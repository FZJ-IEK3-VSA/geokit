from .util import *
from .srsutil import *
from .geomutil import *
from .rasterutil import *
from .vectorutil import *
from .extent import Extent

from io import BytesIO

MaskAndExtent = namedtuple("MaskAndExtent", "mask extent id")
class RegionMask(object):
    """The RegionMask object represents a given region and exposes methods allowing 
    for easy manipulation of geospatial data around that region.

    RegionMask objects are defined by providing a polygon (either via a vector 
    file, an ogr.Geometry, or a Well-Known-Text (WKT) string), a projection system
    to work in, and an extent and pixel resolution to create a matrix mask (i.e.
    boolean values) of.

    * The extent of the generated mask matrix is the tightest fit around the region 
      in units of the pixel resolution. However, the extenT can be defined explcitly 
      if desired
    * The region can be manipulated as a vector polygon via the ".geometry" 
      attribute, which exposes the geometry as an ogr.Geometry. To incoporate this 
      into other vector-handeling libraries it is suggested to use the 
      ".ExportToWkt()" method available via OGR.
    * The region can be manipulated as a raster matrix via the ".mask" attribute 
      which exposes the mask as a boolean numpy.ndarray
    * Any raster source can be easily warped onto the region-mask's extent, 
      projection, and resolution via the ".warp" method
    * Any vector source can be rasterized onto the region-mask's extent, projection,
      and resolution via the ".rasterize" method
    * The default mask set-up is defined by the constant members: DEFAULT_SRS, 
      DEFAULT_RES, and DEFAULT_PAD

    Initializers:
    -------------
    * RegionMask(...) 
        - This is not the preferred way

    * RegionMask.fromVector( ... )
    
    * RegionMask.fromVectorFeature( ... )
    
    * RegionMask.fromGeom( ... )
    
    * RegionMask.fromMask( ... ) 
    
    * RegionMask.load( ... )
        - This function tries to determine which of the other initializers 
          should be used based off the input
    
    """   

    DEFAULT_SRS = 'europe_m'
    DEFAULT_RES = 100
    DEFAULT_PAD = None

    def __init__(s, extent, pixelRes, mask=None, geom=None, attributes=None, **kwargs):
        """The default constructor for RegionMask objects. Creates a RegionMask 
        directly from a matrix mask and a given extent (and optionally a geometry). 
        Pixel resolution is calculated in accordance with the shape of the mask 
        mask and the provided extent

        * Generally one should use the '.load' or else one of the '.fromXXX'
          methods to create RegionMasks

        Parameters:
        -----------
        extent : Extent object
            The geospatial context of the region mask
            * The extent must fit the given pixel sizes
            * All computations using the RegionMask will be evaluated within this
              spatial context
        
        pixelRes : float or tuple
            The RegionMask's native pixel size(s)
            * If float : A pixel size to apply to both the X and Y dimension
            * If (float float) : An X-dimension and Y-dimension pixel size
            * All computations using the RegionMask will generate results in 
              reference to these pixel sizes (i.e. either at this resolution or 
              at some scaling of this resolution)

        mask : numpy-ndarray
            A mask over the context area defining which pixel as inside the region
            and which are outside
            * Must be a 2-Dimensional bool-matrix describing the region, where:
                - 0/False -> "not in the region"
                - 1/True  -> "Inside the region"
            * Either a mask or a geometry must be given, but not both

        geom : ogr-Geomertry 
            A geometric representation of the RegionMask's region
            * Either a mask or a geometry must be given, but not both

        attributes : dict
            Keyword attributes and values to carry along with the RegionMask

        """
        # Check for bad input
        if mask is None and geom is None: raise GeoKitRegionMaskError("Either mask or geom should be defined")    
        if not kwargs.get("mask_plus_geom_is_okay",False):
            if not mask is None and not geom is None: raise GeoKitRegionMaskError("mask and geom cannot be defined simultaneously")

        # Set basic values
        s.extent = extent
        s.srs = extent.srs

        if s.srs is None:
            raise GeoKitRegionMaskError("Extent SRS cannot be None")

        # Set Pixel Size
        if not extent.fitsResolution(pixelRes):
            raise GeoKitRegionMaskError("The given extent does not fit the given pixelSize")

        try:
            pixelWidth, pixelHeight = pixelRes
        except:
            pixelWidth, pixelHeight = pixelRes, pixelRes

        s.pixelWidth = abs(pixelWidth)
        s.pixelHeight = abs(pixelHeight)

        if( s.pixelHeight == s.pixelWidth ):
            s._pixelSize = s.pixelHeight
        else:
            s._pixelSize = None

        # set height and width
        s.width = int(np.round((s.extent.xMax-s.extent.xMin)/s.pixelWidth))
        s.height = int(np.round((s.extent.yMax-s.extent.yMin)/s.pixelHeight))

        # Set mask
        s._mask = mask
        if not mask is None: # test the mask
            # test type
            if(mask.dtype != "bool" and mask.dtype != "uint8" ): 
                raise GeoKitRegionMaskError("Mask must be bool type")
            if(mask.dtype == "uint8"):
                mask = mask.astype("bool")

            if not isclose(extent.xMin+pixelWidth*mask.shape[1], extent.xMax) or not isclose(extent.yMin+pixelHeight*mask.shape[0], extent.yMax):
               raise GeoKitRegionMaskError("Extent and pixels sizes do not correspond to mask shape")

        # Set geometry
        if not geom is None: # test the geometry
            if not isinstance(geom, ogr.Geometry):
                raise GeoKitRegionMaskError("geom is not an ogr.Geometry object")
            
            s._geometry = geom.Clone()
            gSRS = geom.GetSpatialReference()
            if gSRS is None:
                raise GeoKitRegionMaskError("geom does not have an srs")

            if not gSRS.IsSame(s.srs): transform(s._geometry, toSRS=s.srs, fromSRS=gSRS)
        else:
            s._geometry = None

        # set attributes
        s.attributes = {} if attributes is None else attributes

    @staticmethod
    def fromMask(extent, mask, attributes=None):
        """Make a RegionMask directly from amask matrix and extent
        
        Note:
        -----
        Pixel sizes are calculated from the extent boundaries and mask dimensional
        sizes

        Parameters:
        -----------
        extent : Extent object
            The geospatial context of the region mask
            * The extent must fit the given pixel sizes
            * All computations using the RegionMask will be evaluated within this
              spatial context
        
        mask : numpy-ndarray
            A mask over the context area defining which pixel as inside the region
            and which are outside
            * Must be a 2-Dimensional bool-matrix describing the region, where:
                - 0/False -> "not in the region"
                - 1/True  -> "Inside the region"

        attributes : dict
            Keyword attributes and values to carry along with the RegionMask

        Returns:
        --------
        RegionMask

        """

        # get pixelWidth and pixelHeight
        pixelWidth = (extent.xMax-extent.xMin)/(mask.shape[1])
        pixelHeight = (extent.yMax-extent.yMin)/(mask.shape[0])

        return RegionMask(extent=extent, pixelRes=(pixelWidth, pixelHeight), mask=mask, attributes=attributes)

    @staticmethod
    def fromGeom(geom, pixelRes=DEFAULT_RES, srs=DEFAULT_SRS, extent=None, padExtent=DEFAULT_PAD, attributes=None):
        """Make a RasterMask from a given geometry

        Parameters:
        -----------
        geom : ogr-Geomertry or str
            A geometric representation of the RegionMask's region
            * If a string is given, geokit.geom.convertWKT(geom, srs) is called 
              to convert it to an ogr.Geometry

        pixelRes : float or tuple
            The RegionMask's native pixel resolution(s)
            * If float : A pixel size to apply to both the X and Y dimension
            * If (float float) : An X-dimension and Y-dimension pixel size
        
        srs : Anything acceptable to geokit.srs.loadSRS()
            The srs context of the generated RegionMask object
            * This srs is superseded by the srs in an explicitly defined extent 
            * The default srs EPSG3035 is only valid for a European context

        extent : Extent object
            The geospatial context of the generated region mask
            * The extent must fit the given pixel sizes
        
        padExtent : float; optional
            An amount by which to pad the extent before generating the RegionMask

        attributes : dict
            Keyword attributes and values to carry along with the RegionMask

        Returns:
        --------
        RegionMask

        """
        srs = loadSRS(srs)
        # make sure we have a geometry with an srs
        if( isinstance(geom, str)):
            geom = convertWKT(geom, srs)

        geom = geom.Clone() # clone to make sure we're free of outside dependencies

        # set extent (if not given)
        if extent is None:
            extent = Extent.fromGeom(geom).castTo(srs).pad(padExtent).fit(pixelRes)
        else:
            if not extent.srs.IsSame(srs):
                raise GeoKitRegionMaskError("The given srs does not match the extent's srs")
            #extent = extent.pad(padExtent)

        # make a RegionMask object
        return RegionMask(extent=extent, pixelRes=pixelRes, geom=geom, attributes=attributes)


    @staticmethod
    def fromVector(source, where=None, geom=None, pixelRes=DEFAULT_RES, srs=DEFAULT_SRS, extent=None, padExtent=DEFAULT_PAD, **kwargs):
        """Make a RasterMask from a given vector source
        
        Note:
        -----
        Be careful when creating a RegionMask over a large area (such as a country)!
        Using the default pixel size for a large area (such as a country) can 
        easily consume your system's memory

        Parameters:
        -----------
        source : Anything acceptable by loadVector()
            The vector data source to read from

        where : str, int; optional
            If string -> An SQL-like where statement to apply to the source
            If int -> The feature's ID within the dataset
            * Feature attribute name do not need quotes
            * String values should be wrapped in 'single quotes'
            Example: If the source vector has a string attribute called "ISO" and 
                     a integer attribute called "POP", you could use....

                where = "ISO='DEU' AND POP>1000"

        geom : ogr.Geometry; optional
            The geometry to search with
            * All features are extracted which touch this geometry

        pixelRes : float or tuple
            The RegionMask's native pixel resolution(s)
            * If float : A pixel size to apply to both the X and Y dimension
            * If (float float) : An X-dimension and Y-dimension pixel size
        
        srs : Anything acceptable to geokit.srs.loadSRS()
            The srs context of the generated RegionMask object
            * This srs is superseded by the srs in an explicitly defined extent 
            * The default srs EPSG3035 is only valid for a European context

        extent : Extent object
            The geospatial context of the generated region mask
            * The extent must fit the given pixel sizes
            * If not specified, the entire extent of the vector file is assumed
        
        padExtent : float; optional
            An amount by which to pad the extent before generating the RegionMask

        Returns:
        --------
        RegionMask

        """
        # Get all geoms which fit the search criteria
        if isinstance(where, int): 
            geom,attr = extractFeature(source=source, where=where, srs=srs)
        else:
            geoms = extractFeatures(source=source, where=where, srs=srs, onlyGeom=True)
            geom = flatten(geoms)
            attr = None

        # Done!
        return RegionMask.fromGeom(geom, extent=extent, pixelRes=pixelRes, attributes=attr, padExtent=padExtent, srs=srs, **kwargs)

    @staticmethod
    def load(region, **kwargs):
        """Tries to initialize and return a RegionMask in the most appropriate way. 

        Note:
        -----
        If 'region' input is...
            * Already a RegionMask, simply return it
            * A file path, use RegionMask.fromVector
            * An OGR Geometry object, assume is it to be loaded by RegionMask.fromGeom
            * A NumPy array, assume is it to be loaded by RegionMask.fromMask
                - An 'extent' input must also be given
        
        Parameters:
        -----------
        region : Can be RegionMask, str, ogr.Geometry, numpy.ndarray
            The shape  defining the region over which to build the RegionMask
            * See the note above

        where : str, int; optional
            If string -> An SQL-like where statement to apply to the source
            If int -> The feature's ID within the dataset
            * Feature attribute name do not need quotes
            * String values should be wrapped in 'single quotes'
            Example: If the source vector has a string attribute called "ISO" and 
                     a integer attribute called "POP", you could use....

                where = "ISO='DEU' AND POP>1000"

        geom : ogr.Geometry; optional
            The geometry to search with
            * All features are extracted which touch this geometry

        pixelRes : float or tuple
            The RegionMask's native pixel resolution(s)
            * If float : A pixel size to apply to both the X and Y dimension
            * If (float float) : An X-dimension and Y-dimension pixel size
        
        srs : Anything acceptable to geokit.srs.loadSRS()
            The srs context of the generated RegionMask object
            * This srs is superseded by the srs in an explicitly defined extent 
            * The default srs EPSG3035 is only valid for a European context

        extent : Extent object
            The geospatial context of the generated region mask
            * The extent must fit the given pixel sizes
            * If not specified, the entire extent of the vector file is assumed
        
        padExtent : float; optional
            An amount by which to pad the extent before generating the RegionMask

        """
        if isinstance(region, RegionMask): return region
        elif isinstance( region, str): return RegionMask.fromVector(region, **kwargs)
        elif isinstance(region, ogr.Geometry): return RegionMask.fromGeom(region, **kwargs)
        elif isinstance(region, np.ndarray): return RegionMask.fromMask(region, **kwargs)
        else:
            raise GeoKitRegionMaskError("Could not understand region input")


    @property
    def pixelSize(s):
        """The RegionMask's pixel size. 

        !!Only available when pixelWidth equals pixelHeight!!"""
        if s._pixelSize is None:
            raise GeoKitRegionMaskError("pixelSize only accessable when pixelWidth equals pixelHeight")
        return s._pixelSize


    def buildMask(s, **kwargs):
        """Explicitly build the RegionMask's mask matrix. 

        * All kwargs are passed on to a call to geokit.vector.rasterize()

        """
        if s._geometry is None:
            raise GeoKitRegionMaskError("Cannot build mask when geometry is None")

        s._mask = None
        s._mask = s.rasterize(s._geometry, applyMask=False, **kwargs)

    @property
    def mask(s):
        """The RegionMask's mask array as an 2-dimensional boolean numpy array.
        
        * If no mask was given at the time of the RegionMask's creation, then a 
          mask will be generated on first access to the 'mask' property
        * The mask can be rebuilt in a customized way using the 
          RegionMask.buildMask() function
        """
        if(s._mask is None): s.buildMask()
        return s._mask

    @property
    def area(s):
        return s.mask.sum()*s.pixelWidth*s.pixelHeight

    def buildGeometry(s):
        """Explicitly build the RegionMask's geometry"""
        if s._mask is None:
            raise GeoKitRegionMaskError("Cannot build geometry when mask is None")
        
        s._geometry = None
        s._geometry = polygonizeMask( s.mask, bounds=s.extent.xyXY, srs=s.extent.srs, flat=True )

    @property
    def geometry(s):
        """Fetches a clone of the RegionMask's geometry as an OGR Geometry object

        * If a geometry was not provided when the RegionMask was initialized, 
          then one will be generated from the RegionMask's mask matrix in the 
          RegionMask's extent
        * The geometry can always be deleted and rebuild using the 
          RegionMask.rebuildGeometry() function
        """

        if(s._geometry is None): s.buildGeometry()

        return s._geometry.Clone()

    @property
    def vectorPath(s):
        """Returns a path to a vector path on disc which is built only once"""

        if(s._vectorPath is None): 
            s._vectorPath = s._tempFile(ext=".shp")
            createVector(s.geometry, output=s._vectorPath)

        return s._vectorPath

    @property
    def vector(s):
        """Returns a vector saved in memory which is built only once"""

        if(s._vector is None): 
            s._vector = quickVector(s.geometry)

        return s._vector

    def _repr_svg_(s):
        if(not hasattr(s,"svg")):
            f = BytesIO()

            import matplotlib.pyplot as plt
            plt.figure(figsize=(4,4))
            ax = plt.subplot(111)

            g = s.geometry
            h=drawGeoms(g, ax=ax)

            ax.set_aspect('equal')
            ax.autoscale(enable=True)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f, format="svg", dpi=100)

            f.seek(0)
            s.svg = f.read().decode('ascii')

        return s.svg

    def drawMask( s, **kwargs):
        """Draw the region on a matplotlib figure

        * All kwargs (except 'bounds') are passed on to util.drawImage()"""
        return drawImage( s.mask, bounds=s.extent.xyXY, **kwargs )

    def drawGeometry( s, **kwargs):
        """Draw the region on a matplotlib figure

        * All kwargs passed on to geom.drawGeoms()"""
        return drawGeoms( s.geometry, srs=s.srs, **kwargs )

    # def drawImage(s, matrix, ax=None, hideExternal=True, cbar=False, output=None, noBorder=True, geomArgs={}, **imageArgs):
    #     """Draw the region geometry on top of a matrix image

    #     - The image data is assumed to match the boundaries of the RegionMask object

    #     Inputs:
    #         matrix - np.ndarray : The 2-dimensional image data to plot

    #         ax - matplotlib axis object : The axis to draw the figure onto
    #             * If given as 'None', then a fresh axis will be produced and displayed or saved immediately
    #             * When not 'None', then this function returns a handle to the drawn image which can be used however you see fit

    #         hideExternal - T/F : If true, hide values found outside the region

    #         cbar - True/False : Flag indicating whether or not to automatically add a colorbar
    #             * Only operates when an axis has not been given

    #         output - str : A path to save the output figure to
    #             * Only applies when 'ax' is None
    #             * If this is None and 'ax' is None, the figure is displayed immediately

    #         noBorder - T/F : A flag determining whether or not to show the borders of the plot's axis
    #             * Only useful when 'ax' is None

    #         geomArgs - dict : keyword arguments to pass on to the bourder drawing
    #             * All are passed on to a call to RegionMask.drawGeometry(...)

    #         **imageArgs : All other keyword arguements are passed on to the image plotting
    #             * Performed by geokit.raster.drawImage(...)
        
    #     """
    #     from matplotlib.colors import LinearSegmentedColormap

    #     # Do we make our own figure?
    #     if ax is None:
    #         doShow = True
    #         # import some things
    #         import matplotlib.pyplot as plt

    #         # make a figure and axis
    #         plt.figure(figsize=(12,12))
    #         ax = plt.subplot(111)
    #     else: 
    #         doShow=False

    #     # Draw matrix
    #     if not "cmap" in imageArgs:
    #         cmap = LinearSegmentedColormap.from_list('red_green_blue',[(0,91/255,130/255),(171/255,221/255,164/255),(180/255,20/255,20/255)])
    #     else:
    #         cmap = imageArgs.pop("cmap")

    #     vmin = imageArgs.pop("vmin", matrix[s.mask].min())
    #     vmax = imageArgs.pop("vmax", matrix[s.mask].max())

    #     _imageArgs = dict(bounds=s.extent)
    #     _imageArgs.update(imageArgs)
    #     h=drawImage(matrix, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, **_imageArgs)

    #     # Draw region
    #     if hideExternal:
    #         cmap2 = LinearSegmentedColormap.from_list('alpha_to_white',[(1,1,1,0),(1,1,1)])
    #         drawImage(~s.mask, ax=ax, bounds=s.extent, cmap=cmap2)

    #     _geomArgs = dict(simplification=None, fc='None', ec='k', linewidth=3)
    #     _geomArgs.update(geomArgs)
    #     s.drawGeometry(ax=ax, **_geomArgs)

    #     # Done!
    #     if doShow:
    #         ax.set_aspect('equal')
    #         ax.autoscale(enable=True)

    #         if noBorder: plt.axis('off')
    #         if cbar: 
    #             cb = plt.colorbar(h)
    #             cb.ax.tick_params(labelsize=18) 

    #         if output: 
    #             plt.savefig(output, dpi=200)
    #             plt.close()
    #         else: 
    #             plt.show()
    #     else:
    #         return ax


    def _tempFile(s, head="tmp", ext=".tif"):
        """***RM INTERNAL***

        Use this to create a temporary file associated with the RegionMask which 
        will be deleted when the RM goes out of scope.

        !! BEWARE OF EXTERNAL DEPENDANCIES WHEN THE RM IS GOING OUT OF SCOPE, 
        THIS WILL CAUSE A LOT OF ISSUES !!
        """
        if(not hasattr(s,"_TMPDIR")):
            # Create a temporary directory to use with this shape (and associated processes)
            s._TMPDIR = TemporaryDirectory()
        return NamedTemporaryFile(suffix=ext, prefix=head, dir=s._TMPDIR.name, delete=True).name

    def __del__(s):
        if(hasattr(s, "_TMPDIR")): s._TMPDIR.cleanup()

    def _resolve(s, div):
        if(div<0): div = 1.0/abs(int(div))
        return (s.pixelWidth/div, s.pixelHeight/div)

    def applyMask(s, mat, noData=0):
        """Shortcut to apply the RegionMask's mask to an array. Mainly intended
        for internal use

        * When the passed matrix does not have the same extents of the given matrix,
          it is assumed that the RegionMask's mask needs to be scaled so that the
          matrix dimensions match

        * The RM's mask can only be scaled UP, and the given matrix's dimensions
          must be mutiples of the mask's dimensions

        Parameters:
        -----------
        mat : np.ndarray
            The matrix to apply the mask to
            * Must have dimensions equal, or are multiples of, the mask's

        noData : float
            The no-data value to set into matrix's values which are not within 
            the region

        Returns:
        --------
        numpy.ndarray
            
        """
        if(noData is None): noData=0
        # Get size
        Y,X = mat.shape

        # make output array
        out = np.array(mat)

        # Apply mask
        if( s.mask.shape == mat.shape ): # matrix dimensions coincide with mask's data
            out[~s.mask] = noData

        elif( Y>s.height and X>s.width ):
            if( not Y%s.height==0 or not X%s.width==0 ):
                raise GeoKitRegionMaskError("Matrix dimensions must be multiples of mask dimensions")

            yScale = Y//s.height
            xScale = X//s.width

            scaledMask = scaleMatrix(s.mask, (yScale,xScale))
            sel = np.where(~scaledMask)
            out[sel] = noData

        else:
            raise GeoKitRegionMaskError("Could not map mask onto matrix")

        return out
    
    #######################################################################################
    ## Raw value processor
    def _returnBlank(s, resolutionDiv=1, forceMaskShape=False, applyMask=True, noDataValue=None, **kwargs):
        # make output
        if not forceMaskShape and resolutionDiv > 1:
            yN = s.mask.shape[0]*int(resolutionDiv)
            xN = s.mask.shape[1]*int(resolutionDiv)
            output = np.zeros( (yN,xN) )
        else:
            output = np.zeros(s.mask.shape)

        # apply mask, maybe
        if applyMask:
            output = s.applyMask(output, noDataValue)

        # Done
        return output

    def indicateValues(s, source, value, buffer=None, resolutionDiv=1, forceMaskShape=False, applyMask=True, noData=None, **kwargs):
        """
        Indicates those pixels in the RegionMask which correspond to a particular 
        value, or range of values, from a given raster datasource

        Returns a matrix matching the RegionMask's mask dimensions wherein 0 means
        the pixels is not included in the indicated set, and 1 meaning the pixel 
        is included in the indicated set. Intermediate values are also possible. 
        This results from a scenario when the datasource's resolution does not 
        line up perfectly with the RegionMask's resolution and, as a result, a 
        RegionMask pixel overlaps multiple datasource pixels which are not all 
        indicated (or not-indicated). 
        
        * Value processing is performed BEFORE a warp takes place
        * Output from the warp is clipp to values between 0 and 1
        * If a boolean matrix is desired of the result, use "result > 0.5"

        Parameters:
        -----------
        source : str or gdal.Dataset
            The raster datasource to indicate from
            
        value : float or tuple
            The value or range of values to indicate
            * If float : The exact value to accept
              - Maybe cause issues due to float comparison issues. Using an 
                integer is usually better
            * If (float, float) : The inclusive Min and Max values to accept
              - None refers to no bound
              - Ex. (None, 5) -> "Indicate all values equal to and below 5"

        buffer : float; optional
            A buffer region to add around the indicated pixels
            * Units are in the RegionMask's srs
            * The buffering occurs AFTER the indication and warping step and
              so it may not represent the original dataset exactly
              - Buffering can be made more accurate by increasing the 
                'resolutionDiv' input
        
        resolutionDiv : int
            The factor by which to divide the RegionMask's native resolution
            * This is useful if you need to represent very fine details

        forceMaskShape : bool 
            If True, forces the returned matrix to have the same dimension as 
            the RegionMask's mask regardless of the 'resolutionDiv' argument

        applyMask : bool
            When True, the RegionMask's mask will be applied to the outputData
            as described by RegionMask.applyMask

        noData : numeric
            The noData value to use when applying the mask
        
        kwargs -- Passed on to RegionMask.warp()
            * Most notably: 'resampleAlg'


        Returns:
        --------
        numpy.ndarray


        """
        # Unpack value
        if isinstance(value, tuple):
            valueMin,valueMax = value
            valueEquals = None
        else:
            valueMin,valueMax = None,None
            valueEquals = value

        # make processor
        def processor(data):
            ## Find nan values, maybe
            if(not noDataValue is None): 
                nodat = np.isnan(data)
            
            ## Do processing
            if(not valueEquals is None):
                output = data == valueEquals
            else:
                output = np.ones(data.shape, dtype="bool")
            
                if(not valueMin is None):
                    np.logical_and(data >= valueMin, output, output)
                if(not valueMax is None):
                    np.logical_and(data <= valueMax, output, output)
            
            ## Fill nan values, maybe
            if(not noDataValue is None): 
                output[nodat] = noDataValue

            ## Done!
            return output

        # Do processing
        newDS = s.extent.mutateRaster(source, processor=processor, dtype="bool", noData=noData)

        # Warp onto region
        final = s.warp(newDS, dtype="float32", resolutionDiv=resolutionDiv, 
                       applyMask=False, noData=noData, returnMatrix=True, **kwargs)

        # Check for results
        if not (final > 0).any():
            # no results were found
            return s._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape, 
                                  applyMask=applyMask, noDataValue=noDataValue)

        # Apply a buffer if requested
        if not buffer is None:
            geoms = convertMask(final>0.5, bounds=s.extent, srs=s.srs)

            if len(geoms)>0:
                geoms = [g.Buffer(buffer) for g in geoms]
                areaDS = createVector(geoms)
                final = s.rasterize( areaDS, dtype="float32", bands=[1], burnValues=[1], resolutionDiv=resolutionDiv, 
                                     applyMask=False, noDataValue=noDataValue)
            else:
                # no results were found
                return s._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape, 
                                      applyMask=applyMask, noDataValue=noDataValue)

        # apply a threshold incase of funky warping issues
        final[final>1.0] = 1
        final[final<0.0] = 0

        # Make sure we have the mask's shape
        if forceMaskShape:
            if resolutionDiv > 1:
                final = scaleMatrix(final, -1*resolutionDiv)

        # Apply mask?
        if applyMask: final = s.applyMask(final, noDataValue)

        # Return result
        return final

    #######################################################################################
    ## Vector feature indicator
    def indicateFeatures(s, source, where=None, buffer=None, bufferMethod='geom', resolutionDiv=1, forceMaskShape=False, applyMask=True, noData=0, **kwargs):
        """
        Indicates the RegionMask pixels which are found within the features (or 
        a subset of the features) contained in a given vector datasource

        * A Rasterization is performed from the input data set to the 
          RegionMask's mask.
          -See geokit.vector.rasterize or, more specifically gdal.RasterizeOptions
           kwargs for more info on how to control the rasterization step
        
        Parameters:
        -----------
        source : str or gdal.Dataset
            The vector datasource to indicate from
            
        where : str; optional
            An SQL-style filtering string
            * Can be used to filter the input source according to their attributes
            * For tips, see "http://www.gdal.org/ogr_sql.html"
            Ex: 
              where="eye_color='Green' AND IQ>90"

        buffer : float; optional
            A buffer region to add around the indicated pixels
            * Units are in the RegionMask's srs

        bufferMethod : str; optional
            An indicator determining the method to use when buffereing
            * Options are: 'geom' and 'area'
            * If 'geom', the function will attempt to grow each of the geometries
              directly using the ogr library
              - This can fail sometimes when the geometries are particularly 
                complex or if some of the geometries are not valid (as in, they 
                have self-intersections)
            * If 'area', the function will first rasterize the raw geometries and
              will then apply the buffer to the indicated pixels
              - This is the safer option although is not as accurate as the 'geom'
                option since it does not capture the exact edges of the geometries
              - This method can be made more accurate by increasing the 
                'resolutionDiv' input
        
        resolutionDiv : int; optional
            The factor by which to divide the RegionMask's native resolution
            * This is useful if you need to represent very fine details

        forceMaskShape : bool; optional
            If True, forces the returned matrix to have the same dimension as 
            the RegionMask's mask regardless of the 'resolutionDiv' argument

        applyMask : bool; optional
            When True, the RegionMask's mask will be applied to the outputData
            as described by RegionMask.applyMask

        noData : numeric
            The noData value to use when applying the mask
        
        kwargs -- Passed on to RegionMask.rasterize()
            * Most notably: 'allTouched'

        Returns:
        --------
        numpy.ndarray

        """
        # Ensure path to dataSet exists
        source = loadVector(source)
        
        # Do we need to buffer?
        if not buffer is None and bufferMethod == 'geom':
            def doBuffer(geom,attr): return {'geom':geom.Buffer(buffer)}
            source = s.mutateVector(source, where=where, processor=doBuffer, _slim=True)

            where=None # Set where to None since the filtering has already been done

            if source is None: # this happens when the returned dataset is empty
                return s._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape, 
                                  applyMask=applyMask, noDataValue=noDataValue, **kwargs)

        # Do rasterize
        indicated = s.rasterize( source, dtype='float32', bands=[1], burnValues=[1], where=where, resolutionDiv=resolutionDiv, 
                                 applyMask=False, noData=noData)

        # Check for results
        if not (indicated > 0).any():
            # no results were found
            return s._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape, 
                                  applyMask=applyMask, noDataValue=noDataValue)

        # maybe we want to do the other buffer method
        if not buffer is None and bufferMethod == 'area':
            geoms = polygonizeMask(indicated>0.5, bounds=s.extent, srs=s.srs)
            if len(geoms)>0:
                geoms = [g.Buffer(buffer) for g in geoms]
                dataSet = quickVector(geoms)
                final = s.rasterize( dataSet, dtype="float32", bands=[1], burnValues=[1], resolutionDiv=resolutionDiv, 
                                     applyMask=False, noDataValue=noDataValue)
            else:
                return s._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape, 
                                      applyMask=applyMask, noDataValue=noDataValue)

        # Make sure we have the mask's shape
        if forceMaskShape:
            if resolutionDiv > 1:
                final = scaleMatrix(final, -1*resolutionDiv)

        # Apply mask?
        if applyMask: final = s.applyMask(final, noDataValue)

        # Return
        return final

    #######################################################################################
    ## Vector feature indicator
    def indicateGeoms(s, geom, **kwargs):
        """
        Convenience function to indicate values found within a geometry (or a 
        list of geometries)

        * Simply creates a new vector source from the given geometry and then 
          calls RegionMask.indicateFeatures
        * All keywords are passed on to RegionMask.indicateFeatures
        """
        # Ensure geom is a list of geometries
        if isinstance(geom, ogr.Geometry):
            geom = [geom,]
        elif isinstance(geom, list):
            pass
        else: # maybe geom is iterable
            geom = list(geom)

        # Make a vector dataset
        ds = quickVector(geom)

        # Indicate features
        return s.indicateFeatures(ds, **kwargs)

    #######################################################################################
    ## Make a sub region generator
    def subRegions(s, gridSize, asMaskAndExtent=False):
        """Generate a number of sub regions on a grid which combine into the total
        RegionMask area
        """
        # get useful matrix info
        yN, xN = s.mask.shape
        pixelGridSize = int(gridSize/min(s.pixelWidth, s.pixelHeight))

        # Make grid areas
        count = 0
        for ys in range(0, yN, pixelGridSize):

            yn = min(yN, ys+pixelGridSize)
            yMax = s.extent.yMax - ys*s.pixelHeight
            yMin = s.extent.yMax - yn*s.pixelHeight

            for xs in range(0, xN, pixelGridSize):
                xn = min(xN, xs+pixelGridSize)
                xMin = s.extent.xMin + xs*s.pixelWidth
                xMax = s.extent.xMin + xn*s.pixelWidth

                sectionMask = s.mask[ys:yn, xs:xn]
                if not sectionMask.any(): continue

                sectionExtent = Extent( xMin,yMin,xMax,yMax, srs=s.srs ).fit((s.pixelWidth, s.pixelHeight))

                if asMaskAndExtent:
                    yield MaskAndExtent( sectionMask, sectionExtent, count)
                else:
                    yield RegionMask.fromMask(sectionExtent, sectionMask, dict(id=count))

                count+=1


    #############################################################################
    ## CONVENIENCE FUNCTIONS
    def createRaster(s, resolutionDiv=1, **kwargs):
        """Convenience function for geokit.raster.createRaster which sets 'srs',
        'bounds', 'pixelWidth', and 'pixelHeight' inputs

        Parameters:
        -----------
        resolutionDiv : int
            The factor by which to divide the RegionMask's native resolution
            * This is useful if you need to represent very fine details
        
        **kwargs:
            All other keywargs are passed on to geokit.raster.createRaster()
            * See below for argument descriptions

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None

        """
        pW, pH = s._resolve(resolutionDiv)
        return s.extent.createRaster( pixelWidth=pW, pixelHeight=pH, **kwargs)

    def warp(s, source, resolutionDiv=1, returnMatrix=True, applyMask=True, noData=None, **kwargs):
        """Convenience function for geokit.raster.warp() which automatically sets
        'srs', 'bounds', 'pixelWidth', and 'pixelHeight' inputs
        
        Parameters:
        -----------
        source : str
            The path to the raster file to warp

        resolutionDiv : int
            The factor by which to divide the RegionMask's native resolution
            * This is useful if you need to represent very fine details

        returnAsMatrix : bool
            When True, the resulting raster's matrix is return 
            * Should have the same dimensions as the RegionMask's mask matrix

        applyMask : bool
            When True, the RegionMask's mask will be applied to the outputData
            as described by RegionMask.applyMask

        noData : numeric
            The noData value to use when applying the mask
        
        **kwargs:
            All other keywargs are passed on to geokit.raster.warp()

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None

        """
        pW, pH = s._resolve(resolutionDiv)

        # do warp
        if returnMatrix:
            newDS = s.extent.warp(source=source, pixelWidth=px, pixelHeight=pH, **kwargs)
        else:
            if applyMask:
                if "cutline" in kwargs: 
                    raise GeoKitRegionMaskError("Cannot apply both a cutline and the mask when returning the warped dataset")
                newDS = s.extent.warp(source=source, pixelWidth=px, pixelHeight=pH, cutline=s.vector, **kwargs)   
            else:
                newDS = s.extent.warp(source=source, pixelWidth=px, pixelHeight=pH, **kwargs)

        if not returnMatrix: return newDS

        # Read array
        if newDS is None: newDS = kwargs["output"]
        final = extractMatrix(newDS)
        
        # clean up
        del newDS

        # Apply mask, maybe
        if(applyMask): 
            final = s.applyMask(final, noData)
        
        # Return
        if returnMatrix: return final
        else: return kwargs["output"]

    def rasterize(s, source, resolutionDiv=1, returnMatrix=True, applyMask=True, **kwargs):
        """Convenience function for geokit.vector.rasterize() which automatically
        sets the 'srs', 'bounds', 'pixelWidth', and 'pixelHeight' inputs 
        
        Parameters:
        -----------
        source : str
            The path to the vector file to load

        resolutionDiv : int; optional
            The factor by which to divide the RegionMask's native resolution
            * This is useful if you need to represent very fine details

        returnAsMatrix : bool; optional
            When True, the resulting raster's matrix is return 
            * Should have the same dimensions as the RegionMask's mask matrix

        applyMask : bool; optional
            When True, the RegionMask's mask will be applied to the outputData
            as described by RegionMask.applyMask

        noData : numeric; optional
            The noData value to use when applying the mask
        
        **kwargs:
            All other keywargs are passed on to geokit.vector.rasterize()

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None

        """
        pW, pH = s._resolve(resolutionDiv)

        # do rasterization
        if returnMatrix:
            newDS = s.extent.rasterize(source=source, pixelWidth=px, pixelHeight=pH, **kwargs)
        else:
            if applyMask:
                if "cutline" in kwargs: 
                    raise GeoKitRegionMaskError("Cannot apply both a cutline and the mask when returning the rasterized dataset")
                newDS = s.extent.rasterize(source=source, pixelWidth=px, pixelHeight=pH, cutline=s.vector, **kwargs)   
            else:
                newDS = s.extent.rasterize(source=source, pixelWidth=px, pixelHeight=pH, **kwargs)

        if not returnMatrix: return newDS

        # Read array
        if newDS is None: newDS = kwargs["output"]
        final = extractMatrix(newDS)
        
        # clean up
        del newDS

        # Apply mask, maybe
        if(applyMask): 
            final = s.applyMask(final, noData)
            
        # Return
        return final

    def extractFeatures(s, source, **kwargs):
        """Convenience wrapper for geokit.vector.extractFeatures() by setting the 
        'geom' input to the RegionMask's geometry
        
        Parameters:
        -----------
        source : str
            The path to the vector file to load

        **kwargs:
            All other keyword arguments are passed on to vector.extractFeatures()
           
        Returns:
        --------
        * If asPandas is True: pandas.DataFrame or pandas.Series
        * If asPandas is False: generator

        """
        return extractFeatures( source=source, geom=s.geometry, **kwargs )


    def mutateVector(s, source, matchContext=False, **kwargs):
        """Convenience function for geokit.vector.mutateVector which automatically
        sets 'srs' and 'geom' inputs to the RegionMask's srs and geometry

        Note:
        -----
        If this is called without any arguments except for a source, it serves
        to clip the vector source around the RegionMask

        Parameters:
        -----------
        source : Anything acceptable to geokit.vector.loadVector()
            The source to clip

        matchContext : bool; optional
            * If True, transforms all geometries to the RegionMask's srs before 
              mutating
            * If False, only selects the geometries which touch the RegionMask

        **kwargs:
            All other keyword arguments are passed to geokit.vector.mutateVector
            
        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None
        
        """
        # Get the working srs
        if not transform:
            vinfo = vectorInfo( source )
            ext = s.extent.castTo(vinfo.srs)
        else:
            ext = s.extent

        # mutate the source
        return mutateVector(source, srs=srs, geom=s.geometry, **kwargs)

    def mutateRaster(s, source, matchContext=True, warpArgs=None, applyMask=True, processor=None, **mutateArgs):
        """Convenience function for geokit.vector.mutateRaster which automatically
        sets 'bounds'. It also warps the raster to the RegionMask's area 
        and srs before mutating

        Note:
        -----
        If this is called without any arguments except for a source, it serves
        to clip the raster source around the RegionMask, therefore performing
        the same function as RegionMask.warp(..., returnMatrix=False)

        Parameters:
        -----------
        source : Anything acceptable to geokit.raster.loadRaster()
            The source to mutate
        
        matchContext : bool; optional
            * If True, Warp to the RegionMask's boundaries, srs and pixel size 
              before mutating
            * If False, only warp to the RegionMask's boundaries, but keep its 
              srs and resolution intact

        warpArgs : dict; optional
            Arguments to apply to the warping step
            * See geokit.raster.warp()

        processor - function; optional
            The function performing the mutation of the raster's data 
            * The function will take single argument (a 2D numpy.ndarray) 
            * The function must return a numpy.ndarray of the same size as the input
            * The return type must also be containable within a Float32 (int and 
              boolean is okay)
            * See example in geokit.raster.mutateRaster for more info

        applyMask : bool; optional
            When True, the RegionMask's mask will be applied to the outputData
            as described by RegionMask.applyMask

        **kwargs:
            All other keyword arguments are passed to geokit.vector.mutateVector

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None

        """
        output = kwargs.pop("output", None)
        if warpArgs is None: warpArgs = {}

        # Do the warp and mutation
        if matchContext:
            source = s.warp(source, returnMatrix=False, applyMask=applyMask, **warpArgs)

            if processor is None: return source
            else: return mutateRaster(source, output=output, **mutateArgs)

        else:
            if applyMask:
                if "cutline" in warpArgs:
                    raise GeoKitRegionMaskError("Cannot apply both a cutline and the mask during prewarping")
                warpArgs["cutline"] = s.vector

            return s.extent.mutateRaster( source, matchContext=False, warpArgs=warpArgs, **mutateArgs)