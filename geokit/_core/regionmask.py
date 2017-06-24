from .util import *
from .srsutil import *
from .geomutil import *
from .rasterutil import *
from .vectorutil import *
from .extent import Extent

class RegionMask(object):
    """The RegionMask object represents a given region and exposes methods allowing for easy
    manipulation of geospatial data around that region.

    RegionMask objects are defined by providing a polygon (either via a vector file, an OGR 
    Geometry, or a Well-Known-Text (WKT) string), a projection system to work in, and an extent and
    pixel resolution to create a matrix mask (i.e. boolean values) of.

    * The extent of the generated mask matrix is the tightest fit around the region in units
      of the pixel resolution. However, the extend can be defined explcitly defined if desired
    * The region can be manipulated as a vector polygon via the ".geometry" attribute, which 
      exposes the geometry as an OGR Geometry. To incoporate this into other vector-handeling 
      libraries it is suggested to use the ".ExportToWkt()" method available via OGR.
    * The region can be manipulated as a raster matrix via the ".mask" attribute which exposes 
      the mask as a boolean numpy ND-Array
    * Any raster source can be easily warped onto the region-mask's extent, projection, and 
      resolution via the ".warp" method
    * Any vector source can be rasterized onto the region-mask's extent, projection, and 
      resolution via the ".rasterize" method
    * The default mask set-up is defined by the constant members: DEFAULT_SRS, DEFAULT_RES, and DEFAULT_PAD

    Initializers:
        * RegionMask(...) 
            - This is not the preferred way
        * RegionMask.fromVector( ... )
        * RegionMask.fromVectorFeature( ... )
        * RegionMask.fromGeom( ... )
        * RegionMask.fromMask( ... ) 
        * RegionMask.load( ... )
            - This function tries to determine which of the other initializers should be used based off the input
    """   

    DEFAULT_SRS = 'europe_m'
    DEFAULT_RES = 100
    DEFAULT_PAD = None

    """
    def __init__(s, **kwargs):
        raise GeoKitRegionMaskError("Do not directly initialize a RegionMask object, use one of the provided constructors")
    """

    def __init__(s, extent, pixel, mask=None, geom=None, attributes=None, **kwargs):
        """
        The default constructor for RegionMask objects. Creates a RegionMask directly from a matrix mask and a given 
        extent (and optionally a geometry). Pixel resolution is calculated in accordance with the shape of the mask 
        mask and the provided extent

        * Generally one should use the '.load' or else one of the '.fromXXX' methods to create RegionMasks

        Inputs:
            extent - Extent object : The geospatial context of the region mask
                * The extent must fit the given pixel sizes
                * All computations using the RegionMask will be evaluated within this spatial context
            
            pixel : The RegionMask's native pixel size(s)
                - float : A pixel size to apply to both the X and Y dimension
                - (float float) : An X-dimension and Y-dimension pixel size
                * All computations using the RegionMask will generate results in reference to these pixel sizes (i.e.
                  either at this resolution or at some scaling of this resolution)

            mask - numpy-ndarray : A mask over the context area defining which pixel as inside the region and which are 
                                   outside
                * Must be a 2-Dimensional matrix of boolean values describing the region, where:
                    - 0/False -> "not in the region"
                    - 1/True  -> "Inside the region"
                * Either a mask or a geometry must be given, but not both

            geom - ogr-Geomertry object : A geometric representation of the RegionMask's region
                * Either a mask or a geometry must be given, but not both

            attributes - dict : Keyword attributes and values to carry along with the RegionMask
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
        if not extent.fitsResolution(pixel):
            raise GeoKitRegionMaskError("The given extent does not fit the given pixelSize")

        try:
            pixelWidth, pixelHeight = pixel
        except:
            pixelWidth, pixelHeight = pixel, pixel

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

        * Pixel sizes are calculated from the extent boundaries and mask dimensional sizes

        Inputs:
            extent - Extent object : The geospatial context of the region mask
                * All computations using the RegionMask will be evaluated within this spatial context
            
            mask - numpy-ndarray : A mask over the context area defining which pixel as inside the region and which are 
                                   outside
                * Must be a 2-Dimensional matrix of boolean values describing the region, where:
                    - 0/False -> "not in the region"
                    - 1/True  -> "Inside the region"
                * Either a mask or a geometry must be given, but not both

            attributes - dict : Keyword attributes and values to carry along with the RegionMask
        """

        # get pixelWidth and pixelHeight
        pixelWidth = (extent.xMax-extent.xMin)/(mask.shape[1])
        pixelHeight = (extent.yMax-extent.yMin)/(mask.shape[0])

        return RegionMask(extent=extent, pixel=(pixelWidth, pixelHeight), mask=mask, attributes=attributes)

    @staticmethod
    def fromGeom(geom, pixelSize=DEFAULT_RES, srs=DEFAULT_SRS, extent=None, padExtent=DEFAULT_PAD, attributes=None):    
            """
            Make a RasterMask from a given geometry

            Inputs:
                geom : A geometric representation of the RegionMask's region
                    - ogr-Geomertry object
                    - str : A WKT string
                        * Coordinates must correspond to the given 'srs' input
                
                pixelSize : The RegionMask's native pixel size(s)
                    - float : A pixel size to apply to both the X and Y dimension
                    - (float float) : An X-dimension and Y-dimension pixel size
                    * All computations using the RegionMask will generate results in reference to these pixel sizes (i.e.
                      either at this resolution or at some scaling of this resolution)
                    * Units correspond to the 'srs' input's units
                
                srs : The SRS of the created RegionMask
                    - osr.SpatialReference object
                    - an EPSG integer ID
                    - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
                    - a WKT string
                    * If the 'srs' input differs from the geometry's SRS, the geometry will be cast to the given 'srs'
                
                extent : Extent object : An optional geospatial context for the region mask
                    * If 'extent' is None, an extent will be computed accroding to the geometry's envelope and 
                      the RegionMask's pixel size
                    * The extent must fit the given pixel sizes
                    * All computations using the RegionMask will be evaluated within this spatial context
                
                padExtent - float : An optional padding to add around the extent
                    * Must be divisible by the the pixel size

                attributes - dict : Keyword attributes and values to carry along with the RegionMask

            """
            srs = loadSRS(srs)
            # make sure we have a geometry with an srs
            if( isinstance(geom, str)):
                geom = convertWKT(geom, srs)

            geom = geom.Clone() # clone to make sure we're free of outside dependencies

            # set extent (if not given)
            if extent is None:
                extent = Extent.fromGeom(geom).castTo(srs).pad(padExtent).fit(pixelSize)
            else:
                if not extent.srs.IsSame(srs):
                    raise GeoKitRegionMaskError("The given srs does not match the extent's srs")
                #extent = extent.pad(padExtent)

            # make a RegionMask object
            return RegionMask(extent=extent, pixel=pixelSize, geom=geom, attributes=attributes)


    @staticmethod
    def fromVector(source, pixelSize=DEFAULT_RES, srs=DEFAULT_SRS, extent=None, padExtent=DEFAULT_PAD, attributes=None, **kwargs):
        """
        Make a RasterMask from a given vector source

        !NOTE! Becareful when creating a RegionMask over a large area (such as a country)! Using the default pixel 
               size for a large area (such as a country) can easily consume your system's memory resources

        Inputs:
            source : The vector source to extract from
                - path : A path on the file system
                - ogr Dataset
            
            pixelSize : The RegionMask's native pixel size(s)
                - float : A pixel size to apply to both the X and Y dimension
                - (float float) : An X-dimension and Y-dimension pixel size
                * All computations using the RegionMask will generate results in reference to these pixel sizes (i.e.
                  either at this resolution or at some scaling of this resolution)
                * Units correspond to the 'srs' input's units
            
            srs : The SRS of the created RegionMask
                - osr.SpatialReference object
                - an EPSG integer ID
                - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
                - a WKT string
            
            extent : Extent object : An optional geospatial context for the RegionMask
                * If 'extent' is None, an extent will be computed accroding to the vector's envelope and 
                  the RegionMask's pixel size
                * The extent must fit the given pixel sizes
                * All computations using the RegionMask will be evaluated within this spatial context
            
            padExtent - float : An optional padding to add around the extent
                * Must be divisible by the the pixel size

            attributes - dict : Keyword attributes and values to carry along with the RegionMask

            **kwargs : Extra keyword arguments will be passed on to a call to gdal.Rasterize
                * These can be used to filter the vector source's features or to customize the rasterization
                * For example: 
                    - allTouched=True
                    - where="color=='blue'"
        """
        
        # Handel source input
        sourceDS = loadVector(source)
        layer = sourceDS.GetLayer()
        shapeSRS = layer.GetSpatialRef()

        # Get Extent
        if( extent is None ): 
            extent = Extent.fromVector( sourceDS ).castTo(srs).pad(padExtent).fit(pixelSize)

        # Apply a spatial filter on the source layer 
        layer.SetSpatialFilterRect(*extent.castTo(shapeSRS).xyXY)

        # Determine pixelWidth and pixelHeight
        try:
            pWidth, pHeight = pixelSize
        except:
            pWidth, pHeight = pixelSize, pixelSize

        # Create an empty raster
        maskRaster = createRaster(bounds=extent.xyXY, pixelWidth=pWidth, pixelHeight=pHeight, srs=extent.srs)
    
        # Rasterize the shape
        bands = kwargs.pop("bands", [1])
        burnValues = kwargs.pop("burnValues", [1])
        
        err = gdal.Rasterize( maskRaster, sourceDS, bands=bands, burnValues=burnValues, **kwargs)
        if(err != 1):
            raise GeoKitRegionMaskError("Error while rasterizing:\n%s"%err)

        maskRaster.FlushCache()

        # Get final stats
        band = maskRaster.GetRasterBand(1)
        array = band.ReadAsArray().astype('bool')

        # check if layer only has one feature, if so assume it is the geometry we will want
        if(layer.GetFeatureCount()==1):
            ftr = layer.GetFeature(0)
            tmpGeom = ftr.GetGeometryRef()
            geom = tmpGeom.Clone()

            del tmpGeom, ftr
        else: # Otherwise try to get the union of all features
            try:
                geom = flatten([ftr.GetGeometryRef().Clone() for ftr in loopFeatures(layer)])
            except:
                geom=None

        # Check if region geometry is in the given srs. If not, fix it
        srs = loadSRS(srs)
        if( not geom is None and not shapeSRS.IsSame(srs) ):
            geom.TransformTo(srs)

        # do cleanup
        del shapeSRS, layer, sourceDS

        # Done!
        return RegionMask(extent=extent, pixel=pixelSize, mask=array, geom=geom, mask_plus_geom_is_okay=True, attributes=attributes)

    @staticmethod
    def fromVectorFeature(source, select=0, pixelSize=DEFAULT_RES, srs=DEFAULT_SRS, extent=None, padExtent=DEFAULT_PAD, attributes=None):
        """
        Make a RegionMask from a specific feature in a given vector source

        * The RegionMask's attributes dictionary will be automatically extracted from the feature identified in the 
          source

        !NOTE! Becareful when creating a RegionMask over a large area (such as a country)! Using the default pixel 
               size for a large area (such as a country) can easily consume your system's memory resources

        Inputs:
            source : The vector source to extract from
                - path : A path on the file system
                - ogr Dataset

            select : The feature selection description
                - int : The feature ID in the source
                - str : A SQL-style string which filters the features by their attributes
                * Selection must return exactly one feature
            
            pixelSize : The RegionMask's native pixel size(s)
                - float : A pixel size to apply to both the X and Y dimension
                - (float float) : An X-dimension and Y-dimension pixel size
                * All computations using the RegionMask will generate results in reference to these pixel sizes (i.e.
                  either at this resolution or at some scaling of this resolution)
                * Units correspond to the 'srs' input's units
            
            srs : The SRS of the created RegionMask
                - osr.SpatialReference object
                - an EPSG integer ID
                - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
                - a WKT string
            
            extent : Extent object : An optional geospatial context for the RegionMask
                * If 'extent' is None, an extent will be computed accroding to the vector's envelope and 
                  the RegionMask's pixel size
                * The extent must fit the given pixel sizes
                * All computations using the RegionMask will be evaluated within this spatial context
            
            padExtent - float : An optional padding to add around the extent
                * Must be divisible by the the pixel size

            attributes - dict : Keyword attributes and values to carry along with the RegionMask
                * Will be used to update the attributes dictionary of the source's identified feature

        """
        # Get DS, Layer, and Feature
        vecDS = loadVector(source)
        vecLyr = vecDS.GetLayer()

        if( isinstance(select, str) ):
            where = select
            t = vecLyr.SetAttributeFilter(where)
            if( t!=0 ): raise GeoKitRegionMaskError("Error in select statement: \""+where+"\"")
            
            if  vecLyr.GetFeatureCount() > 1: raise GeoKitRegionMaskError("Multiple fetures found")
            if vecLyr.GetFeatureCount() == 0: raise GeoKitRegionMaskError("Zero features found")

            vecFtr = vecLyr.GetNextFeature()
            
        elif isinstance(select, int):
            vecFtr = vecLyr.GetFeature(select)
        else:
            raise GeoKitRegionMaskError("select must be either an SQL 'where' clause, or a feature index")
            
        if vecFtr is None:
            raise GeoKitRegionMaskError("Could not extract feature!")

        # Create a new RegionMask
        geom = vecFtr.GetGeometryRef().Clone()

        return RegionMask.fromGeom( geom, pixelSize=pixelSize, srs=srs, extent=extent, 
                                    padExtent=padExtent, attributes=vecFtr.items() )

    @staticmethod
    def load(region, **kwargs):
        """Tries to initialize and return a RegionMask in the most appropriate way. 

        Meaning, if 'region' input is...
            * already a RegionMask, simply return it
            * a file path...
                - and a "select" kwarg was given, assume it is meant to be loaded by RegionMask.fromVectorFeature
                - and no "select" kwarg is given, assume it is meant to be loaded by RegionMask.fromVector
            * a string but not a file path, assume is it a WKT geometry string to be loaded by RegionMask.fromGeom
            * an OGR Geometry object, assume is it to be loaded by RegionMask.fromGeom
            * a NumPy array, assume is it to be loaded by RegionMask.fromMask
                - An 'extent' input must also be given
        
        * All kwargs are passed on to the called initializer
        """
        if isinstance(region, RegionMask): return region
        elif isinstance( region, str):
            if os.path.isfile(region):
                if 'select' in kwargs:
                    return RegionMask.fromVectorFeature(region, **kwargs)
                else:
                    return RegionMask.fromVector(region, **kwargs)
            else:
                return RegionMask.fromGeom(region, **kwargs)
        elif isinstance(region, ogr.Geometry):
            return RegionMask.fromGeom(region, **kwargs)
        elif isinstance(region, np.ndarray):
            return RegionMask.fromMask(region, **kwargs)
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

        * All kwargs are passed on to a call to gdal.Rasterize
            - 'bands' and 'burnValues' are given as [1]

        """
        if s._geometry is None:
            raise GeoKitRegionMaskError("Cannot build mask when geometry is None")

        s._mask = None

        geomDS = createVector(s._geometry)

        # Create an empty raster
        maskRaster = createRaster(bounds=s.extent.xyXY, pixelWidth=s.pixelWidth, pixelHeight=s.pixelHeight, srs=s.srs)
        
        err = gdal.Rasterize( maskRaster, geomDS, bands=[1], burnValues=[1], **kwargs)
        if(err != 1):
            raise GeoKitRegionMaskError("Error while rasterizing:\n%s"%err)

        maskRaster.FlushCache()

        # Get final stats
        band = maskRaster.GetRasterBand(1)
        s._mask = band.ReadAsArray().astype('bool')

    @property
    def mask(s):
        """The RegionMask's mask array as an 2-dimensional boolean numpy array.
        
        * If no mask was given at the time of the RegionMask's creation, then a mask will be generated on the first call to the 'mask' property
        * The mask can be rebuilt in a customized way using the RegionMask.buildMask() function
        """
        if(s._mask is None): s.buildMask()
        return s._mask

    @property
    def area(s):
        return s.mask.sum()*s.pixelWidth*s.pixelHeight

    def buildGeometry(s):
        """Explicitly build the RM's geometry"""
        if s._mask is None:
            raise GeoKitRegionMaskError("Cannot build geometry when mask is None")
        s._geometry = None
        s._geometry = convertMask( s.mask, bounds=s.extent.xyXY, srs=s.extent.srs, flat=True )

    @property
    def geometry(s):
        """Fetches a clone of the RegionMask's geometry as an OGR Geometry object

        * If a geometry was not provided when the RegionMask was initialized, then one will be generated from RegionMask's mask matrix in the RegionMask's extent
        * The geometry can always be deleted and rebuild using the RegionMask.rebuildGeometry() function
        """

        if(s._geometry is None): s.buildGeometry()

        return s._geometry.Clone()

    def drawMask( s, **kwargs):
        """Draw the region on a matplotlib figure

        * All kwargs (except 'bounds') are passed on to raster.drawMask()"""
        return drawImage( s.mask, bounds=s.extent.xyXY, **kwargs )

    def drawGeometry( s, **kwargs):
        """Draw the region on a matplotlib figure

        * All kwargs passed on to geom.drawPolygons()"""
        return drawGeoms( s.geometry, srs=s.srs, **kwargs )

    def _tempFile(s, head="tmp", ext=".tif"):
        """***RM INTERNAL***

        Use this to create a temporary file associated with the RegionMask which will be deleted when the RM goes out of scope.

        !! BEWARE OF EXTERNAL DEPENDANCIES WHEN THE RM IS GOING OUT OF SCOPE, THIS WILL CAUSE A LOT OF ISSUES !!
        """
        if(not hasattr(s,"_TMPDIR")):
            # Create a temporary directory to use with this shape (and associated processes)
            s._TMPDIR = TemporaryDirectory()
        return NamedTemporaryFile(suffix=ext, prefix=head, dir=s._TMPDIR.name, delete=True).name

    def __del__(s):
        if(hasattr(s, "_TMPDIR")): s._TMPDIR.cleanup()

    def createRaster(s, resolutionDiv=1, **kwargs):
        """Creates a new raster with the same extent and resolution as the parent Mask, but with a optional 
           datatype and filling

        Inputs:
            resolutionDiv -int : A division factor for the raster's resolution
                * Determines the resolution of the "divided" raster
                * Generally this is only used for internal purposes
            
            **kwargs 
                * All keyword arguments are passed on to geokit.raster.createRaster 
                * 'extent', 'pixelWidth', 'pixelHeight', and 'srs' are automatically defined

        """
        # protect from bad inputs
        trash = kwargs.pop("bounds", None)
        trash = kwargs.pop("pixelHeight", None)
        trash = kwargs.pop("pixelWidth", None)
        trash = kwargs.pop("srs", None)

        # Check for division
        if(resolutionDiv<0): resolutionDiv = 1.0/abs(resolutionDiv)
        pW, pH = s.pixelWidth/resolutionDiv, s.pixelHeight/resolutionDiv

        rasDS = createRaster( bounds=s.extent.xyXY, pixelWidth=pW, pixelHeight=pH, srs=s.srs, **kwargs)

        # Return result
        if("output" in kwargs):
            return
        else:
            rasDS.FlushCache()
            return rasDS

    def applyMask(s, mat, noData=0):
        """Shortcut to apply the RegionMask's mask to an array

        * When the passed matrix does not have the same extents of the given matrix, it is assumed that the RM's 
          mask needs to be scaled so that the matrix dimensions match

        * The RM's mask can only be scaled UP, and the given matrix's dimensions must be mutiples of the mask's 
          dimensions

        Inputs:
            mat - np.ndarray : The matrix to apply the mask to
                * A 2D numpy array whose dimensions equal, or are multiples of, the mask's dimensions

            noData - float : The no-data value to set into matrix's values which are not within the region
                
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

    ##############################################################################    
    # Warp raster onto region
    def warp(s, source, dtype=None, resampleAlg='cubic', noDataValue=None, applyMask=True, resolutionDiv=1, **kwargs):
        """Warp a given raster source onto the RM's extent and resolution

        * The source is not clipped around the RM's extent before the warping procedure. This isn't necessary, but if it 
          is desired it is suggested to call '<RegionMaskObject>.extent.clipRaster' (giving the source as an input) and
          then passing the returned value from that into the source input. 
            Ex.
                >>> clippedDS = RM.extent.clipRaster(<source>)
                >>> warpedMatrix = RM.warp( clippedDS )

        Returns a 2D matrix of the warped data fitting the RM's mask

        Inputs:
            source : The raster datasource to warp
                - str -- A path on the filesystem 
                - gdal Dataset

            dtype : The datatype of the warped data
                - str : The datatype as a numpy-readable string
                - type : The data type as a native python type

            resampleAlg - str : The resampling algorithm to use while warping
                * See gdal.WarpOptions for more info
                * The common options are: 'cubic', 'linear', or 'near'
                * Choose the resampling algorithm carefully by considering the raster source's contents!

            noDataValue - float : The no-data-value to use for the warped data
                * Will be cast into the given dtype if one was provided

            applyMask - True/False : Flag determining whether or not to apply the RM's mask to the resulting data

            resolutionDiv - int : A resolution scaling factor to allow for warping onto the same extent as the RM, 
                                  but a higher resolution
                * Generally this is intended for internal use

            **kwargs :
                * Extra keyword arguments are passed on to a call to gdal.Warp()
        """
        # open source and get info
        source = loadRaster(source)
        dsInfo = rasterInfo(source)
        workingExtent = Extent._fromInfo(dsInfo)

        # Test is the extents already match, and a projection is not necessary
        if ( workingExtent == s.extent and isclose(dsInfo.dx, s.pixelWidth/resolutionDiv) and isclose(dsInfo.dy, s.pixelHeight/resolutionDiv)):

            # Read the array
            final = source.GetRasterBand(1).ReadAsArray()

            # Make sure we have the desired datatype
            if(dtype):
                final = final.astype(dtype)

        else: # A projection is required
            # Create a target raster in shape of the mask
            dtype = dtype if dtype else dsInfo.dtype

            targetDS = s.createRaster(resolutionDiv, dtype=dtype)
            if(targetDS is None): raise GeoKitRegionMaskError("Error creating temporary mask-like matrix")

            # Warp working raster to the mask-like raster
            mt = kwargs.pop("multithread", True)
            r = gdal.Warp(targetDS, source, multithread=mt, resampleAlg=resampleAlg, **kwargs)
            targetDS.FlushCache()

            # Get resulting array
            final = targetDS.GetRasterBand(1).ReadAsArray()

            # clean up
            del targetDS

        # do some cleanup
        del source

        # Apply mask, maybe
        if(applyMask): final = s.applyMask(final, noDataValue)
    
        # Return
        return final

    def rasterize(s, source, applyMask=True, dtype="bool", noDataValue=None, resolutionDiv=1, **kwargs):
        """Rasterizes a given vector source onto the RM's extent and resolution. The rasterized data can simply 
        represent inclusion in the vector's features, or it can represent a specific value from the vector features'
        attributes

        Inputs:
            source : The vector source to extract from
                - path : A path on the file system
                - ogr Dataset
            
            applyMask - True/False : Flag determining whether or not to apply the RM's mask to the resulting data

            dtype : The datatype of the rasterized data
                - str : The datatype as a numpy-readable string
                - type : The data type as a native python type
            
            noDataValue - float : The no-data-value to use for the warped data
                * Will be cast into the given dtype if one was provided

            resolutionDiv - int : A resolution scaling factor to allow for rasterizing onto the same extent as the RM, 
                                  but at a higher resolution
                * It is particularly useful in the case when vector features are small compared to the RM's resolution 
                  (such as buildings!)

            **kwargs:
                * All kwargs are passed on as options in a call to gdal.Rasterize()
                * See gdal.RasterizeOptions for more info
                * If neither an 'attributes' or a 'burnValues' option are given, a 'burnValues' option is added 
                  equaling [1]
                * Most notably, the "where" option can be used to filter and select particular features from the 
                  source. Again, see gdal.RasterizeOptions for more info
        """
        # load the vector source
        source = loadVector(source)

        # Create temporary output file
        outputDS = s.createRaster(resolutionDiv, dtype=dtype)

        # Do rasterize
        bands = kwargs.pop("bands",[1])
        if(not( "attribute" in kwargs or "burnValues" in kwargs)):
            kwargs["burnValues"] = [1]

        tmp = gdal.Rasterize( outputDS, source, bands=bands, **kwargs)
        if(tmp==0):
            raise GeoKitRegionMaskError("Rasterization failed!")
        outputDS.FlushCache()

        # Read array
        final = outputDS.GetRasterBand(1).ReadAsArray()

        # Apply mask, maybe
        if(applyMask): final = s.applyMask(final, noDataValue)
               
        # clean up
        del outputDS

        # Return
        return final
    
    #######################################################################################
    ## Raw value processor
    def indicateValues(s, source, value, nanFill=None, forceMaskShape=True, buffer=None, **kwargs):
        """
        Indicates those pixels in the RegionMask which correspond to a particular value, or range of values, from a 
        given raster datasource

        Returns a matrix matching the RegionMask's mask dimensions wherein 0 means the pixels is not included in 
        the indicated set, and 1 meaning the pixel is included in the indicated set. Intermediate values are also 
        possible. This results from a scenario when the datasource's resolution does not line up perfectly
        with the RegionMask's resolution and, as a result, a RegionMask pixel overlaps multiple datasource pixels which
        are not all indicated (or not-indicated). 
        
        * Value processing is performed BEFORE a warp takes place (if one is necesary)
        * Output from the warp is clipp to values between 0 and 1
        * If a boolean matrix is desired of the result, use "result > 0.5"

        Inputs:
            source : The raster datasource to indicate from
                - str : A path on the filesystem 
                - gdal Dataset

            value : The value or range of values to indicate
                - float : The value to accept
                - float, float : The inclusive Min and Max values to accept
                    * None refers to no bound
                    * Ex. (None, 5) -> "Indicate all values equal to and below 5"

            forceMaskShape - True/False : Forces the returned matrix to have the same dimension as the RegionMask's mask
                * Only has effect when a resolutioDiv input is given

            kwargs -- Passed on to RegionMask.warp()
                * Most notably: resampleAlg
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
            ## fill nan values, maybe
            #if(not nanFill is None): data[ np.isnan(data) ] = nanFill
            
            output = np.ones(data.shape, dtype="bool")
            
            if(not valueMin is None):
                np.logical_and(data >= valueMin, output, output)
            if(not valueMax is None):
                np.logical_and(data <= valueMax, output, output)
            if(not valueEquals is None):
                np.logical_and(data == valueEquals, output, output)

            return output

        # Do processing
        clippedDS = s.extent.clipRaster(source)
        processedDS = mutateValues(clippedDS, processor=processor, dtype="bool")

        # Warp onto region
        final = s.warp(processedDS, dtype="float32", **kwargs)

        # Apply a buffer if requested
        if not buffer is None:
            geoms = convertMask(final>0.5, bounds=s.extent, srs=s.srs)
            if len(geoms)>0:
                geoms = [g.Buffer(buffer) for g in geoms]
                areaDS = createVector(geoms)
                final = s.rasterize( areaDS, dtype="bool", bands=[1], burnValues=[1], **kwargs )
            else:
                return np.zeros(s.mask.shape, dtype=bool)

        # apply a threshold incase of funky warping issues
        final[final>1.0] = 1
        final[final<0.0] = 0

        # Make sure we have the mask's shape
        if forceMaskShape:
            rd = kwargs.get("resolutionDiv",None)
            if not rd is None:
                final = scaleMatrix(final, -1*rd)

        # Return result
        return final

    #######################################################################################
    ## Vector feature indicator
    def indicateFeatures(s, dataSet, attribute=None, values=None, forceMaskShape=True, buffer=None, bufferMethod='geom', **kwargs):
        """
        Indicates the RegionMask pixels which are found within the features (or a subset of the features) contained
        in a given vector datasource

        * A Rasterization is performed from the input data set to the RM's mask.
            -See "gdal.RasterizeOptions" kwargs for info on how to control the rasterization step

        Arguments:
            source : The vector source to extract from
                - path : A path on the file system
                - ogr Dataset

            attribute - str : An optional name of a column to fliter the vector features by

            values - list : An optional list of values to accept when filtering the vector features

            forceMaskShape - True/False : Forces the returned matrix to have the same dimension as the RegionMask's mask
                * Only has effect when a resolutioDiv input is given
                
            kwargs -- Passed on to RegionMask.rasterize()
                * Most notably: 'where', 'resolutionDiv', and 'allTouched'
                * For more fine-grained control over the attribute filtering, leave 'attribute' and 'values' as None
                  and use the 'where' keyword
        """
        # Ensure path to dataSet exists
        if( not isinstance(dataSet, gdal.Dataset) and (not os.path.isfile(dataSet))):
            msg = "dataSet path does not exist: {}".format(dataSet)
            raise ValueError(msg)
        
        # Create a where statement if needed
        if attribute:
            where = ""
            for value in values:
                if isinstance(value,str):
                    where += "%s='%s' OR "%(attribute, value)
                elif isinstance(value,int):
                    where += "%s=%d OR "%(attribute, value)
                elif isinstance(value,float):
                    where += "%s=%f OR "%(attribute, value)
                else:
                    raise GeoKitRegionMaskError("Could not determine value type")
            where = where[:-4]
            kwargs["where"] = where
        
        # Do we need to buffer?
        if not buffer is None and bufferMethod == 'geom':
            def doBuffer(geom,attr): return geom.Buffer(buffer)
            dataSet = mutateFeatures(dataSet, srs=s.srs, geom=s.geometry, where=kwargs.pop("where",None),
                                     processor = doBuffer )

        # Do rasterize
        final = s.rasterize( dataSet, dtype="bool", bands=[1], burnValues=[1], **kwargs )

        # maybe we want to do the other buffer method
        if not buffer is None and bufferMethod == 'area':
            geoms = convertMask(final>0.5, bounds=s.extent, srs=s.srs)
            if len(geoms)>0:
                geoms = [g.Buffer(buffer) for g in geoms]
                dataSet = createVector(geoms)
                final = s.rasterize( dataSet, dtype="bool", bands=[1], burnValues=[1], **kwargs )
            else:
                return np.zeros(s.mask.shape, dtype=bool)

        # Make sure we have the mask's shape
        if forceMaskShape:
            rd = kwargs.get("resolutionDiv",None)
            if not rd is None:
                final = scaleMatrix(final, -1*rd)

        # Return
        return final

    #######################################################################################
    ## Vector feature indicator
    def indicateGeoms(s, geom, **kwargs):
        """
        Convenience function to indicate values found within a geometry (or a list of geometries)

        * Simply creates a new vector source from the given geometry and then calls RegionMask.indicateFeatures
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
        ds = createVector(geoms)

        # Indicate features
        return s.indicateFeatures(ds)
        
        '''
        # Do rasterize
        final = s.rasterize( ds, dtype="bool", bands=[1], burnValues=[1], **kwargs )

        # Make sure we have the mask's shape
        if forceMaskShape:
            rd = kwargs.get("resolutionDiv",None)
            if not rd is None:
                final = scaleMatrix(final, -1*rd)

        # Return
        return final
        '''