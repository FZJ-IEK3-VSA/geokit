import numpy as np
from osgeo import ogr
import re
from tempfile import TemporaryDirectory, NamedTemporaryFile
from collections import namedtuple
from io import BytesIO

from . import util as UTIL
from . import srs as SRS
from . import geom as GEOM
from . import raster as RASTER
from . import vector as VECTOR
# from .location import Location, LocationSet
from .extent import Extent


class GeoKitRegionMaskError(UTIL.GeoKitError):
    pass


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

    def __init__(self, extent, pixelRes, mask=None, geom=None, attributes=None, **kwargs):
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
        if mask is None and geom is None:
            raise GeoKitRegionMaskError(
                "Either mask or geom should be defined")
        if not kwargs.get("mask_plus_geom_is_okay", False):
            if not mask is None and not geom is None:
                raise GeoKitRegionMaskError(
                    "mask and geom cannot be defined simultaneously")

        # Set basic values
        self.extent = extent
        self.srs = extent.srs

        if self.srs is None:
            raise GeoKitRegionMaskError("Extent SRS cannot be None")

        # Set Pixel Size)
        if not extent.fitsResolution(pixelRes):
            raise GeoKitRegionMaskError(
                "The given extent does not fit the given pixelRes")

        try:
            pixelWidth, pixelHeight = pixelRes
        except:
            pixelWidth, pixelHeight = pixelRes, pixelRes

        self.pixelWidth = abs(pixelWidth)
        self.pixelHeight = abs(pixelHeight)

        if(self.pixelHeight == self.pixelWidth):
            self._pixelRes = self.pixelHeight
        else:
            self._pixelRes = None

        # set height and width
        # It turns out that I can't set these values here, since sometimes gdal
        # functions can add an extra row (due to float comparison issues?) when
        ## warping and rasterizing
        # int(np.round((self.extent.xMax-s.extent.xMin)/s.pixelWidth))
        self.width = None
        # int(np.round((self.extent.yMax-s.extent.yMin)/s.pixelHeight))
        self.height = None

        # Set mask
        self._mask = mask
        if not mask is None:  # test the mask
            # test type
            if(mask.dtype != "bool" and mask.dtype != "uint8"):
                raise GeoKitRegionMaskError("Mask must be bool type")
            if(mask.dtype == "uint8"):
                mask = mask.astype("bool")

            if not np.isclose(extent.xMin + pixelWidth * mask.shape[1], extent.xMax) or not np.isclose(extent.yMin + pixelHeight * mask.shape[0], extent.yMax):
                raise GeoKitRegionMaskError(
                    "Extent and pixels sizes do not correspond to mask shape")

        # Set geometry
        if not geom is None:  # test the geometry
            if not isinstance(geom, ogr.Geometry):
                raise GeoKitRegionMaskError(
                    "geom is not an ogr.Geometry object")

            self._geometry = geom.Clone()
            gSRS = geom.GetSpatialReference()
            if gSRS is None:
                raise GeoKitRegionMaskError("geom does not have an srs")

            if not gSRS.IsSame(self.srs):
                GEOM.transform(self._geometry, toSRS=self.srs, fromSRS=gSRS)
        else:
            self._geometry = None

        # Set other containers
        self._vector = None
        self._vectorPath = None

        # set attributes
        self.attributes = {} if attributes is None else attributes

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
        pixelWidth = (extent.xMax - extent.xMin) / (mask.shape[1])
        pixelHeight = (extent.yMax - extent.yMin) / (mask.shape[0])

        return RegionMask(extent=extent, pixelRes=(pixelWidth, pixelHeight), mask=mask, attributes=attributes)

    @staticmethod
    def fromGeom(geom, pixelRes=DEFAULT_RES, srs=DEFAULT_SRS, extent=None, padExtent=DEFAULT_PAD, attributes=None, **k):
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
        srs = SRS.loadSRS(srs)
        # make sure we have a geometry with an srs
        if(isinstance(geom, str)):
            geom = GEOM.convertWKT(geom, srs)

        geom = geom.Clone()  # clone to make sure we're free of outside dependencies

        # set extent (if not given)
        if extent is None:
            extent = Extent.fromGeom(geom).castTo(
                srs).pad(padExtent).fit(pixelRes)
        else:
            if not extent.srs.IsSame(srs):
                raise GeoKitRegionMaskError(
                    "The given srs does not match the extent's srs")
            #extent = extent.pad(padExtent)

        # make a RegionMask object
        return RegionMask(extent=extent, pixelRes=pixelRes, geom=geom, attributes=attributes)

    @staticmethod
    def fromVector(source, where=None, geom=None, pixelRes=DEFAULT_RES, srs=DEFAULT_SRS, extent=None, padExtent=DEFAULT_PAD, limitOne=True, **kwargs):
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

        limitOne : bool; optional
            Whether or not to allow more than one feature to be extracted

        Returns:
        --------
        RegionMask

        """
        # Get all geoms which fit the search criteria
        if isinstance(where, int):
            geom, attr = VECTOR.extractFeature(
                source=source, where=where, srs=srs)
        else:
            ftrs = list(VECTOR.extractFeatures(source=source,
                                               where=where, srs=srs, asPandas=False))

            if len(ftrs) == 0:
                raise GeoKitRegionMaskError("Zero features found")
            elif len(ftrs) == 1:
                geom = ftrs[0].geom
                attr = ftrs[0].attr
            else:
                if limitOne:
                    raise GeoKitRegionMaskError(
                        "Multiple fetures found. If you are okay with this, set 'limitOne' to False")
                geom = GEOM.flatten([f.geom for f in ftrs])
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
        if isinstance(region, RegionMask):
            return region
        elif isinstance(region, str):
            return RegionMask.fromVector(region, **kwargs)
        elif isinstance(region, ogr.Geometry):
            return RegionMask.fromGeom(region, **kwargs)
        elif isinstance(region, np.ndarray):
            return RegionMask.fromMask(region, **kwargs)
        else:
            raise GeoKitRegionMaskError("Could not understand region input")

    @property
    def pixelRes(self):
        """The RegionMask's pixel size. 

        !!Only available when pixelWidth equals pixelHeight!!"""
        if self._pixelRes is None:
            raise GeoKitRegionMaskError(
                "pixelRes only accessable when pixelWidth equals pixelHeight")
        return self._pixelRes

    def buildMask(self, **kwargs):
        """Explicitly build the RegionMask's mask matrix. 

        * The 'width' and 'height' attributes for the RegionMask are also set
          when this function is called
        * All kwargs are passed on to a call to geokit.vector.rasterize()

        """
        if self._geometry is None:
            raise GeoKitRegionMaskError(
                "Cannot build mask when geometry is None")

        self._mask = None
        self._mask = self.rasterize(self.vectorPath, applyMask=False,
                                    **kwargs).astype(np.bool)
        self.height, self.width = self._mask.shape

    @property
    def mask(self):
        """The RegionMask's mask array as an 2-dimensional boolean numpy array.

        * If no mask was given at the time of the RegionMask's creation, then a 
          mask will be generated on first access to the 'mask' property
        * The mask can be rebuilt in a customized way using the 
          RegionMask.buildMask() function
        """
        if(self._mask is None):
            self.buildMask()
        return self._mask

    @property
    def area(self):
        return self.mask.sum() * self.pixelWidth * self.pixelHeight

    def buildGeometry(self):
        """Explicitly build the RegionMask's geometry"""
        if self._mask is None:
            raise GeoKitRegionMaskError(
                "Cannot build geometry when mask is None")

        self._geometry = None
        self._geometry = GEOM.polygonizeMask(
            self.mask, bounds=self.extent.xyXY, srs=self.extent.srs, flat=True)

    @property
    def geometry(self):
        """Fetches a clone of the RegionMask's geometry as an OGR Geometry object

        * If a geometry was not provided when the RegionMask was initialized, 
          then one will be generated from the RegionMask's mask matrix in the 
          RegionMask's extent
        * The geometry can always be deleted and rebuild using the 
          RegionMask.rebuildGeometry() function
        """

        if(self._geometry is None):
            self.buildGeometry()

        return self._geometry.Clone()

    @property
    def vectorPath(self):
        """Returns a path to a vector path on disc which is built only once"""

        if(self._vectorPath is None):
            self._vectorPath = self._tempFile(ext=".shp")
            VECTOR.createVector(self.geometry, output=self._vectorPath)

        return self._vectorPath

    @property
    def vector(self):
        """Returns a vector saved in memory which is built only once"""

        if(self._vector is None):
            self._vector = UTIL.quickVector(self.geometry)

        return self._vector

    def _repr_svg_(self):
        if(not hasattr(self, "svg")):
            f = BytesIO()

            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 4))
            ax = plt.subplot(111)

            self.drawSelf(ax=ax)

            ax.set_aspect('equal')
            ax.autoscale(enable=True)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f, format="svg", dpi=100)
            plt.close()

            f.seek(0)
            self.svg = f.read().decode('ascii')

        return self.svg

    def _tempFile(self, head="tmp", ext=".tif"):
        """***RM INTERNAL***

        Use this to create a temporary file associated with the RegionMask which 
        will be deleted when the RM goes out of scope.

        !! BEWARE OF EXTERNAL DEPENDANCIES WHEN THE RM IS GOING OUT OF SCOPE, 
        THIS WILL CAUSE A LOT OF ISSUES !!
        """
        if(not hasattr(self, "_TMPDIR")):
            # Create a temporary directory to use with this shape (and associated processes)
            self._TMPDIR = TemporaryDirectory()
        return NamedTemporaryFile(suffix=ext, prefix=head, dir=self._TMPDIR.name, delete=True).name

    def __del__(self):
        if(hasattr(self, "_TMPDIR")):
            self._TMPDIR.cleanup()

    def _resolve(self, div):
        if(div < 0):
            div = 1.0 / abs(int(div))
        return (self.pixelWidth / div, self.pixelHeight / div)

    def applyMask(self, mat, noData=0):
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
        if(noData is None):
            noData = 0
        # Get size
        Y, X = mat.shape

        # make output array
        out = np.array(mat)

        # Apply mask
        if(self.mask.shape == mat.shape):  # matrix dimensions coincide with mask's data
            out[~self.mask] = noData

        elif(Y > self.height and X > self.width):
            if(not Y % self.height == 0 or not X % self.width == 0):
                raise GeoKitRegionMaskError(
                    "Matrix dimensions must be multiples of mask dimensions")

            yScale = Y // self.height
            xScale = X // self.width

            scaledMask = UTIL.scaleMatrix(self.mask, (yScale, xScale))
            sel = np.where(~scaledMask)
            out[sel] = noData

        else:
            raise GeoKitRegionMaskError("Could not map mask onto matrix")

        return out

    #######################################################################################
    # Raw value processor
    def _returnBlank(self, resolutionDiv=1, forceMaskShape=False, applyMask=True, noData=None, **kwargs):
        # make output
        if not forceMaskShape and resolutionDiv > 1:
            yN = self.mask.shape[0] * int(resolutionDiv)
            xN = self.mask.shape[1] * int(resolutionDiv)
            output = np.zeros((yN, xN))
        else:
            output = np.zeros(self.mask.shape)

        # apply mask, maybe
        if applyMask:
            output = self.applyMask(output, noData)

        # Done
        return output

    def indicateValueToGeoms(self, source, value, contours=False, transformGeoms=True):
        """
        TODO: UPDATE ME
        """
        # Unpack value
        if isinstance(value, tuple):
            valueMin, valueMax = value
            valueEquals = None
        else:
            valueMin, valueMax = None, None
            valueEquals = value

        # make processor
        def processor(data):
            # Do processing
            if(not valueEquals is None):
                output = data == valueEquals
            else:
                output = np.ones(data.shape, dtype="bool")

                if(not valueMin is None):
                    np.logical_and(data >= valueMin, output, output)
                if(not valueMax is None):
                    np.logical_and(data <= valueMax, output, output)

            # Done!
            return output

        # Do processing
        newDS = self.extent.mutateRaster(
            source, processor=processor, dtype="bool", matchContext=False)

        # Make contours
        if contours:
            geomDF = self.extent.contoursFromRaster(
                newDS, [1], transformGeoms=False)
            geoms = geomDF[geomDF.ID == 1].geom
        else:
            geomDF = RASTER.polygonizeRaster(newDS)
            geoms = geomDF[geomDF.value == 1].geom

        if len(geoms) == 0:
            return []
        else:
            geoms = list(geoms)

        if transformGeoms:
            geoms = GEOM.transform(geoms, toSRS=self.srs)

        return geoms

    def indicateValues(self, source, value, buffer=None, resolutionDiv=1, forceMaskShape=False, applyMask=True, noData=None, resampleAlg='bilinear', warpDType=None, bufferMethod='area', preBufferSimplification=None, **kwargs):
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
        * Output from the warp is clipped to values between 0 and 1
        * If a boolean matrix is desired of the result, use "result > 0.5"

        Parameters:
        -----------
        source : str or gdal.Dataset
            The raster datasource to indicate from

        value : numeric, tuple, iterable or str
            The value, range, or set of values to indicate on
            * If float : The exact value to accept
            * If tuple : The inclusive range to accept. Given as (low,high)
              - Assumes exactly 2 values are present
              - If either value is "None", then the range is assumed to be unbounded on that side
            * If any other iterable : The list of exact values to accept 
            * If str : The formatted set of elements to accept 
              - Each element in the set is seperated by a "," 
              - Each element must be either a singluar numeric value, or a range
              - A range element begins with either "[" or "(", and ends with either "]" or ")"
                and should have an '-' in between
                - "[" and "]" imply inclusivity
                - "(" and ")" imply exclusivity
                - Numbers on either side can be omitted, impling no limit on that side
                - Examples:
                  - "[1-5]" -> Indicate values from 1 up to 5, inclusively
                  - "[1-5)" -> Indicate values from 1 up to 5, but not including 5
                  - "(1-]"  -> Indicate values above 1 (but not including 1) up to infinity
                  - "[-5]"  -> Indicate values from negative infinity up to and including 5
                  - "[-]"   -> Indicate values from negative infinity to positive infinity (dont do this..)
              - All whitespaces will be ignored (so feel free to use them as you wish)
              - Example:
                - "[-2),[5-7),12,(22-26],29,33,[40-]" will indicate all of the following:
                  - Everything below 2, but not including 2
                  - Values between 5 up to 7, but not including 7
                  - 12
                  - Values above 22 up to and including 26
                  - 29
                  - 33
                  - Everything above 40, including 40


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

        resampleAlg : str; optional
            The resampling algorithm to use when warping values
            * Options are: 'near', 'bilinear', 'cubic', 'average', 'mode', 'max', 'min'
            * Knowing which option to use can have significant impacts!
                When indicating from a low resolution raster (relative to the region mask),
                then it is best to use one of 'near', 'bilinear', or 'cubic'. However, 
                when indicating from a high resolution raster file (again, relative to the region 
                mask) then one of 'average', 'mode', 'max', or 'min' is likely better.

        warpDType : str or None; optional 
            If given, this controls the raster datatype of the warped indication matrix.
            If not given, then a default datatype is assumed based off `resampleAlg`:
               reampleAlg : assumed dtype
               ----------   -------------
                   'near' : 'uint8'
               'bilinear' : 'float32'
                  'cubic' : 'float32'
                'average' : 'float32'
                   'mode' : 'uint8'
                    'max' : 'uint8'
                    'min' : 'uint8'

        forceMaskShape : bool 
            If True, forces the returned matrix to have the same dimension as 
            the RegionMask's mask regardless of the 'resolutionDiv' argument

        applyMask : bool
            When True, the RegionMask's mask will be applied to the outputData
            as described by RegionMask.applyMask

        noData : numeric
            The noData value to use when applying the mask

        geomsFromContours: bool
            If True, then geometries will be constructed from the function 
            geokit.RegionMask.contoursFromMatrix, as opposed to using
            geokit.RegionMask.polygonizeMask.
            - This will result in simpler geometries which are easier to grow, 
              but which do not strictly follow the edges of the indicated pixels


        bufferMethod : str; optional
            An indicator determining the method to use when buffereing
            * Options are: 'area' and 'contour'
            * If 'area', the function will first rasterize the raw geometries and
              will then apply the buffer to the indicated pixels
              - Uses geokit.RegionMask.polygonizeMask
              - This is the safer option although is not as accurate as the 'geom'
                option since it does not capture the exact edges of the geometries
              - This method can be made more accurate by increasing the 
                'resolutionDiv' input
            * If 'contour', the function will still rasterize the raw geometries, 
              but will then create geometries via mask contours (not the explicit
              pixel edges)
              - Uses geokit.RegionMask.contoursFromMatrix 
              - This option will recreate geometries which are more similar to the 
                original geometries compared to the 'area' method 
              - This method can be made more accurate by increasing the 
                'resolutionDiv' input

        preBufferSimplification: numeric
            If given, then geometries will be simplified (using ogr.Geometry.Simplify)
            using the specified value before being buffered
            - Using this can drastically decrease the time it takes to perform the 
              bufferring procedure, but can decrease accuracy if it is too high

        kwargs -- Passed on to RegionMask.warp()
            * Most notably: 'resampleAlg'


        Returns:
        --------
        numpy.ndarray
        """
        assert bufferMethod in ['area', 'contour']

        # format value input
        if isinstance(value, str):
            pass
        elif isinstance(value, tuple):  # Assume a range in implied
            value = "[{}-{}]".format(
                "" if value[0] is None else value[0],
                "" if value[1] is None else value[1],
            )
        else:
            try:  # Try treating value as an iterable
                _value = ""
                for v in value:
                    _value += "{},".format(v)
                value = _value[:-1]
            except TypeError:  # Value should be just a number
                value = str(value)

        # make processor
        def processor(data):
            # Find nan values, maybe
            if(not noData is None):
                nodat = np.isnan(data)

            # Indicate value elements
            output = np.zeros(data.shape, dtype="bool")
            value_re = re.compile(r"(?P<range>(?P<open>[\[\(])(?P<low>[-+]?(\d*\.\d+|\d+\.?))?-(?P<high>[-+]?(\d*\.\d+|\d+\.?))?(?P<close>[\]\)]))|(?P<value>[-+]?(\d*\.\d+|\d+\.?))")
            for element in value.split(","):
                element = element.replace(" ", "")
                if element == "":
                    continue

                m = value_re.match(element)
                if m is None or (m['value'] is None and m['range'] is None):
                    raise RuntimeError('The element "{}" does not match an expected format'.format(element))

                if m['value'] is not None:
                    update_sel = data == float(m['value'])

                else:  # We are dealing with a range
                    update_sel = np.ones(data.shape, dtype="bool")

                    if m['low'] is not None and m['open'] == "[":
                        np.logical_and(data >= float(m['low']), update_sel, update_sel)

                    elif m['low'] is not None and m['open'] == "(":
                        np.logical_and(data > float(m['low']), update_sel, update_sel)

                    if m['high'] is not None and m['close'] == "]":
                        np.logical_and(data <= float(m['high']), update_sel, update_sel)

                    elif m['high'] is not None and m['close'] == ")":
                        np.logical_and(data < float(m['high']), update_sel, update_sel)

                np.logical_or(update_sel, output, output)

            # Fill nan values, maybe
            if(not noData is None):
                output[nodat] = noData

            # Done!
            return output

        # Do processing
        newDS = self.extent.mutateRaster(
            source, processor=processor, dtype="uint8", noData=noData, matchContext=False)

        # Warp onto region
        if warpDType is None:
            if resampleAlg in ['bilinear', 'cubic', 'average']:
                warpDType = "float32"
            elif resampleAlg in ['near', 'mode', 'max', 'min']:
                warpDType = "uint8"
            else:
                warpDType = "float32"

        final = self.warp(newDS, dtype=warpDType, resolutionDiv=resolutionDiv, resampleAlg=resampleAlg,
                          applyMask=False, noData=noData, returnMatrix=True, **kwargs)

        # Check for results
        if not (final > 0).any():
            # no results were found
            return self._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape,
                                     applyMask=applyMask, noData=noData)

        # Apply a buffer if requested
        if not buffer is None:
            if bufferMethod == 'contour':
                geoms = self.contoursFromMask(final)
            elif bufferMethod == 'area':
                geoms = self.polygonizeMask(final > 0.5, flat=False)

            if preBufferSimplification is not None:
                geoms = [g.Simplify(preBufferSimplification) for g in geoms]

            if len(geoms) > 0:
                geoms = [g.Buffer(buffer) for g in geoms]
                areaDS = VECTOR.createVector(geoms)
                final = self.rasterize(areaDS, dtype="float32", bands=[1], burnValues=[1], resolutionDiv=resolutionDiv,
                                       applyMask=False, noData=noData)
            else:
                # no results were found
                return self._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape,
                                         applyMask=applyMask, noData=noData)

        # apply a threshold incase of funky warping issues
        final[final > 1.0] = 1
        final[final < 0.0] = 0

        # Make sure we have the mask's shape

        if forceMaskShape:
            if resolutionDiv > 1:
                final = UTIL.scaleMatrix(final, -1 * resolutionDiv)

        # Apply mask?
        if applyMask:
            final = self.applyMask(final, noData)

        # Return result
        return final

    #######################################################################################
    # Vector feature indicator
    def indicateFeatures(self, source, where=None, buffer=None, bufferMethod='geom', resolutionDiv=1, forceMaskShape=False, applyMask=True, noData=0, preBufferSimplification=None, **kwargs):
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
            * Options are: 'geom', 'area', and 'contour'
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
            * If 'contour', the function will still rasterize the raw geometries, 
              but will then create geometries via mask contours (not the explicit
              pixel edges)
              - This option will recreate geometries which are more similar to the 
                original geometries compared to the 'area' method 
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

        preBufferSimplification: numeric
            If given, then geometries will be simplified (using ogr.Geometry.Simplify)
            using the specified value before being buffered
            - Using this can drastically decrease the time it takes to perform the 
              bufferring procedure, but can decrease accuracy if it is too high


        kwargs -- Passed on to RegionMask.rasterize()
            * Most notably: 'allTouched'

        Returns:
        --------
        numpy.ndarray

        """
        assert bufferMethod in ['geom', 'area', 'contour']
        # Ensure path to dataSet exists
        source = VECTOR.loadVector(source)

        # Do we need to buffer?
        if buffer == 0:
            buffer = None
        if not buffer is None and bufferMethod == 'geom':
            def doBuffer(ftr):
                if preBufferSimplification is not None:
                    geom = ftr.geom.Simplify(preBufferSimplification)
                else:
                    geom = ftr.geom
                return {'geom': geom.Buffer(buffer)}
            source = self.mutateVector(source, where=where, processor=doBuffer,
                                       matchContext=True, keepAttributes=False, _slim=True)

            where = None  # Set where to None since the filtering has already been done

            if source is None:  # this happens when the returned dataset is empty
                return self._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape,
                                         applyMask=applyMask, noData=noData, **kwargs)

        # Do rasterize
        final = self.rasterize(source, dtype='float32', value=1, where=where, resolutionDiv=resolutionDiv,
                               applyMask=False, noData=noData)
        # Check for results
        if not (final > 0).any():
            # no results were found
            return self._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape,
                                     applyMask=applyMask, noData=noData)

        # maybe we want to do the other buffer method
        if not buffer is None and (bufferMethod == 'area' or bufferMethod == 'contour'):
            if bufferMethod == 'area':
                geoms = self.polygonizeMask(final > 0.5, flat=False)
            elif bufferMethod == 'contour':
                geoms = self.contoursFromMask(final)

            if preBufferSimplification is not None:
                geoms = [g.Simplify(preBufferSimplification) for g in geoms]

            if len(geoms) > 0:
                geoms = [g.Buffer(buffer) for g in geoms]
                dataSet = VECTOR.createVector(geoms)
                final = self.rasterize(dataSet, dtype="float32", bands=[1], burnValues=[1], resolutionDiv=resolutionDiv,
                                       applyMask=False, noData=noData)
            else:
                return self._returnBlank(resolutionDiv=resolutionDiv, forceMaskShape=forceMaskShape,
                                         applyMask=applyMask, noData=noData)

        # Make sure we have the mask's shape
        if forceMaskShape:
            if resolutionDiv > 1:
                final = UTIL.scaleMatrix(final, -1 * resolutionDiv)

        # Apply mask?
        if applyMask:
            final = self.applyMask(final, noData)

        # Return
        return final

    #######################################################################################
    # Vector feature indicator
    def indicateGeoms(self, geom, **kwargs):
        """
        Convenience wrapper to indicate values found within a geometry (or a 
        list of geometries)

        * Simply creates a new vector source from the given geometry and then 
          calls RegionMask.indicateFeatures
        * All keywords are passed on to RegionMask.indicateFeatures
        """
        # Make a vector dataset
        ds = UTIL.quickVector(geom)

        # Indicate features
        return self.indicateFeatures(ds, **kwargs)

    #######################################################################################
    # Make a sub region generator
    def subRegions(self, gridSize, asMaskAndExtent=False):
        """Generate a number of sub regions on a grid which combine into the total
        RegionMask area
        """
        # get useful matrix info
        yN, xN = self.mask.shape
        pixelGridSize = int(gridSize / min(self.pixelWidth, self.pixelHeight))

        # Make grid areas
        count = 0
        for ys in range(0, yN, pixelGridSize):

            yn = min(yN, ys + pixelGridSize)
            yMax = self.extent.yMax - ys * self.pixelHeight
            yMin = self.extent.yMax - yn * self.pixelHeight

            for xs in range(0, xN, pixelGridSize):
                xn = min(xN, xs + pixelGridSize)
                xMin = self.extent.xMin + xs * self.pixelWidth
                xMax = self.extent.xMin + xn * self.pixelWidth

                sectionMask = self.mask[ys:yn, xs:xn]
                if not sectionMask.any():
                    continue

                sectionExtent = Extent(xMin, yMin, xMax, yMax, srs=self.srs).fit(
                    (self.pixelWidth, self.pixelHeight))

                if asMaskAndExtent:
                    yield MaskAndExtent(sectionMask, sectionExtent, count)
                else:
                    yield RegionMask.fromMask(sectionExtent, sectionMask, dict(id=count))

                count += 1

    def subTiles(self, zoom, checkIntersect=True, asGeom=False):
        """Generates tile Extents at a given zoom level which encompass the envoking Regionmask.

        Parameters:
        -----------
        zoom : int
            The zoom level of the expected tile source

        checkIntersect : bool
            If True, exclude tiles which do not intersect with the RegionMask's geometry

        asGeom : bool
            If True, returns tuple of ogr.Geometries in stead of (xi,yi,zoom) tuples

        Returns:
        --------
        Generator of Geometries or (xi,yi,zoom) tuples

        """
        yield from GEOM.subTiles(self.geometry, zoom, checkIntersect=checkIntersect, asGeom=asGeom)

    #############################################################################
    # CONVENIENCE WRAPPERS
    def drawMask(self, ax=None, **kwargs):
        """Convenience wrapper around geokit.util.drawImage which plots the 
        RegionMask's mask over the RegionMask's context.

        * See geokit.util.drawImage for more info on argument options
        * Unless specified, the plotting extent is set to the RegionMask's extent
            - This only plays a role when generating a new axis

        """
        xlim = kwargs.pop("xlim", (self.extent.xMin, self.extent.xMax))
        ylim = kwargs.pop("ylim", (self.extent.yMin, self.extent.yMax))
        return UTIL.drawImage(self.mask, ax=ax, xlim=xlim, ylim=ylim, **kwargs)

    def drawImage(self, matrix, ax=None, drawSelf=True, **kwargs):
        """Convenience wrapper around geokit.util.drawImage which plots matrix data
        which is assumed to match the boundaries of the RegionMask

        * See geokit.util.drawImage for more info on argument options
        * Unless specified, the plotting extent is set to the RegionMask's extent
            - This only plays a role when generating a new axis

        """
        xlim = kwargs.pop("xlim", (self.extent.xMin, self.extent.xMax))
        ylim = kwargs.pop("ylim", (self.extent.yMin, self.extent.yMax))

        ax = UTIL.drawImage(matrix, ax=ax, xlim=xlim, ylim=ylim, **kwargs)

        if drawSelf:
            self.drawSelf(ax=ax, fc='None', ec='k', linewidth=2)
        return ax

    def drawGeoms(self, geoms, ax=None, drawSelf=True, **kwargs):
        """Convenience wrapper around geokit.geom.drawGeoms which plots geometries
        which are then plotted within the context of the RegionMask

        * See geokit.geom.drawGeoms for more info on argument options
        * Geometries are always plotted in the RegionMask's SRS
        * Unless specified, x and y limits are set to the RegionMask's extent
            - This only plays a role when generating a new axis
        """
        xlim = kwargs.pop("xlim", (self.extent.xMin, self.extent.xMax))
        ylim = kwargs.pop("ylim", (self.extent.yMin, self.extent.yMax))
        ax = GEOM.drawGeoms(geoms, ax=ax, srs=self.srs,
                            xlim=xlim, ylim=ylim, **kwargs)

        if drawSelf:
            self.drawSelf(ax=ax, fc='None', ec='k', linewidth=2)
        return ax

    def drawSelf(self, ax=None, **kwargs):
        """Convenience wrapper around geokit.geom.drawGeoms which plots the 
        RegionMask's geometry

        * See geokit.geom.drawGeoms for more info on argument options
        * Geometry are always plotted in the RegionMask's SRS
        * Unless specified, x and y limits are set to the RegionMask's extent
            - This only plays a role when generating a new axis
        """
        xlim = kwargs.pop("xlim", (self.extent.xMin, self.extent.xMax))
        ylim = kwargs.pop("ylim", (self.extent.yMin, self.extent.yMax))
        return GEOM.drawGeoms(self.geometry, ax=ax, srs=self.srs, xlim=xlim, ylim=ylim, **kwargs)

    def drawRaster(self, source, ax=None, drawSelf=True, **kwargs):
        """Convenience wrapper around geokit.raster.drawRaster which plots a raster
        dataset within the context of the RegionMask

        * See geokit.raster.drawRaster for more info on argument options
        * The raster is always warped to the RegionMask's SRS
        * Unless specified, x and y limits are set to the RegionMask's extent
            - This only plays a role when generating a new axis
        """
        xlim = kwargs.pop("xlim", (self.extent.xMin, self.extent.xMax))
        ylim = kwargs.pop("ylim", (self.extent.yMin, self.extent.yMax))
        ax = GEOM.drawGeoms(self.geometry, ax=ax, srs=self.srs,
                            xlim=xlim, ylim=ylim, **kwargs)

        if drawSelf:
            self.drawSelf(ax=ax, fc='None', ec='k', linewidth=2)
        return ax

    def createRaster(self, output=None, resolutionDiv=1, **kwargs):
        """Convenience wrapper for geokit.raster.createRaster which sets 'srs',
        'bounds', 'pixelWidth', and 'pixelHeight' inputs

        Parameters:
        -----------        
        output : str; optional
            A path to an output file to write to

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
        pW, pH = self._resolve(resolutionDiv)
        return self.extent.createRaster(pixelWidth=pW, pixelHeight=pH, output=output, **kwargs)

    def warp(self, source, output=None, resolutionDiv=1, returnMatrix=True, applyMask=True, noData=None, resampleAlg='bilinear', **kwargs):
        """Convenience wrapper for geokit.raster.warp() which automatically sets
        'srs', 'bounds', 'pixelWidth', and 'pixelHeight' inputs

        Note:
        -----
        When creating an 'in memory' raster vs one which is saved to disk, a slightly
        different algorithm is used which can sometimes add an extra row of pixels. Be
        aware of this if you intend to compare value-matricies directly from rasters 
        generated with this function.

        Parameters:
        -----------
        source : str
            The path to the raster file to warp

        output : str; optional
            A path to an output file to write to

        resampleAlg : str; optional
            The resampling algorithm to use when warping values
            * Knowing which option to use can have significant impacts!
            * Options are: 'nearesampleAlg=resampleAlg, r', 'bilinear', 'cubic', 
              'average'

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
        pW, pH = self._resolve(resolutionDiv)

        # do warp
        if returnMatrix:
            newDS = self.extent.warp(source=source, pixelWidth=pW, pixelHeight=pH,
                                     resampleAlg=resampleAlg, output=output, noData=noData, **kwargs)
        else:
            if applyMask:
                if "cutline" in kwargs:
                    raise GeoKitRegionMaskError(
                        "Cannot apply both a cutline and the mask when returning the warped dataset")
                newDS = self.extent.warp(source=source, pixelWidth=pW, pixelHeight=pH, resampleAlg=resampleAlg,
                                         cutline=self.vectorPath, output=output, noData=noData, **kwargs)
            else:
                newDS = self.extent.warp(source=source, pixelWidth=pW, pixelHeight=pH,
                                         resampleAlg=resampleAlg, output=output, noData=noData, **kwargs)

        if not returnMatrix:
            return newDS

        # Read array
        if newDS is None:
            newDS = output
        final = RASTER.extractMatrix(newDS)

        # clean up
        del newDS

        # Apply mask, maybe
        if(applyMask):
            final = self.applyMask(final, noData)

        # Return
        return final

    def rasterize(self, source, output=None, resolutionDiv=1, returnMatrix=True, applyMask=True, noData=None, **kwargs):
        """Convenience wrapper for geokit.vector.rasterize() which automatically
        sets the 'srs', 'bounds', 'pixelWidth', and 'pixelHeight' inputs 

        Note:
        -----
        When creating an 'in memory' raster vs one which is saved to disk, a slightly
        different algorithm is used which can sometimes add an extra row of pixels. Be
        aware of this if you intend to compare value-matricies directly from rasters 
        generated with this function.

        Parameters:
        -----------
        source : str
            The path to the vector file to load

        output : str; optional
            A path to an output file to write to

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
        pW, pH = self._resolve(resolutionDiv)

        # do rasterization
        if returnMatrix:
            newDS = self.extent.rasterize(
                source=source, pixelWidth=pW, pixelHeight=pH, output=output, noData=noData, **kwargs)
        else:
            if applyMask:
                if "cutline" in kwargs:
                    raise GeoKitRegionMaskError(
                        "Cannot apply both a cutline and the mask when returning the rasterized dataset")
                newDS = self.extent.rasterize(source=source, pixelWidth=pW, pixelHeight=pH,
                                              cutline=self.vectorPath, output=output, noData=noData, **kwargs)
            else:
                newDS = self.extent.rasterize(
                    source=source, pixelWidth=pW, pixelHeight=pH, output=output, noData=noData, **kwargs)

        if not returnMatrix:
            return newDS

        # Read array
        if newDS is None:
            newDS = output
        final = RASTER.extractMatrix(newDS)

        # clean up
        del newDS

        # Apply mask, maybe
        if(applyMask):
            final = self.applyMask(final, noData)

        # Return
        return final

    def extractFeatures(self, source, **kwargs):
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
        return VECTOR.extractFeatures(source=source, geom=self.geometry, **kwargs)

    def mutateVector(self, source, matchContext=False, **kwargs):
        """Convenience wrapper for geokit.vector.mutateVector which automatically
        sets 'srs' and 'geom' inputs to the RegionMask's srs and geometry

        * The RegionMask's geometry is always used to select features within the 
        source. If you need a broader scope, try using the RegionMask's extent's
        version of this function

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
        if not matchContext:
            vinfo = VECTOR.vectorInfo(source)
            ext = self.extent.castTo(vinfo.srs)
        else:
            ext = self.extent

        # mutate the source
        return VECTOR.mutateVector(source, srs=ext.srs, geom=self.geometry, **kwargs)

    def mutateRaster(self, source, matchContext=True, warpArgs=None, applyMask=True, processor=None, resampleAlg="bilinear", **mutateArgs):
        """Convenience wrapper for geokit.vector.mutateRaster which automatically
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

        resampleAlg : str; optional
            The resampling algorithm to use when warping values
            * Knowing which option to use can have significant impacts!
            * Options are: 'nearesampleAlg=resampleAlg, r', 'bilinear', 'cubic', 
              'average'

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

        **mutateArgs:
            All other keyword arguments are passed to geokit.vector.mutateVector

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None

        """
        output = mutateArgs.pop("output", None)
        if warpArgs is None:
            warpArgs = {}

        # Do the warp and mutation
        if matchContext:
            source = self.warp(source, returnMatrix=False,
                               applyMask=applyMask, resampleAlg=resampleAlg, **warpArgs)

            if processor is None:
                return source
            else:
                return RASTER.mutateRaster(source, output=output, **mutateArgs)

        else:
            if applyMask:
                if "cutline" in warpArgs:
                    raise GeoKitRegionMaskError(
                        "Cannot apply both a cutline and the mask during prewarping")
                warpArgs["cutline"] = self.vector

            return self.extent.mutateRaster(source, matchContext=False, warpArgs=warpArgs, resampleAlg=resampleAlg, **mutateArgs)

    def polygonizeMatrix(self, matrix, flat=False, shrink=True, _raw=False):
        """Convenience wrapper for geokit.geom.polygonizeMatrix which autmatically
        sets the 'bounds' and 'srs' inputs. The matrix data is assumed to span the
        RegionMask exactly.

        Each unique-valued group of pixels will be converted to a geometry

        Parameters:
        -----------
        matrix : matrix_like
            The matrix which will be turned into a geometry set
              * Must be 2 dimensional
              * Must be integer or boolean type

        flat : bool
            If True, flattens the resulting geometries which share a contiguous matrix
            value into a single geometry object

        shrink : bool
            If True, shrink all geoms by a tiny amount in order to avoid geometry 
            overlapping issues
              * The total amount shrunk should be very very small
              * Generally this should be left as True unless it is ABSOLUTELY 
                necessary to maintain the same area

        Returns:
        --------
        pandas.DataFrame -> With columns:
                                'geom' -> The contiguous-valued geometries
                                'value' -> The value for each geometry

        """
        return GEOM.polygonizeMatrix(matrix, bounds=self.extent.xyXY, srs=self.srs, flat=flat, shrink=shrink, _raw=_raw)

    def polygonizeMask(self, mask, bounds=None, srs=None, flat=True, shrink=True):
        """Convenience wrapper for geokit.geom.polygonizeMask which autmatically
        sets the 'bounds' and 'srs' inputs. The mask data is assumed to span the
        RegionMask exactly

        Each True-valued group of pixels will be converted to a geometry

        Parameters:
        -----------
        mask : matrix_like
            The mask which will be turned into a geometry set
              * Must be 2 dimensional
              * Must be boolean type
              * True values are interpreted as 'in the geometry'

        flat : bool
            If True, flattens the resulting geometries into a single geometry

        shrink : bool
            If True, shrink all geoms by a tiny amount in order to avoid geometry 
            overlapping issues
              * The total amount shrunk should be very very small
              * Generally this should be left as True unless it is ABSOLUTELY 
                neccessary to maintain the same area

        Returns:
        --------
        If 'flat' is True: ogr.Geometry
        else: [ogr.Geometry,  ]

        """
        return GEOM.polygonizeMask(mask, bounds=self.extent.xyXY, srs=self.srs, flat=flat, shrink=shrink)

    def contoursFromRaster(self, raster, contourEdges, applyMask=True, contoursKwargs={}, warpKwargs={}):
        """Convenience wrapper for geokit.raster.contours which automatically
        warps a raster to the invoking RegioNmask

        NOTE:
        -----
        * The raster is first warped to the RegionMask before the contours are
          determined. If this behavior is not desired, consider using the function
          Extent.contoursFromRaster

        Parameters:
        -----------
        raster : The raster datasource to warp from

        contourEdges : [float,]
            The edges to search for withing the raster dataset
            * This parameter can be set as "None", in which case an additional 
                argument should be given to specify how the edges should be determined
                - See the documentation of "GDALContourGenerateEx"
                - Ex. "LEVEL_INTERVAL=10", contourEdges=None

        contoursKwargs : dict
            Keyword arguments to pass on to the contours function
            * See geokit.raster.contours

        warpKwargs : dict
            Keyword arguments to pass on to the raster warp function
            * See geokit.RegionMask.warp

        Returns:
        --------
        pandas.DataFrame 

        With columns:
            'geom' -> The contiguous-valued geometries
            'ID' -> The associated contour edge for each object

        """
        raster = self.warp(raster, applyMask=applyMask,
                           returnMatrix=False, **warpKwargs)
        geoms = RASTER.contours(raster, contourEdges, **contoursKwargs)

        return geoms

    def contoursFromMatrix(self, matrix, contourEdges, contoursKwargs={}, createRasterKwargs={}):
        """Convenience wrapper for geokit.raster.contours which autmatically
        creates a raster for the given matrix (which is assumed to match the 
        domain of the RegionMask)

        Parameters:
        -----------
        matrix : matrix_like
            The matrix which will be turned into a geometry set
              * Must be 2 dimensional

        contourEdges : [float,]
            The edges to search for withing the raster dataset
            * This parameter can be set as "None", in which case an additional 
                argument should be given to specify how the edges should be determined
                - See the documentation of "GDALContourGenerateEx"
                - Ex. "LEVEL_INTERVAL=10", contourEdges=None

        contoursKwargs : dict
            Keyword arguments to pass on to the contours function
            * See geokit.raster.contours

        createRasterKwargs : dict
            Keyword arguments to pass on to the raster creation function
            * See geokit.RegionMask.createRaster

        Returns:
        --------
        pandas.DataFrame 

        With columns:
            'geom' -> The contiguous-valued geometries
            'ID' -> The associated contour edge for each object

        """
        raster = self.createRaster(data=matrix, **createRasterKwargs)
        geoms = RASTER.contours(raster, contourEdges, **contoursKwargs)

        return geoms

    def contoursFromMask(self, mask, truthThreshold=0.5, trueAboveThreshold=True, contoursKwargs={}, createRasterKwargs={}):
        """Convenience wrapper for geokit.raster.contours which autmatically
        creates a raster for the given mask (which is assumed to match the 
        domain of the RegionMask), and extracts the geometries which are indicated 
        in the mask as "True"

        Parameters:
        -----------
        mask : matrix_like
            The mask which will be turned into a geometry set
              * Must be 2 dimensional

        truthThreshold : [float,]
            The value which separates "True" from "False" values
            * Values are True when they are above the threshold unless 
              trueAboveThreshold is set as False

        trueAboveThreshold: bool
            If true, then pixels with values above the threshold are identified
            as "True"

        contoursKwargs : dict
            Keyword arguments to pass on to the contours function
            * See geokit.raster.contours

        createRasterKwargs : dict
            Keyword arguments to pass on to the raster creation function
            * See geokit.RegionMask.createRaster

        Returns:
        --------
        pandas.DataFrame 

        With columns:
            'geom' -> The contiguous-valued geometries
            'ID' -> The associated contour edge for each object

        """
        geomDF = self.contoursFromMatrix(matrix=mask, contourEdges=[truthThreshold, ],
                                         createRasterKwargs=createRasterKwargs,
                                         contoursKwargs=contoursKwargs)

        geoms = geomDF[geomDF.ID == (1 if trueAboveThreshold else 0)].geom

        return geoms
