import os
import sys
import numpy as np
from osgeo import gdal, ogr
from tempfile import TemporaryDirectory, NamedTemporaryFile
import warnings
from collections import OrderedDict, namedtuple
from collections.abc import Iterable
import pandas as pd
from scipy.interpolate import RectBivariateSpline

from . import util as UTIL
from . import srs as SRS
from . import geom as GEOM
from .location import Location, LocationSet


class GeoKitRasterError(UTIL.GeoKitError):
    pass


if("win" in sys.platform):
    COMPRESSION_OPTION = ["COMPRESS=LZW"]
    COMPRESSION_OPTION_STR = "LZW"
else:
    COMPRESSION_OPTION = ["COMPRESS=DEFLATE"]
    COMPRESSION_OPTION_STR = "DEFLATE"

# Basic Loader


def loadRaster(source, mode=0):
    """
    Load a raster dataset from a path to a file on disc

    Parameters:
    -----------
    source : str or gdal.Dataset
        * If a string is given, it is assumed as a path to a raster file on disc
        * If a gdal.Dataset is given, it is assumed to already be an open raster
          and is returned immediately

    Returns:
    --------
    gdal.Dataset

    """
    if(isinstance(source, str)):
        ds = gdal.Open(source, mode)
    else:
        ds = source

    if(ds is None):
        raise GeoKitRasterError(
            "Could not load input dataSource: ", str(source))
    return ds


# GDAL type mapper
_gdalIntToType = dict((v, k) for k, v in filter(
    lambda x: "GDT_" in x[0], gdal.__dict__.items()))
_gdalType = {bool: "GDT_Byte", int: "GDT_Int32", float: "GDT_Float64", "bool": "GDT_Byte",
             "int8": "GDT_Byte", "int16": "GDT_Int16", "int32": "GDT_Int32",
             "int64": "GDT_Int32", "uint8": "GDT_Byte", "uint16": "GDT_UInt16",
             "uint32": "GDT_UInt32", "float32": "GDT_Float32", "float64": "GDT_Float64"}


def gdalType(s):
    """Tries to determine gdal datatype from the given input type"""
    if(s is None):
        return "GDT_Unknown"
    elif(isinstance(s, str)):
        if(hasattr(gdal, s)):
            return s
        elif(s.lower() in _gdalType):
            return _gdalType[s.lower()]
        elif(hasattr(gdal, 'GDT_%s' % s)):
            return 'GDT_%s' % s
        elif(s == "float" or s == "int" or s == "bool"):
            return gdalType(np.dtype(s))

    elif(isinstance(s, int)):
        return _gdalIntToType[s]  # If an int is given, it's probably
        #  the GDAL type indicator (and not a
        #  sample data value)
    elif(isinstance(s, np.dtype)):
        return gdalType(str(s))
    elif(isinstance(s, np.generic)):
        return gdalType(s.dtype)
    elif(s is bool):
        return _gdalType[bool]
    elif(s is int):
        return _gdalType[int]
    elif(s is float):
        return _gdalType[float]
    elif(isinstance(s, Iterable)):
        return gdalType(s[0])
    raise GeoKitRasterError("GDAL type could not be determined")

####################################################################
# Raster writer


def createRaster(bounds, output=None, pixelWidth=100, pixelHeight=100, dtype=None, srs='europe_m', compress=True, noData=None, overwrite=True, fill=None, data=None, meta=None, scale=1, offset=0, creationOptions=dict(), **kwargs):
    """Create a raster file

    NOTE:
    -----
    Raster datasets are always written in the 'yAtTop' orientation. Meaning that 
    the first row of data values (either written to or read from the dataset) will 
    refer to the TOP of the defined boundary, and will then move downward from 
    there

    If a data matrix is given, and a negative pixelWidth is defined, the data 
    will be flipped automatically

    Parameters:
    -----------
    bounds : (xMin, yMix, xMax, yMax) or Extent
        The geographic extents spanned by the raster

    pixelWidth : numeric
        The pixel width of the raster in units of the input srs
        * The keyword 'dx' can be used as well and will override anything given 
        assigned to 'pixelWidth'

    pixelHeight : numeric
        The pixel height of the raster in units of the input srs
        * The keyword 'dy' can be used as well and will override anything given 
          assigned to 'pixelHeight'

    output : str; optional
        A path to an output file 
        * If output is None, the raster will be created in memory and a dataset 
          handel will be returned
        * If output is given, the raster will be written to disk and nothing will
          be returned

    dtype : str; optional
        The datatype of the represented by the created raster's band
        * Options are: Byte, Int16, Int32, Int64, Float32, Float64
        * If dtype is None and data is None, the assumed datatype is a 'Byte'
        * If dtype is None and data is not None, the datatype will be inferred 
          from the given data

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the point to create
          * If not given, longitude/latitude is assumed
          * srs MUST be given as a keyword argument
        * If 'bounds' is an Extent object, the bounds' internal srs will override
          this input

    compress : bool
        A flag instructing the output raster to use a compression algorithm
        * only useful if 'output' has been defined
        * "DEFLATE" used for Linux/Mac, "LZW" used for Windows

    noData : numeric; optional
        Specifies which value should be considered as 'no data' in the created 
        raster
        * Must be the same datatye as the 'dtype' input (or that which is derived)

    fill : numeric; optional
        The initial value given to all pixels in the created raster band
        - numeric
        * Must be the same datatye as the 'dtype' input (or that which is derived)

    overwrite : bool
        A flag to overwrite a pre-existing output file
        * If set to False and an 'output' is specified which already exists,
          an error will be raised

    data : matrix_like
        A 2D matrix to write into the resulting raster
        * array dimensions must fit raster dimensions as calculated by the bounds
          and the pixel resolution

    scale : numeric; optional
        The scaling value given to apply to all values
        - numeric
        * Must be the same datatye as the 'dtype' input (or that which is derived)

    offset : numeric; optional
        The offset value given to apply to all values
        - numeric
        * Must be the same datatye as the 'dtype' input (or that which is derived)

    Returns:
    --------
    * If 'output' is None: gdal.Dataset
    * If 'output' is a string: The path to the output is returned (for easy opening)

    """
    # Check for existing file
    if(not output is None):
        if(os.path.isfile(output)):
            if(overwrite == True):
                os.remove(output)
                if(os.path.isfile(output + ".aux.xml")):
                    os.remove(output + ".aux.xml")
            else:
                raise GeoKitRasterError(
                    "Output file already exists: %s" % output)

    # Ensure bounds is okay
    # bounds = UTIL.fitBoundsTo(bounds, pixelWidth, pixelHeight)

    # Make a raster dataset and pull the band/maskBand objects
    originX = bounds[0]
    originY = bounds[3]  # Always use the "Y-at-Top" orientation

    cols = int(round((bounds[2] - originX) / pixelWidth))
    rows = int(round((originY - bounds[1]) / abs(pixelHeight)))

    # Get DataType
    if(not dtype is None):  # a dtype was given, use it!
        dtype = gdalType(dtype)
    elif (not data is None):  # a data matrix was give, use it's dtype! (assume a numpy array or derivative)
        dtype = gdalType(data.dtype)
    else:  # Otherwise, just assume we want a Byte
        dtype = "GDT_Byte"

    # Open the driver
    opts = OrderedDict()
    if (compress and output is not None):
        opts['COMPRESS'] = COMPRESSION_OPTION_STR
    if creationOptions is not None:
        opts.update(creationOptions)
    opts = ["{}={}".format(k, v) for k, v in opts.items()]

    if(output is None):
        driver = gdal.GetDriverByName('Mem')  # create a raster in memory
        raster = driver.Create('', cols, rows, 1, getattr(gdal, dtype), opts)
    else:

        driver = gdal.GetDriverByName('GTiff')  # Create a raster in storage
        raster = driver.Create(output, cols, rows, 1,
                               getattr(gdal, dtype), opts)

    if(raster is None):
        raise GeoKitRasterError("Failed to create raster")

    # Do the rest in a "try" statement so that a failure wont bind the source
    try:
        raster.SetGeoTransform(
            (originX, abs(pixelWidth), 0, originY, 0, -1 * abs(pixelHeight)))

        # Set the SRS
        if not srs is None:
            rasterSRS = SRS.loadSRS(srs)
            raster.SetProjection(rasterSRS.ExportToWkt())

        # Fill the raster will zeros, null values, or initial values (if given)
        band = raster.GetRasterBand(1)
        if not scale is None:
            band.SetScale(scale)
        if not offset is None:
            band.SetOffset(offset)

        if(not noData is None):
            band.SetNoDataValue(noData)
            if fill is None and data is None:
                band.Fill(noData)

        if(data is None):
            if fill is None:
                band.Fill(0)
            else:
                band.Fill(fill)
        else:
            # make sure dimension size is good
            if not (data.shape[0] == rows and data.shape[1] == cols):
                raise GeoKitRasterError(
                    "Raster dimensions and input data dimensions do not match")

            # See if data needs flipping
            if pixelHeight < 0:
                data = data[::-1, :]

            # Write it!
            band.WriteArray(data)
            band.FlushCache()

            band.ComputeRasterMinMax(0)
            band.ComputeBandStats(0)

        raster.FlushCache()

        # Write MetaData, maybe
        if not meta is None:
            for k, v in meta.items():
                raster.SetMetadataItem(k, v)

        # Return raster if in memory
        if (output is None):
            return raster

        # Done
        return output

    # Handle the fail case
    except Exception as e:
        raster = None
        raise e


def createRasterLike(source, copyMetadata=True, **kwargs):
    """Create a raster described by the given raster info (as returned from a 
    call to rasterInfo() ). 

    * This copies all characteristics of the given raster, including: bounds, 
      pixelWidth, pixelHeight, dtype, srs, noData, and meta. 
    * Any keyword argument which is given will override values found in the 
      source

    """

    if UTIL.isRaster(source):
        source = rasterInfo(source)

    if not isinstance(source, RasterInfo):
        raise GeoKitRasterError("Could not understand source")

    bounds = kwargs.pop("bounds", source.bounds)
    pixelWidth = kwargs.pop("pixelWidth", source.pixelWidth)
    pixelHeight = kwargs.pop("pixelHeight", source.pixelHeight)
    dtype = kwargs.pop("dtype", source.dtype)
    srs = kwargs.pop("srs", source.srs)
    noData = kwargs.pop("noData", source.noData)

    if copyMetadata:
        meta = kwargs.pop("meta", source.meta)
    else:
        meta = None

    return createRaster(bounds=bounds, pixelWidth=pixelWidth, pixelHeight=pixelHeight, dtype=dtype, srs=srs,
                        noData=noData, meta=meta, **kwargs)


####################################################################
# extract the raster as a matrix
def extractMatrix(source, bounds=None, boundsSRS='latlon', maskBand=False, autocorrect=False, returnBounds=False):
    """extract all or part of a raster's band as a numpy matrix

    Note:
    -----
    Unless one is trying to get the entire matrix from the raster dataset, usage
    of this function requires intimate knowledge of the raster's characteristics. 
    In such a case it is probably easier to use Extent.extractMatrix

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource

    bounds: tuple or Extent
        The boundary to clip the raster to before mutating
        * If given as an Extent, the extent is always cast to the source's 
            - native srs before mutating
        * If given as a tuple, (xMin, yMin, xMax, yMax) is expected
            - Units must be in the srs specified by 'boundsSRS'
        * This boundary must fit within the boundary of the rasters source
        * The boundary is always fitted to the source's grid, so the returned
          values do not necessarily match to the boundary which is provided        

    boundsSRS: Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the 'bounds' argument
        * This is ignored if the 'bounds' argument is an Extent object or is None

    autocorrect : bool; optional
        If True, the matrix will search for no data values and change them to 
        numpy.nan
        * Data type will always result in a float, so be careful with large
          matricies

    returnBounds : bool; optional
        If True, return the computed bounds along with the matrix data

    Returns:
    --------
    * If returnBounds is False: numpy.ndarray -> Two dimensional matrix
    * If returnBounds is True: (numpy.ndarray, tuple)
        - ndarray is matrix data
        - tuple is the (xMin, yMin, xMax, yMax) of the computed bounds

    """
    sourceDS = loadRaster(source)  # BE sure we have a raster
    dsInfo = rasterInfo(sourceDS)

    if maskBand:
        mb = sourceDS.GetMaskBand()
    else:
        sourceBand = sourceDS.GetRasterBand(1)  # get band

    # Handle the boundaries
    if not bounds is None:
        # check for extent
        try:
            isExtent = bounds._whatami == "Extent"
        except:
            isExtent = False

        # Ensure srs is okay
        if isExtent:
            bounds = bounds.castTo(dsInfo.srs).fit((dsInfo.dx, dsInfo.dy)).xyXY
        else:
            boundsSRS = SRS.loadSRS(boundsSRS)
            if not dsInfo.srs.IsSame(boundsSRS):
                bounds = GEOM.boundsToBounds(bounds, boundsSRS, dsInfo.srs)
            bounds = UTIL.fitBoundsTo(bounds, dsInfo.dx, dsInfo.dy)

        # Find offsets
        xoff = int(np.round((bounds[0] - dsInfo.xMin) / dsInfo.dx))
        if xoff < 0:
            raise GeoKitRasterError(
                "The given boundary exceeds the raster's xMin value")

        xwin = int(np.round((bounds[2] - dsInfo.xMin) / dsInfo.dx)) - xoff
        if xwin > dsInfo.xWinSize:
            raise GeoKitRasterError(
                "The given boundary exceeds the raster's xMax value")

        if dsInfo.yAtTop:
            yoff = int(np.round((dsInfo.yMax - bounds[3]) / dsInfo.dy))
            if yoff < 0:
                raise GeoKitRasterError(
                    "The given boundary exceeds the raster's yMax value")

            ywin = int(np.round((dsInfo.yMax - bounds[1]) / dsInfo.dy)) - yoff

            if ywin > dsInfo.yWinSize:
                raise GeoKitRasterError(
                    "The given boundary exceeds the raster's yMin value")
        else:
            yoff = int(np.round((bounds[1] - dsInfo.yMin) / dsInfo.dy))
            if yoff < 0:
                raise GeoKitRasterError(
                    "The given boundary exceeds the raster's yMin value")

            ywin = int(np.round((bounds[3] - dsInfo.yMin) / dsInfo.dy)) - yoff
            if ywin > dsInfo.yWinSize:
                raise GeoKitRasterError(
                    "The given boundary exceeds the raster's yMax value")

    else:
        xoff = 0
        yoff = 0
        xwin = None
        ywin = None

    # get Data
    if maskBand:
        data = mb.ReadAsArray(xoff=xoff, yoff=yoff,
                              win_xsize=xwin, win_ysize=ywin)
    else:
        data = sourceBand.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=xwin, win_ysize=ywin)
        if dsInfo.scale is not None and dsInfo.scale != 1.0:
            data = data * dsInfo.scale
        if dsInfo.offset is not None and dsInfo.offset != 0.0:
            data = data + dsInfo.offset

    # Correct 'nodata' values
    if autocorrect:
        noData = sourceBand.GetNoDataValue()
        data = data.astype(np.float)
        data[data == noData] = np.nan

    # make sure we are returing data in the 'flipped-y' orientation
    if not isFlipped(source):
        data = data[::-1, :]

    # Done
    if returnBounds:
        return data, bounds
    else:
        return data


def rasterStats(source, cutline=None, ignoreValue=None, **kwargs):
    """Compute basic statistics of the values contained in a raster dataset. 

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource

    cutline : ogr.Geometry; optional
        The geometry over which to cut out the raster's data
        * Must be a Polygon or MultiPolygon

    ignoreValue : numeric
        A value to ignore when computing the statistics
        * If the raster source has a 'no Data' value, it is automatically 
          ignored

    **kwargs
        * All kwargs are passed on to warp() when 'geom' is given
        * See gdal.WarpOptions for more details
        * For example, 'allTouched' may be useful

    Returns:
    --------
    Results from a call to scipy.stats.describe

    """
    from scipy.stats import describe
    source = loadRaster(source)

    # Get the matrix to calculate over
    if not cutline is None:
        source = warp(source, cutline=cutline, noData=ignoreValue, **kwargs)

    rawData = extractMatrix(source)
    dataInfo = rasterInfo(source)

    # exclude nodata and ignore values
    sel = np.ones(rawData.shape, dtype='bool')

    if not ignoreValue is None:
        np.logical_and(rawData != ignoreValue, sel, sel)

    if not dataInfo.noData is None:
        np.logical_and(rawData != dataInfo.noData, sel, sel)

    # compute statistics
    data = rawData[sel].flatten()
    return describe(data)


####################################################################
# Gradient calculator
def gradient(source, mode="total", factor=1, asMatrix=False, **kwargs):
    """Calculate a raster's gradient and return as a new dataset or simply a matrix

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource

    mode : str; optional
        Determines the type of gradient to compute
        * Options are....
          "total" : Calculates the absolute gradient as a ratio

          "slope" : Same as 'total'

          "north-south" : Calculates the "north-facing" gradient as a ratio where
                          negative numbers indicate a south facing gradient 

          "east-west" : Calculates the "east-facing" gradient as a ratio where 
                        negative numbers indicate a west facing gradient

          "aspect" : calculates the gradient's direction in radians (0 is east)

          "dir" : same as 'aspect'

    factor : numeric or 'latlonToM'
        The scaling factor relating the units of the x & y dimensions to the z 
        dimension
        * If factor is 'latlonToM', the x & y units are assumed to be degrees 
          (lat & lon) and the z units are assumed to be meters. A factor is then 
          computed for coordinates at the source's center.
        * Example: If x,y units are meters and z units are feet, factor should 
          be 0.3048

    asMatrix : bool
        If True, makes the returned value a matrix 
        If False, makes the returned value a raster dataset

    **kwargs : All extra key word arguments are passed on to a final call to 
        'createRaster'
        * Only useful when 'asMatrix' is True

    Returns:
    --------
    * If 'asMatrix' is True: numpy.ndarray
    * If 'asMatrix' is False: gdal.Dataset

    """
    # Make sure source is a source
    source = loadRaster(source)

    # Check mode
    acceptable = ["total", "slope", "north-south",
                  "east-west", 'dir', "ew", "ns", "aspect"]
    if not (mode in acceptable):
        raise ValueError("'mode' not understood. Must be one of: ", acceptable)

    # Get the factor
    sourceInfo = rasterInfo(source)
    if factor == "latlonToM":
        latMid = (sourceInfo.yMax + sourceInfo.yMin) / 2
        R_EARTH = 6371000
        DEGtoRAD = np.pi / 180

        yFactor = R_EARTH * DEGtoRAD  # Get arc length in meters/Degree
        xFactor = R_EARTH * DEGtoRAD * np.cos(latMid * DEGtoRAD)  # ditto...
    else:
        try:
            xFactor, yFactor = factor
        except:
            yFactor = factor
            xFactor = factor

    # Calculate gradient
    arr = extractMatrix(source)

    if mode in ["north-south", "ns", "total", "slope", "dir", "aspect"]:
        ns = np.zeros(arr.shape)
        ns[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / (2 * sourceInfo.dy * yFactor)
        if mode in ["north-south", "ns"]:
            output = ns

    if mode in ["east-west", "total", "slope", "dir", "aspect"]:
        ew = np.zeros(arr.shape)
        ew[:, 1:-1] = (arr[:, :-2] - arr[:, 2:]) / (2 * sourceInfo.dx * xFactor)
        if mode in ["east-west", "ew"]:
            output = ew

    if mode == "total" or mode == "slope":
        output = np.sqrt(ns * ns + ew * ew)

    if mode == "dir" or mode == "aspect":
        output = np.arctan2(ns, ew)

    # Done!
    if asMatrix:
        return output
    else:
        return createRaster(bounds=sourceInfo.bounds, pixelWidth=sourceInfo.dx, pixelHeight=sourceInfo.dy,
                            srs=sourceInfo.srs, data=output, **kwargs)


####################################################################
# Get Raster information

def isFlipped(source):
    source = loadRaster(source)
    xOrigin, dx, trash, yOrigin, trash, dy = source.GetGeoTransform()

    if(dy < 0):
        return True
    else:
        return False


RasterInfo = namedtuple(
    "RasterInfo", "srs dtype flipY yAtTop bounds xMin yMin xMax yMax dx dy pixelWidth pixelHeight noData, xWinSize, yWinSize, meta, source, scale, offset")


def rasterInfo(sourceDS):
    """Returns a named tuple containing information relating to the input raster

    Returns:
    --------
    namedtuple -> ( srs: The spatial reference system (as an OGR object)
                    dtype: The datatype 
                    flipY: A flag which indicates that the raster starts at the 
                             'bottom' as opposed to at the 'top'
                    bounds: The (xMin, yMin, xMax, and yMax) values as a tuple
                    xMin: The minimal X boundary
                    yMin:The minimal Y boundary
                    xMax:The maximal X boundary
                    yMax: The maximal Y boundary
                    pixelWidth: The raster's pixelWidth 
                    pixelHeight: The raster's pixelHeight
                    dx:The raster's pixelWidth
                    dy: The raster's pixelHeight
                    noData: The noData value used by the raster
                    scale: The scale value used by the raster
                    offset: The offset value used by the raster
                    xWinSize: The width of the raster is pixels
                    yWinSize: The height of the raster is pixels
                    meta: The raster's meta data )
    """
    output = {}
    sourceDS = loadRaster(sourceDS)

    # get srs
    srs = SRS.loadSRS(sourceDS.GetProjectionRef())
    output['srs'] = srs

    # get extent and resolution
    sourceBand = sourceDS.GetRasterBand(1)
    output['dtype'] = sourceBand.DataType
    output['noData'] = sourceBand.GetNoDataValue()
    output['scale'] = sourceBand.GetScale()
    output['offset'] = sourceBand.GetOffset()

    xSize = sourceBand.XSize
    ySize = sourceBand.YSize

    xOrigin, dx, _, yOrigin, _, dy = sourceDS.GetGeoTransform()

    xMin = xOrigin
    xMax = xOrigin + dx * xSize

    if(dy < 0):
        yMax = yOrigin
        yMin = yMax + dy * ySize
        dy = -1 * dy
        output["flipY"] = True
        output["yAtTop"] = True
    else:
        yMin = yOrigin
        yMax = yOrigin + dy * ySize
        output["flipY"] = False
        output["yAtTop"] = False

    output['pixelWidth'] = dx
    output['pixelHeight'] = dy
    output['dx'] = dx
    output['dy'] = dy
    output['xMin'] = xMin
    output['xMax'] = xMax
    output['yMin'] = yMin
    output['yMax'] = yMax
    output['xWinSize'] = xSize
    output['yWinSize'] = ySize
    output['bounds'] = (xMin, yMin, xMax, yMax)
    output['meta'] = sourceDS.GetMetadata_Dict()
    output["source"] = sourceDS.GetDescription()

    # clean up
    del sourceBand, sourceDS

    # return
    return RasterInfo(**output)


####################################################################
# extract specific points in a raster
ptValue = namedtuple('value', "data xOffset yOffset inBounds")


def extractValues(source, points, pointSRS='latlon', winRange=0, noDataOkay=True, _onlyValues=False):
    """Extracts the value of a raster at a given point or collection of points. 
       Can also extract a window of values if desired

    * If the given raster is not in the 'flipped-y' orientation, the result will 
      be automatically flipped

    Notes:
    ------
    Generally speaking, interpolateValues() should be used instead of this function

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource

    points : (X,Y) or [(X1,Y1), (X2,Y2), ...] or Location or LocationSet()
        Coordinates for the points to extract
        * All points must be in the same SRS
        * !REMEMBER! For lat and lon coordinates, X is lon and Y is lat 
          (opposite of what you may think...)

    pointSRS : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the point to create
          * If not given, longitude/latitude is assumed
          * Only useful when 'points' is not a LocationSet

    winRange : int 
        The window range (in pixels) to extract the values centered around the 
        closest raster index to the indicated locations.
        * A winRange of 0 will only extract the closest raster value
        * A winRange of 1 will extract a window of shape (3,3) 
        * A winRange of 3 will extract a window of shape (7,7)

    noDataOkay: bool
        If True, an error is raised if a 'noData' value is extracted
        If False, numpy.nan is inserted whenever a 'noData' value is extracted

    Returns: 
    --------
    * If only a single location is given: 
        namedtuple -> (data : The extracted data at the location
                       xOffset : The X index distance from the location to the 
                                 center of the closest raster pixel
                       yOffset : The Y index distance from the location to the 
                                 center of the closest raster pixel
                       inBounds: Flag for whether or not the location is within
                                 The raster's bounds    
                        ) 
    * If Multiple locations are given:
        pandas.DataFrame 
            * Columns are (data, xOffset, yOffset, inBounds)
                - See above for column descriptions
            * Index is 0...N if 'points' input is not a LocationSet
            * Index is the LocationSet is if 'points' input is a LocationSet

    """
    # Be sure we have a raster and srs
    source = loadRaster(source)
    info = rasterInfo(source)
    pointSRS = SRS.loadSRS(pointSRS)

    # Ensure we have a list of point geometries
    try:
        if points._TYPE_KEY_ == "Location":
            asSingle = True
            pointsKey = None
            points = [points.asGeom(info.srs), ]
        elif points._TYPE_KEY_ == "LocationSet":
            asSingle = False
            pointsKey = points
            points = points.asGeom(info.srs)
    except AttributeError:
        pointsKey = None

        def loadPoint(pt, s):
            if isinstance(pt, ogr.Geometry):
                if pt.GetGeometryName() != "POINT":
                    raise GEOM.GeoKitGeomError("Invalid geometry given")
                return pt

            if isinstance(pt, Location):
                return pt.geom

            tmpPt = ogr.Geometry(ogr.wkbPoint)
            tmpPt.AddPoint(*pt)
            tmpPt.AssignSpatialReference(s)

            return tmpPt

        # check for an individual point input
        if isinstance(points, Location) or isinstance(points, tuple) or isinstance(points, ogr.Geometry):
            asSingle = True
            points = [loadPoint(points, pointSRS), ]
        else:  # assume points is iterable
            asSingle = False
            points = [loadPoint(pt, pointSRS) for pt in points]

        # Cast to source srs
        # make sure we're using the pointSRS for the points in the list
        pointSRS = points[0].GetSpatialReference()
        if not pointSRS.IsSame(info.srs):
            points = GEOM.transform(points, fromSRS=pointSRS, toSRS=info.srs)

    # Get x/y values as numpy arrays
    x = np.array([pt.GetX() for pt in points])
    y = np.array([pt.GetY() for pt in points])

    # Calculate x/y indexes
    xValues = (x - (info.xMin + 0.5 * info.pixelWidth)) / info.pixelWidth
    xIndexes = np.round(xValues)
    xOffset = xValues - xIndexes

    if info.yAtTop:
        yValues = ((info.yMax - 0.5 * info.pixelWidth) - y) / abs(info.pixelHeight)
        yIndexes = np.round(yValues)
        yOffset = yValues - yIndexes
    else:
        yValues = (y - (info.yMin + 0.5 * info.pixelWidth)) / info.pixelHeight
        yIndexes = np.round(yValues)
        yOffset = -1 * (yValues - yIndexes)

    # Calculate the starts and window size
    xStarts = xIndexes - winRange
    yStarts = yIndexes - winRange
    window = 2 * winRange + 1

    inBounds = xStarts > 0
    inBounds = inBounds & (yStarts > 0)
    inBounds = inBounds & (xStarts + window < info.xWinSize)
    inBounds = inBounds & (yStarts + window < info.yWinSize)

    if (~inBounds).any():
        msg = "WARNING: One of the given points (or extraction windows) exceeds the source's limits"
        warnings.warn(msg, UserWarning)

    # Read values
    values = []
    band = source.GetRasterBand(1)

    for xi, yi, ib in zip(xStarts, yStarts, inBounds):
        if not ib:
            data = np.empty((window, window))
        else:
            # Open and read from raster
            data = band.ReadAsArray(
                xoff=xi, yoff=yi, win_xsize=window, win_ysize=window)
            if info.scale != 1.0:
                data = data * info.scale
            if info.offset != 0.0:
                data = data + info.offset

            # Look for nodata
            if not info.noData is None:
                nodata = data == info.noData
                if nodata.any():
                    if noDataOkay:
                        # data will neaed to be a float type to represent a nodata value
                        data = data.astype(np.float64)
                        data[nodata] = np.nan
                    else:
                        raise GeoKitRasterError(
                            "No data values found in extractValues with 'noDataOkay' set to False")

            # flip if not in the 'flipped-y' orientation
            if not info.yAtTop:
                data = data[::-1, :]

        if winRange == 0:
            # If winRange is 0, theres no need to return a 2D matrix
            data = data[0][0]

        # Append to values
        values.append(data)

    # Done!
    if asSingle:  # A single point was given, so return a single result
        if _onlyValues:
            return values[0]
        else:
            return ptValue(values[0], xOffset[0], yOffset[0], inBounds[0])
    else:
        if _onlyValues:
            return np.array(values)
        else:
            return pd.DataFrame(dict(data=values, xOffset=xOffset, yOffset=yOffset, inBounds=inBounds), index=pointsKey)

####################################################################
# Shortcut for getting just the raster value


def interpolateValues(source, points, pointSRS='latlon', mode='near', func=None, winRange=None, **kwargs):
    """Interpolates the value of a raster at a given point or collection of points.

    Supports various interpolation schemes: 
        'near', 'linear-spline', 'cubic-spline', 'average', or user-defined


    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource

    points : (X,Y) or [(X1,Y1), (X2,Y2), ...] or Location or LocationSet()
        Coordinates for the points to extract
        * All points must be in the same SRS
        * !REMEMBER! For lat and lon coordinates, X is lon and Y is lat 
          (opposite of what you may think...)

    pointSRS : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the point to create
          * If not given, longitude/latitude is assumed
          * Only useful when 'points' is not a LocationSet

    mode : str; optional
        The interpolation scheme to use
        * options are...
          "near" - Just gets the nearest value (this is default)
          "linear-spline" - calculates a linear spline in between points
          "cubic-spline" - calculates a cubic spline in between points
          "average" - calculates average across a window
          "func" - uses user-provided calculator

    func - function 
        A user defined interpolation function 
        * Only utilized when 'mode' equals "func"
        * The function must take three arguments in this order...
          - A 2 dimensional data matrix
          - A x-index-offset
          - A y-index-offset
        * See the example below for more information

    winRange : int 
        The window range (in pixels) to extract the values centered around the 
        closest raster index to the indicated locations.
        * A winRange of 0 will only extract the closest raster value
        * A winRange of 1 will extract a window of shape (3,3) 
        * A winRange of 3 will extract a window of shape (7,7)
        * Only utilized when 'mode' equals "func"
        * All interpolation schemes have a predefined window range which is 
          appropriate to their use
            - near -> 0
            - linear-spline -> 2
            - cubic-spline -> 4
            - average -> 3
            - func -> 3    


    Returns: 
    --------
    * If only a single location is given: numeric 
        namedtuple -> (data : The extracted data at the location
                       xOffset : The X index distance from the location to the 
                                 center of the closest raster pixel
                       yOffset : The Y index distance from the location to the 
                                 center of the closest raster pixel
                       inBounds: Flag for whether or not the location is within
                                 The raster's bounds    
                        ) 
    * If Multiple locations are given: numpy.ndrray -> (N,)
        - where N is the number of locations

    Example: 
    --------
    "Interpolate" according to the median value in a 5x5 window

    >>> def medianFinder( data, xOff, yOff ):
    >>>     return numpy.median(data)
    >>>
    >>> result = interpolateValues( <source>, <points>, mode='func', 
    >>>                             func=medianFinder, winRange=2)

    """
    # Determine what the user probably wants as an output
    if isinstance(points, tuple) or isinstance(points, ogr.Geometry) or isinstance(points, Location):
        asSingle = True
        # make points a list of length 1 so that the rest works (will be unpacked later)
        points = [points, ]
    else:  # Assume points is already an iterable of some sort
        asSingle = False

    # Do interpolation
    if mode == 'near':
        # Simple get the nearest value
        win = 0 if winRange is None else winRange
        result = extractValues(
            source, points, pointSRS=pointSRS, winRange=win, _onlyValues=True)

    elif mode == "linear-spline":  # use a spline interpolation scheme
        # setup inputs
        win = 2 if winRange is None else winRange
        x = np.linspace(-1 * win, win, 2 * win + 1)
        y = np.linspace(-1 * win, win, 2 * win + 1)

        # get raw data
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)

        # Calculate interpolated values
        result = []
        for v in values.itertuples(index=False):
            rbs = RectBivariateSpline(y, x, v.data, kx=1, ky=1)

            result.append(rbs(v.yOffset, v.xOffset)[0][0])

    elif mode == "cubic-spline":  # use a spline interpolation scheme
        # setup inputs
        win = 4 if winRange is None else winRange
        x = np.linspace(-1 * win, win, 2 * win + 1)
        y = np.linspace(-1 * win, win, 2 * win + 1)

        # Get raw data
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)

        # Calculate interpolated values
        result = []
        for v in values.itertuples(index=False):
            rbs = RectBivariateSpline(y, x, v.data)

            result.append(rbs(v.yOffset, v.xOffset)[0][0])

    elif mode == "average":  # Get the average in a window
        win = 3 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        result = []
        for v in values.itertuples(index=False):
            result.append(v.data.mean())

    elif mode == "func":  # Use a general function processor
        if func is None:
            raise GeoKitRasterError(
                "'func' mode chosen, but no func kwargs was given")
        win = 3 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        result = []
        for v in values.itertuples(index=False):
            result.append(func(v.data, v.xOffset, v.yOffset))

    else:
        raise GeoKitRasterError("Interpolation mode not understood: ", mode)

    # Done!
    if asSingle:
        return result[0]
    else:
        return np.array(result)

####################################################################
# General raster mutator


def mutateRaster(source, processor=None, bounds=None, boundsSRS='latlon', autocorrect=False, output=None, dtype=None, **kwargs):
    """Process all pixels in a raster according to a given function. The boundaries
    of the resulting raster can be changed as long as the new boundaries are within 
    the scope of the original raster, but the resolution cannot

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource

    processor: function; optional
        The function performing the mutation of the raster's data 
        * The function will take single argument (a 2D numpy.ndarray) 
        * The function must return a numpy.ndarray of the same size as the input
        * The return type must also be containable within a Float32 (int and 
          boolean is okay)
        * See example below for more info

    bounds: tuple or Extent
        The boundary to clip the raster to before mutating
        * If given as an Extent, the extent is always cast to the source's native
          srs before mutating
        * If given as a tuple, (xMin, yMin, xMax, yMax) is expected
            - Units must be in the srs specified by 'boundsSRS'
        * This boundary must fit within the boundary of the rasters source
        * The boundary is always fitted to the source's grid, so the returned
          values do not necessarily match to the boundary which is provided        

    boundsSRS: Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the 'bounds' argument
        * This is ignored if the 'bounds' argument is an Extent object or is None

    autocorrect : bool; optional
        If True, then before mutating the matrix extracted from the source will have
        pixels equal to its 'noData' value converted to numpy.nan
        * Data type will always result in a float, so be careful with large
          matricies

    output : str; optional
        A path to an output file 
        * If output is None, the raster will be created in memory and a dataset 
          handel will be returned
        * If output is given, the raster will be written to disk and nothing will
          be returned

    dtype : Type, str, or numpy-dtype; optional
        If given, forces the processed data to be a particular datatype
        * Example
          - A python numeric type  such as bool, int, or float
          - A Numpy datatype such as numpy.uint8 or numpy.float64
          - a String such as "Byte", "UInt16", or "Double"

    **kwargs: 
        * All kwargs are passed on to a call to createRaster()

    Example:
    --------
    If you wanted to assign suitability factors based on a raster containing 
    integer identifiers

    >>> def calcSuitability( data ):
    >>>     # create an ouptut matrix
    >>>     outputMatrix = numpy.zeros( data.shape )
    >>> 
    >>>     # do the processing
    >>>     outputMatrix[ data == 1 ] = 0.1
    >>>     outputMatrix[ data == 2 ] = 0.2 
    >>>     outputMatrix[ data == 10] = 0.4
    >>>     outputMatrix[ np.logical_and(data > 15, data < 20)  ] = 0.5
    >>> 
    >>>     # return the output matrix
    >>>     return outputMatrix
    >>> 
    >>> result = processRaster( <source-path>, processor=calcSuitability )
    """
    # open the dataset and get SRS
    workingDS = loadRaster(source)

    # Get ds info
    dsInfo = rasterInfo(workingDS)

    # Read data into array
    sourceData, bounds = extractMatrix(
        source, bounds=bounds, boundsSRS=boundsSRS, autocorrect=autocorrect, returnBounds=True)
    workingExtent = dsInfo.bounds if (bounds is None) else bounds

    # Perform processing
    processedData = processor(sourceData) if processor else sourceData
    if(dtype and processedData.dtype != dtype):
        processedData = processedData.astype(dtype)

    dtype = gdalType(processedData.dtype)

    # Ensure returned matrix is okay
    if(processedData.shape != sourceData.shape):
        raise GeoKitRasterError("Processed matrix does not have the correct shape \nIs {0} \nShoud be {1}", format(
            processedData.shape, sourceData.shape))
    del sourceData

    # Create an output raster
    if(output is None):
        dtype = gdalType(
            processedData.dtype) if dtype is None else gdalType(dtype)
        return UTIL.quickRaster(dy=dsInfo.dy, dx=dsInfo.dx, bounds=workingExtent, dtype=dtype,
                                srs=dsInfo.srs, data=processedData, **kwargs)

    else:
        createRaster(pixelHeight=dsInfo.dy, pixelWidth=dsInfo.dx, bounds=workingExtent,
                     srs=dsInfo.srs, data=processedData, output=output, **kwargs)

        return output


def indexToCoord(yi, xi, source=None, asPoint=False, bounds=None, dx=None, dy=None, yAtTop=True, srs=None):
    """Convert the index of a raster to coordinate values.

    Parameters:
    -----------
    xi : int
        The x index
        * a numpy array of ints is also acceptable

    yi : int
        The y index
        * a numpy array of ints is also acceptable

    source : Anything acceptable by loadRaster()
        The contentual raster datasource 

    asPoint : bool
        Instruct program to return point geometries instead of x,y coordinates

    Returns:
    --------
    * If 'asPoint' is True: ogr.Geometry
    * If 'asPoint' is False: tuple -> (x,y) coordinates

    """
    if not source is None:
        # Get source info
        if not isinstance(source, RasterInfo):
            source = rasterInfo(source)

        xMin = source.xMin
        # xMax = source.xMax
        yMin = source.yMin
        yMax = source.yMax
        dx = source.dx
        dy = source.dy
        yAtTop = source.yAtTop
    else:
        try:
            xMin, yMin, xMax, yMax = bounds
        except:
            xMin, yMin, xMax, yMax = bounds.xyXY

    # Caclulate coordinates
    if yAtTop:
        x = xMin + dx * (xi + 0.5)
        y = yMax - dy * (yi + 0.5)
    else:
        x = xMin + dx * (xi + 0.5)
        y = yMin + dy * (yi + 0.5)

    # make the output
    if asPoint:
        try:  # maybe x and y are iterable
            output = [GEOM.point((xx, yy), srs=srs) for xx, yy in zip(x, y)]
        except TypeError:  # x and y should be a single point
            output = GEOM.point((x, y), srs=srs)
    else:
        output = np.column_stack([x, y])

    # Done!
    return output

# Raster plotter


def drawRaster(source, srs=None, ax=None, resolution=None, cutline=None, figsize=(12, 12), xlim=None, ylim=None, fontsize=16, hideAxis=False, cbar=True, cbarPadding=0.01, cbarTitle=None, vmin=None, vmax=None, cmap="viridis", cbax=None, cbargs=None, cutlineFillValue=-9999, leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0, zorder=0, **kwargs):
    """Draw a raster as an image on a matplotlib canvas

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource to draw

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the drawn raster data
          * If not given, the raster's internal srs is assumed
          * If the drawing resolution does not match the source's inherent 
            resolution, the source will be warped to the correct format

    ax : matplotlib axis; optional
        The axis to draw the geometries on
          * If not given, a new axis is generated and returned

    resolution : numeric or tuple; optional
        The resolution of the plotted raster data 
        * Lower resolution means more pixels to draw and can be a burden on 
          memory
        * If a tuple is given, resolutions in the X and Y direction are expected
        * Changing the resolution fron the inherent resolution requires a warp

    cutline : str or ogr.Geometry; optional
        The cutline to limit the drawn data too
        * If a string is given, it must be a path to a vector file
        * Values outside of the cutline are given the value 'cutlineFillValue'
        * Requires a warp

    cutlineFillValue : numeric; optional
        The value to give to values outside a cutline
        * Has no effect when cutline is not given

    figsize : (int, int); optional
        The figure size to create when generating a new axis
          * If resultign figure looks wierd, altering the figure size is your best
            bet to make it look nicer

    xlim : (float, float); optional
        The x-axis limits

    ylim : (float, float); optional
        The y-axis limits

    fontsize : int; optional
        A base font size to apply to tick marks which appear
          * Titles and labels are given a size of 'fontsize' + 2

    cbarPadding : float; optional
        The spacing padding to add between the generated axis and the generated
        colorbar axis
          * Only useful when generating a new axis
          * Only useful when 'colorBy' is given

    cbarTitle : str; optional
        The title to give to the generated colorbar
          * If not given, but 'colorBy' is given, the same string for 'colorBy'
            is used
            * Only useful when 'colorBy' is given

    vmin : float; optional
        The minimum value to color
          * Only useful when 'colorBy' is given

    vmax : float; optional
        The maximum value to color
          * Only useful when 'colorBy' is given

    cmap : str or matplotlib ColorMap; optional
        The colormap to use when coloring
          * Only useful when 'colorBy' is given

    cbax : matplotlib axis; optional
        An explicitly given axis to use for drawing the colorbar
          * If not given, but 'colorBy' is given, an axis for the colorbar is 
            automatically generated

    cbargs : dict; optional

    leftMargin : float; optional
        Additional margin to add to the left of the figure
          * Before using this, try adjusting the 'figsize'

    rightMargin : float; optional
        Additional margin to add to the left of the figure
          * Before using this, try adjusting the 'figsize'

    topMargin : float; optional
        Additional margin to add to the left of the figure
          * Before using this, try adjusting the 'figsize'

    bottomMargin : float; optional
        Additional margin to add to the left of the figure
          * Before using this, try adjusting the 'figsize'

    **kwargs : Passed on to a call to warp()
        * Determines how the warping is carried out
        * Consider using 'resampleAlg' or 'workingType' for finer control


    Returns:
    --------
    A namedtuple containing:
       'ax' -> The map axis
       'handles' -> All geometry handles which were created in the order they were 
                    drawn
       'cbar' -> The colorbar handle if it was drawn

    """
    # Create an axis, if needed
    if isinstance(ax, UTIL.AxHands):
        ax = ax.ax
    import matplotlib.pyplot as plt

    if ax is None:
        newAxis = True
        plt.figure(figsize=figsize)

        if not cbar:  # We don't need a colorbar
            if not hideAxis:
                leftMargin += 0.07

            ax = plt.axes([leftMargin,
                           bottomMargin,
                           1 - (rightMargin + leftMargin),
                           1 - (topMargin + bottomMargin)])
            cbax = None

        else:  # We need a colorbar
            rightMargin += 0.08  # Add area on the right for colorbar text
            if not hideAxis:
                leftMargin += 0.07

            cbarExtraPad = 0.05
            cbarWidth = 0.04

            ax = plt.axes([leftMargin,
                           bottomMargin,
                           1 - (rightMargin + leftMargin + cbarWidth + cbarPadding),
                           1 - (topMargin + bottomMargin)])

            cbax = plt.axes([1 - (rightMargin + cbarWidth),
                             bottomMargin + cbarExtraPad,
                             cbarWidth,
                             1 - (topMargin + bottomMargin + 2 * cbarExtraPad)])

        if hideAxis:
            ax.axis("off")
        else:
            ax.tick_params(labelsize=fontsize)
    else:
        newAxis = False

    # Load the raster datasource and check for transformation
    source = loadRaster(source)
    info = rasterInfo(source)

    if not (srs is None and resolution is None and cutline is None and xlim is None and ylim is None):
        if not (xlim is None and ylim is None):
            bounds = (xlim[0], ylim[0], xlim[1], ylim[1],)
        else:
            bounds = None

        if resolution is None:
            xres, yres = None, None
        else:
            try:
                xres, yres = resolution
            except:
                xres, yres = resolution, resolution

        source = warp(source, cutline=cutline, pixelHeight=yres, pixelWidth=xres, srs=srs,
                      bounds=bounds, fill=cutlineFillValue, noData=cutlineFillValue, **kwargs)

    info = rasterInfo(source)

    # Read the Data
    data = extractMatrix(source).astype(float)
    if not cutlineFillValue is None:
        data[data == info.noData] = np.nan

    # Draw image
    ext = (info.xMin, info.xMax, info.yMin, info.yMax,)
    h = ax.imshow(data, extent=ext, vmin=vmin,
                  vmax=vmax, cmap=cmap, zorder=zorder)

    # Draw Colorbar
    if cbar:
        tmp = dict(cmap=cmap, orientation='vertical')
        if not cbargs is None:
            tmp.update(cbargs)

        if cbax is None:
            cbar = plt.colorbar(h, ax=ax, **tmp)
        else:
            cbar = plt.colorbar(h, cax=cbax, **tmp)

        cbar.ax.tick_params(labelsize=fontsize)
        if not cbarTitle is None:
            cbar.set_label(cbarTitle, fontsize=fontsize + 2)

    # Do some formatting
    if newAxis:
        ax.set_aspect('equal')
        ax.autoscale(enable=True)

    if not xlim is None:
        ax.set_xlim(*xlim)
    if not ylim is None:
        ax.set_ylim(*ylim)

    # Done!
    return UTIL.AxHands(ax, h, cbar)

# 3
# Make a geometry from a matrix mask
#
# NOTE TO ME
# This function is VERRYYYYYYY similar to polygonizeMatrix, but since it does not
# not actually have to deal with the matrix data (which is all done by GDAL), this should be
# kept along side polygonizeMatrix


def polygonizeRaster(source, srs=None, flat=False, shrink=True):
    """Polygonize a raster or an integer-valued data matrix

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource to polygonize
        * The Datatype MUST be of boolean of integer type

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the polygons to create
          * If not given, the raster's internal srs is assumed

    flat : bool
        If True, flattens the resulting geometries which share a contiguous 
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

    # Load the information we will need
    source = loadRaster(source)
    band = source.GetRasterBand(1)
    maskBand = band.GetMaskBand()
    if srs is None:
        srs = SRS.loadSRS(source.GetProjectionRef())

    # Do polygonize
    vecDS = gdal.GetDriverByName("Memory").Create(
        '', 0, 0, 0, gdal.GDT_Unknown)
    vecLyr = vecDS.CreateLayer("mem", srs=srs)

    #vecDS = gdal.GetDriverByName("ESRI Shapefile").Create("deleteme.tif", 0, 0, 0, gdal.GDT_Unknown )
    #vecLyr = vecDS.CreateLayer("layer",srs=srs)

    vecField = ogr.FieldDefn("DN", ogr.OFTInteger)
    vecLyr.CreateField(vecField)

    # Polygonize geometry
    result = gdal.Polygonize(band, maskBand, vecLyr, 0)
    if(result != 0):
        raise GEOM.GeoKitGeomError("Failed to polygonize geometry")

    # Check the geoms
    ftrN = vecLyr.GetFeatureCount()
    if(ftrN == 0):
        #raise GlaesError("No features in created in temporary layer")
        msg = "No features in created in temporary layer"
        warnings.warn(msg, UserWarning)
        return

    # Extract geometries and values
    geoms = []
    rid = []
    for i in range(ftrN):
        ftr = vecLyr.GetFeature(i)
        geoms.append(ftr.GetGeometryRef().Clone())
        rid.append(ftr.items()["DN"])

    # Do shrink, maybe
    if shrink:
        # Compute shrink factor
        shrinkFactor = -0.00001
        geoms = [g.Buffer(shrinkFactor) for g in geoms]

    # Do flatten, maybe
    if flat:
        geoms = np.array(geoms)
        rid = np.array(rid)

        finalGeoms = []
        finalRID = []
        for _rid in set(rid):
            smallGeomSet = geoms[rid == _rid]
            finalGeoms.append(GEOM.flatten(smallGeomSet) if len(
                smallGeomSet) > 1 else smallGeomSet[0])
            finalRID.append(_rid)
    else:
        finalGeoms = geoms
        finalRID = rid

    # Cleanup
    vecLyr = None
    vecDS = None
    maskBand = None

    # Done!
    return pd.DataFrame(dict(geom=finalGeoms, value=finalRID))


def contours(source, contourEdges, polygonize=True, unpack=True, **kwargs):
    """Create contour geometries at specified edges for the given raster data

    Notes:
    ====== 
    This function is similar to geokit.geom.polygonizeMatrix, although it only 
    operates on the user-specified edges AND applies the 'Marching Squares'
    algorithm

    See the gdal function "GDALContourGenerateEx" for mor information on the
    specifics of this algorithm

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource to operate on

    contourEdges : [float,]
        The edges to search for withing the raster dataset
          * This parameter can be set as "None", in which case an additional 
            argument should be given to specify how the edges should be determined
            - See the documentation of "GDALContourGenerateEx"
            - Ex. "LEVEL_INTERVAL=10", contourEdges=None

    polygons : bool
        If true, contours are returned as polygons instead of linstrings

    unpack : bool
        If True, Multipolygon/MultiLinestring objects are decomposed

    **kwargs:
        * All keyword arguments are passed on to a call to gdal.ContourGenerateEx
        * They are used to construct the 'options' parameter 
        * Example keys include: LEVEL_INTERVAL, LEVEL_BASE, LEVEL_EXP_BASE, NODATA
        * Do not use the key "ID_FIELD", since this is employed already

    Returns:
    --------
    pandas.DataFrame

    * The column 'geom' corresponds to generated geometry objects
    * The columns 'ID' corresponds to the associated contour edge for each object
    """

    # Open raster
    raster = loadRaster(source)
    band = raster.GetRasterBand(1)
    rasterSRS = SRS.loadSRS(raster.GetProjectionRef())

    # Make temporary vector
    driver = gdal.GetDriverByName("Memory")
    source = driver.Create("", 0, 0, 0, gdal.GDT_Unknown)

    layer = source.CreateLayer("", rasterSRS,
                               ogr.wkbPolygon if polygonize else ogr.wkbLineString)
    field = ogr.FieldDefn("DN", ogr.OFTInteger)
    layer.CreateField(field)

    # Setup contour function
    args = ["{}={}".format(a, b) for a, b in kwargs.items()]
    args.append("ID_FIELD=DN")
    if polygonize:
        args.append("POLYGONIZE=YES")
    if contourEdges is not None:
        opt = "FIXED_LEVELS="
        for edge in contourEdges:
            opt += str(edge) + ","
        args.append(opt[:-1])

    result = gdal.ContourGenerateEx(band, layer, options=args)
    if not result == gdal.CE_None:
        raise GeoKitRasterError("Failed to compute raster contours")
    layer.CommitTransaction()

    IDs = []
    geoms = []
    for ftrid in range(layer.GetFeatureCount()):
        ftr = layer.GetFeature(ftrid)
        geom = ftr.GetGeometryRef()
        value = ftr.GetField(0)

        if unpack:
            for gi in range(geom.GetGeometryCount()):
                geoms.append(geom.GetGeometryRef(gi).Clone())
                IDs.append(value)
        else:
            geoms.append(geom.Clone())
            IDs.append(value)

    # return geoms
    return pd.DataFrame(dict(geom=geoms, ID=IDs))


def warp(source, resampleAlg='bilinear', cutline=None, output=None, pixelHeight=None, pixelWidth=None, srs=None, bounds=None, dtype=None, noData=None, fill=None, overwrite=True, meta=None, **kwargs):
    """Warps a given raster source to another context

    * Can be used to 'warp' a raster in memory to a raster on disk

    Note:
    -----
    Unless manually altered as keyword arguments, the gdal.Warp options 
    'targetAlignedPixels' and 'copyMetadata' are both set to True

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource to draw

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the resulting raster
          * If not given, the raster's internal srs is assumed

    resampleAlg : str; optional
        The resampling algorithm to use when translating pixel values
        * Knowing which option to use can have significant impacts!
        * Options are: 'near', 'bilinear', 'cubic', 'average'

    cutline : str or ogr.Geometry; optional
        The cutline to limit the drawn data too
        * If a string is given, it must be a path to a vector file
        * Values outside of the cutline are given the value 'cutlineFillValue'
        * Requires a warp

    output : str; optional
        The path on disk where the new raster should be created

    pixelHeight : numeric; optional
        The pixel height (y-resolution) of the output raster
        * Only required if this value should be changed

    pixelWidth : numeric; optional
        The pixel width (x-resolution) of the output raster
        * Only required if this value should be changed

    bounds : tuple; optional
        The (xMin, yMin, xMax, yMax) limits of the output raster
        * Only required if this value should be changed

    dtype : Type, str, or numpy-dtype; optional
        If given, forces the processed data to be a particular datatype
        * Only required if this value should be changed
        * Example
          - A python numeric type  such as bool, int, or float
          - A Numpy datatype such as numpy.uint8 or numpy.float64
          - a String such as "Byte", "UInt16", or "Double"

    noData : numeric; optional
        The no-data value to apply to the output raster

    fill : numeric; optional
        The fill data to place into the new raster before warping occurs
        * Does not play a role when writing a file to disk

    **kwargs:
        * All keyword arguments are passed on to a call to gdal.WarpOptions
        * Use these to fine-tune the warping procedure
        * Key Options are (from gdal.WarpOptions):
            format --- output format ("GTiff", etc...)
            targetAlignedPixels --- whether to force output bounds to be multiple
                                    of output resolution
            workingType --- working type (gdal.GDT_Byte, etc...)
            warpMemoryLimit --- size of working buffer in bytes
            creationOptions --- list of creation options
            srcNodata --- source nodata value(s)
            dstNodata --- output nodata value(s)
            multithread --- whether to multithread computation and I/O operations
            cutlineWhere --- cutline WHERE clause
            cropToCutline --- whether to use cutline extent for output bounds
            setColorInterpretation --- whether to force color interpretation of 
                                       input bands to output bands

    Returns:
    --------
    * If 'output' is None: gdal.Dataset
    * If 'output' is a string: The path to the output is returned (for easy opening)

    """
    # open source and get info
    source = loadRaster(source)
    dsInfo = rasterInfo(source)
    if dsInfo.scale != 1.0 or dsInfo.offset != 0.0:
        isAdjusted = True
    else:
        isAdjusted = False

    # Handle potentially missing arguments
    if not srs is None:
        srs = SRS.loadSRS(srs)
    if srs is None:
        srs = dsInfo.srs
        srsOkay = True
    else:
        if srs.IsSame(dsInfo.srs):
            srsOkay = True
        else:
            srsOkay = False

    if bounds is None:
        if srsOkay:
            bounds = dsInfo.bounds
        else:
            bounds = GEOM.boundsToBounds(dsInfo.bounds, dsInfo.srs, srs)

    if pixelHeight is None:
        if srsOkay:
            pixelHeight = dsInfo.dy
        else:
            pixelHeight = (bounds[3] - bounds[1]) / (dsInfo.yWinSize * 1.1)

    if pixelWidth is None:
        if srsOkay:
            pixelWidth = dsInfo.dx
        else:
            pixelWidth = (bounds[2] - bounds[0]) / (dsInfo.xWinSize * 1.1)
    bounds = UTIL.fitBoundsTo(bounds, pixelWidth, pixelHeight)

    if dtype is None:
        dtype = dsInfo.dtype
    dtype = gdalType(dtype)

    if noData is None:
        noData = dsInfo.noData

    # If a cutline is given, create the output
    if not cutline is None:
        if isinstance(cutline, ogr.Geometry):
            tempdir = TemporaryDirectory()
            cutline = UTIL.quickVector(
                cutline, output=os.path.join(tempdir.name, "tmp.shp"))
        # cutline is already a path to a vector
        elif UTIL.isVector(cutline):
            tempdir = None
        else:
            raise GeoKitRasterError(
                "cutline must be a Geometry or a path to a shape file")

    # Workflow depends on whether or not we have an output
    if not output is None:  # Simply do a translate
        if(os.path.isfile(output)):
            if(overwrite == True):
                os.remove(output)
                if(os.path.isfile(output + ".aux.xml")):  # Because QGIS....
                    os.remove(output + ".aux.xml")
            else:
                raise GeoKitRasterError(
                    "Output file already exists: %s" % output)

        # # Check some for bad input configurations
        # if not srs is None:
        #     if (pixelHeight is None or pixelWidth is None):
        #         raise GeoKitRasterError("When warping between srs's and writing to a file, pixelWidth and pixelHeight must be given")

        # Arange inputs
        co = kwargs.pop("creationOptions", COMPRESSION_OPTION)
        copyMeta = kwargs.pop("copyMetadata", True)
        aligned = kwargs.pop("targetAlignedPixels", True)

        # Fix the bounds issue by making them  just a little bit smaller, which should be fixed by gdalwarp
        bounds = (bounds[0] + 0.001 * pixelWidth,
                  bounds[1] + 0.001 * pixelHeight,
                  bounds[2] - 0.001 * pixelWidth,
                  bounds[3] - 0.001 * pixelHeight, )

        # Let gdalwarp do everything...
        opts = gdal.WarpOptions(outputType=getattr(gdal, dtype), xRes=pixelWidth, yRes=pixelHeight, creationOptions=co,
                                outputBounds=bounds, dstSRS=srs, dstNodata=noData, resampleAlg=resampleAlg,
                                copyMetadata=copyMeta, targetAlignedPixels=aligned, cutlineDSName=cutline, **kwargs)

        result = gdal.Warp(output, source, options=opts)
        if not UTIL.isRaster(result):
            raise GeoKitRasterError("Failed to translate raster")

        destRas = output
    else:
        if "cropToCutline" in kwargs:
            msg = "The 'cropToCutline' option is not taken into account when writing to a raster in memory. Try using geokit.Extent.warp instead"
            warnings.warn(msg, UserWarning)

        # Warp to a raster in memory
        destRas = UTIL.quickRaster(bounds=bounds, srs=srs, dx=pixelWidth,
                                   dy=pixelHeight, dtype=dtype, noData=noData, fill=fill)

        # Do a warp
        result = gdal.Warp(
            destRas, source, resampleAlg=resampleAlg, cutlineDSName=cutline, **kwargs)
        destRas.FlushCache()

    # Do we have meta data?
    if not meta is None:
        if isinstance(result, str):
            ds = loadRaster(result, 1)
        else:
            ds = result

        for k, v in meta.items():
            ds.SetMetadataItem(k, v)

        del ds

    # Do we need to readjust?
#    if isAdjusted:
#        if isinstance(result, str):
#            ds = loadRaster(result, 1)
#        else:
#            ds = result
#        band = ds.GetRasterBand(1)
#        band.SetScale(dsInfo.scale)
#        band.SetOffset(dsInfo.offset)
#        band.FlushCache()
#        ds.FlushCache()
#        del band, ds

    # TODO: Should 'result' be deleted at this point?

    # Done!
    if not cutline is None:
        del tempdir
    return destRas
