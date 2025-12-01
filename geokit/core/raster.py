import inspect
import os
import pathlib
import sys
import warnings
from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from tempfile import TemporaryDirectory
from typing import Literal, NamedTuple, Callable
import warnings

import matplotlib.axes._axes
import matplotlib.colorbar
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
from osgeo.gdal import Driver
from scipy.interpolate import RectBivariateSpline

from geokit.core import geom as GEOM
from geokit.core import srs as SRS
from geokit.core import util as UTIL

from geokit.core.location import Location, LocationSet
from geokit.data_types import (
    load_raster_input,
    srs_input,
    numeric,
    gdal_raster_data_types,
    RasterInfo,
    ptValue,
    AxHands,
)


class GeoKitRasterError(UTIL.GeoKitError):
    pass


if "win" in sys.platform:
    COMPRESSION_OPTION = ["COMPRESS=LZW"]
    COMPRESSION_OPTION_STR = "LZW"
else:
    COMPRESSION_OPTION = ["COMPRESS=DEFLATE"]
    COMPRESSION_OPTION_STR = "DEFLATE"

# Basic Loader


def loadRaster(source: load_raster_input, mode=0) -> gdal.Dataset:
    """
    Load a raster dataset from a path to a file on disc.

    Parameters
    ----------
    source : str or gdal.Dataset
        * If a string is given, it is assumed as a path to a raster file on disc
        * If a gdal.Dataset is given, it is assumed to already be an open raster
          and is returned immediately

    Returns
    -------
    gdal.Dataset
    """
    if isinstance(source, pathlib.Path):
        source = str(source)
    if isinstance(source, str):
        ds = gdal.Open(source, mode)
    else:
        ds = source

    if ds is None:
        raise GeoKitRasterError("Could not load input dataSource: ", str(source))
    return ds


# GDAL type mapper
_gdalIntToType = dict((v, k) for k, v in filter(lambda x: "GDT_" in x[0], gdal.__dict__.items()))
_gdalType = {
    bool: "GDT_Byte",
    int: "GDT_Int32",
    float: "GDT_Float64",
    "bool": "GDT_Byte",
    "int8": "GDT_Byte",
    "int16": "GDT_Int16",
    "int32": "GDT_Int32",
    "int64": "GDT_Int32",
    "uint8": "GDT_Byte",
    "uint16": "GDT_UInt16",
    "uint32": "GDT_UInt32",
    "float32": "GDT_Float32",
    "float64": "GDT_Float64",
}


def gdalType(s):
    """Tries to determine gdal datatype from the given input type."""
    if s is None:
        return "GDT_Unknown"
    elif isinstance(s, str):
        if hasattr(gdal, s):
            return s
        elif s.lower() in _gdalType:
            return _gdalType[s.lower()]
        elif hasattr(gdal, "GDT_%s" % s):
            return "GDT_%s" % s
        elif s == "float" or s == "int" or s == "bool":
            return gdalType(np.dtype(s))

    elif isinstance(s, int):
        return _gdalIntToType[s]  # If an int is given, it's probably
        #  the GDAL type indicator (and not a sample data value)
    elif isinstance(s, np.dtype):
        return gdalType(str(s))
    elif isinstance(s, np.generic):
        return gdalType(s.dtype)
    elif s is bool:
        return _gdalType[bool]
    elif s is int:
        return _gdalType[int]
    elif s is float:
        return _gdalType[float]
    elif isinstance(s, type):  # Default to Numpy for all other 'types'
        return gdalType(np.dtype(s))
    elif isinstance(s, Iterable):
        return gdalType(s[0])
    raise GeoKitRasterError("GDAL type could not be determined")


####################################################################
# Raster writer


def createRaster(
    bounds: tuple[numeric, numeric, numeric, numeric],
    output: None | str | pathlib.Path = None,
    pixelWidth: numeric = 100,
    pixelHeight: numeric = 100,
    dtype: None | gdal_raster_data_types = None,
    srs: srs_input | None = None,
    compress: bool = True,
    noData: numeric | None = None,
    overwrite: bool = True,
    fill: numeric | None = None,
    data=None,
    meta: dict | None = None,
    scale: numeric | None = 1,
    offset: numeric = 0,
    creationOptions: dict = dict(),
    raster_band_index: int = 1,
) -> gdal.Dataset | str:
    """Create a raster file.

    NOTE:
    -----
    Raster datasets are always written in the 'yAtTop' orientation. Meaning that
    the first row of data values (either written to or read from the dataset) will
    refer to the TOP of the defined boundary, and will then move downward from
    there

    If a data matrix is given, and a negative pixelWidth is defined, the data
    will be flipped automatically

    Parameters
    ----------
    bounds : (xMin, yMin, xMax, yMax) or Extent
        The geographic extents spanned by the raster

    pixelWidth : numeric
        The pixel width of the raster in units of the input srs
        * The keyword 'dx' can be used as well and will override anything given
        assigned to 'pixelWidth'

    pixelHeight : numeric
        The pixel height of the raster in units of the input srs
        * The keyword 'dy' can be used as well and will override anything given
          assigned to 'pixelHeight'

    output : str, pathlib.Path, None
        A path to an output file
        * If output is None, the raster will be created in memory and a dataset
          handle will be returned
        * If output is given, the raster will be written to disk and nothing will
          be returned

    dtype : gdal_raster_data_types, int, none
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
        * Must be the same datatype as the 'dtype' input (or that which is derived)

    fill : numeric; optional
        The initial value given to all pixels in the created raster band
        - numeric
        * Must be the same datatype as the 'dtype' input (or that which is derived)

    overwrite : bool
        A flag to overwrite a pre-existing output file
        * If set to False and an 'output' is specified which already exists,
          an error will be raised

    data : matrix_like
        A 2D matrix to write into the resulting raster
        * array dimensions must fit raster dimensions as calculated by the bounds
          and the pixel resolution
    meta : dictionary, None
        of key values pairs that is stored in the raster as meta data. Is written
        to the raster using the


    scale : numeric; optional
        The scaling value given to apply to all values
        - numeric
        * Must be the same datatype as the 'dtype' input (or that which is derived)

    offset : numeric; optional
        The offset value given to apply to all values
        - numeric
        * Must be the same datatype as the 'dtype' input (or that which is derived)

    raster_band_index: int, defaults to 1
        Determines which band is written to in the output raster dataset.

    Returns
    -------
    * If 'output' is None: gdal.Dataset
    * If 'output' is a string: The path to the output is returned (for easy opening).
                                It has to be saved as geotiff with .tif suffix.
    """
    # Check for existing file

    if output is not None:
        output = str(output)
        output_path = pathlib.Path(output)
        if output_path.exists():
            if overwrite is True:
                output_path.unlink()
                output_path_with_aux_extension = output_path.with_suffix(".aux.xml")
                if output_path_with_aux_extension.exists():
                    output_path_with_aux_extension.unlink()
            else:
                raise GeoKitRasterError("Output file already exists: %s" % output)

        # check if the directory exists

        elif not output_path.parent.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {os.path.dirname(output)}")

        # check if writeable:
        if not os.access(os.path.dirname(output), os.W_OK):
            print(os.path.dirname(output))
            raise PermissionError(f"Writing permission error for path: {os.path.dirname(output)}")

    # Ensure bounds is okay
    # bounds = UTIL.fitBoundsTo(bounds, pixelWidth, pixelHeight)

    # Make a raster dataset and pull the band/maskBand objects
    x_min = bounds[0]
    y_min = bounds[1]
    x_max = bounds[2]
    y_max = bounds[3]  # Always use the "Y-at-Top" orientation

    cols = int(round((x_max - x_min) / pixelWidth))
    rows = int(round((y_max - y_min) / abs(pixelHeight)))

    # Get DataType
    if dtype is not None:  # a dtype was given, use it!
        dtype = gdalType(dtype)
    elif data is not None:  # a data matrix was give, use it's dtype! (assume a numpy array or derivative)
        dtype = gdalType(data.dtype)
    else:  # Otherwise, just assume we want a Byte
        dtype = "GDT_Byte"

    # Open the driver
    opts = OrderedDict()
    if compress and output is not None:
        opts["COMPRESS"] = COMPRESSION_OPTION_STR
    if creationOptions is not None:
        opts.update(creationOptions)
    opts = ["{}={}".format(k, v) for k, v in opts.items()]

    if output is None:
        driver: gdal.Driver = gdal.GetDriverByName("Mem")  # create a raster in memory
        raster: gdal.Dataset = driver.Create("", cols, rows, 1, getattr(gdal, dtype), opts)
    else:
        driver: gdal.Driver = gdal.GetDriverByName("GTiff")  # Create a raster in storage
        raster: gdal.Dataset = driver.Create(output, cols, rows, 1, getattr(gdal, dtype), opts)

    if raster is None:
        raise GeoKitRasterError("Failed to create raster")

    # Do the rest in a "try" statement so that a failure won't bind the source
    try:
        raster.SetGeoTransform((x_min, abs(pixelWidth), 0, y_max, 0, -1 * abs(pixelHeight)))

        # Set the SRS
        if srs is not None:
            rasterSRS = SRS.loadSRS(srs)
            raster.SetProjection(rasterSRS.ExportToWkt())

        # Fill the raster will zeros, null values, or initial values (if given)
        band: gdal.Band = raster.GetRasterBand(raster_band_index)
        if scale is not None:
            band.SetScale(scale)
        if offset is not None:
            band.SetOffset(offset)

        if noData is not None:
            band.SetNoDataValue(noData)
            if fill is None and data is None:
                band.Fill(noData)

        if data is None:
            if fill is None:
                band.Fill(0)
            else:
                band.Fill(fill)
        else:
            # make sure dimension size is good
            if not (data.shape[0] == rows and data.shape[1] == cols):
                raise GeoKitRasterError(
                    "Raster dimensions and input data dimensions do not match.The data has rows="
                    + str(data.shape[0])
                    + " and collumns="
                    + str(data.shape[1])
                    + ". The raster specification expects rows="
                    + str(rows)
                    + " and columns="
                    + str(cols)
                )

            # See if data needs flipping
            if pixelHeight < 0:
                data = data[::-1, :]

            # Write it!
            band.WriteArray(data)
            band.FlushCache()

            band.ComputeRasterMinMax(0)
            band.ComputeBandStats(0)

        # Write MetaData, maybe
        if meta is not None:
            for k, v in meta.items():
                raster.SetMetadataItem(k, v)

        # writes the raster to the hard drive
        raster.FlushCache()
        # Return raster if in memory
        if output is None:
            return raster

        # Done
        return output

    # Handle the fail case
    except Exception as e:
        raster = None
        raise e


def createRasterLike(source, copyMetadata=True, metadata=None, **kwargs):
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

    if copyMetadata and metadata is not None:
        raise GeoKitRasterError("If metadata is given, copyMetadata cannot be True!")

    bounds = kwargs.pop("bounds", source.bounds)
    pixelWidth = kwargs.pop("pixelWidth", source.pixelWidth)
    pixelHeight = kwargs.pop("pixelHeight", source.pixelHeight)
    dtype = kwargs.pop("dtype", source.dtype)
    srs = kwargs.pop("srs", source.srs)
    noData = kwargs.pop("noData", source.noData)

    if copyMetadata:
        meta = kwargs.pop("meta", source.meta)
    else:
        meta = metadata

    return createRaster(
        bounds=bounds,
        pixelWidth=pixelWidth,
        pixelHeight=pixelHeight,
        dtype=dtype,
        srs=srs,
        noData=noData,
        meta=meta,
        **kwargs,
    )


def saveRasterAsTif(source: gdal.Dataset, output: str, **kwargs):
    """Write a osgeo.gdal.Dataset in memory to a GeoTiff file to disk.

    Parameters
    ----------
    source : osgeo.gdal.Dataset

    output : str
        A path to an output file

    Returns
    -------
    str
        Path to the saved file on disk.
    """
    # assert os.path.isdir(os.path.dirname(output)), 'Output folder does not exist!'
    assert output.split(".")[-1] in [
        "tif",
        "tiff",
    ], "Wrong type specified, use *.tif or *.tiff"

    sourceInfo = rasterInfo(source)
    data = extractMatrix(source)

    return createRaster(
        bounds=sourceInfo.bounds,
        pixelWidth=sourceInfo.dx,
        pixelHeight=sourceInfo.dy,
        noData=sourceInfo.noData,
        dtype=sourceInfo.dtype,
        srs=sourceInfo.srs,
        data=data,
        output=output,
        **kwargs,
    )


####################################################################
# extract the raster as a matrix
def extractMatrix(
    source: load_raster_input,
    bounds=None,
    boundsSRS: srs_input = "latlon",
    maskBand: bool = False,
    autocorrect: bool = False,
    returnBounds: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """Extract all or part of a raster's band as a numpy matrix.

    Note:
    -----
    Unless one is trying to get the entire matrix from the raster dataset, usage
    of this function requires intimate knowledge of the raster's characteristics.
    In such a case it is probably easier to use Extent.extractMatrix

    Parameters
    ----------
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
          matrices

    returnBounds : bool; optional
        If True, return the computed bounds along with the matrix data

    Returns
    -------
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
    if bounds is not None:
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
            bounds = UTIL.fitBoundsTo(bounds, dsInfo.dx, dsInfo.dy, expand=True, startAtZero=True)

        # Find offsets
        xoff = int(np.round((bounds[0] - dsInfo.xMin) / dsInfo.dx))
        if xoff < 0:
            raise GeoKitRasterError("The given boundary exceeds the raster's xMin value")

        xwin = int(np.round((bounds[2] - dsInfo.xMin) / dsInfo.dx)) - xoff
        if xwin > dsInfo.xWinSize:
            raise GeoKitRasterError("The given boundary exceeds the raster's xMax value")

        if dsInfo.yAtTop:
            yoff = int(np.round((dsInfo.yMax - bounds[3]) / dsInfo.dy))
            if yoff < 0:
                raise GeoKitRasterError("The given boundary exceeds the raster's yMax value")

            ywin = int(np.round((dsInfo.yMax - bounds[1]) / dsInfo.dy)) - yoff

            if ywin > dsInfo.yWinSize:
                raise GeoKitRasterError("The given boundary exceeds the raster's yMin value")
        else:
            yoff = int(np.round((bounds[1] - dsInfo.yMin) / dsInfo.dy))
            if yoff < 0:
                raise GeoKitRasterError("The given boundary exceeds the raster's yMin value")

            ywin = int(np.round((bounds[3] - dsInfo.yMin) / dsInfo.dy)) - yoff
            if ywin > dsInfo.yWinSize:
                raise GeoKitRasterError("The given boundary exceeds the raster's yMax value")

    else:
        xoff = 0
        yoff = 0
        xwin = None
        ywin = None

    # get Data
    if maskBand:
        data = mb.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xwin, win_ysize=ywin)
    else:
        data = sourceBand.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xwin, win_ysize=ywin)
        if dsInfo.scale is not None and dsInfo.scale != 1.0:
            data = data * dsInfo.scale
        if dsInfo.offset is not None and dsInfo.offset != 0.0:
            data = data + dsInfo.offset

    # Correct 'nodata' values
    if autocorrect:
        noData = sourceBand.GetNoDataValue()
        data = data.astype(np.float64)
        data[data == noData] = np.nan

    # make sure we are returning data in the 'flipped-y' orientation
    if not isFlipped(source):
        data = data[::-1, :]

    # Done
    if returnBounds:
        return data, bounds
    else:
        return data


def rasterStats(source, cutline=None, ignoreValue=None, **kwargs):
    """Compute basic statistics of the values contained in a raster dataset.

    Parameters
    ----------
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

    Returns
    -------
    Results from a call to scipy.stats.describe
    """
    from scipy.stats import describe

    source = loadRaster(source)

    # Get the matrix to calculate over
    if cutline is not None:
        source = warp(source, cutline=cutline, noData=ignoreValue, **kwargs)

    rawData = extractMatrix(source)
    dataInfo = rasterInfo(source)

    # exclude nodata and ignore values
    sel = np.ones(rawData.shape, dtype="bool")

    if ignoreValue is not None:
        np.logical_and(rawData != ignoreValue, sel, sel)

    if dataInfo.noData is not None:
        np.logical_and(rawData != dataInfo.noData, sel, sel)

    # compute statistics
    data = rawData[sel].flatten()
    return describe(data)


####################################################################
# Gradient calculator
def gradient(source, mode="total", factor=1, asMatrix=False, **kwargs):
    """Calculate a raster's gradient and return as a new dataset or simply a matrix.

    Parameters
    ----------
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

    Returns
    -------
    * If 'asMatrix' is True: numpy.ndarray
    * If 'asMatrix' is False: gdal.Dataset
    """
    # Make sure source is a source
    source = loadRaster(source)

    # Check mode
    acceptable = [
        "total",
        "slope",
        "north-south",
        "east-west",
        "dir",
        "ew",
        "ns",
        "aspect",
    ]
    if mode not in acceptable:
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
        return createRaster(
            bounds=sourceInfo.bounds,
            pixelWidth=sourceInfo.dx,
            pixelHeight=sourceInfo.dy,
            srs=sourceInfo.srs,
            data=output,
            **kwargs,
        )


####################################################################
# Get Raster information


def isFlipped(source):
    source = loadRaster(source)
    xOrigin, dx, trash, yOrigin, trash, dy = source.GetGeoTransform()

    if dy < 0:
        return True
    else:
        return False


def rasterInfo(sourceDS: load_raster_input) -> RasterInfo:
    """Returns a named tuple containing information relating to the input raster.

    Returns
    -------
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
    if sourceDS.GetProjectionRef() == "":
        # return None directly if raster has no srs
        srs = None
    else:
        # generate an srs object if we have srs information
        srs = SRS.loadSRS(sourceDS.GetProjectionRef())
    output["srs"] = srs

    # get extent and resolution
    sourceBand = sourceDS.GetRasterBand(1)
    output["dtype"] = sourceBand.DataType
    output["noData"] = sourceBand.GetNoDataValue()
    output["scale"] = sourceBand.GetScale()
    output["offset"] = sourceBand.GetOffset()

    xSize = sourceBand.XSize
    ySize = sourceBand.YSize

    xOrigin, dx, _, yOrigin, _, dy = sourceDS.GetGeoTransform()

    xMin = xOrigin
    xMax = xOrigin + dx * xSize

    if dy < 0:
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

    output["pixelWidth"] = dx
    output["pixelHeight"] = dy
    output["dx"] = dx
    output["dy"] = dy
    output["xMin"] = xMin
    output["xMax"] = xMax
    output["yMin"] = yMin
    output["yMax"] = yMax
    output["xWinSize"] = xSize
    output["yWinSize"] = ySize
    output["bounds"] = (xMin, yMin, xMax, yMax)
    output["meta"] = sourceDS.GetMetadata_Dict()
    output["source"] = sourceDS.GetDescription()
    output["data_type_name_str"] = gdal.GetDataTypeName(output["dtype"])

    # clean up
    del sourceBand, sourceDS

    raster_info = RasterInfo(**output)

    return raster_info


####################################################################
# extract specific points in a raster


def _convertTupleToOGRPoint(
    point_as_tuple: tuple[float, float],
    point_srs_loaded: osr.SpatialReference,
    point_srs_is_set: bool,
    output_srs: osr.SpatialReference,
) -> ogr.Geometry:
    """This function converts a two dimensional point that is specified as a tuple into
    a ogr.Gemotry object.

    Parameters
    ----------
    point_as_tuple : tuple[float, float]
        The coordinates of the points.
    point_srs_loaded : osr.SpatialReference
        The spatial reference system that the point coordinates are
        specified in.
    point_srs_is_set : bool
        A flag that states if the spatial refrence system has been set
        by th user.
    output_srs : osr.SpatialReference
        The target spatial reference system of the ogr.Gemoetry point

    Returns
    -------
    ogr.Geometry
        The point as an ogr.Geometry in the target spatial refrence system.
    """
    if point_srs_is_set is False:
        _raise_srs_required_exception(object_name="tuple")
    ogr_geometry_point = ogr.Geometry(ogr.wkbPoint)
    ogr_geometry_point.AddPoint(*point_as_tuple)
    ogr_geometry_point.AssignSpatialReference(point_srs_loaded)
    assert isinstance(point_srs_loaded, osr.SpatialReference)
    if not point_srs_loaded.IsSame(output_srs):
        ogr_geometry_point_transformed = GEOM.transform(ogr_geometry_point, fromSRS=point_srs_loaded, toSRS=output_srs)
    else:
        ogr_geometry_point_transformed = ogr_geometry_point

    return ogr_geometry_point_transformed


def _check_if_geometry_is_point(point_to_check: ogr.Geometry):
    if point_to_check.GetGeometryName() != "POINT":
        raise GEOM.GeoKitGeomError(
            "A " + str(point_to_check.GetGeometryName()) + " geometry has been passed as a point."
            "However only 'POINT' Geometries are an appropiate input as points"
        )


def _check_if_srs_is_epsg_4326(srs: osr.SpatialReference):
    if not srs.IsSame(SRS.loadSRS(4326)):
        srs_string = srs.ExportToWkt
        warnings.warn(
            "The Location Object assumes the Spatial Reference Syste EPSG 4326 internally. "
            "Howerver the provided point system in WKT is: " + str(srs_string)
        )


def _raise_srs_required_exception(object_name: str):
    raise GEOM.GeoKitGeomError(
        "A spatial reference systems is required when passing" + str(object_name) + " but None has been provided."
    )


def _transform_ogr_point_spatial_reference_system(
    ogr_point: ogr.Geometry, point_input_SRS_external: osr.SpatialReference | None, output_srs: osr.SpatialReference
) -> ogr.Geometry:
    """This function transforms the ogr.Geometry point transfroms into the output output spatial reference system.

    Before transforming the spatial reference system of the ogr.Geometry, it performs some sanity checks.
    ogr.Geometry object. First, it verifies that the geometry is a point. Then, it verifies that the spatial
    reference system stored in the ogr.geometry object with the spatial reference system provided by the
    user to ensure consistency.

    Parameters
    ----------
    ogr_point : ogr.Geometry
        The point that should be transformed into a new spatial reference system.
    point_input_SRS_external : osr.SpatialReference | None
        The current spatial refrence system of the point. If provided it musst be consistent
        with the one stored in the ogr.Geometry object or ogr.Geometry does not store a
        srs yet.
    output_srs : osr.SpatialReference
        The target spatial refrence system that the point is transfered to

    Returns
    -------
    ogr.Geometry
        The transformed point.

    """
    _check_if_geometry_is_point(point_to_check=ogr_point)
    point_srs_loaded_geometry: osr.SpatialReference = ogr_point.GetSpatialReference()
    if point_srs_loaded_geometry is None and point_input_SRS_external is None:
        # Wrong configuration of the function
        raise GEOM.GeoKitGeomError(
            "The ogr.Geometry Point does not store a spatial reference system, "
            "nor does the user provide one. However, in order to convert the coordinates "
            "into another spatial reference system, the point needs to define a spatial "
            "reference system for its current coordinates."
        )
    elif isinstance(point_srs_loaded_geometry, osr.SpatialReference) and point_input_SRS_external is None:
        # Although the user did not provide a spatial reference system,
        # the OGR geometry stores one, which is an acceptable use of this function.
        pass
    elif isinstance(point_srs_loaded_geometry, osr.SpatialReference) and isinstance(
        point_input_SRS_external, osr.SpatialReference
    ):
        # The user provided a spatial but refrence system and the
        if not point_srs_loaded_geometry.IsSame(point_input_SRS_external):
            raise GEOM.GeoKitGeomError(
                "The user has provided another spatial reference system in the 'point_input_SRS' "
                "argument than the one stored in the ogr.Geometry object. This indicates that the "
                "user assumes that the point is in a different spatial reference system than the "
                "one in which the point is actually stored. To use the point set's internal spatial "
                "reference system, set point_input_SRS to None. The target spatial reference system "
                "is determined by the spatial reference system of the raster anyway. The spatial "
                "reference system that is stored in the ogr.Geometry in WKT is: \n "
                + str(point_srs_loaded_geometry.ExportToWkt())
                + " the user provided spatial reference system is:\n"
                + str(point_input_SRS_external.ExportToWkt())
            )
    elif point_srs_loaded_geometry is None and isinstance(point_input_SRS_external, osr.SpatialReference):
        ogr_point.AssignSpatialReference(point_input_SRS_external)

    if not point_srs_loaded_geometry.IsSame(output_srs):
        ogr_point = GEOM.transform(ogr_point, fromSRS=point_srs_loaded_geometry, toSRS=output_srs)
    return ogr_point


def _convertPointsToListOfOGRPoints(
    points: tuple[float, float]
    | list[tuple[float, float]]
    | ogr.Geometry
    | list[ogr.Geometry]
    | Location
    | LocationSet,
    point_input_SRS: srs_input | None,
    output_srs: osr.SpatialReference,
) -> list[ogr.Geometry]:
    """This function takes tuple[float, float] | ogr.Geometry | list[tuple[float, float]] | Location | LocationSet
    as input and converts them to a list ogr.Geometry objects of the type point.

    Parameters
    ----------
    points : tuple[float, float] | list[tuple[float, float]] | ogr.Geometry | list[ogr.Geometry] | Location | LocationSet
        The points that should be converted into a list of ogr.Geometry objects.
    point_input_SRS : srs_input | None
        The spatial reference system that the points are specified in when passed to the function
    output_srs : osr.SpatialReference
        The target spatial refrence system that the points should be converted to.

    Returns
    -------
    list[ogr.Geometry]
        A list of the points in the target spatial refrence systems as ogr.Geometry objects.

    """
    if point_input_SRS is None:
        point_srs_is_set = False
        point_srs_loaded = None
    else:
        point_srs_is_set = True
        point_srs_loaded = SRS.loadSRS(point_input_SRS)

    # Ensure we have a list of point geometries
    if isinstance(points, Location):
        if point_srs_is_set is True:
            assert isinstance(point_srs_loaded, osr.SpatialReference)
            _check_if_srs_is_epsg_4326(srs=point_srs_loaded)
        points_as_list_of_geom = [points.asGeom(srs=output_srs)]
    elif isinstance(points, LocationSet):
        if point_srs_is_set is True:
            assert isinstance(point_srs_loaded, osr.SpatialReference)
            _check_if_srs_is_epsg_4326(srs=point_srs_loaded)
        points_as_list_of_geom = points.asGeom(
            srs=output_srs,
        )
    elif isinstance(points, tuple):
        points_as_list_of_geom = [
            _convertTupleToOGRPoint(
                point_as_tuple=points,
                point_srs_loaded=point_srs_loaded,
                point_srs_is_set=point_srs_is_set,
                output_srs=output_srs,
            )
        ]
    elif isinstance(points, ogr.Geometry):
        points_as_list_of_geom = [
            _transform_ogr_point_spatial_reference_system(
                ogr_point=points, point_input_SRS_external=point_srs_loaded, output_srs=output_srs
            )
        ]
    elif isinstance(points, list):
        points_as_list_of_geom = []
        for current_point in points:
            if isinstance(current_point, tuple):
                current_point = _convertTupleToOGRPoint(
                    point_as_tuple=current_point,
                    point_srs_loaded=point_srs_loaded,
                    point_srs_is_set=point_srs_is_set,
                    output_srs=output_srs,
                )

            elif isinstance(current_point, ogr.Geometry):
                current_point = _transform_ogr_point_spatial_reference_system(
                    ogr_point=current_point, point_input_SRS_external=point_srs_loaded, output_srs=output_srs
                )

            else:
                raise GEOM.GeoKitGeomError(
                    "The list of points only accepts tuples or ogr.Geometry objects."
                    " However, an object of the following type has been passed: " + str(type(current_point))
                )

            points_as_list_of_geom.append(current_point)
    else:
        raise GEOM.GeoKitGeomError("The point argument")

    return points_as_list_of_geom


def extractValues(
    source: load_raster_input | list[load_raster_input],
    points: tuple[float, float]
    | ogr.Geometry
    | list[tuple[float, float]]
    | list[ogr.Geometry]
    | Location
    | LocationSet,
    pointSRS: srs_input | None = None,
    winRange: int = 0,
    noDataOkay: bool = True,
    _onlyValues: bool = False,
) -> ptValue | pd.DataFrame | numeric | np.ndarray:
    """Extracts the value of a raster at a given point or collection of points.
       Can also extract a window of values if desired.

    * If the given raster is not in the 'flipped-y' orientation, the result will
      be automatically flipped

    Notes
    -----
    Generally speaking, interpolateValues() should be used instead of this function

    Parameters
    ----------
    source : Anything acceptable by loadRaster() or list
        The raster datasource, can be a filepath, a raster dataset etc., see
        RASTER.loadRaster() for details. Alternatively, a list of multiple
        such raster datasources.

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

    _onlyValues: bool
        Return only the extracted data and ommit xOffeset, yOffset, inBounds

    Returns
    -------
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
        or returns numpy array if _onlyValues=True
    """
    # Be sure we have a raster and srs
    if not isinstance(source, list):
        source = [source]
    # make sure all source entries can be interpreted by loadRaster
    try:
        source = [loadRaster(s) for s in source]
    except:
        raise TypeError("At least one source cannot be loaded by geokit.raster.loadRaster().")
    raster_srs = None
    for s in source:
        # load file to make sure it can be interpreted by loadRaster
        if raster_srs is None:
            raster_srs = rasterInfo(s).srs
        else:
            assert raster_srs.IsSame(rasterInfo(s).srs), "All source entries must have the same SRS."

    # Ensure we have a list of point geometries
    points_as_list_of_geom = _convertPointsToListOfOGRPoints(
        points=points, point_input_SRS=pointSRS, output_srs=raster_srs
    )
    if len(points_as_list_of_geom) > 1:
        asSingle = False
    elif len(points_as_list_of_geom) == 1:
        asSingle = True
    else:
        raise GEOM.GeoKitGeomError("No points have been passed to the function extractValues()")

    # get the srcs that are actually overlapping with our points
    src_mapper = {}
    if len(source) > 1:
        for s in source:
            # get bounds and filter indices of points that fall into bounds
            _bounds = rasterInfo(s).bounds
            _indices = [
                i
                for i, p in enumerate(points_as_list_of_geom)
                if (_bounds[0] <= p.GetX() <= _bounds[2]) and (_bounds[1] <= p.GetY() <= _bounds[3])
            ]
            # add source as key plus list of overlapped point indices
            src_mapper[s] = _indices
    else:
        # we have only one source which must be applied to all points
        src_mapper[source[0]] = list(range(len(points_as_list_of_geom)))
        # srcs = source * len(points)

    # iterate over all unique srcs and extract the raster values for the affected points _src by _src

    values = [np.nan] * len(points_as_list_of_geom)  # initialize values with nan for every point
    inBounds = [False] * len(points_as_list_of_geom)  # initialize inbounds with False for every point
    xOffset = [np.nan] * len(points_as_list_of_geom)  # initialize offsets with nan for every point
    yOffset = [np.nan] * len(points_as_list_of_geom)  # initialize offsets with nan for every point

    for _src, _inds in src_mapper.items():
        # get the indices of the points for which data has been extracted already
        _xtrct = [i for i, v in enumerate(values) if np.all(pd.notnull(v))]
        # deduct them from the inds here (they may have been extracted already from an overlapping _src tile)
        _inds = list(set(_inds) - set(_xtrct))

        if len(_inds) == 0:
            # no indices left to extract from this _src
            continue

        # get the points with this _src via list indices
        _points = [points_as_list_of_geom[i] for i in _inds]

        # get the raster info for this _src
        _info = rasterInfo(_src)

        # Get x/y values as numpy arrays
        x = np.array([pt.GetX() for pt in _points])
        y = np.array([pt.GetY() for pt in _points])

        # Calculate x/y indexes
        xValues = (x - (_info.xMin + 0.5 * _info.pixelWidth)) / _info.pixelWidth
        xIndexes = np.round(xValues)
        _xOffset = xValues - xIndexes

        if _info.yAtTop:
            yValues = ((_info.yMax - 0.5 * _info.pixelWidth) - y) / abs(_info.pixelHeight)
            yIndexes = np.round(yValues)
            _yOffset = yValues - yIndexes
        else:
            yValues = (y - (_info.yMin + 0.5 * _info.pixelWidth)) / _info.pixelHeight
            yIndexes = np.round(yValues)
            _yOffset = -1 * (yValues - yIndexes)

        # Calculate the starts and window size
        xStarts = xIndexes - winRange
        yStarts = yIndexes - winRange
        window = 2 * winRange + 1

        _inBounds = xStarts >= 0
        _inBounds = _inBounds & (yStarts >= 0)
        _inBounds = _inBounds & (xStarts + window <= _info.xWinSize)
        _inBounds = _inBounds & (yStarts + window <= _info.yWinSize)

        # Read values
        _values = []
        band = _src.GetRasterBand(1)

        for xi, yi, ib in zip(xStarts, yStarts, _inBounds):
            if not ib:
                data = np.ones((window, window)) * np.nan
            else:
                # Open and read from raster
                data = band.ReadAsArray(xoff=xi, yoff=yi, win_xsize=window, win_ysize=window)
                if (_info.scale != None) and (_info.scale != 1.0):
                    data = data * _info.scale
                if (_info.offset != None) and (_info.offset != 0.0):
                    data = data + _info.offset

                # Look for nodata
                if _info.noData is not None:
                    nodata = data == _info.noData
                    if nodata.any():
                        if noDataOkay:
                            # data will neaed to be a float type to represent a nodata value
                            data = data.astype(np.float64)
                            data[nodata] = np.nan
                        else:
                            raise GeoKitRasterError(
                                "No data values found in extractValues with 'noDataOkay' set to False"
                            )

                # flip if not in the 'flipped-y' orientation
                if not _info.yAtTop:
                    data = data[::-1, :]

            if winRange == 0:
                # If winRange is 0, there's no need to return a 2D matrix
                data = data[0][0]

            # Append to values
            _values.append(data)

            # check if not _inbounds, then replace values with nan
            for i in range(len(_values)):
                if not _inBounds[i]:
                    _values[i] = np.nan * np.ones_like(_values[i])

        # write _values, _inbounds and offsets into the overall container at the respective indices
        for i, v, ib, x, y in zip(_inds, _values, _inBounds, _xOffset, _yOffset):
            values[i] = v
            inBounds[i] = ib
            xOffset[i] = x
            yOffset[i] = y

    if np.isnan(values).any():
        # we still have NaN values left
        msg = f"WARNING: {np.isnan(values).sum()} of the given points (or extraction windows) exceed/s the source's limits. Values were replaced with nan."
        warnings.warn(msg, UserWarning)

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
            return pd.DataFrame(
                dict(data=values, xOffset=xOffset, yOffset=yOffset, inBounds=inBounds),
                index=None,
            )


####################################################################
# Shortcut for getting just the raster value


def interpolateValues(
    source: load_raster_input,
    points: tuple[float, float]
    | ogr.Geometry
    | list[tuple[float, float]]
    | list[ogr.Geometry]
    | Location
    | LocationSet,
    pointSRS: srs_input = "latlon",
    mode: Literal["near", "linear-spline", "cubic-spline", "average"] = "near",
    func: Callable | None = None,
    winRange: int = 0,
    **kwargs,
):
    """Interpolates the value of a raster at a given point or collection of points.

    Supports various interpolation schemes:
        'near', 'linear-spline', 'cubic-spline', 'average', or user-defined

    Parameters
    ----------
    source : Anything acceptable by loadRaster() or list
        The raster datasource, can be a filepath, a raster dataset etc., see
        RASTER.loadRaster() for details. Alternatively, a list of multiple
        such raster datasources.

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

    Returns
    -------
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
        points = [
            points,
        ]
    else:  # Assume points is already an iterable of some sort
        asSingle = False

    # Do interpolation
    if mode == "near":
        # Simple get the nearest value
        win = 0 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win, _onlyValues=True)
        if asSingle is True:
            result = [values]
        else:
            result = values

    elif mode == "linear-spline":  # use a spline interpolation scheme
        # setup inputs
        win = 2 if winRange is None else winRange
        x = np.linspace(-1 * win, win, 2 * win + 1)
        y = np.linspace(-1 * win, win, 2 * win + 1)

        # get raw data
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        if asSingle is True:
            assert isinstance(values, ptValue)
            data_frame_values = pd.DataFrame(
                dict(
                    dict(data=[values.data], xOffset=values.xOffset, yOffset=values.yOffset, inBounds=values.inBounds),
                ),
                index=None,
            )
        else:
            data_frame_values = values
        # Calculate interpolated values
        result = []
        for v in data_frame_values.itertuples(index=False):
            rbs = RectBivariateSpline(y, x, v.data, kx=1, ky=1)

            result.append(rbs(v.yOffset, v.xOffset)[0][0])

    elif mode == "cubic-spline":  # use a spline interpolation scheme
        # setup inputs
        win = 4 if winRange is None else winRange
        x = np.linspace(-1 * win, win, 2 * win + 1)
        y = np.linspace(-1 * win, win, 2 * win + 1)

        # Get raw data
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)

        if asSingle is True:
            assert isinstance(values, ptValue)
            data_frame_values = pd.DataFrame(
                dict(
                    dict(data=[values.data], xOffset=values.xOffset, yOffset=values.yOffset, inBounds=values.inBounds),
                ),
                index=None,
            )
        else:
            data_frame_values = values
        # Calculate interpolated values
        result = []
        for v in data_frame_values.itertuples(index=False):
            rbs = RectBivariateSpline(y, x, v.data)

            result.append(rbs(v.yOffset, v.xOffset)[0][0])

    elif mode == "average":  # Get the average in a window
        win = 3 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        if asSingle is True:
            assert isinstance(values, ptValue)
            data_frame_values = pd.DataFrame(
                dict(
                    dict(data=[values.data], xOffset=values.xOffset, yOffset=values.yOffset, inBounds=values.inBounds),
                ),
                index=None,
            )
        else:
            data_frame_values = values
        result = []
        for v in data_frame_values.itertuples(index=False):
            result.append(v.data.mean())

    elif mode == "func":  # Use a general function processor
        if func is None:
            raise GeoKitRasterError("'func' mode chosen, but no func kwargs was given")
        win = 3 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        if asSingle is True:
            assert isinstance(values, ptValue)
            data_frame_values = pd.DataFrame(
                dict(
                    dict(data=[values.data], xOffset=values.xOffset, yOffset=values.yOffset, inBounds=values.inBounds),
                ),
                index=None,
            )
        else:
            data_frame_values = values
        result = []
        for v in data_frame_values.itertuples(index=False):
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


def mutateRaster(
    source,
    processor=None,
    bounds=None,
    boundsSRS="latlon",
    autocorrect=False,
    output=None,
    dtype=None,
    **kwargs,
):
    """Process all pixels in a raster according to a given function. The boundaries
    of the resulting raster can be changed as long as the new boundaries are within
    the scope of the original raster, but the resolution cannot.

    Parameters
    ----------
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
          matrices

    output : str; optional
        A path to an output file
        * If output is None, the raster will be created in memory and a dataset
          handle will be returned
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
    >>>     # create an output matrix
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
        source,
        bounds=bounds,
        boundsSRS=boundsSRS,
        autocorrect=autocorrect,
        returnBounds=True,
    )
    workingExtent = dsInfo.bounds if (bounds is None) else bounds

    # Perform processing
    processedData = processor(sourceData) if processor else sourceData
    if dtype and processedData.dtype != dtype:
        processedData = processedData.astype(dtype)

    dtype = gdalType(processedData.dtype)

    # Ensure returned matrix is okay
    if processedData.shape != sourceData.shape:
        raise GeoKitRasterError(
            "Processed matrix does not have the correct shape \nIs {0} \nShould be {1}",
            format(processedData.shape, sourceData.shape),
        )
    del sourceData

    # Create an output raster
    if output is None:
        dtype = gdalType(processedData.dtype) if dtype is None else gdalType(dtype)
        return UTIL.quickRaster(
            dy=dsInfo.dy,
            dx=dsInfo.dx,
            bounds=workingExtent,
            dtype=dtype,
            srs=dsInfo.srs,
            data=processedData,
            **kwargs,
        )

    else:
        createRaster(
            pixelHeight=dsInfo.dy,
            pixelWidth=dsInfo.dx,
            bounds=workingExtent,
            srs=dsInfo.srs,
            data=processedData,
            output=output,
            **kwargs,
        )

        return output


def indexToCoord(
    yi: int | np.ndarray,
    xi: int | np.ndarray,
    source=None,
    asPoint: bool = False,
    bounds=None,
    dx=None,
    dy=None,
    yAtTop=True,
    srs=None,
) -> ogr.Geometry | tuple[int | int] | np.ndarray:
    """Convert the index of a raster to coordinate values.

    Parameters
    ----------
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

    Returns
    -------
    * If 'asPoint' is True: ogr.Geometry
    * If 'asPoint' is False: tuple -> (x,y) coordinates
    """
    if source is not None:
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

    # Calculate coordinates
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


def drawSmopyMap(
    bounds,
    zoom,
    tileserver="https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
    tilesize: int = 256,
    maxtiles: int = 100,
    ax=None,
    attribution="© OpenStreetMap contributors",
    attribution_size=12,
    **kwargs,
):
    """
    Draws a basemap using the "smopy" python package.

    NOTE:
    * The basemap is drawn using the Smopy python package. See here: https://github.com/rossant/smopy
    * Be careful to adhere to the usage guidelines of the chosen tile source
        - By default, this source is OSM. See here: https://wiki.openstreetmap.org/wiki/Tile_servers

    !IMPORTANT! If you will publish any images drawn with this method, it's likely that the tile source
    will require an attribution to be written on the image. For example, if using OSM tile (the default),
    you have to write "© OpenStreetMap contributors" clearly on the map. But this is different for each
    tile source!

    Parameters
    ----------
        bounds : (xMin, yMix, xMax, yMax) or Extent
            The geographic extent to be drawn

        zoom : int
            The zoom level to draw (between 1-20)
            * I suggest starting low (e.g. 4), and zooming in until you find a level that suits your needs

        tileserver : string
            The tile server to use

        tilesize : int
            The pixel size of the tiles from 'tileserver'

        maxtiles : int
            The maximum tiles to use when drawing an image
            * Be careful to adhere to the usage conditions stated by your selected tileserver!

        ax : matplotlib.axes
            The matplotlib axes to draw on
            * If 'None', then one will be generated automatically

        kwargs
            All extra keyword arguments are passed on to matplotlib.ax.imshow

    Returns
    -------
        namedtuple
            * .ax     -> The axes draw on
            * .srs    -> The SRS used when drawing (will always be EPSG 3857)
            * .bounds -> The boundaries of the drawn map
    """
    import smopy

    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    try:
        lon_min, lat_min, lon_max, lat_max = bounds.xyXY
    except:
        lon_min, lat_min, lon_max, lat_max = bounds

    tile_box = smopy.get_tile_box((lat_max, lon_min, lat_min, lon_max), zoom)

    img = smopy.fetch_map(
        box=tile_box,
        z=zoom,
        tileserver=tileserver,
        tilesize=tilesize,
        maxtiles=maxtiles,
    )

    tl_lat, tl_lon = smopy.num2deg(tile_box[0], tile_box[1], zoom=zoom)
    br_lat, br_lon = smopy.num2deg(tile_box[2] + 1, tile_box[3] + 1, zoom=zoom)

    xy = SRS.xyTransform(
        [(tl_lon, tl_lat), (br_lon, br_lat)],
        fromSRS=SRS.EPSG4326,
        toSRS=SRS.EPSG3857,
        outputFormat="xy",
    )

    ax.imshow(img, extent=(xy.x.min(), xy.x.max(), xy.y.min(), xy.y.max()), **kwargs)

    if attribution is not None:
        ax.text(
            1,
            0,
            attribution,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            zorder=10,
            fontsize=attribution_size,
        )

    SmopyMap = namedtuple("SmopyMap", "ax srs bounds")

    return SmopyMap(ax, SRS.EPSG3857, (tl_lon, br_lat, br_lon, tl_lat))


def drawRaster(
    source: load_raster_input | pd.DataFrame,
    srs: srs_input | None = None,
    ax: matplotlib.axes._axes.Axes | None | AxHands = None,
    resolution: numeric | None = None,
    cutline=None,
    figsize: tuple[numeric, numeric] = (12, 12),
    xlim: tuple[numeric, numeric] | None = None,
    ylim: tuple[numeric, numeric] | None = None,
    fontsize: int = 16,
    hideAxis: bool = False,
    cbar: bool = True,
    cbarTitle: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap="viridis",
    cbargs=None,
    cutlineFillValue=-9999,
    zorder=0,
    resampleAlg: Literal[
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "rms",
        "mode",
        "max",
        "min",
        "med",
        "Q1",
        "Q3",
        "sum",
    ] = "med",
    **warp_kwargs,
) -> AxHands:
    """Draw a raster as an image on a matplotlib canvas.

    Parameters
    ----------
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
        * Changing the resolution from the inherent resolution requires a warp

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
          * If resultign figure looks weird, altering the figure size is your best
            bet to make it look nicer

    xlim : (float, float); optional
        The x-axis limits

    ylim : (float, float); optional
        The y-axis limits

    fontsize : int; optional
        A base font size to apply to tick marks which appear
          * Titles and labels are given a size of 'fontsize' + 2

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

    cbargs : dict; optional

    resampleAlg : str, optional
        The resampleAlg passed on to a call of warp() if needed, by default "med"

    **kwargs : Passed on to a call to warp()
        * Determines how the warping is carried out
        * Consider using 'resampleAlg' or 'workingType' for finer control

    Returns
    -------
    A namedtuple containing:
       'ax' -> The map axis
       'handles' -> All geometry handles which were created in the order they were
                    drawn
       'cbar' -> The colorbar handle if it was drawn
    """
    if isinstance(ax, AxHands):
        ax = ax.ax
    if ax is None:
        new_main_axis = True
        fig, ax = plt.subplots(figsize=figsize)
    elif isinstance(ax, matplotlib.axes._axes.Axes):
        new_main_axis = False
    else:
        raise Exception(
            "Expected None or matplotlib.axes._axes.Axes object forr the 'ax' argument. However, an object of type: "
            + str(type(ax))
            + " has been provided."
        )
    if hideAxis:
        ax.axis("off")

    if cbar is False:
        cbax = None
    else:
        divider = make_axes_locatable(ax)
        cbax = divider.append_axes("right", size="2.5%", pad=0.05)

    ax.tick_params(labelsize=fontsize)

    # Load the raster datasource and check for transformation
    source = loadRaster(source)
    info = rasterInfo(source)

    if not (srs is None and resolution is None and cutline is None and xlim is None and ylim is None):
        if not (xlim is None and ylim is None):
            bounds = (
                xlim[0],
                ylim[0],
                xlim[1],
                ylim[1],
            )
        else:
            bounds = None

        if resolution is None:
            xres, yres = None, None
        else:
            try:
                xres, yres = resolution
            except:
                xres, yres = resolution, resolution

        source = warp(
            source,
            cutline=cutline,
            pixelHeight=yres,
            pixelWidth=xres,
            srs=srs,
            bounds=bounds,
            fill=cutlineFillValue,
            noData=None,
            resampleAlg=resampleAlg,
            **warp_kwargs,
        )

    info = rasterInfo(source)

    # Read the Data
    data = extractMatrix(source).astype(float)
    if cutlineFillValue is not None:
        data[data == cutlineFillValue] = np.nan

    data[data == info.noData] = np.nan

    # Draw image
    ext = (
        info.xMin,
        info.xMax,
        info.yMin,
        info.yMax,
    )
    h = ax.imshow(
        data,
        extent=ext,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        zorder=zorder,
        interpolation="none",
    )

    # Draw Colorbar
    if cbar is True:
        tmp = dict(cmap=cmap, orientation="vertical")
        if cbargs is not None:
            tmp.update(cbargs)

        cbar_output = plt.colorbar(h, cax=cbax, **tmp)
        cbar_output.ax.tick_params(labelsize=fontsize)
        if cbarTitle is not None:
            cbar_output.set_label(cbarTitle, fontsize=fontsize + 2)
    else:
        cbar_output = None

    # Do some formatting
    ax.set_aspect("equal")
    ax.autoscale(enable=True)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Done!
    return UTIL.AxHands(ax, h, cbar_output)


# 3
# Make a geometry from a matrix mask
#
# NOTE TO ME
# This function is VERRYYYYYYY similar to polygonizeMatrix, but since it does not
# not actually have to deal with the matrix data (which is all done by GDAL), this should be
# kept along side polygonizeMatrix


def polygonizeRaster(source, srs=None, flat=False, shrink=True):
    """Polygonize a raster or an integer-valued data matrix.

    Parameters
    ----------
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

    Returns
    -------
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
    vecDS = gdal.GetDriverByName("Memory").Create("", 0, 0, 0, gdal.GDT_Unknown)
    vecLyr = vecDS.CreateLayer("mem", srs=srs)

    # vecDS = gdal.GetDriverByName("ESRI Shapefile").Create("deleteme.tif", 0, 0, 0, gdal.GDT_Unknown )
    # vecLyr = vecDS.CreateLayer("layer",srs=srs)

    vecField = ogr.FieldDefn("DN", ogr.OFTInteger)
    vecLyr.CreateField(vecField)

    # Polygonize geometry
    result = gdal.Polygonize(band, maskBand, vecLyr, 0)
    if result != 0:
        raise GEOM.GeoKitGeomError("Failed to polygonize geometry")

    # Check the geoms
    ftrN = vecLyr.GetFeatureCount()
    if ftrN == 0:
        # raise GlaesError("No features in created in temporary layer")
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
        geoms = [g.Buffer(float(shrinkFactor)) for g in geoms]

    # Do flatten, maybe
    if flat:
        geoms = np.array(geoms)
        rid = np.array(rid)

        finalGeoms = []
        finalRID = []
        for _rid in set(rid):
            smallGeomSet = geoms[rid == _rid]
            finalGeoms.append(GEOM.flatten(smallGeomSet) if len(smallGeomSet) > 1 else smallGeomSet[0])
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


def contours(
    source: load_raster_input,
    contourEdges: list[float] | None,
    polygonize: bool = True,
    unpack: bool = True,
    raster_band_index: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """Create contour geometries at specified edges for the given raster data.

    Notes
    -----
    This function is similar to geokit.geom.polygonizeMatrix, although it only
    operates on the user-specified edges AND applies the 'Marching Squares'
    algorithm

    See the gdal function "GDALContourGenerateEx" for more information on the
    specifics of this algorithm

    Parameters
    ----------
    source : Anything acceptable by loadRaster()
        The raster datasource to operate on

    contourEdges : list[float] | None
        The edges to search for within the raster dataset
          * This parameter can be set as "None", in which case an additional
            argument should be given to specify how the edges should be determined
            - See the documentation of "GDALContourGenerateEx"
            - Ex. "LEVEL_INTERVAL=10", contourEdges=None

    polygons : bool
        If true, contours are returned as polygons instead of linstrings

    unpack : bool
        If True, Multipolygon/MultiLinestring objects are decomposed

    raster_band_index : int
        The index of the raster which should be loaded from the raster band.

    **kwargs:
        * All keyword arguments are passed on to a call to gdal.ContourGenerateEx
        * They are used to construct the 'options' parameter
        * Example keys include: LEVEL_INTERVAL, LEVEL_BASE, LEVEL_EXP_BASE, NODATA
        * Do not use the key "ID_FIELD", since this is employed already

    Returns
    -------
    pandas.DataFrame

    * The column 'geom' corresponds to generated geometry objects
    * The columns 'ID' corresponds to the associated contour edge for each object
    """
    warnings.warn(
        message="The current behavior of geokits's contours function is deprecated. GDAL has changed"
        " how contours are drawn close to minimum and maximum values, and will discontinue"
        " the current behavior in GDAL 3.11 or 3.12. Geokit will also drop the current"
        " behavior with the next GDAL update. For more information, please see the"
        " following discussion: https://github.com/OSGeo/gdal/issues/12938.",
        category=DeprecationWarning,
    )
    # Open raster
    raster = loadRaster(source)
    band = raster.GetRasterBand(raster_band_index)

    rasterSRS = SRS.loadSRS(raster.GetProjectionRef())

    # Make temporary vector
    driver: Driver = gdal.GetDriverByName("Memory")
    source: gdal.Dataset = driver.Create("", 0, 0, 0, gdal.GDT_Unknown)

    layer: ogr.Layer = source.CreateLayer("", rasterSRS, ogr.wkbPolygon if polygonize else ogr.wkbLineString)
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
    iterator_n = 0
    for ftrid in range(layer.GetFeatureCount()):
        ftr: ogr.Feature = layer.GetFeature(ftrid)
        geom = ftr.GetGeometryRef()
        value = ftr.GetField(0)
        print(iterator_n)
        iterator_n = iterator_n + 1
        if unpack:
            for gi in range(geom.GetGeometryCount()):
                geoms.append(geom.GetGeometryRef(gi).Clone())
                IDs.append(value)
        else:
            geoms.append(geom.Clone())
            IDs.append(value)

    countour_data_frame = pd.DataFrame(dict(geom=geoms, ID=IDs))

    # return geoms
    return countour_data_frame


def warp(
    source: load_raster_input,
    resampleAlg: Literal[
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "rms",
        "mode",
        "max",
        "min",
        "med",
        "Q1",
        "Q3",
        "sum",
    ] = "bilinear",
    cutline: str | ogr.Geometry | None = None,
    output: str | pathlib.Path | None = None,
    pixelHeight: numeric | None = None,
    pixelWidth: numeric | None = None,
    srs: srs_input | None = None,
    bounds: tuple[numeric, numeric, numeric, numeric] | None = None,
    dtype=None,
    noData=None,
    fill: numeric | None = None,
    overwrite: bool = True,
    meta: None | dict[str, str] = None,
    **kwargs,
) -> gdal.Dataset | str:
    """Warps a given raster source to another context.

    * Can be used to 'warp' a raster in memory to a raster on disk

    Note:
    -----
    Unless manually altered as keyword arguments, the gdal.Warp options
    'targetAlignedPixels' and 'copyMetadata' are both set to True

    Parameters
    ----------
    source : Anything acceptable by loadRaster()
        The raster datasource to draw

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the resulting raster
          * If not given, the raster's internal srs is assumed

    resampleAlg : str; optional
        The resampling algorithm to use when translating pixel values
        * Knowing which option to use can have significant impacts!
        * Options are: near , bilinear, cubic,
        cubicspline, lanczos, average, rms, mode,
        max, min, med, Q1, Q3, sum

    cutline : str or ogr.Geometry; optional
        The cutline to limit the drawn data too
        * If a string is given, it must be a path to a vector file
        * Values outside of the cutline are given the value 'cutlineFillValue'
        * Requires a warp

    output : str; pathlib.Path optional
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

    meta: dict; optional: contains a key value pair that is passed to the
          output gdal.dataset using the SetMetadataItem method.

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

    Returns
    -------
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
    if srs is not None:
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
    if cutline is not None:
        if isinstance(cutline, ogr.Geometry):
            tempdir = TemporaryDirectory()
            cutline = UTIL.quickVector(cutline, output=os.path.join(tempdir.name, "tmp.shp"))
        # cutline is already a path to a vector
        elif UTIL.isVector(cutline):
            tempdir = None
        else:
            raise GeoKitRasterError("cutline must be a Geometry or a path to a shape file")

    # Workflow depends on whether or not we have an output
    if isinstance(output, pathlib.Path):
        output = str(output)
    if isinstance(output, str):  # Simply do a translate
        if os.path.isfile(output):
            if overwrite is True:
                os.remove(output)
                if os.path.isfile(output + ".aux.xml"):  # Because QGIS....
                    os.remove(output + ".aux.xml")
            else:
                raise GeoKitRasterError("Output file already exists: %s" % output)

        # # Check some for bad input configurations
        # if not srs is None:
        #     if (pixelHeight is None or pixelWidth is None):
        #         raise GeoKitRasterError("When warping between srs's and writing to a file, pixelWidth and pixelHeight must be given")

        # Arange inputs
        co = kwargs.pop("creationOptions", COMPRESSION_OPTION)
        copyMeta = kwargs.pop("copyMetadata", True)
        aligned = kwargs.pop("targetAlignedPixels", True)

        # Fix the bounds issue by making them  just a little bit smaller, which should be fixed by gdalwarp
        bounds = (
            bounds[0] + 0.001 * pixelWidth,
            bounds[1] + 0.001 * pixelHeight,
            bounds[2] - 0.001 * pixelWidth,
            bounds[3] - 0.001 * pixelHeight,
        )

        # Let gdalwarp do everything...
        gdal_warp_options = gdal.WarpOptions(
            outputType=getattr(gdal, dtype),
            xRes=pixelWidth,
            yRes=pixelHeight,
            creationOptions=co,
            outputBounds=bounds,
            dstSRS=srs,
            dstNodata=noData,
            resampleAlg=resampleAlg,
            copyMetadata=copyMeta,
            targetAlignedPixels=aligned,
            cutlineDSName=cutline,
            **kwargs,
        )

        result_dataset = gdal.Warp(destNameOrDestDS=output, srcDSOrSrcDSTab=source, options=gdal_warp_options)
        if not UTIL.isRaster(result_dataset):
            raise GeoKitRasterError("Failed to translate raster")

        destination_raster = output

    else:
        if "cropToCutline" in kwargs:
            msg = "The 'cropToCutline' option is not taken into account when writing to a raster in memory. Try using geokit.Extent.warp instead"
            warnings.warn(msg, UserWarning)

        # Warp to a raster in memory
        destination_raster = UTIL.quickRaster(
            bounds=bounds,
            srs=srs,
            dx=pixelWidth,
            dy=pixelHeight,
            dtype=dtype,
            noData=noData,
            fill=fill,
        )

        # Do a warp
        result_dataset = gdal.Warp(destination_raster, source, resampleAlg=resampleAlg, cutlineDSName=cutline, **kwargs)

        destination_raster.FlushCache()
    # Do we have meta data?
    if meta is not None:
        if isinstance(destination_raster, str):
            loaded_output_raster = loadRaster(destination_raster, 1)
        else:
            loaded_output_raster = destination_raster

        for k, v in meta.items():
            loaded_output_raster.SetMetadataItem(k, v)

        # FlushCache writes all changes to disk
        loaded_output_raster.FlushCache()

    if cutline is not None:
        del tempdir
    return destination_raster


def warpLike(dataSource, contextSource, copyMetadata=False, **kwargs):
    """
    Convenience function to warp a raster to the context of another raster
    as returned from a call to rasterInfo(contextSource).

    dataSource : Anything acceptable by loadRaster()
        The raster data source to draw
    contextSource : Anything acceptable by loadRaster()
        The raster context source to draw, i.e. the data source raster will be
        warped to this pixelWidth, pixelHeight, bounds, extent, srs etc.
    copyMetadata : bool, optional
        If True, the metadata of the dataSource raster will be copied, else
        metadata will be empty or as possibly provided in kwargs. Defaults to False.
    **kwargs
        All kwargs will be passed on to raster.warp().
    """
    if UTIL.isRaster(dataSource):
        dataInfo = rasterInfo(dataSource)
    if not isinstance(dataInfo, RasterInfo):
        raise GeoKitRasterError("Could not understand dataSource")
    if UTIL.isRaster(contextSource):
        contextInfo = rasterInfo(contextSource)
    if not isinstance(contextInfo, RasterInfo):
        raise GeoKitRasterError("Could not understand contextSource")

    # first get data-related parameters from DATA source
    if copyMetadata:
        if "meta" in kwargs:
            raise GeoKitRasterError("If metadata is given as 'meta' in kwargs, copyMetadata cannot be True!")
        meta = dataInfo.meta
    else:
        meta = kwargs.pop("meta", None)
    print(meta)
    dtype = kwargs.pop("dtype", dataInfo.dtype)
    noData = kwargs.pop("noData", dataInfo.noData)
    if "cutline" in kwargs:
        # make sure that the cells outside are filled with noData if not specified
        fill = kwargs.pop("fill", dataInfo.noData)
    else:
        # set to default of warp() function for consistent behavior
        fill = inspect.signature(warp).parameters["fill"].default

    # then get context related parameters from CONTEXT source
    bounds = kwargs.pop("bounds", contextInfo.bounds)
    pixelWidth = kwargs.pop("pixelWidth", contextInfo.pixelWidth)
    pixelHeight = kwargs.pop("pixelHeight", contextInfo.pixelHeight)
    srs = kwargs.pop("srs", contextInfo.srs)

    return warp(
        source=dataSource,
        bounds=bounds,
        pixelWidth=pixelWidth,
        pixelHeight=pixelHeight,
        dtype=dtype,
        srs=srs,
        noData=noData,
        meta=meta,
        fill=fill,
        **kwargs,
    )


# --------------------------------------------------------------------------------------------
# Sieve raster datasets
# --------------------------------------------------------------------------------------------


def sieve(
    source,
    threshold: int = 100,
    connectedness: Literal[4, 8] = 8,
    mask="none",
    quiet_flag: bool = False,
    output: str | None = None,
    **kwargs,
) -> gdal.Dataset | str:
    """
    Removes raster polygons smaller than a provided threshold size (in pixels) and
    replaces them with the pixel value of the largest neighbour polygon.
    It is useful if you have a large amount of small areas on your raster map.

    Returns
    -------
    * If 'output' is None : gdal.Dataset
    * If 'output' is a string : The path to the output is returned (for easy opening)

    Parameters
    ----------
    source : Anything acceptable by loadRaster()

    threshold (int): minimum polygon size (number of pixels) to retain.

    connectedness (int): either 4 indicating that diagonal pixels are not considered directly
                         adjacent for polygon membership purposes or 8 indicating they are.

    mask (str): 'none' or 'default'. An optional mask band. All pixels in the mask band with a
                value other than zero will be considered suitable for inclusion in polygons.

    quiet_flag (bool): 0 or 1. Callback for reporting algorithm progress

    output : str; optional
        The path on disk where the new raster should be created

    **kwargs:
        * All kwargs are passed on to SieveFilter()
        * See gdal.SieveFilter for more details
    """
    gdal.AllRegister()

    try:
        gdal.SieveFilter
    except:
        print("")
        print('gdal.SieveFilter() not available.  You are likely using "old gen"')
        print("bindings or an older version of the next gen bindings.")
        print("")

    ### Open source file
    # if output is None:
    #     src_ds = gdal.Open( source, gdal.GA_Update )
    # else:
    #     src_ds = gdal.Open( source, gdal.GA_ReadOnly )

    src_ds = loadRaster(source)

    if src_ds is None:
        print("Unable to open %s " % source)

    srcband = src_ds.GetRasterBand(1)

    if mask == "default":
        maskband = srcband.GetMaskBand()
    elif mask == "none":
        maskband = None
    else:
        mask_ds = loadRaster(mask)
        maskband = mask_ds.GetRasterBand(1).GetMaskBand()

    ### Create output file if one is specified.

    if output is not None:
        format = "GTiff"
        driver = gdal.GetDriverByName(format)
        dst_ds = driver.Create(
            output,
            src_ds.RasterXSize,
            src_ds.RasterYSize,
            1,
            srcband.DataType,
            COMPRESSION_OPTION,
        )  # ["COMPRESS=LZW"]

    else:
        format = "Mem"
        driver = gdal.GetDriverByName(format)  # create a raster in memory
        dst_ds = driver.Create(
            "",
            src_ds.RasterXSize,
            src_ds.RasterYSize,
            1,
            srcband.DataType,
            COMPRESSION_OPTION,
        )  # ["COMPRESS=LZW"]

    wkt = src_ds.GetProjection()
    if wkt != "":
        dst_ds.SetProjection(wkt)
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dstband = dst_ds.GetRasterBand(1)

    if quiet_flag:
        prog_func = None
    else:
        prog_func = gdal.TermProgress_nocb

    result = gdal.SieveFilter(
        srcband,
        maskband,
        dstband,
        threshold,
        connectedness,
        callback=prog_func,
        **kwargs,
    )

    dst_ds.FlushCache()

    # Return raster if in memory
    if output is None:
        return dst_ds

    # Return output path to raster if on disk
    else:
        src_ds = None
        dst_ds = None
        mask_ds = None
        return output


def rasterCellNo(points, source=None, bounds=None, cellWidth=None, cellHeight=None):
    """
    Returns the raster cell number for one or multiple points defined by geometry or lon/lat. Cell numeration
    starting in the top left corner cell of the raster with (0,0). Cells with (-1,-1) are out of bounds.

    Args:
        points (osgeo.ogr.Geometry, tuple, iterable): Can be an osgeo.ogr.Geometry point or an iterable thereof, else a (lon, lat) tuple (in EPSG:4326) or an iterable thereof.
        source (gdal.Dataset, str, optional): A gdal.Dataset type raster or a str formatted path to a raster file. Defaults to None.
        bounds (tuple, optional): Raster boundaries in EPSG:4326 in the form of (minX, minY, maxX, maxY). Defaults to None.
        cellWidth (int, float, optional): The cell width in EPSG:4326 units. Defaults to None.
        cellHeight (int, float, optional): The cell height in EPSG:4326 units. Defaults to None.
    NOTE: If source is given, all of the others must be None, else they must be given.

    Returns
    -------
        tuple or iterable: tuple with (X, Y) cell No or an iterable thereof if multiple points were given.
    """
    # check and preprocess points inputs
    if isinstance(points, ogr.Geometry):
        points = [points]
    elif isinstance(points, tuple) and len(points) == 2:
        points = [points]
    assert hasattr(points, "__iter__"), (
        "points must be an osgeo.ogr.Geometry POINT object, a tuple of (lon, lat) or an iterable of any of them."
    )

    if isinstance(points[0], ogr.Geometry):
        if not all([p.GetGeometryName() == "POINT" for p in points]):
            raise TypeError("Only POINT geometries allowed")
        if not all([p.GetSpatialReference().IsSame(SRS.loadSRS(4326)) for p in points]):
            raise ValueError("SRS of all points must be EPSG:4326")
        # convert to lons and lats
        points = [(p.GetX(), p.GetY()) for p in points]
    else:
        assert all(
            [isinstance(x, tuple) and len(x) == 2 and all([isinstance(_x, (int, float)) for _x in x]) for x in points]
        ), (
            "All entries in points must be (lon, lat) tuples with length of 2 and int or float coordinates if given as tuples or iterable thereof."
        )

    # get bounds, cellWidth and cellHeight from the inputs
    if source is not None:
        if not (bounds is None and cellWidth is None and cellHeight is None):
            raise ValueError("bounds, cellWidth and cellHeight must be None when source raster is given.")
        if isinstance(source, str) and not os.path.isfile(source):
            raise FileNotFoundError("source must be an existing raster file if given as string.")
        try:
            sourceRasterInfo = rasterInfo(source)
            bounds = sourceRasterInfo.bounds
            cellWidth = sourceRasterInfo.pixelWidth
            cellHeight = sourceRasterInfo.pixelHeight
            rasterWidth = sourceRasterInfo.xWinSize
            rasterHeight = sourceRasterInfo.yWinSize
        except:
            raise TypeError("source must be path to a gdal.Dataset raster or a gdal.Dataset raster itself if not None.")
        if not sourceRasterInfo.srs.IsSame(SRS.loadSRS(4326)):
            raise ValueError("raster source must be in epsg:4326")
    else:
        if bounds is None or cellWidth is None or cellHeight is None:
            raise ValueError("If source is None, bounds, cellWidth and cellHeight must be given.")
        assert (
            isinstance(bounds, tuple)
            and len(bounds) == 4
            and all([isinstance(x, (int, float)) for x in bounds])
            and bounds[0] < bounds[2]
            and bounds[1] < bounds[3]
        ), "bounds must be a tuple of length = 4 with int or float entries like such: (minX, minY, maxX, maxY)"
        assert isinstance(cellHeight, (int, float)), "cellHeight must be an int or float."
        assert isinstance(cellWidth, (int, float)), "cellWidth must be an int or float."

        # calculate the raster width and height in Nos of cells
        rasterWidth = (bounds[2] - bounds[0]) / cellWidth
        rasterHeight = (bounds[3] - bounds[1]) / cellHeight
        assert np.isclose(rasterWidth % cellWidth, 0) or np.isclose(rasterWidth % cellWidth, cellWidth), (
            f"rasterWidth {rasterWidth} is not a multiple of cellWidth {cellWidth}"
        )
        assert np.isclose(rasterHeight % cellHeight, 0) or np.isclose(rasterHeight % cellHeight, cellHeight), (
            f"rasterHeight {rasterHeight} is not a multiple of cellHeight {cellHeight}"
        )

    # calculate the cell Nos
    cellNos = list()
    outOfBounds = False
    for point in points:
        X = int(np.floor((point[0] - bounds[0]) / cellWidth))
        Y = int(np.floor((bounds[3] - point[1]) / cellHeight))

        # if any dimension out of bounds
        if (X < 0 or X >= rasterWidth) or (Y < 0 or Y >= rasterHeight):
            X = Y = -1
            outOfBounds = True
        # append to cell No list
        cellNos.append((X, Y))

    if outOfBounds:
        warnings.warn("points contain out-of-bounds locations!")

    # return cellNos as tuple for single point, else as list of tuples
    if len(cellNos) == 1:
        return cellNos[0]
    else:
        return cellNos
