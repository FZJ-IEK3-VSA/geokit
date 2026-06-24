"""The Util sub-module contains a number of generally helpful utility functions, classes, and constants."""

import math
import os
import re
import sys
import warnings
from collections import OrderedDict, defaultdict, namedtuple
from collections.abc import Iterable
from decimal import Decimal
from fractions import Fraction
from glob import glob
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import GeneratorType
from typing import NamedTuple

import matplotlib.axis
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib.axes._axes import Axes
from matplotlib.axis import Axis
from osgeo import gdal, ogr, osr
from scipy.interpolate import RectBivariateSpline
from scipy.stats import describe

from geokit.c_data_type_handler import MinimumCDataTypeHandler
from geokit.data_types import AxHands, numeric, srs_input, geokit_c_data_types_literal
from geokit.error import GeoKitError

######################################################################################
# test modules

# The main SRS for lat/lon coordinates
_test = osr.SpatialReference()
res = _test.ImportFromEPSG(4326)

# Quick check if gdal loaded properly
if not res == 0:
    raise RuntimeError("GDAL did not load properly. Check your 'GDAL_DATA' environment variable")

######################################################################################
# An few errors just for me!


#####################################################################
# Testers


def isVector(source):
    """
    Test if loadVector fails for the given input.

    Parameters
    ----------
    source : str
        The path to the vector file to load

    Returns
    -------
    bool -> True if the given input is a vector
    """
    if isinstance(source, gdal.Dataset):
        if source.GetLayerCount() > 0:
            return True
        else:
            return False
    elif isinstance(source, str):
        d = gdal.IdentifyDriver(source)

        meta = d.GetMetadata()
        if meta.get("DCAP_VECTOR", False) == "YES":
            return True
        else:
            return False
    else:
        return False


def isRaster(source):
    """
    Test if loadRaster fails for the given input.

    Parameters
    ----------
    source : str
        The path to the raster file to load

    Returns
    -------
    bool -> True if the given input is a raster
    """
    if isinstance(source, gdal.Dataset):
        try:
            if source.GetLayerCount() == 0:
                return True  # This should always be true?
            else:
                return False
        except:
            return False
    elif isinstance(source, str):
        d = gdal.IdentifyDriver(source)

        meta = d.GetMetadata()
        if meta.get("DCAP_RASTER", False) == "YES":
            return True
        else:
            return False
    else:
        return False


##################################################################
# General funcs
# matrix scaler
def scaleMatrix(mat, scale, strict=True):
    """Scale a 2-dimensional matrix. For example, a 2x2 matrix, with a scale of 2,
    will become a 4x4 matrix. Or scaling a 24x24 matrix with a scale of -3 will
    produce an 8x8 matrix.

    * Scaling UP (positive) results in a dimensionally larger matrix where each
      value is repeated scale^2 times
    * scaling DOWN (negative) results in a dimensionally smaller matrix where each
      value is the average of the associated 'up-scaled' block

    Parameters
    ----------
        mat : numpy.ndarray or [[numeric,],]
            The data to be scaled
              * Must be two dimensional

        scale : int or (int, int)
            The dimensional scaling factor for either both, or independently for
            the Y and X dimensions
              * If scaling down, the scaling factors must be a factor of the their
                associated dimension in the input matrix (unless 'strict' is set
                to False)

        strict : bool
            Whether or not to force a fail when scaling-down by a scaling factor
            which is not a dimensional factor
              * Really intended for internal use...
              * When scaling down by a non-dimensional factor, the matrix will be
                padded with zeros such that the new matrix has dimensional sizes
                which are divisible by the scaling factor. The points which are
                not at the right or bottom boundary are averaged, same as before.
                The points which lie on the edge however, are also averaged across
                all the values which lie in those pixels, but they are corrected
                so that the averaging does NOT take into account the padded zeros.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    INPUT       Scaling Factor      Output
    --------------------------------------

    | 1 2 |             2           | 1 1 2 2 |
    | 3 4 |                         | 1 1 2 2 |
                                    | 3 3 4 4 |
                                    | 3 3 4 4 |

    | 1 1 1 1 |        -2           | 1.5  2.0 |
    | 2 2 3 3 |                     | 5.25 6.75|
    | 4 4 5 5 |
    | 6 7 8 9 |

    | 1 1 1 1 |        -3           | 2.55  3.0 |
    | 2 2 3 3 |   * strict=False    | 7.0    9  |
    | 4 4 5 5 |

    | 6 7 8 9 |       *padded*
                    --------------------------
                   | 1 1 1 1 0 0 |
                   | 2 2 3 3 0 0 |
                   | 4 4 5 5 0 0 |
                   | 6 7 8 9 0 0 |
                   | 0 0 0 0 0 0 |
                   | 0 0 0 0 0 0 |
    """
    # unpack scale
    try:
        yScale, xScale = scale
    except:
        yScale, xScale = scale, scale

    # check for ints
    if not (isinstance(xScale, int) and isinstance(yScale, int)):
        raise ValueError("scale must be integer types")

    if xScale == 0 and yScale == 0:
        return mat  # no scaling (it would just be silly to call this)
    elif xScale > 0 and yScale > 0:  # scale up
        out = np.zeros((mat.shape[0] * yScale, mat.shape[1] * xScale), dtype=mat.dtype)
        for yo in range(yScale):
            for xo in range(xScale):
                out[yo::yScale, xo::xScale] = mat

    elif xScale < 0 and yScale < 0:  # scale down
        xScale = -1 * xScale
        yScale = -1 * yScale
        # ensure scale is a factor of both xSize and ySize
        if strict:
            if not (mat.shape[0] % yScale == 0 and mat.shape[1] % xScale == 0):
                raise GeoKitError("Matrix can only be scaled down by a factor of it's dimensions")
            yPad = 0
            xPad = 0
        else:
            # get the amount to pad in the y direction
            yPad = yScale - mat.shape[0] % yScale
            # get the amount to pad in the x direction
            xPad = xScale - mat.shape[1] % xScale

            if yPad == yScale:
                yPad = 0
            if xPad == xScale:
                xPad = 0

            # Do y-padding
            if yPad > 0:
                mat = np.concatenate((mat, np.zeros((yPad, mat.shape[1]))), 0)
            if xPad > 0:
                mat = np.concatenate((mat, np.zeros((mat.shape[0], xPad))), 1)

        out = np.zeros((mat.shape[0] // yScale, mat.shape[1] // xScale), dtype="float")
        for yo in range(yScale):
            for xo in range(xScale):
                out += mat[yo::yScale, xo::xScale]
        out = out / (xScale * yScale)

        # Correct the edges if a padding was provided
        if yPad > 0:
            # fix the right edge EXCLUDING the bot-left point
            out[:-1, -1] *= yScale / (yScale - yPad)
        if xPad > 0:
            # fix the bottom edge EXCLUDING the bot-left point
            out[-1, :-1] *= xScale / (xScale - xPad)
        if yPad > 0:
            out[-1, -1] *= yScale * xScale / (yScale - yPad) / (xScale - xPad)  # fix the bot-left point

    else:  # we have both a scaleup and a scale down
        raise GeoKitError("Dimensions must be scaled in the same direction")

    return out


# A predefined kernel processor for use in mutateRaster


def KernelProcessor(size, edgeValue=0, outputType=None, passIndex=False):
    """A decorator which automates the production of kernel processors for use
    in mutateRaster (although it could really used for processing any matrix).

    Parameters
    ----------
    size : int
        The number of pixels to expand around a center pixel
        * A 'size' of 0 would make a processing matrix with size 1x1. As in,
          just the value at each point. This would be silly to call...
        * A 'size' of 1 would make a processing matrix of size 3x3. As in, one
          pixel around the center pixel in all directions
        * Processed matrix size is equal to 2*size+1

    edgeValue : numeric; optional
        The value to apply to the edges of the matrix before applying the kernel
        * Will be factored into the kernelling when processing near the edges

    outputType : np.dtype; optional
        The datatype of the processed values
        * Only useful if the output type of the kerneling step is different from
          the matrix input type

    passIndex : bool
        Whether or not to pass the x and y index to the processing function
        * If True, the decorated function must accept an input called 'xi' and
          'yi' in addition to the matrix
        * The xi and yi correspond to the index of the center pixel in the
          original matrix

    Returns
    -------
    function

    Example:
    --------
    * Say we want to make a processor which calculates the average of pixels
      which are within a distance of 2 indices. In other words, we want the
      average of a 5x5 matrix centered around each pixel.
    * Assume that we can use the value -9999 as a no data value

    >>>  @KernelProcessor(2, edgeValue=-9999)
    >>>  def getMean( mat ):
    >>>      # Get only good values
    >>>      goodValues = mat[mat!=-9999]
    >>>
    >>>      # Return the mean
    >>>      return goodValues.mean()
    """
    pass

    def wrapper1(kernel):
        pass

        def wrapper2(matrix):
            # get the original matrix sizes
            yN, xN = matrix.shape

            # make a padded version of the matrix

            # paddedMatrix = (
            #     np.ones((yN + 2 * size, xN + 2 * size),) * edgeValue
            # )
            paddedMatrix = np.full(shape=(yN + 2 * size, xN + 2 * size), fill_value=edgeValue)
            paddedMatrix[size:-size, size:-size] = matrix

            # apply kernel to each pixel
            output = np.zeros((yN, xN), dtype=matrix.dtype if outputType is None else outputType)
            for yi in range(yN):
                for xi in range(xN):
                    slicedMatrix = paddedMatrix[yi : 2 * size + yi + 1, xi : 2 * size + xi + 1]

                    if passIndex:
                        output[yi, xi] = kernel(slicedMatrix, xi=xi, yi=yi)
                    else:
                        output[yi, xi] = kernel(slicedMatrix)

            # done!
            return output

        return wrapper2

    return wrapper1


#############################################################
# internal use source generators


def quickVector(geom, output=None):
    """GeoKit internal for quickly creating a vector datasource."""
    # Create a quick vector source

    if isinstance(geom, ogr.Geometry):
        geom = [
            geom,
        ]
    elif isinstance(geom, list):
        pass
    else:  # maybe geom is iterable
        geom = list(geom)

    # Arrange output
    if output:
        driver = gdal.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Create(output, 0, 0)
    else:
        driver = gdal.GetDriverByName("Memory")
        dataSource = driver.Create("", 0, 0, 0, gdal.GDT_Unknown)

    # Create the layer and write feature
    layer = dataSource.CreateLayer("", geom[0].GetSpatialReference(), geom[0].GetGeometryType())

    for g in geom:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(g)

        # Create the feature
        layer.CreateFeature(feature)
        feature.Destroy()

    # Done!
    if output:
        return output
    else:
        return dataSource


def fitBoundsTo(
    bounds,
    dx,
    dy,
    expand: bool = False,
    startAtZero: bool = False,
    enforce: bool = False,
) -> tuple:
    try:
        xMin, yMin, xMax, yMax = bounds
    except TypeError:
        xMin, yMin, xMax, yMax = bounds.xyXY

    if enforce or (bounds[2] - bounds[0]) % dx != 0 or (startAtZero and (bounds[2] % dx != 0 or bounds[0] % dx != 0)):
        # if expand is demanded or necessary because width would be 0
        if expand or np.round(bounds[0] / dx) * dx == np.round(bounds[2] / dx) * dx:
            # expand such that the original bounds are always fully contained
            xMin = np.floor(bounds[0] / dx) * dx
            xMax = np.ceil(bounds[2] / dx) * dx
        else:
            # round to the nearest cell resolution value
            xMin = np.round(bounds[0] / dx) * dx
            xMax = np.round(bounds[2] / dx) * dx
    if enforce or (bounds[3] - bounds[1]) % dy != 0 or (startAtZero and (bounds[3] % dy != 0 or bounds[1] % dy != 0)):
        # if expand is demanded or necessary because height would be 0
        if expand or np.round(bounds[1] / dy) * dy == np.round(bounds[3] / dy) * dy:
            # expand such that the original bounds are always fully contained
            yMin = np.floor(bounds[1] / dy) * dy
            yMax = np.ceil(bounds[3] / dy) * dy
        else:
            # round to the nearest cell resolution value
            yMin = np.round(bounds[1] / dy) * dy
            yMax = np.round(bounds[3] / dy) * dy

    return xMin, yMin, xMax, yMax


def canonicalGrid(
    bounds: tuple[numeric, numeric, numeric, numeric],
    dx: numeric,
    dy: numeric,
    tol: float = 1e-6,
) -> tuple[tuple[float, float, float, float], int, int]:
    """Return the canonical snapped bounds and exact integer (cols, rows) for a pixel grid.

    This is the single source of truth for translating a (bounds, resolution) request into a
    concrete raster grid, so that the in-memory and on-disk warp paths (and quickRaster) always
    agree.

    Parameters
    ----------
    bounds : tuple[numeric, numeric, numeric, numeric]
        The (xMin, yMin, xMax, yMax) request, snapped to the pixel grid via fitBoundsTo.
    dx : numeric
        The pixel width in x direction.
    dy : numeric
        The pixel height in y direction (sign is ignored).
    tol : float, optional
        Relative tolerance for accepting a span as an integer pixel count, by default 1e-6.

    Returns
    -------
    tuple
        ((xMin, yMin, xMax, yMax), cols, rows) with the bounds snapped to the grid and integer
        pixel counts.
    """
    xMin, yMin, xMax, yMax = fitBoundsTo(bounds, dx, dy)

    def _count(span: numeric, res: numeric) -> int:
        n = span / abs(res)
        nearest = round(n)
        if abs(n - nearest) > tol * max(1.0, nearest):
            raise GeoKitError(
                "bounds span %s is not an integer multiple of resolution %s (got %s pixels)" % (span, res, n)
            )
        return int(nearest)

    return (xMin, yMin, xMax, yMax), _count(xMax - xMin, dx), _count(yMax - yMin, dy)


def _check_fill_return_code(fill_return_code):
    # if =CE_None
    pass


def quickRaster(
    bounds: tuple[numeric, numeric, numeric, numeric],
    srs: osr.SpatialReference | None,
    dx: numeric,
    dy: numeric,
    dtype: geokit_c_data_types_literal | None = None,
    noData: None | numeric | bool = None,
    data: np.ndarray | None = None,
    scale: numeric | None = None,
    offset: numeric | None = None,
) -> gdal.Dataset:
    """
    GeoKit internal for quickly creating a raster datasource.

    Parameters
    ----------
    bounds : tuple[numeric, numeric, numeric, numeric]
        The bounds to create the raster for
    srs : osr.SpatialReference | None
        The SRS to assign to the raster. Please be aware that some operations may not work correctly if no SRS is given.
    dx : numeric
        The pixel width in x direction.
    dy : numeric
        The pixel height in y direction.
    dtype : geokit_c_data_types_literal | None, optional
        The GDAL C datatype to use for the raster band. by default None
    noData : None | numeric | bool, optional
        The value to use for no data in the raster band, by default None
    data : np.ndarray | None, optional
        The data array to write to the raster band, by default None
    scale : numeric | None, optional
        The scale factor to apply to the raster band, by default None
    offset : numeric | None, optional
        The offset to apply to the raster band, by default None

    Returns
    -------
    A raster dataset as gdal.Dataset

    """
    # bounds = fitBoundsTo(bounds, dx, dy)

    # Make a raster dataset and pull the band/maskBand objects
    originX = bounds[0]
    originY = bounds[3]  # Always use the "Y-at-Top" orientation

    cols = int(round((bounds[2] - originX) / dx))
    rows = int(round((originY - bounds[1]) / abs(dy)))

    # Open the driver
    driver: gdal.Driver = gdal.GetDriverByName("Mem")  # create a raster in memory
    # dtype = getattr(gdal, dtype) if isinstance(dtype, str) else dtype
    list_of_scalars = []
    if noData is None:
        pass
    else:
        list_of_scalars.append(noData)

    list_of_datatype_strings = []
    if dtype is None:
        list_of_gdal_data_type_strings = []
    else:
        list_of_gdal_data_type_strings = [dtype]
    dtype_constant = MinimumCDataTypeHandler.get_valid_gdal_data_type_as_constant(
        list_of_numbers=list_of_scalars,
        minimum_gdal_type_list=list_of_gdal_data_type_strings,
        # user_defined_minimum_gdal_type=dtype,
    )
    list_of_datatype_strings.append(dtype)
    raster: gdal.Dataset = driver.Create("", cols, rows, 1, dtype_constant)

    if raster is None:
        raise GeoKitError("Failed to create temporary raster")

    raster.SetGeoTransform((originX, abs(dx), 0, originY, 0, -1 * abs(dy)))

    # Set the SRS
    if srs is not None:
        raster.SetProjection(srs.ExportToWkt())
    else:
        warnings.warn(
            message="No srs given when creating raster. Please be aware that some operations may not work correctly.",
            category=UserWarning,
        )

    # get the band
    band: gdal.Band = raster.GetRasterBand(1)

    # set optionals
    if not noData is None:
        band.SetNoDataValue(noData)

        if data is None:
            no_data_fill_return_code = band.Fill(noData)
            _check_fill_return_code(fill_return_code=no_data_fill_return_code)

    if not scale is None:
        band.SetScale(scale)

    if not offset is None:
        band.SetOffset(offset)

    # add data
    if not data is None:
        band.WriteArray(data)
        band.FlushCache()

    # Done!
    del band
    raster.FlushCache()
    return raster


# Helpful classes
Feature = namedtuple("Feature", "geom attr")

# Image plotter


def drawImage(
    matrix: np.ndarray,
    ax: matplotlib.axis.Axis | None = None,
    xlim: tuple[float | float] | None = None,
    ylim: tuple[float | float] | None = None,
    yAtTop: bool = True,
    scaling: float = 1,
    fontsize: int = 16,
    hideAxis=False,
    figsize: tuple[float, float] = (12, 12),
    cbar: bool = True,
    cbarPadding=0.01,
    cbarTitle=None,
    vmin=None,
    vmax=None,
    cmap="viridis",
    cbax=None,
    cbargs=None,
    leftMargin=0,
    rightMargin=0,
    topMargin=0,
    bottomMargin=0,
    **kwargs,
) -> AxHands:
    """Draw a matrix as an image on a matplotlib canvas.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix data to draw

    ax : matplotlib.axis.Axis; optional
        The axis to draw the geometries on
          * If not given, a new axis is generated and returned

    xlim : tuple[float, float]; optional
        The x-axis limits to draw the matrix on

    ylim : tuple[float, float]; optional
        The y-axis limits to draw the matrix on

    yAtTop : bool; optional
        If True, the first row of data should be plotted at the top of the image

    scaling : numeric; optional
        An integer factor by which to scale the matrix before plotting

    figsize : tuple[float, float]; optional
        The figure size to create when generating a new axis
          * If resulting figure looks weird, altering the figure size is your best
            bet to make it look nicer

    fontsize : int; optional
        A base font size to apply to tick marks which appear
          * Titles and labels are given a size of 'fontsize' + 2

    cbar : bool; optional
        If True, a color bar will be drawn

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

    vmax : float; optional
        The maximum value to color

    cmap : str or matplotlib ColorMap; optional
        The colormap to use when coloring

    cbax : matplotlib axis; optional
        An explicitly given axis to use for drawing the colorbar

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

    Returns
    -------
    A namedtuple containing:
       'ax' -> The map axis
       'handles' -> All geometry handles which were created in the order they were
                    drawn
       'cbar' -> The colorbar handle if it was drawn
    """
    # Create an axis, if needed
    if isinstance(ax, AxHands):
        ax = ax.ax
    if ax is None:
        newAxis = True

        plt.figure(figsize=figsize)

        if not cbar:  # We don't need a colorbar
            if not hideAxis:
                leftMargin += 0.07

            ax = plt.axes(
                [
                    leftMargin,
                    bottomMargin,
                    1 - (rightMargin + leftMargin),
                    1 - (topMargin + bottomMargin),
                ]
            )
            cbax = None

        else:  # We need a colorbar
            rightMargin += 0.08  # Add area on the right for colorbar text
            if not hideAxis:
                leftMargin += 0.07

            cbarExtraPad = 0.05
            cbarWidth = 0.04

            ax = plt.axes(
                [
                    leftMargin,
                    bottomMargin,
                    1 - (rightMargin + leftMargin + cbarWidth + cbarPadding),
                    1 - (topMargin + bottomMargin),
                ]
            )

            cbax = plt.axes(
                [
                    1 - (rightMargin + cbarWidth),
                    bottomMargin + cbarExtraPad,
                    cbarWidth,
                    1 - (topMargin + bottomMargin + 2 * cbarExtraPad),
                ]
            )

        if hideAxis:
            ax.axis("off")
        else:
            ax.tick_params(labelsize=fontsize)
    else:
        newAxis = False

    # handle flipped matrix
    if not yAtTop:
        matrix = matrix[::-1, :]

    # Draw image
    if scaling:
        matrix = scaleMatrix(matrix, scaling, strict=False)

    if not (xlim is None and ylim is None):
        extent = xlim[0], xlim[1], ylim[0], ylim[1]
    else:
        extent = None

    h = ax.imshow(
        matrix,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        **kwargs,
    )

    # Draw Colorbar
    if cbar:
        tmp = dict(cmap=cmap, orientation="vertical")
        if not cbargs is None:
            tmp.update(cbargs)

        if cbax is None:
            cbar = plt.colorbar(h, ax=ax, **tmp)
        else:
            cbar = plt.colorbar(h, cax=cbax)

        cbar.ax.tick_params(labelsize=fontsize)
        if not cbarTitle is None:
            cbar.set_label(cbarTitle, fontsize=fontsize + 2)

    # Do some formatting
    if newAxis:
        ax.set_aspect("equal")
        ax.autoscale(enable=True)

    # Done!
    return AxHands(ax, h, cbar)


def compare_geoms(geoms_1, geoms_2):
    """Compare two lists of geometries and return a list of booleans indicating whether each pair of geometries are equal.
    The order of the lists is important, as the first geometry in the first list will be compared to the first geometry in the second list, and so on.

    Returns
    -------
    list
        A list of booleans indicating whether each pair of geometries are equal.
    """
    equal = map(lambda g1, g2: g1.Equals(g2), geoms_1, geoms_2)

    return list(equal)


def nodata_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        # handle floats/NaN
        return (a == b) or (np.isnan(a) and np.isnan(b))
    except TypeError:
        return a == b
