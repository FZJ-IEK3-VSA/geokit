import datetime
import os
import statistics
from glob import glob
from warnings import warn

import numpy as np
from osgeo import gdal

from geokit.core.raster import RasterInfo
from geokit.core.regionmask import *
from geokit.core.util import GeoKitError, get_common_dtype, nodata_equal
from geokit.raster import (
    createRaster,
    extractMatrix,
    loadRaster,
    rasterInfo,
)


def checkSimilarRasters(
    datasets,
    rtol=0,
):
    """
    Parameters
    ----------
    datasets : string or list
        glob string path describing datasets to combine, alternatively list of
        gdal.Datasets or iterable object with paths.
    rtol : int, float, optional
        The relative tolerance that is allowed for numeric deviations. By
        default 0, i.e. an exact match (within data type accuracy) is required.

    Returns
    -------
    output datasets: list
        List of osgeo.gdal.Datasets with similar contexts.
    """
    assert isinstance(rtol, (int, float)) and rtol >= 0, f"rtol must be a float or int >= 0"
    # ensure we have a list of raster datasets
    if isinstance(datasets, str):
        datasets = glob(datasets)
        if len(datasets) == 0:
            raise FileNotFoundError(f"datasets given as a string but does not lead to any existing files: '{datasets}'")
        datasets.sort()
    if not isinstance(datasets, list):
        raise TypeError(f"datasets must be a list")

    # check and load all datasets
    _datasets = []
    for dataset in datasets:
        if isinstance(dataset, str):
            if not os.path.isfile(dataset):
                raise FileNotFoundError(f"datasets string entry is not an existing file: '{dataset}'")
            _datasets.append(loadRaster(dataset))
        elif isinstance(dataset, gdal.Dataset):
            _datasets.append(dataset)
        else:
            raise TypeError(f"datasets must contain only string or osgeo.gdal.Dataset entries.")
    datasets = _datasets

    # get all raster infos
    infoDataset = [rasterInfo(d) for d in datasets]
    # check all relevant variables
    for rInfo in infoDataset[1:]:
        # same srs is required
        if not infoDataset[0].srs.IsSame(rInfo.srs):
            raise GeoKitError(f"SRS mismatch between datasets.")
        # pixel width and height must be same/similar
        if not np.isclose(infoDataset[0].dx, rInfo.dx, rtol=rtol, atol=0):
            raise GeoKitError(f"dx mismatch between datasets.")
        if not np.isclose(infoDataset[0].dy, rInfo.dy, rtol=rtol, atol=0):
            raise GeoKitError(f"dy mismatch between datasets.")
        # bounds shift must be an exact/close match to a multiple of dx/dy
        diffx = infoDataset[0].bounds[0] - rInfo.bounds[0]
        if not (
            round(diffx / rInfo.dx, 0) == 0
            or np.isclose(diffx / round(diffx / rInfo.dx, 0), rInfo.dx, rtol=rtol, atol=0)
        ):
            raise GeoKitError(f"horizontal bounds shift between datasets is not a multiple of dx.")
        diffy = infoDataset[0].bounds[1] - rInfo.bounds[1]
        if not (
            round(diffy / rInfo.dy, 0) == 0
            or np.isclose(diffy / round(diffy / rInfo.dy, 0), rInfo.dy, rtol=rtol, atol=0)
        ):
            raise GeoKitError(f"vertical bounds shift between datasets is not a multiple of dy.")
        # # noData should be the same
        # if not infoDataset[0].noData == rInfo.noData:
        #     raise GeoKitError(f"noData mismatch between datasets.")

        # noData equality
        if not nodata_equal(infoDataset[0].noData, rInfo.noData):
            raise GeoKitError("noData mismatch between datasets.")

    # make sure the datatypes are the same or can be combined
    dtypes = [rInfo.dtype for rInfo in infoDataset]
    if rtol == 0 and not len(set(dtypes)):
        # no tolerance allowed - assume dtypes must also match exactly
        raise TypeError(f"dtypes or rasters differ but rtol is zero.")
    elif rtol > 0:
        # accept different dtypes as long as they can be combined into one
        get_common_dtype(dtypes=dtypes, fallback=None)  # fail if no common dtype
    # return list of preloaded, similar datasets
    return datasets


def combineSimilarRasters(
    datasets,
    output=None,
    combiningFunc=None,
    verbose=True,
    updateMeta=False,
    allowNumericMismatch=False,
    **kwargs,
):
    """
    Combines several similar raster files into one single raster file.

    Parameters
    ----------
    datasets : string or list
        glob string path describing datasets to combine, alternatively list of
        gdal.Datasets or iterable object with paths.
    output : string, optional
        Filepath to output raster file. If it is an existing file, datasets will
        be added to output. Recommended to create a new file every time though.
        If None, no output dataset will be loaded or created on disk and output
        dataset kept in memory only, by default None
    combiningFunc : [type], optional
        Allows customized functions to combine matrices, by default None
    verbose : bool, optional
        If True, additional status print stamenets will be issued, by default
        True
    updateMeta : bool, optional
        If True, metadata of output dataset will be a combination of all input
        rasters, by default False.
        NOTE: In the case of multiple values for the metadata keys, the last
        dataset metadata will take precedence.
    allowNumericMismatch : bool, optional
        If True, minor deviations in raster context will be ignored/corrected.
        By default False, i.e. only exactly similar rasters will be combined.
    **kwargs
        Will be passed on to geokit.raster.createRaster().

    Returns
    -------
    output dataset: osgeo.gdal.Dataset
        Raster file containing the combined matrices of all input datasets.
    """
    if not isinstance(allowNumericMismatch, bool):
        raise TypeError(f"allowNumericMismatch must be boolean.")

    # CHECK AND PREPROCESS INPUT DATASETS

    # Ensure we have a list of raster datasets
    if isinstance(datasets, str):
        datasets = glob(datasets)
        datasets.sort()
    elif isinstance(datasets, gdal.Dataset):
        datasets = [
            datasets,
        ]
    else:  # assume datasets is iterable
        datasets = list(datasets)

    # make sure we do actually have rasters and they are all indeed "similar"
    if len(datasets) == 0:
        raise GeoKitError("No datasets given")
    rtol = 0
    if allowNumericMismatch:
        rtol = 0.0001  # allow a numeric deviation of 0.01%
    datasets = checkSimilarRasters(datasets=datasets, rtol=rtol)

    # GET REFERENCE CONTEXT FOR THE OUTPUT RASTER

    # determine info for all datasets
    infoSet = [rasterInfo(d) for d in datasets]

    # get reference srs - are all the same thanks to checkSimilarRasters
    srs_ref = infoSet[0].srs

    # get the unique actual dtypes in input rasters
    dtypes = sorted(set([_i.dtype for _i in infoSet]))
    # now get the most lightweight commonly usable dtype
    dtype_ref = get_common_dtype(dtypes=dtypes, fallback=None)

    # get the reference resolution in x and y dir as the most commonly used value
    dx_ref = statistics.mode([_i.pixelWidth for _i in infoSet])
    dy_ref = statistics.mode([_i.pixelHeight for _i in infoSet])

    # try to align all bounds to the first matching raster which has correct x_ref and y_ref resolution
    i_match = next(
        (
            i
            for i, (x, y) in enumerate(
                zip(
                    [_i.pixelWidth for _i in infoSet],
                    [_i.pixelHeight for _i in infoSet],
                )
            )
            if x == dx_ref and y == dy_ref
        ),
        None,
    )
    if i_match is not None:
        # we have a "perfect" raster, use the min. bounds of that one as reference for all other rasters
        boundsXmin_ref = infoSet[i_match].bounds[0]
        boundsYmin_ref = infoSet[i_match].bounds[2]
    else:
        # we do not have any raster which matches both r_ref any y_ref in its resolution. Simply use the first raster.
        boundsXmin_ref = infoSet[0].bounds[0]
        boundsYmin_ref = infoSet[0].bounds[2]
        if verbose:
            print(
                datetime.datetime.now(),
                f"NOTE: None of the rasters matches both reference resolutions in x and y direction. Use first raster bounds as reference.",
                flush=True,
            )

    # calculate the possibly adapted bounds for all datasets
    boundsSet = []
    for _info in infoSet:
        # calculate the new bounds by aligning bottom left corner with boundsXmin_ref/boundsYmin_ref + multiple of cell size
        _bounds_Xmin = boundsXmin_ref + round((_info.bounds[0] - boundsXmin_ref) / dx_ref) * dx_ref
        _bounds_Ymin = boundsYmin_ref + round((_info.bounds[1] - boundsYmin_ref) / dy_ref) * dy_ref
        _bounds = (
            _bounds_Xmin,
            _bounds_Ymin,
            _bounds_Xmin + _info.xWinSize * dx_ref,
            _bounds_Ymin + _info.yWinSize * dy_ref,
        )
        # make sure the number of cells would remain the same in the new bounds
        assert round(_info.xWinSize * dx_ref / _info.dx, 0) == _info.xWinSize, (
            f"The change in bounds width would lead to a different amount of cell columns in the original raster."
        )
        assert round(_info.yWinSize * dy_ref / _info.dy, 0) == _info.yWinSize, (
            f"The change in bounds height would lead to a different amount of cell rows in the original raster."
        )
        boundsSet.append(_bounds)

    # get summary info about the whole dataset group
    dataXMin = min([i[0] for i in boundsSet])
    dataXMax = max([i[2] for i in boundsSet])
    dataYMin = min([i[1] for i in boundsSet])
    dataYMax = max([i[3] for i in boundsSet])

    # get noData value from kwargs, else take from dataset infos
    noData_ref = kwargs.pop("noData", None)
    if noData_ref is None:
        noDataSet = set([i.noData for i in infoSet])
        assert len(noDataSet) == 1  # make sure, is enforced by checkSimilarRasters
        noData_ref = noDataSet.pop()

    # Maybe create a new output dataset
    if isinstance(output, str):
        if not os.path.isfile(output):
            # we will need to create a output source
            createRaster(
                bounds=(dataXMin, dataYMin, dataXMax, dataYMax),
                output=output,
                dtype=dtype_ref,
                pixelWidth=dx_ref,
                pixelHeight=dy_ref,
                noData=noData_ref,
                srs=srs_ref,
                fill=noData_ref,
                **kwargs,
            )
        else:
            warn(
                "WARNING: Overwriting existing output file. Sometimes writing to an non empty output fails. Recommended to write to a non existing location instead and include maser into datasets."
            )
    elif output is None:
        # create raster in memory
        outputDS = createRaster(
            bounds=(dataXMin, dataYMin, dataXMax, dataYMax),
            dtype=dtype_ref,
            pixelWidth=dx_ref,
            pixelHeight=dy_ref,
            noData=noData_ref,
            srs=srs_ref,
            fill=noData_ref,
            **kwargs,
        )
    else:
        raise TypeError(
            "output must be None or a str formatted file path to an existing output file or a file to be created."
        )

    # open output dataset if required and check parameters
    if not output is None:
        outputDS = gdal.Open(output, gdal.GA_Update)
    mInfo = rasterInfo(outputDS)
    mExtent = Extent(mInfo.bounds, srs=mInfo.srs)

    outputBand = outputDS.GetRasterBand(1)

    # Make a meta container
    if updateMeta:
        meta = outputDS.GetMetadata_Dict()

    # Add each dataset to output
    for i in range(len(datasets)):
        if verbose:
            print(
                datetime.datetime.now(),
                f"Now adding raster No. {i + 1}/{len(datasets)}",
            )
        # create dataset extent
        dExtent = Extent(boundsSet[i], srs=srs_ref)

        # extract the dataset's matrix
        dMatrix = extractMatrix(datasets[i])
        if not infoSet[i].yAtTop:
            dMatrix = dMatrix[::-1, :]

        # Calculate starting indices
        idx = mExtent.findWithin(dExtent, (mInfo.dx, mInfo.dy), yAtTop=mInfo.yAtTop)

        # Get output data
        mMatrix = outputBand.ReadAsArray(xoff=idx.xStart, yoff=idx.yStart, win_xsize=idx.xWin, win_ysize=idx.yWin)
        if mMatrix is None:
            raise GeoKitError("mMatrix is None")

        # create selector
        if not combiningFunc is None:
            # update rasterInfo since we might have slightly changed cell/bounds info
            rInfo_dict = infoSet[i]._asdict()
            rInfo_dict["bounds"] = boundsSet[i]
            rInfo_dict["pixelWidth"] = dx_ref
            rInfo_dict["pixelHeight"] = dy_ref
            rInfo_dict["dx"] = dx_ref
            rInfo_dict["dy"] = dy_ref
            rInfo_dict["xMin"] = boundsSet[i][0]
            rInfo_dict["yMin"] = boundsSet[i][1]
            rInfo_dict["xMax"] = boundsSet[i][2]
            rInfo_dict["yMax"] = boundsSet[i][3]
            rInfo_upd = RasterInfo(**rInfo_dict)
            writeMatrix = combiningFunc(mMatrix=mMatrix, mInfo=mInfo, dMatrix=dMatrix, dInfo=rInfo_upd)
        elif not infoSet[i].noData is None:
            sel = dMatrix != infoSet[i].noData
            mMatrix[sel] = dMatrix[sel]
            writeMatrix = mMatrix
        else:
            writeMatrix = dMatrix

        # Add to output
        outputBand.WriteArray(writeMatrix, idx.xStart, idx.yStart)
        outputBand.FlushCache()

        # update metaData, maybe
        if updateMeta:
            meta.update(infoSet[i].meta)

    if updateMeta:
        outputDS.SetMetadata(meta)

    # Write final raster
    outputDS.FlushCache()
    outputBand.ComputeRasterMinMax(0)
    outputBand.ComputeBandStats(0)

    return outputDS
