import os
import sys
from glob import glob
from json import dumps
from os.path import basename
from warnings import warn
import datetime

from osgeo import gdal

from geokit.core.regionmask import *
from geokit.core.util import GeoKitError, get_common_dtype
from geokit.raster import createRaster, extractMatrix, rasterInfo, warp


def combineSimilarRasters(
    datasets,
    output=None,
    combiningFunc=None,
    verbose=True,
    updateMeta=False,
    allowPreWarp=True,
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
        be added to output. Recommended to create a new file everytime though.
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
    allowPreWarp : bool, optional
        If True, minor deviations in raster context will be aligned by a
        preprocessing warping step.
    **kwargs 
        Will be passed on to geokit.raster.createRaster().
    Returns:
    ----------
    output dataset: osgeo.gdal.Dataset
        Raster file containing the combined matrices of all input datasets.
    """
    if not isinstance(allowPreWarp, bool):
        raise TypeError(f"allowPreWarp must be boolean.")

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

    if len(datasets) == 0:
        raise GeoKitError("No datasets given")

    # determine info for all datasets
    infoSet = [rasterInfo(d) for d in datasets]

    # Ensure all input rasters share resolution, srs, datatype, and noData
    if not all([info.srs.IsSame(infoSet[0].srs) for info in infoSet]):
        # all srs must always be the same irrespective of warping or not
        raise GeoKitError(f"SRS does not match in all datasets.")
    else:
        # define the reference srs
        srs_ref = infoSet[0].srs

    try:
        # first try if the ds are in the same context already
        for info in infoSet[1:]:
            if not (info.dx == infoSet[0].dx and info.dy == infoSet[0].dy):
                raise GeoKitError(
                    f"Resolution does not match in datasets. x/y: {info.dx} vs. {infoSet[0].dx} / {info.dy} vs. {infoSet[0].dy}"
                )
            if not (info.dtype == infoSet[0].dtype):
                raise GeoKitError(
                    f"Datatype does not match in datasets: {info.dtype} vs. {infoSet[0].dtype}"
                )
            if not (
                abs((infoSet[0].bounds[0] - info.bounds[0]) % infoSet[0].dx) == 0
                and abs((infoSet[0].bounds[1] - info.bounds[1]) % infoSet[0].dy) == 0
            ):
                raise GeoKitError(
                    f"Boundaries of the different rasters are not aligned with cell resolution."
                )

    except Exception as e:
        # we have a mismatch of at least one relevant context parameter between the rasters in the list
        if not allowPreWarp:
            # pre-warp is not allowed, must fail
            raise e
        if verbose:
            print(
                datetime.datetime.now(),
                f"Resolution, SRS or datatype are not unique in datasets. First warping all datasets to identical context: {e}",
                flush=True,
            )

        # get the unique actual dtypes in input rasters
        dtypes = sorted(set([_i.dtype for _i in infoSet]))
        # now get the most lightweight commonly useable dtype
        # set automated fallback to None - user shall preprocess in such cases
        dtype_ref = get_common_dtype(dtypes=dtypes, fallback=None)

        # get the reference resolution in x and y dir as the most commonly used value
        x_ress = [_i.pixelWidth for _i in infoSet]
        x_ref = max(set(x_ress), key=x_ress.count)
        y_ress = [_i.pixelHeight for _i in infoSet]
        y_ref = max(set(y_ress), key=y_ress.count)

        # get the relative boundaries to align all raster bounds with, so that cells do not partially overlap
        areaOfUse = srs_ref.GetAreaOfUse()
        if areaOfUse is not None:
            # align the rasters to the minimum boundaries of the SRS
            bounds_refXmin = areaOfUse.west_lon_degree
            bounds_refYmin = areaOfUse.south_lon_degree
        else:
            # bounds of SRS unknown, so align all bounds to the first matching raster which has correct x_ref and y_ref resolution
            i_match = next(
                (
                    i
                    for i, (x, y) in enumerate(zip(x_ress, y_ress))
                    if x == x_ref and y == y_ref
                ),
                None,
            )
            if i_match is not None:
                # we have a "perfect" raster, use the min. bounds of that one as reference for all other rasters
                bounds_refXmin = infoSet[i_match].bounds[0]
                bounds_refYmin = infoSet[i_match].bounds[2]
                print(
                    datetime.datetime.now(),
                    f"NOTE: SRS validity bounds could not be extracted from SRS, so the bounds of the first raster which matches both reference resolutions in x and y direction are used as reference (raster #{i_match}).",
                    flush=True,
                )
            else:
                # we do not have any raster which matches both r_ref any y_ref in its resolution. Simply use the first raster.
                bounds_refXmin = infoSet[0].bounds[0]
                bounds_refYmin = infoSet[0].bounds[2]
                print(
                    datetime.datetime.now(),
                    f"NOTE: SRS validity bounds could not be extracted from SRS nor does one raster match both reference resolutions in x and y direction. Use first raster bounds as reference.",
                    flush=True,
                )

        # now (marginally) warp all rasters to the same context first
        _datasets = []
        for i, (_ds, _info) in enumerate(zip(datasets, infoSet)):
            if verbose:
                if isinstance(datasets[i], str):
                    print(
                        datetime.datetime.now(),
                        f"Now pre-warping raster No. {i+1}/{len(datasets)} ({basename(datasets[i])})",
                    )
                else:
                    print(
                        datetime.datetime.now(),
                        f"Now pre-warping raster No. {i+1}/{len(datasets)}",
                    )
            # get the res and make sure it is close enough to the reference for all rasters
            assert np.isclose(
                [_info.pixelWidth, _info.pixelHeight], [x_ref, y_ref]
            ).all()
            # calculate the new bounds, the rule - maintain bottom left bounds and align with bounds_ref
            _bounds_Xmin = (
                bounds_refXmin
                + round((_info.bounds[0] - bounds_refXmin) / x_ref) * x_ref
            )
            _bounds_Ymin = (
                bounds_refYmin
                + round((_info.bounds[1] - bounds_refYmin) / y_ref) * y_ref
            )
            _bounds = (
                _bounds_Xmin,
                _bounds_Ymin,
                _bounds_Xmin + _info.xWinSize * x_ref,
                _bounds_Ymin + _info.yWinSize * y_ref,
            )
            # warp the data to the new context
                _bounds_Xmin+_info.xWinSize*x_ref, 
                _bounds_Ymin+_info.yWinSize*y_ref,
                )
            _dswarped = warp(
                source=_ds,
                resampleAlg="near",
                pixelHeight=y_ref,
                pixelWidth=x_ref,
                srs=srs_ref,
                bounds=_bounds,
                dtype=dtype_ref,
                noData=_info.noData,
                fill=_info.noData,
            )
            _datasets.append(_dswarped)
        # overwrite datasets and calculate a new infoset with updated rasterInfo
        datasets = _datasets
        infoSet = [rasterInfo(d) for d in datasets]

    # Get summary info about the whole dataset group
    dataXMin = min([i.xMin for i in infoSet])
    dataXMax = max([i.xMax for i in infoSet])
    dataYMin = min([i.yMin for i in infoSet])
    dataYMax = max([i.yMax for i in infoSet])

    # Maybe create a new output dataset
    if isinstance(output, str):
        if not os.path.isfile(output):  # we will need to create a output source
            # Determine no data value
            noDataValue = kwargs.pop("noData", None)

            if noDataValue is None:
                noDataSet = set([i.noData for i in infoSet])
                if len(noDataSet) == 1:
                    noDataValue = noDataSet.pop()

            # Create Raster
            dx = infoSet[0].dx
            dy = infoSet[0].dy
            dtype = infoSet[0].dtype

            createRaster(
                bounds=(dataXMin, dataYMin, dataXMax, dataYMax),
                output=output,
                dtype=dtype,
                pixelWidth=dx,
                pixelHeight=dy,
                noData=noDataValue,
                srs=srs_ref,
                fill=noDataValue,
                **kwargs,
            )
        else:
            warn(
                "WARNING: Overwriting existing output file. Sometimes writing to an non empty output fails. Recommended to write to a non existing location instead and include maser into datasets."
            )
    elif output is None:
        # Determine no data value
        noDataValue = kwargs.pop("noData", None)

        if noDataValue is None:
            noDataSet = set([i.noData for i in infoSet])
            if len(noDataSet) == 1:
                noDataValue = noDataSet.pop()

        # Create Raster
        dx = infoSet[0].dx
        dy = infoSet[0].dy
        dtype = infoSet[0].dtype

        outputDS = createRaster(
            bounds=(dataXMin, dataYMin, dataXMax, dataYMax),
            dtype=dtype,
            pixelWidth=dx,
            pixelHeight=dy,
            noData=noDataValue,
            srs=srs_ref,
            fill=noDataValue,
            **kwargs,
        )
    else:
        sys.exist(
            "output must be None or a str formatted file path to an existing output file or a file to be created."
        )

    # Open output dataset if required and check parameters
    if not output is None:
        outputDS = gdal.Open(output, gdal.GA_Update)
    mInfo = rasterInfo(outputDS)
    mExtent = Extent(mInfo.bounds, srs=mInfo.srs)

    if not mInfo.srs.IsSame(srs_ref):
        raise GeoKitError("SRS's do not match output dataset")
    if not (mInfo.dx == infoSet[0].dx and mInfo.dy == infoSet[0].dy):
        raise GeoKitError("Resolution's do not match output dataset")
    if not (mInfo.dtype == infoSet[0].dtype):
        raise GeoKitError("Datatype's do not match output dataset")

    outputBand = outputDS.GetRasterBand(1)

    # Make a meta container
    if updateMeta:
        meta = outputDS.GetMetadata_Dict()

    # Add each dataset to output
    for i in range(len(datasets)):
        if verbose:
            if isinstance(datasets[i], str):
                print(
                    datetime.datetime.now(),
                    f"Now adding raster No. {i+1}/{len(datasets)} ({basename(datasets[i])})",
                )
            else:
                print(
                    datetime.datetime.now(),
                    f"Now adding raster No. {i+1}/{len(datasets)}",
                )
        # create dataset extent
        dExtent = Extent(infoSet[i].bounds, srs=infoSet[i].srs)

        # extract the dataset's matrix
        dMatrix = extractMatrix(datasets[i])
        if not infoSet[i].yAtTop:
            dMatrix = dMatrix[::-1, :]

        # Calculate starting indicies
        idx = mExtent.findWithin(dExtent, (mInfo.dx, mInfo.dy), yAtTop=mInfo.yAtTop)

        # Get output data
        mMatrix = outputBand.ReadAsArray(
            xoff=idx.xStart, yoff=idx.yStart, win_xsize=idx.xWin, win_ysize=idx.yWin
        )
        if mMatrix is None:
            raise GeoKitError("mMatrix is None")

        # create selector
        if not combiningFunc is None:
            writeMatrix = combiningFunc(
                mMatrix=mMatrix, mInfo=mInfo, dMatrix=dMatrix, dInfo=infoSet[i]
            )
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
