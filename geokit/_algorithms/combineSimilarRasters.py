import os
import sys
from glob import glob
from json import dumps
from os.path import basename
from warnings import warn

from osgeo import gdal

from geokit.core.regionmask import *
from geokit.core.util import GeoKitError, get_common_dtype
from geokit.raster import createRaster, extractMatrix, rasterInfo, warp


def combineSimilarRasters(
    datasets, output=None, combiningFunc=None, verbose=True, updateMeta=False, **kwargs
):
    """
    Combines several similar raster files into one single raster file.

    Parameters
    ----------
    datasets : string or list
        glob string path describing datasets to combine, alternatively list of gdal.Datasets or iterable object with paths.
    output : string, optional
        Filepath to output raster file. If it is an existing file, datasets will be added to output. Recommended to create a new file everytime though. If None, no output dataset will be loaded or created on disk and output dataset kept in memory only, by default None
    combiningFunc : [type], optional
        Allows customized functions to combine matrices, by default None
    verbose : bool, optional
        If True, additional status print stamenets will be issued, by default True
    updateMeta : bool, optional
        If True, metadata of output dataset will be a combination of all input rasters, by default False

    Returns:
    ----------
    output dataset: osgeo.gdal.Dataset
        Raster file containing the combined matrices of all input datasets.
    """

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

    # Determine info for all datasets
    infoSet = [rasterInfo(d) for d in datasets]

    # Ensure all input rasters share resolution, srs, datatype, and noData
    try:
        # first try if the ds are in the same context already
        for info in infoSet[1:]:
            if not info.srs.IsSame(infoSet[0].srs):
                raise GeoKitError(f"SRS does not match in datasets: {info.srs} vs. {infoSet[0].srs}")
            if not (info.dx == infoSet[0].dx and info.dy == infoSet[0].dy):
                raise GeoKitError(f"Resolution does not match in datasets. x/y: {info.dx} vs. {infoSet[0].dx} / {info.dy} vs. {infoSet[0].dy}")
            if not (info.dtype == infoSet[0].dtype):
                raise GeoKitError(f"Datatype does not match in datasets: {info.dtype} vs. {infoSet[0].dtype}")
            
    except Exception as e:
        if verbose:
            print(f"Resolution, SRS or datatype are not unique in datasets. First warping all datasets to identical context: {e}", flush=True)
        # get the unique actual dtypes in input rasters
        dtypes = sorted(set([_i.dtype for _i in infoSet]))
        # now get the most lightweight commonly useable dtype
        # set automated fallback to None - user shall preprocess in such cases
        dtype_ref = get_common_dtype(dtypes=dtypes, fallback=None)
        # else (marginally) warp them to the same context first
        _datasets = []
        for i, (_ds, _info) in enumerate(zip(datasets, infoSet)):
            # get the res/srs and make sure it is close enough for all rasters
            _x = _info.pixelWidth
            _y = _info.pixelHeight
            if i==0:
                # define the first resolution and srs as reference
                x_ref = _x
                y_ref = _y
                srs_ref = _info.srs
            else:
                # make sure all other res are at least similar, even if maybe not the same
                assert np.isclose([_x, _y], [x_ref, y_ref]).all()
                assert _info.srs.IsSame(srs_ref) # srs must match
            # calculate the new bounds, the rule - maintain bottom left bounds
            _bounds = ( 
                _info.bounds[0], 
                _info.bounds[0]+_info.xWinSize*x_ref, 
                _info.bounds[2],
                _info.bounds[2]+_info.yWinSize*y_ref,
                )
            _dswarped = warp(
                source = _ds,
                resampleAlg = 'near',
                pixelHeight = y_ref,
                pixelWidth = x_ref,
                srs = srs_ref,
                bounds = _bounds,
                dtype = dtype_ref,
                noData = _info.noData,
                fill = _info.noData,
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
            srs = infoSet[0].srs

            createRaster(
                bounds=(dataXMin, dataYMin, dataXMax, dataYMax),
                output=output,
                dtype=dtype,
                pixelWidth=dx,
                pixelHeight=dy,
                noData=noDataValue,
                srs=srs,
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
        srs = infoSet[0].srs

        outputDS = createRaster(
            bounds=(dataXMin, dataYMin, dataXMax, dataYMax),
            dtype=dtype,
            pixelWidth=dx,
            pixelHeight=dy,
            noData=noDataValue,
            srs=srs,
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

    if not mInfo.srs.IsSame(infoSet[0].srs):
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
                print(f"{i+1}/{len(datasets)} ({basename(datasets[i])})")
            else:
                print(f"{i+1}/{len(datasets)}")
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
