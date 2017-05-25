from geokit._core.regionmask import *


def combineSimilarRasters(master, datasets, flipY=False, **kwargs):
    """!!! WRITE ME !!!"""
    
    # Create empty raster containers
    bands = []
    extents = []
    nodataVals = []

    # Ensure we have a list of raster datasets
    if( isinstance(datasets, str) or isinstance(datasets, gdal.Dataset)):
        datasets = [datasets,]
    else:
        datasets = list(datasets)
    
    # Open all datasets
    for i in range(len(datasets)):
        datasets[i] = loadRaster(datasets[i])
    
    # Open master dataset
    if(os.path.isfile(master)):
        masterDS = gdal.Open(master, gdal.GA_Update)

        # Ensure each datasets has the same projection and resolution as master
        ras = describeRaster(masterDS)

        dx = ras.pixelWidth
        dy = ras.pixelHeight
        srs = ras.srs
        xMin, yMin, xMax, yMax = ras.bounds
        dataType = ras.dtype

        # Iterate over files
        for fi in range(len(datasets)):
            ras = describeRaster(datasets[fi])

            if( dx!=ras.dx or dy!=ras.dy or not srs.IsSame(ras.srs) or dataType!=ras.dtype):
                msg = "Dataset {0} does not match master configuration".format(fi)
                raise RuntimeError(msg)

            # Append containers
            extents.append(ras.bounds)
            bands.append(datasets[fi].GetRasterBand(1))
            nodataVals.append( bands[fi].GetNoDataValue() )

    else: # We will need to make a master dataset
        # Gather collective stats
        first=True
        for ds in datasets:
            ras = describeRaster(ds)

            # Ensure each datasets has the same projection and resolution
            if(first):
                first=False
                dx = ras.pixelWidth
                dy = ras.pixelHeight
                srs = ras.srs
                xMin, yMin, xMax, yMax = ras.bounds
                dataType = ras.dtype
            else:
                if( dx!=ras.dx or dy!=ras.dy or not srs.IsSame(ras.srs) or dataType!=ras.dtype):
                    raise RuntimeError("Dataset parameters do not match")

            # Update master extent
            xMin = min((xMin,ras.xMin))
            yMin = min((yMin,ras.yMin))
            xMax = max((xMax,ras.xMax))
            yMax = max((yMax,ras.yMax))

            # Append containers
            extents.append(ras.bounds)
            bands.append(ds.GetRasterBand(1))
            nodataVals.append( bands[-1].GetNoDataValue() )

        # create master dataset
        createRaster(bounds=(xMin, yMin, xMax, yMax), output=master, dtype=dataType, 
                            pixelWidth=dx, pixelHeight=dy, srs=srs, **kwargs)
        masterDS = gdal.Open(master, gdal.GA_Update)

    masterBand = masterDS.GetRasterBand(1)

    # Add each dataset to master
    for i in range(len(datasets)):
        # Calculate starting indicies
        if(flipY):
            yStart = int((extents[i][1]-yMin)/dy)
        else:
            yStart = int((yMax-extents[i][3])/dy)
        xStart = int((extents[i][0]-xMin)/dx)

        # pull data
        data = bands[i].ReadAsArray()
        if (data is None):
            warnings.warn("Empty raster file? {0}".format(os.path.basename(datasets[i])))
            continue
        
        # Get master data
        mas = masterBand.ReadAsArray(xStart, yStart, data.shape[1], data.shape[0])

        # create selector
        if (not nodataVals[i] is None):
            sel = np.where( data!=nodataVals[i] )
        else: 
            sel = np.where( data!=0 )

        # Add to master
        mas[sel] = data[sel]
        masterBand.WriteArray(mas, xStart, yStart)

    # Write final raster
    masterBand.FlushCache()
    del masterBand
    masterDS.FlushCache()
    calculateStats(masterDS)

