from .util import *
from .extent import *
from .regionMask import *

# Distribute items with a minimal separation within available areas
def placeItemsInMatrix(data, distance, placementDiv=10, threshold=0.10, suitabilityStep=0.10, fastMethod=False):
    """
    !!! REWRITE ME !!!
    Place items within an available areas (in the form of an exclusion matrix) given a 
     minumal distance between items.

    Returns a list of x and y coordinates of placed items

     * The given exclusion matrix must consist of values between 0 and 1 where 0 corresponds 
        to a fully excluded pixel and 1 corresponds to a fully avilable pixel.
     * Values between 0 and 1 will be handled such that the ratio of the item density found in a 
        contiguous value zone vs zones with full availability *should* be the given value
     * A maximum of ONE item will be placed in each pixel

    Input Arguments:
        data
            np.ndarray -- The matrix of values between zero and one wherein items will be placed

        distance
            float -- The distance between items in pixels
            * Must be >= 1

        placementDiv (10)
            int -- The number of divisions determining where within each pixel to place an item
                    can be placed

        threshold (0.10)
            float -- The mimimal factor value on which items can be placed

        output (None)
            str -- Path to an output shapefile
            * Assumes overwrite is desired!

        suitabilityStep
            float -- The step decrement in searching for available locations
            * The algorithm will search first for values equal to one, and assign item locations.
              Afterwards, the algorithm will search for values between 1 and 1-suitabilityStep, then
               1-suitabilityStep and 1-2*suitabilityStep, and so forth until threshold is reached

        method (False)
            bool -- The method indicator
            * False instructs the algorithm to only save item locations during runtime, and 
               then when searching within a new pixel calculates a tailored exclusion area based 
               off the pixel's suitability. 
               - This method will produce better results! 
            * True instructs the algorithm to save the exclusion area of each new turbine (based 
               off the suitability of their associated pixels), and then when searching within a new pixel
               allows new turbines to be placed in the first identified unexcluded location. 
               - This method is roughly 5-10x faster than the previous method, but items in pixels with lower
                 suitabilities may be closer than they should be to items in pixels of higher suitabilities.
    """

    # ensure inputs are correct type
    if(not type(placementDiv) is int): placementDiv = int(placementDiv)
    threshold = int(threshold*100)
    suitabilityStep = int(suitabilityStep*100)

    # Get the useful sizes
    ySize = data.shape[0]
    xSize = data.shape[1]
    divXSize = (xSize*placementDiv)
    divYSize = (ySize*placementDiv)
    
    maxDivDist = int(placementDiv*distance/np.sqrt(0.01*threshold))

    # Make stamps
    stamps = [None,]
    for width in range(1,maxDivDist+1):
        stamps.append( np.zeros((2*width+1, 2*width+1), dtype="bool"))
        tmp = np.zeros((width+1,width+1), dtype="bool")
        
        for yi in np.arange(0,width+1):
            for xi in np.arange(width,-1,-1):
                if(xi*xi+yi*yi <= width*width):
                    tmp[yi,:xi+1] = True
                    break
                    
        stamps[width][0:width+1,0:width+1] = tmp[::-1,::-1]
        stamps[width][width:,0:width+1] = tmp[:,::-1]
        stamps[width][0:width+1,width:] = tmp[::-1,:]
        stamps[width][width:,width:] = tmp[:,:]

    # make empty coordinate list
    xPlacements = []
    yPlacements = []
    exclusionDists = []

    # Initialize a placement exclusion matrix
    itemExclusion = np.zeros((divYSize+2*maxDivDist, divXSize+2*maxDivDist), dtype="bool")

    # Loop over suitability groups
    suitabilityMax = 100+suitabilityStep # So that suitabilities equal to 1 are handled first
    while suitabilityMax > threshold:

        # Get the indexes which are in the suitability range
        yIndexes, xIndexes = np.where(np.logical_and(data< 0.01*suitabilityMax, 
                                                     data>= 0.01*(suitabilityMax-suitabilityStep)))

        # Cast indexes to divided indicies
        yDivStartIndexes = yIndexes*placementDiv+maxDivDist
        xDivStartIndexes = xIndexes*placementDiv+maxDivDist

        # Loop over potential placements
        for i in range(len(yIndexes)):
            # Get the extent of the current pixel in the DIVIDED matrix
            yDivStart = yDivStartIndexes[i]
            yDivEnd = yDivStart + placementDiv + 1
            xDivStart = xDivStartIndexes[i]
            xDivEnd = xDivStart + placementDiv + 1

            # Get search distance
            suit = data[yIndexes[i], xIndexes[i]] # Get suitability at point
            exclusionDist = distance/np.sqrt(suit)
            width = int(placementDiv*exclusionDist)

            # Search for availability
            if(fastMethod):
                a = np.where( ~itemExclusion[yDivStart:yDivEnd, xDivStart:xDivEnd] ) # Get any point that is available
            else:
                itemLocs = np.where(itemExclusion[yDivStart-width:yDivEnd+width, xDivStart-width:xDivEnd+width])
                if itemLocs[0].size==0:
                    a = (np.array([0]), np.array([0]))
                else:
                    subSize = placementDiv+4*width+1
                    subItems = np.zeros((subSize, subSize), dtype="bool")

                    # Create an exclusion matrix accoring to the CURRENT suitability
                    for yi,xi in zip(itemLocs[0], itemLocs[1]):
                        yS = yi
                        yE = yS+2*width+1
                        xS = xi
                        xE = xS+2*width+1

                        subItems[yS:yE,xS:xE] = np.logical_or( subItems[yS:yE,xS:xE], stamps[width] )

                    t = 2*width
                    a = np.where( ~subItems[t:-t, t:-t])

            if a[0].size == 0: continue # No available space was found

            # Get the first available point
            yDiv = a[0][0]+yDivStart
            xDiv = a[1][0]+xDivStart

            # Add to placement list
            yPlacements.append( float(yDiv-maxDivDist)/placementDiv )
            xPlacements.append( float(xDiv-maxDivDist)/placementDiv )
            exclusionDists.append( exclusionDist )

            # Stamp the exclusion matrix
            if(fastMethod):
                # Stamp the area of the CURRENT suitability
                yS = yDiv-width
                yE = yDiv+width+1
                xS = xDiv-width
                xE = xDiv+width+1

                itemExclusion[yS:yE, xS:xE] = np.logical_or(itemExclusion[yS:yE, xS:xE], stamps[width])
            else:
                # Stamp just the item's location
                itemExclusion[yDiv, xDiv] = True

        # Decrement the suitability window
        suitabilityMax -= suitabilityStep

    # Finish
    return (np.array(yPlacements), np.array(xPlacements), np.array(exclusionDists))

def placeItemsInRaster(raster, distance, output=None, outputAsPoints=True, overwrite=False, **kwargs):
    """!!! WRITE ME !!!"""
    # Get Raster info and data
    ds = loadRaster(raster)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()

    info = describeRaster(ds)
    scale = (info.pixelHeight+info.pixelWidth)/2

    # place items
    yIndexLocs, xIndexLocs, exc = placeItemsInMatrix(data=data, distance=distance/scale, **kwargs)

    # Fix loctions
    if(info.flipY):
        yStart = info.yMax
        dy = -1*info.pixelHeight
    else:
        yStart = info.yMin
        dy = info.pixelHeight
    xStart = info.xMin

    yLocs = yStart+yIndexLocs*dy
    xLocs = xStart+xIndexLocs*info.pixelWidth

    exc = exc*scale

    # Return if output is None
    if (output is None):
        # Combine all points into a single multipoint object
        points = ogr.Geometry(ogr.wkbMultiPoint)
        points.AssignSpatialReference(info.srs)
        for x,y in zip(xLocs,yLocs):
            pt = ogr.Geometry(ogr.wkbPoint)
            pt.AddPoint(x,y)
            points.AddGeometry(pt)

        # return
        return points
    
    # Make an output file
    geoms = []
    items = []
    for y,x,e in zip(yLocs,xLocs,exc):
        tmp = ogr.Geometry(ogr.wkbPoint)
        tmp.AddPoint(x,y)
        tmp.AssignSpatialReference(info.srs)
        if( not outputAsPoints):
            geoms.append(tmp.Buffer(e/2))
            items.append(e)
        else:
            geoms.append(tmp)
            items.append(e)

    createVector( geoms, output=output, srs=info.srs, fieldVals={"exclusion":items}, 
                  fieldDef={"exclusion":"Real"}, overwrite=overwrite)
    return

# A better growing function
def growMatrix(mat, dist, div=10, simplifyGeom=False):
    """
    !!!! REWRITE ME !!!!
    Grow the mask by a given pixel distance. The returned matrix corresponds to the factor of each pixel which would be contained within the grown area. use "edgeDivision" to increase the accuracy of the result around edges (also causes increased computation time).

    Input keywords:
        data - (required)
            : np.ndarray -- The data overwhich to grow

        distance - (required) 
            : float -- The distance to grow in pixels

        edgeDivision - (default 0)
            : int -- The number of pixel divisions to use while processing the edges of the grown regions

        processes - (1)   
            : int -- Number of multiple processes to spawn while processing
            * One will not invoke creation of a pool object

        batchSize (1,000,000)  
            : int -- The working matrix size
            * Fiddle with this and "processes" to optimize for system

        centerGrown (False)
            : bool -- Indicates whether the regions are grown from pixel centers or edges

    ########################################################
    EXAMPLE 1: 

        Initial data:
        |    0,    0,    0,    0,    0 |
        |    0,    0,    0,    0,    0 |
        |    0,    0,    1,    0,    0 |
        |    0,    0,    0,    0,    0 |
        |    0,    0,    0,    0,    0 |

        Grown by 1 pixel:
        |    0,    0,    0,    0,    0 |
        |    0,    0,    1,    0,    0 |
        |    0,    1,    1,    1,    0 |
        |    0,    0,    1,    0,    0 |
        |    0,    0,    0,    0,    0 |

        Grown by one pixel with infinite edgeDivision:
        |    0,    0,    0,    0,    0 |
        |    0, pi/4,    1, pi/4,    0 |
        |    0,    1,    1,    1,    0 |
        |    0, pi/4,    1, pi/4,    0 |
        |    0,    0,    0,    0,    0 |

        Grown by one pixel with infinite edgeDivision and centerGrown:
        |    0,     0,    0,     0,    0 |
        |    0, .0788, .456, .0788,    0 |
        |    0,  .456,    1,  .456,    0 |
        |    0, .0788, .456, .0788,    0 |
        |    0,     0,    0,     0,    0 |

    ############################################################
    EXAMPLE 2:

        Initial data:
        |    0,    0,    0,    0,    0 |
        |    0,    1,    0,    1,    0 |
        |    0,    0,    0,    0,    0 |
        |    0,    0,    0,    0,    0 |
        |    0,    0,    1,    0,    0 |

        Grown by 1 pixel:
        |    0,    1,    0,    1,    0 |
        |    1,    1,    1,    1,    1 |
        |    0,    1,    0,    1,    0 |
        |    0,    0,    1,    0,    0 |
        |    0,    1,    1,    1,    0 |

        Grown by one pixel with high edgeDivision:
        | 0.772,   1.0, 0.947,   1.0, 0.772 |
        |   1.0,   1.0,   1.0,   1.0,   1.0 |
        | 0.772,   1.0, 0.947,   1.0, 0.772 |
        |   0.0, 0.772,   1.0, 0.772,   0.0 |
        |   0.0,   1.0,   1.0,   1.0,   0.0 |

        Grown by one pixel with infinite edgeDivision and centerGrown:
        | 0.079, 0.456, 0.157, 0.456, 0.079 |
        | 0.456,   1.0, 0.912,   1.0, 0.456 |
        | 0.079, 0.535, 0.612, 0.535, 0.079 |
        |   0.0, 0.456,   1.0, 0.456,   0.0 |
        |   0.0, 0.079, 0.456, 0.079,   0.0 |
    """
    # Check for empty matrix
    if( mat.max() == 0):
        return mat

    # Create a regionMask object
    extent = Extent(0,0,mat.shape[1],mat.shape[0], srs=EPSG3035)
    rm = RegionMask(mat,extent)

    # make geometry
    geom = rm.geometry

    if(simplifyGeom):
        previousArea = geom.Area()
        geom = geom.SimplifyPreserveTopology(0.80)
        
        areaPercDelta = abs(geom.Area()-previousArea)/previousArea
        if( areaPercDelta > 0.05):
            msg = "Geometry changed %f%% after simplifying "% (areaPercDelta*100)
            warnings.warn(msg)

    # Grow region
    geom = geom.Buffer(dist)

    # Make a new region mask
    newRM = RegionMask.fromGeom(geom, pixelSize=1/div, extent=rm.extent)

    # Done!
    return scaleMatrix( newRM.mask, -1*div)
    
def combineRasters(master, datasets, flipY=False, **kwargs):
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

#############################################################################################
def coordinateFilter(lats, lons, geom, wktSRS='latlon' ):
    """!!! WRITE ME !!!"""

    # Ensure lats and lons are 2 dim arrays
    if( len(lats.shape)==1 and len(lons.shape)==1 ): # lats and lons are 1D
        lons, lats = np.meshgrid(lons,lats) # yes, it needs to be 'opposite'

    if( not (len(lats.shape)==2 and len(lons.shape)==2 and lats.shape==lons.shape)):
        raise ValueError("Lats and lons shapes do not match")

    # Ensure we are working with a geometry object
    if(isinstance(geom, str)): # Assume geom is a wkt string
        geom = ogr.CreateGeometryFromWkt(geom)
        geom.AssignSpatialReference(wktSRS)
    else:
        geom = geom.Clone() # Just in case the object is tethered outside

    # Maybe geom needs to be projected?
    geomSRS = geom.GetSpatialReference()
    if( not geomSRS.IsSame(EPSG4326) ):
        geom = geom.Clone() # Clone incase it it still used outside the function
        geom.TransformTo(EPSG4326)

    # Get extent of geom
    extent = Extent.fromGeom(geom)

    # Slice lats and lons around extent 
    tmp = np.logical_and( lats>=extent.yMin, lats<=extent.yMax )
    np.logical_and( lons>=extent.xMin, tmp, tmp )
    np.logical_and( lons<=extent.xMax, tmp, tmp )

    sel = np.where(tmp)

    # Iterate over extent-contained points and see if they are in region
    goodVals = dict(lat=[], lon=[], latI=[], lonI=[])

    xMap, yMap = np.meshgrid(range(lats.shape[1]), range(lats.shape[0]) )

    for lat, lon, lonI, latI in zip( lats[sel], lons[sel], xMap[sel], yMap[sel] ):
        # make a point
        pt = ogr.Geometry( ogr.wkbPoint )
        pt.AddPoint( lon, lat )
        if(geom.Contains(pt)):
            goodVals["lat"].append(lat)
            goodVals["lon"].append(lon)
            goodVals["latI"].append(latI)
            goodVals["lonI"].append(lonI)

    # Done!
    return pd.DataFrame(goodVals)