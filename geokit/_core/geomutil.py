from .util import *
from .srsutil import *

####################################################################
# Geometry convenience functions

def makePoint(*args, srs='latlon'):
    """Make a simple point geometry
    
    Usage:
        makePoint(x, y [,srs])
        makePoint( (x, y) [,srs] )

    """
    if len(args)==1:
        x,y = args[0]
    elif len(args)==2:
        x = args[0]
        y = args[1]
    else:
        raise GeoKitGeomError("Too many positional inputs. Did you mean to specify \"srs=\"?")

    """make a point geometry from given coordinates (x,y) and srs"""
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(x,y)
    pt.AssignSpatialReference(loadSRS(srs))
    return pt

def makeBox(*args, srs=None):
    """Make an ogr polygon object from extents

    Usage:
        makeBox(xMin, yMin, xMax, yMax [, srs])
        makeBox( (xMin, yMin, xMax, yMax) [, srs])
    """
    if (len(args)==1):
        xMin,yMin,xMax,yMax = args[0]
    elif (len(args)==4):
        xMin,yMin,xMax,yMax = args
    else:
        raise GeoKitGeomError("Incorrect number positional inputs (only accepts 1 or 4). Did you mean to specify \"srs=\"?")

    # make sure inputs are good
    xMin = float(xMin)
    xMax = float(xMax)
    yMin = float(yMin)
    yMax = float(yMax)

    # make box
    box = ogr.Geometry(ogr.wkbPolygon)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x,y in [(xMin,yMin),(xMax,yMin),(xMax,yMax),(xMin,yMax),(xMin,yMin)]:
        ring.AddPoint(x,y)

    box.AddGeometry(ring)
    if(srs): 
        srs = loadSRS(srs)
        box.AssignSpatialReference(srs)
    return box

def makeEmpty(name, srs=None):
    """
    name options: Point, MultiPoint, Line, MultiLine, Polygon, MultiPolygon, ect...
    """
    if not hasattr(ogr,"wkb"+name):
        raise GeoKitGeomError("Could not find geometry named: "+name)
    geom = ogr.Geometry(getattr(ogr,"wkb"+name))

    if not srs is None:
        geom.AssignSpatialReference( loadSRS(srs))

    return geom

#################################################################################3
# Make a geometry from a WKT string
def convertWKT( wkt, srs=None):
    """Make a geometry from a WKT string"""
    geom = ogr.CreateGeometryFromWkt( wkt ) # Create new geometry from string 
    if geom is None: # test for success
        raise GeoKitGeomError("Failed to create geometry")
    if srs:
        geom.AssignSpatialReference(loadSRS(srs)) # Assign the given srs
    return geom

#################################################################################3
# Make a geometry from a matrix mask
def convertMask( mask, bounds=None, srs=None, flat=False):
    """Create a geometry from a matrix mask"""
    # Make sure we have a numpy array
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask, dtype='bool')

    # Make sure mask is bool type
    if not mask.dtype == np.bool:
        raise ValueError("mask must be of bool type to be turned into a geometry")

    # Make boundaries if not given
    if bounds is None:
        bounds = (0,0,mask.shape[1], mask.shape[0]) # bounds in xMin, yMin, xMax, yMax
        pixelHeight = 1
        pixelWidth  = 1

    try: # first try for a tuple
        xMin, yMin, xMax, yMax = bounds
    except: # next assume the user gave an extent object
        try:
            xMin, yMin, xMax, yMax = bounds.xyXY
            srs = bounds.srs
        except:
            raise GeoKitGeomError("Could not understand 'bounds' input")

    pixelHeight = (yMax-yMin)/mask.shape[0]
    pixelWidth  = (xMax-xMin)/mask.shape[1]

    if not srs is None: srs=loadSRS(srs)

    # Make a raster dataset and pull the band/maskBand objects

    cols = round((xMax-xMin)/pixelWidth) # used 'round' instead of 'int' because this matched GDAL behavior better
    rows = round((yMax-yMin)/abs(pixelHeight))
    originX = xMin
    originY = yMax # Always use the "Y-at-Top" orientation
    
    # Get DataType
    dtype = "GDT_Byte"
        
    # Open the driver
    driver = gdal.GetDriverByName('Mem') # create a raster in memory
    raster = driver.Create('', cols, rows, 1, getattr(gdal,dtype))

    if(raster is None):
        raise GeoKitGeomError("Failed to create temporary raster raster")

    raster.SetGeoTransform((originX, abs(pixelWidth), 0, originY, 0, -1*abs(pixelHeight)))
    
    # Set the SRS
    if not srs is None:
        rasterSRS = loadSRS(srs)
        raster.SetProjection( rasterSRS.ExportToWkt() )

    # Set data into band
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray( mask )

    band.FlushCache()
    raster.FlushCache()

    #rasDS = createRaster(bounds=bounds, data=mask, noDataValue=0, pixelWidth=pixelWidth, pixelHeight=pixelHeight, srs=srs)

    # Do a polygonize
    rasBand = raster.GetRasterBand(1)
    maskBand = rasBand.GetMaskBand()

    # Open an empty vector dataset, layer, and field
    driver = gdal.GetDriverByName("Memory")
    tmp_driver = gdal.GetDriverByName("ESRI Shapefile")
    t = TemporaryDirectory()
    tmp_dataSource = tmp_driver.Create( t.name+"tmp.shp", 0, 0 )

    vecDS = driver.CreateCopy("MEMORY", tmp_dataSource)
    t.cleanup()

    vecLyr = vecDS.CreateLayer("mem",srs=srs)

    field = ogr.FieldDefn("DN", ogr.OFTInteger)
    vecLyr.CreateField(field)

    # Polygonize geometry
    result = gdal.Polygonize(rasBand, maskBand, vecLyr, 0)
    if( result != 0):
        raise GeoKitGeomError("Failed to polygonize geometry")

    # Check how many features were created
    ftrN = vecLyr.GetFeatureCount()

    if( ftrN == 0):
        #raise GlaesError("No features in created in temporary layer")
        print("No features in created in temporary layer")
        if flatten: return None
        else: return []

    # If only one feature created, return it
    if( ftrN==1 ):
        ftr = vecLyr.GetFeature(0)
        if flatten:
            final = ftr.GetGeometryRef().Clone()
        else:
            final = [ftr.GetGeometryRef().Clone(),]

    # Check if the user doesn't want a flat geometry
    
    geoms = []
    for i in range(ftrN):
        ftr = vecLyr.GetFeature(i)
        geoms.append(ftr.GetGeometryRef().Clone())

    final = flatten(geoms) if flat else geoms
        
    # Cleanup
    vecLyr = None
    vecDS = None
    maskBand = None
    rasBand = None
    raster = None

    return final

# geometry transformer
def transform( geoms, toSRS='europe_m', fromSRS=None, segment=None):
    """Transforms a geometry, or a list of geometries, from one SRS to another"""
    # make sure geoms is a list
    if isinstance(geoms, ogr.Geometry):
        geoms = [geoms, ]
    else: # assume geoms is iterable
        try:
            geoms = list(geoms)
        except Exception as e:
            print("Could not determine geometry SRS")
            raise e

    # make sure geoms is a list
    if fromSRS is None:
        fromSRS = geoms[0].GetSpatialReference()
        
    # load srs's
    fromSRS = loadSRS(fromSRS)
    toSRS = loadSRS(toSRS)

    # make a transformer
    trx = osr.CoordinateTransformation(fromSRS, toSRS)

    # Do transformation
    geoms = [g.Clone() for g in geoms]
    if not segment is None: [g.Segmentize(segment) for g in geoms]
    
    r = [g.Transform(trx) for g in geoms]
    if sum(r)>0: # check fro errors
        raise GeoKitGeomError("Errors in geometry transformations")
        
    # Done!
    if len(geoms)==1: return geoms[0]
    else: return geoms

#################################################################################3
# Flatten a list of geometries
def flatten( geoms ):
    """Flatten a list of geometries into a single geometry"""
    if not isinstance(geoms,list):
        geoms = list(geoms)
        try: # geoms is not a list, but it might be iterable
            geoms = list(geoms)
        except:
            raise ValueError("argument must be a list of geometries")

    if len(geoms) == 0: return None
    
    ## Combine geometries by iteratively union-ing nearby (according to index) geometries
    ##  * example, given a list of geometries (A,B,C,D,E,F,G,H,I,J):
    ##       [ A  B  C  D  E  F  G  H  I  J ]
    ##       [  AB    CD    EF    GH    IJ  ]
    ##       [    ABCD        EFGH      IJ  ]
    ##       [        ABCDEFGH          IJ  ]
    ##       [               ABCDEFGHIJ     ]  <- Result is a list with only one geometry. 
    ##                                            This becomes the resulting geometry  
    ##
    ##  * I had to do this because unioning one by one was much too slow since it would create
    ##    a single large geometry which had to be manipulated for EACH new addition

    while( len(geoms)>1):
        newGeoms = []
        for gi in range(0,len(geoms),2):
            try:
                newGeoms.append(geoms[gi].Union(geoms[gi+1]))
            except IndexError: # should only occur when length of geoms is odd
                newGeoms.append(geoms[gi])
        geoms = newGeoms
    return geoms[0]
