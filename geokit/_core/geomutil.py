from .util import *
from .srsutil import *

POINT = ogr.wkbPoint
MULTIPOINT = ogr.wkbMultiPoint
LINE = ogr.wkbLineString
MULTILINE = ogr.wkbMultiLineString
POLYGON = ogr.wkbPolygon
MULTIPOLYGON = ogr.wkbMultiPolygon

####################################################################
# Geometry convenience functions
def makePoint(*args, srs='latlon'):
    """Make a simple point geometry
    
    Usage:
        makePoint(x, y [,srs])
        makePoint( (x, y) [,srs] )
    
    * srs must be given as a keyword argument
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
    if not srs is None:
        pt.AssignSpatialReference(loadSRS(srs))
    return pt

def makeBox(*args, srs=None):
    """Make an ogr polygon object from extents

    Usage:
        makeBox(xMin, yMin, xMax, yMax [, srs])
        makeBox( (xMin, yMin, xMax, yMax) [, srs])

    * srs must be given as a keyword argument
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

def makePolygon(outerRing, *args, srs=None):
    """Creates an OGR Polygon obect from a given set of points

    Inputs:
        outerRing : The polygon's outer edge
            - an iterable of (x,y) coordinates
            - an Nx2 numpy array also works

        *args : The inner edges of the polygon
            - iterable (x,y) coordinate sets
            * Each input forms a single edge
            * Inner rings cannot interset the outer ring or one another 
            * NOTE! For proper drawing, inner rings must be given in the opposite orientation as 
              the outer ring (clockwise vs counterclockwise)

        srs : The Spatial reference system to apply to the created geometry
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string
            * srs must be given as a keyword argument
    
    Example: Make a diamond cut out of a box

        box = [(-2,-2), (-2,2), (2,2), (2,-2), (-2,-2)]
        diamond = [(0,1), (-0.5,0), (0,-1), (0.5,0), (0,1)]

        geom = makePolygon( box, diamond )
    """
    if not srs is None: srs = loadSRS(srs)

    # Make the complete geometry
    g = ogr.Geometry(ogr.wkbPolygon)
    if not srs is None: g.AssignSpatialReference(srs)

    # Make the outer ring
    otr = ogr.Geometry(ogr.wkbLinearRing)
    if not srs is None: otr.AssignSpatialReference(srs)
    [otr.AddPoint(x,y) for x,y in outerRing]
    g.AddGeometry(otr)

    # Make the inner rings (maybe)
    for innerRing in args:
        tmp = ogr.Geometry(ogr.wkbLinearRing)
        if not srs is None: tmp.AssignSpatialReference(srs)
        [tmp.AddPoint(x,y) for x,y in innerRing]

        g.AddGeometry(tmp)
        
    # Make sure geom is valid
    g.CloseRings()
    if not g.IsValid(): raise GeoKitGeomError("Polygon is invalid")

    # Done!
    return g

def makeLine(points, srs=None):
    """Creates an OGR Polygon obect from a given set of points

    Inputs:
        points : The line's point coordinates
            - an iterable of (x,y) coordinates
            - Also works for an Nx2 numpy array
            * Forms the outermost edge of the polygon

        srs : The Spatial reference system to apply to the created geometry
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string
            * srs must be given as a keyword argument
    """

    # Make the complete geometry
    g = ogr.Geometry(ogr.wkbLineString)
    if not srs is None: g.AssignSpatialReference(srs)

    # Make the line
    [g.AddPoint(x,y) for x,y in points]
    #g.AddGeometry(otr)

    # Ensure valid
    if not g.IsValid(): raise GeoKitGeomError("Polygon is invalid")

    # Done!
    return g


def makeEmpty(name, srs=None):
    """
    Make an generic OGR geometry

    name options: Point, MultiPoint, Line, MultiLine, Polygon, MultiPolygon, ect...

    *Not for the feint of heart*
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
def convertMask( mask, bounds=None, srs=None, flat=False, shrink=True, cornerConnected=False):
    """Create a geometry set from a matrix mask

    Inputs:
        mask : The matrix which will be turned into a geometry
            - a 2D boolean matrix
            * True values are interpreted as 'in the geometry'

        bounds : Determines the boundary context for the given mask and will scale the geometry's coordinates accordingly
            - (xMin, yMin, xMax, yMax)
            - geokit.Extent object
            * If a boundary is not given, the geometry coordinates will correspond to the mask's indicies
            * If the boundary is given as an Extent object, an srs input is not required

        srs : The Spatial reference system to apply to the created geometries
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string
            * This input is ignored if bounds is a geokit.Extent object

        flat : If True, flattens the resulting geometries into a single geometry object
            - True/False

        shrink : If True, shrink all geoms by a tiny amount in order to avoid geometry overlapping issues
            * The total amount shrunk should be very small
            * Generally this should be left as True unless it is ABSOLUTELY neccessary to maintain the same area
    """
    
    # Make sure we have a boolean numpy matrix
    if not isinstance(mask, np.ndarray):
        raise GeoKitGeomError("Mask must be a 2D boolean numpy ndarray")
    if(mask.dtype != "bool" and mask.dtype != "uint8" ): 
        raise GeoKitGeomError("Mask must be a 2D boolean numpy ndarray")
    if(mask.dtype == "uint8"):
        mask = mask.astype("bool")

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

    cols = int(round((xMax-xMin)/pixelWidth)) # used 'round' instead of 'int' because this matched GDAL behavior better
    rows = int(round((yMax-yMin)/abs(pixelHeight)))
    originX = xMin
    originY = yMax # Always use the "Y-at-Top" orientation
    
    # Get DataType
    dtype = "GDT_Byte"
        
    # Open the driver
    driver = gdal.GetDriverByName('Mem') # create a raster in memory
    raster = driver.Create('', cols, rows, 1, getattr(gdal,dtype))

    if(raster is None):
        raise GeoKitGeomError("Failed to create temporary raster")

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
    #driver = gdal.GetDriverByName("Memory")
    #tmp_driver = gdal.GetDriverByName("ESRI Shapefile")
    #t = TemporaryDirectory()
    #tmp_dataSource = tmp_driver.Create( t.name+"tmp.shp", 0, 0 )

    #vecDS = driver.CreateCopy("MEMORY", tmp_dataSource)
    #t.cleanup()

    vecDS = gdal.GetDriverByName("Memory").Create( '', 0, 0, 0, gdal.GDT_Unknown )
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
        if flat: return None
        else: return []

    # If only one feature created, set it
    geoms = []
    for i in range(ftrN):
        ftr = vecLyr.GetFeature(i)
        geoms.append(ftr.GetGeometryRef().Clone())

    # Do shrink, maybe
    if shrink: 
        # Compute shrink factor
        shrinkFactor = -0.001*(xMax-xMin)/mask.shape[1]
        geoms = [g.Buffer(shrinkFactor) for g in geoms]

    # Do flatten, maybe
    if flat:
        final = flatten(geoms) if len(geoms)>1 else geoms[0]
    else:
        final = geoms
        
    # Cleanup
    vecLyr = None
    vecDS = None
    maskBand = None
    rasBand = None
    raster = None

    return final

# geometry transformer
def transform( geoms, toSRS='europe_m', fromSRS=None, segment=None):
    """Transforms a geometry, or a list of geometries, from one SRS to another

    Inputs:
        geoms : The geometry or geometries to transform
            - ogr Geometry object
            - and iterable of og Geometry objects
            * All geometries must have the same spatial reference

        toSRS : The srs of the output geometries
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string

        fromSRS : The srs of the input geometries
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string
            * If fromSRS is None, the geometry's internal srs will be used
            * If fromSRS is given, it will override any geometry's SRS (will likely cause weird outputs)

        segment : An optional segmentation length of the input geometries
            - float
            * Units are in the input geometry's native unit
            * If given, the input geometries will be segmented such that no line segment is longer than the given 
              segment size
              - use this for a more detailed transformation!
        """
    # make sure geoms is a list
    if isinstance(geoms, ogr.Geometry):
        returnSingle = True
        geoms = [geoms, ]
    else: # assume geoms is iterable
        returnSingle = False
        try:
            geoms = list(geoms)
        except Exception as e:
            print("Could not determine geometry SRS")
            raise e

    # make sure geoms is a list
    if fromSRS is None:
        fromSRS = geoms[0].GetSpatialReference()
        if fromSRS is None:
            raise GeoKitGeomError("Could not determine fromSRS from geometry")
        
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
    if returnSingle: return geoms[0]
    else: return geoms

#################################################################################3
# Flatten a list of geometries
def flatten( geoms ):
    """Flatten a list of geometries into a single geometry object

    Example:
        - A list of Polygons/Multipolygons will become a single Multipolygon
        - A list of Linestrings/MultiLinestrings will become a single MultiLinestring"""
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


def drawGeoms(geoms, ax=None, srs=None, simplification=None, **mplargs):
    """Draw geometries onto a matplotlib figure
    
    * Each geometry type is displayed as an appropriate plotting type
        -> Points/ Multipoints are displayed as points using plt.plot(...)
        -> Lines/ MultiLines are displayed as lines using plt.plot(...)
        -> Polygons/ MultiPolygons are displayed as patches using the descartes library
    * Each geometry can be given its own set of matplotlib plotting parameters

    Inputs:
        geoms : The geometry/geometries to draw
            - a single ogr Geometry object
            - a tuple -- (ogr Geometry, dict of plotting parameters for this geometry)
            - a dict containing at least one 'geom' key pointing to an ogr Geometry object
                * other kwargs are treated as plotting parameters for the geometry
            - An iterable of any the above
            * All geometries must have the same native SRS
        
        ax  : a matplotlib axeis object to plot on
            * if ax is None, the function will create its own axis and automatically show it

        srs : The SRS to draw the geometries in
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string
            * if srs is None, the geometries' native SRS will be used

        simplification : A simplification factor to apply onto each geometry whichs ensure no line segment is shorter
                         than the given value
            * float
            - Units are the same as the 'srs' input
            - Using this will make plotting easier and use less resources

        **mplargs : matplotlib keyword arguments to apply to each geometry
            * Specified keyword arguments for each geometry inherit from mplargs 
    
    Retuns: A list of matplotlib handels to the created items
    """
    # do some imports
    from descartes import PolygonPatch
    from json import loads

    showPlot = False
    if ax is None:
        showPlot = True
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,12))
        ax = plt.subplot(111)

    # Be sure we have a list
    if isinstance(geoms,ogr.Geometry) or isinstance(geoms, dict) or isinstance(geoms, tuple):
        geoms = [geoms,]
    else: #Assume its an iterable
        geoms = list(geoms)

    # Separate geoms into geometries and mpl-parameters
    mplParams = []
    tmpGeoms = []
    for g in geoms:
        params = mplargs.copy()
        if isinstance(g, ogr.Geometry): 
            tmpGeoms.append(g)
            mplParams.append(params)
        
        elif isinstance(g,tuple):
            tmpGeoms.append(g[0])

            params.update(g[1])
            mplParams.append(params)

        elif isinstance(g,dict):
            tmpGeoms.append(g.pop('geom'))

            params.update(g)
            mplParams.append(params)

    geoms = tmpGeoms
    # Test the first geometry to see if the srs needs transforming
    if not srs is None:
        srs = loadSRS(srs   )
        gSRS = geoms[0].GetSpatialReference()
        
        if not gSRS.IsSame(srs):
            geoms = transform(geoms, toSRS=srs, fromSRS=gSRS)


    # Apply simplifications if required
    if not simplification is None:
        geoms = [g.Simplify(simplification) for g in geoms]

    # Consider matplotlib parameters
    if isinstance(mplParams, dict):
        mplForEachGeom = False
    else:
        mplForEachGeom = True
        mplParams = list(mplParams) # make sure mplParams is a list

    # unpack multigeomsgons and make into shapey objects
    polys = []
    polyParams = []
    lines = []
    lineParams = []
    points = []
    pointParams = []

    for gi in range(len(geoms)):
        g = geoms[gi]
        if g.GetGeometryName()=="POLYGON":
            polys.append( loads(g.ExportToJson()))
            if mplForEachGeom: polyParams.append(mplParams[gi])
            else: polyParams.append(mplParams)

        elif g.GetGeometryName()=="MULTIPOLYGON":
            for sgi in range(g.GetGeometryCount()):
                gg = g.GetGeometryRef(sgi)
                polys.append( loads(gg.ExportToJson()))
                if mplForEachGeom: polyParams.append(mplParams[gi])
                else: polyParams.append(mplParams)

        elif g.GetGeometryName()=="LINESTRING":
            pts = np.array(g.GetPoints())
            lines.append( (pts[:,0], pts[:,1]) )
            if mplForEachGeom: lineParams.append(mplParams[gi])
            else: lineParams.append(mplParams)

        elif g.GetGeometryName()=="LINEARRING":
            g.CloseRings()
            pts = np.array(g.GetPoints())
            lines.append( (pts[:,0], pts[:,1]) )
            if mplForEachGeom: lineParams.append(mplParams[gi])
            else: lineParams.append(mplParams)

        elif g.GetGeometryName()=="MULTILINESTRING":
            for sgi in range(g.GetGeometryCount()):
                gg = g.GetGeometryRef(sgi)
                pts = np.array(gg.GetPoints())
                lines.append( (pts[:,0], pts[:,1]) )
                if mplForEachGeom: lineParams.append(mplParams[gi])
                else: lineParams.append(mplParams)

        elif g.GetGeometryName()=="POINT":
            points.append( ([g.GetX(),], [g.GetY()] ) )
            
            tmp = mplParams[gi].copy() if mplForEachGeom else mplParams.copy()
            tmp["linestyle"] = tmp.get('linestyle','None')
            tmp["marker"] = tmp.get('marker','o')
            pointParams.append(tmp)

        elif g.GetGeometryName()=="MULTIPOINT":
            x = []
            y = []
            for sgi in range(g.GetGeometryCount()):
                gg = g.GetGeometryRef(sgi)
                pts = np.array(gg.GetPoints())
                x.append( gg.GetX() )
                y.append( gg.GetY() )
            points.append((x,y))
            
            tmp = mplParams[gi].copy() if mplForEachGeom else mplParams.copy()
            tmp["linestyle"] = tmp.get('linestyle','None')
            tmp["marker"] = tmp.get('marker','o')
            pointParams.append(tmp)

        else:
            raise GeoKitGeomError("Function can't understand geometry: "+g.GetGeometryName())

    ### Do Plotting
    # make patches
    handels = []
    patches = [PolygonPatch(sp, **kwargs) for sp, kwargs in zip(polys, polyParams)]
    handels.extend([ax.add_patch(p) for p in patches])

    # make lines
    handels.extend([ax.plot( *xy, **kwargs) for xy, kwargs in zip(lines, lineParams) ])

    # make points
    handels.extend([ax.plot( *xy, **kwargs) for xy, kwargs in zip(points, pointParams) ])

    # done!
    if showPlot: 
        ax.set_aspect('equal')
        ax.autoscale(enable=True)
        plt.show()

    else: return handels

def partitionArea2(geom, gridRes, resolution):
    # test if the geometry is more than 200% of the target area, if not just return it
    #if geom.Area() <= 2.0*targetArea: return np.array([geom.Clone(), ]), np.array([0,])

    # Get geometry information
    xMin, xMax, yMin, yMax = geom.GetEnvelope()
    srs = geom.GetSpatialReference()
    if srs is None: srs=EPSG3035

    #if min(xMax-xMin,yMax-yMin) <= 2*resolution: raise GeoKitGeomError("resolution is too large compared to the geometry") 

    # make a vector dataset
    vec = quickVector(geom)

    # Turn geometry into a matrix
    #res = min((xMax-xMin)/resolutionFactor, (yMax-yMin)/resolutionFactor)
    res = resolution
    raster = quickRaster(bounds=(xMin,yMin,xMax,yMax), srs=srs, dx=res, dy=res)

    # rasterize vector onto raster
    result = gdal.Rasterize(raster, vec, bands=[1], burnValues=[1])
    if not result==1:
        raise GeoKitGeomError("Rasterization failed")

    # Get raster info and data
    matrix = raster.GetRasterBand(1).ReadAsArray()>0.5
    yN, xN = matrix.shape

    xOrigin, dx, trash, yOrigin, trash, dy = raster.GetGeoTransform()
    dx = abs(dx)
    dy = abs(dy)

    xMin = xOrigin
    yMin = yOrigin - dy*yN
    xMax = xOrigin + dx*xN
    yMax = yOrigin
    
    # convert target area to a pixel area
    #targetPixelArea = targetArea/dy/dx
    pixelGridSize = int(gridRes/min( dy, dx))

    # Make an empty areas matrix
    noData = 2**30-1
    isOkay = 0
    areas = np.zeros(matrix.shape, dtype=np.int32)+noData
    areas[matrix] = isOkay

    # Loop over all available points which are not too close to the borders
    count = 1
    for ys in range(0, yN, pixelGridSize):
        for xs in range(0, xN, pixelGridSize):
            yn = min(yN, ys+pixelGridSize)
            xn = min(xN, xs+pixelGridSize)

            areas[ys:yn, xs:xn][matrix[ys:yn, xs:xn]] = count
            count += 1

    # Make a new raster of sub-geom indicated pixels
    raster = None
    raster = quickRaster(bounds=(xMin,yMin,xMax,yMax), srs=srs, dx=res, dy=res, noData=noData, data=areas)
    band = raster.GetRasterBand(1)
    maskBand = band.GetMaskBand()

    # Do polygonize
    vecDS = gdal.GetDriverByName("Memory").Create( '', 0, 0, 0, gdal.GDT_Unknown )
    vecLyr = vecDS.CreateLayer("mem",srs=srs)
    
    vecField = ogr.FieldDefn("DN", ogr.OFTInteger)
    vecLyr.CreateField(vecField)

    # Polygonize geometry
    result = gdal.Polygonize(band, maskBand, vecLyr, 0)
    if( result != 0):
        raise GeoKitGeomError("Failed to polygonize geometry")

    # Check the geoms
    ftrN = vecLyr.GetFeatureCount()

    if( ftrN == 0):
        #raise GlaesError("No features in created in temporary layer")
        print("No features in created in temporary layer")
        if flat: return None
        else: return []

    # Read the geoms
    geoms = []
    values = []
    for i in range(ftrN):
        ftr = vecLyr.GetFeature(i)
        geoms.append(ftr.GetGeometryRef().Clone())
        values.append(ftr.items()["DN"])

    geoms = np.array(geoms)
    values = np.array(values)

    return geoms,values

    # flatten the geometries
    flatValues = []
    flatGeoms = []

    for g in geoms[values==0]:
        flatGeoms.append(g)
        flatValues.append(0)

    for val in set(values):
        if val==0: continue # skip the unpartitioned areas
        flatValues.append(val)

        geomList = geoms[values==val]
        if geomList.size==1:
            flatGeoms.append(geomList[0])
        else:
            flatGeoms.append(flatten(geomList))

    flatValues = np.array(flatValues)
    flatGeoms = np.array(flatGeoms)

    return flatGeoms, flatValues


def partitionArea(geom, targetArea, resolution, steps=20, minRatio=0.25):
    # test if the geometry is more than 200% of the target area, if not just return it
    #if geom.Area() <= 2.0*targetArea: return np.array([geom.Clone(), ]), np.array([0,])

    # Get geometry information
    xMin, xMax, yMin, yMax = geom.GetEnvelope()
    srs = geom.GetSpatialReference()
    if srs is None: srs=EPSG3035

    #if min(xMax-xMin,yMax-yMin) <= 2*resolution: raise GeoKitGeomError("resolution is too large compared to the geometry") 

    # make a vector dataset
    vec = quickVector(geom)

    # Turn geometry into a matrix
    #res = min((xMax-xMin)/resolutionFactor, (yMax-yMin)/resolutionFactor)
    res = resolution
    raster = quickRaster(bounds=(xMin,yMin,xMax,yMax), srs=srs, dx=res, dy=res)

    # rasterize vector onto raster
    result = gdal.Rasterize(raster, vec, bands=[1], burnValues=[1])
    if not result==1:
        raise GeoKitGeomError("Rasterization failed")

    # Get raster info and data
    matrix = raster.GetRasterBand(1).ReadAsArray()
    yN, xN = matrix.shape

    xOrigin, dx, trash, yOrigin, trash, dy = raster.GetGeoTransform()
    dx = abs(dx)
    dy = abs(dy)

    xMin = xOrigin
    yMin = yOrigin - dy*yN
    xMax = xOrigin + dx*xN
    yMax = yOrigin
    
    # convert target area to a pixel area
    targetPixelArea = targetArea/dy/dx

    # get minimum/maximum pixel radius
    minRadius = np.sqrt(targetPixelArea/np.pi)
    maxArea = targetPixelArea/minRatio
    maxRadius = np.sqrt(maxArea/np.pi)
    mRI = int(np.ceil(maxRadius))

    # pad the availability matrix so that we can evaluate the kernel on the edges
    tmp = np.zeros((yN+2*mRI, xN+2*mRI), dtype=bool)
    tmp[mRI:-mRI, mRI:-mRI] = matrix
    matrix = tmp

    # make a set of inclusion stamps
    y = x = np.arange(-mRI,mRI+1)
    xx,yy = np.meshgrid(x,y)

    stamps = [ np.sqrt(xx*xx+yy*yy)<=radius for radius in np.linspace(minRadius,maxRadius,steps) ]
    stampSizes = [stamp.sum() for stamp in stamps]

    # Make an empty areas matrix
    noData = 2**30-1
    isOkay = 0
    areas = np.zeros(matrix.shape, dtype=np.int32)+noData
    areas[matrix] = isOkay

    # Loop over all available points which are not too close to the borders
    count = 1
    for yi, xi in np.argwhere(matrix):
        # Be sure index isn't already taken by something
        if not areas[yi,xi] == isOkay: continue
        #print("\n",count)
        for stamp in stamps:
            # start searching for an acceptable areas
            areaMat = np.logical_and(matrix[yi-mRI:yi+mRI+1, xi-mRI:xi+mRI+1], stamp)
            area = areaMat.sum()
            
            # If the applicable search area has gotten TOO large, go to the next pixel
            if area > targetPixelArea*1.1: break

            # Check if the area is acceptable
            if area >= targetPixelArea: # break out if we have found an area which satisfies the target area
                # Now that we know something has been found, add the areas to areas matrix indicated by the count
                areas[yi-mRI:yi+mRI+1, xi-mRI:xi+mRI+1][areaMat] = count
                count += 1

                # also remove these pixels from the availability matrix
                matrix[yi-mRI:yi+mRI+1, xi-mRI:xi+mRI+1][areaMat] = 0
                break

    # unpad the areas matrix
    areas = areas[mRI:-mRI, mRI:-mRI]

    # Make a new raster of sub-geom indicated pixels
    raster = None
    raster = quickRaster(bounds=(xMin,yMin,xMax,yMax), srs=srs, dx=res, dy=res, noData=noData, data=areas)
    band = raster.GetRasterBand(1)
    maskBand = band.GetMaskBand()

    # Do polygonize
    vecDS = gdal.GetDriverByName("Memory").Create( '', 0, 0, 0, gdal.GDT_Unknown )
    vecLyr = vecDS.CreateLayer("mem",srs=srs)
    
    vecField = ogr.FieldDefn("DN", ogr.OFTInteger)
    vecLyr.CreateField(vecField)

    # Polygonize geometry
    result = gdal.Polygonize(band, maskBand, vecLyr, 0)
    if( result != 0):
        raise GeoKitGeomError("Failed to polygonize geometry")

    # Check the geoms
    ftrN = vecLyr.GetFeatureCount()

    if( ftrN == 0):
        #raise GlaesError("No features in created in temporary layer")
        print("No features in created in temporary layer")
        if flat: return None
        else: return []

    # Read the geoms
    geoms = []
    values = []
    for i in range(ftrN):
        ftr = vecLyr.GetFeature(i)
        geoms.append(ftr.GetGeometryRef().Clone())
        values.append(ftr.items()["DN"])

    geoms = np.array(geoms)
    values = np.array(values)

    # flatten the geometries
    flatValues = []
    flatGeoms = []

    for g in geoms[values==0]:
        flatGeoms.append(g)
        flatValues.append(0)

    for val in set(values):
        if val==0: continue # skip the unpartitioned areas
        flatValues.append(val)

        geomList = geoms[values==val]
        if geomList.size==1:
            flatGeoms.append(geomList[0])
        else:
            flatGeoms.append(flatten(geomList))

    flatValues = np.array(flatValues)
    flatGeoms = np.array(flatGeoms)

    return flatGeoms, flatValues

import time
CLOCK = 0

def partitionArea3(geom, targetArea, growStep=None, _startPoint=0):
    global CLOCK
    if growStep is None:
        growStep = np.sqrt(targetArea/np.pi)/2

    # Be sure we are working with a polygon
    if geom.GetGeometryName()=="POLYGON":
        pass
    elif geom.GetGeometryName()=="MULTIPOLYGON":
        results = []
        for gi in range(geom.GetGeometryCount()):
            tmpResults = partitionArea3(geom.GetGeometryRef(gi), targetArea, growStep)
            results.extend( tmpResults )

        return results
    else:
        raise GeoKitGeomError("Geometry is not a polygon or multipolygon object")
        
    # Check the geometry's size
    #st = time.clock()
    gArea = geom.Area()
    if gArea < 1.5*targetArea: return [geom.Clone(), ]

    # Find the most starting boundary coordinate
    boundary = geom.Boundary()
    if boundary.GetGeometryCount() == 0:
        coords = boundary.GetPoints()
    else:
        coords = boundary.GetGeometryRef(0).GetPoints()
    
    #xc, yc = geom.Centroid().GetPoints()[0]
    y = np.array([c[1] for c in coords])
    x = np.array([c[0] for c in coords])

    if _startPoint==0: # start from the TOP-LEFT
        yStart = y.max()
        xStart = x[y==yStart].min()
    elif _startPoint==1: # Start from the LEFT-TOP
        xStart = x.min()
        yStart = y[x==xStart].max()
    elif _startPoint==2: # Start from the RIGHT-TOP
        xStart = x.max()
        yStart = y[x==xStart].max()
    elif _startPoint==3: # Start from the BOT-RIGHT
        yStart = y.min()
        xStart = x[y==yStart].max()
    elif _startPoint==4: # Start from the BOT-LEFT
        yStart = y.min()
        xStart = x[y==yStart].min()
    elif _startPoint==5: # Start from the LEFT-BOT
        xStart = x.min()
        yStart = y[x==xStart].min()
    else:
        raise GeoKitGeomError("Start point failure. There may be an infinite loop in one of the geometries")
    #i = np.argmax( np.power(y-yc,2)+np.power(x-xc,2) 


    start = makePoint(xStart, yStart, srs=geom.GetSpatialReference())

    # start searching
    searchGeom = start.Buffer(growStep).Intersection(geom)
    sgArea = searchGeom.Area()

    if gArea < 2*targetArea: # use a slightly smalled target area when the whole geometry
                             #  is close to twice the target area in order to increase the 
                             #  liklihood of a usable leftover
        workingTarget = 0.9*targetArea
    else: workingTarget = targetArea
    #clock=int((time.clock()-st)*1e9)
    #st = time.clock()
    
    while sgArea<workingTarget:
        newGeom = searchGeom.Buffer(growStep).Intersection(geom)
        newArea = newGeom.Area()
        if newArea > workingTarget*1.1:
            #print("special")
            dA = (newArea-sgArea)/growStep
            weightedGrowStep = (workingTarget-sgArea)/dA
            searchGeom = searchGeom.Buffer( weightedGrowStep ).Intersection(geom)
            break

        searchGeom = newGeom
        sgArea = newArea
    #clock=int((time.clock()-st)*1e9)

    # fix the search geometry
    #  - For some reason the searchGeometry will sometime create a geometry and a linestring,
    #    in these cases the real geometry was always the second item...
    if searchGeom.GetGeometryName() == "GEOMETRYCOLLECTION":
        for gi in range(searchGeom.GetGeometryCount()):
            g = searchGeom.GetGeometryRef(gi)
            if g.GetGeometryName()=="POLYGON":
                searchGeom = g.Clone()
                break

    # Check the left over geometry, maybe some poops have been created and we need to glue them together
    st = time.clock()
    outputGeom = searchGeom.Simplify(growStep/20)
    geomToDo = []

    leftOvers = geom.Difference(searchGeom)
    if leftOvers.GetGeometryName() == "MULTIPOLYGON":
        for gi in range(leftOvers.GetGeometryCount()):
            leftOver = leftOvers.GetGeometryRef(gi)
            if leftOver.Area() < targetArea*0.5:
                outputGeom = outputGeom.Union(leftOver)
            else:
                geomToDo.append(leftOver)
    elif leftOvers.GetGeometryName() == "POLYGON":
        geomToDo.append(leftOvers)
    else:
        raise GeoKitGeomError("FATAL ERROR: Difference did not result in a polygon")

    CLOCK += int((time.clock()-st)*1e9)

    
    # make an output array
    if outputGeom.Area()<targetArea*1.5:
        output = [outputGeom]
    else:
        # the search geom plus some (or maybe all) of the left over poops total an area which is too large,
        #  so it will need recomputing. But in order to decrease the liklihhod of an infinite loop,
        #  use a difference starting point than the one used before
        #  - This will loop a maximum of 6 times before an exception is raised
        output = partitionArea3( outputGeom, targetArea, growStep, _startPoint+1) 
 
    for g in geomToDo:
        tmpOutput = partitionArea3( g, targetArea, growStep)
        output.extend(tmpOutput)

    # Done!
    return output