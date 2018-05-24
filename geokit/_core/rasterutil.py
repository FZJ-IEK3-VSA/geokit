from .util import *
from .srsutil import *
from .geomutil import *
from .location import *

if( "win" in sys.platform):COMPRESSION_OPTION = ["COMPRESS=LZW"]
else: COMPRESSION_OPTION = ["COMPRESS=DEFLATE"]

# Basic Loader
def loadRaster(source):
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
    if(isinstance(source,str)):
        ds = gdal.Open(source)
    else:
        ds = source

    if(ds is None):
        raise GeoKitRasterError("Could not load input dataSource: ", str(source))
    return ds


# GDAL type mapper
_gdalIntToType = dict((v,k) for k,v in filter(lambda x: "GDT_" in x[0], gdal.__dict__.items()))
_gdalType={bool:"GDT_Byte", int:"GDT_Int32", float:"GDT_Float64","bool":"GDT_Byte", 
           "int8":"GDT_Byte", "int16":"GDT_Int16", "int32":"GDT_Int32", 
           "int64":"GDT_Int32", "uint8":"GDT_Byte", "uint16":"GDT_UInt16", 
           "uint32":"GDT_UInt32", "float32":"GDT_Float32", "float64":"GDT_Float64"}
def gdalType(s):
    """Tries to determine gdal datatype from the given input type"""
    if( isinstance(s,str) ):
        if( hasattr(gdal, s)): return s
        elif( s.lower() in _gdalType): return _gdalType[s.lower()]
        elif( hasattr(gdal, 'GDT_%s'%s)): return 'GDT_%s'%s
        elif( s == "float" or s=="int" or s=="bool" ): return gdalType( np.dtype(s) )
    
    elif( isinstance(s,int) ): return _gdalIntToType[s] # If an int is given, it's probably
                                                        #  the GDAL type indicator (and not a 
                                                        #  sample data value)
    elif( isinstance(s,np.dtype) ): return gdalType(str(s))
    elif( isinstance(s,np.generic) ): return gdalType(s.dtype)
    elif( s is bool ): return _gdalType[bool]
    elif( s is int ): return _gdalType[int]
    elif( s is float ): return _gdalType[float]
    elif( isinstance(s,Iterable) ): return gdalType( s[0] )
    raise GeoKitRasterError("GDAL type could not be determined")  

####################################################################
# Raster writer
def createRaster( bounds, output=None, pixelWidth=100, pixelHeight=100, dtype=None, srs='europe_m', compress=True, noData=None, overwrite=False, fill=None, data=None, meta=None, **kwargs):
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

    Returns:
    --------
    * If 'output' is None: gdal.Dataset
    * If 'output' is a string: None

    """
    # Check for existing file
    if(not output is None):
        if( os.path.isfile(output) ):
            if(overwrite==True):
                #print( "Removing existing raster file - " + output )
                os.remove(output)
                if(os.path.isfile(output+".aux.xml")):
                    os.remove(output+".aux.xml")
            else:
                raise GeoKitRasterError("Output file already exists: %s" %output)

    # Calculate axis information
    try: # maybe the user passed in an Extent object, test for this...
        xMin, yMin, xMax, yMax = bounds 
    except TypeError:
        xMin, yMin, xMax, yMax = bounds.xyXY
        srs = bounds.srs

    # fix origins to multiples of the resolutions
    originX = float(np.round(xMin/pixelWidth)*pixelWidth)
    originY = float(np.round(yMax/pixelHeight)*pixelHeight) # Always use the "Y-at-Top" orientation

    cols = int(round((xMax-originX)/pixelWidth)) # used 'round' instead of 'int' because this matched GDAL behavior better
    rows = int(round((originY-yMin)/abs(pixelHeight)))

    
    # Get DataType
    if( not dtype is None): # a dtype was given, use it!
        dtype = gdalType( dtype )
    elif (not data is None): # a data matrix was give, use it's dtype! (assume a numpy array or derivative)
        dtype = gdalType( data.dtype )
    else: # Otherwise, just assume we want a Byte
        dtype = "GDT_Byte"
        
    # Open the driver
    if(output is None):
        driver = gdal.GetDriverByName('Mem') # create a raster in memory
        raster = driver.Create('', cols, rows, 1, getattr(gdal,dtype))
    else:
        if (compress): opts = COMPRESSION_OPTION
        else: opts = []
            
        driver = gdal.GetDriverByName('GTiff') # Create a raster in storage
        raster = driver.Create(output, cols, rows, 1, getattr(gdal, dtype), opts)

    if(raster is None):
        raise GeoKitRasterError("Failed to create raster")

    # Do the rest in a "try" statement so that a failure wont bind the source
    try:
        raster.SetGeoTransform((originX, abs(pixelWidth), 0, originY, 0, -1*abs(pixelHeight)))
        
        # Set the SRS
        if not srs is None:
            rasterSRS = loadSRS(srs)
            raster.SetProjection( rasterSRS.ExportToWkt() )

        # Fill the raster will zeros, null values, or initial values (if given)
        band = raster.GetRasterBand(1)

        if( not noData is None):
            band.SetNoDataValue(noData)
            if fill is None and data is None:
                band.Fill(noData)

        if( data is None ):
            if fill is None:
                band.Fill(0)
            else:
                band.Fill(fill)
        else:
            # make sure dimension size is good
            if not (data.shape[0]==rows and data.shape[1]==cols):
                raise GeoKitRasterError("Raster dimensions and input data dimensions do not match")
            
            # See if data needs flipping
            if pixelHeight<0:
                data=data[::-1,:]

            # Write it!
            band.WriteArray( data )
            band.FlushCache()

            band.ComputeRasterMinMax(0)
            band.ComputeBandStats(0)
        
        raster.FlushCache()

        # Write MetaData, maybe
        if not meta is None:
            for k,v in meta.items():
                raster.SetMetadataItem(k,v)

        # Return raster if in memory
        if ( output is None): 
            return raster

        # Done
        return

    # Handle the fail case
    except Exception as e:
        raster = None
        raise e


def createRasterLike( rasterInfo, **kwargs):
    """Create a raster described by the given raster info (as returned from a 
    call to rasterInfo() ). 

    * This copies all characteristics of the given raster, including: bounds, 
      pixelWidth, pixelHeight, dtype, srs, noData, and meta. 
    * Any keyword argument which is given will override values found in the 
      rasterInfo

    """

    bounds = kwargs.pop("bounds", rasterInfo.bounds)
    pixelWidth = kwargs.pop("pixelWidth", rasterInfo.pixelWidth)
    pixelHeight = kwargs.pop("pixelHeight", rasterInfo.pixelHeight)
    dtype = kwargs.pop("dtype", rasterInfo.dtype)
    srs = kwargs.pop("srs", rasterInfo.srs)
    noData = kwargs.pop("noData", rasterInfo.noData)
    meta = kwargs.pop("meta", rasterInfo.meta)

    return createRaster( bounds=bounds, pixelWidth=pixelWidth, pixelHeight=pixelHeight, dtype=dtype, srs=srs, 
                         noData=noData, meta=meta, **kwargs)



####################################################################
# extract the raster as a matrix
def extractMatrix(source, xOff=0, yOff=0, xWin=None, yWin=None, maskBand=False ):
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
        
    xOff : int 
        The index offset in the x-dimension

    yOff : int
        The index offset in the y-dimension

    xWin: int
        The window size in the x-dimension

    yWin : int
        The window size in the y-dimension

    Returns:
    --------
    numpy.ndarray -> Two dimensional matrix

    """
    sourceDS = loadRaster(source) # BE sure we have a raster
    sourceBand = sourceDS.GetRasterBand(1) # get band
    if maskBand: mb = sourceBand.GetMaskBand()

    # set kwargs
    kwargs={}
    kwargs["xoff"] = xOff
    kwargs["yoff"] = yOff
    if not xWin is None: kwargs["win_xsize"] = xWin
    if not yWin is None: kwargs["win_ysize"] = yWin

    # get Data and check for flip
    if maskBand: data = mb.ReadAsArray(**kwargs)
    else: data = sourceBand.ReadAsArray(**kwargs)

    # make sure we are returing data in the 'flipped-y' orientation
    if not isFlipped(source): data = data[::-1,:]
    return data

# Cutline extracter
cutlineInfo = namedtuple("cutlineInfo","data info")
def extractCutline(source, geom, cropToCutline=True, **kwargs):
    """Extracts a cutout of a raster source's data which is within a geometry 

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource

    geom : ogr.Geometry
        The geometry over which to cut out the raster's data
        * Must be a Polygon or MultiPolygon

    cropToCutline : bool
        A flag which restricts the bounds of the returned matrix to that which 
        most closely matches the geometry

    **kwargs
        * All kwargs are passed on to a call to gdal.Warp
        * See gdal.WarpOptions for more details
        * For example, 'allTouched' may be useful

    Returns:
    --------
    namedtuple -> ( "data":numpy.ndarray, 
                    "info":rasterInfo for the context of the created matrix )

    """
    #### TODO: Update this function to use warp()
    # make sure we have a polygon or multipolygon geometry
    if not isinstance(geom, ogr.Geometry):
        raise GeoKitGeomError("Geom must be an OGR Geometry object")
    if not geom.GetGeometryName() in ["POLYGON","MULTIPOLYGON"]:
        raise GeoKitGeomError("Geom must be a Polygon or MultiPolygon type")

    # make geom into raster srs
    source = loadRaster(source)
    rInfo = rasterInfo(source)

    if not geom.GetSpatialReference().IsSame(rInfo.srs):
        geom.TransformTo(rInfo.srs)

    # make a quick vector dataset
    t = TemporaryDirectory()
    vecName = quickVector(geom,output=os.path.join(t.name,"tmp.shp"))
    
    # Do cutline
    cutName = os.path.join(t.name,"cut.tif")
    
    cutDS = gdal.Warp(cutName, source, cutlineDSName=vecName, cropToCutline=cropToCutline, **kwargs)
    cutDS.FlushCache()

    cutInfo = rasterInfo(cutDS)

    # Almost Done!
    returnVal = cutlineInfo(extractMatrix(cutDS), cutInfo)

    # cleanup
    t.cleanup()

    # Now Done!
    return returnVal

def rasterStats( source, cutline=None, ignoreValue=None, **kwargs):
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
        np.logical_and(rawData!= ignoreValue, sel, sel)

    if not dataInfo.noData is None:
        np.logical_and(rawData!= dataInfo.noData, sel,sel)

    # compute statistics
    data = rawData[sel].flatten()
    return describe(data)


####################################################################
# Gradient calculator
def gradient( source, mode ="total", factor=1, asMatrix=False, **kwargs):
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
    acceptable = ["total", "slope", "north-south" , "east-west", 'dir', "ew", "ns", "aspect"]
    if not ( mode in acceptable):
        raise ValueError("'mode' not understood. Must be one of: ", acceptable)

    # Get the factor
    sourceInfo = rasterInfo(source)
    if factor == "latlonToM":
        lonMid = (sourceInfo.xMax + sourceInfo.xMin)/2
        latMid = (sourceInfo.yMax + sourceInfo.yMin)/2
        R_EARTH = 6371000
        DEGtoRAD = np.pi/180

        yFactor = R_EARTH*DEGtoRAD # Get arc length in meters/Degree
        xFactor = R_EARTH*DEGtoRAD*np.cos(latMid*DEGtoRAD) # ditto...
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
        ns[1:-1,:] = (arr[2:,:] - arr[:-2,:])/(2*sourceInfo.dy*yFactor)
        if mode in ["north-south","ns"]: output=ns

    if mode in ["east-west", "total", "slope", "dir", "aspect"]:
        ew = np.zeros(arr.shape)
        ew[:,1:-1] = (arr[:,:-2] - arr[:,2:])/(2*sourceInfo.dx*xFactor)
        if mode in ["east-west","ew"]: output=ew
    
    if mode == "total" or mode == "slope":
        output = np.sqrt(ns*ns + ew*ew)

    if mode == "dir" or mode == "aspect":
        output = np.arctan2(ns,ew)

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

    if( dy<0 ): return True
    else: return False

RasterInfo = namedtuple("RasterInfo","srs dtype flipY yAtTop bounds xMin yMin xMax yMax dx dy pixelWidth pixelHeight noData, xWinSize, yWinSize, meta")
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
                    xWinSize: The width of the raster is pixels
                    yWinSize: The height of the raster is pixels
                    meta: The raster's meta data )
    """
    output = {}
    sourceDS = loadRaster(sourceDS)

    # get srs
    srs = loadSRS( sourceDS.GetProjectionRef() )
    output['srs']=srs

    # get extent and resolution
    sourceBand = sourceDS.GetRasterBand(1)
    output['dtype'] = sourceBand.DataType
    output['noData'] = sourceBand.GetNoDataValue()
    
    
    xSize = sourceBand.XSize
    ySize = sourceBand.YSize

    xOrigin, dx, trash, yOrigin, trash, dy = sourceDS.GetGeoTransform()
    
    xMin = xOrigin
    xMax = xOrigin+dx*xSize

    if( dy<0 ):
        yMax = yOrigin
        yMin = yMax+dy*ySize
        dy = -1*dy
        output["flipY"]=True
        output["yAtTop"]=True
    else:
        yMin = yOrigin
        yMax = yOrigin+dy*ySize
        output["flipY"]=False
        output["yAtTop"]=False

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

    # clean up 
    del sourceBand, sourceDS

    # return
    return RasterInfo(**output)

####################################################################
# extract specific points in a raster
ptValue = namedtuple('value',"data xOffset yOffset inBounds")
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
    pointSRS = loadSRS(pointSRS)

    # Ensure we have a list of point geometries
    try:
        if points._TYPE_KEY_ == "Location": 
            asSingle=True
            pointsKey=None
            points = [points.asGeom(info.srs), ]
        elif points._TYPE_KEY_ == "LocationSet": 
            asSingle=False
            pointsKey = points
            points = points.asGeom(info.srs)
    except AttributeError:
        pointsKey=None

        def loadPoint(pt,s):
            if isinstance(pt,ogr.Geometry):
                if pt.GetGeometryName()!="POINT":
                    raise GeoKitGeomError("Invalid geometry given")
                return pt

            if isinstance(pt, Location):
                return pt.geom
        
            tmpPt = ogr.Geometry(ogr.wkbPoint)
            tmpPt.AddPoint(*pt)
            tmpPt.AssignSpatialReference(s)
            
            return tmpPt
        
        if isinstance(points, Location) or isinstance(points, tuple) or isinstance(points, ogr.Geometry): # check for an individual point input
            asSingle = True
            points = [loadPoint( points, pointSRS), ]
        else: # assume points is iterable
            asSingle = False
            points = [loadPoint( pt, pointSRS) for pt in points]
        
        # Cast to source srs
        pointSRS = points[0].GetSpatialReference() # make sure we're using the pointSRS for the points in the list
        if not pointSRS.IsSame(info.srs): 
            points=transform(points, fromSRS=pointSRS, toSRS=info.srs)
    
    # Get x/y values as numpy arrays
    x = np.array([pt.GetX() for pt in points])
    y = np.array([pt.GetY() for pt in points])
    
    # Calculate x/y indexes
    xValues = (x-(info.xMin+0.5*info.pixelWidth))/info.pixelWidth
    xIndexes = np.round( xValues )
    xOffset = xValues-xIndexes
    
    if info.yAtTop:
        yValues = ((info.yMax-0.5*info.pixelWidth)-y)/abs(info.pixelHeight)
        yIndexes = np.round( yValues )
        yOffset = yValues-yIndexes
    else:
        yValues = (y-(info.yMin+0.5*info.pixelWidth))/info.pixelHeight
        yIndexes = np.round( yValues )
        yOffset = -1*(yValues-yIndexes)

    # Calculate the starts and window size
    xStarts = xIndexes-winRange
    yStarts = yIndexes-winRange
    window = 2*winRange+1

    inBounds = xStarts>0
    inBounds = inBounds & (yStarts>0)
    inBounds = inBounds & (xStarts+window<info.xWinSize)
    inBounds = inBounds & (yStarts+window<info.yWinSize)

    if (~inBounds).any():
        print("WARNING: One of the given points (or extraction windows) exceeds the source's limits")

    # Read values
    values = []
    band = source.GetRasterBand(1)

    for xi,yi,ib in zip(xStarts, yStarts, inBounds):
        if not ib:
            data = np.zeros((window,window))
            data[:,:] = np.nan
        else:
            # Open and read from raster
            data = band.ReadAsArray(xoff=xi, yoff=yi, win_xsize=window, win_ysize=window)

            # Look for nodata
            if not info.noData is None:
                nodata = data == info.noData
                if nodata.any():
                    if noDataOkay:
                        # data will neaed to be a float type to represent a nodata value
                        data = data.astype(np.float64)
                        data[nodata] = np.nan
                    else:
                        raise GeoKitRasterError("No data values found in extractValues with 'noDataOkay' set to False")

            # flip if not in the 'flipped-y' orientation
            if not info.yAtTop:
                data=data[::-1,:]

        if winRange==0: data=data[0][0] # If winRange is 0, theres no need to return a 2D matrix

        # Append to values
        values.append(data)

    # Done!
    if asSingle: # A single point was given, so return a single result
        if _onlyValues: return values[0]
        else: return ptValue(values[0], xOffset[0], yOffset[0], inBounds[0])
    else:
        if _onlyValues: return np.array(values)
        else: return pd.DataFrame(dict(data=values, xOffset=xOffset, yOffset=yOffset, inBounds=inBounds), index=pointsKey)

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
        asSingle=True
        points = [points, ] # make points a list of length 1 so that the rest works (will be unpacked later)
    else: # Assume points is already an iterable of some sort
        asSingle=False

    # Do interpolation
    if mode=='near':
        # Simple get the nearest value
        result = extractValues(source, points, pointSRS=pointSRS, winRange=0, _onlyValues=True)
        
    elif mode=="linear-spline": # use a spline interpolation scheme
        # setup inputs
        win = 2 if winRange is None else winRange
        x = np.linspace(-1*win,win,2*win+1)
        y = np.linspace(-1*win,win,2*win+1)

        # get raw data
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)

        # Calculate interpolated values
        result=[]
        for v in values.itertuples(index=False):
            rbs = RectBivariateSpline(y,x,v.data, kx=1, ky=1)

            result.append(rbs(v.yOffset,v.xOffset)[0][0])

    elif mode=="cubic-spline": # use a spline interpolation scheme
        # setup inputs
        win = 4 if winRange is None else winRange
        x = np.linspace(-1*win,win,2*win+1)
        y = np.linspace(-1*win,win,2*win+1)
        
        # Get raw data
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        
        # Calculate interpolated values
        result=[]
        for v in values.itertuples(index=False):
            rbs = RectBivariateSpline(y,x,v.data)

            result.append(rbs(v.yOffset,v.xOffset)[0][0])

    elif mode == "average": # Get the average in a window
        win = 3 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        result = []
        for v in values.itertuples(index=False):
            result.append(v.data.mean())

    elif mode == "func": # Use a general function processor
        if func is None:
            raise GeoKitRasterError("'func' mode chosen, but no func kwargs was given")
        win = 3 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        result = []
        for v in values.itertuples(index=False):
            result.append( func(v.data, v.xOffset, v.yOffset) )

    else:
        raise GeoKitRasterError("Interpolation mode not understood: ", mode)

    # Done!
    if asSingle: return result[0]
    else: return np.array(result)
    
####################################################################
# General raster mutator
def mutateRaster(source, processor, output=None, dtype=None, **kwargs):
    """Process all pixels in a raster according to a given function

    * Creates a gdal dataset with the resulting data
    * If the user wishes to generate an output file (by giving an 'output' input),
      then nothing will be returned to help avoid dependency issues. If no output 
      is provided, the function will return a gdal dataset for immediate use

    Parameters:
    -----------
    source : Anything acceptable by loadRaster()
        The raster datasource

    processor - function
        The function performing the mutation of the raster's data 
        * The function will take single argument (a 2D numpy.ndarray) 
        * The function must return a numpy.ndarray of the same size as the input
        * The return type must also be containable within a Float32 (int and 
          boolean is okay)
        * See example below for more info

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
    workingExtent = dsInfo.bounds

    # Read data into array
    sourceBand = workingDS.GetRasterBand(1)
    sourceData = sourceBand.ReadAsArray()

    # Perform processing
    processedData = processor( sourceData ) if processor else sourceData
    if(dtype and processedData.dtype!=dtype):
        processedData = processedData.astype(dtype)

    # Ensure returned matrix is okay
    if( processedData.shape != sourceData.shape ):
        raise GeoKitRasterError( "Processed matrix does not have the correct shape \nIs {0} \nShoud be {1}",format(rawSuitability.shape, sourceData.shape ))
    del sourceData

    # Check if flipping is required
    if not dsInfo.yAtTop:
        processedData = processedData[::-1,:]
    
    # Create an output raster
    if(output is None):
        return quickRaster( dy=dsInfo.dy, dx=dsInfo.dx, bounds=workingExtent, 
                            srs=dsInfo.srs, data=processedData, **kwargs )

    else:
        outDS = createRaster( pixelHeight=dsInfo.dy, pixelWidth=dsInfo.dx, bounds=workingExtent, 
                              srs=dsInfo.srs, data=processedData, output=output, **kwargs )

        return

# A predefined kernel processor for use in mutateRaster
def KernelProcessor(size, edgeValue=0, outputType=None, passIndex=False):
    """A decorator which automates the production of kernel processors for use 
    in mutateRaster (although it could really used for processing any matrix)
    
    Parameters:
    -----------
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
    
    Returns:
    --------
    function

    Example:
    --------
    * Say we want to make a processor which calculates the average of pixels 
      which are within a distance of 2 indcies. In other words, we want the 
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
    def wrapper1(kernel):
        def wrapper2(matrix):
            # get the original matrix sizes
            yN, xN = matrix.shape

            # make a padded version of the matrix
            paddedMatrix = np.ones((yN+2*size,xN+2*size), dtype=matrix.dtype)*edgeValue
            paddedMatrix[size:-size,size:-size] = matrix

            # apply kernel to each pixel
            output = np.zeros((yN,xN), dtype = matrix.dtype if outputType is None else outputType)
            for yi in range(yN):
                for xi in range(xN):
                    slicedMatrix = paddedMatrix[yi:2*size+yi+1, xi:2*size+xi+1]
                    
                    if passIndex: output[yi,xi] = kernel(slicedMatrix, xi=xi, yi=yi)
                    else: output[yi,xi] = kernel(slicedMatrix)

            # done!
            return output
        return wrapper2
    return wrapper1

def indexToCoord( yi, xi, source, asPoint=False):
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
    # Get source info
    if not isinstance(source, RasterInfo): source = rasterInfo(source)

    # Caclulate coordinates
    if source.yAtTop: 
        x = source.xMin+source.pixelWidth*xi
        y = source.yMax-source.pixelHeight*yi
    else:
        x = source.xMin+source.pixelWidth*xi
        y = source.yMin+source.pixelHeight*yi

    # make the output
    if asPoint: 
        try: # maybe x and y are iterable
            output = [makePoint((xx,yy),srs=source.srs) for xx,yy in zip(x,y)]
        except TypeError: # x and y should be a single point
            output = makePoint((x,y),srs=source.srs)
    else: output = x,y

    # Done!
    return output

### Raster plotter
def drawRaster(source, srs=None, ax=None, resolution=None, cutline=None, figsize=(12,12), xlim=None, ylim=None, fontsize=16, hideAxis=False, margin=(0,0,0,0), cbarPadding=0.01, cbarTitle=None, vmin=None, vmax=None, cmap="viridis", cbax=None, cbargs=None, cutlineFillValue=-9999,**kwargs):
    """Draw a matrix as an image on a matplotlib canvas

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

    hideAxis : bool; optional
        Instructs the created axis to hide its boundary
          * Only useful when generating a new axis

    margin : (float, float, float, float, ); optional
        Additional margins to add around a generated axis
          * Useful if, for whatever reason, the plot isn't fitting right in the 
            final figure
          * Before using this, try adjusting the 'figsize'

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
    if isinstance(ax, AxHands):ax = ax.ax

    if ax is None:
        newAxis=True
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        margin = margin[0],margin[1],margin[2]+0.08,margin[3]
        cbarExtraPad = 0.05
        cbarWidth = 0.04
        
        ax = plt.axes([margin[0], margin[1], 1-margin[2]-cbarWidth-cbarPadding, 1-margin[3]-margin[1]])
        cbax = plt.axes([1-margin[2]-cbarWidth, 
                         margin[1]+cbarExtraPad, 
                         cbarWidth, 
                         1-margin[3]-margin[1]-2*cbarExtraPad])

        if hideAxis: ax.axis("off")
        else: ax.tick_params(labelsize=fontsize)
    else:
        newAxis=False

    # Load the raster datasource and check for transformation
    source = loadRaster(source)
    info = rasterInfo(source)

    if not (srs is None and resolution is None and cutline is None and xlim is None and ylim is None):
        
        if xlim is None: xlim=info.xMin, info.xMax
        if ylim is None: ylim=info.yMin, info.yMax
        
        bounds = (xlim[0], ylim[0], xlim[1], ylim[1])

        if resolution is None: xres,yres = None,None
        else:
            try:    xres,yres = resolution
            except: xres,yres = resolution,resolution

        source = warp(source, cutline=cutline, pixelHeight=yres, pixelWidth=xres, srs=srs, 
                      bounds=bounds, fill=cutlineFillValue, noData=cutlineFillValue, **kwargs)

    info = rasterInfo(source)

    # Read the Data
    data = extractMatrix(source).astype(float)
    if not cutlineFillValue is None: 
        data[data==info.noData] = np.nan

    # Draw image
    ext=(info.xMin,info.xMax,info.yMin,info.yMax,)
    h = ax.imshow( data, extent=ext, vmin=vmin, vmax=vmax, cmap=cmap)

    # Draw Colorbar
    tmp = dict(cmap=cmap, orientation='vertical')
    if not cbargs is None: tmp.update( cbargs )
    
    # if cbax is None: cbar = ax.colorbar(h)
    # else: cbar = cbax.colorbar(h)

    if cbax is None:  cbar = plt.colorbar( h, ax=ax )
    else: cbar = plt.colorbar( h, cax=cbax )

    cbar.ax.tick_params(labelsize=fontsize)
    if not cbarTitle is None:
        cbar.set_label( cbarTitle , fontsize=fontsize+2 )

    # Do some formatting
    if newAxis:
        ax.set_aspect('equal')
        ax.autoscale(enable=True)

    if not xlim is None: ax.set_xlim(*xlim)
    if not ylim is None: ax.set_ylim(*ylim)

    # Done!
    return AxHands( ax, h, cbar)

#################################################################################3
# Make a geometry from a matrix mask
# 
# NOTE TO ME
# This function is VERRYYYYYYY similar to polygonizeMatrix, but since it does not
# not actually have to deal with the matrix data (which is all done by GDAL), this should be
# kept along side polygonizeMatrix
def polygonizeRaster( source, srs=None, flat=False, shrink=True):
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
        srs = loadSRS(source.GetProjectionRef())

    # Do polygonize
    vecDS = gdal.GetDriverByName("Memory").Create( '', 0, 0, 0, gdal.GDT_Unknown )
    vecLyr = vecDS.CreateLayer("mem", srs=srs)

    #vecDS = gdal.GetDriverByName("ESRI Shapefile").Create("deleteme.tif", 0, 0, 0, gdal.GDT_Unknown )
    #vecLyr = vecDS.CreateLayer("layer",srs=srs)
    
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
            smallGeomSet = geoms[rid==_rid]
            finalGeoms.append(flatten( smallGeomSet ) if len(smallGeomSet)>1 else smallGeomSet[0])
            finalRID.append(_rid)
    else:
        finalGeoms = geoms
        finalRID = rid
        
    # Cleanup
    vecLyr = None
    vecDS = None
    maskBand = None
    rasBand = None
    raster = None

    # Done!
    return pd.DataFrame(dict(geom=finalGeoms, value=finalRID))
    
def warp(source, resampleAlg='bilinear', cutline=None, output=None, pixelHeight=None, pixelWidth=None, srs=None, bounds=None, dtype=None, noData=None, fill=None, **kwargs):
    """Warps a given raster source to another context

    * Can be used to 'warp' a raster in memory to a raster on disk

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

    Returns:
    --------
    * If 'output' is None: gdal.Dataset
    * If 'output' is a string: None

    """
    # open source and get info
    source = loadRaster(source)
    dsInfo = rasterInfo(source)

    # Handle arguments
    if pixelHeight is None: pixelHeight=dsInfo.dy
    if pixelWidth is None: pixelWidth=dsInfo.dx
    if srs is None: srs=dsInfo.srs
    if bounds is None: bounds=dsInfo.bounds
    if dtype is None: dtype=dsInfo.dtype
    if noData is None: noData=dsInfo.noData

    srs = loadSRS(srs)
    dtype = gdalType(dtype)

    # Check some inputs in case they are bad (since GDAL does not give good error reports)
    if pixelHeight > (bounds[2]-bounds[0]): raise GeoKitRasterError("pixelHeight is too large compare to boundary")
    if pixelWidth > (bounds[3]-bounds[1]): raise GeoKitRasterError("pixelWidth is too large compare to boundary")

    # Test if warping is even really needed
    if ( output is None and 
         np.isclose(dsInfo.bounds[0], bounds[0]) and 
         np.isclose(dsInfo.bounds[1], bounds[1]) and 
         np.isclose(dsInfo.bounds[2], bounds[2]) and 
         np.isclose(dsInfo.bounds[3], bounds[3]) and  
         np.isclose(dsInfo.dx, pixelWidth) and 
         np.isclose(dsInfo.dy, pixelHeight) and
         dtype == dsInfo.dtype and
         noData == dsInfo.noData and 
         srs.IsSame(dsInfo.srs)):
            return source
    
    # If a cutline is given, create the output
    if not cutline is None:
        if isinstance(cutline, ogr.Geometry):
            tempdir = TemporaryDirectory()
            cutline = quickVector(cutline, output=os.path.join(tempdir.name,"tmp.shp"))

        elif not isinstance(cutline, str) and not isVector(cutline):
            raise GeoKitRasterError("cutline must be a Geometry or a path to a shape file")
        else: # cutline is already a path to a vector
            tempdir = None

    # Workflow depends on whether or not we have an output
    if not output is None: # Simply do a translate
        co = kwargs.pop("creationOptions", COMPRESSION_OPTION)
        
        copyMeta = kwargs.pop("copyMetadata", True)
        aligned = kwargs.pop("targetAlignedPixels", True)

        opts = gdal.WarpOptions( outputType=getattr(gdal,dtype), xRes=pixelWidth, yRes=pixelHeight, creationOptions=co, 
                                 outputBounds=bounds, dstSRS=srs, dstNodata=noData, resampleAlg=resampleAlg, 
                                 copyMetadata=copyMeta, targetAlignedPixels=aligned, cutlineDSName=cutline, 
                                 **kwargs)

        result = gdal.Warp( output, source, options=opts )
        if not isRaster(result): raise GeoKitRasterError("Failed to translate raster")

        destRas = None
    else:
        # Warp to a raster in memory
        destRas = quickRaster(bounds=bounds, srs=srs, dx=pixelWidth, dy=pixelHeight, dType=dtype, noData=noData, fill=fill)

        # Do a warp
        result = gdal.Warp(destRas, source, resampleAlg=resampleAlg, cutlineDSName=cutline, **kwargs)
        #print(result)
        #if( result != 0): raise GeoKitRasterError("Failed to warp raster")
        destRas.FlushCache()

    # Done!
    if not cutline is None: del tempdir
    return destRas
