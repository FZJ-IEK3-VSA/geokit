from .util import *
from .srsutil import *
from .geomutil import *

####################################################################
# INTERNAL FUNCTIONS

# Basic Loader
def loadRaster(x):
    """
    ***GeoKit INTERNAL***
    Load a raster dataset from various sources.
    """
    if(isinstance(x,str)):
        ds = gdal.Open(x)
    else:
        ds = x

    if(ds is None):
        raise GeoKitRasterError("Could not load input dataSource: ", str(x))
    return ds


# GDAL type mapper
_gdalIntToType = dict((v,k) for k,v in filter(lambda x: "GDT_" in x[0], gdal.__dict__.items()))
_gdalType={bool:"GDT_Byte", int:"GDT_Int32", float:"GDT_Float64","bool":"GDT_Byte", 
           "int8":"GDT_Byte", "int16":"GDT_Int16", "int32":"GDT_Int32", 
           "int64":"GDT_Int32", "uint8":"GDT_Byte", "uint16":"GDT_UInt16", 
           "uint32":"GDT_UInt32", "float32":"GDT_Float32", "float64":"GDT_Float64"}
def gdalType(s):
    """Try to determine gdal datatype from the given input type"""
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

# raster stat calculator
def calculateStats( source ):
    """GeoKit INTERNAL: Calculates the statistics of a raster and writes results into the raster
    * Assumes that the raster is writable
    """
    if isinstance(source,str):
        source = gdal.Open(source, 1)
    if source is None:
        raise GeoKitRasterError("Failed to open source: ", source)

    band = source.GetRasterBand(1)
    band.ComputeBandStats(0)
    band.ComputeRasterMinMax(0)


####################################################################
# Raster writer
def createRaster( bounds, output=None, pixelWidth=100, pixelHeight=100, dtype=None, srs='europe_m', compress=True, noData=None, overwrite=False, fill=None, data=None, **kwargs):
    """
    Create a raster file

    !NOTE! Raster datasets are always written in the 'yAtTop' orientation. Meaning that the first row of data values 
           (either written to or read from the dataset) will refer to the TOP of the defined boundary, and will then 
           move downward from there

    * If a data matrix is given, and a negative pixelWidth is defined, the data will be flipped automatically

    Return: None or gdal raster dataset (depending on whether an output path is given)

    Keyword inputs:
        bounds : The geographic extents spanned by the raster
            - (xMin, yMix, xMax, yMax)
            - geokit.Extent object
        
        output : A path to an output file 
            - string
            * If output is None, the raster will be created in memory and a dataset handel will be returned
            * If output is given, the raster will be written to disk and nothing will be returned
        
        pixelWidth : The pixel width of the raster in units of the input srs
            - float
            * The keyword 'dx' can be used as well and will override anything given assigned to 'pixelWidth'
        
        pixelHeight : The pixel height of the raster in units of the input srs
            - float
            * The keyword 'dy' can be used as well and will override anything given assigned to 'pixelHeight'

        dtype : The datatype of the represented by the created raster's band
            - string
            * Options are: Byte, Int16, Int32, Int64, Float32, Float64
            * If dtype is None and data is None, the assumed datatype is a 'Byte'
            * If dtype is None and data is not None, the datatype will be inferred from the given data
        
        srs : The Spatial reference system to apply to the created raster
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string
            * If 'bounds' is an Extent object, the bounds' internal srs will override the 'srs' input

        compress :  A flag instructing the output raster to use a compression algorithm
            - True/False
            * only useful if 'output' has been defined
            * "DEFLATE" used for Linux/Mac, "LZW" used for windows
        
        noData : Specifies which value should be considered as 'no data' in the created raster
            - numeric
            * Must be the same datatye as the 'dtype' input (or that which is derived)

        fill : The initial value given to all pixels in the created raster band
            - numeric
            * Must be the same datatye as the 'dtype' input (or that which is derived)

        overwrite : A flag to overwrite a pre-existing output file
            - True/False

        data : A 2D matrix to write into the resulting raster
            - np.ndarray
            * array dimensions must fit raster dimensions!!
    """

    # Fix some inputs for backwards compatibility
    pixelWidth = kwargs.pop("dx",pixelWidth)
    pixelHeight = kwargs.pop("dy",pixelHeight)
    fillValue = kwargs.pop("fillValue",fill)
    noDataValue = kwargs.pop("noDataValue",noData)

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

    cols = int(round((xMax-xMin)/pixelWidth)) # used 'round' instead of 'int' because this matched GDAL behavior better
    rows = int(round((yMax-yMin)/abs(pixelHeight)))
    originX = xMin
    originY = yMax # Always use the "Y-at-Top" orientation
    
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
        opts = []
        if (compress):
            if( "win" in sys.platform):
                opts = ["COMPRESS=LZW"]
            else:   
                opts = ["COMPRESS=DEFLATE"]
        driver = gdal.GetDriverByName('GTiff') # Create a raster in storage
        raster = driver.Create(output, cols, rows, 1, getattr(gdal, dtype), opts)

    if(raster is None):
        raise GeoKitRasterError("Failed to create raster")

    raster.SetGeoTransform((originX, abs(pixelWidth), 0, originY, 0, -1*abs(pixelHeight)))
    #raster.SetGeoTransform((originX, abs(pixelWidth), 0, originY, 0, pixelHeight))
    
    # Set the SRS
    if not srs is None:
        rasterSRS = loadSRS(srs)
        raster.SetProjection( rasterSRS.ExportToWkt() )

    # Fill the raster will zeros, null values, or initial values (if given)
    band = raster.GetRasterBand(1)

    if( not noDataValue is None):
        band.SetNoDataValue(noDataValue)
        if fillValue is None and data is None:
            band.Fill(noDataValue)

    if( data is None ):
        if fillValue is None:
            band.Fill(0)
        else:
            band.Fill(fillValue)
            #band.WriteArray( np.zeros((rows,cols))+fillValue )
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
    raster.FlushCache()

    # Return raster if in memory
    if ( output is None): 
        return raster

    # Calculate stats if data was given
    if(not data is None): 
        calculateStats(raster)
    
    return

####################################################################
# extract the raster as a matrix
def extractMatrix(source, xOff=0, yOff=0, xWin=None, yWin=None ):
    """extract all or part of a raster's band as a numpy matrix
    
    * Unless one is trying to get the entire matrix from the raster dataset, usage of this function requires 
      intimate knowledge of the raster's characteristics. In this case it probably easier to use Extent.extractMatrix

    Inputs:
        source : The raster datasource
            - str -- A path on the filesystem 
            - gdal Dataset

        xOff - int : The index offset in the x-dimension

        yOff - int : The index offset in the y-dimension

        xWin - int : The window size in the x-dimension

        yWin - int : The window size in the y-dimension
    """

    sourceDS = loadRaster(source) # BE sure we have a raster
    sourceBand = sourceDS.GetRasterBand(1) # get band

    # set kwargs
    kwargs={}
    kwargs["xoff"] = xOff
    kwargs["yoff"] = yOff
    if not xWin is None: kwargs["win_xsize"] = xWin
    if not yWin is None: kwargs["win_ysize"] = yWin

    # get Data
    return sourceBand.ReadAsArray(**kwargs)

# Cutline extracter
cutlineInfo = namedtuple("cutlineInfo","data info")
def extractCutline(source, geom, cropToCutline=True, **kwargs):
    """extract a cutout of a raster source's data which is within a given geometry 

    Inputs:
        source : The raster datasource
            - str -- A path on the filesystem 
            - gdal Dataset

        geom : The geometry overwhich to cut out the raster's data
            - ogr Geometry object
            * Must be a Polygon or MultiPolygon

        cropToCutline : A flag which restricts the bounds of the returned matrix to that which most closely matches 
                        the geometry
            - True/False

        **kwargs
            * All kwargs are passes on to a call to gdal.Warp
            * See gdal.WarpOptions for more details
            * For example, 'allTouched' may be useful

    Returns:
        ( matrix-data, a rasterInfo output in the context of the created matrix )
    """
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

def stats( source, geom=None, ignoreValue=None, **kwargs):
    """Compute basic statistics of the values contained in a raster dataset.

    * Can clip the raster with a geometry
    * Can ignore certain values if the raster doesnt have a no-data-value
    * All kwargs are passed on to 'extractCutline' when a 'geom' input is given
    """

    source = loadRaster(source)

    # Get the matrix to calculate over
    if geom is None:
        rawData = extractMatrix(source)
        dataInfo = rasterInfo(source)
    else:
        rawData, dataInfo = extractCutline(source, geom, **kwargs)

    # exclude nodata and ignore values
    sel = np.ones(rawData.shape, dtype='bool')

    if not ignoreValue is None:
        np.logical_and(rawData!= ignoreValue, sel,sel)


    if not dataInfo.noData is None:
        np.logical_and(rawData!= dataInfo.noData, sel,sel)

    # compute statistics
    data = rawData[sel].flatten()
    return describe(data)


####################################################################
# Gradient calculator
def gradient( source, mode ="total", factor=1, asMatrix=False, **kwargs):
    """Calculate a raster's gradient and return as a new dataset or simply a matrix

    Inputs:
        source : The raster datasource
            - str -- A path on the filesystem 
            - gdal Dataset

        mode : Determines the type of gradient to compute
            - str
            * Options are....
                - "total" : Calculates the absolute gradient as a ratio
                    * use arctan(...) to compute the gradient in degrees/radians
                - "slope" : Same as 'total'
                - "north-south" : Calculates the "north-facing" gradient (negative numbers indicate a south facing 
                                gradient) as a ratio (use arctan to convert to degrees/radians)
                - "ns" : Same as 'north-south'
                - "east-west" : Calculates the "east-facing" gradient (negative numbers indicate a west facing gradient) 
                              as a ratio (use arctan to convert to degrees/radians)
                - "ew" : Same as 'east-west'
                - "aspect" : calculates the gradient's direction in radians (0 is east)
                - "dir" : same as 'aspect'

        factor : The scaling factor relating the units of the x & y dimensions to the z dimension
            - float
            - str
            * If factor is 'latlonToM', the x & y units are assumed to be degrees (lat & lon) and the z units are 
              assumed to be meters. A factor is then computed for coordinates at the source's center.
            * Example: 
                - If x,y units are meters and z units are feet, factor should be 0.3048

        asMatrix : Flag which makes the returned value a matrix (as opposed to a raster dataset)
            - True/False

        **kwargs : All extra key word arguments are passed on to a final call to 'createRaster'
            * These have no effect when asMatrix is True
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
Info = namedtuple("Info","srs dtype flipY yAtTop bounds xMin yMin xMax yMax dx dy pixelWidth pixelHeight noData, xWinSize, yWinSize")
def rasterInfo(sourceDS):
    """Returns a named tuple containing information relating to the input raster

    Includes:
        srs - The spatial reference system (as an OGR object)
        dtype - The datatype (as an ????)
        flipY - A flag which indicates that the raster starts at the 'bottom' as opposed to at the 'top'
        bounds - The xMin, yMin, xMax, and yMax values as a tuple
        xMin, yMin, xMax, yMax - The individual boundary values
        pixelWidth, pixelHeight - The raster's pixelWidth and pixelHeight
        dx, dy - Shorthand names for pixelWidth (dx) and pixelHeight (dy)
        noData - The noData value used by the raster
        xWinSize - The width of the raster is pixels
        yWinSize - The height of the raster is pixels
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

    # clean up 
    del sourceBand, sourceDS

    # return
    return Info(**output)

####################################################################
# extract specific points in a raster
ptValue = namedtuple('value',"data xOffset yOffset")
def extractValues(source, points, pointSRS='latlon', winRange=0):
    """Extracts the value of a raster at a given point or collection of points. Can also extract a window of values if 
       desired

    * If the given raster is not in the 'flipped-y' orientation, the result will be automatically flipped

    Inputs:
        source : The raster datasource
            - str -- A path on the filesystem 
            - gdal Dataset

        points : Coordinates for the points to extract
            - ogr-Point -- A single OGR point-geometry object
            - (x, y) -- A single pair of coordinates
            - An iterable containing either of the above (must be one or the other)
            * All points must be in the same SRS
            * !REMEMBER! For lat and lon coordinates, X is lon and Y is lat 
              (opposite of what you may think...)

        pointSRS : The SRS to use when points are input as x & y coordinates
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string

        winRange - int : The window range (in pixels) to extract around each located point
            * A winRange of 0 will only extract the closest raster value
            * A winRange of 1 will extract a window of shape (3,3) centered around the closest raster value.
            * A winRange of 3 will extract a window of shape (7,7) centered around the closest raster value.

    Returns: A list of namedtuples consisting of (the point data, the x-offset, the y-offset )
        * If only a single point is given, a single tuple will be returned
        * If a winRange of 0 is given, the point data will just be a single value. Otherwise it will be a 2D matrix 
        * Offsets are in 'index' units
    """
    # Be sure we have a raster and srs
    source = loadRaster(source)
    info = rasterInfo(source)
    pointSRS = loadSRS(pointSRS)

    # Ensure we have a list of point geometries
    def loadPoint(pt,s):
        """GeoKit internal. Shortcut for making points"""
        if isinstance(pt,ogr.Geometry): 
            if pt.GetGeometryType()!=POINT:
                raise GeoKitGeomError("Invalid geometry given")
            return pt

        tmpPt = ogr.Geometry(ogr.wkbPoint)
        tmpPt.AddPoint(*pt)
        tmpPt.AssignSpatialReference(s)
        
        return tmpPt
    
    if isinstance(points, tuple) or isinstance(points, ogr.Geometry): # check for an individual point input
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

    if xStarts.min()<0 or yStarts.min()<0 or (xStarts.max()+window)>info.xWinSize or (yStarts.max()+window)>info.yWinSize:
        raise GeoKitRasterError("One of the given points (or extraction windows) exceeds the source's limits")

    # Read values
    values = []
    band = source.GetRasterBand(1)

    for xi,yi in zip(xStarts, yStarts):
        # Open and read from raster
        data = band.ReadAsArray(xoff=xi, yoff=yi, win_xsize=window, win_ysize=window)

        # flip if not in the 'flipped-y' orientation
        if not info.yAtTop:
            data=data[::-1,:]

        if winRange==0: data=data[0][0] # If winRange is 0, theres no need to return a 2D matrix

        # Append to values
        values.append(data)

    # Done!
    if asSingle: # A single point was given, so return a single result
        return ptValue(values[0], xOffset[0], yOffset[0])
    else:
        return [ptValue(d,xo,yo) for d,xo,yo in zip(values, xOffset, yOffset)]

####################################################################
# Shortcut for getting just the raster value
def interpolateValues(source, points, pointSRS='latlon', mode='near', func=None, winRange=None, **kwargs):
    """interpolates a single value from a raster at each point

    * Supports various interpolation schemes
        - 'near'
        - 'linear-spline'
        - 'cubic-spline'
        - 'average'
        - user-defined

    Inputs:
        source : The raster datasource
            - str -- A path on the filesystem 
            - gdal Dataset

        points : Coordinates for the points to extract
            - ogr-Point -- A single OGR point-geometry object
            - (x, y) -- A single pair of coordinates
            - An iterable containing either of the above (must be one or the other)
            * All points must be in the same SRS
            * !REMEMBER! For lat and lon coordinates, X is lon and Y is lat 
              (opposite of what you may think...)

        pointSRS : The SRS to use when points are input as x & y coordinates
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string
        
        mode - str : The interpolation scheme to use
            * options are...
                - "near" - Just gets the nearest value (this is default)
                - "linear-spline" - calculates a linear spline in between points
                - "cubic-spline" - calculates a cubic spline in between points
                - "average" - calculates average across a window
                - "func" - uses user-provided calculator
        
        func - function : A user defined interpolation function 
            * Only utilized when 'mode' equals "func"
            * The function must take three arguments in this order...
                - A 2 dimensional data matrix
                - A x-index-offset
                - A y-index-offset
            * See the example for more information

        winRange - int : Enforces a window range to extract before interpolating
            * All interpolation schemes have a predefined window range which is appropriate to their use
                - near -> 0
                - linear-spline -> 2
                - cubic-spline -> 4
                - average -> 3
                - func -> 3                
            * Useful when using "average" or "func" mode to control the window size
            * Doesn't have much (if any) effect on the other schemes
        
        Returns:
            - If a single point was given, returns a single interpolated value
            - If multiple points (or at least a list containing a single point) was given, returns an array of
              interpolated values corresponding to each of the specified points

        Example:
            - "Interpolate" according to the median value in a window or range 2

            >>> def medianFinder( data, xOff, yOff ):
            >>>     return numpy.median(data)
            >>>
            >>> result = interpolateValues( <source>, <points>, mode='func', func=medianFinder, winRange=2)
        
        """
    # Determine what the user probably wants as an output
    if isinstance(points, tuple) or isinstance(points, ogr.Geometry):
        asSingle=True
        points = [points, ] # make points a list of length 1 so that the rest works (will be unpacked later)
    else: # Assume points is already an iterable of some sort
        asSingle=False

    # Do interpolation
    if mode=='near':
        # Simple get the nearest value
        values = extractValues(source, points, pointSRS=pointSRS, winRange=0)
        if isinstance(values, list):
            result = [d for d,xo,yo in values]
        else:
            result = [values.data, ]

    elif mode=="linear-spline": # use a spline interpolation scheme
        # setup inputs
        win = 2 if winRange is None else winRange
        x = np.linspace(-1*win,win,2*win+1)
        y = np.linspace(-1*win,win,2*win+1)

        # get raw data
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)

        # Calculate interpolated values
        result=[]
        for z,xo,yo in values:
            rbs = RectBivariateSpline(y,x,z, kx=1, ky=1)

            result.append(rbs(yo,xo)[0][0])

    elif mode=="cubic-spline": # use a spline interpolation scheme
        # setup inputs
        win = 4 if winRange is None else winRange
        x = np.linspace(-1*win,win,2*win+1)
        y = np.linspace(-1*win,win,2*win+1)
        
        # Get raw data
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        
        # Calculate interpolated values
        result=[]
        for z,xo,yo in values:
            rbs = RectBivariateSpline(y,x,z)

            result.append(rbs(yo,xo)[0][0])

    elif mode == "average": # Get the average in a window
        win = 3 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        result = []
        for z,xo,yo in values:
            result.append(z.mean())

    elif mode == "func": # Use a general function processor
        if func is None:
            raise GeoKitRasterError("'func' mode chosen, but no func kwargs was given")
        win = 3 if winRange is None else winRange
        values = extractValues(source, points, pointSRS=pointSRS, winRange=win)
        result = []
        for v in values:
            result.append( func(*v) )

    else:
        raise GeoKitRasterError("Interpolation mode not understood: ", mode)

    # Done!
    if asSingle: return result[0]
    else: return np.array([r for r in result])
    
####################################################################
# General raster mutator
def mutateValues(source, processor, output=None, dtype=None, **kwargs):
    """
    Process a raster according to a given function

    * Creates a gdal dataset with the resulting data
    * If the user wishes to generate an output file (by giving an 'output' input), then nothing will be returned to 
      help avoid dependance issues. If no output is provided, however, the function will return a gdal dataset for 
      immediate use

    Inputs:
        source : The raster datasource
            - str -- A path on the filesystem 
            - gdal Dataset

        processor - function : The function performing the mutation of the raster's data 
            * The function will take single argument (a 2D numpy.ndarray) 
            * The function must return a numpy.ndarray of the same size as the input
            * The return type must also be containable within a Float32 (int and boolean is okay)
            * See example below for more info

        output - str : An optional output path for the new raster
            * Using None implies results are contained in memory
            * Not giving an output will cause the function to return a gdal dataset, otherwise it will return nothing
        
        dtype : An optional argument which forces the processed data to be a particular datatype
            - A python nnumeric type (bool, int, float)
            - A Numpy datatype

        **kwargs: 
            * All kwargs are passed on to a call to createRaster, which is generating the resulting dataset

    Example:
        If you wanted to assign suitability factors based on a raster containing integer identifiers (like in the 
        CLC dataset!)

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
    outDS = createRaster( pixelHeight=dsInfo.dy, pixelWidth=dsInfo.dx, bounds=workingExtent, 
                          srs=dsInfo.srs, data=processedData, output=output, **kwargs )

    # Done!
    if(output is None):
        if(outDS is None): raise GeoKitRasterError("Error creating temporary working raster")
        outDS.FlushCache() # just for good measure

        return outDS
    else:
        calculateStats(output)
        return

def drawImage(data, bounds=None, ax=None, scaling=None, yAtTop=True, **kwargs):
    """Draw a matrix as an image on a matplotlib canvas

    Inputs:
        data : The data to plot as a 2D matrix
            - numpy.ndarray

        bounds : The spatial context of the matrix's boundaries
            - (xMin, yMin, xMax, yMax)
            - geokit.Extent object
            * If bounds is None, the plotted matrix will be bounded by the matrix's dimension sizes

        ax : An optional matplotlib axes to plot on
            * If ax is None, the function will draw and create its own axis

        scaling - int : An optional scaling factor used to scale down the data matrix
            * Used to decrease strain on the system's resources for visualing the data
            * make sure to use a NEGATIVE integer to scale down (positive will scale up and make a larger matrix)

        yAtTop - True/False : Flag indicating that the data is in the typical y-index-starts-at-top orientation
            * If False, the data matrix will be flipped before plotting

        **kwargs : Passed on to a call to matplotlib's imshow function
            * Determines the visual characteristics of the drawn image

    """
    showPlot = False
    if ax is None:
        showPlot = True
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,12))
        ax = plt.subplot(111)

    # If bounds is none, make a boundary
    if bounds is None:
        xMin,yMin,xMax,yMax = 0,0,data.shape[1],data.shape[0] # bounds = xMin,yMin,xMax,yMax
    else:
        try:
            xMin,yMin,xMax,yMax = bounds
        except: # maybe bounds is an ExtentObject
            xMin,yMin,xMax,yMax = bounds.xyXY

    # Set extent
    extent = (xMin,xMax,yMin,yMax)
    
    # handle flipped data
    if not yAtTop: data=data[::-1,:]

    # Draw image
    if scaling: data=scaleMatrix(data,scaling,strict=False)
    h = ax.imshow( data, extent=extent, **kwargs)

    # Done!
    if showPlot:
        ax.set_aspect('equal')
        ax.autoscale(enable=True)
        plt.show()
    else:
        return h
