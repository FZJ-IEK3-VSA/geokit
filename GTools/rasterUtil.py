from .util import *

####################################################################
# INTERNAL FUNCTIONS

# Basic Loader
def loadRaster(x):
    """
    ***GIS INTERNAL***
    Load a raster dataset from various sources.
    """
    if(isinstance(x,str)):
        ds = gdal.Open(x)
    else:
        ds = x

    if(ds is None):
        raise GToolsRasterError("Could not load input dataSource: ", str(x))
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
    raise GToolsRasterError("GDAL type could not be determined")  

# raster stat calculator
def calculateStats( source ):
    """GIS INTERNAL: Calculates the statistics of a raster and writes results into the raster
    * Assumes that the raster is writable
    """
    if isinstance(source,str):
        source = gdal.Open(source, 1)
    if source is None:
        raise GToolsRasterError("Failed to open source: ", source)

    band = source.GetRasterBand(1)
    band.ComputeBandStats(0)
    band.ComputeRasterMinMax(0)


####################################################################
# Raster writer
def createRaster( bounds, output=None, pixelWidth=100, pixelHeight=100, dtype=None, srs='europe_m', compress=True, noDataValue=None, overwrite=False, fillValue=None, data=None):
    """
    Create a raster file

    NOTE! Raster datasets are always written in the 'flipped-y' orientation. Meaning that the first row of data values (either written to or read from the dataset) will refer to the TOP of the defined boundary, and will then move downward from there

        * If a data matrix is given, and a negative pixelWidth is defined, the data will be flipped automatically

    Return: None or gdal raster dataset (depending on whether an output path is given)

    Keyword inputs:
        bounds
            (xMin, yMix, xMax, yMax) -- the extents of the raster file to create
        
        output - (None) 
            string : path to the output file
            * If not defined, raster will be made in memory and a dataset will be returned
        
        pixelWidth - (100)
            float : the raster's pixel width corrsponding to the input SRS
        
        pixelHeight - (100)
            float : the raster's pixel height corrsponding to the input SRS
        
        dtype - (None) 
            str : the data type which the raster file will expect
            * Options are: Byte, Int16, Int32, Int64, Float32, Float64
        
        srs - (EPSG3035) 
            int : EPSG integer
            str : WKT string representing the desired spatial reference
            osr.SpatialReference : the raster's reference system

        compress - (True) 
            bool : use compression on the final file
            * only useful if 'output' has been defined
            * "DEFLATE" used for Linux/Mac, "LZW" used for windows
        
        noDataValue - (None)
            float : the raster's 'noData' value

        fillValue - (None) 
            float : the raster's initial values
            * None implies a no fill value of zero

        overwrite - (False) 
            bool : force overwriting the output file if it exists

        data - (None) 
            np.ndarray : A dataset to write into the final raster.
            * array dimensions must fit raster dimensions
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
                raise GToolsRasterError("Output file already exists: %s" %output)

    # Calculate axis information
    try: # maybe the user passed in an Extent object, test for this...
        xMin, yMin, xMax, yMax = bounds 
    except TypeError:
        xMin, yMin, xMax, yMax = bounds.xyXY

    cols = round((xMax-xMin)/pixelWidth) # used 'round' instead of 'int' because this matched GDAL behavior better
    rows = round((yMax-yMin)/abs(pixelHeight))
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
        raise GToolsRasterError("Failed to create raster")

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
            raise GToolsRasterError("Raster dimensions and input data dimensions do not match")
        
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
# Fetch the raster as a matrix
def rasterMatrix(source, xOff=0, yOff=0, xWin=None, yWin=None ):
    """ Fetch all or part of a raster as a numpy matrix"""

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


####################################################################
# Gradient calculator
def rasterGradient( source, mode ="total", factor=1, asMatrix=False, asDegree=False, **kwargs):
    """Calculate a raster's gradient and return as a new dataset or simply a matrix

    Inputs:
        source
            str : A path to the raster dataset
            gdal.Dataset : A previously opened gdal raster dataset

        mode - "total"
            str : The mode to use when calculating
                * Options are....
                    "total" - Calculates the absolute gradient
                    "slope" - Same as 'total'
                    "north-south" - Calculates the "north-facing" gradient (negative numbers indicate a south facing gradient)
                    "ns" - Same as 'north-south'
                    "east-west" - Calculates the "east-facing" gradient (negative numbers indicate a west facing gradient)
                    "ew" - Same as 'east-west'
                    "dir" - calculates the gradient's direction

        !!!!!!!!!!FILL IN THE REST LATER!!!!!!!
    """
    # Make sure source is a source
    source = loadRaster(source)

    # Check mode
    acceptable = ["total", "slope", "north-south" , "east-west", 'dir', "ew", "ns"]
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
    arr = rasterMatrix(source)
    
    if mode in ["north-south", "ns", "total", "slope", "dir"]:
        ns = np.zeros(arr.shape)
        ns[1:-1,:] = (arr[2:,:] - arr[:-2,:])/(2*sourceInfo.dy*yFactor)
        if mode in ["north-south","ns"]: output=ns

    if mode in ["east-west", "total", "slope", "dir"]:
        ew = np.zeros(arr.shape)
        ew[:,1:-1] = (arr[:,:-2] - arr[:,2:])/(2*sourceInfo.dx*xFactor)
        if mode in ["east-west","ew"]: output=ew
    
    if mode == "total" or mode == "slope":
        output = np.sqrt(ns*ns + ew*ew)

    if mode == "dir":
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
    """Returns a named tuple of the input raster's information.
    

    Includes:
        srs - The spatial reference system (as an OGR object)
        dtype - The datatype (as an ????)
        flipY - A flag which indicates that the raster starts at the 'bottom' as opposed to at the 'top'
        bounds - The xMin, yMin, xMax, and yMax values as a tuple
        xMin, yMin, xMax, yMax - The individual boundary values
        pixelWidth, pixelHeight - The raster's pixelWidth and pixelHeight
        dx, dy - Shorthand names for pixelWidth (dx) and pixelHeight (dy)
        noData - The noData value used by the raster
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

def describeRaster(x): 
    print("CHANGE CALL TO 'rasterInfo'")
    return rasterInfo(x)

####################################################################
# Fetch specific points in a raster
def rasterValues(source, points, winRange=0):
    """Extracts the value of a raster a a given point or collection of points. Can also extract a window of values if desired

    * If the given raster is not in the 'flipped-y' orientation, the result will be automatically flipped

    Returns a tuple consisting of: list of value-windows at each of the given points : a list of x and y offsets from the located index
        * If only a single point is given, it will still need to be accessed as the first element of the returned list
        * If a winRange of 0 is given, the actual value will still need to be accessed as the first column of the first row in the value-window
        * Offsets are in 'index' units

    Inputs:
        source
            str -- Path to an input shapefile
            * either source or wkt must be provided

        points
            [ogr-Point, ] -- An array of OGR point-geometry objects
            ogr-Point -- A single OGR point-geometry object
            (lat, lon) -- A single lattitude/longitude pair
            [ (lat,lon), ] -- A list of latitude longitude pairs

        winRange - (0)
            int -- The number of raster pixels to include around the located points
            * Result extends 'winRange' pixes in all directions, so a winRange of 0 will result in a returned a window of shape (1,1), a winRange of 1 will result in a returned window of shape (3,3), and so on...
    """
    # Be sure we have a raster
    source = loadRaster(source)
    info = rasterInfo(source)

    # See if point is a point geometry array, if not make it into one
    if isinstance(points, ogr.Geometry):
        points = [points, ]
    elif not isinstance(points[0], ogr.Geometry):
        try:
            points = [pointGeom(points[0], points[1], EPSG4326),]
        except:
            points = [pointGeom(pt[0], pt[1], EPSG4326) for pt in points]
    pointSRS = points[0].GetSpatialReference()

    # Cast to source srs
    if not pointSRS.IsSame(info.srs):
        trx = osr.CoordinateTransformation(points[0].GetSpatialReference(), info.srs)
        for pt in points:
            pt.Transform(trx)
    
    # Get x/y values as numpy arrays
    x = np.array([pt.GetX() for pt in points])
    y = np.array([pt.GetY() for pt in points])
    
    # Calculate x/y indexes
    xValues = (x-(info.xMin+0.5*info.pixelWidth))/info.pixelWidth
    xIndexes = np.round( xValues )
    xOffset = xValues-xIndexes
    
    if info.flipY:
        yValues = ((info.yMax-0.5*info.pixelWidth)-y)/abs(info.pixelHeight)
        yIndexes = np.round( yValues )
        yOffset = yValues-yIndexes
    else:
        yValues = (y-(info.yMin+0.5*info.pixelWidth))/info.pixelHeight
        yIndexes = np.round( yValues )
        yOffset = -1*(yValues-yIndexes)
    

    offsets = list(zip(xOffset,yOffset))

    # Calculate the starts and window size
    xStarts = xIndexes-winRange
    yStarts = yIndexes-winRange
    window = 2*winRange+1

    if xStarts.min()<0 or yStarts.min()<0 or (xStarts.max()+window)>info.xWinSize or (yStarts.max()+window)>info.yWinSize:
        raise GToolsRasterError("One of the given points (or extraction windows) exceeds the source's limits")

    # Read values
    values = []
    band = source.GetRasterBand(1)

    for xi,yi in zip(xStarts, yStarts):
        # Open and read from raster
        data = band.ReadAsArray(xoff=xi, yoff=yi, win_xsize=window, win_ysize=window)

        # flip if not in the 'flipped-y' orientation
        if not info.flipY:
            data=data[::-1,:]

        # Append to values
        values.append(data)

    # Done!
    return values, offsets

####################################################################
# Shortcut for getting a single point value
def rasterValue(source, point):
    """Convenience wrapper for 'rasterValues' to simply get the closest raster value at the given point"""
    return rasterValues(source, point)[0][0][0]

####################################################################
# General raster mutator
def rasterMutate(source, extent=None, processor=None, output=None, dtype=None, **kwargs):
    """
    Process a raster according to a given function

    Returns or creates a gdal dataset with the resulting data

    * If the user wishes to generate an output file (by giving an 'output' input), then nothing will be returned to help avoid dependance issues. If no output is provided, however, the function will return a gdal dataset for immediate use

    Inputs:
        source 
            str -- The path to the raster to processes
            gdal.Dataset -- The input data source as a gdal dataset object

        extent: (None)
            Extent Object -- A geographic extent to clip the source dataset to before processing
            * If the Extent is not in the same SRS as the source's SRS, then it will be cast to the source's SRS, resulting in a new extent which will be >= the original extent
            
        processor: (None) 
            func -- A function for processing the source data
            * The function will take single argument (a 2D numpy.ndarray) 
            * The function must return a numpy.ndarray of the same size as the input
            * The return type must also be containable within a Float32 (int and boolean is okay)
            * See example below for more info

        output: (None)
            str -- A path to a resulting output raster
            * Using None implies results are contained in memory
            * Not giving an output will cause the function to return a gdal dataset, otherwise it will return nothing
        
        kwargs: 
            * All kwargs are passed on to a call to createRaster, which is generating the resulting dataset
            * Do not provide the following inputs since they are defined in the function:
                - pixelHeight
                - pixelWidth
                - bounds
                - srs
                - data

    Example:
        If you wanted to assign suitability factors based on integer identifiers (like in the CLC dataset!)

        def calcSuitability( data ):
            # create an ouptut matrix
            outputMatrix = numpy.zeros( data.shape )

            # do the processing
            outputMatrix[ data == 1 ] = 0.1
            outputMatrix[ data == 2 ] = 0.2 
            outputMatrix[ data == 10] = 0.4
            outputMatrix[ np.logical_and(data > 15, data < 20)  ] = 0.5

            # return the output matrix
            return outputMatrix

        result = processRaster( <source-path>, processor=calcSuitability )
    """
    # open the dataset and get SRS
    rasDS = loadRaster(source)

    # Clip input dataset to the region (if required)
    workingDS = extent.clipRaster(rasDS) if extent else rasDS

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
        raise GToolsRasterError( "Processed matrix does not have the correct shape \nIs {0} \nShoud be {1}",format(rawSuitability.shape, sourceData.shape ))
    del sourceData

    # Create an output raster
    outDS = createRaster( pixelHeight=dsInfo.dy, pixelWidth=dsInfo.dx, bounds=workingExtent, 
                          srs=dsInfo.srs, data=processedData, output=output, **kwargs )

    # Done!
    if(output is None):
        if(outDS is None): raise GToolsRasterError("Error creating temporary working raster")
        outDS.FlushCache() # just for good measure

        return outDS
    else:
        calculateStats(output)
        return

