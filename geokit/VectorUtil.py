from .util import *
from .rasterUtil import *

####################################################################
# INTERNAL FUNCTIONS

# Loaders Functions
def loadVector(x, stringOnly=False):
    """
    ***GIS INTERNAL***
    Load a vector dataset from various sources.
    """
    if(isinstance(x,str)):
        ds = gdal.OpenEx(x)
    else:
        ds = x
    
    if(ds is None):
        raise GeoKitVectorError("Could not load input dataSource: ", str(x))
    return ds

# Feature looper
def loopFeatures(source):
    """Loops over an input layer's features

      * Will reset the reading counter before looping is initiated
    """
    if(isinstance(source,str)): # assume input source is a path to a datasource
        ds = ogr.Open(source)
        layer = ds.GetLayer()
    else: # otherwise, assume input source is an ogr layer object
        # loop over all features
        layer = source
        layer.ResetReading()

    while(True):
        ftr = layer.GetNextFeature()
        if ftr: yield ftr
        else: return

# OGR type map
_ogrIntToType = dict((v,k) for k,v in filter(lambda x: "OFT" in x[0], gdal.__dict__.items()))
_ogrStrToType = {"bool":"OFTInteger", "int8":"OFTInteger", "int16":"OFTInteger", 
                 "int32":"OFTInteger", "int64":"OFTInteger64", "uint8":"OFTInteger", 
                 "uint16":"OFTInteger", "uint32":"OFTInteger", "float32":"OFTReal", 
                 "float64":"OFTReal", "string":"OFTString", "Object":"OFTString"}

def ogrType(s):
    """Tries to determine the corresponding OGR type according to the input"""

    if( isinstance(s,str) ): 
        if( hasattr(ogr, s)): return s
        elif( s.lower() in _ogrStrToType): return _ogrStrToType[s.lower()]
        elif( hasattr(ogr, 'OFT%s'%s)): return 'OFT%s'%s
        return "OFTString"

    elif( s is str ): return "OFTString"
    elif( isinstance(s,np.dtype) ): return ogrType(str(s))
    elif( isinstance(s,np.generic) ): return ogrType(s.dtype)
    elif( s is bool ): return "OFTInteger"
    elif( s is int): return "OFTInteger64"
    elif( isinstance(s,int) ): return _ogrIntToType[s]
    elif( s is float ): return "OFTReal"
    elif( isinstance(s,Iterable) ): return ogrType(s[0])

    raise ValueError("OGR type could not be determined")


####################################################################
# Geometry convenience functions

def makePoint(x,y,srs='latlon'):
    """make a point geometry from given coordinates (x,y) and srs"""
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(x,y)
    pt.AssignSpatialReference(loadSRS(srs))
    return pt

def makeBox(xMin,yMin,xMax,yMax, srs=None):
    """Make an ogr polygon object from extents"""

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

def makeEmptyGeom(name, srs=None):
    """
    name options: Point, MultiPoint, Line, MultiLine, Polygon, MultiPolygon, ect...
    """
    if not hasattr(ogr,"wkb"+name):
        raise GeoKitVectorError("Could not find geometry named: "+name)
    geom = ogr.Geometry(getattr(ogr,"wkb"+name))

    if not srs is None:
        geom.AssignSpatialReference(loadSRS(srs))

    return geom

def makeGeomFromWkt( wkt, srs=None):
    """Make a geometry from a WKT string"""
    geom = ogr.CreateGeometryFromWkt( wkt ) # Create new geometry from string 
    if not isinstance(geom,ogr.Geometry): # test for success
        raise GeoKitVectorError("Failed to create geometry")
    if srs:
        srs = loadSRS(srs)
        geom.AssignSpatialReference(srs) # Assign the given srs
    return geom

#################################################################################3
# Make a geometry from a matrix mask
def makeGeomFromMask( mask, bounds=None, srs=None, flatten=False):
    """Create a geometry from a matrix mask"""
    # Make sure mask is bool type
    if not mask.dtype == np.bool:
        raise ValueError("mask must be of bool type to be turned into a geometry")

    # Make boundaries if not given
    if bounds is None:
        bounds = (0,0,mask.shape[1], mask.shape[0]) # bounds in xMin, yMin, xMax, yMax
        pixelHeight = 1
        pixelWidth  = 1

    else:
        try: # first try for a tuple
            xMin, yMin, xMax, yMax = bounds
        except: # next assume the user gave an extent object
            try:
                xMin, yMin, xMax, yMax = bounds.xyXY
                srs = bounds.srs
            except:
                raise GeoKitVectorError("Could not underdtane 'bounds' input")


        pixelHeight = (yMax-yMin)/mask.shape[0]
        pixelWidth  = (xMax-xMin)/mask.shape[1]

    if not srs is None: srs=loadSRS(srs)

    # Make a raster dataset and pull the band/maskBand objects
    rasDS = createRaster(bounds=bounds, data=mask, noDataValue=0, pixelWidth=pixelWidth, pixelHeight=pixelHeight, srs=srs)

    rasBand = rasDS.GetRasterBand(1)
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
        raise GeoKitVectorError("Failed to polygonize geometry")

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
    elif not flatten:
        final = [ftr.GetGeometryRef().Clone() for ftr in loopFeatures(vecLyr)]
    else: # Flatten is true, so combine all features into a single geometry
        final = geomListFlatten([ftr.GetGeometryRef().Clone() for ftr in loopFeatures(vecLyr)])
        
    # Cleanup
    vecLyr = None
    vecDS = None
    maskBand = None
    rasBand = None
    rasDS = None

    return final

# geometry transformer
def geomTransform( geom, fromSRS='europe_m', toSRS='latlon'):
    # load srs's
    fromSRS = loadSRS(fromSRS)
    toSRS = loadSRS(toSRS)

    # make a transformer
    trx = osr.CoordinateTransformation(fromSRS, toSRS)

    # Do transformation
    if isinstance(geom, ogr.Geometry):
        geom = geom.Clone()
        r = geom.Transform(trx)
        print("CHECK ME FOR ERROR: ",r)
    else: # assume geom is iterable
        geom = [g.Clone(trx) for g in geom]
        r = [g.Transform(trx) for g in geom]
        print("CHECK ME FOR ERRORS: ",r)
        
    # Done!
    return geom

#################################################################################3
# Flatten a list of geometries
def geomListFlatten( geoms ):
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



####################################################################
# geometry transformer
def coordTransform(x, y, fromSRS='europe_m', toSRS='latlon'):
    # load srs's
    fromSRS = loadSRS(fromSRS)
    toSRS = loadSRS(toSRS)

    # make a transformer
    trx = osr.CoordinateTransformation(fromSRS, toSRS)

    # Do transformation
    try:
        out = trx.TransformPoint(x,y)
    except NotImplementedError:
        out = trx.TransformPoints(list(zip(x,y)))
        
    # Done!
    return out


####################################################################
# Vector feature count
def vectorCount(source):
    ds = loadVector(source)
    layer = ds.GetLayer()
    return layer.GetFeatureCount()


####################################################################
# Vector feature count
vecInfo = namedtuple("vecInfo","srs bounds xMin yMin xMax yMax count attributes")
def vectorInfo(source):
    info = {}

    vecDS = loadVector(source)
    vecLyr = vecDS.GetLayer()
    info["srs"] = vecLyr.GetSpatialRef()
    
    xMin, xMax, yMin, yMax = vecLyr.GetExtent()
    info["bounds"] = (xMin, yMin, xMax, yMax)
    info["xMin"] = xMin
    info["xMax"] = xMax
    info["yMin"] = yMin
    info["yMax"] = yMax

    info["count"] = vecLay.GetFeatureCount()

    info["attributes"] = []
    layerDef = vecLyr.GetLayerDefn()
    for i in range(layerDef.GetFieldCount()):
        info["attributes"].append(layerDef.GetFieldDefn(i).GetName())

    return vecInfo(**info)

####################################################################
# Iterable to loop over vector items
def vectorItems(source, geom=None, where=None, outputSRS=None):
    """Loops over an input vector sources's features

    !!! WARN ABOUT FEATURE MAY NOT BE IN THE REGION (JUST THE REGION EXTENT) !!!
    !!! HANDEL THIS OUTSIDE OF THE FUNC

    * Iteravely returns (feature-geometry, feature-fields)
    """

    ds = loadVector(source)
    layer = ds.GetLayer()

    trx=None
    if(not outputSRS is None):
        lyrSRS = layer.GetSpatialRef()
        if (not lyrSRS.IsSame(outputSRS)):
            trx = osr.CoordinateTransformation(lyrSRS, outputSRS)
    
    if (not geom is None):
        if(geom.GetSpatialReference() is None):
            raise RuntimeError("Input geom must have a srs")
        if(not geom.GetSpatialReference().IsSame(layer.GetSpatialRef())):
            geom = geom.Clone()
            geom.TransformTo(layer.GetSpatialRef())
            #trx = osr.CoordinateTransformation(layer.GetSpatialRef(), geom.GetSpatialReference())
        layer.SetSpatialFilter(geom)

    if(not where is None):
        r = layer.SetAttributeFilter(where)
        if( r!=0): raise RuntimeError("Error applying where statement")

    for ftr in loopFeatures(layer):
        oGeom = ftr.GetGeometryRef().Clone()
        if ( not trx is None): oGeom.Transform(trx)
        oItems = ftr.items().copy()

        yield (oGeom, oItems)

def vectorItem(**kwargs):
    """convenience function to get a single geometry from a source"""
    getter = vectorItems(**wargs)

    # Get first result
    geom,attr = next(getter)

    # try to get a second result
    s = nect(getter)
    if not s is None:
        raise GeoKitVectorError("More than one feature found")

    # Done!
    return geom,attr    

####################################################################
# Create a vector
def createVector( geoms, output=None, srs=None, fieldVals=None, fieldDef=None, overwrite=False):
    """
    Create a vector file from a list of geometries, optionally including field values 

    keyword inputs:
        geoms
            [ogr-geometry,] : A list of ogr Geometry objects
                * All geometries must share the same type (point, line, polygon, ect...)
                * All geometries must share the same SRS
                * If geometry SRS differ from an input srs, then all geometries will be 
                  projected to the input srs
            [str,] : a list of WKT strings
                * All geometries must share the same type (point, line, polygon, ect...)
                * SRS must be provided via the srs input

        output - None
            str : The path to the output file
                * Assumed to be of "ESRI Shapefile" format
                * Will create a number of files with different extensions

        srs - None
            int : The SRS to use as an EPSG integer
            str : The SRS to use as a WKT string
            osr.SpatialReference : The SRS to use

        fieldVals - None
            pandas-DataFrame : A DataFrame of the field names(taken from column names) and 
                values corresponding to the geometries
            dict : A dictionry of the field names (taken from keys) and and associated lists of the geometry's field values
        
        fieldDef - {}
            dict : Mapping of field names to data types to write into file
            * Options are defined from ogr.OFT[...]
              - ex. Integer, Real, String

        overwrite - False
            bool : Flag to overwrite preexisting file
    """
    if(srs): srs = loadSRS(srs)

    # Search for file
    if(output):
        exists = os.path.isfile(output)
        if ( exists and overwrite ): 
            os.remove(output)

    # make geom or wkt list into a list of ogr.Geometry objects
    finalGeoms = []
    if ( isinstance(geoms, str) or isinstance(geoms, ogr.Geometry)): # geoms is a single geometry
        geoms = [geoms,]

    if( isinstance(geoms[0], ogr.Geometry) ): # Test if the first geometry is an ogr-Geometry type
                                              #  (Assume all geometries in the array have the same type)
        geomSRS = geoms[0].GetSpatialReference()
        
        # Set up some variables for the upcoming loop
        doTransform = False
        setSRS = False

        if( srs and geomSRS and not srs.IsSame(geomSRS)): # Test if a transformation is needed
            trx = osr.CoordinateTransformation(geomSRS, srs)
            doTransform = True
        elif( srs is None and geomSRS): # Test if an srs was NOT given, but the incoming geometries have an SRS already
            srs = geomSRS
        elif( srs and geomSRS is None): # Test if an srs WAS given, but the incoming geometries do NOT have an srs already
                                        # In which case, assume the geometries are meant to have the given srs
            setSRS = True

        # Create geoms
        for i in range(len(geoms)):
            finalGeoms.append( geoms[i].Clone() ) # clone just incase the geometry is tethered outside of function
            if(doTransform): finalGeoms[i].Transform( trx ) # Do transform if necessary
            if(setSRS): finalGeoms[i].AssignSpatialReference(srs) # Set srs if necessary

    elif( isinstance(geoms[0], str)): # Otherwise, the geometry array should contain only WKT strings
        if( srs is None):
            raise ValueError("srs must be given when passing wkt strings")

        # Create geoms
        finalGeoms = [makeGeomFromWkt(wkt,srs) for wkt in geoms]

    else:
        raise ValueError("Geometry inputs must be ogr.Geometry objects or WKT strings")
        
    geoms = finalGeoms # Overwrite geoms array with the conditioned finalGeoms
    
    # Determine geometry types
    types = set()
    for g in geoms:
        types.add(g.GetGeometryName()) # Get a set of all geometry type-names (POINT, POLYGON, ect...)
    
    if( types.issubset({'POINT','MULTIPOINT'})): 
        geomType = ogr.wkbPoint
    elif( types.issubset({'LINESTRING','MULTILINESTRING'})): 
        geomType = ogr.wkbLineString
    elif( types.issubset({'POLYGON','MULTIPOLYGON'})):  
        geomType = ogr.wkbPolygon
    else: 
        #geomType = ogr.wkbGeometryCollection
        raise RuntimeError("Could not determine output shape's geometry type")

    # Create a driver and datasource
    #driver = ogr.GetDriverByName("ESRI Shapefile")
    #dataSource = driver.CreateDataSource( output )
    if output:
        driver = gdal.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Create( output, 0,0 )
    else:
        driver = gdal.GetDriverByName("Memory")
        
        # Using 'Create' from a Memory driver leads to an error. But creating
        #  a temporary shapefile driver (it doesnt actually produce a file, I think)
        #  and then using 'CreateCopy' seems to work
        tmp_driver = gdal.GetDriverByName("ESRI Shapefile")
        t = TemporaryDirectory()
        tmp_dataSource = tmp_driver.Create( t.name+"tmp.shp", 0, 0 )

        dataSource = driver.CreateCopy("MEMORY", tmp_dataSource)
        t.cleanup()
        del tmp_dataSource, tmp_driver, t


    # Create the layer
    if(output): layerName = os.path.splitext(os.path.basename(output))[0]
    else: layerName="Layer"

    layer = dataSource.CreateLayer( layerName, srs, geomType )

    # Setup fieldVals and fieldDef dicts
    if( not fieldVals is None):
        # Ensure fieldVals is a dataframe
        if( not isinstance(fieldVals, pd.DataFrame)):
            fieldVals = pd.DataFrame(fieldVals) # If not, try converting it
        
        # check if length is good
        if(fieldVals.shape[0]!=len(geoms)):
            raise GeoKitVectorError("Values table length does not equal geometry count")

        # Ensure fieldDefs exists and is a dict
        if( fieldDef is None): # Try to determine data types
            fieldDef = {}
            for k,v in fieldVals.items():
                fieldDef[k] = ogrType(v.dtype)

        elif( isinstance(fieldDef, dict) ): # Assume fieldDef is a dict mapping fields to inappropriate                                             
                                            #   datatypes. Fix these to ogr indicators!
            for k in fieldVals.columns:
                try:
                    fieldDef[k] = ogrType(fieldDef[k])
                except KeyError as e:
                    print("\"%s\" not in attribute definition table"%k)
                    raise e

        else: # Assume a single data type was given. Apply to all fields
            _type = ogrType(fieldDef)
            fieldDef = {}
            for k in fieldVals.keys():
                fieldDef[k] = _type

        # Write field definitions to layer
        for fieldName, dtype in fieldDef.items():
            layer.CreateField(ogr.FieldDefn(fieldName, getattr(ogr,dtype)))

        # Ensure list lengths match geom length
        for k,v in fieldVals.items():
            if( len(v) != len(geoms) ): raise RuntimeError("'{}' length does not match geom list".format(k))      

    # Create features
    for gi in range(len(geoms)):
        # Create a blank feature
        feature = ogr.Feature(layer.GetLayerDefn())
        
        # Fill the attributes, if required
        if( not fieldVals is None ):
            for fieldName, value in fieldVals.items():
                _type = fieldDef[fieldName]

                # cast to basic type
                if( _type == "OFTString" ): 
                    val = str(value[gi])
                elif( _type == "OFTInteger" or _type == "OFTInteger64" ): val = int(value[gi])
                else: val = float(value[gi]) 
                
                # Write to feature                
                feature.SetField(fieldName, val)

        # Set the Geometry
        feature.SetGeometry( geoms[gi] )

        # Create the feature
        layer.CreateFeature( feature )
        feature.Destroy() # Free resources (probably not necessary here)

    # Finish
    if( output ): return
    else: return dataSource


####################################################################
# mutuate a vector
def vectorMutate(source, extent=None, processor=None, where=None, srs=None, fieldTypes=None, output=None, **kwargs):
    """Process a vector dataset according to a given function

    Returns or creates an ogr dataset with the resulting data

    * If the user wishes to generate an output file (by giving an 'output' input), then nothing will be returned to help avoid dependance issues. If no output is provided, however, the function will return a dataset for immediate use

    Inputs:
        source 
            str -- The path to the raster to processes
            gdal.Dataset -- The input data source as a gdal dataset object

        extent: (None)
            Extent Object -- A geographic extent to clip the source around to before processing
            * If the Extent is not in the same SRS as the source's SRS, then it will be cast to the source's SRS, resulting in a new extent which will be >= the original extent
            * Uses the "SetSpatialFilterRect" method which means all features which extent into (or are contained within) this rectangle will be included

        processor: (None) 
            func -- A function for processing the source data
            * If none is given, feature will will be "processed" by simply returning themselves
            * The function will take 2 arguments: an ogr.Geometry object and an attribute dictionary 
            * The function must return a tuple containing the geometry and an attribute dictionary
                - The returned geometry can be an ogr.Geometry object or a WKT string
                - If the user wishes to process the objects using shapely, the input geometries should first be cast to WKT strings (using <geometry>.ExportToWkt()) before being used to initialize a shapely object. After processing, the shapely objects should be cast back into WKT strings
                - If a WKT string is returned, it is assumed to be in the same srs as the "srs" input provided with this function
            * The attribute dictionary should always contain the same names for all features
            * The attribute dictionary's values should only be numeric types and strings
            * See example below for more info

        where: (None)
            str -- the "where" statement used to filter features by attribute
            * follows the SQL-where syntax

        srs: (None)
            skt-string, osr.SpatialReference, EPSG-int -- The spatial referenece to use while processing and in the final dataset
            * If no SRS is given, use the source's default SRS
            * If the given SRS is different from the source's SRS, all feature geometries will be cast to the given SRS before processing

        fieldTypes: (None)
            dict -- The datatypes to use for the calculated attributes
            * Use this to control what datatype the fields are written as into the final dataset
            * Pattern is: dict(<field-name>=<datatype>)
            * Can use numpy types, native python types, strings, or ogr identifiers

        output: (None)
            str -- A path to a resulting output vector
            * Using None implies results are contained in memory
            * Not giving an output will cause the function to return a gdal dataset, otherwise it will return nothing

        kwargs: 
            * All kwargs are passed on to a call to createVector, which is generating the resulting dataset
            * Do not provide the following inputs since they are defined in the function:
                - fieldDef

    Example:
        Say you had a vector source which containes point geometries, where each feature also had an float-attribute called "value". If you wanted to create a new vector set wherein you have circle geometries at the same locations as the original points and whose radius is equal to the original features' "value" attribute. Furthermore, let's say you only want to do this for feature's who's "value" attribute is greater than zero and less than 10. Do as such:

        def growPoints( geom, attr ):
            # Create a new geom
            newGeom = geom.Buffer(attr["radius"])

            # Return the new geometry/attribute set
            return (newGeom, attr)

        result = processVector( <source-path>, where="value>0 AND value<10", processor=growPoints )
    """

    # Open source
    vecDS = loadVector(source)
    vecLyr = vecDS.GetLayer()
    vecSRS = vecLyr.GetSpatialRef()
    if(vecSRS is None): raise GeoKitVectorError("Could not determine source SRS")

    # Apply filters to source
    if(extent):
        vecLyr.SetSpatialFilterRect(*extent.castTo(vecSRS).xyXY)

    if(not where is None):
        r = vecLyr.SetAttributeFilter(where)
        if( r!=0): raise GeoKitVectorError("Error applying where statement")

    # TEST THE FEATURES!!!!
    if( vecLyr.GetFeatureCount()==0 ): 
        print("No geometries found, returning None")
        return None #No geometries found!!

    # Sometimes the filtering seems to fail in an odd way when we're applying filters 
    #  which return no geometries. layer.GetFeatureCount() will return something above 0,
    #  but calling layer.GetNextFeature() will return None. So, we will just do a test for 
    #  this, too...
    if( vecLyr.GetNextFeature() is None ): 
        print("No geometries found, returning None")
        return None #No geometries found!!
    vecLyr.ResetReading()
    
    # See if a projection to a working srs is necessary
    if( srs is None):
        srs = vecSRS
    else:
        srs=loadSRS(srs)

    if( vecSRS.IsSame(srs) ):
        projectionReq = False
    else:
        projectionReq = True
        workingTrx = osr.CoordinateTransformation(vecSRS, srs)

    # process leftover features
    geoms = []
    values = {}
    if processor is None: # if no processor, simply return the geometry and items
        processor = lambda g,i: (g,i)

    for ftr in loopFeatures(vecLyr):
        _g = ftr.GetGeometryRef().Clone()

        # Project geom, maybe?
        if(projectionReq):
            _g.Transform(workingTrx)

        # apply processor
        r = processor( _g, ftr.items() )
        if(r is None): continue # if processor returns None, skip

        # check for good values
        goodOutput = True
        try:
            g,v = r
            if( not (isinstance(g, str) or isinstance(g, ogr.Geometry))):
                goodOutput = False
            
            if( not isinstance(v, dict)):
                v = dict(value=v)

            for _k,_v in v.items():
                if( not (isinstance(_v,int) or isinstance(_v,float) or isinstance(_v,str))):
                    print("Invalid field value found: " + str(_k) + " - " + str(_v))
                    goodOutput = False

        except TypeError: 
            goodOutput = False

        if (not goodOutput):
            raise GeoKitVectorError( "Error encountered while processing")

        # Look if wkt was returned. If so, convert to geom
        if( isinstance(g, str) ):
            g = ogr.CreateGeometryFromWkt( g )
            g.AssignSpatialReference( srs )

        # Append
        geoms.append(g)
        for k,v in v.items():
            if(k in values):
                values[k].append(v)
            else:
                values[k] = [v,]

    if( len(geoms)==0 ):
        raise GeoKitVectorError("Invalid geometry count")

    # Create a new shapefile from the results 
    newDS = createVector( geoms, srs=srs, fieldVals=values, fieldDef=fieldTypes, output=output, **kwargs )

    if not output is None: 
        return
    else:
        if(newDS is None):
            raise GeoKitVectorError("Failed to create working shapefile")
        return newDS



####################################################################
# 



####################################################################
# 
