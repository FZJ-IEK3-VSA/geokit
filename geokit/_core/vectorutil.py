from .util import *
from .srsutil import *
from .geomutil import *
from .rasterutil import *

####################################################################
# INTERNAL FUNCTIONS

# Loaders Functions
def loadVector(x, stringOnly=False):
    """
    ***GeoKit INTERNAL***
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
    """Geokit internal

    *Loops over an input layer's features
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

def filterLayer(layer, geom=None, where=None):
    """GeoKit internal

    Filters an ogr Layer object accordint to a geometry and where statement
    """
    if (not geom is None):
        if isinstance(geom,ogr.Geometry):
            if(geom.GetSpatialReference() is None):
                raise GeoKitVectorError("Input geom must have a srs")
            if(not geom.GetSpatialReference().IsSame(layer.GetSpatialRef())):
                geom = geom.Clone()
                geom.TransformTo(layer.GetSpatialRef())
            layer.SetSpatialFilter(geom)
        else:
            if isinstance(geom, tuple): # maybe geom is a simple tuple
                xMin, yMin, xMax, yMax = geom 
            else:
                try: # maybe geom is an extent object
                    xMin, yMin, xMax, yMax = geom.castTo(layer.GetSpatialRef()).xyXY
                except:
                    raise GeoKitVectorError("Geom input not understood")
            layer.SetSpatialFilterRect(xMin, yMin, xMax, yMax)

    if(not where is None):
        r = layer.SetAttributeFilter(where)
        if( r!=0): raise GeoKitVectorError("Error applying where statement")

####################################################################
# Vector feature count
def countFeatures(source, geom=None, where=None):
    """Returns the number of features found in the given source

    * Use 'geom' to filter by an ogr Geometry object
    * Use 'where' to filter by an SQL-style where statement
    """
    ds = loadVector(source)
    layer = ds.GetLayer()
    filterLayer(layer, geom, where)
    return layer.GetFeatureCount()


####################################################################
# Vector feature count
vecInfo = namedtuple("vecInfo","srs bounds xMin yMin xMax yMax count attributes")
def vectorInfo(source):
    """Extract general information about a vector source

    Determines:
        srs : The source's SRS system
        bounds : The source's boundaries (in the srs's units)
        xMin : The source's xMin boundaries (in the srs's units)
        yMin : The source's xMax boundaries (in the srs's units)
        xMax : The source's yMin boundaries (in the srs's units)
        yMax : The source's yMax boundaries (in the srs's units)
        count : The number of features in the source
        attributes : The attribute titles for the source's features
    """
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

    info["count"] = vecLyr.GetFeatureCount()

    info["attributes"] = []
    layerDef = vecLyr.GetLayerDefn()
    for i in range(layerDef.GetFieldCount()):
        info["attributes"].append(layerDef.GetFieldDefn(i).GetName())

    return vecInfo(**info)

####################################################################
# Iterable to loop over vector items
def extractFeatures(source, geom=None, where=None, outputSRS=None, onlyGeom=False, onlyAttr=False):
    """Creates a generator which extracte the features contained within the source
    
    * Iteravely returns (feature-geometry, feature-fields)    
    
    !NOTE! Be careful when filtering by a geometry as the extracted features may not necessarily be IN the given shape
        * Sometimes they may only overlap
        * Sometimes they are only in the geometry's envelope
        * To be sure an extracted geometry fits the selection criteria, you may need to do further processing

    Inputs:
        source : The vector source to extract from
            - path : A path on the file system
            - ogr Dataset
        
        geom : A spatial context filtering the features which are extracted
            - ogr Geometry 
            - (xMin, yMin, xMax, yMax) : boundary extents in the vector source's native srs
            - A geokit.Extent object

        where - str : An SQL-style where statement

        outputSRS : An SRS which instructs the function to output the feature's geometries in a particular SRS
        
        onlyGeom : True/False flag determining whether or not only the feature geometry is returned
        onlyAttr : True/False flag determining whether or not only the feature attributes are returned
            * If both onlyGeom and onlyAttr are false, a namedtuple is returned containing both
    
    Examples:
        - If you wanted to iterate over features in a source which have an attribute called 'color' 
            * You only want features whose color attribute equals 'blue'

        >>> for geom, attribute in extractFeatures(<source>, where="color='blue'"):
        >>>     ...

        - If you simply ant a list of all the geometries described above

        >>> geoms = [geom for geom, attribute in extractFeatures(<source>, where="color='blue'")] 
    """

    ds = loadVector(source)
    layer = ds.GetLayer()
    filterLayer(layer, geom, where)

    trx=None
    if(not outputSRS is None):
        outputSRS = loadSRS(outputSRS)
        lyrSRS = layer.GetSpatialRef()
        if (not lyrSRS.IsSame(outputSRS)):
            trx = osr.CoordinateTransformation(lyrSRS, outputSRS)
    
    for ftr in loopFeatures(layer):
        oGeom = ftr.GetGeometryRef().Clone()
        if ( not trx is None): oGeom.Transform(trx)
        oItems = ftr.items().copy()

        if onlyGeom:
            yield oGeom
        elif onlyAttr: 
            yield oItems
        else:
            yield Feature(oGeom, oItems)
        

def extractFeature(source, feature=None, geom=None, where=None, outputSRS=None, onlyGeom=False, onlyAttr=False):
    """convenience function to get a single geometry from a source using extractFeatures
    
    Returns a tuple containing:
        - The identified feature's geometry as an ogr Geometry object
        - The identified attribute values as a dictionary

    * See geokit.vector.extractFeatures for more info on inputs
    * This function will fail if more than one feature is identified
    """
    if feature is None:
        getter = extractFeatures(source, geom, where, outputSRS)

        # Get first result
        fGeom, fItems = next(getter)

        # try to get a second result
        try:
            s = next(getter)
        except StopIteration:
            pass
        else:
            raise GeoKitVectorError("More than one feature found")
    else:
        ds = loadVector(source)
        lyr = ds.GetLayer()
        ftr = lyr.GetFeature(feature)

        fGeom = ftr.GetGeometryRef().Clone()
        fItems = ftr.items().copy()

    if(not outputSRS is None):
        outputSRS = loadSRS(outputSRS)
        if (not fGeom.GetSpatialReference().IsSame(outputSRS)):
            fGeom.TransformTo( outputSRS )

    # Done!
    if onlyGeom:
        return fGeom
    elif onlyAttr: 
        return fItems
    else:
        return Feature(fGeom, fItems)

def extractAsDataFrame(source, **kwargs):
    """Extracts a vector source and formats it as a Pandas DataFrame

    * All kwargs are passes on to extractFeatures
        - Useful for filtering or to set the outputSRS
        - Do not use 'onlyGeoms' or 'onlyAttr'!
    """
    fields = defaultdict(list)
    fields["geom"] = []
    for g,a in extractFeatures(source, **kwargs):
        fields["geom"].append(g.Clone())
        for k,v in a.items():
            fields[k].append(v)

    df = pd.DataFrame(fields)
    return df


####################################################################
# Create a vector
def createVector( geoms, output=None, srs=None, fieldVals=None, fieldDef=None, overwrite=False):
    """
    Create a vector file from a list of geometries, optionally including field values 

    keyword inputs:
        geoms : The geometries to write into the raster file
            - ogr Geometry
            - str : A WKT string representing a single geometry
            - An iterable of either of the above (exclusively one or the other)
                * All geometries must share the same type (point, line, polygon, ect...)
                * All geometries must share the same SRS
            * If geometry SRS differs from the 'srs' input, then all geometries will be projected to the input srs
            * SRS must be provided via the 'srs' input when the input geometries are WKT strings

        output : An optional output path
            str - The path on the file system
            * If output is None, the vector dataset will be created in memory
            * Assumed to be of "ESRI Shapefile" format
            * Will create a number of files with different extensions

        srs : The Spatial reference system to apply to the created vector
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string

        fieldVals : Attribute values to assign to each geometry
            - pandas-DataFrame : A DataFrame of the field names(taken from column names) and corresponding values
            - dict : A dictionry of the field names (taken from keys) and associated lists of the geometry's field values
            * The order of each column/list will correspond to the order geometries are written into the dataset
            * The length of each column/list must match the number of geometries
        
        fieldDef - dict : A dictionary enforcing the datatype of each attribute when written into the final dataset
            * Options are defined from ogr.OFT[...]
              - ex. Integer, Real, String
            * The ogrType function can be used to map typical python and numpy types to appropriate ogr types

        overwrite - True/False : A Flag determining whether prexisting files should be overwritten
            * Only used when output is not None
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
        finalGeoms = [convertWKT(wkt,srs) for wkt in geoms]

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

    #### Wrap the whole writing function in a 'try' statement in case it fails
    try:
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
                        val = str(value.iloc[gi])
                    elif( _type == "OFTInteger" or _type == "OFTInteger64" ): val = int(value.iloc[gi])
                    else: val = float(value.iloc[gi]) 
                    
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
    
    ##### Delete the datasource in case it failed
    except Exception as e:
        dataSource is None
        raise e



####################################################################
# mutuate a vector
def mutateFeatures(source, processor, srs=None, geom=None, where=None, fieldDef=None, output=None, **kwargs):
    """Process a vector dataset according to a given function

    Returns or creates an ogr dataset containing the resulting data

    * If the user wishes to generate an output file (by giving an 'output' input), then nothing will be returned to help 
      avoid dependance issues. If no output is provided, however, the function will return a dataset for immediate use

    Inputs:
        source : The vector source to extract from
            - path : A path on the file system
            - ogr Dataset

        processor - function : The processing function performing the mutation 
            * The function will take 2 arguments: an ogr.Geometry object and an attribute dictionary 
            * The function must return a tuple containing the geometry and an attribute dictionary
                - The returned geometry can be an ogr.Geometry object or a WKT string
                - If the user wishes to process the objects using shapely, the input geometries should first be cast to 
                  WKT strings (using <geometry>.ExportToWkt()) before being used to initialize a shapely object. After 
                  processing, the shapely objects should be cast back into WKT strings
                - If a WKT string is returned, it is assumed to be in the same srs as the "srs" input
            * The attribute dictionary should always contain the same names for all features
            * The attribute dictionary's values should only be numeric types and strings
            * See example below for more info

        where : str -- the "where" statement used to filter features by attribute
            * follows the SQL-where syntax

        srs : The Spatial reference system to apply to the created vector
            - osr.SpatialReference object
            - an EPSG integer ID
            - a string corresponding to one of the systems found in geokit.srs.SRSCOMMON
            - a WKT string
            * If no SRS is given, use the source's default SRS
            * If the given SRS is different from the source's SRS, all feature geometries will be cast to the given SRS before processing

        fieldDef: (None)
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
    filterLayer(vecLyr, geom, where)

    # TEST THE FEATURES!!!!
    if( vecLyr.GetFeatureCount()==0 ):
        return None #No geometries found!!

    # Sometimes the filtering seems to fail in an odd way when we're applying filters 
    #  which return no geometries. layer.GetFeatureCount() will return something above 0,
    #  but calling layer.GetNextFeature() will return None. So, we will just do a test for 
    #  this, too...
    if( vecLyr.GetNextFeature() is None ):
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
    noValues = True

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
            if isinstance(r,str) or isinstance(r,ogr.Geometry):
                g,v = r,None
            else:
                g,v = r
                
            if( not (isinstance(g, str) or isinstance(g, ogr.Geometry))):
                goodOutput = False
            
            if not v is None:
                noValues = False
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
        if noValues == False:
            for k,v in v.items():
                if(k in values):
                    values[k].append(v)
                else:
                    values[k] = [v,]
        else: 
            values=None

    if( len(geoms)==0 ):
        raise GeoKitVectorError("Invalid geometry count")

    # Create a new shapefile from the results 
    newDS = createVector( geoms, srs=srs, fieldVals=values, fieldDef=fieldDef, output=output, **kwargs )

    if not output is None: 
        return
    else:
        if(newDS is None):
            raise GeoKitVectorError("Failed to create working shapefile")
        return newDS
