import os
import numpy as np
from osgeo import gdal, ogr, osr
from tempfile import TemporaryDirectory
import warnings
from collections import namedtuple, defaultdict, OrderedDict
from collections.abc import Iterable
import pandas as pd

from . import util as UTIL
from . import srs as SRS
from . import geom as GEOM
from . import raster as RASTER


class GeoKitVectorError(UTIL.GeoKitError):
    pass


####################################################################
# INTERNAL FUNCTIONS

# Loaders Functions


def loadVector(x):
    """
    Load a vector dataset from a path to a file on disc

    Parameters:
    -----------
    source : str or gdal.Dataset
        * If a string is given, it is assumed as a path to a vector file on disc
        * If a gdal.Dataset is given, it is assumed to already be an open vector
          and is returned immediately

    Returns:
    --------
    gdal.Dataset

    """
    if(isinstance(x, str)):
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

    Parameters:
    -----------
    source : Anything acceptable by loadVector()
        The vector datasource to read from

    """
    if(isinstance(source, str)):  # assume input source is a path to a datasource
        ds = ogr.Open(source)
        layer = ds.GetLayer()
    else:  # otherwise, assume input source is an ogr layer object
        # loop over all features
        layer = source
        layer.ResetReading()

    while(True):
        ftr = layer.GetNextFeature()
        if ftr:
            yield ftr
        else:
            return


# OGR type map
_ogrIntToType = dict((v, k) for k, v in filter(
    lambda x: "OFT" in x[0], gdal.__dict__.items()))
_ogrStrToType = {"bool": "OFTInteger", "int8": "OFTInteger", "int16": "OFTInteger",
                 "int32": "OFTInteger", "int64": "OFTInteger64", "uint8": "OFTInteger",
                 "uint16": "OFTInteger", "uint32": "OFTInteger", "float32": "OFTReal",
                 "float64": "OFTReal", "string": "OFTString", "Object": "OFTString"}


def ogrType(s):
    """Tries to determine the corresponding OGR type according to the input"""

    if(isinstance(s, str)):
        if(hasattr(ogr, s)):
            return s
        elif(s.lower() in _ogrStrToType):
            return _ogrStrToType[s.lower()]
        elif(hasattr(ogr, 'OFT%s' % s)):
            return 'OFT%s' % s
        return "OFTString"

    elif(s is str):
        return "OFTString"
    elif(isinstance(s, np.dtype)):
        return ogrType(str(s))
    elif(isinstance(s, np.generic)):
        return ogrType(s.dtype)
    elif(s is bool):
        return "OFTInteger"
    elif(s is int):
        return "OFTInteger64"
    elif(isinstance(s, int)):
        return _ogrIntToType[s]
    elif(s is float):
        return "OFTReal"
    elif(isinstance(s, Iterable)):
        return ogrType(s[0])

    raise ValueError("OGR type could not be determined")


def filterLayer(layer, geom=None, where=None):
    """GeoKit internal

    Filters an ogr Layer object accordint to a geometry and where statement
    """
    if (not geom is None):
        if isinstance(geom, ogr.Geometry):
            if(geom.GetSpatialReference() is None):
                raise GeoKitVectorError("Input geom must have a srs")
            if(not geom.GetSpatialReference().IsSame(layer.GetSpatialRef())):
                geom = geom.Clone()
                geom.TransformTo(layer.GetSpatialRef())
            layer.SetSpatialFilter(geom)
        else:
            if isinstance(geom, tuple):  # maybe geom is a simple tuple
                xMin, yMin, xMax, yMax = geom
            else:
                try:  # maybe geom is an extent object
                    xMin, yMin, xMax, yMax = geom.castTo(
                        layer.GetSpatialRef()).xyXY
                except:
                    raise GeoKitVectorError("Geom input not understood")
            layer.SetSpatialFilterRect(xMin, yMin, xMax, yMax)

    if(not where is None):
        r = layer.SetAttributeFilter(where)
        if(r != 0):
            raise GeoKitVectorError("Error applying where statement")

####################################################################
# Vector feature count


def countFeatures(source, geom=None, where=None):
    """Returns the number of features found in the given source and within a 
    given geometry and/or where-statement

    Parameters:
    -----------
    source : Anything acceptable by loadVector()
        The vector datasource to read from

    geom : ogr.Geometry; optional
        The geometry to search within
        * All features are extracted which touch this geometry

    where : str; optional
        An SQL-like where statement to apply to the source
        * Feature attribute name do not need quotes
        * String values should be wrapped in 'single quotes'
        Example: If the source vector has a string attribute called "ISO" and 
                 a integer attribute called "POP", you could use....

            where = "ISO='DEU' AND POP>1000"

    Returns:
    --------
    int

    """
    ds = loadVector(source)
    layer = ds.GetLayer()
    filterLayer(layer, geom, where)
    return layer.GetFeatureCount()


####################################################################
# Vector feature count
vecInfo = namedtuple(
    "vecInfo", "srs bounds xMin yMin xMax yMax count attributes source")


def vectorInfo(source):
    """Extract general information about a vector source

    Determines:

    Parameters:
    -----------
    source : Anything acceptable by loadVector()
        The vector datasource to read from

    Returns:
    --------
    namedtuple -> (srs : The source's SRS system,
                   bounds : The source's boundaries (in the srs's units),
                   xMin : The source's xMin boundaries (in the srs's units),
                   yMin : The source's xMax boundaries (in the srs's units),
                   xMax : The source's yMin boundaries (in the srs's units),
                   yMax : The source's yMax boundaries (in the srs's units),
                   count : The number of features in the source,
                   attributes : The attribute titles for the source's features,)

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

    info["source"] = vecDS.GetDescription()

    info["attributes"] = []
    layerDef = vecLyr.GetLayerDefn()
    for i in range(layerDef.GetFieldCount()):
        info["attributes"].append(layerDef.GetFieldDefn(i).GetName())

    return vecInfo(**info)

####################################################################
# Iterable to loop over vector items


def _extractFeatures(source, geom, where, srs, onlyGeom, onlyAttr, skipMissingGeoms, ):
    # Do filtering
    source = loadVector(source)
    layer = source.GetLayer()
    filterLayer(layer, geom, where)

    # Make a transformer
    trx = None
    if(not srs is None):
        srs = SRS.loadSRS(srs)
        lyrSRS = layer.GetSpatialRef()
        if (not lyrSRS.IsSame(srs)):
            trx = osr.CoordinateTransformation(lyrSRS, srs)

    # Yield features and attributes
    for ftr in loopFeatures(layer):
        if not onlyAttr:
            oGeom = ftr.GetGeometryRef()

            if oGeom is None:
                if skipMissingGeoms:
                    continue
            else:
                oGeom = oGeom.Clone()
                if (not trx is None):
                    oGeom.Transform(trx)
        else:
            oGeom = None

        if not onlyGeom:
            oItems = ftr.items().copy()
        else:
            oItems = None

        if onlyGeom:
            yield oGeom
        elif onlyAttr:
            yield oItems
        else:
            yield UTIL.Feature(oGeom, oItems)


def extractFeatures(source, where=None, geom=None, srs=None, onlyGeom=False, onlyAttr=False, asPandas=True, indexCol=None, skipMissingGeoms=True, **kwargs):
    """Creates a generator which extract the features contained within the source

    * Iteratively returns (feature-geometry, feature-fields)    

    Note:
    -----
    Be careful when filtering by a geometry as the extracted features may not 
    necessarily be IN the given shape
    * Sometimes they may only overlap
    * Sometimes they are only in the geometry's envelope
    * To be sure an extracted geometry fits the selection criteria, you may 
      still need to do further processing

    Parameters:
    -----------
    source : Anything acceptable by loadVector()
        The vector data source to read from

    geom : ogr.Geometry; optional
        The geometry to search with
        * All features are extracted which touch this geometry

    where : str; optional
        An SQL-like where statement to apply to the source
        * Feature attribute name do not need quotes
        * String values should be wrapped in 'single quotes'
        Example: If the source vector has a string attribute called "ISO" and 
                 a integer attribute called "POP", you could use....

            where = "ISO='DEU' AND POP>1000"

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the geometries to extract
          * If not given, the source's inherent srs is used
          * If srs does not match the inherent srs, all geometries will be 
            transformed

    onlyGeom : bool; optional
        If True, only feature geometries will be returned

    onlyAttr : bool; optional
        If True, only feature attributes will be returned

    asPandas : bool; optional
        Whether or not the result should be returned as a pandas.DataFrame (when
        onlyGeom is False) or pandas.Series (when onlyGeom is True)

    indexCol : str; optional
        The feature identifier to use as the DataFrams's index
        * Only useful when as DataFrame is True

    skipMissingGeoms : bool; optional
        If True, then the parser will not read a feature which are missing a geometry

    Returns:
    --------
    * If asPandas is True: pandas.DataFrame or pandas.Series
    * If asPandas is False: generator

    """
    # arrange output
    if not asPandas:
        return _extractFeatures(
            source=source,
            geom=geom,
            where=where,
            srs=srs,
            onlyGeom=onlyGeom,
            onlyAttr=onlyAttr,
            skipMissingGeoms=skipMissingGeoms)
    else:
        fields = defaultdict(list)
        fields["geom"] = []

        for g, a in _extractFeatures(
                source=source,
                geom=geom,
                where=where,
                srs=srs,
                onlyGeom=False,
                onlyAttr=False,
                skipMissingGeoms=skipMissingGeoms):
            fields["geom"].append(g.Clone())
            for k, v in a.items():
                fields[k].append(v)

        df = pd.DataFrame(fields)
        if not indexCol is None:
            df.set_index(indexCol, inplace=True, drop=False)

        if onlyGeom:
            return df["geom"]
        elif onlyAttr:
            return df.drop("geom", axis=1)
        else:
            return df


def extractFeature(source, where=None, geom=None, srs=None, onlyGeom=False, onlyAttr=False, **kwargs):
    """Convenience function calling extractFeatures which assumes there is only
    one feature to extract

    * Will raise an error if multiple features are found

    Parameters:
    -----------
    source : Anything acceptable by loadVector()
        The vector datasource to read from

    where : str, int; optional
        If string -> An SQL-like where statement to apply to the source
        If int -> The feature's ID within the dataset
        * Feature attribute name do not need quotes
        * String values should be wrapped in 'single quotes'
        Example: If the source vector has a string attribute called "ISO" and 
                 a integer attribute called "POP", you could use....

            where = "ISO='DEU' AND POP>1000"

    geom : ogr.Geometry; optional
        The geometry to search with
        * All features are extracted which touch this geometry

    outputSRS : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the geometries to extract
          * If not given, the source's inherent srs is used
          * If srs does not match the inherent srs, all geometries will be 
            transformed

    onlyGeom : bool; optional
        If True, only feature geometries will be returned

    onlyAttr : bool; optional
        If True, only feature attributes will be returned

    Returns:
    --------
    * If onlyGeom and onlyAttr are False: namedtuple -> (geom, attr)
    * If onlyGeom is True: ogr.Geometry
    * If onlyAttr is True: dict

    """
    if isinstance(where, int):
        ds = loadVector(source)
        lyr = ds.GetLayer()
        ftr = lyr.GetFeature(where)

        fGeom = ftr.GetGeometryRef().Clone()
        fItems = ftr.items().copy()

    else:
        getter = _extractFeatures(source=source, geom=geom, where=where, srs=srs,
                                  onlyGeom=onlyGeom, onlyAttr=onlyAttr, skipMissingGeoms=False)

        # Get first result
        res = next(getter)

        if onlyGeom:
            fGeom = res
        elif onlyAttr:
            fItems = res
        else:
            (fGeom, fItems) = res

        # try to get a second result
        try:
            next(getter)
        except StopIteration:
            pass
        else:
            raise GeoKitVectorError("More than one feature found")

    if(not srs is None and not onlyAttr):
        srs = SRS.loadSRS(srs)
        if (not fGeom.GetSpatialReference().IsSame(srs)):
            fGeom.TransformTo(srs)

    # Done!
    if onlyGeom:
        return fGeom
    elif onlyAttr:
        return fItems
    else:
        return UTIL.Feature(fGeom, fItems)


def extractAsDataFrame(source, indexCol=None, geom=None, where=None, srs=None, **kwargs):
    """Convenience function calling extractFeatures and structuring the output as
    a pandas DataFrame

    * Geometries are written to a column called 'geom'
    * Attributes are written to a column of the same name

    Parameters:
    -----------
    source : Anything acceptable by loadVector()
        The vector datasource to read from

    indexCol : str; optional
        The feature identifier to use as the DataFrams's index

    geom : ogr.Geometry; optional
        The geometry to search within
        * All features are extracted which touch this geometry

    where : str; optional
        An SQL-like where statement to apply to the source
        * Feature attribute name do not need quotes
        * String values should be wrapped in 'single quotes'
        Example: If the source vector has a string attribute called "ISO" and 
                 a integer attribute called "POP", you could use....

            where = "ISO='DEU' AND POP>1000"

    outputSRS : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the geometries to extract
          * If not given, the source's inherent srs is used
          * If srs does not match the inherent srs, all geometries will be 
            transformed

    Returns:
    --------
    pandas.DataFrame

    """
    warnings.warn("This function will be removed in favor of geokit.vector.extractFeatures", DeprecationWarning)
    return extractFeatures(source=source, indexCol=indexCol, geom=geom, where=where, srs=srs, **kwargs)


####################################################################
# Create a vector
def createVector(geoms, output=None, srs=None, fieldVals=None, fieldDef=None, overwrite=True):
    """
    Create a vector on disk from geometries or a DataFrame with 'geom' column

    Parameters:
    -----------
    geoms : ogr.Geometry or [ogr.Geometry, ] or pandas.DataFrane
        The geometries to write into the raster file
        * If a DataFRame is given, it must have a column called 'geom'
        * All geometries must share the same type (point, line, polygon, ect...)
        * All geometries must share the same SRS
        * If geometry SRS differs from the 'srs' input, then all geometries will 
          be projected to the input srs

    output : str; optional
        A path on disk to create the output vector
        * If output is None, the vector dataset will be created in memory
        * Assumed to be of "ESRI Shapefile" format
        * Will create a number of files with different extensions

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the vector to create
          * If not given, the geometries' inherent srs is used
          * If srs does not match the inherent srs, all geometries will be 
            transformed

    fieldVals : dict of lists or pandas.DataFrame; optional
        Explicit attribute values to assign to each geometry
        * If a dict is given, must be a dictionary of the attribute names and 
          associated lists of the attribute values
        * If a DataFrame is given, the field names are taken from column names 
          and attribute values from the corresponding column data
        * The order of each column/list will correspond to the order geometries 
          are written into the dataset
        * The length of each column/list MUST match the number of geometries
        * All values in a single column/list must share the same type
            - Options are int, float, or str

    fieldDef : dict; optional
        A dictionary specifying the datatype of each attribute when written into
        the final dataset
        * Options are defined from ogr.OFT[...]
          - ex. Integer, Real, String
        * The ogrType() function can be used to map typical python and numpy types
          to appropriate ogr types

    overwrite : bool; optional
        Determines whether the prexisting files should be overwritten
        * Only used when output is not None

    Returns:
    --------
    * If 'output' is None: gdal.Dataset
    * If 'output' is given: None

    """
    if(srs):
        srs = SRS.loadSRS(srs)

    # Search for file
    if(output):
        exists = os.path.isfile(output)
        if (exists and overwrite):
            os.remove(output)
        elif(exists and not overwrite):
            raise GeoKitVectorError(
                "%s exists but 'overwrite' is not True" % output)

    # make geom or wkt list into a list of ogr.Geometry objects
    finalGeoms = []
    geomIndex = None
    if (isinstance(geoms, str) or isinstance(geoms, ogr.Geometry)):  # geoms is a single geometry
        geoms = [geoms, ]
    elif isinstance(geoms, pd.Series):
        geomIndex = geoms.index
        geoms = geoms.values
    elif isinstance(geoms, pd.DataFrame):
        if not fieldVals is None:
            raise GeoKitVectorError(
                "fieldVals must be None when geoms input is a DataFrame")

        fieldVals = geoms.copy()
        geoms = geoms.geom.values
        fieldVals.drop("geom", inplace=True, axis=1)

    if len(geoms) == 0:
        raise GeoKitVectorError("Empty geometry list given")

    # Test if the first geometry is an ogr-Geometry type
    if(isinstance(geoms[0], ogr.Geometry)):
        #  (Assume all geometries in the array have the same type)
        geomSRS = geoms[0].GetSpatialReference()

        # Set up some variables for the upcoming loop
        doTransform = False
        setSRS = False

        if(srs and geomSRS and not srs.IsSame(geomSRS)):  # Test if a transformation is needed
            trx = osr.CoordinateTransformation(geomSRS, srs)
            doTransform = True
        # Test if an srs was NOT given, but the incoming geometries have an SRS already
        elif(srs is None and geomSRS):
            srs = geomSRS
        # Test if an srs WAS given, but the incoming geometries do NOT have an srs already
        elif(srs and geomSRS is None):
            # In which case, assume the geometries are meant to have the given srs
            setSRS = True

        # Create geoms
        for i in range(len(geoms)):
            # clone just incase the geometry is tethered outside of function
            finalGeoms.append(geoms[i].Clone())
            if(doTransform):
                finalGeoms[i].Transform(trx)  # Do transform if necessary
            if(setSRS):
                finalGeoms[i].AssignSpatialReference(
                    srs)  # Set srs if necessary

    # Otherwise, the geometry array should contain only WKT strings
    elif(isinstance(geoms[0], str)):
        if(srs is None):
            raise ValueError("srs must be given when passing wkt strings")

        # Create geoms
        finalGeoms = [GEOM.convertWKT(wkt, srs) for wkt in geoms]

    else:
        raise ValueError(
            "Geometry inputs must be ogr.Geometry objects or WKT strings")

    geoms = finalGeoms  # Overwrite geoms array with the conditioned finalGeoms

    # Determine geometry types
    types = set()
    for g in geoms:
        # Get a set of all geometry type-names (POINT, POLYGON, ect...)
        types.add(g.GetGeometryName())

    if(types.issubset({'POINT', 'MULTIPOINT'})):
        geomType = ogr.wkbPoint
    elif(types.issubset({'LINESTRING', 'MULTILINESTRING'})):
        geomType = ogr.wkbLineString
    elif(types.issubset({'POLYGON', 'MULTIPOLYGON'})):
        geomType = ogr.wkbPolygon
    else:
        #geomType = ogr.wkbGeometryCollection
        raise RuntimeError("Could not determine output shape's geometry type")

    # Create a driver and datasource
    #driver = ogr.GetDriverByName("ESRI Shapefile")
    #dataSource = driver.CreateDataSource( output )
    if output:
        driver = gdal.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Create(output, 0, 0)
    else:
        driver = gdal.GetDriverByName("Memory")

        # Using 'Create' from a Memory driver leads to an error. But creating
        #  a temporary shapefile driver (it doesnt actually produce a file, I think)
        #  and then using 'CreateCopy' seems to work
        tmp_driver = gdal.GetDriverByName("ESRI Shapefile")
        t = TemporaryDirectory()
        tmp_dataSource = tmp_driver.Create(t.name + "tmp.shp", 0, 0)

        dataSource = driver.CreateCopy("MEMORY", tmp_dataSource)
        t.cleanup()
        del tmp_dataSource, tmp_driver, t

    # Wrap the whole writing function in a 'try' statement in case it fails
    try:
        # Create the layer
        if(output):
            layerName = os.path.splitext(os.path.basename(output))[0]
        else:
            layerName = "Layer"

        layer = dataSource.CreateLayer(layerName, srs, geomType)

        # Setup fieldVals and fieldDef dicts
        if(not fieldVals is None):
            # Ensure fieldVals is a dataframe
            if(not isinstance(fieldVals, pd.DataFrame)):
                # If not, try converting it
                fieldVals = pd.DataFrame(fieldVals)
            if not geomIndex is None:
                fieldVals = fieldVals.loc[geomIndex]

            # check if length is good
            if(fieldVals.shape[0] != len(geoms)):
                raise GeoKitVectorError(
                    "Values table length does not equal geometry count")

            # Ensure fieldDefs exists and is a dict
            if(fieldDef is None):  # Try to determine data types
                fieldDef = {}
                for k, v in fieldVals.items():
                    fieldDef[k] = ogrType(v.dtype)

            elif(isinstance(fieldDef, dict)):  # Assume fieldDef is a dict mapping fields to inappropriate
                #   datatypes. Fix these to ogr indicators!
                for k in fieldVals.columns:
                    try:
                        fieldDef[k] = ogrType(fieldDef[k])
                    except KeyError as e:
                        print("\"%s\" not in attribute definition table" % k)
                        raise e

            else:  # Assume a single data type was given. Apply to all fields
                _type = ogrType(fieldDef)
                fieldDef = {}
                for k in fieldVals.keys():
                    fieldDef[k] = _type

            # Write field definitions to layer
            for fieldName, dtype in fieldDef.items():
                layer.CreateField(ogr.FieldDefn(
                    str(fieldName), getattr(ogr, dtype)))

            # Ensure list lengths match geom length
            for k, v in fieldVals.items():
                if(len(v) != len(geoms)):
                    raise RuntimeError(
                        "'{}' length does not match geom list".format(k))

        # Create features
        for gi in range(len(geoms)):
            # Create a blank feature
            feature = ogr.Feature(layer.GetLayerDefn())

            # Fill the attributes, if required
            if(not fieldVals is None):
                for fieldName, value in fieldVals.items():
                    _type = fieldDef[fieldName]

                    # cast to basic type
                    if(_type == "OFTString"):
                        val = str(value.iloc[gi])
                    elif(_type == "OFTInteger" or _type == "OFTInteger64"):
                        val = int(value.iloc[gi])
                    else:
                        val = float(value.iloc[gi])

                    # Write to feature
                    feature.SetField(str(fieldName), val)

            # Set the Geometry
            feature.SetGeometry(geoms[gi])

            # Create the feature
            layer.CreateFeature(feature)
            feature.Destroy()  # Free resources (probably not necessary here)

        # Finish
        if(output):
            return output
        else:
            return dataSource

    # Delete the datasource in case it failed
    except Exception as e:
        dataSource is None
        raise e


def createGeoJson(geoms, output=None, srs=4326, topo=False, fill=''):
    """Convert a set of geometries to a geoJSON object"""
    if(srs):
        srs = SRS.loadSRS(srs)

    # arrange geom, index, and data
    if isinstance(geoms, ogr.Geometry):  # geoms is a single geometry
        finalGeoms = [geoms, ]
        data = None
        index = [0, ]

    elif isinstance(geoms, pd.Series):
        index = geoms.index
        finalGeoms = geoms.values
        data = None

    elif isinstance(geoms, pd.DataFrame):
        index = geoms.index
        finalGeoms = geoms.geom.values
        data = geoms.loc[:, geoms.columns != 'geom']
        data["_index"] = index
    else:
        finalGeoms = list(geoms)
        data = None
        index = list(range(len(finalGeoms)))

    if len(finalGeoms) == 0:
        raise GeoKitVectorError("Empty geometry list given")

    # Transform?
    if not srs is None:
        finalGeoms = GEOM.transform(finalGeoms, toSRS=srs)

    # Make JSON object
    from io import BytesIO
    if not output is None and not isinstance(output, str):
        if not output.writable():
            raise GeoKitVectorError("Output object is not writable")

        if topo:
            fo = BytesIO()
        else:
            fo = output
    elif isinstance(output, str) and not topo:
        fo = open(output, "wb")
    else:
        fo = BytesIO()

    fo.write(
        bytes('{"type":"FeatureCollection","features":[', encoding='utf-8'))

    for j, i, g in zip(range(len(index)), index, finalGeoms):

        fo.write(bytes('%s{"type":"Feature",' %
                       ("" if j == 0 else ","), encoding='utf-8'))
        if data is None:
            fo.write(
                bytes('"properties":{"_index":%s},' % str(i), encoding='utf-8'))
        else:
            fo.write(bytes('"properties":%s,' %
                           data.loc[i].fillna(fill).to_json(), encoding='utf-8'))

        fo.write(bytes('"geometry":%s}' % g.ExportToJson(), encoding='utf-8'))
        #fo.write(bytes('"geometry": {"type": "Point","coordinates": [125.6, 10.1] }}', encoding='utf-8'))
    fo.write(bytes("]}", encoding='utf-8'))
    fo.flush()

    # Put in the right format
    if topo:
        from topojson import conversion
        from io import TextIOWrapper

        fo.seek(0)
        topo = conversion.convert(TextIOWrapper(
            fo), object_name="primary")  # automatically closes fo
        topo = str(topo).replace("'", '"')

    # Done!
    if output is None:
        if topo:
            return topo
        else:
            fo.seek(0)
            geojson = fo.read()
            fo.close()
            return geojson.decode('utf-8')

    elif isinstance(output, str):
        if topo:
            with open(output, "w") as fo:
                fo.write(topo)
        else:
            pass  # we already wrote to the file!
        return output

    else:
        if topo:
            output.write(bytes(topo, encoding='utf-8'))
        else:
            pass  # We already wrote to the file!
        return None

####################################################################
# mutuate a vector


def mutateVector(source, processor=None, srs=None, geom=None, where=None, fieldDef=None, output=None, keepAttributes=True, _slim=False, **kwargs):
    """Process a vector dataset according to an arbitrary function

    Note:
    -----
    If this is called without a processor, it simply clips the vector source to
    the selection criteria given by 'geom' and 'where' as well as translates
    all geometries to 'srs'

    Parameters:
    -----------
    source : Anything acceptable by loadVector()
        The vector datasource to read from

    processor : function; optional
        The function which mutates each feature
        * If no function is given, this stage is skipped
        * The function will take 1 arguments: a pandas.Series object containing
          one 'geom' key indicating the geometry and the other keys indicating
          attributes
        * The function must return something understandable by pd.Series  
          containing the geometry under the index 'geom' and any other keys
            - These will be used to update the old geometries and attributes 
        * The attribute dictionary's values should only be numerics and strings
        * See example below for more info

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the vector to create
          * If not given, the source's inherent srs is used
          * If the given SRS is different from the source's SRS, all feature 
            geometries will be cast to the given SRS before processing

    geom : ogr.Geometry; optional
        The geometry to search within
        * All features are extracted which touch this geometry

    where : str; optional
        An SQL-like where statement to apply to the source
        * Feature attribute name do not need quotes
        * String values should be wrapped in 'single quotes'
        Example: If the source vector has a string attribute called "ISO" and 
                 a integer attribute called "POP", you could use....

            where = "ISO='DEU' AND POP>1000"

    fieldDef : dict; optional
        A dictionary specifying the datatype of each attribute when written into
        the final dataset
        * Options are defined from ogr.OFT[...]
          - ex. Integer, Real, String
        * The ogrType() function can be used to map typical python and numpy types
          to appropriate ogr types

    output : str; optional
        A path on disk to create the output vector
        * If output is None, the vector dataset will be created in memory
        * Assumed to be of "ESRI Shapefile" format
        * Will create a number of files with different extensions

    keepAttributes : bool; optional
        If True, the old attributes will be kept in the output vector
            * Unless they are over written by the processor
        If False, only the newly specified attributes are kept

    Returns:
    --------
    * If 'output' is None: gdal.Dataset
    * If 'output' is given: None

    Example:
    --------
    Say you had a vector source which contains point geometries, and where each 
    feature also had an float-attribute called "value". You want to create a new
    vector set wherein you have circle geometries at the same locations as the 
    original points and whose radius is equal to the original features' "value"
    attribute. Furthermore, let's say you only want to do this for feature's who's 
    "value" attribute is greater than zero and less than 10. Do as such...

    >>> def growPoints( row ):
    >>>     # Create a new geom
    >>>     newGeom = row.geom.Buffer(row.radius)
    >>>
    >>>     # Return the new geometry/attribute set
    >>>     return { 'geom':newGeom }
    >>>
    >>> result = processVector( <source-path>, where="value>0 AND value<10", 
    >>>                         processor=growPoints )
    >>>

    """
    # Extract filtered features
    geoms = extractFeatures(source, geom=geom, where=where, srs=srs)
    if geoms.size == 0:
        return None

    # Hold on to the SRS in case we need it
    if srs is None:
        vecds = loadVector(source)
        veclyr = vecds.GetLayer()
        srs = veclyr.GetSpatialRef()

    # Do processing
    if not processor is None:
        result = geoms.apply(lambda x: pd.Series(processor(x)), axis=1)
        if keepAttributes:
            for c in result.columns:
                geoms[c] = result[c].values
        else:
            geoms = result

        if not "geom" in geoms:
            raise GeoKitVectorError(
                "There is no 'geom' field in the resulting vector table")

        # make sure the geometries have an srs
        if not geoms.geom[0].GetSpatialReference():
            srs = SRS.loadSRS(srs)
            geoms.geom.apply(lambda x: x.AssignSpatialReference(srs))

    # Create a new shapefile from the results
    if _slim:
        return createVector(geoms.geom)
    else:
        return createVector(geoms, srs=srs, output=output, **kwargs)


def rasterize(source, pixelWidth, pixelHeight, srs=None, bounds=None, where=None, value=1, output=None, dtype=None, compress=True, noData=None, overwrite=True, fill=None, **kwargs):
    """Rasterize a vector datasource onto a raster context

    Note:
    -----
    When creating an 'in memory' raster vs one which is saved to disk, a slightly
    different algorithm is used which can sometimes add an extra row of pixels. Be
    aware of this if you intend to compare value-matricies directly from rasters 
    generated with this function.

    Parameters:
    -----------
    source : str or ogr.Geometry
        If str, the path to the vector file to load
        If ogr.Geometry, an Polygon geometry
            - Will be immediately turned into a vector

    pixelWidth : numeric
        The pixel width of the raster in the working srs
        * Is 'srs' is not given, these are the units of the source's inherent srs

    pixelHeight : numeric
        The pixel height of the raster in the working srs
        * Is 'srs' is not given, these are the units of the source's inherent srs

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the point to create
        * If 'bounds' is an Extent object, the bounds' internal srs will override
          this input

    bounds : (xMin, yMix, xMax, yMax) or Extent; optional
        The geographic extents spanned by the raster
        * If not given, the whole bounds spanned by the input is used

    where : str; optional
        An SQL-like where statement to use to filter the vector before rasterizing

    value : numeric, str
        The values to burn into the raster
        * If a numeric is given, all pixels are burned with the specified value
        * If a string is given, then one the feature attribute names is expected

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

    Returns:
    --------
    * If 'output' is None: gdal.Dataset
    * If 'output' is a string: The path to the output is returned (for easy opening)

    """
    # Normalize some inputs
    if isinstance(source, ogr.Geometry):
        source = createVector(source)
    else:
        source = loadVector(source)

    # Get the vector's info
    vecinfo = vectorInfo(source)

    if srs is None:
        srs = vecinfo.srs
        srsOkay = True
    else:
        srs = SRS.loadSRS(srs)
        if srs.IsSame(vecinfo.srs):
            srsOkay = True
        else:
            srsOkay = False

    # Look for bounds input
    if bounds is None:
        bounds = vecinfo.bounds
        if not srsOkay:
            bounds = GEOM.boundsToBounds(bounds, vecinfo.srs, srs)
    else:
        try:
            bounds = bounds.xyXY  # Get a tuple from an Extent object
        except:
            pass  # Bounds should already be a tuple

    bounds = UTIL.fitBoundsTo(bounds, pixelWidth, pixelHeight)

    # Determine DataType is not given
    if dtype is None:
        if value == 1:  # Assume we want a bool matrix
            dtype = "GDT_Byte"
        else:  # assume float
            dtype = "GDT_Float32"
    else:
        dtype = RASTER.gdalType(dtype)

    # Collect rasterization options
    if output is None and not 'bands' in kwargs:
        kwargs["bands"] = [1]

    if isinstance(value, str):
        kwargs["attribute"] = value
    else:
        kwargs["burnValues"] = [value, ]

    # Do 'in memory' rasterization
    # We need to follow this path in both cases since the below fails when simultaneously rasterizing and writing to disk (I couldn't figure out why...)
    if output is None or not srsOkay:
        # Create temporary output file
        outputDS = UTIL.quickRaster(bounds=bounds, srs=srs, dx=pixelWidth,
                                    dy=pixelHeight, dtype=dtype, noData=noData, fill=fill)

        # Do rasterize
        tmp = gdal.Rasterize(outputDS, source, where=where, **kwargs)
        if(tmp == 0):
            raise RASTER.GeoKitRasterError("Rasterization failed!")
        outputDS.FlushCache()

        if output is None:
            return outputDS
        else:
            ri = RASTER.rasterInfo(outputDS)
            RASTER.createRasterLike(ri, output=output,
                                    data=RASTER.extractMatrix(outputDS))
            return output

    # Do a rasterization to a file on disk
    else:
        # Check for existing file
        if(os.path.isfile(output)):
            if(overwrite == True):
                os.remove(output)
                if(os.path.isfile(output + ".aux.xml")):  # Because QGIS....
                    os.remove(output + ".aux.xml")
            else:
                raise RASTER.GeoKitRasterError(
                    "Output file already exists: %s" % output)

        # Arrange some inputs
        aligned = kwargs.pop("targetAlignedPixels", True)

        if not "creationOptions" in kwargs:
            if compress:
                co = RASTER.COMPRESSION_OPTION
            else:
                co = []
        else:
            co = kwargs.pop("creationOptions")

        # Fix the bounds issue by making them  just a little bit smaller, which should be fixed by gdalwarp
        bounds = (bounds[0] + 0.001 * pixelWidth,
                  bounds[1] + 0.001 * pixelHeight,
                  bounds[2] - 0.001 * pixelWidth,
                  bounds[3] - 0.001 * pixelHeight, )

        # Do rasterize
        tmp = gdal.Rasterize(output, source, outputBounds=bounds, xRes=pixelWidth, yRes=pixelHeight,
                             outputSRS=srs, noData=noData, where=where,
                             creationOptions=co, targetAlignedPixels=aligned, **kwargs)

        if not UTIL.isRaster(tmp):
            raise RASTER.GeoKitRasterError("Rasterization failed!")

        return output
