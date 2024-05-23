import os
import copy
import numpy as np
from osgeo import gdal, ogr, osr
from tempfile import TemporaryDirectory
import warnings
from collections import namedtuple, defaultdict, OrderedDict
from collections.abc import Iterable
import pandas as pd
from binascii import hexlify
import numbers

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
    if isinstance(x, str):
        ds = gdal.OpenEx(x)
    else:
        ds = x

    if ds is None:
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
    if isinstance(source, str):  # assume input source is a path to a datasource
        ds = ogr.Open(source)
        layer = ds.GetLayer()
    else:  # otherwise, assume input source is an ogr layer object
        # loop over all features
        layer = source
        layer.ResetReading()

    while True:
        ftr = layer.GetNextFeature()
        if ftr:
            yield ftr
        else:
            return


# OGR type map
_ogrIntToType = dict(
    (v, k) for k, v in filter(lambda x: "OFT" in x[0], gdal.__dict__.items())
)
_ogrStrToType = {
    "bool": "OFTInteger",
    "int8": "OFTInteger",
    "int16": "OFTInteger",
    "int32": "OFTInteger",
    "int64": "OFTInteger64",
    "uint8": "OFTInteger",
    "uint16": "OFTInteger",
    "uint32": "OFTInteger",
    "float32": "OFTReal",
    "float64": "OFTReal",
    "string": "OFTString",
    "Object": "OFTString",
}


def ogrType(s):
    """Tries to determine the corresponding OGR type according to the input"""

    if isinstance(s, str):
        if hasattr(ogr, s):
            return s
        elif s.lower() in _ogrStrToType:
            return _ogrStrToType[s.lower()]
        elif hasattr(ogr, "OFT%s" % s):
            return "OFT%s" % s
        return "OFTString"

    elif s is str:
        return "OFTString"
    elif isinstance(s, np.dtype):
        return ogrType(str(s))
    elif isinstance(s, np.generic):
        return ogrType(s.dtype)
    elif s is bool:
        return "OFTInteger"
    elif s is int:
        return "OFTInteger64"
    elif isinstance(s, int):
        return _ogrIntToType[s]
    elif s is float:
        return "OFTReal"
    elif isinstance(s, Iterable):
        return ogrType(s[0])

    raise ValueError("OGR type could not be determined")


def filterLayer(layer, geom=None, where=None):
    """GeoKit internal

    Filters an ogr Layer object accordint to a geometry and where statement
    """
    if not geom is None:
        if isinstance(geom, ogr.Geometry):
            if geom.GetSpatialReference() is None:
                raise GeoKitVectorError("Input geom must have a srs")
            if not geom.GetSpatialReference().IsSame(layer.GetSpatialRef()):
                geom = geom.Clone()
                geom.TransformTo(layer.GetSpatialRef())
            layer.SetSpatialFilter(geom)
        else:
            if isinstance(geom, tuple):  # maybe geom is a simple tuple
                xMin, yMin, xMax, yMax = geom
            else:
                try:  # maybe geom is an extent object
                    xMin, yMin, xMax, yMax = geom.castTo(layer.GetSpatialRef()).xyXY
                except:
                    raise GeoKitVectorError("Geom input not understood")
            layer.SetSpatialFilterRect(xMin, yMin, xMax, yMax)

    if not where is None:
        r = layer.SetAttributeFilter(where)
        if r != 0:
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
    "vecInfo", "srs bounds xMin yMin xMax yMax count attributes source"
)


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
# List layers within a multi-layer vector dataset e.g. a geopackage
def listLayers(
    source,
):
    """Returns the layer names for each layer that is stored in a geopackage.

    Parameters
    ----------
    source :  Anything acceptable by loadVector()
        The vector datasource to read from

    Returns
    -------
    list
        A list of layer names for the source geopackage.
    """
    layer_names = []
    ds = loadVector(source)

    # Loop over the layers to get their names.
    for i in range(ds.GetLayerCount()):
        name = ds.GetLayer(i).GetName()
        layer_names.append(name)
    return layer_names


####################################################################
# Iterable to loop over vector items


def _extractFeatures(
    source,
    geom,
    where,
    srs,
    onlyGeom,
    onlyAttr,
    skipMissingGeoms,
    layerName=None,
    spatialPredicate="Touches",
):
    # Check spatialPredicate
    avail_predicates=["Touches", "Overlaps", "CentroidWithin"]
    assert spatialPredicate in avail_predicates, f"'spatialPredicate' needs to be one of the available spatial predicates filters: '{','.join(avail_predicates)}'. Here: {spatialPredicate}"
    if spatialPredicate!="Touches":
        # spatialPredicate is not default, will take effect only when filter geom is given
        assert isinstance(geom, ogr.Geometry), f"spatialPredicate '{spatialPredicate}' requires geom argument to be an osgeo.ogr.Geometry object"

    # Do filtering
    source = loadVector(source)
    if not layerName is None:
        layer = source.GetLayerByName(layerName)
        filterLayer(layer, geom, where)
    else:
        layer = source.GetLayer()
        filterLayer(layer, geom, where)

    # Make a transformer
    trx = None
    if not srs is None:
        srs = SRS.loadSRS(srs)
        lyrSRS = layer.GetSpatialRef()
        if not lyrSRS.IsSame(srs):
            trx = osr.CoordinateTransformation(lyrSRS, srs)

    # deal with non-geometry type geoms if needed
    if geom is not None and not isinstance(geom, ogr.Geometry):
        if isinstance(geom, tuple):  # maybe geom is a simple tuple
            _geom = GEOM.box(geom, srs=layer.GetSpatialRef())
        else:
            try:  # maybe geom is an extent object
                _geom = GEOM.box(geom.castTo(layer.GetSpatialRef()).xyXY, srs=layer.GetSpatialRef())
            except:
                raise GeoKitVectorError("Geom input not understood")
    elif isinstance(geom, ogr.Geometry):
        _geom = copy.copy(geom)
    else:
        _geom = None
    # adapt srs of filter geom to extracted geometry if needed
    if _geom is not None and layer.GetSpatialRef() is not None and not _geom.GetSpatialReference().IsSame(layer.GetSpatialRef()):
        _geom = GEOM.transform(_geom, toSRS = layer.GetSpatialRef())

    # Yield features and attributes
    _warned=False # initialize flag
    for ftr in loopFeatures(layer):
        oGeom = ftr.GetGeometryRef()

        if oGeom is None and skipMissingGeoms:
            continue
        
        if layer.GetSpatialRef() is not None:
            assert oGeom.GetSpatialReference().IsSame(layer.GetSpatialRef()), f"SRS of extracted geometry deviates from overall layer SRS!"

        # base extraction is "Touches", now apply apply more sophisticated spatialPredicate if required
        if _geom is not None:
            assert oGeom is not None # make sure
            # apply different geometrical filter methods
            if spatialPredicate == "Overlaps":
                if "POLYGON" in oGeom.GetGeometryName():
                    # this means extracted (multi)polygons must have an area overlap (partial or full)
                    if not (_geom.Overlaps(oGeom) or oGeom.Within(_geom)):
                        # skip those _geoms which do not overlap the filter geom (line touch only)
                        continue
                elif "LINESTRING" in oGeom.GetGeometryName():
                    # "overlap' for lines means that extracted lines actually intersect the filter geom beyond its boundary
                    if oGeom.Intersects(_geom) and oGeom.Intersection(_geom).Equals(oGeom):
                        # the line touches the filter geom boundary along its whole length, no "true overlap"
                        # will also not be extracted by neighboring filter geoms
                        if not _warned:
                            warnings.warn(f"At least one line geometry of the vector data to be extracted lies completely on (filter) geom boundary and will not be extracted.")
                            _warned = True
                    if not (oGeom.Intersects(_geom) and oGeom.Intersection(_geom).Equals(oGeom.Intersection(_geom.GetBoundary()))):
                        # the intersecting part of the filter geom and its boundary are alike, i.e. the line touches only, skip
                        continue
                elif "POINT" in oGeom.GetGeometryName():
                    # overlap means that at least one point must be within the filter geom
                    if oGeom.GetGeometryName() == "POINT":
                        _oGeomMulti = [oGeom]
                    elif oGeom.GetGeometryName() == "MULTIPOINT":
                        _oGeomMulti = copy.copy(oGeom)
                    else:
                        raise TypeError(f"Unknown extracted point geometry of type '{oGeom.GetGeometryName()}'.")
                    if all([p.Intersects(_geom.GetBoundary()) for p in _oGeomMulti]):
                        # all points are on the filter geom boundary, these points will not be extracted here nor by neighboring filter geoms
                        if not _warned:
                            warnings.warn(f"At least one (multi)point geometry of the vector data to be extracted lies completely on (filter) geom boundary and will not be extracted.")
                            _warned = True
                    if not any([p.Within(_geom) for p in _oGeomMulti]):
                        # no point falls within the filter geom, skip
                        continue
                else:
                    raise TypeError(f"Unknown extracted geometry of type '{oGeom.GetGeometryName()}'.")
            
            elif spatialPredicate == "CentroidWithin":
                # applies to all extracted geom types alike: the centroid must fall "within" the filter _geom (works also for point and line filter geoms)
                if oGeom.Centroid().Intersects(geom.GetBoundary()):
                    # the Centroid falls right on the filter geom boundary, will be extracted neither here nor by neighboring filter geoms
                    if not _warned:
                        warnings.warn(f"Centroid of at least one geometry of the vector data to be extracted lies exactly on (filter) geom boundary and will not be extracted.")
                        _warned = True
                if not oGeom.Centroid().Within(_geom):
                    # the extracted geom centroid lies outside filter geom (or on its boundary if filter geom is a polygon!), skip
                    continue
        if oGeom is not None:
            oGeom = oGeom.Clone()
            if not trx is None:
                oGeom.Transform(trx)

        if onlyAttr:
            oGeom = None

        if not onlyGeom:
            oItems = ftr.items().copy()
        else:
            oItems = None

        if onlyGeom:
            assert not onlyAttr, f"onlyGeom cannot be combined with onlyAttr."
            yield oGeom
        elif onlyAttr:
            yield oItems
        else:
            yield UTIL.Feature(oGeom, oItems)


def extractFeatures(
    source,
    where=None,
    geom=None,
    srs=None,
    onlyGeom=False,
    onlyAttr=False,
    asPandas=True,
    indexCol=None,
    skipMissingGeoms=True,
    layerName=None,
    spatialPredicate="Touches",
    **kwargs,
):
    """Creates a generator which extract the features contained within the source

    * Iteratively returns (feature-geometry, feature-fields)

    Note:
    -----
    Be careful when filtering by a geometry as the extracted features may not
    necessarily be IN the given shape
    * Sometimes they may only overlap
    * Sometimes they are only in the geometry's envelope
    * To be sure an extracted geometry fits the selection criteria, you may
      still need to do further processing or use extractAndClipFeatures()

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

    layerName : str; optional
        The name of the layer to extract from the source vector dataset (only applicable in case of a geopackage).

    spatialPredicate : str, optional
        Applies only in combination with given 'geom' filter. If "Touches",
        all geometries will be extracted that simply touch the filter 
        geom. If "Overlaps", geometries to be extracted must overlap (for
        lines, this represents an "Intersect") either partially or 
        completely (i.e. it includes "Within"), and if "CentroidWithin" 
        the centroid of the extracted geom must be within or on the 
        filter geom. By default "Touches".
        NOTE: When filter geom is a polygon, centroids exactly on the 
        filter geom boundary will NOT be extracted. 

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
            skipMissingGeoms=skipMissingGeoms,
            layerName=layerName,
            spatialPredicate=spatialPredicate,
        )
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
            skipMissingGeoms=skipMissingGeoms,
            layerName=layerName,
            spatialPredicate=spatialPredicate,
        ):
            fields["geom"].append(g.Clone())
            for k, v in a.items():
                fields[k].append(v)

        df = pd.DataFrame(fields)
        if not indexCol is None:
            df.set_index(indexCol, inplace=True, drop=False)

        if onlyGeom:
            assert not onlyAttr, f"onlyGeom cannot be combined with onlyAttr."
            return df["geom"]
        elif onlyAttr:
            return df.drop("geom", axis=1)
        else:
            return df


def extractFeature(
    source, where=None, geom=None, srs=None, onlyGeom=False, onlyAttr=False, **kwargs
):
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
        getter = _extractFeatures(
            source=source,
            geom=geom,
            where=where,
            srs=srs,
            onlyGeom=onlyGeom,
            onlyAttr=onlyAttr,
            skipMissingGeoms=False,
        )

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

    if not srs is None and not onlyAttr:
        srs = SRS.loadSRS(srs)
        if not fGeom.GetSpatialReference().IsSame(srs):
            fGeom.TransformTo(srs)

    # Done!
    if onlyGeom:
        return fGeom
    elif onlyAttr:
        return fItems
    else:
        return UTIL.Feature(fGeom, fItems)


def extractAsDataFrame(
    source, indexCol=None, geom=None, where=None, srs=None, **kwargs
):
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
    warnings.warn(
        "This function will be removed in favor of geokit.vector.extractFeatures",
        DeprecationWarning,
    )
    return extractFeatures(
        source=source, indexCol=indexCol, geom=geom, where=where, srs=srs, **kwargs
    )


def extractAndClipFeatures(
    source,
    geom,
    where=None,
    srs=None,
    onlyGeom=False,
    indexCol=None,
    skipMissingGeoms=True,
    layerName=None,
    scaleAttrs=None,
    minShare=0.001,
    **kwargs,
):
    """
    Extracts features from a source and clips them to the boundaries of a given geom.
    Optionally scales numeric attribute values linearly to the overlapping area share.

    Parameters:
    -----------
    source : Anything acceptable by loadVector()
        The vector data source to read from

    geom : ogr.Geometry
        The geometry to search with
        * All features touching this geometry are extracted and clipped to the geometry boundaries.

    where : str; optional
        An SQL-like where statement to apply to the source
        * Feature attribute name do not need quotes
        * String values should be wrapped in 'single quotes'
        Example: If the source vector has a string attribute called "ISO" and
                 a integer attribute called "POP", you could use....

            where = "ISO='DEU' AND POP>1000"

    srs : Anything acceptable to geokit.srs.loadSRS(); optional
        The srs of the geometries to extract
          * If not given, the geom's inherent srs is used
          * If srs does not match the inherent srs, all geometries will be
            transformed

    onlyGeom : bool; optional
        If True, only feature geometries will be returned

    indexCol : str; optional
        The feature identifier to use as the DataFrams's index
        * Only useful when as DataFrame is True

    skipMissingGeoms : bool; optional
        If True, then the parser will not read a feature which are missing a geometry

    layerName : str; optional
        The name of the layer to extract from the source vector dataset (only applicable in case of a geopackage).

    scaleAttrs : str or list, optional
        attribute names of the source with numeric values. The values will be scaled linearly with the
        area share of the feature overlapping the geom.

    minShare : float
        The min. area share of a polygon that falls either inside or
        outside the clipping geom. Allows to deal with imperfect boundary
        alignments. 0 means that all clipped geoms, however small they
        may be, are considered. Example: If minShare=0.001 (0.1%), a
        polygon that overlaps with the clipping geom by 99.92% of its
        area will NOT be clipped. If another polygon also at
        minShare=0.001 overlaps by only 0.06% of its area, it will be
        disregarded. By default 0.001.

    Returns:
    --------
    * pandas.DataFrame or pandas.Series
    """
    assert (
        0 <= minShare <= 1
    ), f"minShare must be greater or equal to 0 and less or equal to 1.0"
    # assert and preprocess input source
    if isinstance(source, pd.DataFrame):
        # check validity of input dataframe
        if not "geom" in source.columns:
            raise AttributeError(
                f"source is given as a pd.DataFrame but has not 'geom' column."
            )
        if not isinstance(source.geom.iloc[0], ogr.Geometry):
            raise TypeError(
                f"source is given as a pd.DataFrame but value in 'geom' column is not of type osgeo.ogr.Geometry."
            )
        # return empty dataframe with empty expected "areaShare" column if no geometries contained since vector cannot be created without geometries
        if len(source) == 0:
            source["areaShare"] = None
            return source
        # generate a vector from source dataframe
        source = createVector(source)
    elif isinstance(source, str):
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"source is given as a string but is not an existing filepath: {source}"
            )
        # load as vector file
        source = loadVector(source)
    elif not isinstance(source, gdal.Dataset):
        raise TypeError(
            f"source must either be a pd.DataFrame, a gdal.Dataset vector instance or a str formatted shapefile path."
        )

    # extract only the overlapping geoms, first define srs
    if srs is None:
        srs = geom.GetSpatialReference()
    else:
        geom = GEOM.transform(geom, toSRS=srs)
    df = extractFeatures(
        source=source,
        geom=geom,
        where=where,
        srs=srs,
        onlyGeom=onlyGeom,
        indexCol=indexCol,
        skipMissingGeoms=skipMissingGeoms,
        layerName=layerName,
        **kwargs,
    )
    if scaleAttrs is None:
        scaleAttrs = []
    elif isinstance(scaleAttrs, str):
        scaleAttrs = [scaleAttrs]
    else:
        assert isinstance(
            scaleAttrs, list
        ), f"scaleAttrs must be a str or a list thereof if not None."
    if len(df) > 0:
        for _attr in scaleAttrs:
            if not _attr in list(df.columns):
                raise AttributeError(
                    f"'{_attr}' was given as scaleAttrs but is not an attribute of the source dataframe."
                )
            if not all([isinstance(_val, numbers.Number) for _val in df[_attr]]):
                raise TypeError(
                    f"All values in column '{_attr}' in scaleAttrs must be numeric."
                )

    # check if we have any features to clip at all
    if len(df) == 0:
        # if not, add the mandatory areaShare column in case that it is not there already and return empty dataframe
        df["areaShare"] = None
        return df
    # else add the expected areaShare column
    assert not "areaShare" in list(
        df.columns
    ), f"source data must not contain a 'areaShare' attribute."
    df["areaShare"] = 1.0
    # check if we need to clip the geometries at all
    if df.geom.iloc[0].GetGeometryName()[:5] == "POINT":
        # we have only points and no further clipping is needed
        return df

    # else we need to add an ID column and generate a new vector
    assert not "clippingID" in list(
        df.columns
    ), f"source data must not contain a 'clippingID' attribute."
    df["clippingID"] = range(len(df))
    _vec = createVector(df)
    # extract only these features intersected by the outer geom boundary
    outer_df = extractFeatures(
        source=_vec,
        geom=geom.Boundary(),
        where=where,
        srs=srs,
        indexCol=indexCol,
        skipMissingGeoms=skipMissingGeoms,
        layerName=layerName,
        **kwargs,
    )
    del _vec
    if len(outer_df) == 0:
        # we have no features intersecting with the geom boundary, return all included features

        return df.drop(columns="clippingID")

    # else clip these features that are intersected by the geom
    _clippedIDs = list()
    _clippedGeoms = list()
    _areaShares = list()
    for i, feat in zip(outer_df.clippingID, outer_df.geom):
        _clipped = feat.Intersection(geom)
        _areaShare = _clipped.Area() / feat.Area()
        if _areaShare >= 1.0 - minShare:
            # the feature is only touched by the boundary (or protrudes minimally outside the geom edge) -> will not be reduced
            continue
        elif _areaShare <= 0.0 + minShare:
            # the feature is fully outside the geom and only touches the geom boundary (or overlaps only minimally with geom)
            # -> set clipped feature geometry to np.nan to filter out later
            _clipped = np.nan
        _clippedGeoms.append(_clipped)
        _areaShares.append(_areaShare)
        _clippedIDs.append(i)

    if len(_clippedIDs) == 0:
        # we have not clipped any feature at all, return df
        return df.drop(columns="clippingID")

    # else replace the original feature geometries with the clipped ones where needed and add area shares
    df.loc[df.clippingID.isin(_clippedIDs), "geom"] = _clippedGeoms
    df.loc[df.clippingID.isin(_clippedIDs), "areaShare"] = _areaShares
    for _attr in scaleAttrs:
        df[_attr] = df.apply(lambda x: x[_attr] * x.areaShare, axis=1)

    # drop nan geometries that do not (sufficiently) overlap filter geom
    df = df[~df.geom.isna()].reset_index(drop=True)

    # return the adapted dataframe
    return df.drop(columns="clippingID")


####################################################################
# Create a vector
def createVector(
    geoms,
    output=None,
    srs=None,
    driverName="ESRI Shapefile",
    layerName="default",
    fieldVals=None,
    fieldDef=None,
    checkAllGeoms=False,
    overwrite=True,
):
    """
    Create a vector on disk from geometries or a DataFrame with 'geom' column

    Parameters:
    -----------
    geoms : ogr.Geometry or [ogr.Geometry, ] or pandas.DataFrane
        The geometries to write into the vector file
        * If a DataFrame is given, it must have a column called 'geom'
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

    driverName : str; optional
        The name of the driver to use when creating the vector.
        Currently supported options are:
            - ESRI Shapefile
            - GPKG

        For a list of all supported vector drivers by OGR, see: https://gdal.org/drivers/vector/index.html

    layerName : str; optional
        The name of the layer to create within the vector. (Only applicable for GeoPackages)
        * If the layer already exists, it will be overwritten

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

    checkAllGeoms : bool, optional
        If True, all geoms will be asserted in object type and exact srs. Else, only
        the first geom in the geom column/iterable will be checked fpr performance reasons.
        By default False.

    overwrite : bool; optional
        Determines whether the prexisting files should be overwritten
        * Only used when output is not None

    Returns:
    --------
    * If 'output' is None: gdal.Dataset
    * If 'output' is given: None

    """
    if srs:
        srs = SRS.loadSRS(srs)

    # make geom or wkt list into a list of ogr.Geometry objects
    finalGeoms = []
    geomIndex = None
    if isinstance(geoms, str) or isinstance(
        geoms, ogr.Geometry
    ):  # geoms is a single geometry
        geoms = [
            geoms,
        ]
    elif isinstance(geoms, pd.Series):
        geomIndex = geoms.index
        geoms = geoms.values
    elif isinstance(geoms, pd.DataFrame):
        if not fieldVals is None:
            raise GeoKitVectorError(
                "fieldVals must be None when geoms input is a DataFrame"
            )

        fieldVals = geoms.copy()
        geoms = geoms.geom.values
        fieldVals.drop("geom", inplace=True, axis=1)

    if len(geoms) == 0:
        raise GeoKitVectorError("Empty geometry list given")

    # Test if the first geometry is an ogr-Geometry type
    if isinstance(geoms[0], ogr.Geometry):
        #  (Assume all geometries in the array have the same type)
        geomSRS = geoms[0].GetSpatialReference()
        if checkAllGeoms:
            # check if all other geoms have the same SRS
            assert all(
                [geomSRS.IsSame(g.GetSpatialReference()) for g in geoms]
            ), f"Not all geoms have the same SRS, srs of first geom: {geomSRS}"
        # Set up some variables for the upcoming loop
        doTransform = False
        setSRS = False

        if (
            srs and geomSRS and not srs.IsSame(geomSRS)
        ):  # Test if a transformation is needed
            trx = osr.CoordinateTransformation(geomSRS, srs)
            doTransform = True
        # Test if an srs was NOT given, but the incoming geometries have an SRS already
        elif srs is None and geomSRS:
            srs = geomSRS
        # Test if an srs WAS given, but the incoming geometries do NOT have an srs already
        elif srs and geomSRS is None:
            # In which case, assume the geometries are meant to have the given srs
            setSRS = True

        # Create geoms
        for i in range(len(geoms)):
            # clone just incase the geometry is tethered outside of function
            finalGeoms.append(geoms[i].Clone())
            if doTransform:
                finalGeoms[i].Transform(trx)  # Do transform if necessary
            if setSRS:
                finalGeoms[i].AssignSpatialReference(srs)  # Set srs if necessary

    # Otherwise, the geometry array should contain only WKT strings
    elif isinstance(geoms[0], str):
        if srs is None:
            raise ValueError("srs must be given when passing wkt strings")

        # Create geoms
        finalGeoms = [GEOM.convertWKT(wkt, srs) for wkt in geoms]

    else:
        raise ValueError("Geometry inputs must be ogr.Geometry objects or WKT strings")

    geoms = finalGeoms  # Overwrite geoms array with the conditioned finalGeoms

    # Determine geometry types
    types = set()
    for g in geoms:
        # Get a set of all geometry type-names (POINT, POLYGON, ect...)
        types.add(g.GetGeometryName())

    if types.issubset({"POINT", "MULTIPOINT"}):
        geomType = ogr.wkbPoint
    elif types.issubset({"LINESTRING", "MULTILINESTRING"}):
        geomType = ogr.wkbLineString
    elif types.issubset({"POLYGON", "MULTIPOLYGON"}):
        geomType = ogr.wkbPolygon
    else:
        # geomType = ogr.wkbGeometryCollection
        raise RuntimeError("Could not determine output shape's geometry type")

    # Create a driver and datasource
    # driver = gdal.GetDriverByName("ESRI Shapefile")
    # dataSource = driver.Create(output, 0, 0)

    if output is not None and overwrite:
        # Search for directory
        if (
            os.path.dirname(output) == ""
        ):  # If no directory is given, assume current directory
            output = os.path.join(os.getcwd(), output)

        elif not os.path.isdir(
            os.path.dirname(output)
        ):  # If directory does not exist, raise error
            raise FileNotFoundError(
                f"Directory {os.path.dirname(output)} does not exist"
            )

        # Remove file if it exists
        if os.path.isfile(output):
            os.remove(output)

        driver = ogr.GetDriverByName(driverName)
        dataSource = driver.CreateDataSource(output)

    elif output is not None and overwrite == False:
        warnings.warn("Overwriting existing file")
        dataSource = ogr.Open(output, 1)
        assert dataSource is not None, f"Could not open {output}"

    else:
        # driver = ogr.GetDriverByName("Memory")
        # dataSource = driver.CreateDataSource("")

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
        if output is not None and overwrite == False:
            layerName = layerName
            assert layerName not in listLayers(
                output
            ), f"Layer name '{layerName}' already exists in {output}. Please Specify a new layer name or set overwrite=True."

        else:
            layerName = layerName

        layer = dataSource.CreateLayer(layerName, srs, geomType)
        assert layer is not None, "Could not create layer!"

        # Setup fieldVals and fieldDef dicts
        if not fieldVals is None:
            # Ensure fieldVals is a dataframe
            if not isinstance(fieldVals, pd.DataFrame):
                # If not, try converting it
                fieldVals = pd.DataFrame(fieldVals)
            if not geomIndex is None:
                fieldVals = fieldVals.loc[geomIndex]

            # check if length is good
            if fieldVals.shape[0] != len(geoms):
                raise GeoKitVectorError(
                    "Values table length does not equal geometry count"
                )

            # Ensure fieldDefs exists and is a dict
            if fieldDef is None:  # Try to determine data types
                fieldDef = {}
                for k, v in fieldVals.items():
                    fieldDef[k] = ogrType(v.dtype)

            elif isinstance(
                fieldDef, dict
            ):  # Assume fieldDef is a dict mapping fields to inappropriate
                #   datatypes. Fix these to ogr indicators!
                for k in fieldVals.columns:
                    try:
                        fieldDef[k] = ogrType(fieldDef[k])
                    except KeyError as e:
                        print('"%s" not in attribute definition table' % k)
                        raise e

            else:  # Assume a single data type was given. Apply to all fields
                _type = ogrType(fieldDef)
                fieldDef = {}
                for k in fieldVals.keys():
                    fieldDef[k] = _type

            # Write field definitions to layer
            for fieldName, dtype in fieldDef.items():
                layer.CreateField(ogr.FieldDefn(str(fieldName), getattr(ogr, dtype)))

            # Ensure list lengths match geom length
            for k, v in fieldVals.items():
                if len(v) != len(geoms):
                    raise RuntimeError("'{}' length does not match geom list".format(k))

        # Create features
        for gi in range(len(geoms)):
            # Create a blank feature
            feature = ogr.Feature(layer.GetLayerDefn())

            # Fill the attributes, if required
            if not fieldVals is None:
                for fieldName, value in fieldVals.items():
                    _type = fieldDef[fieldName]

                    # cast to basic type
                    if _type == "OFTString":
                        val = str(value.iloc[gi])
                    elif _type == "OFTInteger" or _type == "OFTInteger64":
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
        if output:
            return output
        else:
            return dataSource

    # Delete the datasource in case it failed
    except Exception as e:
        dataSource = None
        raise e


def createGeoJson(geoms, output=None, srs=4326, topo=False, fill=""):
    """Convert a set of geometries to a geoJSON object"""
    if srs:
        srs = SRS.loadSRS(srs)

    # arrange geom, index, and data
    if isinstance(geoms, ogr.Geometry):  # geoms is a single geometry
        finalGeoms = [
            geoms,
        ]
        data = None
        index = [
            0,
        ]

    elif isinstance(geoms, pd.Series):
        index = geoms.index
        finalGeoms = geoms.values
        data = None

    elif isinstance(geoms, pd.DataFrame):
        index = geoms.index
        finalGeoms = geoms.geom.values
        data = geoms.loc[:, geoms.columns != "geom"]
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

    fo.write(bytes('{"type":"FeatureCollection","features":[', encoding="utf-8"))

    for j, i, g in zip(range(len(index)), index, finalGeoms):
        fo.write(
            bytes('%s{"type":"Feature",' % ("" if j == 0 else ","), encoding="utf-8")
        )
        if data is None:
            fo.write(bytes('"properties":{"_index":%s},' % str(i), encoding="utf-8"))
        else:
            fo.write(
                bytes(
                    '"properties":%s,' % data.loc[i].fillna(fill).to_json(),
                    encoding="utf-8",
                )
            )

        fo.write(bytes('"geometry":%s}' % g.ExportToJson(), encoding="utf-8"))
        # fo.write(bytes('"geometry": {"type": "Point","coordinates": [125.6, 10.1] }}', encoding='utf-8'))
    fo.write(bytes("]}", encoding="utf-8"))
    fo.flush()

    # Put in the right format
    if topo:
        from topojson import conversion
        from io import TextIOWrapper

        fo.seek(0)
        topo = conversion.convert(
            TextIOWrapper(fo), object_name="primary"
        )  # automatically closes fo
        topo = str(topo).replace("'", '"')

    # Done!
    if output is None:
        if topo:
            return topo
        else:
            fo.seek(0)
            geojson = fo.read()
            fo.close()
            return geojson.decode("utf-8")

    elif isinstance(output, str):
        if topo:
            with open(output, "w") as fo:
                fo.write(topo)
        else:
            pass  # we already wrote to the file!
        return output

    else:
        if topo:
            output.write(bytes(topo, encoding="utf-8"))
        else:
            pass  # We already wrote to the file!
        return None


####################################################################
# mutuate a vector


def createGeoDataFrame(dfGeokit: pd.DataFrame):
    """Creates a gdf from an Reskit shape pd.DataFrame

    Parameters
    ----------
    dfGeokit : pd.DataFrame
        Reskit shape pd.DataFrame, need a 'geom' column.

    Returns
    -------
    gpd.GeoDataFrame
        Same as the previos, just as an GeodataFrame
    """
    assert isinstance(dfGeokit, pd.DataFrame)
    assert "geom" in dfGeokit.columns

    # import the required external packages - these are not part of the requirements.yml and are possibly not installed
    try:
        import shapely
    except:
        raise ImportError(
            "'shapely' is required for geokit.vector.createGeoDataFrame() but is not installed in the current environment."
        )

    try:
        import geopandas as gpd
    except:
        raise ImportError(
            "'geopandas' is required for geokit.vector.createGeoDataFrame() but is not installed in the current environment."
        )

    # get values
    values = {}
    for col in dfGeokit.columns:
        # geoms need to be converted to shapely
        if col == "geom":
            values["geometry"] = [
                shapely.wkt.loads(g.ExportToWkt()) for g in dfGeokit.geom
            ]  # this takes some time :/
        # other values are just stored as they are
        else:
            values[col] = list(dfGeokit[col])

    # get srs as Well knwon text
    crs = dfGeokit.geom.iloc[0].GetSpatialReference().ExportToWkt()

    # create gdf and set index
    gdf = gpd.GeoDataFrame(values, crs=crs)
    gdf = gdf.set_index(dfGeokit.index, drop=True)

    # over and out!
    return gdf


def mutateVector(
    source,
    processor=None,
    srs=None,
    geom=None,
    where=None,
    fieldDef=None,
    output=None,
    keepAttributes=True,
    _slim=False,
    **kwargs,
):
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
                "There is no 'geom' field in the resulting vector table"
            )

        # make sure the geometries have an srs
        if not geoms.geom[0].GetSpatialReference():
            srs = SRS.loadSRS(srs)
            geoms.geom.apply(lambda x: x.AssignSpatialReference(srs))

    # Create a new shapefile from the results
    if _slim:
        return createVector(geoms.geom)
    else:
        return createVector(geoms, srs=srs, output=output, **kwargs)


def rasterize(
    source,
    pixelWidth,
    pixelHeight,
    srs=None,
    bounds=None,
    where=None,
    value=1,
    output=None,
    dtype=None,
    compress=True,
    noData=None,
    overwrite=True,
    fill=None,
    **kwargs,
):
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
    if output is None and not "bands" in kwargs:
        kwargs["bands"] = [1]

    if isinstance(value, str):
        kwargs["attribute"] = value
    else:
        kwargs["burnValues"] = [
            value,
        ]

    # Do 'in memory' rasterization
    # We need to follow this path in both cases since the below fails when simultaneously rasterizing and writing to disk (I couldn't figure out why...)
    if output is None or not srsOkay:
        # Create temporary output file
        outputDS = UTIL.quickRaster(
            bounds=bounds,
            srs=srs,
            dx=pixelWidth,
            dy=pixelHeight,
            dtype=dtype,
            noData=noData,
            fill=fill,
        )

        # Do rasterize
        tmp = gdal.Rasterize(outputDS, source, where=where, **kwargs)
        if tmp == 0:
            raise RASTER.GeoKitRasterError("Rasterization failed!")
        outputDS.FlushCache()

        if output is None:
            return outputDS
        else:
            ri = RASTER.rasterInfo(outputDS)
            RASTER.createRasterLike(
                ri, output=output, data=RASTER.extractMatrix(outputDS)
            )
            return output

    # Do a rasterization to a file on disk
    else:
        # Check for existing file
        if os.path.isfile(output):
            if overwrite == True:
                os.remove(output)
                if os.path.isfile(output + ".aux.xml"):  # Because QGIS....
                    os.remove(output + ".aux.xml")
            else:
                raise RASTER.GeoKitRasterError(
                    "Output file already exists: %s" % output
                )

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
        bounds = (
            bounds[0] + 0.001 * pixelWidth,
            bounds[1] + 0.001 * pixelHeight,
            bounds[2] - 0.001 * pixelWidth,
            bounds[3] - 0.001 * pixelHeight,
        )

        # Do rasterize
        tmp = gdal.Rasterize(
            output,
            source,
            outputBounds=bounds,
            xRes=pixelWidth,
            yRes=pixelHeight,
            outputSRS=srs,
            noData=noData,
            where=where,
            creationOptions=co,
            targetAlignedPixels=aligned,
            **kwargs,
        )

        if not UTIL.isRaster(tmp):
            raise RASTER.GeoKitRasterError("Rasterization failed!")

        return output
