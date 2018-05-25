from helpers import *
from geokit.vector import *
from geokit.geom import box
from geokit.util import GeoKitError
import geokit as gk

## ogrType
def test_ogrType():
    if( ogrType(bool) != "OFTInteger" ): error("ogr type")
    if( ogrType("float32") != "OFTReal" ): error("ogr type")
    if( ogrType("Integer64") != "OFTInteger64" ): error("ogr type")
    if( ogrType(NUMPY_FLOAT_ARRAY) != "OFTReal" ): error("ogr type")
    if( ogrType(NUMPY_FLOAT_ARRAY.dtype) != "OFTReal" ): error("ogr type")

    print( "ogrType passed" )


def test_countFeatures():
    if countFeatures(MULTI_FTR_SHAPE_PATH) != 4: error("Simple vector count")

    if countFeatures(MULTI_FTR_SHAPE_PATH, geom=box(5.89,48.77,6.89,49.64, srs=EPSG4326)) != 2:
        error("Vector count - same SRS, geom filter")

    if countFeatures(MULTI_FTR_SHAPE_PATH, geom=box(4022802,2867575, 4104365,2938843, srs=EPSG3035)) != 2:
        error("Vector count - different SRS, geom filter")

    if countFeatures(MULTI_FTR_SHAPE_PATH, where="name LIKE 'mo%'") != 2:
        error("Vector count - where filter")

    print( "countFeatures passed" )

def test_extractFeatures():
    # test basic
    vi = list(extractFeatures(BOXES, asPandas=False))

    if len(vi)!=3: error("extractFeatures 1 - count mismatch") 

    if (vi[0][0].Area()!=1.0): error("extractFeatures 1 - geom mismatch")
    if (vi[0][1]['name']!="harry"): error("extractFeatures 1 - attribute mismatch")

    if (vi[1][0].Area()!=4.0): error("extractFeatures 1 - geom mismatch")
    if (vi[1][1]['name']!="ron"): error("extractFeatures 1 - attribute mismatch")

    if (vi[2][0].Area()!=9.0): error("extractFeatures 1 - geom mismatch")
    if (vi[2][1]['name']!="hermoine"): error("extractFeatures 1 - attribute mismatch")

    # test clip
    vi = list(extractFeatures(BOXES, geom=box(0,0,3,3, srs=EPSG4326), asPandas=False))

    if len(vi)!=2: error("extractFeatures 2 - count mismatch")   

    if (vi[0][0].Area()!=1.0): error("extractFeatures 2 - geom mismatch")
    if (vi[0][1]['name']!="harry"): error("extractFeatures 2 - attribute mismatch")

    if (vi[1][0].Area()!=4.0): error("extractFeatures 2 - geom mismatch")
    if (vi[1][1]['name']!="ron"): error("extractFeatures 2 - attribute mismatch")

    # test srs change and attribute filter
    vi = list(extractFeatures(BOXES, where="smart>0", srs=EPSG3035, asPandas=False))

    if len(vi)!=1: error("extractFeatures 3 - count mismatch")   
    if (not vi[0][0].GetSpatialReference().IsSame(EPSG3035)): error("extractFeatures 3 - srs mismatch")
    if (vi[0][1]['name']!="hermoine"): error("extractFeatures 3 - attribute mismatch")

    # Test loading as a dataframe
    vi = extractFeatures(BOXES, asPandas=True)
    if vi.shape!=(3,3): error("extractFeatures 4 - shape mismatch") 

    if (vi.geom[0].Area()!=1.0): error("extractFeatures 4 - geom mismatch")
    if (vi['name'][0]!="harry"): error("extractFeatures 4 - attribute mismatch")

    if (vi.geom[1].Area()!=4.0): error("extractFeatures 4 - geom mismatch")
    if (vi['name'][1]!="ron"): error("extractFeatures 4 - attribute mismatch")

    if (vi.geom[2].Area()!=9.0): error("extractFeatures 4 - geom mismatch")
    if (vi['name'][2]!="hermoine"): error("extractFeatures 4 - attribute mismatch")

    print( "extractFeatures passed" )

def test_extractFeature():
    # test succeed
    geom, attr = extractFeature(BOXES, where=1)
    if (geom.Area()!=4.0): error("extractFeature 1 - geom mismatch")
    if (attr['name']!="ron"): error("extractFeature 1 - attribute mismatch")

    geom, attr = extractFeature(BOXES, where="name='harry'")
    if (geom.Area()!=1.0): error("extractFeature 2 - geom mismatch")
    if (attr['name']!="harry"): error("extractFeature 2 - attribute mismatch")

    # test fail
    try:
        geom, attr = extractFeature(BOXES, where="smart=0")
        error("extractFeature 3 - fail test")
    except GeoKitError:
        pass
    else:
        error("extractFeature 3 - fail test")

    print( "extractFeature passed" )

## Create shape file
def test_createVector():
    ## Setup
    out1 = result("util_shape1.shp")
    out2 = result("util_shape2.shp")
    out3 = result("util_shape3.shp")

    ######################
    # run and check

    # Single WKT feature, no attributes
    createVector(POLY, out1, srs=EPSG4326, overwrite=True)

    ds = ogr.Open(out1)
    ly = ds.GetLayer()
    if (ly.GetFeatureCount()!=1): error("Vector creation 1 - feature count mismatch")
    if (not ly.GetSpatialRef().IsSame(EPSG4326)): error("Vector creation - srs mismatch")

    # Single GEOM feature, with attributes
    createVector(GEOM_3035, out2, fieldVals={"id":1,"name":["fred",],"value":12.34}, fieldDef={"id":"int8", "name":str, "value":float}, overwrite=True )

    ds = ogr.Open(out2)
    ly = ds.GetLayer()
    ftr = ly.GetFeature(0)
    attr = ftr.items()
    if (type(attr["id"])!=int or type(attr["name"])!=str or type(attr["value"])!=float): error("Vector creation 2 - attribute type mismatch")
    if (ftr.items()["id"]!=1): error("Vector creation  2- int attribute mismatch")
    if (ftr.items()["name"]!="fred"): error("Vector creation 2 - str attribute mismatch")
    if (ftr.items()["value"]!=12.34): error("Vector creation 2 - float attribute mismatch")

    # Multiple GEOM features, attribute-type definition, srs-cast
    createVector(SUB_GEOMS, out3, srs=EPSG3035, fieldDef={"newField":"Real"}, fieldVals={"newField":range(3)} )

    ds = ogr.Open(out3)
    ly = ds.GetLayer()
    if (ly.GetFeatureCount()!=3): error("Vector creation 3 - feature count mismatch")
    if (not ly.GetSpatialRef().IsSame(EPSG3035)): error("Vector creation 3- srs mismatch")
    for i in range(3):
        ftr = ly.GetFeature(i)
        if not (i-ftr.items()["newField"])<0.0001: error("Vector creation 3 - value mismatch")

        geomCheck = SUB_GEOMS[i].Clone()
        geomCheck.TransformTo(EPSG3035)

        if not (ftr.GetGeometryRef().Area()-geomCheck.Area())<0.0001: error("Vector creation 3 - geometry area mismatch")

    # Multiple points, save in memory
    memVec = createVector(POINT_SET, srs=EPSG4326)

    ly = memVec.GetLayer()
    if ly.GetFeatureCount() != len(POINT_SET): error("Vector creation 4 - feature count mismatch")
    for i in range(len(POINT_SET)):
        ftr = ly.GetFeature(i)
        if not ftr.GetGeometryRef() != ogr.CreateGeometryFromWkt(POINT_SET[i]):
          error("Vector creation 4 - feature mismatch")

    print( "createVector passed" )

def test_mutateVector():
    # Setup
    ext_small = (6.1, 50.7, 6.25, 50.9)
    box_aachen = box(AACHEN_SHAPE_EXTENT, srs=EPSG4326)
    box_aachen.TransformTo(EPSG3035)

    sentance = ["Never","have","I","ever","ridden","on","a","horse","Did","you","know","that","?"]
    sentanceSmall = ["Never","have","I","ever","you"]

    ## simple repeater
    ps1 = mutateVector( AACHEN_POINTS, processor=None )
    
    res1 = extractFeatures(ps1)
    if res1.shape[0]!=13: error( "mutateVector 1 - item count")
    for i in range(13):
        if not res1.geom[i].GetSpatialReference().IsSame(EPSG4326): error("mutateVector 1 - geom srs")
        if not res1.geom[i].GetGeometryName()=="POINT": error("mutateVector 1 - geom type")
        if res1['word'][i] != sentance[i]: error("mutateVector 1 - attribute writing")

    ## spatial filtering
    ps2 = mutateVector( AACHEN_POINTS, processor=None, geom=ext_small )

    res2 = extractFeatures(ps2)
    if res2.shape[0]!=5: error( "mutateVector 2 - item count")
    for i in range(5):
        if not (res2['word'][i] == sentanceSmall[i]): error("mutateVector 2 - attribute writing")

    ## attribute and spatial filtering
    ps3 = mutateVector( AACHEN_POINTS, processor=None, geom=ext_small, where="id<5" )

    res3 = extractFeatures(ps3)
    if res3.shape[0]!=4: error( "mutateVector 3 - item count")
    for i in range(4):
        if not (res3['word'][i] == sentanceSmall[i]): error("mutateVector 3 - attribute writing")

    ## Test no items found
    ps4 = mutateVector( AACHEN_POINTS, processor=None, where="id<0" )

    if not ps4 is None: error("mutateVector 4 - no items found")

    ## Simple grower func ina new srs
    def growByWordLength(ftr):
        size = len(ftr["word"])*1000
        newGeom = ftr.geom.Buffer(size)

        return {'geom':newGeom, "size":size}

    output5 = result("mutateVector5.shp")
    mutateVector( AACHEN_POINTS, processor=growByWordLength, srs=EPSG3035, output=output5, overwrite=True)
    ps5 = loadVector(output5)

    res5 = extractFeatures(ps5)
    if res5.shape[0]!=13: error( "mutateVector 5 - item count")
    for i in range(13):
        if not res5.geom[i].GetSpatialReference().IsSame(EPSG3035): error("mutateVector 5 - geom srs")
        if not res5.geom[i].GetGeometryName()=="POLYGON": error("mutateVector 5 - geom type")
        if not (res5['word'][i] == sentance[i]): error("mutateVector 5 - attribute writing")
        if not (res5['size'][i] == len(sentance[i])*1000): error("mutateVector 5 - attribute writing")

        # test if the new areas are close to what they shoud be 
        area = 1000*len(sentance[i])*1000*len(sentance[i])*np.pi
        if not abs(1 - area/res5.geom[i].Area())<0.001: error("mutateVector 5 - geom area")

    ## Test inline processor, with filtering, and writign to file
    mutateVector( AACHEN_ZONES, srs=4326, geom=box_aachen, where="YEAR>2000", processor=lambda ftr: {'geom':ftr.geom.Centroid(), "YEAR":ftr["YEAR"]}, output=result("mutateVector6.shp"), overwrite=True)


    print( "mutateVector passed" )

def test_loadVector(): print("loadVector is trivial")
def test_vectorInfo(): print("vectorInfo is trivial")
def test_rasterize(): 

    # Simple vectorization to file
    r = rasterize(source=AACHEN_ZONES, pixelWidth=250, pixelHeight=250, output=result("rasterized1.tif"))
    mat1 = gk.raster.extractMatrix(r)
    compare(mat1.mean(), 0.13731291, "rasterization - simple")

    # Simple vectorization to mem
    r = rasterize(source=AACHEN_ZONES, pixelWidth=250, pixelHeight=250, )
    mat2 = gk.raster.extractMatrix(r)
    compare( (mat2-mat1).mean(), 0, "rasterization - memory")

    # Change srs to disc
    r = rasterize(source=AACHEN_ZONES, srs=4326, pixelWidth=0.01, pixelHeight=0.01, output=result("rasterized2.tif"))
    mat = gk.raster.extractMatrix(r)
    compare(mat.mean(), 0.12456277, "rasterization - simple")

    # Write attribute values to disc
    r = rasterize(source=AACHEN_ZONES, value="YEAR", pixelWidth=250, pixelHeight=250, output=result("rasterized3.tif"), noData=-1)
    mat = gk.raster.extractMatrix(r, autocorrect=True)
    compare(np.isnan(mat).sum(), 49570, "rasterization - nan values")
    compare(np.nanmean(mat), 1995.84283904, "rasterization - simple")

    # Write attribute values to mem, and use where clause
    r = rasterize(source=AACHEN_ZONES, value="YEAR", pixelWidth=250, pixelHeight=250, noData=-1, where="YEAR>2000")
    mat = gk.raster.extractMatrix(r, autocorrect=True)
    compare(np.isnan(mat).sum(), 54445, "rasterization - nan values")
    compare(np.nanmean(mat), 2004.96384743, "rasterization - simple") 

    print( "rasterize passed")   

if __name__=="__main__":
    test_loadVector()
    test_ogrType()
    test_countFeatures()
    test_vectorInfo()
    test_extractFeatures()
    test_extractFeature()
    test_createVector()
    test_mutateVector()
    test_rasterize()
