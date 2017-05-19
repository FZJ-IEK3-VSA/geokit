from helpers import *

def makePoint_():
  print("MAKE 'makePoint' TESTER!!!!!!!!!!!!")
def makeEmptyGeom_():
  print("MAKE 'makeEmptyGeom' TESTER!!!!!!!!!!!!")
def makeGeomFromWkt_():
  print("MAKE 'makeGeomFromWkt' TESTER!!!!!!!!!!!!")
def makeGeomFromMask_():
  print("MAKE 'makeGeomFromMask' TESTER!!!!!!!!!!!!")
def geomTransform_():
  print("MAKE 'geomTransform' TESTER!!!!!!!!!!!!")
def geomListFlatten_():
  print("MAKE 'geomListFlatten' TESTER!!!!!!!!!!!!")
def coordTransform_():
  print("MAKE 'coordTransform' TESTER!!!!!!!!!!!!")
def vectorCount_():
  print("MAKE 'vectorCount' TESTER!!!!!!!!!!!!")
def vectorInfo_():
  print("MAKE 'vectorInfo' TESTER!!!!!!!!!!!!")
def vectorMutate_():
  print("MAKE 'vectorMutate' TESTER!!!!!!!!!!!!")

## box
def makeBox_():
  # fun func
  b1 = makeBox(0,0,5,10, srs=EPSG3035)
  
  # check results
  if( b1.Area() != 50 ): error("Box Creation")

def vectorItems_():
  # test basic
  vi = list(vectorItems(BOXES))
  
  if len(vi)!=3: error("vectorItems 1 - count mismatch") 

  if (vi[0][0].Area()!=1.0): error("vectorItems 1 - geom mismatch")
  if (vi[0][1]['name']!="harry"): error("vectorItems 1 - attribute mismatch")

  if (vi[1][0].Area()!=4.0): error("vectorItems 1 - geom mismatch")
  if (vi[1][1]['name']!="ron"): error("vectorItems 1 - attribute mismatch")

  if (vi[2][0].Area()!=9.0): error("vectorItems 1 - geom mismatch")
  if (vi[2][1]['name']!="hermoine"): error("vectorItems 1 - attribute mismatch")

  # test clip
  vi = list(vectorItems(BOXES, geom=makeBox(0,0,3,3, EPSG4326)))

  if len(vi)!=2: error("vectorItems 2 - count mismatch")   

  if (vi[0][0].Area()!=1.0): error("vectorItems 2 - geom mismatch")
  if (vi[0][1]['name']!="harry"): error("vectorItems 2 - attribute mismatch")

  if (vi[1][0].Area()!=4.0): error("vectorItems 2 - geom mismatch")
  if (vi[1][1]['name']!="ron"): error("vectorItems 2 - attribute mismatch")

  # test srs change and attribute filter
  vi = list(vectorItems(BOXES, where="smart>0", outputSRS=EPSG3035))

  if len(vi)!=1: error("vectorItems 3 - count mismatch")   

  if (not vi[0][0].GetSpatialReference().IsSame(EPSG3035)): error("vectorItems 3 - srs mismatch")
  if (vi[0][1]['name']!="hermoine"): error("vectorItems 3 - attribute mismatch")

def vectorItem_():
  print("MAKE 'vectorItem' TESTER!!!!!!!")


## ogrType
def ogrType_():
  if( ogrType(bool) != "OFTInteger" ): error("ogr type")
  if( ogrType("float32") != "OFTReal" ): error("ogr type")
  if( ogrType("Integer64") != "OFTInteger64" ): error("ogr type")
  if( ogrType(NUMPY_FLOAT_ARRAY) != "OFTReal" ): error("ogr type")
  if( ogrType(NUMPY_FLOAT_ARRAY.dtype) != "OFTReal" ): error("ogr type")

## Create shape file
def createVector_():
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

def vectorMutate_():
  # Setup
  ext_small = Extent.from_xyXY((6.1,50.7,6.25,50.9), EPSG4326)
  ext_aachen = Extent.fromVector(AACHEN_SHAPE_PATH)

  sentance = ["Never","have","I","ever","ridden","on","a","horse","Did","you","know","that","?"]
  sentanceSmall = ["Never","have","I","ever","you"]

  ## simple repeater
  ps1 = vectorMutate( AACHEN_POINTS )

  res1 = list(vectorItems(ps1))
  if len(res1)!=13: error( "vectorMutate 1 - item count")
  for i in range(13):
    if not res1[i][0].GetSpatialReference().IsSame(EPSG4326): error("vectorMutate 1 - geom srs")
    if not res1[i][0].GetGeometryName()=="POINT": error("vectorMutate 1 - geom type")
    if res1[i][1]['word'] != sentance[i]: error("vectorMutate 1 - attribute writing")

  ## spatial filtering
  ps2 = vectorMutate( AACHEN_POINTS, extent=ext_small )

  res2 = list(vectorItems(ps2))
  if len(res2)!=5: error( "vectorMutate 2 - item count")
  for i in range(5):
    if not (res2[i][1]['word'] == sentanceSmall[i]): error("vectorMutate 2 - attribute writing")

  ## attribute and spatial filtering
  ps3 = vectorMutate( AACHEN_POINTS, extent=ext_small, where="id<5" )

  res3 = list(vectorItems(ps3))
  if len(res3)!=4: error( "vectorMutate 3 - item count")
  for i in range(4):
    if not (res3[i][1]['word'] == sentanceSmall[i]): error("vectorMutate 3 - attribute writing")

  ## Test no items found
  ps4 = vectorMutate( AACHEN_POINTS, where="id<0" )

  if not ps4 is None: error("vectorMutate 4 - no items found")

  ## Simple grower func ina new srs
  def growByWordLength(g,i):
    size = len(i["word"])*1000
    newGeom = g.Buffer(size)
    i['size'] = size

    return newGeom,i

  ps5 = vectorMutate( AACHEN_POINTS, processor=growByWordLength, srs=EPSG3035)#, output="DELETEME.shp" )

  res5 = list(vectorItems(ps5))
  if len(res5)!=13: error( "vectorMutate 5 - item count")
  for i in range(13):
    if not res5[i][0].GetSpatialReference().IsSame(EPSG3035): error("vectorMutate 5 - geom srs")
    if not res5[i][0].GetGeometryName()=="POLYGON": error("vectorMutate 5 - geom type")
    if not (res5[i][1]['word'] == sentance[i]): error("vectorMutate 5 - attribute writing")
    if not (res5[i][1]['size'] == len(sentance[i])*1000): error("vectorMutate 5 - attribute writing")

    # test if the new areas are close to what they shoud be 
    area = 1000*len(sentance[i])*1000*len(sentance[i])*np.pi
    if not abs(1 - area/res5[i][0].Area())<0.001: error("vectorMutate 5 - geom area")

  ## Test inline processor, with filtering, and writign to file
  vectorMutate( AACHEN_ZONES, extent=ext_aachen, where="YEAR>2000", processor=lambda g,i: (g.Centroid(), {"YEAR":i["YEAR"]}), output=result("algorithms_vectorMutate_6.shp"), overwrite=True)

if __name__=="__main__":
  makePoint_()
  makeEmptyGeom_()
  makeGeomFromWkt_()
  makeGeomFromMask_()
  geomTransform_()
  geomListFlatten_()
  coordTransform_()
  vectorCount_()
  vectorInfo_()
  vectorMutate_()
  makeBox_()
  vectorItems_()
  vectorItem_()
  ogrType_()
  createVector_()