from .helpers import *
from geokit import vector, raster, geom, util
from os.path import join, dirname
import pandas as pd
import pytest
from functools import reduce

# ogrType


def test_ogrType():
    assert vector.ogrType(bool) == "OFTInteger"
    assert vector.ogrType("float32") == "OFTReal"
    assert vector.ogrType("Integer64") == "OFTInteger64"
    assert vector.ogrType(NUMPY_FLOAT_ARRAY) == "OFTReal"
    assert vector.ogrType(NUMPY_FLOAT_ARRAY.dtype) == "OFTReal"


def test_countFeatures():
    # Simple vector count
    assert vector.countFeatures(MULTI_FTR_SHAPE_PATH) == 4

    #  same SRS, geom filter
    cnt = vector.countFeatures(
        MULTI_FTR_SHAPE_PATH, geom=geom.box(5.89, 48.77, 6.89, 49.64, srs=EPSG4326)
    )
    assert cnt == 2

    # different SRS, geom filter
    cnt = vector.countFeatures(
        MULTI_FTR_SHAPE_PATH,
        geom=geom.box(4022802, 2867575, 4104365, 2938843, srs=EPSG3035),
    )
    assert cnt == 2

    # where filter
    cnt = vector.countFeatures(MULTI_FTR_SHAPE_PATH, where="name LIKE 'mo%'")
    assert cnt == 2


def test_extractFeatures():
    # test basic
    vi = list(vector.extractFeatures(BOXES, asPandas=False))

    assert len(vi) == 3  # count mismatch

    assert vi[0][0].Area() == 1.0  # geom mismatch
    assert vi[0][1]["name"] == "harry"  # attribute mismatch

    assert vi[1][0].Area() == 4.0  # geom mismatch
    assert vi[1][1]["name"] == "ron"  # attribute mismatch

    assert vi[2][0].Area() == 9.0  # geom mismatch
    assert vi[2][1]["name"] == "hermoine"  # attribute mismatch

    # test clip
    vi = list(
        vector.extractFeatures(
            BOXES, geom=geom.box(0, 0, 3, 3, srs=EPSG4326), asPandas=False
        )
    )

    assert len(vi) == 2  # count mismatch

    assert vi[0][0].Area() == 1.0  # geom mismatch
    assert vi[0][1]["name"] == "harry"  # attribute mismatch

    assert vi[1][0].Area() == 4.0  # geom mismatch
    assert vi[1][1]["name"] == "ron"  # attribute mismatch

    # test srs change and attribute filter
    vi = list(
        vector.extractFeatures(BOXES, where="smart>0", srs=EPSG3035, asPandas=False)
    )

    assert len(vi) == 1  # count mismatch
    assert vi[0][0].GetSpatialReference().IsSame(EPSG3035)  # srs mismatch
    assert vi[0][1]["name"] == "hermoine"  # attribute mismatch

    # Test loading as a dataframe
    vi = vector.extractFeatures(BOXES, asPandas=True)
    assert vi.shape == (3, 3)  # shape mismatch

    assert vi.geom[0].Area() == 1.0  # geom mismatch
    assert vi["name"][0] == "harry"  # attribute mismatch

    assert vi.geom[1].Area() == 4.0  # geom mismatch
    assert vi["name"][1] == "ron"  # attribute mismatch

    assert vi.geom[2].Area() == 9.0  # geom mismatch
    assert vi["name"][2] == "hermoine"  # attribute mismatch

    # Test spatialPredicate
    # define a filter geom such that box "harry" is overlapping, box
    # "ron" is included and box "hermoine" is touching only
    _filtergeom = geom.box(0.5, 0.5, 5, 7)
    vi = vector.extractFeatures(BOXES, geom=_filtergeom, spatialPredicate="Overlaps")

    # hermoine must now be gone, so make sure we have length 2
    assert vi.shape == (2, 3)  # shape mismatch

    assert vi.geom[0].Area() == 1.0  # geom mismatch
    assert vi["name"][0] == "harry"  # attribute mismatch

    assert vi.geom[1].Area() == 4.0  # geom mismatch
    assert vi["name"][1] == "ron"  # attribute mismatch

    # sub test: make sure that also boxes are extracted that are LARGER than the filter geom when spatialPredicate="Overlaps"
    vi = vector.extractFeatures(
        vector.createVector(geom.polygon([(0, 0), (0, 8), (8, 8), (8, 0)])),
        geom=_filtergeom,
        spatialPredicate="Overlaps",
    )

    # hagrid must be considered even if larger in all directions than the _filtergeom
    assert vi.shape == (1, 1)  # single geom found

    assert vi.geom[0].Area() == 64.0  # geom area mismatch

    # now extract only the geom whose centroid is within this same filter geom
    vi = vector.extractFeatures(
        BOXES, geom=_filtergeom, spatialPredicate="CentroidWithin"
    )

    # hermoine AND harry must now be gone, so make sure we have length 1
    assert vi.shape == (1, 3)  # shape mismatch

    assert vi.geom[0].Area() == 4.0  # geom mismatch
    assert vi["name"][0] == "ron"  # attribute mismatch


def test_extractFeature():
    # test succeed
    geom, attr = vector.extractFeature(BOXES, where=1)
    assert geom.Area() == 4.0  # geom mismatch
    assert attr["name"] == "ron"  # attribute mismatch

    geom, attr = vector.extractFeature(BOXES, where="name='harry'")
    assert geom.Area() == 1.0  # geom mismatch
    assert attr["name"] == "harry"  # attribute mismatch

    # test fail
    try:
        geom, attr = vector.extractFeature(BOXES, where="smart=0")
        assert False
    except util.GeoKitError:
        assert True
    else:
        assert False


def test_extractAndClipFeatures():
    multiFeatures = vector.extractFeatures(MULTI_FTR_SHAPE_PATH)
    multiFeatures["testAttr"] = 100
    box = geom.box((6.6, 49.0, 7.6, 50.0), srs=4326)

    clipped = vector.extractAndClipFeatures(
        source=multiFeatures, geom=box, where="id < 4", scaleAttrs="testAttr"
    )

    assert len(clipped) == 2
    assert all(np.isclose(clipped.areaShare.values, np.array([0.827164, 1.0])))
    assert all(np.isclose(clipped.testAttr.values, np.array([82.716413, 100.0])))


# Create shape file


def test_createVector(tmpdir):
    # Setup
    out1 = result("util_shape1.shp")
    out2 = result("util_shape2.shp")
    out3 = result("util_shape3.shp")

    ######################
    # run and check

    # assure that check for directory works as expected
    with pytest.raises(FileNotFoundError):
        vector.createVector(
            geoms=POLY,
            output=join(dirname(__file__), "nonexisting_folder", "util_shape1.shp"),
            srs=EPSG4326,
            overwrite=True,
        )

    # Single WKT feature, no attributes
    vector.createVector(geoms=POLY, output=out1, srs=EPSG4326, overwrite=True)

    ds = ogr.Open(out1)
    ly = ds.GetLayer()
    assert ly.GetFeatureCount() == 1  # feature count mismatch
    assert ly.GetSpatialRef().IsSame(EPSG4326)  # srs mismatch

    # Single GEOM feature, with attributes
    vector.createVector(
        GEOM_3035,
        out2,
        fieldVals={
            "id": 1,
            "name": [
                "fred",
            ],
            "value": 12.34,
        },
        fieldDef={"id": "int8", "name": str, "value": float},
        overwrite=True,
    )

    ds = ogr.Open(out2)
    ly = ds.GetLayer()
    ftr = ly.GetFeature(0)
    attr = ftr.items()
    assert type(attr["id"]) == int  # attribute type mismatch
    assert type(attr["name"]) == str  # attribute type mismatch
    assert type(attr["value"]) == float  # attribute type mismatch
    assert ftr.items()["id"] == 1  # int attribute mismatch
    assert ftr.items()["name"] == "fred"  # str attribute mismatch
    assert ftr.items()["value"] == 12.34  # float attribute mismatch

    # Multiple GEOM features, attribute-type definition, srs-cast
    vector.createVector(
        SUB_GEOMS,
        out3,
        srs=EPSG3035,
        fieldDef={"newField": "Real"},
        fieldVals={"newField": range(3)},
    )

    ds = ogr.Open(out3)
    ly = ds.GetLayer()
    assert ly.GetFeatureCount() == 3  # feature count mismatch
    assert ly.GetSpatialRef().IsSame(EPSG3035)  # srs mismatch
    for i in range(3):
        ftr = ly.GetFeature(i)
        assert np.isclose(i, ftr.items()["newField"])

        geomCheck = SUB_GEOMS[i].Clone()
        geomCheck.TransformTo(EPSG3035)

        assert np.isclose(ftr.GetGeometryRef().Area(), geomCheck.Area())

    # Multiple points, save in memory
    memVec = vector.createVector(POINT_SET, srs=EPSG4326)

    ly = memVec.GetLayer()
    assert ly.GetFeatureCount() == len(POINT_SET)
    for i in range(len(POINT_SET)):
        ftr = ly.GetFeature(i)
        assert ftr.GetGeometryRef() != ogr.CreateGeometryFromWkt(POINT_SET[i])

    # Multiple points, save to shapefile
    output_shp = tmpdir.mkdir("temp_1").join("mpoints_temp.shp").__str__()

    # create a vector in memory
    vec_shp = vector.extractFeatures(vector.createVector(POINT_SET, srs=EPSG4326))

    # create a vector on disk
    vector.createVector(POINT_SET, output=output_shp, srs=EPSG4326)

    vec_shp_disk = vector.extractFeatures(output_shp)

    assert len(vec_shp) == len(vec_shp_disk)
    test_equal_shp = util.compare_geoms(
        vec_shp["geom"].to_list(), vec_shp_disk["geom"].to_list()
    )

    assert all(test_equal_shp)
    assert all([g.IsValid() for g in vec_shp_disk["geom"]])

    # Multiple point layers, save to geopackage
    output_gpkg = tmpdir.mkdir("temp_2").join("mpoints_temp.gpkg").__str__()

    # create new vector layer in memory
    vec_gpkg_lyr_1 = vector.extractFeatures(
        vector.createVector(POINT_SET, layerName="layer_1", srs=EPSG4326)
    )

    # create new geopackage on disk
    vector.createVector(
        POINT_SET, output=output_gpkg, layerName="layer_1", srs=EPSG4326
    )

    # append new layer to existing geopackage
    vector.createVector(
        POINT_SET,
        output=output_gpkg,
        layerName="layer_2",
        srs=EPSG4326,
        overwrite=False,
    )

    layers = vector.listLayers(output_gpkg)
    assert len(layers) == 2
    assert "layer_1" in layers
    assert "layer_2" in layers

    vec_gpkg_lyr_1_disk = vector.extractFeatures(output_gpkg, layerName="layer_1")

    assert len(vec_gpkg_lyr_1) == len(vec_gpkg_lyr_1_disk)
    test_equal_gpkg = util.compare_geoms(
        vec_shp["geom"].to_list(), vec_shp_disk["geom"].to_list()
    )
    assert all(test_equal_gpkg)

    assert all([g.IsValid() for g in vec_gpkg_lyr_1_disk["geom"]])


def test_mutateVector():
    # Setup
    ext_small = (6.1, 50.7, 6.25, 50.9)
    box_aachen = geom.box(AACHEN_SHAPE_EXTENT, srs=EPSG4326)
    box_aachen.TransformTo(EPSG3035)

    sentance = [
        "Never",
        "have",
        "I",
        "ever",
        "ridden",
        "on",
        "a",
        "horse",
        "Did",
        "you",
        "know",
        "that",
        "?",
    ]
    sentanceSmall = ["Never", "have", "I", "ever", "you"]

    # simple repeater
    ps1 = vector.mutateVector(AACHEN_POINTS, processor=None)

    res1 = vector.extractFeatures(ps1)
    assert res1.shape[0] == 13  # item count")
    for i in range(13):
        assert res1.geom[i].GetSpatialReference().IsSame(EPSG4326)  # geom srs
        assert res1.geom[i].GetGeometryName() == "POINT"  # geom type
        assert res1["word"][i] == sentance[i]  # attribute writing

    # spatial filtering
    ps2 = vector.mutateVector(AACHEN_POINTS, processor=None, geom=ext_small)

    res2 = vector.extractFeatures(ps2)
    assert res2.shape[0] == 5  # item count
    for i in range(5):
        assert res2["word"][i] == sentanceSmall[i]  # attribute writing

    # attribute and spatial filtering
    ps3 = vector.mutateVector(
        AACHEN_POINTS, processor=None, geom=ext_small, where="id<5"
    )

    res3 = vector.extractFeatures(ps3)
    assert res3.shape[0] == 4  # item count
    for i in range(4):
        assert res3["word"][i] == sentanceSmall[i]  # attribute writing

    # Test no items found
    ps4 = vector.mutateVector(AACHEN_POINTS, processor=None, where="id<0")

    assert ps4 is None  # no items found

    # Simple grower func in a new srs
    def growByWordLength(ftr):
        size = len(ftr["word"]) * 10
        newGeom = ftr.geom.Buffer(float(size))

        return {"geom": newGeom, "size": size}

    output5 = result("mutateVector5.shp")
    vector.mutateVector(
        AACHEN_POINTS,
        processor=growByWordLength,
        srs=EPSG3035,
        output=output5,
        overwrite=True,
    )
    ps5 = vector.loadVector(output5)

    res5 = vector.extractFeatures(ps5)
    assert res5.shape[0] == 13  # item count
    for i in range(13):
        assert res5.geom[i].GetSpatialReference().IsSame(EPSG3035)  # geom srs
        assert res5.geom[i].GetGeometryName() == "POLYGON"  # geom type
        assert res5["word"][i] == sentance[i]  # attribute writing
        assert res5["size"][i] == len(sentance[i]) * 10  # attribute writing

        # test if the new areas are close to what they shoud be
        area = np.power(10 * len(sentance[i]), 2) * np.pi
        assert np.isclose(area, res5.geom[i].Area(), rtol=1.0e-3)  # geom area

    # Test inline processor, with filtering, and writing to file
    vector.mutateVector(
        AACHEN_ZONES,
        srs=4326,
        geom=box_aachen,
        where="YEAR>2000",
        processor=lambda ftr: {"geom": ftr.geom.Centroid(), "YEAR": ftr["YEAR"]},
        output=result("mutateVector6.shp"),
        overwrite=True,
    )


def test_loadVector():
    assert util.isVector(vector.loadVector(BOXES))


def test_vectorInfo():
    vi = vector.vectorInfo(AACHEN_ZONES)
    for a, b in zip(
        vi.attributes,
        [
            "PK_UID",
            "SITE_CODE",
            "PARENT_ISO",
            "ISO3",
            "SITE_NAME",
            "SITE_AREA",
            "YEAR",
            "DESIGNATE",
            "CDDA_disse",
        ],
    ):
        assert a == b

    assert np.isclose(
        vi.bounds,
        (4037376.3605322, 3045182.1758945677, 4092345.479879612, 3109991.6386917345),
    ).all()

    assert np.isclose(vi.count, 115)
    assert vi.srs.IsSame(EPSG3035)
    assert np.isclose(vi.xMin, 4037376.3605322)
    assert np.isclose(vi.yMin, 3045182.1758945677)
    assert np.isclose(vi.xMax, 4092345.479879612)
    assert np.isclose(vi.yMax, 3109991.6386917345)


def test_rasterize():
    # Simple vectorization to file
    r = vector.rasterize(
        source=AACHEN_ZONES,
        pixelWidth=250,
        pixelHeight=250,
        output=result("rasterized1.tif"),
    )
    mat1 = raster.extractMatrix(r)
    assert np.isclose(mat1.mean(), 0.13910192)

    # Simple vectorization to mem
    r = vector.rasterize(
        source=AACHEN_ZONES,
        pixelWidth=250,
        pixelHeight=250,
    )
    mat2 = raster.extractMatrix(r)
    assert np.isclose(mat2, mat1).all()

    # Change srs to disc
    r = vector.rasterize(
        source=AACHEN_ZONES,
        srs=4326,
        pixelWidth=0.01,
        pixelHeight=0.01,
        output=result("rasterized2.tif"),
    )
    mat = raster.extractMatrix(r)
    assert np.isclose(mat.mean(), 0.12660478)

    # Write attribute values to disc
    r = vector.rasterize(
        source=AACHEN_ZONES,
        value="YEAR",
        pixelWidth=250,
        pixelHeight=250,
        output=result("rasterized3.tif"),
        noData=-1,
    )
    mat = raster.extractMatrix(r, autocorrect=True)
    assert np.isclose(np.isnan(mat).sum(), 48831)
    assert np.isclose(np.nanmean(mat), 1995.84283904)

    # Write attribute values to mem, and use where clause
    r = vector.rasterize(
        source=AACHEN_ZONES,
        value="YEAR",
        pixelWidth=250,
        pixelHeight=250,
        noData=-1,
        where="YEAR>2000",
    )
    mat = raster.extractMatrix(r, autocorrect=True)
    assert np.isclose(np.isnan(mat).sum(), 53706)
    assert np.isclose(np.nanmean(mat), 2004.96384743)


def test_createGeoDataFrame():
    import geopandas as gpd

    dfIn = pd.DataFrame()
    dfIn["geom"] = [geom.point(7, 51, srs=3857), geom.point(7.5, 52, srs=3857)]
    dfIn["data_column"] = ["abc", 123]

    dfOut = vector.createGeoDataFrame(dfGeokit=dfIn)

    assert "data_column" in dfOut.columns
    assert (dfOut["data_column"] == dfIn["data_column"]).all()
    assert (
        gpd.points_from_xy([7, 7.5], [51, 52], [0, 0], crs="EPSG:3857")
        == dfOut["geometry"]
    ).all()
    assert dfOut["geometry"].crs.to_epsg() == 3857
