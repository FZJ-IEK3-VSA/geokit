from .helpers import *
from geokit import RegionMask, Extent, geom, vector, raster, util
import pytest


def test_RegionMask___init__():
    ext = Extent(0, 0, 100, 100, srs=EPSG3035)
    # test succeed
    rm = RegionMask(ext, 1, mask=MASK_DATA)
    rm = RegionMask(ext, (1, 1), mask=MASK_DATA)
    rm = RegionMask(ext, 2, geom=GEOM)

    # no mask or geometry given
    try:
        rm = RegionMask(ext, 1)
        assert False
    except util.GeoKitRegionMaskError as e:
        assert str(e) == 'Either mask or geom should be defined'
    else:
        assert False

    # bad pixel given
    try:
        rm = RegionMask(ext, (1, 3), geom=GEOM)
        assert False
    except util.GeoKitRegionMaskError as e:
        assert str(e) == 'The given extent does not fit the given pixelRes'
    else:
        assert False

    # bad mask given
    try:
        rm = RegionMask(ext, (1, 2), mask=MASK_DATA)
        assert False
    except util.GeoKitRegionMaskError as e:
        assert str(e) == 'Extent and pixels sizes do not correspond to mask shape'
    else:
        assert False

    # bad geom given
    try:
        rm = RegionMask(ext, (1, 2), geom="bananas")
        assert False
    except util.GeoKitRegionMaskError as e:
        assert str(e) == 'geom is not an ogr.Geometry object'
    else:
        assert False

    # bad geom given 2
    badGeom = GEOM.Clone()
    badGeom.AssignSpatialReference(None)

    try:
        rm = RegionMask(ext, (1, 2), geom=badGeom)
        assert False
    except util.GeoKitRegionMaskError as e:
        assert str(e) == 'geom does not have an srs'
    else:
        assert False


def test_RegionMask_fromMask():
    ext = Extent(0, 0, 100, 100, srs=EPSG3035)
    rm = RegionMask.fromMask(
        mask=MASK_DATA, extent=ext, attributes={"hats": 5})

    assert rm.mask.sum() == MASK_DATA.sum()  # mask
    assert rm.extent == ext  # extent
    assert rm.srs.IsSame(ext.srs)  # srs


def test_RegionMask_fromGeom():
    # fromGeom with wkt
    rm1 = RegionMask.fromGeom(geom.convertWKT(
        POLY, srs='latlon'), pixelRes=1000)
    assert (rm1.extent.xXyY == (4329000.0, 4771000.0, 835000.0, 1682000.0))
    assert (rm1.extent.srs.IsSame(EPSG3035))
    assert rm1.mask.sum() == 79274

    # fromGeom with geometry
    dxy = 0.05
    rm2 = RegionMask.fromGeom(GEOM, pixelRes=dxy, srs=EPSG4326, padExtent=0.2)

    assert rm2.extent == Extent(9.90, 30.30, 14.80, 38.30)
    assert rm2.extent.srs.IsSame(EPSG4326)

    g = GEOM.Clone()
    g.TransformTo(EPSG4326)
    assert np.isclose(rm2.mask.sum()*dxy*dxy, g.Area(), rtol=1e-3)

    # fromGeom with geometry and extent
    dxy = 0.05
    definedExtent = Extent(9.50, 30.25, 14.80, 38.35)
    rm3 = RegionMask.fromGeom(
        GEOM, pixelRes=dxy, srs=EPSG4326, extent=definedExtent)

    assert rm3.extent == definedExtent
    assert rm3.extent.srs.IsSame(EPSG4326)

    g = GEOM.Clone()
    g.TransformTo(EPSG4326)
    assert np.isclose(rm3.mask.sum()*dxy*dxy, g.Area(), rtol=1e-3)


def test_RegionMask_fromVector():
    # fromVector with a padded extent and defined srs
    rm0 = RegionMask.fromVector(
        AACHEN_SHAPE_PATH, pixelRes=0.001, srs=EPSG4326, padExtent=0.1)
    assert rm0.mask.sum() == 90296

    # fromVector - ID select
    rm1 = RegionMask.fromVector(MULTI_FTR_SHAPE_PATH, where=1)

    assert (rm1.extent == Extent(
        4069100, 2867000, 4109400, 2954000, srs=EPSG3035))
    assert (rm1.attributes["name"] == "dog")

    ds = ogr.Open(MULTI_FTR_SHAPE_PATH)
    lyr = ds.GetLayer()
    ftr = lyr.GetFeature(1)
    g = ftr.GetGeometryRef().Clone()
    g.TransformTo(EPSG3035)
    # check if total areas are close to one another
    assert np.isclose(rm1.mask.sum()*100*100, g.Area(), rtol=1e-3)

    # fromVector - 'where' select
    rm2 = RegionMask.fromVector(
        MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelRes=0.01, where="name='monkey'")

    assert (rm2.extent == Extent(6.83, 49.52, 7.53, 49.94))
    assert (rm2.attributes["id"] == 3)

    ftr = lyr.GetFeature(3)
    g = ftr.GetGeometryRef().Clone()
    g.TransformTo(EPSG4326)
    assert np.isclose(rm2.mask.sum(), 1948.0)

    # fromVector - 'where' select fail no features
    try:
        rm3 = RegionMask.fromVector(
            MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelRes=0.01, where="name='monkeyy'")
        assert False
    except util.GeoKitRegionMaskError as e:
        assert str(e) == 'Zero features found'
    else:
        assert False

    # fromVector - 'where' finds many features
    try:
        rm4 = RegionMask.fromVector(
            MULTI_FTR_SHAPE_PATH, srs=EPSG4326, pixelRes=0.01, where=r"name like 'mo%'")
        assert False
    except util.GeoKitRegionMaskError as e:
        assert 'Multiple fetures found' in str(e)
    else:
        assert False


@pytest.mark.skip("No test implemented")
def test_RegionMask_load():

    print("RegionMask_load not tested...")


def test_RegionMask_pixelRes():
    # test succeed
    rm1 = RegionMask.fromMask(Extent(0, 0, 100, 100, srs=EPSG3035), MASK_DATA)
    ps = rm1.pixelRes
    assert ps == 1

    # test fail
    rm2 = RegionMask.fromMask(Extent(0, 0, 100, 200, srs=EPSG3035), MASK_DATA)
    try:
        ps = rm2.pixelRes
        assert False
    except util.GeoKitRegionMaskError as e:
        assert str(
            e) == 'pixelRes only accessable when pixelWidth equals pixelHeight'
    else:
        assert False


def test_RegionMask_buildMask():
    # Build from another srs
    rm = RegionMask.load(AACHEN_SHAPE_PATH, srs=EPSG3035, pixelRes=100)
    assert np.isclose(rm.mask.sum(), 70944)
    assert np.isclose(rm.mask.std(), 0.498273451386)


@pytest.mark.skip("No test implemented")
def test_RegionMask_area():
    print("RegionMask_area not tested...")


def test_RegionMask_buildGeometry():
    # setup
    rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)
    rm2.buildMask()  # Be sure the mask is in place

    # Get the "real" geometry
    ds = ogr.Open(AACHEN_SHAPE_PATH)
    lyr = ds.GetLayer()
    ftr = lyr.GetFeature(0)
    realGeom = ftr.GetGeometryRef().Clone()
    realGeom.TransformTo(EPSG3035)

    # check initial geom and real geom area
    assert np.isclose(rm2.geometry.Area(), realGeom.Area(), rtol=1e-3)

    # destroy and recreate
    rm2.buildGeometry()
    assert np.isclose(rm2.geometry.Area(), realGeom.Area(), rtol=1e-3)


def test_RegionMask_vectorPath():
    rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)
    vec = rm2.vectorPath

    # Temp vector is created
    assert isfile(vec)

    # Temp vector is deleted
    del rm2
    assert not isfile(vec)


def test_RegionMask_vector():
    rm2 = RegionMask.fromVector(AACHEN_SHAPE_PATH)
    vec = rm2.vector
    vec.GetLayer()

    # Temp vector is created
    assert util.isVector(vec)

    # Temp vector is still not deleted
    del vec
    assert util.isVector(rm2._vector)


@pytest.mark.skip("No test implemented")
def test_RegionMask__repr_svg_():
    print("RegionMask__repr_svg_ not tested...")


@pytest.mark.skip("No test implemented")
def test_RegionMask_drawMask():
    print("RegionMask_drawMask not tested...")


@pytest.mark.skip("No test implemented")
def test_RegionMask_drawGeometry():
    print("RegionMask_drawGeometry not tested...")


def test_RegionMask_applyMask():
    # setup
    rm = RegionMask.fromGeom(geom.point(6.20, 50.75).Buffer(
        0.05), srs=EPSG4326, pixelRes=0.001)

    data1 = np.arange(rm.mask.size).reshape(rm.mask.shape)
    data2 = np.arange(rm.mask.shape[0]*3*rm.mask.shape[1]
                      * 3).reshape((rm.mask.shape[0]*3, rm.mask.shape[1]*3))

    # test applying
    data1 = rm.applyMask(data1)
    assert data1.sum() == 39296070
    assert np.isclose(data1.std(), 3020.0893432)

    data2 = rm.applyMask(data2.astype('int64'))
    assert data2.sum() == 3183264630
    assert np.isclose(data2.std(), 27182.1342973)

    #rm.createRaster(output=result("regionMask_applyMask_1.tif"), data=data1, overwrite=True)
    #rm.createRaster(3, output=result("regionMask_applyMask_2.tif"), data=data2, overwrite=True)


@pytest.mark.skip("No test implemented")
def test_RegionMask__returnBlank():
    print("RegionMask__returnBlank not tested...")


def test_RegionMask_indicateValues():
    # Setup
    rm = RegionMask.fromVector(AACHEN_SHAPE_PATH, pixelRes=0.001, srs=EPSG4326)

    # Testing valueMin (with srs change)
    res1 = rm.indicateValues(CLC_RASTER_PATH, value=(20, None))

    assert np.isclose(res1.sum(), 30969.6796875, 1e-6)
    assert np.isclose(res1.std(), 0.3489773, 1e-6)

    # Testing valueMax (with srs change)
    res2 = rm.indicateValues(CLC_RASTER_PATH, value=(None, 24))
    assert np.isclose(res2.sum(), 82857.5078125, 1e-6)
    assert np.isclose(res2.std(), 0.4867994, 1e-6)

    # Testing valueEquals (with srs change)
    res3 = rm.indicateValues(CLC_RASTER_PATH, value=7, resampleAlg="cubic")
    assert np.isclose(res3.sum(), 580.9105835, 1e-4)
    assert np.isclose(res3.std(), 0.0500924, 1e-6)

    # Testing range
    res4 = rm.indicateValues(CLC_RASTER_PATH, value=(20, 24))

    combi = np.logical_and(res1 > 0.5, res2 > 0.5)
    # Some pixels will not end up the same due to warping issues
    assert ((res4 > 0.5) != combi).sum() < res4.size*0.001

    # Testing buffering
    res5 = rm.indicateValues(CLC_RASTER_PATH, value=(
        1, 2), buffer=0.01, resolutionDiv=2, forceMaskShape=True)
    assert np.isclose(res5.sum(), 65030.75000000, 1e-4)

    # make sure we get an empty mask when nothing is indicated
    res6 = rm.indicateValues(CLC_RASTER_PATH, value=2000, buffer=0.01,
                             resolutionDiv=2, forceMaskShape=True, noData=-1)
    assert np.isclose(res6.sum(), -113526.0, 1e-4)


def test_RegionMask_indicateFeatures():
    # setup
    rm = RegionMask.fromVector(AACHEN_SHAPE_PATH)

    # Simple case
    res = rm.indicateFeatures(NATURA_PATH, where="SITECODE='DE5404303'")
    #print("%.7f"%res.sum(), "%.7f"%res.std())
    assert np.isclose(res.sum(), 649, 1e-6)
    assert np.isclose(res.std(), 0.0646270, 1e-6)

    # Buffered Cases
    res2 = rm.indicateFeatures(
        NATURA_PATH, where="SITETYPE='B'", buffer=300, resolutionDiv=3, forceMaskShape=True)
    #print("%.7f"%res2.sum(), "%.7f"%res2.std())
    assert np.isclose(res2.sum(), 13670.5555556, 1e-6)

    res3 = rm.indicateFeatures(NATURA_PATH, where="SITETYPE='B'", buffer=300,
                               bufferMethod='area', resolutionDiv=5, forceMaskShape=True)
    #print("%.7f"%res3.sum(), "%.7f"%res3.std())
    assert np.isclose(res3.sum(), 13807.320000, 1e-6)

    # No indication case
    res4 = rm.indicateFeatures(NATURA_PATH, where="SITETYPE='D'", buffer=300,
                               bufferMethod='area', resolutionDiv=2, forceMaskShape=True, noData=-1)
    #print("%.7f"%res4.sum(), "%.7f"%res4.std())

    assert np.isclose(res4.sum(), -83792, 1e-6)


@pytest.mark.skip("No test implemented")
def test_RegionMask_indicateGeoms():
    print("RegionMask_indicateGeoms not tested...")


@pytest.mark.skip("No test implemented")
def test_RegionMask_subRegions():
    print("RegionMask_subRegions not tested...")


def test_RegionMask_createRaster():
    rm = RegionMask.fromGeom(geom.point(6.20, 50.75).Buffer(
        0.05), srs=EPSG4326, pixelRes=0.001)

    # Create a raster like the mask
    ds = rm.createRaster()

    dsInfo = raster.rasterInfo(ds)
    assert np.isclose(dsInfo.xMin,  6.15)
    assert np.isclose(dsInfo.xMax,  6.25)
    assert np.isclose(dsInfo.yMin, 50.70)
    assert np.isclose(dsInfo.yMax, 50.80)
    assert dsInfo.srs.IsSame(EPSG4326)
    assert dsInfo.dtype == gdal.GDT_Byte

    # Fill a raster with mask data
    out2 = result("rasterMast_createRaster_2.tif")
    rm.createRaster(output=out2, data=rm.mask, overwrite=True)

    ds = gdal.Open(out2)
    band = ds.GetRasterBand(1)
    assert np.isclose(band.ReadAsArray(), rm.mask).all()

    # The function is not meant for scaling down
    # # test Scaling down
    # scaledData = scaleMatrix(rm.mask,-4)
    # ds = rm.createRaster(resolutionDiv=1/4, data=scaledData, overwrite=True)

    # band = ds.GetRasterBand(1)
    # if (band.ReadAsArray()-scaledData).any(): error("createRaster 3 - data mismatch")

    # test Scaling up
    scaledData = util.scaleMatrix(rm.mask, 2)

    ds = rm.createRaster(resolutionDiv=2, data=scaledData, overwrite=True)
    band = ds.GetRasterBand(1)
    assert np.isclose(band.ReadAsArray(), scaledData).all()


def test_RegionMask_warp():
    # setup
    rm_3035 = RegionMask.fromGeom(geom.point(6.20, 50.75).Buffer(0.05))
    rm = RegionMask.fromGeom(geom.point(6.20, 50.75).Buffer(
        0.05), srs=EPSG4326, pixelRes=0.0005)

    # basic warp Raster
    warped_1 = rm_3035.warp(CLC_RASTER_PATH)

    assert warped_1.dtype == np.uint8
    assert warped_1.shape == rm_3035.mask.shape
    assert np.isclose(warped_1.sum(), 88128)
    assert np.isclose(warped_1.std(), 9.52214123991)
    #rm_3035.createRaster(data=warped_1, output=result("regionMask_warp_1.tif"), overwrite=True)

    # basic warp Raster (FLIP CHECK!)
    warped_1f = rm_3035.warp(CLC_FLIPCHECK_PATH)

    assert warped_1f.dtype == np.uint8
    assert warped_1f.shape == rm_3035.mask.shape
    assert np.isclose(warped_1f.sum(), 88128)
    assert np.isclose(warped_1f.std(), 9.52214123991)
    #rm_3035.createRaster(data=warped_1f, output=result("regionMask_warp_1f.tif"), overwrite=True)

    assert (warped_1 == warped_1f).all()

    # basic warp Raster with srs change
    warped_2 = rm.warp(CLC_RASTER_PATH)
    assert warped_2.dtype == np.uint8
    assert warped_2.shape == rm.mask.shape
    assert np.isclose(warped_2.sum(), 449627)
    assert np.isclose(warped_2.std(), 9.07520801659)
    #rm.createRaster(data=warped_2, output=result("regionMask_warp_2.tif"), overwrite=True)

    # Define resample alg and output type
    warped_3 = rm.warp(CLC_RASTER_PATH, dtype="float", resampleAlg='near')

    assert warped_3.dtype == np.float64
    assert warped_3.shape == rm.mask.shape
    assert np.isclose(warped_3.sum(), 449317.0)
    assert np.isclose(warped_3.std(), 9.37570375729)
    #rm.createRaster(data=warped_3, output=result("regionMask_warp_3.tif"), overwrite=True)

    # define a resolution div
    warped_4 = rm.warp(CLC_RASTER_PATH, resolutionDiv=5,
                       resampleAlg='near', noData=0)

    assert warped_4.dtype == np.uint8
    assert warped_4.shape == (rm.mask.shape[0]*5, rm.mask.shape[1]*5)
    assert np.isclose(warped_4.sum(), 11240881)
    assert np.isclose(warped_4.std(), 9.37633272361)
    #rm.createRaster(5, data=warped_4, output=result("regionMask_warp_4.tif"), noData=0, overwrite=True)


def test_RegionMask_rasterize():
    # setup
    rm = RegionMask.fromVector(AACHEN_SHAPE_PATH, pixelRes=0.001, srs=EPSG4326)

    # simple rasterize
    rasterize_1 = rm.rasterize(AACHEN_ZONES)

    assert rasterize_1.dtype == np.uint8
    assert rasterize_1.shape == rm.mask.shape
    assert np.isclose(rasterize_1.sum(), 47191)
    assert np.isclose(rasterize_1.std(), 0.42181050527)
    #rm.createRaster(data=rasterize_1, output=result("regionMask_rasterize_1.tif"), overwrite=True)

    # attribute rasterizing
    rasterize_2 = rm.rasterize(AACHEN_ZONES, value="YEAR", dtype="int16")

    assert rasterize_2.dtype == np.int16
    assert rasterize_2.shape == rm.mask.shape
    assert np.isclose(rasterize_2.sum(), 94219640)
    assert np.isclose(rasterize_2.std(), 842.177748527)
    #rm.createRaster(data=rasterize_2, output=result("regionMask_rasterize_2.tif"), overwrite=True)

    # where statement and resolution div
    rasterize_3 = rm.rasterize(
        AACHEN_ZONES, value=10, resolutionDiv=5, where="YEAR>2000", dtype=float)

    assert rasterize_3.dtype == np.float64
    assert rasterize_3.shape == (rm.mask.shape[0]*5, rm.mask.shape[1]*5)
    assert np.isclose(rasterize_3.sum(), 4578070.0)
    assert np.isclose(rasterize_3.std(), 2.85958813405)
    #rm.createRaster(data=scaleMatrix(rasterize_3,-5), output=result("regionMask_rasterize_3.tif"), overwrite=True)


@pytest.mark.skip("No test implemented")
def test_RegionMask_extractFeatures():
    print("RegionMask_extractFeatures not tested...")


@pytest.mark.skip("No test implemented")
def test_RegionMask_mutateVector():
    print("RegionMask_mutateVector not tested...")


@pytest.mark.skip("No test implemented")
def test_RegionMask_mutateRaster():
    print("RegionMask_mutateRaster not tested...")


@pytest.mark.skip("No test implemented")
def test_contoursFromRaster():
    print("Nothing to do :(")


@pytest.mark.skip("No test implemented")
def test_contoursFromMatrix():
    print("Nothing to do :(")


@pytest.mark.skip("No test implemented")
def test_contoursFromMask():
    print("Nothing to do :(")
