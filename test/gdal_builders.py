"""Rasters and vectors built with plain GDAL, so that GeoKit's type logic is not involved in the inputs."""

from osgeo import gdal, ogr, osr

import geokit


def spatial_reference_from_epsg(epsg: int) -> osr.SpatialReference:
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(epsg)
    spatial_reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return spatial_reference


def create_gdal_raster(
    matrix,
    gdal_type,
    noData=None,
    scale=None,
    offset=None,
    epsg=3035,
    pixel_size=100,
    x_min=0.0,
    y_max=None,
    driver="MEM",
    path="",
):
    """A raster built with plain GDAL. ``y_max`` is the top edge (defaults to rows * pixel_size)."""
    rows, columns = matrix.shape
    if y_max is None:
        y_max = rows * pixel_size

    dataset = gdal.GetDriverByName(driver).Create(path, columns, rows, 1, gdal_type)
    dataset.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    dataset.SetProjection(spatial_reference_from_epsg(epsg).ExportToWkt())

    band = dataset.GetRasterBand(1)
    band.WriteArray(matrix)
    if noData is not None:
        band.SetNoDataValue(noData)
    if scale is not None:
        band.SetScale(scale)
    if offset is not None:
        band.SetOffset(offset)
    band.FlushCache()
    return dataset


def write_gdal_geotiff(path, matrix, gdal_type, **raster_kwargs):
    """Write ``matrix`` to a GeoTIFF with plain GDAL and return its path as a string."""
    dataset = create_gdal_raster(matrix, gdal_type, driver="GTiff", path=str(path), **raster_kwargs)
    dataset = None  # closing the dataset flushes it to disk
    return str(path)


def band_type_name(raster) -> str:
    """The GDAL name of the type of the first band, such as "Byte" or "Float32"."""
    dataset = geokit.raster.loadRaster(raster)
    gdal_type = dataset.GetRasterBand(1).DataType
    return gdal.GetDataTypeName(gdal_type)


def square_polygon(x_min: float, size: float = 1000.0):
    """A square in EPSG:3035 whose lower left corner is (x_min, 0)."""
    corners = [(x_min, 0), (x_min + size, 0), (x_min + size, size), (x_min, size), (x_min, 0)]
    return geokit.geom.polygon(corners, srs=3035)


def write_gdal_geopackage(path, field_name, field_type, field_values):
    """Write one 1000 m square per value, side by side, to a GeoPackage with plain OGR and return its path."""
    data_source = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    layer = data_source.CreateLayer("squares", spatial_reference_from_epsg(3035), ogr.wkbPolygon25D)
    layer.CreateField(ogr.FieldDefn(field_name, field_type))
    for index, field_value in enumerate(field_values):
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(square_polygon(2000 * index))
        if field_value is not None:  # None leaves the field NULL
            feature.SetField(field_name, field_value)
        layer.CreateFeature(feature)
    data_source = None  # closing the data source writes the file
    return str(path)
