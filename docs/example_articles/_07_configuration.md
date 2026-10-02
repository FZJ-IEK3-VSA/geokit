# Further Configuration and Insights

At some point, you may need to configure ETHOS.GeoKit more precisely.


## Configure SRS

Many operations require you to set a spatial reference system, and there are multiple ways to configure it, as demonstrated in [this example](../Examples/_07_configuration_options/_01_srs.ipynb).

## Data Types

GDAL stores raster bands and vector fields in fixed C types, and ETHOS.GeoKit chooses them for you. [This example](../Examples/_07_configuration_options/_02_c_datatypes.ipynb) shows the `dtype` parameter with its modes `"auto"`, `"preserve_input"` and `"smallest"`, explicit types, the helpers of `geokit.dtypes`, and how field types follow from column types. Why GeoKit chooses the types it does is explained in [Data Types](../explanation/data_types/index.md).