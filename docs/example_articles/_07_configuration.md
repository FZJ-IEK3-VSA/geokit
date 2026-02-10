# Further Configuration and Insights

At some point, you may need to configure Geokit more precisely.


## Configure SRS

Many operations require you to set a spatial reference system, and there are multiple ways to configure it, as demonstrated in [this example](../Examples/_07_configuration_options/_01_srs.ipynb).

## Internal Data Type Handling

GeoKit uses GDAL, which uses C data types. GeoKit automates many processes related to this topic, as shown in [this example](../Examples/_07_configuration_options/_02_c_datatypes.ipynb).