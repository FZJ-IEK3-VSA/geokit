# Further Configuration and Insights

At some point, you may need to configure ETHOS.GeoKit more precisely.


## Configure SRS

Many operations require you to set a spatial reference system, and there are multiple ways to configure it, as demonstrated in [this example](../Examples/_07_configuration_options/_01_srs.ipynb).

## Internal Data Type Handling

ETHOS.GeoKit uses GDAL, which uses C data types. ETHOS.GeoKit automates many processes related to this topic, as shown in [this example](../Examples/_07_configuration_options/_02_c_datatypes.ipynb).