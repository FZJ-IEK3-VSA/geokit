#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geokit as gk
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# # Genreal geokit build up
# 
# type gk. and use auto completion (press tab) to see suggestions. Here you can see the modules:
# - srs: spatial reference system
# - geom: geometries
# - raster: raster files
# - vector: vector files
# - region mask: *ontop of all*
# - Extent: *brings *all together*
# - ...
# 
# 
# Each module holdes all necessary function for its operation.
# Some more special function are directly available under gk.#function name, but are defined within each module.
# The modules are stored under ./geokit/core/#modulename

# For each function, the ducumentation can be assessed by '?'

# In[2]:


gk.drawGeoms


# # Spatial Reference System (SRS)

# same as:
# - CRS - Coordinate reference system
# - 'projection'
# - proj4 system
# 
# There are different types coordinate systems, for example EPSG4326. This can be accessed by typing 'gk.srs.EPSG4326'. This will create a SRS-Objectk. This object can be later used within other functions, where a SRS is used as an input parameter e.g: gk.raster.extractValues(srs = gk.srs.EPSG4326)
# 
# Commonley used SRS:
# - Basic latitude and longitude - EPSG4326
# - Lambert azimuthal equal area (LAEA) - EPSG3035
# - Mercator - EPSG3857

# In[3]:


print(type(gk.srs.EPSG4326))


# In[4]:


print(gk.srs.EPSG4326)
# print(gk.srs.EPSG3035)
# print(gk.srs.EPSG3857)


# ## Loading of other SRS
# Other SRS can be loaded from epsg.io or spacialreference.org. For Exampe EPSG:2004 would be loaded by the EPSG-name:

# In[5]:


print(gk.srs.loadSRS(2004))


# Otherwise, SRS can be loaded from their Proj4 string:

# In[6]:


poland_special_srs = gk.srs.loadSRS(
    'PROJCS["PUWG_42_Strefa_4",GEOGCS["GCS_Pulkovo_1942",DATUM["Pulkovo_1942",SPHEROID["Krasovsky_1940",6378245,298.3]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",4500000],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",21],PARAMETER["Scale_Factor",1],PARAMETER["Latitude_Of_Origin",0],UNIT["Meter",1]]'
)

print(poland_special_srs)


# ## Center the SRS to coordinate 
# SRS can be centerd to a coordinate, for example centered to Aachen:

# In[7]:


aachen_centered_srs = gk.srs.centeredLAEA(6.083, 50.775)
print(aachen_centered_srs)


# ## Transformation between coordinate systems

# In[8]:


# transforming points between SRS's
new_points = gk.srs.xyTransform(
    [
        (6.083, 50.775),
        (6.083, 50.875),
        (6.083, 50.975),
        (7.083, 50.175),
        (7.583, 50.775),
    ],
    fromSRS=gk.srs.EPSG4326,
    toSRS=aachen_centered_srs,
    outputFormat="xy",
)


# In[9]:


for x, y in zip(new_points.x, new_points.y):
    print(x, y)


# ## Export the SRS
# The coordinate systems can be exported. File formats for that are: WKT or Proj4

# In[10]:


# Export as WKT
aachen_centered_srs.ExportToWkt()


# In[11]:


# Export to Proj4
aachen_centered_srs.ExportToProj4()


# ## Load SRS

# In[12]:


gk.srs.loadSRS(aachen_centered_srs.ExportToWkt())


# In[ ]:




