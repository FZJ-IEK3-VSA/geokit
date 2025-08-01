# GeoKit - **Geo**spatial tool**kit** for Python

GeoKit communicates directly with functions and objects within the [GDAL (Geospatial Data Abstraction Library)](https://gdal.org/) and exposes them in such a way that is particularly useful for programmatic general purpose geospatial analyses. It is part of the [ETHOS (**E**nergy **T**ransformation **P**at**H**way **O**ptimization **S**uite)](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services). 
It gives low overhead control of fundamental operations; such as reading, writing, and mutating geospatial data sets, manipulating and translating geometries, warping and resampling raster data, and much more.
Via the RegionMask object, GeoKit even allows for seamless integration of information expressed across multiple geospatial datasets in many formats and reference systems into the context of a single region.


GeoKit is not intended to replace the GDAL library, as only very small subset of GDAL's capabilities are exposed. Nor is it intended to compete with other libraries with similar functionalities.
Instead GeoKit evolved in an ad hoc manner in order to realize the Geospatial Land Eligibility for Energy Systems (<a href="https://github.com/FZJ-IEK3-VSA/glaes">GLAES</a>) model which is intended for rapid land eligibility analyses of renewable energy systems and is also available on GitHub.
Nevertheless, GeoKit quickly emerged as a general purpose GIS toolkit with capabilities far beyond computing land eligibility.
Therefore, it is our pleasure to offer it to anyone who is interested in its use.