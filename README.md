<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster" alt="Forschungszentrum Juelich Logo" width="230px"></a> 

# GeoKit - Geospatial tools for Python
GeoKit communicates directly with functions and objects within the GDAL Library and exposes them in such a way that is particularly useful for general purpose geospatial operations with as little overhead as possible. It gives fundamental control over fundamental operations, such as reading, writing, and manipulating geospatial data sets, manipulating and translating geometries, warping and resampling raster data, and much more. Via the RegionMask object, GeoKit allows for seamless integration of information expressed across multiple geospatial datasets in any format and reference system into the context of a single region. 

GeoKit is not intended to replace the GDAL library, as only very small subset of GDAL's capabilities are exposes. Nor is it intended to compete with over libraries with similar functionalities.
Instead it evolved in an ad hoc manner in order to realize the Geospatial Land Eligibility for Energy Systems (GLAES) model which is intended for rapid land eligibility analyses of renewable energy systems and is also available on GitHub.
Nevertheless, GeoKit quickly emerged as a general purpose GIS toolset with capabilities far beyond computing land eligibility.
Therefore, it is our pleasure to offer it to anyone who is interested in its use.

## Features
* Direct exposure of functions and objects in the GDAL library
* Reading, writing, and manipulating raster and vector datasets
* Translation between data formats and projection systems
* Direct conversion of raster data into NumPy matrices

## Installation

First clone a local copy of the repository to your computer

	git clone https://github.com/FZJ-IEK3-VSA/geokit.git
	
Then install GeoKit via pip as follow
	
	cd geokit
	pip install .
	
Or install directly via python as 

	python setup.py install
	
	
## Examples

More detailed examples of GeoKit's capabilites will be added soon. 

## License

!!!!!!!!!!!!!!!!!!! I NEED TO DO THIS TOO !!!!!!!!!!!!!!!!!!!!!!!!!


Copyright (C) 2016-2017 Leander Kotzur (FZJ IEK-3), Peter Markewitz (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

## About Us 
<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://fz-juelich.de/SharedDocs/Bilder/IEK/IEK-3/Abteilungen2015/VSA_DepartmentPicture_2017.jpg?__blob=normal" alt="Abteilung VSA"></a> 

We are the [Process and Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the Forschungszentrum Jülich. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.


## Acknowledgement

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050 – A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
