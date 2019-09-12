<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster" alt="Forschungszentrum Juelich Logo" width="230px"></a> 

# GeoKit - **Geo**spatial tool**kit** for Python

GeoKit communicates directly with functions and objects within the Geospatial Data Abstraction Library (<a href="www.gdal.org">GDAL</a>) and exposes them in such a way that is particularly useful for programmatic general purpose geospatial analyses. 
It gives low overhead control of fundamental operations; such as reading, writing, and mutating geospatial data sets, manipulating and translating geometries, warping and resampling raster data, and much more. 
Via the RegionMask object, GeoKit even allows for seamless integration of information expressed across multiple geospatial datasets in many formats and reference systems into the context of a single region. 

GeoKit is not intended to replace the GDAL library, as only very small subset of GDAL's capabilities are exposed. Nor is it intended to compete with other libraries with similar functionalities.
Instead GeoKit evolved in an ad hoc manner in order to realize the Geospatial Land Eligibility for Energy Systems (<a href="https://github.com/FZJ-IEK3-VSA/glaes">GLAES</a>) model which is intended for rapid land eligibility analyses of renewable energy systems and is also available on GitHub.
Nevertheless, GeoKit quickly emerged as a general purpose GIS toolkit with capabilities far beyond computing land eligibility.
Therefore, it is our pleasure to offer it to anyone who is interested in its use.

[![DOI](https://zenodo.org/badge/114900977.svg)](https://zenodo.org/badge/latestdoi/114900977)

## Features
* Direct exposure of functions and objects in the GDAL library
* Reading, writing, and manipulating raster and vector datasets
* Translation between data formats and projection systems
* Direct conversion of raster data into NumPy matrices

## Installation

First clone a local copy of the repository to your computer

	git clone https://github.com/FZJ-IEK3-VSA/geokit.git

Be sure GDAL and netCDF4 are previously installed
	* If using Anaconda, this can be accomplished via:

	conda install -c conda-forge gdal>=2.0.0 netCDF4

!For Windows users! 
	* Sometimes a path variable will need to be set to tell the system where to find the GDAL dependancies
	* Path variables name must be: "GDAL_DATA"
	* When installed with Anaconda, path should be: "<anaconda-top-directory>\Library\share\gdal"
	* As of GeoKit version 1.1.0, GeoKit will attempt to add this path automatically at runtime
	
Then install GeoKit via pip as follow
	
	cd geokit
	pip install -e .
	
Or install directly via python as 

	python setup.py install
	
	
## Examples

See the [Examples page](Examples/)

## Docker

We are trying to get GeoKit to work within a Docker container. Try it out!

* First pull the image with:
```bash
docker pull sevberg/geokit:latest
```

* You can then start a basic python interpreter with:
```bash
docker run -it sevberg/geokit:latest -c "python"
```

* Or you can start a jupyter notebook using:
```bash
docker run -it \
    -p 8888:8888 \
    sevberg/geokit:latest \
    -c "jupyter notebook --notebook-dir=/notebook-dir --ip='*' --port=8888 --no-browser --allow-root"
```
 - Which can then be connected to at the address "localhost:8888:<API-KEY>"
 - The API Key can be found from the output of the earlier command

* Finally, you might want to mount a volume to access geospatial data. For this you can use:
```bash
docker run -it \
    --mount target=/notebook-dir,type=bind,src=<PATH-TO-DIRECTORY> \
    -p 8888:8888 \
    sevberg/geokit:latest  \
    -c "jupyter notebook --notebook-dir=/notebook-dir --ip='*' --port=8888 --no-browser --allow-root"
```
## License

MIT License

Copyright (c) 2017 David Severin Ryberg (FZJ IEK-3), Jochen Linßen (FZJ IEK-3), Martin Robinius (FZJ IEK-3), Detlef Stolten (FZJ IEK-3)

You should have received a copy of the MIT License along with this program.  
If not, see <https://opensource.org/licenses/MIT>

## About Us 
<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="http://fz-juelich.de/SharedDocs/Bilder/IEK/IEK-3/Abteilungen2015/VSA_DepartmentPicture_2017.jpg?__blob=normal" alt="Abteilung VSA"></a> 

We are the [Process and Systems Analysis](http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the Forschungszentrum Jülich. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.


## Acknowledgment

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050 – A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
