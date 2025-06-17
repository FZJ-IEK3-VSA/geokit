| Name |  Version  | Test on pull request | Daily test of conda-forge package|
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-geokit-green.svg)](https://anaconda.org/conda-forge/geokit) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/geokit.svg)](https://anaconda.org/conda-forge/geokit) | [![.github/workflows/test_conda_forge_version.yml](https://github.com/FZJ-IEK3-VSA/geokit/actions/workflows/test_development_version.yml/badge.svg)](https://github.com/FZJ-IEK3-VSA/geokit/actions/workflows/test_development_version.yml) | [![.github/workflows/test_conda_forge_version.yml](https://github.com/FZJ-IEK3-VSA/geokit/actions/workflows/test_conda_forge_version.yml/badge.svg)](https://github.com/FZJ-IEK3-VSA/geokit/actions/workflows/test_conda_forge_version.yml) |

<a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/FJZ_IEK-3_logo.svg?raw=True" alt="Forschungszentrum Juelich Logo" width="300px"></a> 

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

- Direct exposure of functions and objects in the GDAL library
- Reading, writing, and manipulating raster and vector datasets
- Translation between data formats and projection systems
- Direct conversion of raster data into NumPy matrices

## Installation

### Installation via conda-forge
The easiest way to install GeoKit into a new environment is from `conda-forge` with:

```bash
conda create -n geokit -c conda-forge geokit
```

or into an existing environment with:
```bash
conda install -c conda-forge geokit
```

### Installation from a local folder

1. First clone a local copy of the repository to your computer, and move into the created directory

```
git clone https://github.com/FZJ-IEK3-VSA/geokit.git
cd geokit
```

1. (Alternative) If you want to use the 'dev' branch (or another branch) then use:

```
git checkout dev
```

2. When using [Anaconda](https://www.anaconda.com/) / [(Micro-)Mamba](https://mamba.readthedocs.io/en/latest/) (recommended), GeoKit should be installable to a new environment with:

```
conda env create --file requirements.yml
conda activate geokit
pip install . --no-deps
```

2. (Alternative) Or into an existing environment with:

```
conda env update --file requirements.yml -n <ENVIRONMENT-NAME>
conda activate geokit
pip install . --no-deps
```

2. (Alternative) If you want to install GeoKit in editable mode, and also with jupyter notebook and with testing functionalities use:

```
conda env create --file requirements-dev.yml
conda activate geokit
pip install . --no-deps -e
```

## Examples

See the [Examples page](Examples/)

## Docker

We are trying to get GeoKit to work within a Docker container. Try it out!

- First pull the image with:

```bash
docker pull sevberg/geokit:latest
```

- You can then start a basic python interpreter with:

```bash
docker run -it sevberg/geokit:latest -c "python"
```

- Or you can start a jupyter notebook using:

```bash
docker run -it \
    -p 8888:8888 \
    sevberg/geokit:latest \
    -c "jupyter notebook --ip='*' --port=8888 --no-browser --allow-root"
```

- Which can then be connected to at the address "localhost:8888:<API-KEY>"
- The API Key can be found from the output of the earlier command

* Finally, you might want to mount a volume to access geospatial data. For this you can use:

```bash
docker run -it \
    --mount target=/notebooks,type=bind,src=<PATH-TO-DIRECTORY> \
    -p 8888:8888 \
    sevberg/geokit:latest  \
    -c "jupyter notebook --notebook-dir=/notebooks --ip='*' --port=8888 --no-browser --allow-root"
```

## License

MIT License

Active Developers: Julian Schönau, Rachel Maier, Christoph Winkler, Shitab Ishmam, David Franzmann, Julian Belina, Noah Pflugradt, Heidi Heinrichs, Jochen Linßen, Detlef Stolten 

Alumni: David Severin Ryberg, Martin Robinius, Stanley Risch

You should have received a copy of the MIT License along with this program.  
If not, see <https://opensource.org/licenses/MIT>

## About Us

<a href="https://www.fz-juelich.de/en/iek/iek-3"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/iek3-square.png?raw=True" alt="Institute image IEK-3" width="280" align="right" style="margin:0px 10px"/></a>

We are the <a href="https://www.fz-juelich.de/en/iek/iek-3">Institute of Energy and Climate Research - Techno-economic Systems Analysis (IEK-3)</a> belonging to the <a href="https://www.fz-juelich.de/en">Forschungszentrum Jülich</a>. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Acknowledgment

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>
