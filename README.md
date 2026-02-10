<table style="border:0; border-collapse:collapse;">
	<tr>
		<td style="border:0;">
			<img src="./docs/visualizations/logos/geokit_logo.svg" alt="GeoKit logo" width="300px">
		</td>
		<td style="border:0;">
			<a href="https://www.fz-juelich.de/en/ice/ice-2"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/JSA-Header.svg?raw=True" alt="Logo für Forschungszentrum Juelich - Juelich System Analysis" width="300px"></a>
		</td>
	</tr>
</table>

| Name                                                                                                             | Version                                                                                                             | Tests on pull requests                                                                                                                                                                                                       | Docstring Style                                                                  | Documentation Coverage                                  |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-geokit-green.svg)](https://anaconda.org/conda-forge/geokit) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/geokit.svg)](https://anaconda.org/conda-forge/geokit) | [![.github/workflows/test_current_branch.yml](https://github.com/FZJ-IEK3-VSA/geokit/actions/workflows/test_current_branch.yml/badge.svg)](https://github.com/FZJ-IEK3-VSA/geokit/actions/workflows/test_current_branch.yml) | ![NumPy docstring style](https://img.shields.io/badge/%20style-numpy-459db9.svg) | ![Documentation Coverage](./docs/interrogate_badge.svg) |



---

## Documentation Overview

GeoKit is a Python toolkit designed to efficiently handle geospatial data and spatial operations.

GeoKit provides low-overhead control of fundamental geospatial operations including:
- Reading, writing, and mutating geospatial datasets
- Manipulating and translating geometries between coordinate systems
- Warping and resampling raster data
- Seamlessly integrating multiple geospatial datasets through the **RegionMask** object

The RegionMask object is particularly powerful, allowing seamless integration of information expressed across multiple geospatial datasets in various formats and reference systems into the context of a single region. An extensice Documentation about GeoKit can be found at https://geokit.readthedocs.io/.

GeoKit is part of the [ETHOS (**E**nergy **T**ransformation **P**at**H**way **O**ptimization **S**uite)](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services). It builds upon the software [GDAL (Geospatial Data Abstraction Library)](https://gdal.org/) and exposes its capabilities in a way that is particularly useful for programmatic, general-purpose geospatial analyses.

## Installation

If you just want to use GeoKit, install it from Conda Forge. To download and execute all the examples or develop the source code, install it from the source. Both installations require conda or mamba, which can be used interchangeably. We recommend the [Miniforge installer](https://conda-forge.org/download/).

### Installation via conda-forge (Recommended)

The easiest way to install GeoKit into a new environment is from `conda-forge`:

```bash
conda create -n geokit -c conda-forge geokit
```

Or into an existing environment with:

```bash
conda install -c conda-forge geokit
```

### Installation from Source

1. Clone the repository and navigate to it:

```bash
git clone https://github.com/FZJ-IEK3-VSA/geokit.git
cd geokit
```

2. (Optional) Switch to the development branch:

```bash
git checkout dev
```

3. Create a new environment:

```bash
conda env create --file requirements-dev
conda activate geokit
pip install . --no-deps
```

4. (Alternative) Update an existing environment:

```bash
conda env update --file requirements-dev -n <ENVIRONMENT-NAME>
conda activate geokit
pip install . --no-deps
```

## Getting Started

The best way to learn GeoKit is through hands-on examples. This documentation includes:

- **Example notebooks** in the `docs/Examples` folder demonstrating real-world use cases
- **Detailed guides** in the `docs/example_articles` folder explaining key concepts
- **API documentation** providing comprehensive reference information
- **Source code** in the `geokit` folder for advanced users

Start with the [Introduction to GeoKit](docs/example_articles/_00_introduction.md) to understand the fundamentals, or jump directly to the capability area that interests you most.

### About GeoKit

GeoKit evolved from the Geospatial Land Eligibility for Energy Systems (<a href="https://github.com/FZJ-IEK3-VSA/glaes">GLAES</a>) model, which is intended for rapid land eligibility analyses of renewable energy systems. However, GeoKit quickly emerged as a versatile, general-purpose GIS toolkit with capabilities far extending beyond computing land eligibility.

## License

MIT License

Active developers: Christoph Winkler, Shitab Ishmam, Julian Belina, Noah Pflugradt, Heidi Heinrichs, Jochen Linßen, Detlef Stolten

Alumni: David Severin Ryberg, Martin Robinius, Stanley Risch, Julian Schönau, Rachel Maier, David Franzmann

You should have received a copy of the MIT License along with this program.  
If not, see <https://opensource.org/licenses/MIT>

## About Us

<a href="https://www.fz-juelich.de/en/ice/ice-2"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/iek3-square.png?raw=True" alt="Institute image ICE-2" width="280" align="right" style="margin:0px 10px"/></a>

We are the <a href="https://www.fz-juelich.de/en/ice/ice-2">Institute of Climate and Energy Systems (ICE) - Jülich Systems Analysis</a> belonging to the <a href="https://www.fz-juelich.de/en">Forschungszentrum Jülich</a>. Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.


## Contributions and Support
Every contributions are welcome:
- If you have a question want to report a bug or have feature request, please open an [Issue](https://github.com/FZJ-IEK3-VSA/geokit/issues/new). We will then take care of the issue as soon as possible.
- If you want to contribute with additional features or code improvements, open a [Pull request](https://github.com/FZJ-IEK3-VSA/geokit/pulls).

## Code of Conduct
Please respect our [code of conduct](CODE_OF_CONDUCT.md).

## Acknowledgment

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050: A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/"><img src="https://www.helmholtz.de/fileadmin/user_upload/05_aktuelles/Marke_Design/logos/HG_LOGO_S_ENG_RGB.jpg" alt="Helmholtz Logo" width="200px" style="float:right"></a>



