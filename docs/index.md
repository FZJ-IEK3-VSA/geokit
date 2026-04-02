<table style="border:0; border-collapse:collapse;">
	<tr>
		<td style="border:0;">
			<img src="./visualizations/logos/geokit_logo.svg" alt="ETHOS.GeoKit logo" width="300px">
		</td>
		<td style="border:0;">
			<a href="https://www.fz-juelich.de/en/ice/ice-2"><img src="https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/JSA-Header.svg?raw=True" alt="Logo für Forschungszentrum Juelich - Juelich System Analysis" width="300px"></a>
		</td>
	</tr>
</table>

| Name                                                                                                             | Version                                                                                                             | Tests                                                                                                                                                                                                    | Pytest Coverage                                                                                                                                                 | Docstring Style                                                                  | Documentation Coverage                                  |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-geokit-green.svg)](https://anaconda.org/conda-forge/geokit) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/geokit.svg)](https://anaconda.org/conda-forge/geokit) | [![Tests](https://github.com/FZJ-IEK3-VSA/geokit/actions/workflows/test_push.yml/badge.svg?branch=dev)](https://github.com/FZJ-IEK3-VSA/geokit/actions/workflows/test_push.yml) | [![codecov](https://codecov.io/gh/FZJ-IEK3-VSA/geokit/graph/badge.svg)](https://codecov.io/gh/FZJ-IEK3-VSA/geokit) | ![NumPy docstring style](https://img.shields.io/badge/%20style-numpy-459db9.svg) | ![Documentation Coverage](./interrogate_badge.svg) |



---

## Documentation Overview

ETHOS.GeoKit is a Python toolkit designed to efficiently handle geospatial data and spatial operations. It provides low-overhead control of fundamental geospatial operations including:
- Reading, writing, and mutating geospatial datasets
- Manipulating and translating geometries between coordinate systems
- Warping and resampling raster data
- Seamlessly integrating multiple geospatial datasets through the **RegionMask** object

The RegionMask object is particularly powerful, allowing seamless integration of information expressed across multiple geospatial datasets in various formats and reference systems into the context of a single region. Extensive documentation about ETHOS.GeoKit can be found at https://geokit.readthedocs.io/.

ETHOS.GeoKit is part of the [ETHOS (**E**nergy **T**ransformation **P**at**H**way **O**ptimization **S**uite)](https://www.fz-juelich.de/de/ice/ice-2/leistungen/model-services). It builds upon the software [GDAL (Geospatial Data Abstraction Library)](https://gdal.org/) and exposes its capabilities in a way that is particularly useful for programmatic, general-purpose geospatial analyses. Geokit is for example used in [GLAES](https://github.com/FZJ-IEK3-VSA/glaes) and [RESKit](https://github.com/FZJ-IEK3-VSA/reskit).

## Installation

If you just want to use ETHOS.GeoKit, install it from Conda Forge. To download and execute all the examples or develop the source code, install it from the source. Both installations require conda or mamba, which can be used interchangeably. We recommend the [Miniforge installer](https://conda-forge.org/download/).

### Installation via conda-forge (Recommended)

The easiest way to install ETHOS.GeoKit into a new environment is from `conda-forge`:

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

The best way to learn ETHOS.GeoKit is through hands-on examples. This documentation includes:

- **Example notebooks** in the `Examples` folder demonstrating real-world use cases
- **Detailed guides** in the `example_articles` folder explaining key concepts
- **API documentation** providing comprehensive reference information
- **Source code** in the `geokit` folder for advanced users

Start with the [Introduction to ETHOS.GeoKit](./example_articles/_00_introduction.md) to understand the fundamentals, or jump directly to the capability area that interests you most.

## Contributions and Support
All contributions are welcome:
- If you have a question, want to report a bug, or have a feature request, please open an [Issue](https://github.com/FZJ-IEK3-VSA/geokit/issues/new). We will then take care of the issue as soon as possible.
- If you want to contribute with additional features or code improvements, open a [Pull request](https://github.com/FZJ-IEK3-VSA/geokit/pulls).

## License

MIT License

Active developers: Christoph Winkler, Shitab Ishmam, Julian Belina, Noah Pflugradt, Heidi Heinrichs, Jochen Linßen, Detlef Stolten

Alumni: David Severin Ryberg, Martin Robinius, Stanley Risch, Julian Schönau, Rachel Maier, David Franzmann

You should have received a copy of the MIT License along with this program.  
If not, see <https://opensource.org/licenses/MIT>

## About Us 

We are the <a href="https://www.fz-juelich.de/en/ice/ice-2">Institute of Climate and Energy Systems – Jülich Systems Analysis (ICE-2)</a> at the <a href="https://www.fz-juelich.de/en"> Forschungszentrum Jülich</a>.
Our work focuses on independent, interdisciplinary research in energy, bioeconomy, infrastructure, and sustainability. We support a just, greenhouse gas–neutral transformation through open models and policy-relevant science.

## Code of Conduct
Please respect our [code of conduct](https://github.com/FZJ-IEK3-VSA/README_assets/blob/main/CODE_CONDUCT.md).


## Acknowledgment

This work was supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050: A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

<a href="https://www.helmholtz.de/en/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-White-RGB.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-Dark-Blue-RGB.svg">
    <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-Dark-Blue-RGB.svg" alt="Helmholtz Logo" width="200px" style="float:left">
  </picture>
</a>


