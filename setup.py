from setuptools import setup, find_packages

setup(
    name='geokit',
    version='1.3.0',
    author='GeoKit Developer Team',
    url='https://github.com/FZJ-IEK3-VSA/geokit',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "gdal>2.2.0,<=3.2.1",
        "numpy",
        "descartes",
        "pandas",
        "scipy",
        "matplotlib",
        "smopy",
    ]
)
