from setuptools import setup, find_packages

setup(
    name='geokit',
    version='1.1.2',
    author='Severin Ryberg',
    url='https://github.com/FZJ-IEK3-VSA/geokit',
    packages = find_packages(),
    include_package_data=True,
    install_requires = [
        "gdal>=2.0.0",
        "numpy",
        "descartes",
        "pandas",
        "scipy",
        "matplotlib",
    ]
)
