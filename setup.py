from setuptools import setup, find_packages

setup(
    name='geokit',
    version='1.0.2',
    author='Severin Ryberg',
    url='http://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html',
    packages = find_packages(),
    include_package_data=True,
    install_requires = [
        "gdal>=2.0.0",
        "numpy>=1.11.2",
        "descartes",
        "pandas",
        "scipy",
        "matplotlib",
    ]
)
