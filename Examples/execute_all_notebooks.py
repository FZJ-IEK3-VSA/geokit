import pathlib

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

current_directory = pathlib.Path(__file__).parent

list_of_notebook_names = [
    r"example_01_srs.ipynb",
    r"example_02_geometry.ipynb",
    r"example_03_vector.ipynb",
    r"example_04_raster.ipynb",
    r"example_05_Extent.ipynb",
    r"example_06_RegionMask.ipynb",
]

for current_notebook_name in list_of_notebook_names:
    print("Execute notebook: ", current_notebook_name)
    with open(pathlib.Path(current_directory, current_notebook_name)) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor()
        ep.preprocess(nb)
    print("Notebook execution is terminated")
print("Done")
