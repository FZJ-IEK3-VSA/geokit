combineSimilarRasters
=====================

.. py:module:: combineSimilarRasters


Functions
---------

.. autoapisummary::

   combineSimilarRasters.combineSimilarRasters


Module Contents
---------------

.. py:function:: combineSimilarRasters(datasets, output=None, combiningFunc=None, verbose=True, updateMeta=False, **kwargs)

   Combines several similar raster files into one single raster file.

   :param datasets: glob string path describing datasets to combine, alternatively list of gdal.Datasets or iterable object with paths.
   :type datasets: string or list
   :param output: Filepath to output raster file. If it is an existing file, datasets will be added to output. Recommended to create a new file everytime though. If None, no output dataset will be loaded or created on disk and output dataset kept in memory only, by default None
   :type output: string, optional
   :param combiningFunc: Allows customized functions to combine matrices, by default None
   :type combiningFunc: [type], optional
   :param verbose: If True, additional status print stamenets will be issued, by default True
   :type verbose: bool, optional
   :param updateMeta: If True, metadata of output dataset will be a combination of all input rasters, by default False
   :type updateMeta: bool, optional
   :param Returns:
   :param ----------:
   :param output dataset: Raster file containing the combined matrices of all input datasets.
   :type output dataset: osgeo.gdal.Dataset


