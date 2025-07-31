geokit.get_test_data
====================

.. py:module:: geokit.get_test_data


Attributes
----------

.. autoapisummary::

   geokit.get_test_data.all_file_name_dict
   geokit.get_test_data.list_of_all_shape_file_extensions
   geokit.get_test_data.root_dir


Functions
---------

.. autoapisummary::

   geokit.get_test_data.get_test_data
   geokit.get_test_data.get_test_shape_file
   geokit.get_test_data.get_all_shape_files
   geokit.get_test_data.create_hash_dict


Module Contents
---------------

.. py:data:: all_file_name_dict

.. py:function:: get_test_data(file_name: str, data_cache_folder: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath('data'), no_download: bool = True) -> str

.. py:data:: list_of_all_shape_file_extensions
   :value: ['.shp', '.dbf', '.shx', '.prj', '.sbn', '.sbx', '.ain', '.aih', '.ixs', '.mxs', '.atx',...


.. py:function:: get_test_shape_file(file_name_without_extension: str, extension: Literal['.shp', '.dbf', '.shx', '.prj', '.sbn', '.sbx', '.ain', '.aih', '.ixs', '.mxs', '.atx', '.shp.xml', '.cpg', '.qix'], data_cache_folder: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath('data'), no_download: bool = True) -> str

.. py:function:: get_all_shape_files(data_cache_folder: pathlib.Path = pathlib.Path(__file__).parent.parent.joinpath('data'), no_download: bool = True)

.. py:function:: create_hash_dict(list_of_file_paths: list[pathlib.Path], alg: str = 'sha256') -> dict[str, str]

.. py:data:: root_dir

