geokit.core.util
================

.. py:module:: geokit.core.util

.. autoapi-nested-parse::

   The Util sub-module contains a number of generally helpful utility functions, classes, and constants



Attributes
----------

.. autoapisummary::

   geokit.core.util._test
   geokit.core.util.res


Exceptions
----------

.. autoapisummary::

   geokit.core.util.GeoKitError


Classes
-------

.. autoapisummary::

   geokit.core.util.Feature
   geokit.core.util.AxHands


Functions
---------

.. autoapisummary::

   geokit.core.util.isVector
   geokit.core.util.isRaster
   geokit.core.util.scaleMatrix
   geokit.core.util.KernelProcessor
   geokit.core.util.quickVector
   geokit.core.util.fitBoundsTo
   geokit.core.util.quickRaster
   geokit.core.util.drawImage
   geokit.core.util.compare_geoms


Module Contents
---------------

.. py:data:: _test

.. py:data:: res

.. py:exception:: GeoKitError

   Bases: :py:obj:`Exception`


   Common base class for all non-exit exceptions.

   Initialize self.  See help(type(self)) for accurate signature.


.. py:function:: isVector(source)

   Test if loadVector fails for the given input

   Parameters:
   -----------
   source : str
       The path to the vector file to load

   Returns:
   --------
   bool -> True if the given input is a vector



.. py:function:: isRaster(source)

   Test if loadRaster fails for the given input

   Parameters:
   -----------
   source : str
       The path to the raster file to load

   Returns:
   --------
   bool -> True if the given input is a raster



.. py:function:: scaleMatrix(mat, scale, strict=True)

   Scale a 2-dimensional matrix. For example, a 2x2 matrix, with a scale of 2,
   will become a 4x4 matrix. Or scaling a 24x24 matrix with a scale of -3 will
   produce an 8x8 matrix.

   * Scaling UP (positive) results in a dimensionally larger matrix where each
     value is repeated scale^2 times
   * scaling DOWN (negative) results in a dimensionally smaller matrix where each
     value is the average of the associated 'up-scaled' block

   Parameters:
   -----------
       mat : numpy.ndarray or [[numeric,],]
           The data to be scaled
             * Must be two dimensional

       scale : int or (int, int)
           The dimensional scaling factor for either both, or independently for
           the Y and X dimensions
             * If scaling down, the scaling factors must be a factor of the their
               associated dimension in the input matrix (unless 'strict' is set
               to False)

       strict : bool
           Whether or not to force a fail when scaling-down by a scaling factor
           which is not a dimensional factor
             * Really intended for internal use...
             * When scaling down by a non-dimensional factor, the matrix will be
               padded with zeros such that the new matrix has dimensional sizes
               which are divisible by the scaling factor. The points which are
               not at the right or bottom boundary are averaged, same as before.
               The points which lie on the edge however, are also averaged across
               all the values which lie in those pixels, but they are corrected
               so that the averaging does NOT take into account the padded zeros.

   Returns:
   --------

   numpy.ndarray


   Examples:
   ---------

   INPUT       Scaling Factor      Output
   -----       --------------      ------

   | 1 2 |             2           | 1 1 2 2 |
   | 3 4 |                         | 1 1 2 2 |
                                   | 3 3 4 4 |
                                   | 3 3 4 4 |


   | 1 1 1 1 |        -2           | 1.5  2.0 |
   | 2 2 3 3 |                     | 5.25 6.75|
   | 4 4 5 5 |
   | 6 7 8 9 |


   | 1 1 1 1 |        -3           | 2.55  3.0 |
   | 2 2 3 3 |   * strict=False    | 7.0    9  |
   | 4 4 5 5 |
   | 6 7 8 9 |       *padded*
                   -------------
                  | 1 1 1 1 0 0 |
                  | 2 2 3 3 0 0 |
                  | 4 4 5 5 0 0 |
                  | 6 7 8 9 0 0 |
                  | 0 0 0 0 0 0 |
                  | 0 0 0 0 0 0 |



.. py:function:: KernelProcessor(size, edgeValue=0, outputType=None, passIndex=False)

   A decorator which automates the production of kernel processors for use
   in mutateRaster (although it could really used for processing any matrix)

   Parameters:
   -----------
   size : int
       The number of pixels to expand around a center pixel
       * A 'size' of 0 would make a processing matrix with size 1x1. As in,
         just the value at each point. This would be silly to call...
       * A 'size' of 1 would make a processing matrix of size 3x3. As in, one
         pixel around the center pixel in all directions
       * Processed matrix size is equal to 2*size+1

   edgeValue : numeric; optional
       The value to apply to the edges of the matrix before applying the kernel
       * Will be factored into the kernelling when processing near the edges

   outputType : np.dtype; optional
       The datatype of the processed values
       * Only useful if the output type of the kerneling step is different from
         the matrix input type

   passIndex : bool
       Whether or not to pass the x and y index to the processing function
       * If True, the decorated function must accept an input called 'xi' and
         'yi' in addition to the matrix
       * The xi and yi correspond to the index of the center pixel in the
         original matrix

   Returns:
   --------
   function

   Example:
   --------
   * Say we want to make a processor which calculates the average of pixels
     which are within a distance of 2 indicies. In other words, we want the
     average of a 5x5 matrix centered around each pixel.
   * Assume that we can use the value -9999 as a no data value

   >>>  @KernelProcessor(2, edgeValue=-9999)
   >>>  def getMean( mat ):
   >>>      # Get only good values
   >>>      goodValues = mat[mat!=-9999]
   >>>
   >>>      # Return the mean
   >>>      return goodValues.mean()



.. py:function:: quickVector(geom, output=None)

   GeoKit internal for quickly creating a vector datasource


.. py:function:: fitBoundsTo(bounds, dx, dy, expand: bool = False, startAtZero: bool = False, enforce: bool = False) -> tuple

.. py:function:: quickRaster(bounds, srs, dx, dy, dtype='GDT_Byte', noData=None, fill=None, data=None, scale=None, offset=None)

   GeoKit internal for quickly creating a raster datasource


.. py:class:: Feature

   Bases: :py:obj:`tuple`


   .. py:attribute:: geom


   .. py:attribute:: attr


.. py:class:: AxHands

   Bases: :py:obj:`tuple`


   .. py:attribute:: ax


   .. py:attribute:: handles


   .. py:attribute:: cbar


.. py:function:: drawImage(matrix, ax=None, xlim=None, ylim=None, yAtTop=True, scaling=1, fontsize=16, hideAxis=False, figsize=(12, 12), cbar=True, cbarPadding=0.01, cbarTitle=None, vmin=None, vmax=None, cmap='viridis', cbax=None, cbargs=None, leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0, **kwargs)

   Draw a matrix as an image on a matplotlib canvas

   Parameters:
   -----------
   matrix : numpy.ndarray
       The matrix data to draw

   ax : matplotlib axis; optional
       The axis to draw the geometries on
         * If not given, a new axis is generated and returned

   xlim : (float, float); optional
       The x-axis limits to draw the marix on

   ylim : (float, float); optional
       The y-axis limits to draw the marix on

   yAtTop : bool; optional
       If True, the first row of data should be plotted at the top of the image

   scaling : numeric; optional
       An integer factor by which to scale the matrix before plotting

   figsize : (int, int); optional
       The figure size to create when generating a new axis
         * If resultign figure looks wierd, altering the figure size is your best
           bet to make it look nicer

   fontsize : int; optional
       A base font size to apply to tick marks which appear
         * Titles and labels are given a size of 'fontsize' + 2

   cbar : bool; optional
       If True, a color bar will be drawn

   cbarPadding : float; optional
       The spacing padding to add between the generated axis and the generated
       colorbar axis
         * Only useful when generating a new axis
         * Only useful when 'colorBy' is given

   cbarTitle : str; optional
       The title to give to the generated colorbar
         * If not given, but 'colorBy' is given, the same string for 'colorBy'
           is used
           * Only useful when 'colorBy' is given

   vmin : float; optional
       The minimum value to color

   vmax : float; optional
       The maximum value to color

   cmap : str or matplotlib ColorMap; optional
       The colormap to use when coloring

   cbax : matplotlib axis; optional
       An explicitly given axis to use for drawing the colorbar

   cbargs : dict; optional

   leftMargin : float; optional
       Additional margin to add to the left of the figure
         * Before using this, try adjusting the 'figsize'

   rightMargin : float; optional
       Additional margin to add to the left of the figure
         * Before using this, try adjusting the 'figsize'

   topMargin : float; optional
       Additional margin to add to the left of the figure
         * Before using this, try adjusting the 'figsize'

   bottomMargin : float; optional
       Additional margin to add to the left of the figure
         * Before using this, try adjusting the 'figsize'

   Returns:
   --------
   A namedtuple containing:
      'ax' -> The map axis
      'handles' -> All geometry handles which were created in the order they were
                   drawn
      'cbar' -> The colorbar handle if it was drawn



.. py:function:: compare_geoms(geoms_1, geoms_2)

   Compare two lists of geometries and return a list of booleans indicating whether each pair of geometries are equal.
   The order of the lists is important, as the first geometry in the first list will be compared to the first geometry in the second list, and so on.

   :returns: A list of booleans indicating whether each pair of geometries are equal.
   :rtype: list


