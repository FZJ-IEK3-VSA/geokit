from .util import *
from .srsutil import *
from .geomutil import *
from .rasterutil import *
from .vectorutil import *


IndexSet = namedtuple("IndexSet","xStart yStart xWin yWin xEnd yEnd")

class Extent(object):
    """Geographic extent

    The Extent object represents geographic extents of an area and exposes useful methods which depend on those extents. This includes:
        - Easily representing the boundaries as (xMin, xMax, yMin, yMax) or (xMin, yMin, xMax, yMax)
        - Casting to another projection system
        - Padding and shifting the boundaries
        - "Fitting" the boundaries onto a given resolution
        - Clipping a given raster file

    Initialization is accomplished via:
        - Extent(xMin, yMin, xMax, yMax [, srs])
        - Extent.from_xyXY( (xMin, yMin, xMax, yMax) [, srs])
        - Extent.from_xXyY( (xMin, xMax, yMin, yMax) [, srs])
        - Extent.fromGeom( geom [, srs] )
        - Extent.fromVector( vector-file-path )
        - Extent.fromRaster( raster-file-path )
    """
    def __init__(s, *args, srs='latlon'):
        # Unpack args
        if len(args)==1:
            xMin, yMin, xMax, yMax = args[0]
        elif len(args)==4:
            xMin, yMin, xMax, yMax = args
        else:
            raise GeoKitExtentError("Incorrect number of positional arguments givin in init (accepts 1 or 4). Is an srs given as 'srs=...'?")
        
        # Ensure good inputs
        xMin, xMax = min(xMin, xMax), max(xMin, xMax)
        yMin, yMax = min(yMin, yMax), max(yMin, yMax) 

        s.xMin = xMin
        s.xMax = xMax
        s.yMin = yMin
        s.yMax = yMax
        s.srs  = loadSRS(srs)

        s._box = makeBox(s.xMin, s.yMin, s.xMax, s.yMax, srs=s.srs)

    @staticmethod
    def from_xXyY(bounds, srs='latlon'):
        """Create extent from explicitly defined boundaries

        Inputs:
          bounds - (xMin, xMax, yMin, yMax)
          srs - The Spatial Reference system to use (default EPSG4326)
        """
        return Extent(bounds[0], bounds[2], bounds[1], bounds[3], srs)

    @staticmethod
    def fromGeom( geom, srs=None ):
        """Create extent around a given geometry

        Inputs:
          geom - ogr.Geometry, str
            * The geometry to use when generating the extent
            * If a string is given, it is assumed to be a WKT string. In this case, an srs must also be provided
          srs - osr.SpatialReference, int, str (default None)
            * The srs of the given geometry
            * If an ogr.Geometry is given for the "geom" input, then that geometry will be transformed into the given srs. Unless no srs is given in which the geometry's native srs will be used
        """
        # ensure we have an osr.SpatialReference object
        if not srs is None:
            srs = loadSRS(srs)

        # Ensure geom is an ogr.Geometry object
        if isinstance(geom, str):
            if srs is None: raise ValueError("srs must be provided when geom is a string")
            geom = convertWKT(geom, srs)
        elif (srs is None):
            srs = geom.GetSpatialReference()

        # Test if a reprojection is required
        if not srs.IsSame( geom.GetSpatialReference()):
            geom.TransformTo(srs)

        # Read Envelope
        xMin, xMax, yMin, yMax = geom.GetEnvelope()

        # Done!
        return Extent(xMin, yMin, xMax, yMax, srs=geom.GetSpatialReference())

    @staticmethod
    def fromVector( source ):
        """Create extent around a vector source

        Inputs:
          source - str
            * The path to a vector file
        """
        shapeDS = loadVector(source)

        shapeLayer = shapeDS.GetLayer()
        shapeSRS = shapeLayer.GetSpatialRef()
        
        xMin,xMax,yMin,yMax = shapeLayer.GetExtent()

        return Extent(xMin, yMin, xMax, yMax, srs=shapeSRS)

    @staticmethod
    def fromRaster( source ):
        """Create extent around a raster source

        Inputs:
          source - str
            * The path to a vector file
        """
        dsInfo = rasterInfo(source)

        xMin,yMin,xMax,yMax = dsInfo.bounds

        return Extent(xMin, yMin, xMax, yMax, srs=dsInfo.srs)

    @staticmethod
    def _fromInfo(info):
        return Extent( info.xMin, info.yMin, info.xMax, info.yMax, srs=info.srs)

    @property
    def xyXY(s): 
        """Returns a tuple of the extent boundaries in order:
            xMin, yMin, xMax, yMax"""
        return (s.xMin, s.yMin, s.xMax, s.yMax)

    @property
    def xXyY(s): 
        """Returns a tuple of the extent boundaries in order:
            xMin, xMax, yMin, yMax"""
        return (s.xMin, s.xMax, s.yMin, s.yMax)

    @property
    def box(s): 
        """Returns a new rectangular ogr.Geometry object representing the extent"""
        return s._box.Clone()

    def __eq__(s,o):
        #if (s.xyXY != o.xyXY): return False
        if (not s.srs.IsSame(o.srs) ): return False
        if not isclose(s.xMin, o.xMin): return False
        if not isclose(s.xMax, o.xMax): return False
        if not isclose(s.yMin, o.yMin): return False
        if not isclose(s.yMin, o.yMin): return False
        return True

    def __str__(s):
        out = ""
        out += "xMin: %f\n"%s.xMin
        out += "xMax: %f\n"%s.xMax
        out += "yMin: %f\n"%s.yMin
        out += "yMax: %f\n"%s.yMax
        out += "srs: %s\n"%s.srs.ExportToWkt()

        return out

    def pad(s, pad): 
        """Pad the edges of the extent by some amount

        Inputs:
          pad - float
            * The amount to pad the edges within the extent's reference system
            * Can also accept a negative padding
        """
        if pad is None or pad == 0 : return s
        return Extent(s.xMin-pad, s.yMin-pad, s.xMax+pad, s.yMax+pad, srs=s.srs)

    def shift(s, dx=0, dy=0): 
        """Shift the edges of the extent by some amount in either direction

        Inputs:
          dx - float (default 0)
            * The amount to shift the x-edges within the extent's reference system
          dy - float (default 0)
            * The amount to shift the y-edges within the extent's reference system
        
        """
        return Extent(s.xMin+dx, s.yMin+dy, s.xMax+dx, s.yMax+dy, srs=s.srs)

    def fitsResolution(s, unit, tolerance=1e-6):
        try:
            unitX, unitY = unit
        except:
            unitX, unitY = unit, unit

        xSteps = (s.xMax-s.xMin)/unitX
        xResidual = abs(xSteps-np.round(xSteps))
        if xResidual > tolerance:
            return False

        ySteps = (s.yMax-s.yMin)/unitY
        yResidual = abs(ySteps-np.round(ySteps))
        if yResidual > tolerance:
            return False

        return True

    def fit(s, unit, dtype=None):
        """Fit the extent to a given pixel resolution

        * The extent is always expanded to fit onto the given unit

        Inputs:
          unit - float, (float, float)
            * The unit (or (xUnit, yUnit) ) dimension to fit the extent onto
          dtype - type (default None)
            * An optional caster which will force the output dimensions to be the given data type
        """
        try:
            unitX, unitY = unit
        except:
            unitX, unitY = unit, unit
        
        # Look for bad sizes
        if (unitX> s.xMax-s.xMin): raise GeoKitExtentError("Unit size is larger than extent width")
        if (unitY> s.yMax-s.yMin): raise GeoKitExtentError("Unit size is larger than extent width")

        # Calculate new extent
        newXMin = s.xMin-s.xMin%unitX
        newYMin = s.yMin-s.yMin%unitY

        tmp = s.xMax%unitX
        if(tmp == 0): newXMax = s.xMax
        else: newXMax = (s.xMax+unitX)-tmp

        tmp = s.yMax%unitY
        if(tmp == 0): newYMax = s.yMax
        else: newYMax = (s.yMax+unitY)-tmp

        # Done!
        if dtype is None or isinstance(unitX,dtype):
            return Extent( newXMin, newYMin, newXMax, newYMax, srs=s.srs )
        else:
            return Extent( dtype(newXMin), dtype(newYMin), dtype(newXMax), dtype(newYMax), srs=s.srs )

    def corners(s, asPoints=False):
        """Shortcut function to get the four corners of the extent as ogr geometry points

        * The extent is always expanded to fit onto the given unit

        Inputs:
          unit - float, (float, float)
            * The unit (or (xUnit, yUnit) ) dimension to fit the extent onto
          dtype - type (default None)
            * An optional caster which will force the output dimensions to be the given data type
        """

        if (asPoints):
            # Make corner points
            bl = makePoint( s.xMin, s.yMin )
            br = makePoint( s.xMax, s.yMin )
            tl = makePoint( s.xMin, s.yMax )
            tr = makePoint( s.xMax, s.yMax )

        else:
            # Make corner points
            bl = ( s.xMin, s.yMin)
            br = ( s.xMax, s.yMin)
            tl = ( s.xMin, s.yMax)
            tr = ( s.xMax, s.yMax)

        return (bl, br, tl, tr)

    def castTo(s, targetSRS):
        """
        Transforms an extent from a source SRS to a target SRS. 
        Note: The resulting region will be equal to or (almost certainly) larger than the origional
        
        keyword inputs:
            targetSRS - (required)
                : int -- The target SRS to use as an EPSG integer
                : str -- The target SRS to use as a WKT string
                : osr.SpatialReference -- The target SRS to use
        """
        targetSRS=loadSRS(targetSRS)

        # Check for same srs
        if( targetSRS.IsSame(s.srs)):
            return s

        # Create a transformer
        targetSRS = loadSRS(targetSRS)
        transformer = osr.CoordinateTransformation(s.srs, targetSRS)

        # Transform and record points
        X = []
        Y = []

        for point in s.corners(True):
            try:
                point.Transform( transformer )
            except Exception as e:
                print("Could not transform between the following SRS:\n\nSOURCE:\n%s\n\nTARGET:\n%s\n\n"%(s.srs.ExportToWkt(), targetSRS.ExportToWkt()))
                raise e

            
            X.append(point.GetX())
            Y.append(point.GetY())

        # return new extent
        return Extent(min(X), min(Y), max(X), max(Y), srs=targetSRS)

    def inSourceExtent(s, source):
        """Tests if the extent box intersects the extent box of of the given source"""
        sourceExtent = Extent.fromVector(source).castTo(s.srs)
        return s._box.Intersects(sourceExtent.box)

    def filterSources(s, sources):
        """Filter a list of sources whose's envelope overlaps a given extent.

        Input 'sources' can either be:
            * A list of vector sources
            * A glob string which will generate a list of source paths
                - see glob.glob for more info
        """
        # create list of searchable files
        if isinstance(sources, str):
            directoryList = glob(sources)
        else:
            directoryList = sources

        return filter(s.inSourceExtent, directoryList)
    
    def contains(s, extent, res=None):
        """Tests if the extent contains another given extent

        * If an optional resolution ('res') is given, the containment value is also dependant on whether or not the given extent fits within the larger extent AND is situated along the given resolution

        Inputs:
            extent - Extent object
                * The other extent
            res - float, (float, float) (default None)
                * The resolution on which the interior extent should fit
        """
        # test raw bounds
        if( not extent.srs.IsSame(s.srs) or 
            extent.xMin < s.xMin or extent.yMin < s.yMin or
            extent.xMax > s.xMax or extent.yMax > s.yMax):
            return False

        if( res ):
            # unpack resolution
            try: dx, dy = res
            except: dx, dy = res, res

            # Test for factor of resolutions
            thresh = dx/1000
            if( (extent.xMin - s.xMin)%dx>thresh or
                (extent.yMin - s.yMin)%dy>thresh or
                (s.xMax - extent.xMax)%dx>thresh or
                (s.yMax - extent.yMax)%dy>thresh ):
                return False
        return True

    def findWithin(s, extent, res=100, yAtTop=True):
        """Finds the given extent within the main extent according to the given resolution. Assumes the two extents are a part of the same 'grid'
    
        Inputs:
            extent - Extent
                * The other extent

            res - float, (float, float) (default 100)
                * The resolution of the 'grid'

            yAtTop - bool (default True)
                * Instructs the offsetting to begin from yMax instead of from yMin

        Returns:
            (xOffset, yOffset, xWindowSize, yWindowSize)
        """

        # test srs
        if not s.srs.IsSame(extent.srs):
            raise GeoKitExtentError("extents are not of the same srs")

        # try to unpack the resolution
        try:
            dx,dy = res
        except:
            dx,dy = res,res

        # Get offsets
        tmpX = (extent.xMin - s.xMin)/dx
        xOff = int(np.round(tmpX))

        if( yAtTop ):
            tmpY = (s.yMax - extent.yMax)/dy
        else:
            tmpY = (extent.yMin - s.yMin)/dy
        yOff = int(np.round(tmpY))

        if not (isclose(xOff, tmpX) and isclose(yOff, tmpY)):
            raise GeoKitExtentError("The extents are not relatable on the given resolution")

        # Get window sizes
        tmpX = (extent.xMax - extent.xMin)/dx
        xWin = int(np.round(tmpX))

        tmpY = (extent.yMax - extent.yMin)/dy
        yWin = int(np.round(tmpY))

        if not (isclose(xWin, tmpX) and isclose(yWin, tmpY)):
            raise GeoKitExtentError("The extents are not relatable on the given resolution")

        # Done!
        return IndexSet(xOff, yOff, xWin, yWin, xOff+xWin, yOff+yWin)
    
    def extractMatrix(s, source):
        """Extracts the extent from the given raster source as a matrix. The called extent must fit somewhere within the raster's grid

        Returns: extracted-data-matrix:
                * The data matrix from the raster which corresponds to the called extent

        !NOTE! - If a raster is given which is not in the 'flipped-y' orientation, it will be clipped in its native state, but the returned matrix will be automatically flipped
        """
        
        # open the dataset and get description
        rasDS = loadRaster(source)
        rasInfo = rasterInfo(rasDS)
        rasExtent = Extent._fromInfo(rasInfo)

        # Find the main extent within the raster extent
        try:
            xO, yO, xW, yW, xE, yE = rasExtent.findWithin(s, res=(rasInfo.dx, rasInfo.dy), yAtTop=rasInfo.yAtTop)
        except GeoKitExtentError:
            raise GeoKitExtentError( "The extent does not appear to fit within the given raster")
            
        # Extract and return the matrix
        arr = fetchMatrix(rasDS, xOff=xO, yOff=yO, xWin=xW, yWin=yW)

        # make sure we are returing data in the 'flipped-y' orientation
        if not rasInfo.flipY:
            arr = arr[::-1,:]
        
        return arr

    def clipRaster(s, source, output=None):
        """
        Clip a given raster around the extent object while maintaining the original source's
        projection and resolution

        * Returns a gdal.Datasource if an 'output' path is not provided and asMatrix is False
        * Creates a raster file if and 'output' path is provided. Will return nothing if asMatrix is False
        * Always returns a matrix and the associated extent if asMatrix is True
            - order is: (Matrix, Extent)

        Inputs:
            source - gdal.Datasource, str 
                * The datasource to clip
                * If a string is given, it must be a path to a datasource which gdal can open

            output - str (None)
                * A path to an output file

            asMatrix - bool (default False)
                * Instructs the method to return the clipped raster as a matrix
        """

        # open the dataset and get description
        rasDS = loadRaster(source)
        ras = rasterInfo(rasDS)

        # Find an extent which contains the region in the given raster system
        if( not ras.srs.IsSame(s.srs) ):
            extent = s.castTo(ras.srs)
        else:
            extent = s

        # Ensure new extent fits the pixel size
        extent = extent.fit((ras.dx, ras.dy))

        # create target raster
        outputDS = createRaster(bounds=extent.xyXY, pixelWidth=ras.dx, pixelHeight=ras.dy, 
                                srs=ras.srs, dtype=ras.dtype, noDataValue=ras.noData, overwrite=True,
                                output=output)
        # Do warp
        if (not output is None): # Reopen the outputDS if it is None (only happens when an output path was given)
            outputDS = gdal.Open(output, gdal.GA_Update)

        gdal.Warp(outputDS, source, outputBounds=extent.xyXY) # Warp source onto the new raster

        # Done
        if(output is None):
            return outputDS
        else:
            return

