from .util import *
from .srs import *
from .geom import *
from .raster import *
from .vector import *

IndexSet = namedtuple("IndexSet","xStart yStart xWin yWin xEnd yEnd")

class Extent(object):
    """Geographic extent

    The Extent object represents geographic extents of an area and exposes useful
    methods which depend on those extents. This includes:
        - Easily representing the boundaries as (xMin, xMax, yMin, yMax) or 
          (xMin, yMin, xMax, yMax)
        - Casting to another projection system
        - Padding and shifting the boundaries
        - "Fitting" the boundaries onto a given resolution
        - Clipping a given raster file

    Initialization:
    ---------------
    * Extent(xMin, yMin, xMax, yMax [, srs])
    * Extent.from_xyXY( (xMin, yMin, xMax, yMax) [, srs])
    * Extent.from_xXyY( (xMin, xMax, yMin, yMax) [, srs])
    * Extent.fromGeom( geom [, srs] )
    * Extent.fromVector( vector-file-path )
    * Extent.fromRaster( raster-file-path )
    * Extent.load( args )
    """
    def __init__(s, *args, srs='latlon'):
        """Create extent from explicitly defined boundaries
        
        Usage:
        ------
        Extent(xMin, yMin, xMax, yMax [, srs=<srs>])
        Extent( (xMin, yMin, xMax, yMax) [, srs=<srs>])
        
        Where:    
            xMin - The minimal x value in the respective SRS
            yMin - The minimal y value in the respective SRS
            xMax - The maximal x value in the respective SRS
            yMax - The maximal y value in the respective SRS
            srs - The Spatial Reference system to use

        """
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

        s._box = box(s.xMin, s.yMin, s.xMax, s.yMax, srs=s.srs)

    @staticmethod
    def from_xXyY(bounds, srs='latlon'):
        """Create an Extent from explicitly defined boundaries

        Parameters:
        -----------
        bounds : tuple 
            The (xMin, xMax, yMin, yMax) values for the extent

        srs : Anything acceptable to geokit.srs.loadSRS(); optional
            The srs of the input coordinates
              * If not given, lat/lon coordinates are assumed
        
        Returns:
        --------
        Extent

        """
        return Extent(bounds[0], bounds[2], bounds[1], bounds[3], srs=srs)

    @staticmethod
    def fromGeom( geom ):
        """Create extent around a given geometry
        
        Parameters:
        -----------
        geom : ogr.Geometry 
            The geometry from which to extract the extent

        Returns:
        --------
        Extent

        """
        # Read Envelope
        xMin, xMax, yMin, yMax = geom.GetEnvelope()

        # Done!
        return Extent(xMin, yMin, xMax, yMax, srs=geom.GetSpatialReference())

    @staticmethod
    def fromVector( source ):
        """Create extent around the contemts of a vector source
        
        Parameters:
        -----------
        source : Anything acceptable by loadVector()
            The vector datasource to read from

        Returns:
        --------
        Extent

        """
        shapeDS = loadVector(source)

        shapeLayer = shapeDS.GetLayer()
        shapeSRS = shapeLayer.GetSpatialRef()
        
        xMin,xMax,yMin,yMax = shapeLayer.GetExtent()

        return Extent(xMin, yMin, xMax, yMax, srs=shapeSRS)

    @staticmethod
    def fromRaster( source ):
        """Create extent around the contents of a raster source
        
        Parameters:
        -----------
        source : Anything acceptable by loadRaster()
            The vector datasource to read from

        Returns:
        --------
        Extent

        """
        dsInfo = rasterInfo(source)

        xMin,yMin,xMax,yMax = dsInfo.bounds

        return Extent(xMin, yMin, xMax, yMax, srs=dsInfo.srs)

    @staticmethod
    def fromLocationSet( locs ):
        """Create extent around the contents of a LocationSet object
        
        Parameters:
        -----------
        locs : LocationSet
            
        Returns:
        --------
        Extent

        """
        lonMin, latMin, lonMax, latMax = locs.getBounds()
        return Extent(lonMin, latMin, lonMax, latMax, srs=EPSG4326)


    @staticmethod
    def load( source, **kwargs ):
        """Attempts to load an Extent from a variety of inputs in the most 
        appropriate manner

        One Extent initializer (.fromXXX) is called depending on the inputs
            
            source is LocationSet -> Extent.fromLocationSet( source )
            source is ogr.Geometry -> Extent.fromGeom( source )
            source is not a string: -> Extent(*source, **kwargs)
            source is a string:
                First try: Extent.fromVector(source)
                Then try: Extent.fromRaster(source)

        If none of the above works, an error is raised

        Returns:
        --------
        Extent

        """
        if isinstance(source, LocationSet): return Extent.fromLocationSet(source)
        elif isinstance(source, ogr.Geometry): return Extent.fromGeom(source)
        elif isVector(source): return Extent.fromVector(source)
        elif isRaster(source): return Extent.fromRaster(source)
        
        try: # Maybe the source is an iterable giving xyXY 
            vals = list(source)
            if len(vals)==4: return Extent( vals, **kwargs)
        except: pass

        raise GeoKitExtentError("Could not load the source")

    @staticmethod
    def _fromInfo(info):
        """GeoKit internal

        Creates an Extent from rasterInfo's returned value
        """
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
    def ylim(s): 
        """Returns a tuple of the y-axis extent boundaries in order:
            yMin, yMax
        """
        return (s.yMin, s.yMax)

    @property
    def xlim(s): 
        """Returns a tuple of the x-axis extent boundaries in order:
            xMin, xMax
        """
        return (s.xMin, s.xMax)

    @property
    def box(s): 
        """Returns a rectangular ogr.Geometry object representing the extent"""
        return s._box.Clone()

    def __eq__(s,o):
        #if (s.xyXY != o.xyXY): return False
        if (not s.srs.IsSame(o.srs) ): return False
        if not np.isclose(s.xMin, o.xMin): return False
        if not np.isclose(s.xMax, o.xMax): return False
        if not np.isclose(s.yMin, o.yMin): return False
        if not np.isclose(s.yMin, o.yMin): return False
        return True

    def __repr__(s):
        out = ""
        out += "xMin: %f\n"%s.xMin
        out += "xMax: %f\n"%s.xMax
        out += "yMin: %f\n"%s.yMin
        out += "yMax: %f\n"%s.yMax
        out += "srs: %s\n"%s.srs.ExportToWkt()

        return out

    def __str__(s):
        return "(%.5f,%.5f,%.5f,%.5f)"%s.xyXY


    def pad(s, pad): 
        """Pad the extent in all directions
        
        Parameters:
        -----------
        pad : float
            The amount to pad in all directions
            * In units of the extent's srs
            * Can also accept a negative padding
        
        Returns:
        --------
        Extent

        """
        # Check for no input pads
        if pad is None: return s
        
        # try breaking apart by x and y component
        try: 
            xpad, ypad = pad
        except: 
            xpad = pad
            ypad = pad

        # Pad the extent
        return Extent(s.xMin-xpad, s.yMin-ypad, s.xMax+xpad, s.yMax+ypad, srs=s.srs)

    def shift(s, dx=0, dy=0): 
        """Shift the extent in the X and/or Y dimensions
        
        Parameters:
        -----------
        dx : float
            The amount to shift in the x dimension
            * In units of the extent's srs

        dy : float
            The amount to shift in the y dimension
            * In units of the extent's srs
        
        Returns:
        --------
        Extent

        """
        return Extent(s.xMin+dx, s.yMin+dy, s.xMax+dx, s.yMax+dy, srs=s.srs)

    def fitsResolution(s, unit, tolerance=1e-6):
        """Test if calling Extent first around the given unit(s) (at least within
        an error defined by 'tolerance')
        
        Parameters:
        -----------
        unit : numeric or tuple
            The unit value(s) to check
            * If float, a single resolution value is assumed for both X and Y dim
            * If tuple, resolutions for both dimensions (x, y)

        tolerance : float
            The tolerance to allow when comparing float values

        Returns:
        --------
        Extent

        Examples:
        ---------
        >>> ex = Extent( 100, 100, 300, 500)
        >>> ex.fitsResolution(25) # True!
        >>> ex.fitsResolution( (25, 10) ) # True!
        >>> ex.fitsResolution(33) # False!
        >>> ex.fitsResolution( (25, 33) ) # False!
        
        """
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
        
        Note:
        -----
        The extent is always expanded to fit onto the given unit


        Parameters:
        -----------
        unit : numeric or tuple
            The unit value(s) to check
            * If numeric, a single value is assumed for both X and Y dim
            * If tuple, resolutions for both dimensions (x, y)

        dtype : Type or np.dtype
            The final data type of the boundary values

        Returns:
        --------
        Extent
        
        """
        try:
            unitX, unitY = unit
        except:
            unitX, unitY = unit, unit
        
        # Look for bad sizes
        if (unitX> s.xMax-s.xMin): raise GeoKitExtentError("Unit size is larger than extent width")
        if (unitY> s.yMax-s.yMin): raise GeoKitExtentError("Unit size is larger than extent width")

        # Calculate new extent
        newXMin = np.floor(s.xMin/unitX)*unitX
        newYMin = np.floor(s.yMin/unitY)*unitY
        newXMax = np.ceil(s.xMax/unitX)*unitX
        newYMax = np.ceil(s.yMax/unitY)*unitY

        # Done!
        if dtype is None or isinstance(unitX,dtype):
            return Extent( newXMin, newYMin, newXMax, newYMax, srs=s.srs )
        else:
            return Extent( dtype(newXMin), dtype(newYMin), dtype(newXMax), dtype(newYMax), srs=s.srs )

    def corners(s, asPoints=False):
        """Returns the four corners of the extent as ogr.gGometry points or as (x,y)
        coordinates in the extent's srs

        """

        if (asPoints):
            # Make corner points
            bl = point( s.xMin, s.yMin, srs=s.srs )
            br = point( s.xMax, s.yMin, srs=s.srs )
            tl = point( s.xMin, s.yMax, srs=s.srs )
            tr = point( s.xMax, s.yMax, srs=s.srs )

        else:
            # Make corner points
            bl = ( s.xMin, s.yMin)
            br = ( s.xMax, s.yMin)
            tl = ( s.xMin, s.yMax)
            tr = ( s.xMax, s.yMax)

        return (bl, br, tl, tr)

    def castTo(s, srs):
        """
        Creates a new Extent by transforming an extent from the original Extent's
        srs to a target SRS. 
        
        Note:
        -----
        The resulting region spanned by the extent will be equal-to or (almost 
        certainly) larger than the original
        
        Parameters:
        -----------
        srs : Anything acceptable to geokit.srs.loadSRS()
            The srs to cast the Extent object to 

        Returns:
        --------
        Extent
        
        """
        srs=loadSRS(srs)

        # Check for same srs
        if( srs.IsSame(s.srs)):
            return s

        # Create a transformer
        srs = loadSRS(srs)
        transformer = osr.CoordinateTransformation(s.srs, srs)

        # Transform and record points
        X = []
        Y = []

        for pt in s.corners(True):
            try:
                pt.Transform( transformer )
            except Exception as e:
                print("Could not transform between the following SRS:\n\nSOURCE:\n%s\n\nTARGET:\n%s\n\n"%(s.srs.ExportToWkt(), srs.ExportToWkt()))
                raise e

            
            X.append(pt.GetX())
            Y.append(pt.GetY())

        # return new extent
        return Extent(min(X), min(Y), max(X), max(Y), srs=srs)

    def inSourceExtent(s, source):
        """Tests if the extent box is at least partially contained in the extent-box
        of the given vector or raster source"""
        if isinstance(source, str): 

            if isVector(source): isvec = True
            elif isRaster(source): isvec = False
            else: raise GeoKitExtentError("Could not handle source: "+source)

        elif isinstance(source, gdal.Dataset):
            if source.GetLayer() is None: isvec = False
            else: isvec = True

        elif isinstance( source, ogr.DataSource): isvec = True
        else:
            raise GeoKitExtentError("Source type could not be determined")

        # Check extent for inclusion
        if isvec: sourceExtent = Extent.fromVector(source).castTo(s.srs)
        else: sourceExtent = Extent.fromRaster(source).castTo(s.srs)
        
        for p in sourceExtent.corners(asPoints=True): # Are the source's corners in the main object's extent?
            if s._box.Contains(p): return True
        for p in s.corners(asPoints=True): # Are the main object's corners in the sources's extent?
            if sourceExtent._box.Contains(p): return True

        return False

    def filterSources(s, sources):
        """Filter a list of sources by those whose's envelope overlaps the Extent.
        
        Note:
        -----
        Creates a filter object which can be immediately iterated over, or else 
        can be cast as a list

        Parameters:
        -----------
        sources : list or str 
            The sources to filter
            * An iterable of vector/raster sources
            * An iterable of paths pointing to vector/raster sources
            * A glob string which will generate a list of source paths
                - see glob.glob for more info
        
        Returns:
        --------
        filter

        """
        # create list of searchable files
        if isinstance(sources, str):
            directoryList = glob(sources)
        else:
            directoryList = sources

        return filter(s.inSourceExtent, directoryList)
    
    def containsLoc(s, locs, srs=None):
        """Test if the extent contains a location or an iterable of locations

        Parameters:
        -----------
        locs : Anything acceptable to LocationSet()
            The locations to be checked
        

        srs : Anything acceptable to geokit.srs.loadSRS()
            The srs to cast the Extent object to 
        
        Returns:
        --------
        * If a single location is checker: bool
        * If multiple locations are checked: numpy.ndarray

        """
        boxG = s.box # initialize the box

        # Normalize locations
        locs = LocationSet(locs).asXY(s.srs) 

        # Do tests
        sel = np.ones(locs.shape[0], dtype=bool)
        sel *= locs[:,0] >= s.xMin 
        sel *= locs[:,0] <= s.xMax
        sel *= locs[:,1] >= s.yMin 
        sel *= locs[:,1] <= s.yMax

        # Done!
        if sel.size == 1: return sel[0] 
        else: return sel
    
    def contains(s, extent, res=None):
        """Tests if the extent contains another given extent
        
        Note:
        -----
        If an optional resolution ('res') is given, the containment value is also
        dependent on whether or not the given extent fits within the larger extent
        AND is situated along the given resolution

        Parameters:
        -----------
        extent : Extent
            The Extent object to test for containment
        
        res : numeric or tuple
            The X & Y resolution to enforce 
        
        Returns:
        --------
        bool

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
        """Finds the indexes of the given extent within the main extent according
        to the given resolution.
        
        Note:
        -----
        * Use this to compute the index offsets and window sizes of a window 
          within a raster dataset
        * The two extents MUST share the same SRS

        Parameters:
        -----------
        extent : Extent
            The extent to find within the calling extent
        
        res : numeric or tuple
            A resolution to check containment on

        yAtTop : bool; optional
            Instructs the offsetting to begin from yMax instead of from yMin

        Returns:
            tuple -> (xOffset, yOffset, xWindowSize, yWindowSize)

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

        if not (np.isclose(xOff, tmpX) and np.isclose(yOff, tmpY)):
            raise GeoKitExtentError("The extents are not relatable on the given resolution")

        # Get window sizes
        tmpX = (extent.xMax - extent.xMin)/dx
        xWin = int(np.round(tmpX))

        tmpY = (extent.yMax - extent.yMin)/dy
        yWin = int(np.round(tmpY))

        if not (np.isclose(xWin, tmpX) and np.isclose(yWin, tmpY)):
            raise GeoKitExtentError("The extents are not relatable on the given resolution")

        # Done!
        return IndexSet(xOff, yOff, xWin, yWin, xOff+xWin, yOff+yWin)
    

    #############################################################################
    ## CONVENIENCE FUNCTIONS
    def createRaster(s, pixelWidth, pixelHeight, **kwargs):
        """Convenience function for geokit.raster.createRaster which sets 'bounds'
        and 'srs' inputs

        * The input resolution MUST fit within the extent
        
        Parameters:
        -----------
        pixelWidth : numeric
            The pixel width of the raster in units of the input srs
            * The keyword 'dx' can be used as well and will override anything given 
            assigned to 'pixelWidth'
        
        pixelHeight : numeric
            The pixel height of the raster in units of the input srs
            * The keyword 'dy' can be used as well and will override anything given 
              assigned to 'pixelHeight'
        
        **kwargs:
            All other keyword arguments are passed on to geokit.raster.createRaster()

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None

        """
        if not s.fitsResolution((pixelWidth,pixelHeight)):
            raise GeoKitExtentError("The given resolution does not fit to the Extent boundaries")
        return createRaster( bounds=s.xyXY, pixelWidth=pixelWidth, pixelHeight=pixelHeight, srs=s.srs, **kwargs)
        
    def extractMatrix(s, source):
        """Extracts the extent directly from the given raster source as a matrix.
        
        Note:
        -----
        The called extent must fit somewhere within the raster's grid

        Returns:
        --------
        numpy.ndarray

        """
        
        # open the dataset and get description
        rasDS = loadRaster(source)
        rasInfo = rasterInfo(rasDS)
        rasExtent = Extent._fromInfo(rasInfo)

        # Find the main extent within the raster extent
        xO, yO, xW, yW, xE, yE = rasExtent.findWithin(s, res=(rasInfo.dx, rasInfo.dy), yAtTop=rasInfo.yAtTop)
        
        if xO<0 or yO<0 or xE>rasInfo.xWinSize or yE>rasInfo.yWinSize: 
            raise GeoKitExtentError( "The extent does not appear to fit within the given raster")
            
        # Extract and return the matrix
        arr = extractMatrix(rasDS, xOff=xO, yOff=yO, xWin=xW, yWin=yW)
        
        return arr

    def warp(s, source, pixelWidth, pixelHeight, strict=True, **kwargs):
        """Convenience function for geokit.raster.warp() which automatically sets the 
        'srs' and 'bounds' input. 

        Note:
        -----
        When creating an 'in memory' raster vs one which is saved to disk, a slightly
        different algorithm is used which can sometimes add an extra row of pixels. Be
        aware of this if you intend to compare value-matricies directly from rasters 
        generated with this function.
        
        Parameters:
        -----------
        source : str
            The path to the vector file to load
        
        pixelHeight : numeric; optional
            The pixel height (y-resolution) of the output raster
            * Only required if this value should be changed
        
        pixelWidth : numeric; optional
            The pixel width (x-resolution) of the output raster
            * Only required if this value should be changed

        strict : bool
            If True, raise an error if trying to warp to a pixelWidth and 
            pixelHeight which does not fit into the Extent
    
        **kwargs:
            All other keyword arguments are passed on to geokit.raster.warp()

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None

        """
        if strict and not s.fitsResolution((pixelWidth,pixelHeight)):
            raise GeoKitExtentError("The given resolution does not fit to the Extent boundaries")
        return warp(source=source, pixelWidth=pixelWidth, pixelHeight=pixelHeight, 
                    srs=s.srs, bounds=s.xyXY, **kwargs)

    def rasterize(s, source, pixelWidth, pixelHeight, strict=True, **kwargs):
        """Convenience function for geokit.vector.rasterize() which automatically
        sets the 'srs' and 'bounds' input. 

        Note:
        -----
        When creating an 'in memory' raster vs one which is saved to disk, a slightly
        different algorithm is used which can sometimes add an extra row of pixels. Be
        aware of this if you intend to compare value-matricies directly from rasters 
        generated with this function.
        
        Parameters:
        -----------
        source : str
            The path to the vector file to load
        
        pixelHeight : numeric; optional
            The pixel height (y-resolution) of the output raster
            * Only required if this value should be changed
        
        pixelWidth : numeric; optional
            The pixel width (x-resolution) of the output raster
            * Only required if this value should be changed

        strict : bool
            If True, raise an error if trying to rasterize to a pixelWidth and 
            pixelHeight which does not fit into the Extent
    
        **kwargs:
            All other keyword arguments are passed on to geokit.raster.warp()
        
        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None
        
        """
        if strict and not s.fitsResolution((pixelWidth,pixelHeight)):
            raise GeoKitExtentError("The given resolution does not fit to the Extent boundaries")
        return rasterize(source=source, pixelWidth=pixelWidth, pixelHeight=pixelHeight, 
                         srs=s.srs, bounds=s.xyXY, **kwargs)

    def extractFeatures(s, source, **kwargs):
        """Convenience wrapper for geokit.vector.extractFeatures() by setting the 
        'geom' input to the extent's box
        
        Parameters:
        -----------
        source : str
            The path to the vector file to load

        **kwargs:
            All other keyword arguments are passed on to vector.extractFeatures()
           
        Returns:
        --------
        * If asPandas is True: pandas.DataFrame or pandas.Series
        * If asPandas is False: generator

        """
        return extractFeatures( source=source, geom=s._box, **kwargs )


    def mutateVector(s, source, matchContext=False, **kwargs):
        """Convenience function for geokit.vector.mutateVector which automatically
        sets 'srs' and 'geom' input to the Extent's srs and geometry

        Note:
        -----
        If this is called without any arguments except for a source, it serves
        to clip the vector source around the extent

        Parameters:
        -----------
        source : Anything acceptable to geokit.vector.loadVector()
            The source to clip

        matchContext : bool; optional
            * If True, transforms all geometries to the Extent's srs before 
              mutating
            * If False, the Extent is cast to the source's srs, and all filtering
              and mutating happens in that context 

        **kwargs:
            All other keyword arguments are passed to geokit.vector.mutateVector
            

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None
        
        """
        # Get the working srs
        if not matchContext:
            vinfo = vectorInfo( source )
            ext = s.castTo(vinfo.srs)
        else:
            ext = s

        # mutate the source
        return mutateVector(source, srs=ext.srs, geom=ext._box, **kwargs)

    def mutateRaster(s, source, pixelWidth=None, pixelHeight=None, matchContext=False, warpArgs=None, processor=None, resampleAlg='bilinear', **mutateArgs):
        """Convenience function for geokit.vector.mutateRaster which automatically
        warps the raster to the extent's area and srs before mutating

        Note:
        -----
        If this is called without any arguments except for a source, it serves
        to clip the raster source around the Extent, therefore performing
        the same function as Extent.warp(...) on an Extent which has been cast
        to the source's srs

        Parameters:
        -----------
        source : Anything acceptable to geokit.raster.loadRaster()
            The source to mutate
        
        pixelHeight : numeric
            The pixel height (y-resolution) of the output raster
        
        pixelWidth : numeric
            The pixel width (x-resolution) of the output raster

        matchContext : bool; optional
            * If True, Warp to the Extent's boundaries and srs before mutating
                - pixelHeight and pixelWidth MUST be provided in this case
            * If False, only warp to the Extent's boundaries, but keep its 
              srs and resolution intact

        warpArgs : dict; optional
            Arguments to apply to the warping step
            * See geokit.raster.warp()

        processor - function; optional
            The function performing the mutation of the raster's data 
            * The function will take single argument (a 2D numpy.ndarray) 
            * The function must return a numpy.ndarray of the same size as the input
            * The return type must also be containable within a Float32 (int and 
              boolean is okay)
            * See example in geokit.raster.mutateRaster for more info

        resampleAlg : str; optional
            The resampling algorithm to use while warping
            * Knowing which option to use can have significant impacts!
            * Options are: 'near', 'bilinear', 'cubic', 'average'

        **kwargs:
            All other keyword arguments are passed to geokit.vector.mutateVector

        Returns:
        --------
        * If 'output' is None: gdal.Dataset
        * If 'output' is a string: None

        """
        if warpArgs is None: warpArgs = {}

        if processor is None: # We wont do a mutation without a processor, since everything else
                              # can be handled by Warp. Therefore we pass on any 'output' that is 
                              # given to the warping stage, unless one was already given
            warpArgs["output"] = warpArgs.get("output", mutateArgs["output"])
        
        # Warp the source
        if matchContext:
            if pixelWidth is None or pixelHeight is None:
                raise GeoKitExtentError("pixelWidth and pixelHeight must be provided when matchContext is True")

            source = s.warp( source, resampleAlg=resampleAlg, pixelWidth=pixelWidth, pixelHeight=pixelWidth, strict=True, **warpArgs )
        else:
            if not "srs" in mutateArgs:
                source = loadRaster(source)
                srs = source.GetProjectionRef()

            ext = s.castTo(srs)
            source = ext.warp( source, resampleAlg=resampleAlg, pixelWidth=pixelWidth, pixelHeight=pixelWidth, strict=False, **warpArgs )   

        # mutate the source
        if not processor is None: return mutateRaster(source, processor=processor, **mutateArgs)
        else: return source
