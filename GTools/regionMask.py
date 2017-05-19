from .extent import *

RM_DEFAULT_SRS = 'europe_m'
RM_DEFAULT_RES = 100
RM_DEFAULT_PAD = None

class RegionMask(object):
    """The RegionMask object represents a given region and exposes methods allowing for easy
    manipulation of geospatial data around that region.

    RegionMask objects are defined by providing a polygon (either via a vector file, an OGR 
    Geometry, or a Well-Known-Text (WKT) string), a projection system to work in, and an extent and
    pixel resolution to create a matrix mask (i.e. boolean values) of.

    * The extent of the generated mask matrix is the tightest fit around the region in units
      of the pixel resolution. However, the extend can be defined explcitly defined if desired
    * The region can be manipulated as a vector polygon via the ".geometry" attribute, which 
      exposes the geometry as an OGR Geometry. To incoporate this into other vector-handeling 
      libraries it is suggested to use the ".ExportToWkt()" method available via OGR.
    * The region can be manipulated as a raster matrix via the ".mask" attribute which exposes 
      the mask as a boolean numpy ND-Array
    * Any raster source can be easily warped onto the region-mask's extent, projection, and 
      resolution via the ".warp" method
    * Any vector source can be rasterized onto the region-mask's extent, projection, and 
      resolution via the ".rasterize" method
    * The default mask set-up is as follows:
       SRS -------- EPSG3035
       RESOLUTION - 100m
       EXTENT ----- tight (as described above)

    Initialization:
        * RegionMask(mask-array, geographic-extent[, geometry ]) ### This is not the preferred way
        * RegionMask.fromVector( vector-data-source )
        * RegionMask.fromVectorFeature( vector-data-source, featureID )
        * RegionMask.fromGeom( OGR-Geometry or WKT-string, spatial-reference-system ) 
    """
    def __init__(s, data, extent, geom=None):
        """The default constructor for RegionMask objects. Creates a RegionMask directly from a matrix mask and a given extent (and optionally a geometry). Pixel resolution is calculated in accordance with the shape of the data mask and the provided extent

        * Generally one should use one of the .fromXXX methods to create RegionMasks

        Inputs:
            data - numpy-ndarray object
                * A 2-Dimensional numpy array of boolean values describing the region, where:
                    - 0/False -> "not in the region"
                    - 1/True  -> "Inside the region"

            extent - Extent object
                * The geographic extent and reference of the area represented by the data mask

            geom - ogr-Geomertry object (default None)
                * An OGR Geometry object which represents the region.
                ! Be careful that this geometry matches with the required "data" input. This is not checked within the constructor
        """

        # Mask type must be bool
        if(data.dtype != "bool" and data.dtype != "uint8" ): 
            raise ValueError("Mask data must be bool type")
        if(data.dtype == "uint8"):
            data = data.astype("bool")

        # set values
        s.mask = data
        s.extent = extent
        s.srs = extent.srs

        s.pixelWidth = (extent.xMax-extent.xMin)/(data.shape[1])
        s.pixelHeight = (extent.yMax-extent.yMin)/(data.shape[0])

        if( s.pixelHeight == s.pixelWidth ):
            s._pixelSize = s.pixelHeight
        else:
            s._pixelSize = None

        s.shape = data.shape

        s.height, s.width = data.shape

        # Create empty attributes dictionary
        s.attributes = {}

        # Set geometry
        s._geometry = geom

    @staticmethod
    def fromVector(source, pixelSize=RM_DEFAULT_RES, srs=RM_DEFAULT_SRS, extent=None, padExtent=RM_DEFAULT_PAD, **kwargs):    
        """
        Make a raster mask of a given shape or shapefile

        Returns a RegionMask object 

        Inputs:
            source - (None)
                : str -- Path to an input shapefile
                * either source or wkt must be provided
            
            pixelSize - (RM_DEFAULT_RES)
                : float -- the raster's pixel size in units of the srs

            srs - (RM_DEFAULT_SRS)
                : int -- The WKT SRS to use as an EPSG integer
                : str -- The WKT SRS to use as a WKT string
                : osr.SpatialReference -- The WKT SRS to use

            extent - (None) 
                : (xMin, yMix, xMax, yMax) -- the extents of the raster file to create
            
            padExtent - RM_DEFAULT_PAD
                float -- An extra paddind to ad onto the extent

            *Further kwargs are passed on to a gdal.Rasterize call. Use these to further customize the final mask
              Ex:
                - allTouched=True
                - where="color=='blue'"
        """
        # Handel source input
        sourceDS = loadVector(source)
        layer = sourceDS.GetLayer()
        shapeSRS = layer.GetSpatialRef()

        # Get Extent
        if( extent is None ): 
            extent = Extent.fromVector( sourceDS ).castTo(srs).fit(pixelSize)

        if( padExtent ): 
            extent = extent.pad(padExtent).fit(pixelSize)

        # Apply a spatial filter on the source layer 
        layer.SetSpatialFilterRect(*extent.castTo(shapeSRS).xyXY)

        # Determine pixelWidth and pixelHeight
        try:
            pWidth, pHeight = pixelSize
        except:
            pWidth, pHeight = pixelSize, pixelSize
        
        # Create an empty raster
        maskRaster = createRaster(bounds=extent.xyXY, pixelWidth=pWidth, pixelHeight=pHeight, srs=extent.srs)
        
        # Rasterize the shape
        bands = kwargs.pop("bands", [1])
        burnValues = kwargs.pop("burnValues", [1])
        
        err = gdal.Rasterize( maskRaster, sourceDS, bands=bands, burnValues=burnValues, **kwargs)
        if(err != 1):
            raise GToolsRegionMaskError("Error while rasterizing:\n%s"%err)

        maskRaster.FlushCache()

        # Get final stats
        band = maskRaster.GetRasterBand(1)
        array = band.ReadAsArray()

        # check if layer only has one feature, if so assume it is the geometry we will want
        if(layer.GetFeatureCount()==1):
            ftr = layer.GetFeature(0)
            tmpGeom = ftr.GetGeometryRef()
            geom = tmpGeom.Clone()

            del tmpGeom, ftr
        else: # Otherwise try to get the union of all features
            try:
                geom = geomListFlatten([ftr.GetGeometryRef() for ftr in loopFeatures(layer)])
            except:
                geom=None

        # Check if region geometry is in the given srs. If not, fix it
        srs = loadSRS(srs)
        if( not geom is None and not shapeSRS.IsSame(srs) ):
            geom.TransformTo(srs)

        # do cleanup
        del shapeSRS, layer, sourceDS

        # Done!
        return RegionMask(array, extent, geom)

    @staticmethod
    def fromGeom(geom,  wktSRS=EPSG4326, pixelSize=RM_DEFAULT_RES, srs=RM_DEFAULT_SRS, extent=None, padExtent=RM_DEFAULT_PAD, **kwargs):    
        """
        Make a raster mask of a given shape or shapefile

        Returns a Mask object 

        Keyword inputs:
            geom - (None)
                : str -- A WKT string representing the shape
                * either source or wkt must be provided

            wktSRS - (None)
                : int -- The WKT SRS to use as an EPSG integer
                : str -- The WKT SRS to use as a WKT string
                : osr.SpatialReference -- The WKT SRS to use
                * Must be defined if wkt is given
            
            pixelSize - (RM_DEFAULT_RES)
                : float -- the raster's pixel size in units of the srs

            srs - (RM_DEFAULT_SRS)
                : int -- The WKT SRS to use as an EPSG integer
                : str -- The WKT SRS to use as a WKT string
                : osr.SpatialReference -- The WKT SRS to use

            extent - (None) 
                : (xMin, yMix, xMax, yMax) -- the extents of the raster file to create
            
            padExtent - RM_DEFAULT_PAD
                float -- An extra paddind to ad onto the extent

            *Further kwargs are passed on to a gdal.Rasterize call. Use these to further customize the final mask
              Ex:
                - allTouched=True
                - where="color=='blue'"
        """
        # make a vector dataset
        if( isinstance(geom, str)):
            if srs is None: raise ValueError("srs must be provided when geom is a string")
            geom = makeGeomFromWkt(geom, wktSRS)

        ds = createVector( geoms=[geom,] )

        # Call 'fromVector'
        kwargs["pixelSize"]=pixelSize
        kwargs["extent"]=extent
        kwargs["srs"]=srs
        kwargs["padExtent"]=padExtent

        return RegionMask.fromVector(ds, **kwargs)

    @staticmethod
    def fromVectorFeature(source, select=0, pixelSize=RM_DEFAULT_RES, srs=RM_DEFAULT_SRS, extent=None, padExtent=RM_DEFAULT_PAD, **kwargs):
        """Load a specific feature from a shapefile as a Region, including feature attributes

        Inputs:
            source
                str -- Path to the source shapefile
                ogr.Dataset -- The source dataset 

            select - (0)
                int -- The feature index to select
                str -- An SQL where statement to fiter source features

            pixelSize - (RM_DEFAULT_RES)
                : float -- the raster's pixel size in units of the srs

            srs - (RM_DEFAULT_SRS)
                : int -- The WKT SRS to use as an EPSG integer
                : str -- The WKT SRS to use as a WKT string
                : osr.SpatialReference -- The WKT SRS to use

            extent - (None) 
                : (xMin, yMix, xMax, yMax) -- the extents of the raster file to create
            
            padExtent - RM_DEFAULT_PAD
                float -- An extra paddind to ad onto the extent

            *Further kwargs are passed on to a gdal.Rasterize call. Use these to further customize the final mask
              Ex:
                - allTouched=True
                - where="color=='blue'"
            
        """
        # Get DS, Layer, and Feature
        vecDS = loadVector(source)
        vecLyr = vecDS.GetLayer()

        if( isinstance(select,str) ):
            where = select
            t = vecLyr.SetAttributeFilter(where)
            if( t!=0 ): raise GToolsRegionMaskError("Error in select statement: \""+where+"\"")
            
            if vecLyr.GetFeatureCount()  > 1: raise GToolsRegionMaskError("Multiple fetures found")
            if vecLyr.GetFeatureCount() == 0: raise GToolsRegionMaskError("Zero features found")

            vecFtr = vecLyr.GetNextFeature()
            
        else: # Assume select is an integer
            if not isinstance(select, int):
                raise ValueError("select must be either an SQL 'where' clause, or a feature index")
            vecFtr = vecLyr.GetFeature(select)
            
        if vecFtr is None:
            raise GToolsRegionMaskError("Could not extract feature!")

        # Create a new Region
        geom = vecFtr.GetGeometryRef()

        kwargs["pixelSize"]=pixelSize
        kwargs["extent"]=extent
        kwargs["srs"]=srs
        kwargs["padExtent"]=padExtent
        region = RegionMask.fromGeom( geom, **kwargs )

        # Add attributes to region object
        region.attributes = vecFtr.items().copy()

        return region

    @property
    def envelope(s):
        """Shortcut to obtain the envelope of the RasterMask's extent as an OGR Geometry"""
        if( not hasattr(s,"_envelope")):
            xMin, yMin, xMax, yMan = s.extent.xyXY
            s._envelope = makeBox(xMin, yMin, xMax, yMan, s.srs)
        return s._envelope.Clone()

    @property
    def pixelSize(s):
        """The RegionMask's pixel size. 

        !!Only available when pixelWidth equals pixelHeight!!"""
        if s._pixelSize is None:
            raise GToolsRegionMaskError("Function only accessable when pixelWidth equals pixelHeight")
        return s._pixelSize

    @property
    def geometry(s):
        """Fetched a clone of the RegionMask's geometry (as an OGR Geometry object).

        * If a geometry was not provided when the RegionMask was initialized, then one will be generated from RegionMask's mask matrix in the RegionMask's extent
        * The geometry can always be deleted and rebuild using the RegionMask.rebuildGeometry() function
        """

        if(s._geometry is None):
            s._geometry = makeGeomFromMask( s.mask, s.extent, flatten=True )

        return s._geometry.Clone()

    def draw( s, method='image', plotOb=None, downScaleFactor=5, srs=None, **kwargs ):
        """Draw the region on a matplotlib figure

        !!Currently in testing!!

        Returns a handle to the drawn object

        Inputs:
            method - (default 'line')
                str -- The method to use when drawing
                    * If 'line' draw the RegionMask's geometry's boundary as a line using plt.plot( xVals, yVals)
                    * If 'image' draw the RegionMask's mask matrix as an image using plt.imshow
                        !NOTE! imshow will draw the mask on an index-grid, and will not line up with actual coordinates
            plotOb - (default None)
                matplotlib.pyplot object -- The plot object to use when plotting
                    * If None is given, generate one in the function and then call 'plt.show()'

            downScaleFactor - (default 0)
                int -- The amount to downscale the visualized data
                    * If method is 'line', downScaleFactor will simple choose every N-th boundary coordinate (where N is equal to the downScaleFactor)
                    * if method is 'image, downScaleFactor controls the downScaling of RM's mask matrix via metisGIS.scaleMatrix(...)

            kwargs are passed on to either plt.plot or plt.imshow depending on the given method

        """
        # Test if we need to import matplotlib
        doShow = False
        if plotOb is None:
            import matplotlib.pyplot as plotOb
            doShow = True

        # Do "line" draw
        if method == 'line':
            print("DONT FORGET BUG! - fails for multipolygon geoms")
            # Make the boundary
            boundary = s.geometry.Boundary()

            # Do transform, maybe
            if srs and not srs.IsSame(s.srs):
                boundary.TransformTo(srs)
            
            # get the points!
            pts = np.array(boundary.GetPoints())
                
            if downScaleFactor == 0:
                xPts = pts[:,0]
                yPts = pts[:,1]

            elif downScaleFactor > 0:
                xPts = [x.mean() for x in np.array_split(pts[:,0], pts.shape[0]//downScaleFactor)]
                yPts = [y.mean() for y in np.array_split(pts[:,1], pts.shape[0]//downScaleFactor)]

                xPts.append(xPts[0])
                yPts.append(yPts[0])

            else:
                raise RuntimeError("downScaleFactor must be integers >= 0")

            h = plotOb.plot(xPts,yPts, **kwargs)

        # Do "image" draw
        elif method == 'image':
            
            if downScaleFactor == 0:
                drawMat = s.mask
            elif downScaleFactor > 0:
                drawMat = scaleMatrix(s.mask, -1*downScaleFactor, strict=False)    
            else:
                raise RuntimeError("downScaleFactor must be integers >= 0")
            
            h =  plotOb.imshow(drawMat, **kwargs)

        # Do "image" draw
        elif method == 'contour':
            
            if downScaleFactor == 0:
                drawMat = s.mask
            elif downScaleFactor > 0:
                drawMat = scaleMatrix(s.mask, -1*downScaleFactor, strict=False)    
            else:
                raise RuntimeError("downScaleFactor must be integers >= 0")
            
            y = np.linspace(s.extent.yMin,s.extent.yMax, drawMat.shape[0])
            x = np.linspace(s.extent.xMin,s.extent.xMax, drawMat.shape[1])
            
            h =  plotOb.contourf(x,y,drawMat[::-1,:], **kwargs)
        
        # All done!
        if (doShow):
            plotOb.show()
        else:
            return h

    def rebuildGeometry(s):
        """Rebuild the RM's geometry. Use this if the RM was initialized with an incorrect geometry.

        * This should never happen, but, just in case....
        """
        s._geometry = None
        return s.geometry

    def _tempFile(s, head="tmp", ext=".tif"):
        """***RM INTERNAL***

        Use this to create a temporary file associated with the RegionMask which will be deleted when the RM goes out of scope.

        !! BEWARE OF EXTERNAL DEPENDANCIES WHEN THE RM IS GOING OUT OF SCOPE, THIS WILL CAUSE A LOT OF ISSUES !!
        """
        if(not hasattr(s,"_TMPDIR")):
            # Create a temporary directory to use with this shape (and associated processes)
            s._TMPDIR = TemporaryDirectory()
        return NamedTemporaryFile(suffix=ext, prefix=head, dir=s._TMPDIR.name, delete=True).name

    def __del__(s):
        if(hasattr(s, "_TMPDIR")): s._TMPDIR.cleanup()

    def extractAttributes(s, source, header=""):
        """
        Extract the attributes of a vector source which contains a given geometry
        * The first feature which contains at least 90% of the given geometry will be selected

        Input arguments:
            source - (required) 
                str -- The path to the containing vector data source

            header - (default "")
                str -- A header to add to the source's fields when inserting into the RM's attribute dictionary
        """

        # Open master ds
        masterDS = loadVector(source)
        masterLyr = masterDS.GetLayer()
        masterSRS = masterLyr.GetSpatialRef()

        # Filter source 
        masterLyr.SetSpatialFilterRect(*s.extent.castTo(masterSRS).xyXY)

        # Cast region, maybe?
        regGeom = s.geometry
        if( not masterSRS.IsSame(regGeom.GetSpatialReference())):
            regGeom.TransformTo(masterSRS)

        regArea = regGeom.Area()

        # Find containing feature
        featureFound = False
        for masterFtr in loopFeatures(masterLyr):
            masterGeom = masterFtr.GetGeometryRef()

            overlap = regGeom.Intersection(masterGeom)
            overlapArea = overlap.Area()

            if ( overlapArea/regArea >= 0.95 ):
                featureFound = True
                containingFtr = masterFtr
                break
        
        if( not featureFound ):
            raise GToolsRegionMaskError("No containing feature found")

        # cleanup
        del masterLyr, masterDS

        # assign into attributed
        for k,v in containingFtr.items().items():
            s.attributes[header+k] = v

    def createRaster(s, resolutionDiv=1, **kwargs):
        """Creates a new raster with the same extent and resolution as the parent Mask, but with a optional datatype and fill.

            * All kwargs are passed on to reslo.gis.createRaster (again, except extent and resolution)
            * extent, pixelWidth, pixelHeight, and srs are automatically defined

            Inputs:
                resolutionDiv: int -- A division factor for the raster's resolution
                    * Determines the resolution of the "divided" raster
                    * Generally this is only used for internal purposes

        """
        # protect from bad inputs
        trash = kwargs.pop("bounds", None)
        trash = kwargs.pop("pixelHeight", None)
        trash = kwargs.pop("pixelWidth", None)
        trash = kwargs.pop("srs", None)

        # Check for division
        if(resolutionDiv<0): resolutionDiv = 1.0/abs(resolutionDiv)
        pW, pH = s.pixelWidth/resolutionDiv, s.pixelHeight/resolutionDiv

        rasDS = createRaster( bounds=s.extent.xyXY, pixelWidth=pW, pixelHeight=pH, srs=s.srs, **kwargs)

        # Return result
        if("output" in kwargs):
            return
        else:
            rasDS.FlushCache()
            return rasDS

    def applyMask(s, mat, noData=0):
        """Shortcut to apply the RM's mask to an array

            * When the passed matrix does not have the same extents of the, it is assumed that the RM's mask needs to be scaled so that the matrix dimensions match

            * The RM's mask can only be scalled UP, and the given matrix's dimensions must be mutiples of the mask's dimensions

        Inputs:
            mat: 
                np.ndarray -- The matrix to apply the mask to
                * A 2D numpy array

            noData: (default 0.0)
                float -- The noData value to set the mask-disqualified values in the given matrix to 
        """
        if(noData is None): noData=0
        # Get size
        Y,X = mat.shape

        # make output array
        #out = np.zeros(mat.shape, dtype=mat.dtype)
        #if( noData!=0 ): out.fill(noData)
        out = np.array(mat)

        # Apply mask
        if( s.shape == mat.shape ): # matrix dimensions coincide with mask's data
            sel = np.where(~s.mask)
            out[sel] = noData

        elif( Y>s.height and X>s.width ):
            if( not Y%s.height==0 or not X%s.width==0 ):
                raise GToolsRegionMaskError("Matrix dimensions must be multiples of mask dimensions")

            yScale = Y//s.height
            xScale = X//s.width

            scaledMask = scaleMatrix(s.mask, (yScale,xScale))
            sel = np.where(~scaledMask)
            out[sel] = noData

        else:
            raise GToolsRegionMaskError("Could not map mask onto matrix")

        return out

    ##############################################################################    
    # Warp raster onto region
    def warp(s, source, dtype=None, resampleAlg='cubic', noDataValue=None, applyMask=True, resolutionDiv=1, **kwargs):
        """Warp a given raster source onto the RM's extent and resolution

        * The source is not clipped around the RM's extent before the warping procedure. This isn't necessary, but if it is desired it is suggested to call 'clipRaster' from the RM's extent (givign the source as an input) and then passing the returned value from that into the source input. 
            Ex.
                -> clippedDS = RM.extent.clipRaster(<source>)
                -> warpedMatrix = RM.warp( clippedDS )

        Returns a 2D matrix of the warped data fitting the RM's mask

        Inputs:
            source:
                str -- The data source to warp
                gdal-Dataset -- The dataset to warp
                * Must be a GDAL-Readable source

            dtype: (None)
                str -- the datatype as a numpy-readable string
                type -- the data type as a native python type
                * This controls the datatype of the warped data

            resampleAlg: ('cubic')
                str -- The resampling algorithm to use when warping
                * See gdal.WarpOptions for more info
                * The common options are: cubic, linear, near

            noDataValue: (None)
                float -- The no-data-value to use for the warped data
                * Will be cast into the given dtype if one was provided

            applyMask: (True)
                bool -- Flag determining whether or not to apply the RM's mask to the resulting data

            resolutionDiv: (1)
                int: A resolution scaling factor to allow for warping onto the same extent as the RM, but a higher resolution
                * Generally this is intended for internal use

            * All kwargs are passed on to a call to gdal.Warp()
        """
        # open source and get info
        source = loadRaster(source)
        dsInfo = rasterInfo(source)
        workingExtent = Extent._fromInfo(dsInfo)

        # Test is the extents already match, and a projection is not necessary
        if ( workingExtent == s.extent and isclose(dsInfo.dx, s.pixelWidth/resolutionDiv) and isclose(dsInfo.dy, s.pixelHeight/resolutionDiv)):

            # Read the array
            final = source.GetRasterBand(1).ReadAsArray()

            # Make sure we have the desired datatype
            if(dtype):
                final = final.astype(dtype)

        else: # A projection is required
            # Create a target raster in shape of the mask
            dtype = dtype if dtype else dsInfo.dtype

            targetDS = s.createRaster(resolutionDiv, dtype=dtype)
            if(targetDS is None): raise GToolsRegionMaskError("Error creating temporary mask-like matrix")

            # Warp working raster to the mask-like raster
            mt = kwargs.pop("multithread", True)
            r = gdal.Warp(targetDS, source, multithread=mt, resampleAlg=resampleAlg, **kwargs)
            targetDS.FlushCache()

            # Get resulting array
            final = targetDS.GetRasterBand(1).ReadAsArray()

            # clean up
            del targetDS

        # do some cleanup
        del source

        # Apply mask, maybe
        if(applyMask): final = s.applyMask(final, noDataValue)
    
        # Return
        return final

    def rasterize(s, source, applyMask=True, dtype="bool", noDataValue=None, resolutionDiv=1, **kwargs):
        """Rasterizes a given vector source onto the RM's extent and resolution

        Returns a 2D-matrix which matches the RM's mask

        Inputs:
            source:
                str -- A path to a vector source
                gdal-Dataset -- A dataset object for the vector source

            applyMask: (True)
                bool -- Flag determining whether or not to apply the RM's mask to the resulting data

            dtype: (bool)
                str -- the datatype as a numpy-readable string
                type -- the data type as a native python type
                * This controls the datatype of the rasterized data

            noDataValue: (None)
                float -- The no-data-value to use for the warped data
                * Will be cast into the given dtype if one was provided

            resolutionDiv: (1)
                int: A resolution scaling factor to allow for rasterizing onto the same extent as the RM, but at a higher resolution
                * Generally this is intended for internal use
                * It is particularly useful in the case when vector features are small compared to the RM's resolution (such as buildings!)

            kwargs:
                * All kwargs are passed on as options in a call to gdal.Rasterize()
                * See gdal.RasterizeOptions for more info
                * If neither an 'attributes' or a 'burnValues' option are given, a 'burnValues' option is added equaling [1]
                * Most notably, the "where" option can be used to filter and select particular features from the source. Again, see gdal.RasterizeOptions for more info
        """
        # load the vector source
        source = loadVector(source)

        # Create temporary output file
        outputDS = s.createRaster(resolutionDiv, dtype=dtype)

        # Do rasterize
        bands = kwargs.pop("bands",[1])
        if(not( "attribute" in kwargs or "burnValues" in kwargs)):
            kwargs["burnValues"] = [1]

        tmp = gdal.Rasterize( outputDS, source, bands=bands, **kwargs)
        if(tmp==0):
            raise GToolsRegionMaskError("Rasterization failed!")
        outputDS.FlushCache()

        # Read array
        final = outputDS.GetRasterBand(1).ReadAsArray()

        # Apply mask, maybe
        if(applyMask): final = s.applyMask(final, noDataValue)
               
        # clean up
        del outputDS

        # Return
        return final
    
    #######################################################################################
    ## Raw value processor
    def indicateValues(s, dataSet, value, nanFill=None, forceMaskShape=True, **kwargs):
        """
        Convenience function to indicates a particular value or range of values from a given raster datasource onto the RegionMask.

        * An 'AND' operation is performed among the different subsets of min, max, and equal 
          specifiers (assuming they are not 'None')
        * Value processing is performed BEFORE a warp takes place (if one is necesary)
        * After warping, all values >0.5 are set as True. Others are set as False

        Returns a raster mask matching the RegionMask's mask dimensions where 0 means the pixels is not included in the indicated set, and 1 meaning the pixel is included in the indicated set. 

        Arguments:
            dataSet
                str -- Path to the raster file over which to process
                gdal.Dataset -- The input data source as a gdal dataset object

            value
                float -- The value to accept
                float, float -- The inclusion Min and Max to accept
                * When giving a min/max range, None refers to no bound

            nanFill - (None)
                numeric (datasource dtype) -- If the raw datasource data contains nan values, fill them with this number before processing

            kwargs -- Passed on to RegionMask.warp()
                * Most notably: resampleAlg
        """
        # Unpack value
        if isinstance(value, tuple):
            valueMin,valueMax = value
            valueEquals = None
        else:
            valueMin,valueMax = None,None
            valueEquals = value

        # make processor
        def processor(data):
            # fill nan values, maybe
            if(not nanFill is None): data[ np.isnan(data) ] = nanFill
            
            output = np.ones(data.shape, dtype="bool")
            
            if(not valueMin is None):
                np.logical_and(data >= valueMin, output, output)
            if(not valueMax is None):
                np.logical_and(data <= valueMax, output, output)
            if(not valueEquals is None):
                np.logical_and(data == valueEquals, output, output)

            return output

        # Do processing
        processedDS = rasterMutate(dataSet, s.extent, processor=processor, dtype="bool")

        # Warp onto region
        final = s.warp(processedDS, dtype="float32", **kwargs)

        # apply a threshold incase of funky warping issues
        final[final>1.0] = 1
        final[final<0.0] = 0

        # Make sure we have the mask's shape
        if forceMaskShape:
            rd = kwargs.get("resolutionDiv",None)
            if not rd is None:
                final = scaleMatrix(final, -1*rd)

        # Return result
        return final

    #######################################################################################
    ## Vector processor
    def indicateAreas(s, dataSet, whereField=None, whereValues=None, forceMaskShape=True, **kwargs):
        """
        Convenience function to indicates the RegionMask (RM) pixels which are found within the areas (or a subset of the areas) defined in a given vector datasource

        * A Rasterization is performed from the input data set to the RM's mask.
          See "gdal.RasterizeOptions" kwargs for info on how to control the 
          rasterization step.

        Arguments:
            dataSet
                str -- Path to the vector file over which to process
                ogr.Dataset -- The input data source as an initialized ogr dataset object

            whereField - (None)
                str -- The name of a column to fliter the vector features by

            whereField - (None)
                list -- The values of a filter-column to accept when filtering the vector features
                
            kwargs -- Passed on to RegionMask.rasterize()
                * Most notably: 'where', 'resolutionDiv', and 'allTouched'
        """
        # Ensure path to dataSet exists
        if( not isinstance(dataSet, gdal.Dataset) and (not os.path.isfile(dataSet))):
            msg = "dataSet path does not exist: {}".format(dataSet)
            raise ValueError(msg)
        
        # Create a where statement if needed
        if whereField:
            where = ""
            for value in whereValues:
                if isinstance(value,str):
                    where += "%s='%s' OR "%(whereField, value)
                elif isinstance(value,int):
                    where += "%s=%d OR "%(whereField, value)
                elif isinstance(value,float):
                    where += "%s=%f OR "%(whereField, value)
                else:
                    raise GToolsRegionMaskError("Could not determine value type")
            where = where[:-4]
            kwargs["where"] = where
        
        # Do rasterize
        final = s.rasterize( dataSet, dtype="bool", bands=[1], burnValues=[1], **kwargs )

        # Make sure we have the mask's shape
        if forceMaskShape:
            rd = kwargs.get("resolutionDiv",None)
            if not rd is None:
                final = scaleMatrix(final, -1*rd)

        # Return
        return final
