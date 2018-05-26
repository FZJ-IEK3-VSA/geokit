from geokit.core.regionmask import *

#############################################################################################
def filterCoordinates(lats, lons, geom):
    """
    Filters a set of coordinates by those which are found within a given geometry

    Returns a dataframe detailing which coordinates (and their corresponding indexes) were identified

    Inputs:
        lats : y-dimension coordinates
            - list
            - numpy array
            * If 'lats' is 2-dimensional, its shape must fit that of 'lons'

        lons : x-dimension coordinates
            - list
            - numpy array
            * If 'lons' is 2-dimensional, its shape must fit that of 'lats'

        geom - ogr Geometry : The Geometry to filter by

    """

    # Ensure lats and lons are 2 dim arrays
    if( len(lats.shape)==1 and len(lons.shape)==1 ): # lats and lons are 1D
        lons, lats = np.meshgrid(lons,lats) # yes, it needs to be 'opposite'

    if( not (len(lats.shape)==2 and len(lons.shape)==2 and lats.shape==lons.shape)):
        raise ValueError("Lats and lons shapes do not match")

    # Ensure we are working with a geometry object
    if not isinstance(geom, ogr.Geometry): # Assume geom is a wkt string
        raise GeoKitError("geom input must be an OGR Geometry object")
    else:
        geom = geom.Clone() # Just in case the object is tethered outside

    # Maybe geom needs to be projected?
    geomSRS = geom.GetSpatialReference()
    if( not geomSRS.IsSame(EPSG4326) ):
        geom = geom.Clone() # Clone incase it it still used outside the function
        geom.TransformTo(EPSG4326)

    # Get extent of geom
    extent = Extent.fromGeom(geom)

    # Slice lats and lons around extent 
    tmp = np.logical_and( lats>=extent.yMin, lats<=extent.yMax )
    np.logical_and( lons>=extent.xMin, tmp, tmp )
    np.logical_and( lons<=extent.xMax, tmp, tmp )

    sel = np.where(tmp)

    # Iterate over extent-contained points and see if they are in region
    goodVals = dict(lat=[], lon=[], latI=[], lonI=[])

    xMap, yMap = np.meshgrid(range(lats.shape[1]), range(lats.shape[0]) )

    for lat, lon, lonI, latI in zip( lats[sel], lons[sel], xMap[sel], yMap[sel] ):
        # make a point
        pt = ogr.Geometry( ogr.wkbPoint )
        pt.AddPoint( lon, lat )
        if(geom.Contains(pt)):
            goodVals["lat"].append(lat)
            goodVals["lon"].append(lon)
            goodVals["latI"].append(latI)
            goodVals["lonI"].append(lonI)

    # Done!
    return pd.DataFrame(goodVals)