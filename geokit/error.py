class GeoKitError(Exception):
    pass


class GeoKitGeomError(GeoKitError):
    pass


class GeoKitRegionMaskError(GeoKitError):
    pass


class GeoKitRasterError(GeoKitError):
    pass


class GeoKitExtentError(GeoKitError):
    pass


class GeoKitLocationError(GeoKitError):
    pass


class GeoKitSRSError(GeoKitError):
    pass


class GeoKitVectorError(GeoKitError):
    """Marks an error that is specific to geokit behavior.

    Parameters
    ----------
    UTIL : _type_
        _description_
    """

    pass


class GeoKitCDataError(GeoKitError):
    pass
