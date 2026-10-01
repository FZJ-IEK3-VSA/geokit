from warnings import warn
from sys import platform


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


class GeoKitDataTypeError(GeoKitError):
    """A data-type request that GeoKit cannot carry out.

    Raised for a bare integer or an unsupported type passed as ``dtype``, and for a noData, fill or burn value
    that the chosen type cannot store. See ``geokit.dtypes``.
    """


class GeoKitDataTypeWarning(UserWarning):
    """A data-type check found a possible loss: precision above 2**53, or pixels created without a noData value.

    ``geokit.dtypes.set_options(checks=False)`` turns these warnings off.
    """


class GeokitMultiProcessingWarning(Warning):
    multiProcessingWarningMessage = (
        "Multiprocessing has been set to 'False' because it is not available for Windows or Mac."
        " To deactivate this warning, please set the multiProcess variable to False. On Windows and "
        "Mac, new processes must be spawned, which requires the serialisation of the method to be "
        "executed via multiprocessing. However, Geokit contains objects that cannot be serialised by Pickle. "
        "On Linux, however, new processes are inherited and no serialisation is required."
    )


def checkMultiProcessingAvailability(multiProcess: bool) -> bool:
    """Multiprocessing is not available on all operating systems. If the user wants to
    to use multiprocessing on an unsupported operating system, multiprocessing will be
    deactivated and a warning appears.

    Parameters
    ----------
    multiProcess : bool
        A flag indicating whether multiprocessing should be used as indicated by the user.
        If multiprocessing is not available for the operating system, multiprocessing will be deactivated and a warning appears.

    Returns
    -------
    bool
        The corrected value for multiprocessing availability.
    """
    if platform == "linux" or platform == "linux2":
        multiProcessCorrected = multiProcess
    elif platform == "darwin" and multiProcess is True:
        multiProcessCorrected = False
        warn(message=GeokitMultiProcessingWarning.multiProcessingWarningMessage, category=GeokitMultiProcessingWarning)
    elif platform == "win32" and multiProcess is True:
        multiProcessCorrected = False
        warn(message=GeokitMultiProcessingWarning.multiProcessingWarningMessage, category=GeokitMultiProcessingWarning)
    elif multiProcess is False:
        multiProcessCorrected = multiProcess
    else:
        multiProcessCorrected = False
        warn(message=GeokitMultiProcessingWarning.multiProcessingWarningMessage, category=GeokitMultiProcessingWarning)
    return multiProcessCorrected
