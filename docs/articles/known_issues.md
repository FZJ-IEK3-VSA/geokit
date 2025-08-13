# Known Issues

## Incompatibility with MacOS systems

Geokit shows various incompatibilities with Mac OS X for Geokit <1.5. These are mainly caused by GDAL and its inconsistent behaviour on different operating systems. These variations are addressed in newer GDAL versions, which require updates to Geokit. If you are using a MacOS machine, please use Geokit 1.5.0 or higher.

## Availability of multiprocessing for RegionMask on Windows, MacOS and Linux

The Geokit multiprocessing feature of Regionmask is only available on Linux, due to different multiprocessing implementations on MacOS and Windows compared to Linux. On MacOS and Windows, new processes are started for processes invoked by multiprocessing. This requires the function to be serialised. However, many GDAL objects cannot be serialised, which causes incompatibility.

In version 1.4.1, where multiprocessing was introduced, this caused the two tests, 'test_RegionMask_indicateValues' and
'test_RegionMask_indicateFeatures', to fail with the following warning:

'
UserWarning: Memory efficient multiProcess failed, returning to safe linear processing.
'
Therefore, starting with version 1.5.0, multiprocessing is automatically disabled on Windows and macOS systems, and a warning is displayed.

On Linux, however, new processes inherit from the initial process. This means that no serialisation with pickle is required. You can read more on this topic here: https://pythonforthelab.com/blog/differences-between-multiprocessing-windows-and-linux/




