# Known Issues

## Incompatibility with macOS Systems

GeoKit shows various incompatibilities with Mac OS X for GeoKit < 1.5. These are mainly caused by GDAL and its inconsistent behaviour on different operating systems. These variations are addressed in newer GDAL versions, which require updates to GeoKit. If you are using a macOS machine, please use GeoKit 1.5.0 or higher.

## Availability of Multiprocessing for RegionMask on Windows, macOS, and Linux

The GeoKit multiprocessing feature of RegionMask is only available on Linux, due to different multiprocessing implementations on macOS and Windows compared to Linux. On macOS and Windows, new processes are started for each multiprocessing task. This requires the function to be serialised. However, many GDAL objects cannot be serialised, which causes incompatibility.

In version 1.4.1, where multiprocessing was introduced, this caused the two tests, `test_RegionMask_indicateValues` and
`test_RegionMask_indicateFeatures`, to fail with the following warning:

```python
UserWarning: Memory efficient multiProcess failed, returning to safe linear processing.
```

Therefore, starting with version 1.5.0, multiprocessing is automatically disabled on Windows and macOS systems, and a warning is displayed.

On Linux, however, new processes inherit from the initial process. This means that no serialisation with pickle is required. You can read more on this topic here: https://pythonforthelab.com/blog/differences-between-multiprocessing-windows-and-linux/




