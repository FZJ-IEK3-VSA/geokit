# Release 1.7.0 Issue Summary

This document provides a comprehensive overview of all issues included in the Release 1.7.0 milestone.

**Total Issues:** 22
**All Issues Status:** Closed ✓

## Overview by Category

- **Plotting/Visualization Enhancements:** 5 issues
- **Raster Operations Improvements:** 6 issues
- **Vector/Geometry Operations:** 5 issues
- **Data Handling:** 2 issues
- **Code Quality & Developer Experience:** 3 issues
- **Bug Fixes & Stability:** 1 issues

---

## Plotting/Visualization Enhancements

### Issue #294: allow mixed NaN and value geometry plots

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/294

**Description:**
### Purpose of the improvement

Currently, gk.drawGeoms() can only color geom dataframes according to value columns which are purely not NaN. As soon as one NaN only is in the data iterable, all geoms will be plotted "empty". There is a workaround but it is a bit clumsy and has limitations, see below.

```python
import geokit as gk
import pandas as pd
import numpy as np

box1=gk.geom.box(0,0,1,1)
box2=gk.geom.box(1,0,2,1)
box3=gk.geom.box(2,0,3,1)

df = pd.DataFrame()
df["geom"] = [box1, box2, box3]
df["value"] = [43, np.nan, 0]

# first show how it looks today - all fields empty, even if box1 and 3 have values
gk.drawGeoms(df, colorBy="value", draw_cbar=True)
```
<img width="1102" height="362" alt="Image" src="https://github.com/user-attachments/assets/2364cdb1-155a-4520-88db-327bd08d7de3" />

```python
# second show the current work around:
# generate a plot of only the "colored" geoms with values
# extract the axis of the first plot
# the add the empty plots onto the same axis with 

... (truncated, see issue for full details)

---

### Issue #293: draw_cbar arg not backward compatible

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/293

**Description:**
### Purpose of the improvement

I think the new option to allow NOT to draw a colorbar in gk.drawGeoms() is a very nice and meaningful one! But why was it set to draw_cbar=False as a default? That means that all plots turn out different now, unless one adds the argument everywhere in the code? Can we default it to True instead? 
Also, the arg lacks a description in the docstring so far and should not be named in underscore style but instead drawcbar, possibly with a capital C and/or B. Even if I would not choose it for a new project, everything else here is camelCase and PEP8 is very clear about consistency throughout a project or even module...

### Proposal

Rename arg, add docstring desciption and default to True

---

### Issue #278: gk.drawImage does not work with ax input

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/278

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [x] I have confirmed this bug exists on the lastest commit on the dev branch.

- [ ] I have confirmed this bug exists for a specific version the software.


### Issue Description

plotting into the same axis is allowed in gk.drawImage() but leads to this error due to how pyplot is imported only in an if ax is None statement:

---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
Cell In[60], [line 11](vscode-notebook-cell:?execution_count=60&line=11)
      8 mx_zero[mx_zero>0] = np.nan
     10 ax = gk.drawImage(mx)
---> [11](vscode-notebook-cell:?execution_count=60&line=11) gk.drawImage(mx_zero, ax=ax)

File ~/libs/geokit/geokit/core/util.py:683, in drawImage(matrix, ax, xlim, ylim, yAtTop, scaling, fontsize, hideAxis, figsize, cbar, cbarPadding, cbarTitle, vmin, 

... (truncated, see issue for full details)

---

### Issue #264: Fix GeoKit Plotting Functions

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/264

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [x] I have confirmed this bug exists on the lastest commit on the dev branch.

- [ ] I have confirmed this bug exists for a specific version the software.


### Issue Description

There are multiple issues with GeoKit's plotting functions, which are likely related. This issue collects them all together to track progress. 

### Reproducible Example

```python
The problems are described in the subissues.
```

### Expected Behavior

The plotting should work as expected

### pip list -v

```bash

```

---

### Issue #29: allow colorBy str values when drawing geoms

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/29

**Description:**
gk.drawGeoms(colorBy=...) only accepts columns with numeric values. Allow plotting also based on other values.
This issue was originally created by: @chrisjwin

---

## Raster Operations Improvements

### Issue #286: Remove ComputeStatistics in rasterInfo where possible

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/286

**Description:**
### Purpose of the improvement

Calling ComputeStatistics in rasterInfo is computationally expensive. This function is used to determine the minimum and maximum values of the raster, but this information is not required each time and should therefore only be computed when necessary.

### Proposal

Add a flag to the rasterInfo that only calculates the minimum and maximum values when necessary.

---

### Issue #285: deal with rasters without srs in warp

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/285

**Description:**
### Purpose of the improvement

Rasters do not necessarily need a spatial reference (the whole geom system works with and without SRS as well), and consequently, gk.raster.createRaster() allows creating rasters with and without (optional) srs argument. 

However, gk.raster.warp() does not accept rasters without srs at all. While that would be understandable when the SRS shall be warped (one cannot warp from no srs to some srs), it also applies when srs is not part of the warp. The error message is very non-descriptive though, one has to know osgeo well to guess the reason.

See example code here:

```python
import geokit as gk
import numpy as np

# generate the same raster twice, once with and once without srs
arr = np.array([[50,100,150], [200,250,255]])
rstr_withsrs = gk.raster.createRaster(
    data=arr,
    bounds=(0,0,3,2),
    pixelWidth=1,
    pixelHeight=1,
    srs=4326,
)

rstr_nosrs = gk.raster.createRaster(
    data=arr,
    bounds=(0,0,3,2),
    pixelWidth=1,
    pixelHeigh

... (truncated, see issue for full details)

---

### Issue #274: raster.interpolateValues() cannot deal with point lists of len = 1 anymore

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/274

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [ ] I have confirmed this bug exists on the lastest commit on the dev branch.

- [x] I have confirmed this bug exists for a specific version the software.


### Issue Description

When data for a list of points with only one point tuple shall be extracted, the code fails now with the following error:

> ---------------------------------------------------------------------------
> AttributeError                            Traceback (most recent call last)
Cell In[1], [line 3](vscode-notebook-cell:?execution_count=1&line=3)
      1 import geokit as gk
----> [3](vscode-notebook-cell:?execution_count=1&line=3) gk.raster.interpolateValues(
      4     source='/fast/home/c-winkler/libs/RESKit/reskit/weather/Era5Source/data/ERA5_surface_solar_radiation_downwards_mean.tiff', 
      5     points=[(-79.94, 37.27)], 
      6     mode='linear-spline')
File ~/libs/geokit/g

... (truncated, see issue for full details)

---

### Issue #268: warp() assigns wrong datatype depending on input for fill and noData

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/268

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [x] I have confirmed this bug exists on the lastest commit on the dev branch.

- [ ] I have confirmed this bug exists for a specific version the software.


### Issue Description

Depending on the values for noData and fill, the warp function assigns the wrong data types to the output matrix. This results in uncaught overflow errors. 

The following faulty output results from the reproducible example:

>[[5 0 0]
> [2 3 7]]

### Reproducible Example

```python
import numpy as np

    import geokit as gk

    raster_matrix_2x3 = np.array(
        [
            [5, 255, 0],
            [2, 3, 7],
        ],
        dtype=np.uint8,
    )

    raster = gk.raster.createRaster(
        bounds=[0, 0, 3, 2],
        pixelWidth=1,
        pixelHeight=1,
        data=raster_matrix_2x3,
        srs=4326,
        noData=255,
        # output=intermediate_raster_tif_str,
  

... (truncated, see issue for full details)

---

### Issue #261: test_contours, test_Extent_contoursFromRaster and test_polygonize_matrix_all_false prompt warnings

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/261

**Description:**
### Purpose of the improvement

test_contours and  test_Extent_contoursFromRaster prompt the following warning

> DeprecationWarning: The current behavior of geokits's contours function is deprecated. GDAL has changed how contours are drawn close to minimum and maximum values, and will discontinue the current behavior in GDAL 3.11 or 3.12. Geokit will also drop the current behavior with the next GDAL update. For more information, please see the following discussion: https://github.com/OSGeo/gdal/issues/12938.

test_poylgonizeMatrix_all_false prompts the following warning

>  UserWarning: No features created in temporary layer


### Proposal

The warning in test_contours and  test_Extent_contoursFromRaster should be silenced during the test runs.

The cause the warning in test_poylgonizeMatrix_all_false should be resolved

---

### Issue #216: warp cannot deal with meta argument

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/216

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch of geokit.

- [x] I have confirmed this bug exists on the lastest commit on the dev branch of geokit.

- [ ] I have confirmed this bug exists for a specific version of geokit.


### Issue Description

When passing a meta argument of any kind to geokit.raster.warp(), the execution fails with the following error (note that the print statements were added to dev for debugging):
```
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[1], [line 8](vscode-notebook-cell:?execution_count=1&line=8)
      6 mx=raster.extractMatrix(SINGLE_HILL_PATH)
      7 rInfo = raster.rasterInfo(SINGLE_HILL_PATH)
----> [8](vscode-notebook-cell:?execution_count=1&line=8) raster.warp(
      9     source=ELEVATION_PATH,
     10     meta={'AREA_OR_POINT': 'Area'}, 
     11     bounds=

... (truncated, see issue for full details)

---

## Vector/Geometry Operations

### Issue #276: LocationSet cannot deal with integer arrays

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/276

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [ ] I have confirmed this bug exists on the lastest commit on the dev branch.

- [x] I have confirmed this bug exists for a specific version the software.


### Issue Description

LocationSet canm be initialized from integer and float lists, and from float arrays - but not from integer arrays. See reproducible example, the last step fails. Has been confirmed for 270_numpybool but can very easily reproduced with any branch via below example

### Reproducible Example

```python
import numpy as np
import geokit as gk

locsFloat = [(5.0,51.0), (7.0,52.0)]
locsInt = [(5,51), (7,52)]

# generate from float list
gk.LocationSet(locsFloat)
print("LocationSet from float list worked.")

# generate from int list
gk.LocationSet(locsInt)
print("LocationSet from int list worked.")

# generate from float array
gk.LocationSet(np.array(locsFloat))
print("LocationSet from float 

... (truncated, see issue for full details)

---

### Issue #265: extractValues Cannot Handle LocationSet as input for Points

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/265

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [x] I have confirmed this bug exists on the lastest commit on the dev branch.

- [ ] I have confirmed this bug exists for a specific version the software.


### Issue Description

The extractValues() function cannot handle a LocationSet as the input for the points argument, even though the docstring states that it can. 

Using a LocationSet as the input results in an error. 

> Index(...) must be called with a collection of some kind,  , Lon      , Lat
> 0, 6.06590  , 50.51939 
> 1, 6.02141  , 50.61491 
> 2, 6.37163  , 50.84602 
>  was passed
>   File "C:\Programming\geokit\geokit\core\raster.py", line 1127, in extractValues
>         dict(data=values, xOffset=xOffset, yOffset=yOffset, inBounds=inBounds),
>            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>     ...<2 lines>...
>     
>     
>   File "C:\Programming\geokit\test\te

... (truncated, see issue for full details)

---

### Issue #245: Regionmask.load() does not provide coordinate system to RegionMask from Vector

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/245

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [x] I have confirmed this bug exists on the lastest commit on the dev branch.

- [ ] I have confirmed this bug exists for a specific version the software.


### Issue Description

Region.load() does not provide a srs to RegionMask.fromVector()

```
if isinstance(region, RegionMask):
            return region
        elif isinstance(region, str):
            return RegionMask.fromVector(region, start_raster=start_raster, **kwargs)
        elif isinstance(region, ogr.Geometry):
            return RegionMask.fromGeom(region, start_raster=start_raster, **kwargs)
        elif isinstance(region, np.ndarray):
            return RegionMask.fromMask(region, **kwargs)
        else:
            raise GeoKitRegionMaskError("Could not understand region input")
```

This make Region.load() unsuable for a string input

### Reproducible Example

```python
Run the test test_Ex

... (truncated, see issue for full details)

---

### Issue #244: Check for file existence before reading vector file

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/244

**Description:**
### Purpose of the improvement

Non-existent files lead to unreadable error messages when trying to extractFeatures().

### Proposal

All versions of extractFeatures lead back to loadVector. This should start with a check for existence of the file, else a clear FileNotFoundError should be raised

---

### Issue #95: polygonize matrix fails, if matrix is only false

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/95

**Description:**

This issue was originally created by: @d-franzmann

---

## Data Handling

### Issue #284: How should fill values be applied

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/284

**Description:**
The fill value is the value that is for example being set to values outside of the coverage when a raster is warped to a larger geographical context, or to the "outside" area when a cutline geometry is applied which geospatially delineates the raster data of interest.

What it does, however, may surprise users: Instead of just filling the "additional" areas or those outside the cutout, it also replaces all noData values. This could be interpreted as a consequent application of the input raster as a "masking raster" but means that warping a raster with a fill value (other than the same as the noData value) leads to the removal of all noData values in the existing raster! 

See below example. The initial raster contains a noData cell (value 255)

```python
import numpy as np
import geokit as gk

arr = np.array([[50,100,150], [200,250,255]])
rstr_255noData = gk.raster.createRaster(
    data=arr,
    bounds=(0,0,3,2),
    pixelWidth=1,
    pixelHeight=1,
    noData=255,
    srs=4326,
)
gk.

... (truncated, see issue for full details)

---

### Issue #270: deprecated numpy datatypes

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/270

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [x] I have confirmed this bug exists on the lastest commit on the dev branch.

- [ ] I have confirmed this bug exists for a specific version the software.


### Issue Description

data_types.py defines np.bool as second option for inBounds, but datatype was removed in favor of bool alone, leading to this error:

> Exception has occurred: AttributeError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
module 'numpy' has no attribute 'bool'.
`np.bool` was a deprecated alias for the builtin `bool`. To avoid this error in existing code, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdo

... (truncated, see issue for full details)

---

## Code Quality & Developer Experience

### Issue #289: Enable autocompletion for geokit

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/289

**Description:**
### Purpose of the improvement

Currently, autocompletion for Geokit fails very often. An import looks like this, for example:

<img width="280" height="17" alt="Image" src="https://github.com/user-attachments/assets/03106428-9bcc-4554-9402-206441227a29" />

However, it's supposed to look like this:




### Proposal

It requires three steps to make it work.

1. Use the modern type checker and Python Language Server 'ty' from the Rust creators: https://marketplace.visualstudio.com/items?itemName=astral-sh.ty.
2. Use direct imports.

For example, don't import extractFeatures.

like this:

```
import geokit as gk

gk.extractFeatures
```

Import it directly from the location of the method, for example:

```
import geokit.core.vector

geokit.core.vector.extractFeatures
```

or 

```
import geokit.core.vector

geokit.core.vector.extractFeatures
```
3. In environments where Geokit has only been added as an editable installation, switch the build backend from Setuptools to Hatchling to make th

... (truncated, see issue for full details)

---

### Issue #271: typeguard

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/271

**Description:**
### Version Checks (indicate both or one)

- [ ] I have confirmed this bug exists on the lastest commit on the master branch

- [x] I have confirmed this bug exists on the lastest commit on the dev branch.

- [x] I have confirmed this bug exists for a specific version the software.


### Issue Description

3 Tests (test_extractValues, test_extractValues_location and test_interpolateValues) all related to extractValues() fail due to a typeguard check, both in the 270_numpybool branch as well as on the latest dev. FYI @julian-belina 

### Reproducible Example

```python
See pipeline result here: https://github.com/FZJ-IEK3-VSA/geokit/actions/runs/19745610642/job/56579165962
```

### Expected Behavior

Tests should pass

### pip list -v

```bash
see pipeline setup
```

---

### Issue #269: Enable "ARG" in Ruff to check for unused arguments

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/269

**Description:**
### Purpose of the improvement

There are currently many unused keyworded arguments that can cause unintended behavior.

### Proposal

Enable the "ARG" linter in Ruff to check for unused arguments. Then, pass them on or remove them from the respective function.

---

## Bug Fixes & Stability

### Issue #254: Add more retry attempts for downloads in Example 7

**Status:** closed ✓

**Link:** https://github.com/FZJ-IEK3-VSA/geokit/issues/254

**Description:**
### Purpose of the improvement

Example 7 sometimes fails due to download issues with Zenodo. Zenodo returns an empty string, resulting in the following error message:
> raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)
> 
> JSONDecodeError: Expecting value: line 1 column 1 (char 0)

The error is usually resolved by simply rerunning the jobs.

### Proposal

The number of download retry attempts should be increased by setting 

`retry_if_failed = 5`

in the pooch.create() method.

---

