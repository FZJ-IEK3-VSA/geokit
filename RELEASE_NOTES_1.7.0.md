# GeoKit Release 1.7.0

**Release Date:** February 2026

We are pleased to announce the release of GeoKit 1.7.0! This release includes 22 resolved issues focused on improving plotting capabilities, raster operations, data handling, and overall code quality.

## 🎨 Plotting & Visualization Enhancements

This release brings significant improvements to GeoKit's plotting and visualization capabilities:

- **#294**: allow mixed NaN and value geometry plots
- **#293**: draw_cbar arg not backward compatible
- **#278**: gk.drawImage does not work with ax input
- **#264**: Fix GeoKit Plotting Functions
- **#29**: allow colorBy str values when drawing geoms

## 🗺️ Raster Operations Improvements

Enhanced raster processing capabilities with better handling of edge cases:

- **#286**: Remove ComputeStatistics in rasterInfo where possible
- **#285**: deal with rasters without srs in warp
- **#274**: raster.interpolateValues() cannot deal with point lists of len = 1 anymore
- **#268**: warp() assigns wrong datatype depending on input for fill and noData
- **#261**: test_contours, test_Extent_contoursFromRaster and test_polygonize_matrix_all_false prompt warnings
- **#216**: warp cannot deal with meta argument

## 📐 Vector & Geometry Operations

Improvements to vector and geometry handling:

- **#276**: LocationSet cannot deal with integer arrays
- **#265**: extractValues Cannot Handle LocationSet as input for Points
- **#245**: Regionmask.load() does not provide coordinate system to RegionMask from Vector
- **#244**: Check for file existence before reading vector file
- **#95**: polygonize matrix fails, if matrix is only false

## 💾 Data Handling & Type Support

Better data handling and type compatibility:

- **#284**: How should fill values be applied
- **#270**: deprecated numpy datatypes

## 🔧 Code Quality & Developer Experience

Improvements to code quality, testing, and developer experience:

- **#289**: Enable autocompletion for geokit
- **#271**: typeguard
- **#269**: Enable "ARG" in Ruff to check for unused arguments

## 🐛 Bug Fixes & Stability

Additional bug fixes and stability improvements:

- **#254**: Add more retry attempts for downloads in Example 7

## 📊 Release Statistics

- **Total Issues Resolved:** 22
- **Issue Status:** All closed ✓
- **Categories:** 6 major improvement areas

## 🙏 Acknowledgments

Thank you to all contributors who helped make this release possible!

## 📚 Documentation

For more information about using GeoKit, please refer to our documentation at https://fzj-iek3-vsa.github.io/geokit/

## 🔗 Links

- **GitHub Repository:** https://github.com/FZJ-IEK3-VSA/geokit
- **Milestone:** [Release 1.7.0](https://github.com/FZJ-IEK3-VSA/geokit/milestone/1)
- **Full Changelog:** See all [closed issues in this milestone](https://github.com/FZJ-IEK3-VSA/geokit/issues?q=milestone%3A%22Release+1.7.0%22+is%3Aclosed)

---

*GeoKit is developed and maintained by Forschungszentrum Jülich - IEK-3*
