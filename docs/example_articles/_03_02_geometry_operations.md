# Geometry Operations

You can apply various operations on geometric objects:

## 1. [Centroid Extraction](../Examples/_03_geometries/_02_geometry_operations/_01_get_centroid_of_geom.ipynb)
- Retrieve centroids from any geometry type (lines, polygons, etc.)

## 2.  [Boundary Extraction](../Examples/_03_geometries/_02_geometry_operations/_02_get_boundary.ipynb)
- Extract the boundary of geometries using `Boundary()`

## 3.  [Spatial Reference System Transformation](../Examples/_03_geometries/_02_geometry_operations/_03_transform_srs.ipynb)
- Transform geometries between coordinate systems using `transform()`
- Extract vertices from transformed geometries via `extractVerticies()`
- Handle antimeridian edge cases with `revert360degProj` flag
- Clip or shift out-of-bounds geometries with `fixOutOfBoundsGeoms()`

## 4. [Area Calculation](../Examples/_03_geometries/_02_geometry_operations/_05_get_areas.ipynb)
- Calculate geometry areas using `Area()` method

## 5. [Spatial Relations](../Examples/_03_geometries/_02_geometry_operations/_06_spatial_relations_of_geometries.ipynb)
- Test geometric relationships such as: Contains, Crosses, Touches, Overlaps
- Evaluate spatial predicates between geometry pairs

## 6. [Set Operations](../Examples/_03_geometries/_02_geometry_operations/_07_intersection_and_union.ipynb)
- Compute geometry intersections using `Intersection()`
- Compute geometry unions using `Union()`
- Ensure operations occur within the same spatial reference system