# GeoKit Geometry Capabilities Summary

GeoKit provides comprehensive geometry creation and operations capabilities, demonstrated through notebooks organized into two categories:

## **Geometry Creation**

### 1. Points ([_01_create_points.ipynb](../../Examples/_03_geometries/_01_create_geometries/_01_create_points.ipynb))
- Create 2D points from coordinate tuples or separate x, y arguments
- Supports multiple spatial reference systems
- Visualize points with `drawGeoms()`
- Export points to shapefile format

### 2. LineStrings ([_02_create_lines.ipynb](../../Examples/_03_geometries/_01_create_geometries/_02_create_lines.ipynb))
- Create linestring geometries from ordered coordinate lists
- Supports visualization and export to vector files

### 3. MultiLineStrings ([_03_create_multi_lines.ipynb](../../Examples/_03_geometries/_01_create_geometries/_03_create_multi_lines.ipynb))
- Combine multiple linestrings into a single multilinestring geometry using `flatten()`
- Visualize and export multilinestring collections

### 4. Polygons ([_04_create_polygon.ipynb](../../Examples/_03_geometries/_01_create_geometries/_04_create_polygon.ipynb))
- Create polygon geometries from coordinate lists
- Export polygon geometries to shapefile format

### 5. MultiPolygons ([_05_create_multi_polygon.ipynb](../../Examples/_03_geometries/_01_create_geometries/_05_create_multi_polygon.ipynb))
- Combine multiple polygons into multipolygon geometries
- Use `flatten()` to merge polygon collections

## **Geometry Operations**

### 1. Centroid Extraction ([_01_get_centroid_of_geom.ipynb](../../Examples/_03_geometries/_02_geometry_operations/_01_get_centroid_of_geom.ipynb))
- Retrieve centroids from any geometry type (lines, polygons, etc.)
- Maintain spatial reference system through `AssignSpatialReference()`

### 2. Boundary Operations ([_02_get_boundary.ipynb](../../Examples/_03_geometries/_02_geometry_operations/_02_get_boundary.ipynb))
- Extract the boundary of geometries using `Boundary()`
- Convert polygons to linestring boundaries

### 3. Spatial Reference System Transformation ([_03_transform_srs.ipynb](../../Examples/_03_geometries/_02_geometry_operations/_03_transform_srs.ipynb))
- Transform geometries between coordinate systems using `transform()`
- Extract vertices from transformed geometries via `extractVerticies()`
- Handle antimeridian edge cases with `revert360degProj` flag
- Clip or shift out-of-bounds geometries with `fixOutOfBoundsGeoms()`

### 4. Area Calculation ([_05_get_areas.ipynb](../../Examples/_03_geometries/_02_geometry_operations/_05_get_areas.ipynb))
- Calculate geometry areas using `Area()` method
- Support for different coordinate systems and projections

### 5. Spatial Relations ([_06_spatial_relations_of_geometries.ipynb](../../Examples/_03_geometries/_02_geometry_operations/_06_spatial_relations_of_geometries.ipynb))
- Test geometric relationships: `Contains()`, `Crosses()`, `Touches()`, `Overlaps()`
- Evaluate spatial predicates between geometry pairs

### 6. Set Operations ([_07_intersection_and_union.ipynb.ipynb](../../Examples/_03_geometries/_02_geometry_operations/_07_intersection_and_.ipynb))
- Compute geometry intersections using `Intersection()`
- Compute geometry unions using `Union()`
- Ensure operations occur within the same spatial reference system
