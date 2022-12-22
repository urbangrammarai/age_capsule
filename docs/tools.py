import cuspatial
import geopandas
import cupy
import warnings

def sjoin_gpu(
    pts_gdf, 
    poly_gdf, 
    scale=5, 
    max_depth=7, 
    max_size=125,
    pts_cols=['LMK_KEY', 'UPRN', 'CONSTRUCTION_AGE_BAND'],
    poly_cols=['id', 'type']
):
    '''
    Spatial Join on a GPU
    ...
    
    Adapted from:
    
    > https://docs.rapids.ai/api/cuspatial/stable/user_guide/users.html#cuspatial.quadtree_point_in_polygon
    
    Arguments
    ---------
    pts_gdf : geopandas.GeoDataFrame/cuspatial.GeoDataFrame
              Table with points
    poly_gdf : geopandas.GeoDataFrame/cuspatial.GeoDataFrame
               Table with polygons
    scale : int
            [From `cuspatial` docs. Default=5] A scaling function that increases the size of the point 
            space from an origin defined by `{x_min, y_min}`. This can increase the likelihood of 
            generating well-separated quads.
            
    max_depth : int
                [From `cuspatial` docs. Default=7] In order for a quadtree to index points effectively, 
                it must have a depth that is log-scaled with the size of the number of points. Each level 
                of the quad tree contains 4 quads. The number of available quads $q$
                for indexing is then equal to $q = 4^d$ where $d$ is the max_depth parameter. With an input 
                size of 10m points and `max_depth` = 7, points will be most efficiently packed into the leaves
                of the quad tree.
    max_size : int
               [From `cuspatial` docs. Default=125] Maximum number of points allowed in a node before it's 
               split into 4 leaf nodes. 
    pts_cols : list
               [Optional. Default=['UPRN', 'CONSTRUCTION_AGE_BAND']] Column names in `pts_gdf` to be 
               joined in the output
    poly_cols : list
                [Optional. Default=['id', 'type']] Column names in `poly_gdf` to be joined in the output 
    
    Returns
    -------
    sjoined : cudf.DataFrame
              Table with `pts_cols` and `poly_cols` spatially joined
    '''
    pts_gpu = gpufy_table(pts_gdf)
    poly_gpu = gpufy_table(poly_gdf)
    
    x_points = pts_gpu['geometry'].points.x
    y_points = pts_gpu['geometry'].points.y
    polygons = poly_gpu['geometry'].polygons

    poly_bboxes = cuspatial.polygon_bounding_boxes(
        polygons.part_offset,
        polygons.ring_offset,
        polygons.x,
        polygons.y
    )
    point_indices, quadtree = cuspatial.quadtree_on_points(
        x_points,
        y_points,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        scale,
        max_depth,
        max_size
    )
    intersections = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree,
        poly_bboxes,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        scale,
        max_depth
    )
    polygons_and_points = cuspatial.quadtree_point_in_polygon(
        intersections,
        quadtree,
        point_indices,
        x_points,
        y_points,
        polygons.part_offset,
        polygons.ring_offset,
        polygons.x,
        polygons.y
    )
    sjoined = (# Join point table columns
        polygons_and_points
        .assign(
            remapped_point_index=polygons_and_points['point_index'].map(point_indices)
        )
        .set_index('remapped_point_index')
        .drop('point_index', axis=1)
        .join(
            pts_gpu
            .assign(pts_id=cupy.arange(len(pts_gpu)))
            .set_index('pts_id')
            [pts_cols]
        )
         # Join polygon table columns
        .set_index('polygon_index')
        .join(
            poly_gpu
            .assign(poly_id=cupy.arange(len(poly_gpu)))
            .set_index('poly_id')
            [poly_cols]
        )
    )
    return sjoined

def gpufy_table(tab):
    if type(tab) == geopandas.geodataframe.GeoDataFrame:
        warnings.warn("Copying table to the GPU...")
        return cuspatial.from_geopandas(tab)
    else:
        return tab
    