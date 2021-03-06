""" Module for extracting population estimates from Bangladesh data"""
import os
import sys
import code
from tqdm import tqdm
from datetime import datetime
import dateutil.parser
from osgeo import ogr
from osgeo import osr
from shapely.geometry import Polygon

def get_pop_polygons(shapefile_dir):

    population_polygons = {}
    all_paths = []
    for root, dirs, files in os.walk(shapefile_dir):
        all_paths += [os.path.join(root, file) for file in files if file.lower().endswith('.shp')]

    for path in tqdm(all_paths):
        file = os.path.basename(path)
        # Extract the associated date from the end of the filename
        curr_date = dateutil.parser.parse(os.path.splitext(file)[0][-8:])

        sf = ogr.Open(path)

        layer = sf.GetLayer()
        # get projected spatial reference
        sr = layer.GetSpatialRef()
        # get geographic spatial reference
        geogr_sr = sr.CloneGeogCS()
        # define reprojection
        proj_to_geog = osr.CoordinateTransformation(sr, geogr_sr)
        curr_date_entries = []
        for area in layer:
            area_shape = area.GetGeometryRef()
            area_polygon = area_shape.GetGeometryRef(0)
            # transform coordinates
            area_polygon.Transform(proj_to_geog)
            no_of_polygon_vertices = area_polygon.GetPointCount()
            polygon_points = []
            for vertex in range(no_of_polygon_vertices):
                lon, lat, z = area_polygon.GetPoint(vertex)
                polygon_points.append((lat, lon))
            curr_poly = Polygon(polygon_points)
            curr_date_entries.append((curr_poly, area))

        population_polygons[curr_date] = curr_date_entries

    return population_polygons

def population_lookup(bounding_polygon, reference_date, population_polygons):
    """
    Compute a population estimate by overlapping all population polygons
    with the provided polygon and multiply by an area-based population
    factor. Use the provided reference date to pull the right set of
    shapefiles.
    """
    all_dates = list(population_polygons.keys())
    date_distances = [abs(date - reference_date) for date in all_dates]
    min_date_distance = min(date_distances)
    min_index = date_distances.index(min_date_distance)

    population_polygons = population_polygons[all_dates[min_index]]

    if not bounding_polygon.is_valid:
        bounding_polygon = bounding_polygon.buffer(0)

    total_population = 0
    total_area = 0
    for polygon, record in population_polygons:
        try:
            intersection_polygon = polygon.intersection(bounding_polygon)
            if not intersection_polygon.is_empty:
                population_multiplier = intersection_polygon.area/polygon.area
                total_population += (population_multiplier*get_population_from_record(record))
                total_area += intersection_polygon.area/polygon.area
        except:
            pass

    return total_population, total_area


def get_population_from_record(record):
    """ The shapefile population records aren't consistent, so we use this
    method to search for and return the population entry. """
    record_keys = record.keys()
    possible_keys = ['Total_Pop', 'Population', 'Total_INDs']
    for key in possible_keys:
        if key in record_keys:
            return record[key]

    raise RuntimeError('Could not find valid population key! Possible keys are: {}'.format(record_keys))
