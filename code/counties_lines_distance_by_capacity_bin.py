#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 08:35:24 2025

@author: sr2626
"""

#%% install packages

import subprocess
import sys
import os
import warnings

# suppress warnings
warnings.filterwarnings('ignore')

# check system environment for packages and install if necessary
# package versions that were used at time of writing are commented above

# Pandas 2.0.0
try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas'])
finally:
    import pandas as pd

# Numpy 1.24.2
try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'numpy'])
finally:
    import numpy as np

# GeoPandas 0.12.2
try:
    import geopandas as gpd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'geopandas'])
finally:
    import geopandas as gpd

# Matplotlib 3.7.1
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])
finally:
    import matplotlib.pyplot as plt

# Shapely 2.0.1
try:
    from shapely.geometry import Point, LineString, MultiLineString, Polygon
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'shapely'])
finally:
    from shapely.geometry import Point, LineString, MultiLineString, Polygon
    
# GeoPy 2.3.0
try:
    from geopy.distance import distance, geodesic
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'geopy'])
finally:
    from geopy.distance import distance, geodesic

#%% user parameters

# 1. root directory
directory_path = os.path.realpath(__file__)[:-47]
os.chdir(directory_path)

# 2. data path
data_path = os.path.join(directory_path, "data")

# 3. Image output path
output_path = os.path.join(directory_path, "figures")


#%% load shapefile

counties_path = os.path.join(directory_path, "shapefile", "cb_2023_us_county_500k.shp")

counties_shp = gpd.read_file(counties_path)
#%% load data
stations_path = os.path.join(data_path, "grid_input", "vertices.csv")
df_stations = pd.read_csv(stations_path)

lines_path = os.path.join(data_path, "grid_input", "links_cap.csv")
df_lines = pd.read_csv(lines_path)

intersections_path = os.path.join(data_path, "grid_input", "intersections.csv")
df_intersections = pd.read_csv(intersections_path)

#%% process line data
# keep only intersections that touch two stations
df_intersections = df_intersections.drop_duplicates(subset=['line_id', 'station_id'], keep='first')
df_intersections = df_intersections[df_intersections['line_id'].duplicated(keep=False)]

# keep only lines that touch stations on both ends
df_s2slines = df_lines.merge(df_intersections, 
                             on=['line_id'], 
                             how='left', 
                             indicator=True)

df_s2slines = df_s2slines[df_s2slines['_merge'] == 'both']
df_s2slines = df_s2slines.drop_duplicates('line_id')
df_s2slines = df_s2slines[['line_id', 'longitude_1', 'latitude_1', 'longitude_2', 'latitude_2', 'voltage_kv', 'max_capacity_mw']]

# turn endpoints into point geometries
df_s2slines['point_A'] = df_s2slines.apply(lambda x: Point(x['longitude_1'], x['latitude_1']), axis = 1)
df_s2slines['point_B'] = df_s2slines.apply(lambda x: Point(x['longitude_2'], x['latitude_2']), axis = 1)
df_s2slines['line'] = df_s2slines.apply(lambda x: LineString([x['point_A'], x['point_B']]), axis = 1)


#%% find edges between counties

# make the line dataframe a geodataframe and change the CRS
df_s2slines = gpd.GeoDataFrame(df_s2slines, geometry = 'point_A')
df_s2slines = df_s2slines.set_crs("EPSG:4326")
counties_shp = counties_shp.to_crs("EPSG:4326")

# calculate which county intersects point_A in a given line
edges = gpd.sjoin(df_s2slines, counties_shp, how='left', predicate='intersects')
edges = edges[['line_id', 'point_A', 'point_B', 'line', 'GEOID','NAME', 'voltage_kv', 'max_capacity_mw']]

# set point_B to the geometry and calculate which commuter zone intersects point_B
edges = gpd.GeoDataFrame(edges, geometry = 'point_B')
edges = edges.set_crs("EPSG:4326")
edges = gpd.sjoin(edges, counties_shp, how='left', predicate='intersects')


# clean up the edges table 
edges = edges[['line_id', 'point_A', 'point_B', 'line','GEOID_left', 'NAME_left','GEOID_right', 'NAME_right', 'voltage_kv', 'max_capacity_mw']]
edges = edges.rename(columns={ 'GEOID_left': 'GEOID_A','NAME_left':'NAME_A', 'GEOID_right': 'GEOID_B','NAME_right':'NAME_B'})

edges_us = edges.dropna(subset=['GEOID_A','GEOID_B'])

#%% calculate the line distances

edges_us['latA'] = edges_us['point_A'].apply(lambda x: x.y)  # Latitude of point_A
edges_us['lonA'] = edges_us['point_A'].apply(lambda x: x.x)  # Longitude of point_A
edges_us['latB'] = edges_us['point_B'].apply(lambda x: x.y)  # Latitude of point_B
edges_us['lonB'] = edges_us['point_B'].apply(lambda x: x.x)  # Longitude of point_B

# Create tuples for distance calculation
edges_us['tupleA'] = list(zip(edges_us['latA'], edges_us['lonA']))
edges_us['tupleB'] = list(zip(edges_us['latB'], edges_us['lonB']))

# Calculate distance between point_A and point_B in kilometers
edges_us['distance_km'] = edges_us[['tupleA', 'tupleB']].apply(
    lambda x: distance(x['tupleA'], x['tupleB']).km, axis=1
)

# Create a new column 'same_county' that checks if GEOID_A is equal to GEOID_B
edges_us['same_county'] = edges_us['GEOID_A'] == edges_us['GEOID_B']

#%% Split lines that cross multiple counties
# Filter lines that stay within the same county
edges_within_county = edges_us[edges_us['same_county'] == True]

# Filter lines that cross county boundaries
edges_crossing_county = edges_us[edges_us['same_county'] == False]
edges_crossing_county = gpd.GeoDataFrame(edges_crossing_county,geometry='line',crs='EPSG:4326')

# Split at county boundaries
edges_split = gpd.overlay(edges_crossing_county, counties_shp, how='intersection')

# Function to calculate geodesic length of a LineString or MultiLineString
def geodesic_length(geometry):
    total_distance = 0
    
    # If it's a LineString, calculate distance between consecutive points
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)  # Extract coordinates from LineString
        for i in range(len(coords) - 1):
            total_distance += geodesic((coords[i][1], coords[i][0]), (coords[i+1][1], coords[i+1][0])).km
    
    # If it's a MultiLineString, iterate over each LineString part and calculate distance
    elif isinstance(geometry, MultiLineString):
        for line in geometry.geoms:  # Access each individual LineString
            coords = list(line.coords)  # Extract coordinates from each LineString
            for i in range(len(coords) - 1):
                total_distance += geodesic((coords[i][1], coords[i][0]), (coords[i+1][1], coords[i+1][0])).km
    
    return total_distance

# Calculate the length for split edges
edges_split['length_km'] = edges_split['geometry'].apply(geodesic_length)

# Keep specific columns
edges_split_final = edges_split[['line_id','GEOID', 'NAME', 'voltage_kv', 'max_capacity_mw','length_km','same_county']]
edges_split_final = edges_split_final.rename(columns={ 'length_km': 'distance_km'})
edges_within_county_final = edges_within_county[['line_id','GEOID_A', 'NAME_A', 'voltage_kv', 'max_capacity_mw','distance_km','same_county']]
edges_within_county_final = edges_within_county_final.rename(columns={ 'GEOID_A': 'GEOID','NAME_A':'NAME'})

# Merge the two datasets
final_edges = pd.concat([edges_within_county_final, edges_split_final], ignore_index=True)

#%% Create capacity bins
# cap capacities at 5000 for bin construction
final_edges['max_capacity_mw'].where(final_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# Define the bins and labels
bins = [0, 1000, 2000, 3000, 4000, 5000]
labels = [1, 2, 3, 4, 5]

# Create a new column with the index based on max_capacity_mv
final_edges['capacity_index'] = pd.cut(final_edges['max_capacity_mw'], bins=bins, labels=labels, right=False)

# Group by capacity index and calculate stats
capacity_stats = final_edges.groupby('capacity_index')['max_capacity_mw'].agg(['count', 'mean', 'min', 'max', 'std']).reset_index()

#%% Calculate total distance by county for each capacity bin and the number of lines by county

final_edges['within_county'] = final_edges['same_county'].astype(int)  # True becomes 1, False becomes 0
final_edges['cross_county'] = (~final_edges['same_county']).astype(int)  # False becomes 1, True becomes 0

# Group by GEOID and capacity_index, then sum the distances
county_distance_stats = (
    final_edges
    .groupby(['GEOID', 'capacity_index'])['distance_km']
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)

# Group by GEOID  and sum the 'within_county' and 'cross_country' columns for each county
county_count_lines_stats = (
    final_edges
    .groupby(['GEOID'])
    .agg({
        'within_county': 'sum',  # Total within-county lines
        'cross_county': 'sum'   # Total cross-country lines
    })
    .reset_index()
)

# Merge both DataFrames on 'GEOID'
county_stats = pd.merge(county_distance_stats, county_count_lines_stats, on='GEOID', how='left')


# Create a mapping of GEOID to NAME using the counties_shp DataFrame
geoid_to_name = counties_shp.set_index('GEOID')[['NAME']].to_dict(orient='index')

# Apply the mapping to get the NAME based on the GEOID
county_stats['NAME'] = county_stats['GEOID'].apply(lambda x: geoid_to_name.get(x, {}).get('NAME'))

# Rearrange the columns to put 'NAME' first
county_stats = county_stats[['NAME', 'GEOID','within_county','cross_county', 1, 2, 3, 4, 5]]

# Rename the columns 
county_stats.columns = ['county_name', 'county_geo_id', 
                        'count_lines_within_county','count_lines_cross_county',
                          'distance_km_max_cap_mw_1000', 
                          'distance_km_max_cap_mw_2000', 
                          'distance_km_max_cap_mw_3000', 
                          'distance_km_max_cap_mw_4000', 
                          'distance_km_max_cap_mw_5000']

# Ensure all counties from counties_shp are in the county_stats table
full_county_stats = pd.merge(counties_shp[['GEOID', 'NAME']], county_stats, 
                              left_on='GEOID', right_on='county_geo_id', 
                              how='left')

# Fill NaN values with 0 for counties with no data
full_county_stats.fillna(0, inplace=True)

# Now, full_county_stats contains all counties, with 0s for those with no power line data
full_county_stats = full_county_stats[['NAME', 'GEOID', 
                                       'count_lines_within_county', 'count_lines_cross_county',
                                       'distance_km_max_cap_mw_1000', 'distance_km_max_cap_mw_2000',
                                       'distance_km_max_cap_mw_3000', 'distance_km_max_cap_mw_4000',
                                       'distance_km_max_cap_mw_5000']]

# Rename columns if necessary
full_county_stats.columns = ['county_name', 'county_geo_id', 
                              'lines_count_within_county', 'lines_count_cross_county',
                              'lines_distance_km_max_cap_mw_1000', 'lines_distance_km_max_cap_mw_2000',
                              'lines_distance_km_max_cap_mw_3000', 'lines_distance_km_max_cap_mw_4000', 
                              'lines_distance_km_max_cap_mw_5000']

# Save the results
full_county_stats.to_excel(os.path.join(data_path, "counties_output","counties_lines_distance_by_capacity_bin.xlsx"),index=False)








