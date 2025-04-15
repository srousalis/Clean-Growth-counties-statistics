# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 15:33:24 2025

@author: srous
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
directory_path = os.path.realpath(__file__)[:-41]
os.chdir(directory_path)

#%% load shapefile

counties_path = os.path.join(directory_path, "shapefile", "cb_2023_us_county_500k.shp")
counties_shp = gpd.read_file(counties_path)

#%% load data
df_gpp = pd.read_csv(os.path.join(directory_path, "input","gpp_bleed.csv"))

# sort types of energy sources 
df_gpp_renw = df_gpp[(df_gpp['primary_fuel'] == 'Solar') |
                     (df_gpp['primary_fuel'] == 'Wind') |
                     (df_gpp['primary_fuel'] == 'Biomass') |
                     (df_gpp['primary_fuel'] == 'Waste') |
                     (df_gpp['primary_fuel'] == 'Geothermal')]
df_gpp_solr = df_gpp[df_gpp['primary_fuel'] == 'Solar']
df_gpp_wind = df_gpp[df_gpp['primary_fuel'] == 'Wind']
df_gpp_fossil = df_gpp[(df_gpp['primary_fuel'] == 'Gas') |
                      (df_gpp['primary_fuel'] == 'Oil') |
                      (df_gpp['primary_fuel'] == 'Coal') |
                      (df_gpp['primary_fuel'] == 'Hydro') ]

#%%
# Define fuel type groups
fuel_categories = {
    'total': {
        'fuels': None  # include all fuels
    },
    'renewable': {
        'fuels': ['Solar', 'Wind', 'Biomass', 'Waste', 'Geothermal']
    },
    'solar': {
        'fuels': ['Solar']
    },
    'wind': {
        'fuels': ['Wind']
    },
    'fossil': {
        'fuels': ['Gas', 'Oil', 'Coal', 'Hydro']  # Hydro treated as fossil per instructions
    }
}

# Flatten all fuel types defined in categories (excluding 'total')
defined_fuels = set()
for key, value in fuel_categories.items():
    if value['fuels'] is not None:
        defined_fuels.update(value['fuels'])

# Get unique fuels in dataset
all_fuels = set(df_gpp['primary_fuel'].unique())

# Identify others
others_fuels = list(all_fuels - defined_fuels)

# Add to categories
fuel_categories['others'] = {
    'fuels': others_fuels
}

# Loop through fuel categories and process
agg_results = {}

for category, details in fuel_categories.items():
    fuels = details['fuels']
    
    # Filter df_gpp based on fuel types if specified
    if fuels:
        df_filtered = df_gpp[df_gpp['primary_fuel'].isin(fuels)].copy()
    else:
        df_filtered = df_gpp.copy()
    
    # Create GeoDataFrame
    df_filtered['coordinates'] = [Point(xy) for xy in zip(df_filtered['longitude'], df_filtered['latitude'])]
    df_filtered = gpd.GeoDataFrame(df_filtered, geometry='coordinates')
    df_filtered = df_filtered.set_crs("EPSG:4326")
    
    # Spatial join with counties shapefile
    df_joined = gpd.sjoin(df_filtered, counties_shp, how='left', predicate='intersects')
    df_joined = df_joined[df_joined['GEOID'].notna()]
    
    # Aggregate capacity and plant counts
    df_agg = df_joined.groupby(['GEOID', 'NAME']).agg(
        capacity_mw=('capacity_mw', 'sum'),
        plant_count=('capacity_mw', 'count')
    ).reset_index()
    
    # Store results in separate variables using globals()
    globals()[f'df_joined_{category}'] = df_joined
    globals()[f'df_agg_{category}'] = df_agg

    # Print summary
    print(f"Category: {category.upper()}")
    print("  Total plants in USA:", df_filtered[df_filtered['country'] == 'USA'].shape[0])
    print("  Matched plants in USA:", df_joined.shape[0])
    print("  Aggregated plant count:", df_agg['plant_count'].sum())
    print("-" * 50)

# Initialize a new DataFrame for the result
full_county_stats = counties_shp[['NAME', 'GEOID']].copy()

# Loop through each aggregation category and merge the data
for category, details in fuel_categories.items():
    # Get the aggregation result for the category (e.g., df_agg_solar, df_agg_fossil)
    df_agg_category = globals().get(f'df_agg_{category}')
    
    # Merge aggregated data with the counties DataFrame on both 'NAME' and 'GEOID'
    if df_agg_category is not None:
        # Merge on NAME and GEOID, using a left join to keep all counties
        merged = full_county_stats.merge(df_agg_category[['NAME', 'GEOID', 'capacity_mw', 'plant_count']], 
                                 on=['NAME', 'GEOID'], how='left')
        
        # Rename columns to reflect the category (e.g., capacity_mw_solar, plant_count_solar)
        merged.rename(columns={
            'capacity_mw': f'capacity_mw_{category}',
            'plant_count': f'plant_count_{category}'
        }, inplace=True)
        
        # Update result_df to include the new merged columns
        full_county_stats = merged
    else:
        # If no data for this category, just add NaN columns
        full_county_stats[f'capacity_mw_{category}'] = np.nan
        full_county_stats[f'plant_count_{category}'] = np.nan

# Ensure missing values are filled with NaN (already done by merge with 'how=left')
full_county_stats.fillna(0, inplace=True)

full_county_stats.rename(columns={
    'NAME': 'county_name',
    'GEOID': 'county_geo_id'
}, inplace=True)
full_county_stats.to_excel(os.path.join(directory_path, "output","counties_plants_capacity_by_type.xlsx"),index=False)

full_county_stats.columns = ['county_name', 'county_geo_id', 
                               'plant_capacity_mw_total','plant_count_total',
                               'plant_capacity_mw_renewable', 'plant_count_renewable',
                               'plant_capacity_mw_solar', 'plant_count_solar', 
                               'plant_capacity_mw_wind', 'plant_count_wind',
                               'plant_capacity_mw_fossil', 'plant_count_fossil',
                               'plant_capacity_mw_other','plant_count_other']

#%%
lines_capacity_stats = pd.read_excel(os.path.join(directory_path, "output","counties_lines_distance_by_capacity_bin.xlsx"))


statistics_counties = lines_capacity_stats.copy()

columns_to_copy = ['plant_capacity_mw_total','plant_count_total',
                   'plant_capacity_mw_renewable', 'plant_count_renewable',
                   'plant_capacity_mw_solar', 'plant_count_solar', 
                   'plant_capacity_mw_wind', 'plant_count_wind',
                    'plant_capacity_mw_fossil', 'plant_count_fossil',
                    'plant_capacity_mw_other','plant_count_other']

for column in columns_to_copy:
    statistics_counties[column] = full_county_stats[column]


statistics_counties.to_excel(os.path.join(directory_path, "output","counties_lines_plants_statistics.xlsx"),index=False)












