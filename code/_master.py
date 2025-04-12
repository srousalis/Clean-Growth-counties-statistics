# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 18:23:39 2025

@author: srous
"""

#%% user parameters
import sys
import os
#%% run scripts

# root directory (changes the directory automatically in submodules, no need to change directories in the called scripts)
master_directory_path = os.path.realpath(__file__)[:-15]

# code directory
code_path = os.path.join(master_directory_path, "code")

# include code folder as part of module search
sys.path.append(code_path)

#%%


print("Running scripts.")
        
import counties_lines_distance_by_capacity_bin
print("STEP 1: Lines Statistics finished.")

import counties_plants_capacity_by_type
print("STEP 2: Plants Statistics finished.")
        
print("Run completed.")




















