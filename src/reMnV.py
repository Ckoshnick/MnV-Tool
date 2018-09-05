# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 08:44:44 2018

@author: koshnick
"""

import matplotlib.pyplot as plt
import mnv14 as mnv
import pandas as pd
import json



# Load stuff and rerun prediction - save results

# Load JSON
# Load Raw Data
# Append Data to Raw Data from PI

# Pipe Data and JSON to DK
# - Plots will say if modifications need making
# - Post period will need to be flexible / smart?

# Pipe DK and JSON to MC
# Run model and show plots
# Save these results somehow? - WIth create archive? Is chaining bad?

# How will this work with central archiving? Save twice?

def read_json_params():
    # Ding Json params locally, split into two dicts
    # return two objects

    jsonName = 'all Params.json' # TODO: Move this to a class varaible
    with open(jsonName) as f:
        data = json.load(f)
    print(data)

    return data


def read_raw_data():
    # find raw data locally, load into pd.dataframe
    # return data
    pass

def extend_data_with_pi(data, endDate='y', tags=[]):

    # Find last date in data, make that start date for datapull
    # Default end date to yesterday
    # Use column headings as tags
    # if custom tags, then pull those and replace columns??? (tricky)
    # Pull data, concat and return
    # Save lengthened data (??)
    # return last date and data
    pass

def DK_II(data, dataParams):
    # make DK
    # Default Clean
    # Defualt plots
    # return DK
    pass

def MC_II(pre, post, modelParams):
    # Make Mod
    # Default plots
    pass

def save_results():
    # ??
    # Return
    pass


A = read_json_params()