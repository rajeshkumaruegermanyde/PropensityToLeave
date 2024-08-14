# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 09:42:03 2024

@author: RajeshKumar
"""

# employee dataset quality report using pandas

import pandas as pd
import numpy as np


data=pd.read_csv('data/employee_hr_data.csv')

data_types = pd.DataFrame(
    data.dtypes,
    columns=['Data Type']
)

data_counts = pd.DataFrame(
    data.count(),
    columns=['Data Count']
)


missing_data = pd.DataFrame(
    data.isnull().sum(),
    columns=['Missing Values']
)
unique_values = pd.DataFrame(
    columns=['Unique Values']
)
for row in list(data.columns.values):
    unique_values.loc[row] = [data[row].nunique()]
    

min_values = pd.DataFrame(
    columns=['Minimum Value']
)
for row in list(data.columns.values):
    min_values.loc[row] = [data[row].min()]



maximum_values = pd.DataFrame(
    columns=['Maximum Value']
)
for row in list(data.columns.values):
    maximum_values.loc[row] = [data[row].max()]
    
    
dq_report = data_types.join(data_counts).join(missing_data).join(unique_values).join(min_values).join(maximum_values)
dq_report.to_csv('data/dqr.csv')

    
    
    