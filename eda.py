# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 09:42:03 2024

@author: RajeshKumar
"""

# employee attrition analysis and retention

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


data=pd.read_csv('data/employee_hr_data.csv')

# show top 5 rows
print(data.head())
# data shape
print(f"Data shape: {data.shape}")

# data info
print(data.info())

# checking nulls

print(data.isnull().sum())


# data heatmaps
cols=['YearsWithCurrManager','YearsInCurrentRole','YearsAtCompany','WorkLifeBalance','TotalWorkingYears','StockOptionLevel','RelationshipSatisfaction','PerformanceRating',
'OverTime','NumCompaniesWorked','MonthlyIncome','MaritalStatus','JobSatisfaction','JobRole','JobLevel',
'JobInvolvement','Gender','EnvironmentSatisfaction','EducationField','Education','DistanceFromHome','Department',
'BusinessTravel','Age','Attrition']

edata = data[cols]

edata['Gender']=edata['Gender'].apply(lambda x:1 if x=='Male' else 0)

edata['Attrition']=edata['Attrition'].apply(lambda x:1 if x=='Yes' else 0)

# change overtime field from text yes/no to categorical 1/0

edata['OverTime']=edata['OverTime'].apply(lambda x:1 if x=='Yes' else 0)

edata['MaritalStatus']=edata['MaritalStatus'].apply(lambda x:1 if x=='Married' else 0)

ncols=[]
for c in cols:
  if edata[c].dtype=='object':
    print(c)
  else:
    ncols.append(c)

edata=edata[ncols]


plt.figure(figsize=(20,20))
sns.heatmap(edata.corr(), cmap="YlGnBu", annot=True)

plt.show()


# Univariate Analysis

fig=plt.figure(figsize=(20,20))
i=0
for col in edata:
  if i==16:
    break
  sub=fig.add_subplot(4, 4, i+1)
  sub.set_xlabel(col)
  sub.title.set_text(col)
  edata[col].plot(kind='hist')
  i=i+1

col=ncols[16:]
edata=edata[col]


fig=plt.figure(figsize=(20,20))
i=0
for col in edata:
  sub=fig.add_subplot(3, 3, i+1)
  sub.set_xlabel(col)
  sub.title.set_text(col)
  edata[col].plot(kind='hist')
  i=i+1
  
  
# Bivariate Analysis
cols=['YearsWithCurrManager','YearsSinceLastPromotion','YearsInCurrentRole','YearsAtCompany','WorkLifeBalance',
'TrainingTimesLastYear','TotalWorkingYears','StockOptionLevel','RelationshipSatisfaction','PerformanceRating',
'PercentSalaryHike','OverTime','NumCompaniesWorked','MonthlyIncome','MaritalStatus','JobSatisfaction','JobRole','JobLevel',
'JobInvolvement','HourlyRate','Gender','EnvironmentSatisfaction','EducationField','Education','DistanceFromHome','Department',
'BusinessTravel','Age','Attrition']

edata = data[cols]

fig=plt.figure(figsize=(14,55))

for i in range (len(cols)):
  col=cols[i]
  sub=fig.add_subplot(16, 2, i+1)
  sns.countplot(data=edata, x=col, hue='Attrition', palette='RdYlBu' )
  

