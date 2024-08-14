import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split

def outlierp_removal(df, feature):
    lower = np.percentile(df[feature],1)
    upper = np.percentile(df[feature], 99)
    
    indexu = df[ (df[feature] >= upper) ].index
    indexl = df[ (df[feature] <= lower) ].index

    df.drop(indexu , inplace=True)
    df.drop(indexl , inplace=True)


# Load the dataset
data=pd.read_csv('data/employee_hr_data.csv')
print(f"Shape before outlier removal: {data.shape}")

# numeric columns
ncols=['YearsWithCurrManager','YearsInCurrentRole','YearsAtCompany',
       'DailyRate','HourlyRate',
      'TotalWorkingYears','NumCompaniesWorked','MonthlyIncome',
      'MonthlyRate','DistanceFromHome','Age','PercentSalaryHike',
      'YearsSinceLastPromotion','OverTime',
      'Education',
      'EnvironmentSatisfaction',
      'JobLevel',
      'JobInvolvement',
      'JobSatisfaction',
      'Gender',
      'Attrition'
]


# Box Plots

edata = data[ncols]

fig=plt.figure(figsize=(22,40))
for i in range (len(ncols)-1):
  colname=ncols[i]
  sub=fig.add_subplot(6, 4, i+1)
  sns.boxplot(data=edata, hue='Attrition', y=colname, palette='RdYlBu_r')
        


# create boxplots to visualize high outliers and select one which has max outliers

ocols=['YearsWithCurrManager','YearsInCurrentRole','YearsAtCompany',
               'DailyRate',
              'TotalWorkingYears','NumCompaniesWorked','MonthlyIncome',
              'DistanceFromHome','PercentSalaryHike',
              'YearsSinceLastPromotion']


# create a figure of size width=20 units and height=30 units
fig=plt.figure(figsize=(15,26))
i=0
for column in data.columns:
    if(i==12):
        break
    if column in ocols:
        sub=fig.add_subplot(6,2,i+1)
        #sub.set_xlabel(column)
        sub.title.set_text(column)
        data.boxplot([column])
        i=i+1
plt.show()





# based on boxplot visualization, there are significant outliers only in
# MonthlyIncome field

ocols=['MonthlyIncome']


for f in ocols:
    outlierp_removal(data, f)
    
print(f"Shape after outlier removal: {data.shape}")
