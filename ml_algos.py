import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from numpy import percentile
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from numpy import mean
from numpy import std
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
from keras import layers


# Percentil method to detect and remove outliers
def outlierp_removal(df, feature):
    a=df.shape[0]
    lower_limit = np.percentile(df[feature],1)
    upper_limit = np.percentile(df[feature], 99)
    #Q4=np.percentile(df[feature], 100)
    #dff=df[(df[feature] < lower_limit) & (df[feature] > upper_limit)]
    con1=(df[feature] > upper_limit)
    con2=(df[feature] < lower_limit)
    df=df[~con1]
    df=df[~con2]
    b=a-df.shape[0]
    print(f"Feature: {feature} shape: {df.shape} removed={b}")
    return df



# Load the employee dataset
df=pd.read_csv('data/employee_hr_data.csv')
 
# Feature set selected from feature selection methods    
    
fsc=[
        'OverTime',
        'TotalWorkingYears',
        'JobLevel',
        'YearsInCurrentRole',
        'MonthlyIncome',
        'Age',
        'JobRole',
        'YearsWithCurrManager',
        'MonthlyRate',
        'DailyRate',
        'DistanceFromHome',
        'MaritalStatus',
        'Attrition'
        ]


# Features that require outlier removal
orf=['MonthlyIncome', ]


data=df[fsc]
print(f"Data shape before outlier removal: {data.shape}")

#print(data['Attrition'].value_counts()['Yes'])


# remove outliers    
for feature in orf:
    data=outlierp_removal(data, feature)


print(f"Data shape after outlier removal: {data.shape}")


XX=pd.get_dummies(data.drop(['Attrition'], axis=1), dtype=int)
y=data['Attrition'].apply(lambda x:1 if x=='Yes' else 0)


from sklearn.preprocessing import PowerTransformer
ptr = PowerTransformer()
X=ptr.fit_transform(XX)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#print(y_train.sum())
#print(y_test.sum())

models=[
        ("LRC", LogisticRegression() ),
        ("DTC",DecisionTreeClassifier() ),
        ("RFC",RandomForestClassifier() ),
        ("SVC",SVC(kernel='rbf', gamma='scale', C=1) ),
        ("KNN",KNeighborsClassifier()),
        ("GNB", GaussianNB())
        ]



for name, model in models:
    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)

    print(f"Model Name: {name}")
    # print accuracy
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100}")

    # print classification report
    #print(classification_report(y_test,y_pred))
    print('Precision:', metrics.precision_score(y_test, y_pred))
    print('Recall:', metrics.recall_score(y_test, y_pred))
    print('F1 Score:', metrics.f1_score(y_test, y_pred))


    # define confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(confusion_matrix)

    sns.heatmap(confusion_matrix,
            annot=True,
            fmt='g',
            xticklabels=['Active','Attrition'],
            yticklabels=['Active','Attrition'])

    # display matrix
    plt.ylabel('Actual',fontsize=12)
    plt.xlabel('Prediction ( '+name+' )'    ,fontsize=12)
    plt.show()

    
    
    

