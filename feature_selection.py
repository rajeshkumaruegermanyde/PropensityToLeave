import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Load the employee dataset
df=pd.read_csv('data/employee_hr_data.csv')
X=pd.get_dummies(df.drop(['EmployeeNumber','Attrition','EmployeeCount','Over18','StandardHours'], axis=1), dtype=int)
y=df['Attrition'].apply(lambda x:1 if x=='Yes' else 0)


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

XX=np.array(X_scaled)
yy=np.array(y)

# Create a random forest classifier object
rfc = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
rfc.fit(XX, yy)

# Get feature importances from the trained model
importances = rfc.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]


# Select the top 10 features
num_features = 10
top_indices = indices[:num_features]
top_importances = importances[top_indices]


# Print the top 10 feature rankings
print("Top 10 feature rankings:")
for f in range(num_features):  # Use num_features instead of 10
    print(f"{f+1}. {X.columns[indices[f]]}: {importances[indices[f]]}")
# Plot the top 10 feature importances in a horizontal bar chart
plt.barh(range(num_features), top_importances, align="center")
plt.yticks(range(num_features), X.columns[top_indices])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()



#Correlation-based Feature Selection

# Calculate feature correlations with target variable
correlations = np.abs(np.corrcoef(X.T, y)[:X.shape[1], -1])
sorted_indices = correlations.argsort()[::-1]

# Select the top k features
k = 10
selected_features = X.columns[sorted_indices[:k]]
top_correlations = correlations[sorted_indices[:k]]

print("Selected Features with correlation:")
print(selected_features)
print(top_correlations)








