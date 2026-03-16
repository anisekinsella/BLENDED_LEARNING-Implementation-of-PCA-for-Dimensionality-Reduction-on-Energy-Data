# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Standardization: Standardize the data using StandardScaler to have zero mean and unit variance.
2. Covariance Matrix Computation: Compute the covariance matrix of the standardized data.
3. Eigenvalue Decomposition: Perform eigenvalue decomposition on the covariance matrix to obtain eigenvectors and eigenvalues.
4. Transformation: Select top k eigenvectors (principal components) and transform the data onto these components.
   
## Program:
```

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('HeightsWeights.csv')
print("First 5 rows of the dataset:")
print(data.head())

# Select features
X = data[['Height(Inches)', 'Weight(Pounds)']]

# Plot original data
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title("Original Data Distribution")
plt.show()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Create PCA dataframe
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Plot PCA results
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

```

## Output:
<img width="846" height="750" alt="image" src="https://github.com/user-attachments/assets/be9533e3-74a2-4c1d-9a90-fea5e54de899" />
<img width="853" height="641" alt="image" src="https://github.com/user-attachments/assets/5c71ac13-6c5b-4f5d-bf5f-6118e86ce744" />



## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
