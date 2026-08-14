# K-Means is distance-based, so we'll need to scale data before applying it.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, scale

# loading data from CSV file
data = pd.read_csv("Task2-customer_segmentation/Mall_Customers.csv")

#inspecting the data
# print(data.head())
# print(data.describe())
# print(data.info())
#print(data.isnull().sum())
#print(data.duplicated().sum())
# data has no null values or duplicates, so no cleaning is necessary

# visually inspecting the relationship between annual income and spending score
# plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.title('Customer Segmentation')
# plt.show()

# defining features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# scaling the features
scale = StandardScaler()
X_scaled = scale.fit_transform(X)

# finding optimal k
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=7, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# plotting the elbow curve
# plt.plot(range(1, 11), inertia, marker='o')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.title('Elbow Method for Optimal k')
# plt.show()

# based on the elbow method, we can choose k=5 for clustering
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=7, n_init=10)

clusters = kmeans.fit_predict(X_scaled)

data['Cluster'] = clusters # adding cluster column to the original data
#print(data.head()) # just checking to see it's there lol

print(kmeans.inertia_) # checking inertia for best_k

# visualizing the clusters and unscaling the data for better understanding
# eventhough both unscaled and scaled data graphs will look the same because scaling is a linear transformation.

centroids = scale.inverse_transform(kmeans.cluster_centers_)

for cluster in range(best_k):
    cluster_points = X[data['Cluster'] == cluster]
    plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], label=f'Cluster {cluster}')

plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='black', marker='X', label='Centroids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation with K-Means Clustering')
plt.legend()
plt.show()

