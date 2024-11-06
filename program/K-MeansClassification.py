# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


iris = load_iris()
X = iris.data  
y = iris.target 

kmeans = KMeans(n_clusters=3, random_state=42) 
kmeans.fit(X)


cluster_labels = kmeans.labels_

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)


plt.figure(figsize=(10, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering of Iris Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid()
plt.show()

print("Cluster Centers:")
print(kmeans.cluster_centers_)