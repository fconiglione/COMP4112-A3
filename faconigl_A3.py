# COMP 4112 Introduction to Data Science
# Assignment 3, Unsupervised learning
# Francesco Coniglione (st#1206780)

"""
In this assignment you will cluster a total of 2 datasets we have looked at in the course. You can select
the datasets you want to cluster. Additionally, try and report some information regarding the clustering.
There are no specific requirements for this assignment other than clustering of 2 chosen datasets; the
amount of detail you want to report about the clustering is up to you, but ideally it should show some
evidence of effort and experimentation.
"""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering # From https://scikit-learn.org/dev/modules/generated/sklearn.cluster.AgglomerativeClustering.html
from sklearn import cluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Dataset 1: auto.tsv

auto_file = pd.read_csv("Auto.csv")
X_auto = auto_file.select_dtypes(include=['number']) # Select only numerical columns

# KMeans clustering

k_means_auto = cluster.KMeans(n_clusters=3) # After some experimentation, 3 clusters seems to be the best fit
k_means_auto.fit(X_auto)

print("Auto Dataset KMeans Clustering Results:")
print(k_means_auto.labels_[:20])
print("Silhouette Score: ", silhouette_score(X_auto, k_means_auto.labels_))

# Agglomerative clustering

agg_clust_auto = AgglomerativeClustering(n_clusters=2)
agg_clust_auto_labels = agg_clust_auto.fit_predict(X_auto)

print("\nAuto Dataset Agglomerative Clustering Results:")
print(agg_clust_auto_labels[:20])
print("Silhouette Score (Agglomerative): ", silhouette_score(X_auto, agg_clust_auto_labels))

# Plotting the clusters for three features

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X_auto.iloc[:, 0], X_auto.iloc[:, 1], X_auto.iloc[:, 2], c=k_means_auto.labels_, cmap='viridis', edgecolor='k')
ax.set_title("Auto Dataset KMeans Clustering")
ax.set_xlabel("mpg")
ax.set_ylabel("cylinders")
ax.set_zlabel("displacement")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_auto.iloc[:, 0], X_auto.iloc[:, 1], X_auto.iloc[:, 2], c=agg_clust_auto_labels, cmap='viridis', edgecolor='k')
ax2.set_title("Auto Dataset Agglomerative Clustering")
ax2.set_xlabel("mpg")
ax2.set_ylabel("cylinders")
ax2.set_zlabel("displacement")

# Divider
print("\n---------------------------------------------------\n")

# Dataset 2: Credit.csv

credit_file = pd.read_csv("Credit.csv")
X_credit = credit_file.select_dtypes(include=['number']) # Select only numerical columns

# KMeans clustering

k_means_credit = cluster.KMeans(n_clusters=3) # After some experimentation, 3 clusters seems to be the best fit
k_means_credit.fit(X_credit)
credit_labels_kmeans = k_means_credit.labels_

print("Credit Dataset KMeans Clustering Results:")
print(k_means_credit.labels_[:20])
print("Silhouette Score: ", silhouette_score(X_credit, k_means_credit.labels_))

# Agglomerative clustering

agg_clust_credit = AgglomerativeClustering(n_clusters=3)
agg_clust_credit_labels = agg_clust_credit.fit_predict(X_credit)

print("\nCredit Dataset Agglomerative Clustering Results:")
print(agg_clust_credit_labels[:20])
print("Silhouette Score (Agglomerative): ", silhouette_score(X_credit, agg_clust_credit_labels))

# Plotting the clusters for three features

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(X_credit.iloc[:, 0], X_credit.iloc[:, 1], X_credit.iloc[:, 2], c=credit_labels_kmeans, cmap='viridis', edgecolor='k')
ax.set_title("Credit Dataset KMeans Clustering")
ax.set_xlabel("Income")
ax.set_ylabel("Limit")
ax.set_zlabel("Rating")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_credit.iloc[:, 0], X_credit.iloc[:, 1], X_credit.iloc[:, 2], c=agg_clust_credit_labels, cmap='viridis', edgecolor='k')
ax2.set_title("Credit Dataset Agglomerative Clustering")
ax2.set_xlabel("Income")
ax2.set_ylabel("Limit")
ax2.set_zlabel("Rating")
plt.show()

# Referenced example code from ex1_kmeans_basic_iris.py and k-means-clustering-visual.py