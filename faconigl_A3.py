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
from sklearn.cluster import AgglomerativeClustering
from sklearn import cluster
from sklearn.metrics import silhouette_score

# Dataset 1: email.tsv

email_file = pd.read_csv("email.tsv", sep="\t")
X_email = email_file.select_dtypes(include=['number']) # Select only numerical columns

# KMeans clustering

k_means_email = cluster.KMeans(n_clusters=2)
k_means_email.fit(X_email)

print("Email Dataset KMeans Clustering Results:")
print(k_means_email.labels_[:20])
print("Silhouette Score: ", silhouette_score(X_email, k_means_email.labels_))

# Agglomerative clustering

agg_clust_email = AgglomerativeClustering(n_clusters=2)
agg_clust_email_labels = agg_clust_email.fit_predict(X_email)

print("\nEmail Dataset Agglomerative Clustering Results:")
print(agg_clust_email_labels[:20])
print("Silhouette Score (Agglomerative): ", silhouette_score(X_email, agg_clust_email_labels))

# Divider
print("\n---------------------------------------------------\n")

# Dataset 2: Credit.csv

credit_file = pd.read_csv("Credit.csv")
X_credit = credit_file.select_dtypes(include=['number']) # Select only numerical columns

# KMeans clustering

k_means_credit = cluster.KMeans(n_clusters=3)
k_means_credit.fit(X_credit)

print("Credit Dataset KMeans Clustering Results:")
print(k_means_credit.labels_[:20])
print("Silhouette Score: ", silhouette_score(X_credit, k_means_credit.labels_))

# Agglomerative clustering

agg_clust_credit = AgglomerativeClustering(n_clusters=3)
agg_clust_credit_labels = agg_clust_credit.fit_predict(X_credit)

print("\nCredit Dataset Agglomerative Clustering Results:")
print(agg_clust_credit_labels[:20])
print("Silhouette Score (Agglomerative): ", silhouette_score(X_credit, agg_clust_credit_labels))

# Referenced example code from ex1_kmeans_basic_iris.py and k-means-clustering-visual.py