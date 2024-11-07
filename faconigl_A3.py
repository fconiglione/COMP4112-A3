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
from sklearn import cluster

# Dataset 1: email.tsv

email_file = pd.read_csv("email.tsv", sep="\t")
X_email = email_file.select_dtypes(include=['number'])

k_means_email = cluster.KMeans(n_clusters=2)
k_means_email.fit(X_email)

print(k_means_email.labels_[:20])

print("Email Dataset Clustering Results:")

# Dataset 2: Credit.csv

credit_file = pd.read_csv("Credit.csv")
X_credit = credit_file.select_dtypes(include=['number'])

k_means_credit = cluster.KMeans(n_clusters=2)
k_means_credit.fit(X_credit)

print(k_means_credit.labels_[:20])

print("Credit Dataset Clustering Results:")