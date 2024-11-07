from sklearn import cluster, datasets
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import homogeneity_score

X_iris, y_iris = datasets.load_iris(return_X_y=True)

print(len(X_iris))

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(X_iris)

print(k_means.labels_[::10])
print(y_iris[::10])

print(homogeneity_score(k_means.labels_[::10], y_iris[::10]))
print(completeness_score(k_means.labels_[::10], y_iris[::10]))
print(v_measure_score(k_means.labels_[::10], y_iris[::10]))