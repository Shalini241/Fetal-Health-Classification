import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

features = np.genfromtxt('fetal_health.csv', missing_values=0, skip_header=1, delimiter=',', dtype=float)
features = features[1:,:]

baseline_value = features[:, 0].reshape(-1, 1)
accelerations = features[:, 1].reshape(-1, 1)
fetal_movement = features[:, 2].reshape(-1, 1)
uterine_contractions = features[:, 3].reshape(-1, 1)
light_decelerations = features[:, 4].reshape(-1, 1)
severe_decelerations = features[:, 5].reshape(-1, 1)
prolongued_decelerations = features[:, 6].reshape(-1, 1)
abnormal_short_term_variability = features[:, 7].reshape(-1, 1)
mean_value_of_short_term_variability = features[:, 8].reshape(-1, 1)
percentage_of_time_with_abnormal_long_term_variability = features[:, 9].reshape(-1, 1)
mean_value_of_long_term_variability = features[:, 10].reshape(-1, 1)
histogram_width = features[:, 11].reshape(-1, 1)
histogram_min = features[:, 12].reshape(-1, 1)
histogram_max = features[:, 13].reshape(-1, 1)
histogram_number_of_peaks = features[:, 14].reshape(-1, 1)
histogram_number_of_zeroes = features[:, 15].reshape(-1, 1)
histogram_mode = features[:, 16].reshape(-1, 1)
histogram_mean = features[:, 17].reshape(-1, 1)
histogram_median = features[:, 18].reshape(-1, 1)
histogram_variance = features[:, 19].reshape(-1, 1)
histogram_tendency = features[:, 20].reshape(-1, 1)

wcss = []
ones = np.empty_like(abnormal_short_term_variability)
X = np.c_[abnormal_short_term_variability, ones]

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for baseline_value')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(X[:,0], X[:,1])
plt.show()