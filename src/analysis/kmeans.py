from sklearn.cluster import KMeans


def cluster_covariance_matrix(C, num_clusters=10):

    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=0,
    )

    labels = kmeans.fit_predict(C)

    ordering = labels.argsort()

    C_clustered = C[ordering][:, ordering]

    return C_clustered