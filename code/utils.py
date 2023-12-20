import numpy as np
from sklearn.metrics import pairwise_distances
from collections import Counter


class KMeans:
    def __init__(self, k, init='random', max_iterations=100):
        self.k = k
        self.init = init
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None
        self.inertia_ = 0  # Inertia attribute

    def _initialize_centroids(self, X):
        if self.init == 'random':
            indices = np.random.choice(X.shape[0], self.k, replace=False)
            self.centroids = X[indices]
        elif self.init == 'k-means++':
            self.centroids = [X[np.random.randint(X.shape[0])]]  # Choose the first centroid randomly
            for _ in range(1, self.k):
                distances_to_centroids = pairwise_distances(X, np.array(self.centroids))
                min_distances = np.min(distances_to_centroids, axis=1)
                probabilities = min_distances / min_distances.sum()
                cumulative_probabilities = np.cumsum(probabilities)
                random_value = np.random.rand()
                next_centroid_index = np.where(cumulative_probabilities >= random_value)[0][0]
                self.centroids.append(X[next_centroid_index])
            self.centroids = np.array(self.centroids)
        elif self.init == 'KR':
            # Kaufman and Rousseeuw initialization
            distances = pairwise_distances(X)
            centrality = distances.sum(axis=1)
            first_centroid_index = np.argmin(centrality)
            self.centroids = [X[first_centroid_index]]
            for _ in range(1, self.k):
                distances_to_centroids = pairwise_distances(X, np.array(self.centroids))
                min_distances = np.min(distances_to_centroids, axis=1)
                next_centroid_index = np.argmax(min_distances)
                self.centroids.append(X[next_centroid_index])
            self.centroids = np.array(self.centroids)
        elif self.init == 'KKZ':
            # Katsavounidis et al. initialization
            norms = np.linalg.norm(X, axis=1)
            first_centroid_index = np.argmax(norms)
            self.centroids = [X[first_centroid_index]]
            for _ in range(1, self.k):
                distances_to_centroids = pairwise_distances(X, np.array(self.centroids))
                min_distances = np.min(distances_to_centroids, axis=1)
                next_centroid_index = np.argmax(min_distances)
                self.centroids.append(X[next_centroid_index])
            self.centroids = np.array(self.centroids)
        else:
            raise ValueError(f"Unsupported initialization method: {self.init}")

    def _closest_centroid(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _move_centroids(self, X, closest):
        self.centroids = np.array([X[closest == i].mean(axis=0) for i in range(self.k)])

    def _compute_inertia(self, X, closest):
        # Calculate the sum of squared distances between each point and its assigned centroid
        distances = np.linalg.norm(X - self.centroids[closest], axis=1)
        self.inertia_ = np.sum(distances ** 2)

    def fit(self, X, visualization=True):
        self._initialize_centroids(X)
        for iteration in range(self.max_iterations):
            closest = self._closest_centroid(X)
            self._move_centroids(X, closest)
            self._compute_inertia(X, closest)  # Update inertia after moving centroids

            if np.array_equal(self.labels, closest):
                break
            self.labels = closest

    def predict(self, X):
        return self._closest_centroid(X)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit_transform(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        cov = np.cov(X_centered.T)

        # Eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Store first n eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # Project the data onto the principal components
        X_transformed = np.dot(X_centered, self.components)

        return X_transformed
    

class MyStandardScaler:
    def fit(self, X):

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        :param X: array-like, shape [n_samples, n_features]
                   The data used to scale along the features axis.
        :return: The transformed array.
        """
        self.fit(X)
        return self.transform(X)
    

def clustering_accuracy(ground_truth, predictions):
    """
    Calculate the clustering accuracy.

    """
    # Ensure ground_truth and predictions are numpy arrays
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    # Create a mapping from predicted clusters to the most common ground truth label
    cluster_to_label_mapping = {}
    for cluster in np.unique(predictions):
        # Find indices where the prediction equals the current cluster
        indices = np.where(predictions == cluster)[0]

        # Find the most common ground truth label in these indices
        most_common_label = Counter(ground_truth[indices]).most_common(1)[0][0]
        cluster_to_label_mapping[cluster] = most_common_label

    # Calculate the accuracy
    correct_predictions = sum([gt == cluster_to_label_mapping[pred]
                               for gt, pred in zip(ground_truth, predictions)])
    cluster_accuracy = correct_predictions / len(ground_truth)

    return cluster_accuracy

def davies_bouldin_score(X, labels):
    """
    Compute the Davies-Bouldin score.
    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster distances
    to between-cluster distances. Thus, clusters which are farther apart and less
    dispersed will result in a better score.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row corresponds to a single data point.
    labels : array-like, shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score : float
        The resulting Davies-Bouldin score.
    """
    n_clusters = len(np.bincount(labels))
    cluster_k = [X[labels == k] for k in range(n_clusters)]
    centroids = [np.mean(k, axis=0) for k in cluster_k]
    variances = [np.mean(pairwise_distances(k, [centroid])) for k, centroid in zip(cluster_k, centroids)]

    db_index = np.mean([max((variances[i] + variances[j]) /
                            pairwise_distances([centroids[i]], [centroids[j]])[0][0]
                            for j in range(n_clusters) if i != j)
                        for i in range(n_clusters)])

    return db_index

def create_label_mapping(labels):
    """
    Create a mapping from unique labels to integers.
    """
    return {label: idx for idx, label in enumerate(np.unique(labels))}

def map_labels(labels, mapping):
    """
    Map the labels to integers based on the provided mapping.
    """
    return [mapping[label] for label in labels]

