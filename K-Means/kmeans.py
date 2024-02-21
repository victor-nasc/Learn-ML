import numpy as np


# Victor Nascimento Ribeiro
# 02/2024
    
class KMeans:
    def __init__(self, k, max_iter=5000):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
    
    
    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iter):
            # Assign data points to the closest centroid
            labels = self._assign_clusters(X)
            
            # Update centroids based on the assigned data points
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break
            
            self.centroids = new_centroids
        
        self.labels = self._assign_clusters(X)
    
    
    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    
    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            new_centroids[i] = X[labels == i].mean(axis=0)
        return new_centroids