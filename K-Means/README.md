# K-Means

K-means is a popular **clustering** algorithm used in machine learning. It partitions data into $k$ clusters by iteratively assigning each data point to the nearest centroid and then updating the centroids based on the mean of the points in each cluster. It repeats this process until the centroids no longer change significantly or a specified number of iterations is reached.

## Formulation

Given a dataset $X=(X_1,X_2,\dots,X_N)$, K-means clustering aims to partition the $N$ data instances into $k \leq N$ sets $S = \lbrace S_1, S_2, \dots, S_k \rbrace$ so as to minimize the within-cluster variance. 

The objective is find

$$
\hat{y} = \mathop{\arg \min}\limits_{S} \sum_{i=1}^{k} \sum_{X_n \in S_i}^{} \lVert X_n - \mu_i \lVert^2 
$$

where $\mu_i$ is the mean (also called **centroid**) of points in $S_i$

$$
\mu_i = \frac{1}{|S_i|} \sum_{X_n \in S_i}^{} X_n
$$



## Algorithm

Randomly select $k$ data points from the dataset as initial cluster centroids.

- 1 Assignment Step:
  - For each data point, calculate the distance between the point and each centroid.
  - Assign each observation to the cluster with the nearest mean.
  - $S_i^{(t)} = \lbrace X_n : \lVert X_n - \mu_i^{(t)}\lVert^2 \leq \lVert X_n - \mu_j^{(t)}\lVert^2\quad\forall j, 1 \leq j \leq k \rbrace$
  - Each data instance is assigned to exactly one cluster
- 2 Update step:
  - After assigning each data point to a cluster, calculate the new centroid for each cluster by taking the mean of all data points assigned to that cluster.
  - The new centroid becomes the representative point for that cluster.
  - $\mu_i^{(t+1)} = \dfrac{1}{|S_i^{(t)}|} \sum\limits_{X_n \in S_i^{(t)}} X_n$

Repeat until
- **Convergence**: The centroids do not change significantly between iterations.
- Maximum number of iterations is reached.