# k-Nearest Neighbors (KNN)

The k-Nearest Neighbors (KNN) algorithm is a simple yet effective supervised learning algorithm used for classification and regression tasks.

Given a dataset with labeled data points, the KNN algorithm classifies new data points based on the majority class of their $k$ nearest neighbors in the feature space.





## Algorithm Steps:

1. **Calculate Distance:** Measure the <ins>distance</ins> between the new data point and all other data points in the dataset. 
  
2. **Find Neighbors:** Identify the $k$ nearest neighbors of the new data point based on the calculated distances.
   
3. **Majority Voting:** For classification tasks, count the occurrences of each class among the $k$ nearest neighbors. Assign the class label that appears <ins>most frequently</ins> as the predicted class for the new data point. For regression tasks, the predicted value can be the mean or median of the k nearest neighbors target values.

<div align="center">
  <img src="https://www.codespeedy.com/wp-content/uploads/2020/02/Terrorism-Detection-and-Classification-using-kNN-Algorithm.png" alt="Image description" width="400">
  <p>KNN visualization</p>
</div>

In the figure, with $k = 3$ the new data should be labeled as class B.





## Distance Calculation

Given $P = (p_1, p_2, \dots, p_n)$ and $Q = (q_1, q_2, \dots, q_n)$ points in n-dimensional, let's calculate their distances using different approaches.

### Euclidian Distance

Euclidean distance is a measure of the straight-line distance between two points in a Euclidean space, which is essentially the space we are most familiar with in everyday life. 

$$d(P,Q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}$$


### Manhattan distance

Manhattan distance is a measure of distance moving only horizontally or vertically in the space.

$$d(P,Q) = |p_1 - q_1| + |p_2 - q_2| + \dots + |p_n - q_n|$$


### Minkowski distance

The Minkowski distance is a generalization of both the Euclidean distance and the Manhattan distance.

$$d(P,Q) = \big{(} \sum\limits^{n}_ {i = 1} |p_i - q_i|^p \big{)}^{\dfrac{1}{p}}$$




## Advantages and Disadvantages:
- *Advantages:* Simple to implement, robust to noisy data, effective for both classification and regression tasks.
- *Disadvantages:* Computationally expensive for large datasets, sensitive to irrelevant features, and the choice of $k$ can impact performance.
