# Learn-ML

#### Author: Victor Nascimento Ribeiro


## 1. Introduction 

This repository aims to introduce readers to basic notions and fundamental concepts in the field of study on Machine Learning, along with some code implementations. For a good understanding, it is essential to be familiar with concepts of Differential Calculus, Linear Algebra and Statistics. No prior knowledge of Machine Learning is required :)

We will cover **Supervised Machine Learning**, mathematically formulate this type of problem, explain how training is done and delve into some algorithms.




## 2. Supervised Learning

Let's start with a question: How does the human being learn? This may be a somewhat complex question, but in a simplified way, we can say that one of the forms of learning is through observations. In childhood, during literacy, we are presented with vowels along with their phonemes; our brain associates each shape with its sound. In a way, when we see any of the vowels in a different handwriting, we can still recognize it. Initially, we will face difficulties, but with each mistake, we are corrected by our teachers and parents. As we see that letter over and over again, after numerous errors and corrections, we finally learn to recognize it with rare mistakes. Supervised Machine Learning works in a very similar way, and this will become evident as we progress through the topics.





### 2.1 Mathematical Formulation

Let's see another example. Imagine you're a doctor and want to decide if a patient $j$ is ill. To do this, you ask them to fill out a list of symptoms, indicating the intensity of each. Let's formulate this problem mathematically. Consider $\mathcal{X}$ as the input hypothesis space, where $X_j = (x_1, x_2, \dots, x_d) \in \mathcal{X}$ represents the list of symptoms of size $d$ filled out by patient $j$. Also, consider $\mathcal{Y}$ as the output space, where $y_j \in \mathcal{Y}$ represents whether the person is ill or not. It can be said that there is a relationship between inputs and outputs, which can be described by a function $f: \mathcal{X} \xrightarrow{} \mathcal{Y}$, called the **Target Function**.

The problem now becomes finding the Target Function $f$. However, most of the time, there is no analytical solution to find it. In such cases, we use Machine Learning. Now, denoting $X = (X_1, X_2, \dots, X_N)$ and $y = (y_1, y_2, \dots, y_N)$ from a set of $N$ observations (patients), we define $\mathcal{D} = {(X_i,y_i) \in \mathcal{X} \times \mathcal{Y}: i = 1,\dots,N }$, which we call **Dataset**. Having a family of functions $\mathcal{H}$, we want to find a function $h_w: \mathcal{X} \xrightarrow{} \mathcal{Y}$, with $h_w \in \mathcal{H}$, such that $h_w(X) \approx f(X)$ for any pair $(X,y) \in \mathcal{X} \times \mathcal{Y}$. In general, the family $\mathcal{H}$ is the family of linear functions, where the real coefficients of the function are denoted by the **weight vector** $w = (w_1, \dots, w_d) \in \mathbb{R}^{d}$, and $b \in \mathcal{R}$ called the **bias**, so $h_w(X_i) = b + w_1 x_{i1} + \dots + w_d x_{id}$. The learning algorithm will adjust $w$ and $b$ to maximize this approximation.

<div align="center">
  <img src="https://i.imgur.com/BT3aVap.png" alt="Image description" width="600">
  <p>Figure 1: Representation of components in Supervised Machine Learning</p>
</div>






## 2.2 Notation

- $N$: Number os samples in the dataset
- $d$: Number of features in each data instance

$$
X = 
\begin{bmatrix} 
\text{---} & X_1 & \text{---} \\
\text{---} & X_2 & \text{---} \\ 
& \vdots \\ 
\text{---} & X_N & \text{---} 
\end{bmatrix} = 
\begin{bmatrix} 
x_{11} & x_{12} & ... & x_{1d} \\
x_{21} & x_{22} & ... & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\ 
x_{N1} & x_{N2} & ... & x_{Nd} 
\end{bmatrix}\quad\quad
\begin{split} 
  & X \in \mathcal{R}^{N\times d} & \text{(Dataset)} \\
  & X_i \in \mathcal{R}^d & \text{(Data instance)} \\ 
  & x_{ij} \in \mathcal{R} & \text{(Feature)}
\end{split}
$$

$$w = (w_1, w_2, ... , w_d)^T\quad w_i \in \mathbb{R}\quad\text{(Weights)}$$

$$\hat{y_i} = b + w_1x_{i1} + \dots + w_dx_{id} \quad\text{(Prediction)}$$

$$y_i \quad\text{(Target)}\quad$$

#### Artificial component to simplify notatiton:
$$ X_i = (\mathbf{1}, x_{i1}, x_{i2}, ... , x_{id})\quad(x_{i0} = 1)$$

$$ w = (\mathbf{w_0}, w_1, w_2, ... , w_d)^T\quad (w_0 \text{ is the bias } b)$$

Thus $\hat{y} = w^T \cdot X, \quad$ where $(\cdot)$ denotes the dot product which will be omitted from now on.

Now that the problem has been formulated, the next step is to understand how the learning algorithm is executed.






## 2.3. Training

To measure how far $h_w$ is from $f$, we need to define a **Loss Function** or cost function $L(w)$. Let's denote $h_w(X)$ as $\hat{y}$, where $h_w \in \mathcal{H}$ is a linear function. From a dataset $\mathcal{D}$. Learning occurs through the minimization of the Loss Function; the smaller it is, the closer $h_w$ is to $f$. To achieve this, let's understand how the **Gradient Descent** algorithm can be used to find the minimum point of this function by adjusting the weights $w$.






### 2.3.1 Gradient Descent

The gradient vector is a vector normal to the contour of a function at a given point, indicating the direction and sense in which the point should be moved to maximize the function's value. Consequently, the negative of this vector points in the direction and sense of the greatest decrease.

$$\nabla L(w) = \Big{[} \dfrac{\partial L}{\partial w_1}, \dfrac{\partial L}{\partial w_2}, \dots, \dfrac{\partial L}{\partial w_d} \Big{]} ^{T}$$

The main idea of the Gradient Descent algorithm is to choose a vector $w$, initially with arbitrary values, and at each iteration $t$, update it by a small fraction $\eta$ (**learning rate**) in the opposite direction of the gradient of the Loss Function, for a predefined number of iterations called **epochs**. 

$$ w(t + 1) = w(t) - \eta \nabla L(w) $$


<div align="center">
  <img src="https://i.imgur.com/dlZ3GPz.jpeg" alt="Image description" width="200">
  <p>Figure 2: Loss Function optimization</p>
</div>


#### 2.3.1.1 Stochastic Gradient Descent (SGD):

   - __Overview__: In SGD, instead of using the entire dataset for each iteration, only a single random data point is used to update the model parameters.
   - __Pros__ : Faster updates, especially for large datasets. It can escape local minima due to the stochastic nature.
   - __Cons__: Noisy updates, might oscillate around the minimum. Might take a longer time to converge.

#### 2.3.1.2 Batch Gradient Descent:

   - __Overview__: In Batch GD, the entire dataset is used for each iteration to update the model parameters.
   - __Pros__ : Smooth convergence, less noisy updates. Suitable for smaller datasets.
   - __Cons__: Computationally expensive for large datasets. May take longer to converge.

#### 2.3.1.3 Mini-Batch Gradient Descent:

   - __Overview__: Mini-Batch GD strikes a balance between SGD and Batch GD by using a small, random subset (mini-batch) of the data for each iteration.
   - __Pros__ : Combines advantages of both SGD and Batch GD. Suitable for a wide range of dataset sizes.
   - __Cons__: Still has some noise in updates, and the choice of mini-batch size is a hyperparameter.





## 2.3.2 Weight Initialization

There are several ways to initialize $w$, with the simplest being to start with random values. However, there are more intelligent ways to choose initial values, ensuring that the function converges in fewer iterations. To delve deeper into this topic, read Chapter 3 of [III](http://neuralnetworksanddeeplearning.com/).




## 2.4. Classification and Regression

In **classification** problems, the output $y$ corresponds to a label of a class, meaning a discrete value representing a category. Let's go back to the doctor's example: based on a patient's symptoms, deciding whether they are ill or not. This is a binary classification problem, where the output has two possible labels: 0 for a healthy patient and 1 for a sick patient. Note that the output doesn't necessarily have to be binary; it can contain an arbitrary number of classes.

Now, in **regression** problems, the output corresponds to a continuous numerical value. For example, given a person's height, determining their weight. In this case, the output can vary over any real value, characterizing regression. Similar to classification, the output can contain an arbitrary number of values.




## 3. Unsupervised Learning


Unsupervised Learning is a type of machine learning where the algorithm learns patterns from **unlabeled data without any explicit guidance or supervision**. In unsupervised learning, the system tries to identify the underlying structure in the data on its own.

There are two main types of unsupervised learning techniques:

**- Clustering**: Clustering algorithms group similar data points together based on certain criteria. The goal is to partition the data into clusters such that data points within the same cluster are more similar to each other than to those in other clusters. Examples of clustering algorithms include k-means, hierarchical clustering, and DBSCAN.

**- Dimensionality reduction**: Dimensionality reduction techniques aim to reduce the number of features in the data while preserving its essential structure. This is often done to simplify the dataset and make it more manageable for analysis. Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE) are common dimensionality reduction techniques.


## References

I.  Abu-Mostafa, Yaser S., Magdon-Ismail, Malik and Lin, Hsuan-Tien. Learning From Data. : AMLBook, 2012.

II. Learning from Data - Caltech [online course](https://work.caltech.edu/telecourse)

III. M. A. Nielsen. Neural networks and deep learning, 2018 - [online book](http://neuralnetworksanddeeplearning.com/)

