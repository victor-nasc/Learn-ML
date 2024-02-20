# Support Vector Machines

Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates different classes in the feature space, maximizing the margin between classes while minimizing classification error.


## Setup

Given a set of labeled data points $(X_i,y_i)$ where ​the class label $y_i=\lbrace −1,1\rbrace$ (binary classification), the goal of SVM is to find the optimal hyperplane that separates the data points into different classes while maximizing the margin.

$$
h_w(X_i)
\begin{cases}
    +1, & \text{if } w^T X_i \geq 1\\
    -1, & \text{if } w^T X_i \leq 1
\end{cases}
\implies y_i(w^T X_i) \geq 1
$$

$$i=1,2,\dots,N$$


## Margin

<div align="center">
  <img src="https://www.reneshbedre.com/assets/posts/svm/svm_linear.webp" alt="Image description" width="500">
  <p>SVM Visual Representation</p>
</div>

The hyperplane can be represented as $w^T z = 0$, where $w$ is the weight vector **perpendicular to the hyperplane**.

The distance of a point $z$ from the hyperplane can be expressed as:

$$d(z,w) = \frac{|w^T z|}{\lVert w \rVert}$$

To find the margin size, we can utilize the fact that the closest point $\bar{X}$ (support vectors) to the decision boundary lies on the hyperplane, which implies that $w^T \bar{X} = 1$.

Let $\bar{X} = z + k \frac{w^T}{\lVert w \rVert}$ for an $z$ in the decision boudary and $k \in \mathcal{R}$, thus

$$1 = w^T \bar{X} = w^T (z + k \frac{w^T}{\lVert w \rVert}) = \underbrace{w^T z}_{0} + w^T k \frac{w^T}{\lVert w \rVert} = k \frac{{\lVert w \rVert}^2}{\lVert w \rVert}= k \lVert w \rVert$$

$$\implies k = \frac{1}{\lVert w \rVert}$$

So the margin size is $\frac{2}{\lVert w \rVert}$




## Optimization 

The margin size in a SVM is determined by $\frac{2}{\lVert w \rVert}$. In the optimization process, the objective is to maximize this margin size. This is equivalent to minimizing $\frac{\lVert w \rVert}{2}$.

To achieve this we will use **Lagrange Multipliers**

$$
\begin{cases}
    \frac{\lVert w \rVert}{2} & \text{Objective Function}\\
    y_i(w^T X_i) \geq 1 & \text{Constraint} 
\end{cases}
$$

To extend SVM to cases in which the data are not linearly separable (**Soft-margin**), the **hinge loss function** is helpful.

$$max\big{(} 0, y_i(w^T X_i) -1 \big{)}$$

Thus

$$\mathcal{L}(X,y,\lambda) = \frac{\lVert w \rVert}{2} + \sum_{i=1}^{N} \alpha_i max\big{(} 0, y_i(w^T X_i) -1\big{)}$$

Let's calculate it's gradient

$$\nabla \mathcal{L}(w) = \begin{bmatrix} \dfrac{\partial \mathcal{L}}{\partial w_0} & \dfrac{\partial \mathcal{L}}{\partial w_1} & ... & \dfrac{\partial \mathcal{L}}{\partial w_d} \end{bmatrix} ^{T}$$

Where

$$\frac{\partial \mathcal{L}}{\partial w_j} = w + \sum_{i=1}^{N} \alpha_i y_i X_i $$

We can simplify it by substituting $\alpha_i, i=1,2,\dots,N$ for $C \in \mathcal{R}$, thus

$$\frac{\partial \mathcal{L}}{\partial w_j} = w + C \sum_{i=1}^{N} y_i X_i$$
