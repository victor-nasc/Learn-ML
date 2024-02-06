# Multinomial Logistic Regression

Multinomial logistic regression, also known as Softmax Regression, is a classification algorithm that extends logistic regression to handle **multiple classes**. It calculates the probabilities of each class for a given input and then normalizes them using the **softmax** function. The class with the highest probability is predicted as the output, making it suitable for multi-class classification problems.





## Hypotesis

Since our target $f$ is such that $0 \leq f(X) \leq 1$, let us consider hypotheses of the same type: 


$$\hat{y} = h_w(X) = \phi(w^{T}X) \approx f(X)$$

where $K$ represents the number of classes

$${\phi(z)}_ {j} =  \frac{e^{z_j}}{\sum\limits^{K}_{i = 1} e^{z_i}} \quad j=1,2,\dots,K \quad z \in \mathcal{R}^K$$

$$ 0 \leq {\phi(z)}_ {j} \leq 1 \implies 0 \leq h_w(X) \leq 1$$ 

known as **Softmax Function**






### Notation

To use Softmax Regression we need to treat the data in a different way

- $N$ = Dataset size 
- $D$ = Features 
- $K$ = Number of Classes 

$$
W = 
\begin{bmatrix} 
w_{11} & w_{12} & ... & w_{1K} \\ 
w_{21} & w_{22} & ... & w_{2K} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
w_{D1} & w_{D2} & ... & w_{DK} 
\end{bmatrix}, \quad\quad\quad
\hat{Y} =  
\begin{bmatrix} 
\hat{p}_ {11} & \hat{p}_ {12} & ... & \hat{p}_ {1K} \\ 
\hat{p}_ {21} & \hat{p}_ {22} & ... & \hat{p}_ {2K} \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\hat{p}_ {N1} & \hat{p}_ {N2} & ... & \hat{p}_{NK} 
\end{bmatrix} = 
\begin{bmatrix} 
\text{---} & \hat{y}_1 & \text{---} \\
\text{---} & \hat{y}_2 & \text{---} \\ 
& \vdots \\ 
\text{---} & \hat{y}_N & \text{---} 
\end{bmatrix}
$$

Note that $\hat{y_i} = \phi(w^T X_i) \in \mathcal{R}^K$ represents the <ins>probabilities for each class</ins>

The predicted class of $X_i$ is $argmax(\hat{y}_i)$.

The target $y$ must be **One-hot encoded**:

ex.: $K = 4, \quad N = 3$

$$
y = 
\begin{bmatrix} 
y_0 \\ 
y_1 \\ 
y_2 
\end{bmatrix} = 
\begin{bmatrix} 
\mathbf{0} \\ 
\mathbf{2} \\ 
\mathbf{3} 
\end{bmatrix} \implies 
Y = 
\underbrace{\begin{bmatrix} 
\mathbf{1} & 0 & 0 & 0 \\ 
0 & 0 & \mathbf{1} & 0 \\ 
0 & 0 & 0 & \mathbf{1} 
\end{bmatrix}}_{N \times K}
\in \lbrace 0,1 \rbrace
$$





## Formulation

We want an approach that estimates the K conditionals 

$$f(X_i) = P(y_i=k|X_i), \quad k = 1, 2,\dots, K$$

To do this we will use a generalization of the <ins>Logistic regression</ins> for the case of multiple classes and called **Multinomial Logistic Regression** using **softmax** function

if $h_w(X_i) \approx f(X_i)$, then

$$ h_w(X_i) = \hat{P_w}(y_i=k|X_i) = \hat{p}_ {ik} = \phi(w^T X_i)_ k = \frac{e^{(w^{T}X_i)_ k}}{\sum\limits^{K}_ {j = 1} e^{(w^{T}X_i)_j}} \approx f(X_i)$$

$$k = 1,2,...,K, \quad i = 1,2,...,N$$

Note that $\sum\limits^{K}_ {j = 1} \hat{p}_{ij} = 1, \forall i$




### Maximum likelihood estimation

Let's find a $w$ that maximizes the likelihood of observing the examples. 

Assuming examples are $i.i.d.$ (independent and identically distributed), the likelihood function can be written as:

$$
\prod_{k = 1}^{K} \hat{P_w}(y=k|X) = 
\prod_{n = 1}^{N}  \prod_{k = 1}^{K} \hat{P_w}(y_n=k|X_n) = 
\prod_{n = 1}^{N} \prod_{k = 1}^{K} \Bigg{(} \frac{e^{(w^{T}X_n)_ k}}{\sum\limits^{K}_ {j = 1} e^{(w^{T}X_n)_ j}} \Bigg{)}^{\large y_{nk}} =
\prod_{n = 1}^{N} \prod_{k = 1}^{K} {\phi(w^T X_n)_ k}^{\large y_{nk}}
$$






### Optimization problem

find $w$ that **maximizes**

$$\prod_{n = 1}^{N} \prod_{k = 1}^{K} \Big{(} {\phi(w^T X_n)_ k}^{\large y_{nk}} \Big{)}$$


Or, equivalently, **minimizes**

$$ 
\begin{split}
  L(w) & = -\frac{1}{N} \ln \Bigg{[} \prod_{n = 1}^{N} \prod_{k = 1}^{K} \Big{(} {\phi(w^T X_n)_ k}^{\large y_{nk}} \Big{)} \Bigg{]} \\ 
  & = -\frac{1}{N} \sum_{n = 1}^{N} \sum_{k = 1}^{K} \ln \Big{(} {\phi(w^T X_n)_ k}^{\large y_{nk}} \Big{)} \\
  & = -\frac{1}{N} \sum_{n = 1}^{N} \sum_{k = 1}^{K} y_{nk} \ln \big{(} \phi(w^T X_n)_ k \big{)} \\
  & = -\frac{1}{N} \sum_{n = 1}^{N} \sum_{k = 1}^{K} y_{nk} \ln \hat{p}_{nk} \\
  & \quad \quad \text{Loss Function}
\end{split}
$$








### Gradient Descent

Let's calculate Loss gradient

$$\nabla L(w) = \begin{bmatrix} \dfrac{\partial L}{\partial w_0} & \dfrac{\partial L}{\partial w_1} & ... & \dfrac{\partial L}{\partial w_D} \end{bmatrix} ^{T}$$

where

$$ 
\begin{split} 
\frac{\partial L}{\partial w_{j}} & = 
\frac{\partial}{\partial w_j} \Bigg{[} -\frac{1}{N} \sum_{n = 1}^{N} \sum_{k = 1}^{K} y_{nk} \ln \Bigg{(} \frac{e^{(w^{T}X_n)_ k}}{\sum\limits^{K}_ {j = 1} e^{(w^{T}X_n)_ j}} \Bigg{)} \Bigg{]} \\ 
& = -\frac{1}{N} \frac{\partial}{\partial w_j} \Bigg{[} \sum_{n = 1}^{N} \sum_{k = 1}^{K} y_{nk} (w^{T}X_n)_ k - \sum_{n = 1}^{N} \sum_{k = 1}^{K} y_{nk} \ln \Big{(} \sum\limits^{K}_ {j = 1} e^{(w^{T}X_n)_ j} \Big{)} \Bigg{]} \\ 
& = -\frac{1}{N} \Bigg{[} \sum_{n = 1}^{N} y_{nj} x_{nj} - \sum_{n = 1}^{N} \Bigg{(} \frac{e^{(w^{T}X_n)_ j} x_{nj}}{\sum\limits^{K}_ {j = 1} e^{(w^{T}X_n)_ j}}\Bigg{)} \Bigg{]} \\
& = \frac{1}{N} \Bigg{[} \sum_{n = 1}^{N} -(y_{nj} x_{nj}) + (\hat{p}_ {nj} x_{nj} ) \Bigg{]} \\
& = \frac{1}{N} \Bigg{[} \sum_{n = 1}^{N} (\hat{p}_ {nj} -y_{nj}) x_{nj} \Bigg{]}
\end{split} 
$$




