# Logistic Regression

Logistic regression is a statistical model used for **binary classification**, predicting the probability of an outcome belonging to one of two classes. It's named "logistic" because it employs the logistic function to map a linear combination of input features to a probability score, which is then transformed into a binary decision boundary.



## Hypotesis

Since our target $f$ is such that $0 \leq f(X) \leq 1$, let us consider
hypotheses of the same type: 

$$\hat{y} = h_w(X) = \theta(w^{T}X) \approx f(X)$$

where 

$$ \theta(z) = \frac{e^{z}}{e^{z}+1} = \frac{1}{1+e^{-z}}$$

$$ 0 \leq \theta(z) \leq 1 \implies 0 \leq h_w(X) \leq 1$$ 

known as **Sigmoid Activation Function**


<div align="center">
  <img src="https://vitalflux.com/wp-content/uploads/2020/05/sigmoid-function-plot-1.png" alt="Image description" width="400">
  <p>Sigmoid Function</p>
</div>





## Formulation with $y_i \in \{-1,1\}$


Assume our target is: $f(X_i) = \hat{P_w}(y_i = +1|X_i), \quad i=1,2,\dots,N$

if $h_w(X_i) \approx f(X_i)$, then

$$ \hat{P_w}(y_i |X_i) = \begin{cases}
        h_w(X_i) & \text{, if } y_i = +1\\
        1 - h_w(X_i) & \text{, if } y_i = -1
        \end{cases} $$
        
should be a good estimate of $\hat{P_w}(y_i |X_i)$

note that: 

$$1 - \theta(z) = 1 - \frac{e^{z}}{e^{z}+1} = \frac{e^{z} + 1 - e^{z}}{e^{z}+1} = \frac{1}{e^{z}+1} = \theta(-z)$$

thus 

$$1 − \theta(z) = \theta(−z)$$

then we can write $\   \hat{P_w}(y |X) = \theta(y w^T X)$





### Maximum likelihood estimation

Let's find a $w$ that **maximizes** the likelihood of observing the examples. 

Assuming examples are $i.i.d.$ (independent and identically distributed), the likelihood function can be written as:

$$\prod_{i = 1}^{N} \hat{P_w}(y_i |X_i) = \prod_{i = 1}^{N} \theta(y_i w^T X_i)$$






### Optimization problem
We want to find $w$ that **maximizes** 

$$\prod_{i = 1}^{N} \theta(y_iw^{T}X_i)$$


Or, equivalently, **minimizes**

$$ 
\begin{split}
  L(w) & = -\frac{1}{N} \ln{\Big(\prod_{i = 1}^{N} \theta(y_iw^{T}X_i)\Big)} \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} \ln{\Big(\theta(y_iw^{T}X_i)\Big)} \\ 
  & = \frac{1}{N} \sum_{i = 1}^{N} \ln{\Big(\frac{1}{\theta(y_iw^{T}X_i)}\Big)} \\ 
  & = \frac{1}{N} \sum_{i = 1}^{N} \ln{\Big(1 + e^{-y_iw^{T}X_i}}\Big) \\
  & \quad \quad \quad \text{ Loss Function}
  \end{split}
$$






### Gradient Descent

Let's calculate Loss gradient

$$\nabla L(w) = \begin{bmatrix} \dfrac{\partial L}{\partial w_0} & \dfrac{\partial L}{\partial w_1} & ... & \dfrac{\partial L}{\partial w_d} \end{bmatrix} ^{T}$$

where

$$
\begin{split} 
  \nabla L(w)
  & = \frac{\partial}{\partial w} \Big{[} \frac{1}{N} \sum_{i = 1}^{N} \ln{\Big(1 + e^{-y_iw^{T}X_i}}\Big) \Big{]}\\ 
  & = \frac{1}{N} \sum_{i = 1}^{N} -y_ix_i\frac{e^{-y_iw^{T}X_i}}{(1 + e^{-y_iw^{T}X_i})} \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} y_ix_i\frac{1}{(1 + e^{y_iw^{T}X_i})} 
\end{split} 
$$










## Formulation with $y_i \in \{0,1\}$


Note that for $y_i \in \{ 0,1 \}$, we can write

$$
\begin{split}  
  \hat{P_w}(y_i |X_i) 
  & = \hat{P_w}(y_i = 1|X_i)^{y_i} \hat{P_w}(y_i = 0|X_i)^{1−y_i} \\ 
  & = \hat{P_w}(y_i = 1|X_i)^{y_i}\big[1 − \hat{P_w}(y_i = 1|X_i)\big]^{1−y_i} 
\end{split} 
$$





### Maximum likelihood estimation

As well as in the interpretation $\   y \in \{-1,1\}$

Let's find a $w$ that **maximizes** the likelihood of observing the examples. 

Assuming examples are $i.i.d.$ (independent and identically distributed), the likelihood function can be written as:

$$
\begin{split} 
  \prod_{i=1}^N \hat{P_w}(y_i|X_i) & =  
  \prod_{i=1}^N \hat{P_w}(y_i = 1|X_i)^{y}\big[1 − \hat{P_w}(y_i = 1|X_i)\big]^{1−y_i} \\ 
  & \approx \prod_{i=1}^N \big(\theta(w^{T}X_i)\big)^{y_i}\big(1-\theta(w^{T}X_i)\big)^{1-y_i}\\ 
  & = \prod_{i=1}^N \hat{y_i}^{y_i} (1 - \hat{y_i})^{1-y_i} 
\end{split} 
$$

Remember that we defined $\hat{y_i} = \theta(w^{T}X_i)$ for this problem






### Optimization problem

Following the same steps seen before, we have that the maximization of the function above is equivalent to minimizing

$$ 
\begin{split} 
  L(w) & = 
  -\frac{1}{N} \ln{\prod_{i=1}^N \hat{y_i}^{y_i} (1 - \hat{y_i})^{1-y_i}}  \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} \ln{\Big(\hat{y_i}^{y_i} (1 - \hat{y_i})^{1-y_i})\Big)} \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} \ln{\hat{y_i}^{y_i}} + \ln{\big((1 - \hat{y_i})^{1-y_i}\big)} \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} y_i\ln{\hat{y_i}} + (1-y_i) \ln{(1 - \hat{y_i})} \\
  & \quad \text{ Binary Cross-Entropy Loss Function}
\end{split}
$$

Let's analyze the meaning of this loss function. When $y_i = 1$ (indicating $X_i$ is in the positive class) and $\hat{y_i}$ is close to 1, we have $\ln \hat{y_i} \approx 0$, making $y_i \ln \hat{y_i} \approx 0$. Simultaneously, the second term $(1 - y_i) \ln(1 - \hat{y_i})$ is equal to zero. When $y_i = 0$ (indicating $X_i$ is in the negative class) and $\hat{y_i}$ is close to 0, we encounter a similar situation: the term $y_i \ln \hat{y_i}$ is equal to 0, and $1 - \hat{y_i}$ is close to 1, making the second term close to zero. On the other hand, if $y_i$ and $\hat{y_i}$ are not close, one of the terms can have a large value. Therefore, when we minimize the function, we are effectively forcing $\hat{y_i}$ to approach $y_i$.


### Gradient Descent

First let's calculate the derivative of $\hat{y_i} = \theta(w^{T}X_i)$

$$
\begin{split} 
  \frac{\partial\theta(w^{T}X_i)}{\partial w_j} 
  & = \frac{\partial}{\partial w_j} \Big(\frac{1}{1+e^{-w^{T}X_i}}\Big) \\ 
  & = \frac{e^{-w^{T}X_i}}{(1 + e^{-w^{T}X_i})^{2}}  x_{ij}\\ 
  & = \frac{1}{(1 + e^{-w^{T}X_i})}  \frac{e^{-w^{T}X_i}}{(1 + e^{-w^{T}X_i})}  x_{ij}\\ 
  & = \frac{1}{(1 + e^{-w^{T}X_i})}  \frac{(1 + e^{-w^{T}X_i}) - 1}{(1 + e^{-w^{T}X_i})}  x_{ij}\\ 
  & = \frac{1}{(1 + e^{-w^{T}X_i})}  \biggl(\frac{1 + e^{-w^{T}X_i}}{(1 + e^{-w^{T}X_i})} - \frac{1} {1+e^{-w^{T}X_i}} \biggl)  x_{ij}\\ 
  & = \frac{1}{(1 + e^{-w^{T}X_i})}  \biggl(1 - \frac{1} {1+e^{-w^{T}X_i}} \biggl)  x_{ij}\\ 
  & = \theta(w^{T}X_i) (1 - \theta(w^{T}X_i))  x_{ij} \\ 
  & = \hat{y_i} (1 - \hat{y_i}) x_{ij} \\
  &
\end{split} $$


Now the Binary Cross-Entropy's Gradient

$$\nabla L(w) = \begin{bmatrix} \dfrac{\partial L}{\partial w_0} & \dfrac{\partial L}{\partial w_1} & ... & \dfrac{\partial L}{\partial w_d} \end{bmatrix} ^{T}$$

where

$$
\begin{split} 
  \frac{\partial L}{\partial w_j} & = 
  -\frac{1}{N} \sum_{i = 1}^{N} \frac{\partial}{\partial w_j} \Big(y_i\ln{\hat{y_i}} + (1-y_i) \ln{(1 - \hat{y_i})} \Big) \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} y_i\frac{\hat{y_i} (1 - \hat{y_i}) x_{ij}}{\hat{y_i}} + (1-y_i) \frac{\big(-\hat{y_i} (1 - \hat{y_i}) x_{ij}\big)}{(1 - \hat{y_i})} \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} y_i (1 - \hat{y_i}) x_{ij} + (1-y_i) (-\hat{y_i}) x_{ij} \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} \big(y_i - \hat{y_i} y_i -\hat{y_i} + \hat{y_i} y_i \big)x_{ij} \\ 
  & = -\frac{1}{N} \sum_{i = 1}^{N} \big(y_i -\hat{y_i} \big)x_{ij}
\end{split}
$$



