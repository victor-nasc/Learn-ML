# Linear Regression

Linear regression is a statistical method used for modeling the relationship between a dependent variable (output) and one or more independent variables (inputs) by fitting a linear equation to observed data. The goal is to find the best-fitting line (or hyperplane in multiple dimensions) that minimizes the sum of squared differences between the predicted and actual values (MSE), allowing for prediction and understanding of the linear relationship between variables.

$$MSE = \frac{1}{N} \sum_{i=1}^N (\hat{y_i} - y_i)^2\quad\text{(Loss Function)}$$

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*-y7VmmWRh2SpqHqxLYHSBA.png" alt="Image description" width="700">
  <p>2D Linear Regression Visialisation</p>
</div>




## Analytical Solution

Solution based on matrix algebra. Let's write the residual vector:

$$
\begin{bmatrix}
  \hat{y_1}-y_1 \\
  \hat{y_2}-y_2 \\
  \vdots \\
  \hat{y_N}-y_N
\end{bmatrix}
= \begin{bmatrix}
  \hat{y_1} \\
  \hat{y_2} \\
  \vdots \\
  \hat{y_N}
\end{bmatrix} - 
\begin{bmatrix}
  y_1 \\
  y_2 \\
  \vdots \\
  y_N
\end{bmatrix}
= \begin{bmatrix}
  w^TX_1 \\
  w^TX_2 \\
  \vdots \\
  w^TX_N
\end{bmatrix} - y
= \begin{bmatrix}
  w_0x_{10} + w_1x_{11} + ... + w_dx_{1d} \\ 
  w_0x_{20} + w_1x_{21} + ... + w_dx_{2d} \\ 
  \vdots \\ 
  w_0x_{N0} + w_1x_{N1} + ... + w_dx_{Nd}
\end{bmatrix} - y
= \underbrace{\begin{bmatrix}
  1 & x_{11} & ... & x_{1d} \\ 
  1 & x_{21} & ... & x_{2d} \\ 
  \vdots \\ 
  1 & x_{N1} & ... & x_{Nd}
\end{bmatrix}}_{\huge{X}}
\begin{bmatrix} 
  w_0 \\ 
  w_1 \\ 
  \vdots \\ 
  w_d 
\end{bmatrix} - y = Xw - y
$$

Thus, the vector of residuals can be expressed as:

$$\begin{bmatrix} 
  \hat{y_1}-y_1 \\ 
  \hat{y_2}-y_2 \\ 
  \vdots \\ 
  \hat{y_N}-y_N 
\end{bmatrix} = Xw - y$$

We need the square of the residuals to get MSE:

$$\begin{bmatrix}
  (\hat{y_1}-y_1)^{2} \\ 
  (\hat{y_2}-y_2)^{2} \\ 
  \vdots \\ 
  (\hat{y_N}-y_N)^{2} 
\end{bmatrix}  = (Xw - y)^T(Xw - y) = \lVert Xw - y \rVert^{2}$$

Then

$$MSE = \frac{1}{N} \sum_{i=1}^N (\hat{y_i} - y_i)^2 = \frac{1}{N} \lVert Xw - y \rVert^{2}$$


The solution is the one that minimizes the Loss Function (MSE):

$$\frac{\partial}{\partial w} \Big{[} \frac{1}{N} \lVert Xw - y \rVert^{2} \Big{]}= \frac{2}{N} X^T(Xw - y) = \mathbf{0}$$

$$\implies X^TXw = X^Ty$$ 

$$\implies w = (X^TX)^{-1}X^Ty = X^{\dagger}y$$

where $X^{\dagger} = (X^{T}X)^{-1}X^{T}$ is the pseudo-inverse of $X$

With this, we can calculate $w$ where the derivative of MSE is minimum; this is the optimal solution.

$$ w = X^{\dagger}y$$


## Iterative Solution
### Gradient Descent

Let's calculate MSE's gradient

$$\nabla MSE = \Big{[} \frac{\partial MSE}{\partial w_1}, \frac{\partial MSE}{\partial w_2}, \dots, \frac{\partial MSE}{\partial w_d} \Big{]} ^{T}$$

where

$$
\begin{split}
  \frac{\partial MSE}{\partial w_j} & = \frac{\partial}{\partial w_j} \Big{[} \frac{1}{N} \sum_{i = 1}^{N} (\hat{y_i} - y_i)^{2} \Big{]}\\
                                    & = \frac{1}{N} \sum_{i = 1}^{N} 2(\hat{y_i} - y_i) \frac{\partial}{\partial w_j} (\hat{y_i} - y_i) \\
                                    & = \frac{2}{N} \sum_{i = 1}^{N} (\hat{y_i} - y_i) \frac{\partial}{\partial w_j} \big{(}(w_0 + w_1x_{i1} + ... + w_jx_{ij} + ... + w_dx_{id}) - y\big{)} \\
                                    & = \frac{2}{N} \sum_{i = 1}^{N} (\hat{y_i} - y_i)x_{ij}
\end{split}
$$

Thus

$$\nabla MSE = \frac{2}{N} X^T \boldsymbol{\cdot} (\hat{y} - y)$$
