# Linear Regression

#### Author: Victor Nascimento Ribeiro - January 2024

Linear regression is a statistical method used for modeling the relationship between a dependent variable (output) and one or more independent variables (inputs) by fitting a linear equation to observed data. The goal is to find the best-fitting line (or hyperplane in multiple dimensions) that minimizes the sum of squared differences between the predicted and actual values (MSE), allowing for prediction and understanding of the linear relationship between variables.


<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*-y7VmmWRh2SpqHqxLYHSBA.png" alt="Image description" width="700">
  <p>2D Linear Regression Visialisation</p>
</div>



## Formulation

- $N$: Number os samples in the dataset
- $d$: Number of features in each data instance

$$X = (X_1, X_2, ... , X_N)\quad X_i \in \mathbb{R}^d\quad\text{(Dataset)}$$

$$X_i = (x_1, x_2, ... , x_d)\quad x_i \in \mathbb{R}\quad\text{(Data instance)}$$

$$y_i \in \mathbb{R}\quad\text{(Target)}\quad$$

$$w = (w_1, w_2, ... , w_d)\quad w_i \in \mathbb{R}\quad\text{(Weights)}$$

$$\hat{y_i} = w_1X_1 + \dots + w_NX_N + b\quad\text{(Prediction)}$$

$$MSE = \frac{1}{N} \sum_{i=1}^N (\hat{y_i} - y_i)^2\quad\text{(Loss Function)}$$

#### Artificial component to simplify notatiton:
$$ X_i = (\mathbf{1}, x_1, x_2, ... , x_d)\quad(x_0 = 1)$$

$$ w = (\mathbf{w_0}, w_1, w_2, ... , w_d)\quad (w_0 \text{ is the bias } b)$$

Thus $\hat{y_i} = w^TX_i $





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

$$\frac{\partial}{\partial w} \frac{1}{N} \lVert Xw - y \rVert^{2} = \frac{2}{N} X^T(Xw - y) = \mathbf{0}$$

$$\implies X^TXw = X^Ty$$ 

$$\implies w = (X^TX)^{-1}X^Ty = X^{\dagger}y$$

where $X^{\dagger} = (X^{T}X)^{-1}X^{T}$ is the pseudo-inverse of $X$

With this, we can calculate $w$ where the derivative of MSE is minimum; this is the optimal solution.

$$ w = X^{\dagger}y$$


## Iterative Solution
### Gradient Descent

Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or of the approximate gradient) of the function at the current point. The learning rate, denoted by $\alpha$, determines the size of these steps, influencing the convergence speed and stability of the algorithm. 

Let's calculate MSE's gradient


$$ \frac{\partial MSE}{\partial w_j} = \frac{\partial}{\partial w_j} \frac{1}{N} \sum_{i = 1}^{N} (\hat{y_i} - y_i)^{2} $$

$$ = \frac{1}{N} \sum_{i = 1}^{N} 2(\hat{y_i} - y_i) \frac{\partial}{\partial w_j} (\hat{y_i} - y_i) $$

$$ = \frac{2}{N} \sum_{i = 1}^{N} (\hat{y_i} - y_i) \frac{\partial}{\partial w_j} ((w_0 + w_1x_{i1} + ... + w_jx_{ij} + ... + w_dx_{id}) - y) $$

$$ = \frac{2}{N} \sum_{i = 1}^{N} (\hat{y_i} - y_i)x_{ij} = \frac{2}{N} X^T \boldsymbol{\cdot} (\hat{y} - y)$$

### 1: Stochastic Gradient Descent

**w** &larr; small random value 

**repeat:**
   - **for all** $(x,y)$ **do**
      - $\hat{y} = w^{T}x$
      - $w_j = w_j - \frac{1}{N} \alpha (\hat{y} - y) x_j\quad j = 0, 1, 2, ... , d$
   - **end for**
     
**until** number of iterations == epochs

return $w$


### 2: Batch Gradient

**w** &larr; small random value 

**repeat:**
   - $\Delta w_j = 0\quad j = 0, 1, 2, ... ,d$
   - **for all** $(x,y)$ **do**
      - $\hat{y} = w^{T}x$
      - $\Delta w_j = \Delta w_j - \frac{1}{N} \alpha (\hat{y} - y) x_j\quad j = 0, 1, 2, ... , d$
   - **end for**
   - $w_j = w_j + \alpha \Delta w_j\quad j = 0, 1, 2, ..., d $
     
**until** number of iterations == epochs

return $w$


### 3: Mini-Batch Gradient

Mini-batch gradient descent is an optimization algorithm that combines aspects of both stochastic and batch gradient descent. It divides the training dataset into small batches and updates the model parameters based on the average gradient of each batch. This approach strikes a balance between the efficiency of stochastic gradient descent and the stability of batch gradient descent, making it suitable for large datasets.


## References
- https://work.caltech.edu/telecourse (lecture 3)
 - Abu-Mostafa, Yaser S., Magdon-Ismail, Malik and Lin, Hsuan-Tien. Learning From Data. : AMLBook, 2012.
