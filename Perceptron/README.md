# Perceptron Learning Algorithm

The Perceptron Learning Algorithm (PLA) is a simple algorithm for learning a binary classifier, particularly for **linearly separable** datasets

A linearly separable dataset is one in which the instances of different classes can be perfectly separated by a linear decision boundary, typically a straight line in two dimensions or a hyperplane in higher dimensions. This means that it's possible to draw a line or a plane that cleanly separates all data points of one class from those of another, making classification straightforward for algorithms like the Perceptron.



<div align="center">
  <img src="https://i0.wp.com/cmdlinetips.com/wp-content/uploads/2021/02/Linearly_Separable_Data_Example.png?fit=539%2C234&ssl=1" alt="Image description" width="500">
  <p>Problems Separability</p>
</div>


## Hypotesis

Our target $f$ is such that $f(X) \in \lbrace -1,+1 \rbrace $

$$\hat{y} = h_w(X) = sign \lbrace (w^T X) \rbrace \quad i = 1,2,...,N$$


<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:916/1*_xbbcmL06_nXADsGK1TAug.png" alt="Image description" width="500">
  <p>Graphical Visualization</p>
</div>

## Algorithm

**Input:** $\mathcal{D} = \lbrace (X_i, y_i) \in \mathcal{R}^d \times \lbrace −1, +1 \rbrace : i = 1, \dots, N \rbrace $

**Output:** $w \in \mathcal{R}^{d+1}$ 

$w \leftarrow \text{small random values}$

**while** there exists $(X_i,y_i) \in \mathcal{D}$ such that $sign(w^T X_i) \neq y_i$ **do**

$\quad w \leftarrow w + y_i X_i$

**end while**

**return $w$**
 



## Convergence 
To se a detailed proof for the algorithm convergence see: [Shivaram Kalyanakrishnan](https://www.cse.iitb.ac.in/~shivaram/teaching/old/cs344+386-s2017/resources/classnote-1.pdf)



