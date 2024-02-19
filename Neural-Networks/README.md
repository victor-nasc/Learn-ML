# Feedforwards Neural Networks

Feedforward Neural Networks are models capable of achieving great success in various applications. They are excellent candidates for problems involving learning because they can efficiently approximate complex target functions. Let's start with the basics to understand how networks work and then explain how their training is done using data.


<div align="center">
  <img src="https://i.imgur.com/eFvsL0l.jpeg" alt="Image description" width="500">
  <p>Figure 1: Feedforward Neural Network Representation</p>
</div>






## 1. Neurons

Neurons, also called nodes, are the fundamental structure for the functioning of networks.

<div align="center">
  <img src="https://i.imgur.com/lRU5rrG.png" alt="Image description" width="450">
  <p>Figure 2: Artificial Neuron Representation</p>
</div>

Remember that in our formulation $x_0 = 1$ and $w_0 = b$ 

They are responsible for receiving an input, associating weights to it, as mentioned in the main repository page, and propagating their output through an **activation function** $(\mathbf{\varphi})$. These functions introduce non-linearity to the network, enabling it to learn complex patterns in the data, and play a crucial role in shaping the output of neural networks, influencing their ability to learn and generalize from the input data.

<div align="center">
  <img src="https://i.imgur.com/0QdWDv1.png" alt="Image description" width="500">
  <p>Figure 3: Activation Functions Examples</p>
</div>





## 2. Structure and Notation

Neural networks are organized into layers indexed by $l = (0,1,\dots,L)$, where $L$ denotes the total number of layers. In Figure 1, $L = 3$. The **input layer**, represented in blue, corresponds to $l = 0$, while the **output layer**, shown in green, corresponds to $l = L$. The intermediary layers, indexed by $l = (1,\dots,L-1)$, are referred to as **hidden layers**. We represent the layer number as $^{(l)}$. Each layer comprises $d^{(l)}$ nodes, visualized as circles. For instance, in Figure 1, $d^{(2)} = 6$.

Now let's examine a single node $j$ from any layer $l$ of the network in detail. It's worth noting that $0 \leq j < d^{(l)}$.

<div align="center">
  <img src="https://i.imgur.com/VOgCwHU.png" width="400">
  <p>Figure 4: Diagram of an arbitrary node $j$</p>
</div>


The input $x^{(l-1)}_{ji}$ represents the $i$-th input of node $j$, and the same applies to the weight $w^{(l)} _{ji}$. The reason why $x$ belongs to $^{(l-1)}$ instead of $^{(l)}$ might be obscure for now, but it will make sense in the next topic.

We have

$$s^{(l)}_ j = w^{(l)}_ {j0} + w^{(l)}_ {j1} x^{(l-1)}_ 1 + \dots + w^{(l)}_ {j{d^{(l)}}} x^{(l-1)}_ {d^{(l)}}$$

and finally $z^{(l)}_j$ is the final output of neuron 

$$z^{(l)}_j = \varphi (s^{(l)}_j)$$ 


Let us also denote 

$$w^{(l)} = \lbrace (w^{(l)}_{ji}) : j=0, \dots ,d^{(l-1)} \land i=0,\dots,d^{(l)} \rbrace$$

$$X^{(l)} = \lbrace (x^{(l)} _{ji}) : j=0, \dots ,d^{(l-1)} \land i=0,\dots,d^{(l)} \rbrace$$

$$W = (w^{(1)}, w^{(2)}, \dots, w^{(L)})$$

which we will use later.


In neural networks, our family of functions is $\mathcal{H}_{nn}$, representing the architecture of our neural network according to the dimensions of $L$ and $d$. In this case, $h_W \in \mathcal{H} _{nn}$ is computed through **Forward Propagation**.







## Forward Propagation

Feedforward networks are fully connected, meaning all nodes in a layer $l$ receive as input all outputs from the node $l-1$. To compute $h$, it's necessary to process and propagate the input signal until the output layer, referred to as the **forward pass**. Each <ins>neuron receives as input the activation from neurons in the previous layer and propagates its output to be used as input by neurons in the subsequent layers</ins>. The forward pass is depicted in the diagram below. It's important to note that $X^{(l)} = z^{(l)}$.


$$ 
\large{X^{(0)} \xrightarrow{w^{(1)}} s^{(1)} \xrightarrow{\varphi} z^{(1)} = X^{(1)}  \xrightarrow{w^{(2)}} s^{(2)} \xrightarrow{\varphi} z^{(2)} = X^{(2)} \xrightarrow{} \dots \xrightarrow{} X^{(L-1)} \xrightarrow{w^{(L)}} s^L \xrightarrow{\varphi} z^L = h_W(X)}
$$

It's crucial to understand how each weight influences the final outcome of the forward pass; this dependency relationship is essential to understand the training algorithm.

Recall that the Loss Lunction is a function of $W$, and with $g(X) = \hat{y}$ calculated, we can utilize it to evaluate the value of our Loss Function, subsequently adjusting the weights to minimize it. Let's delve into how this process is accomplished in the next section.





## Backpropagation Algorithm

Backpropagation is the algorithm used to train neural networks. Recalling our goal is to use **gradient descent** to adjust the weights of the network in order to minimize a cost function. For a single neuron, this is a simple task, but in neural networks, the algorithm becomes much more complex.

Therefore, we want to calculate

$$
\nabla L(W) = \Big{[}\frac{\partial L}{\partial w^{(1)}}, \frac{\partial L}{\partial w^{(2)}}, \dots, \frac{\partial L}{\partial w^{(L)}}\Big{]}
$$

The algorithm exploits the fact that the gradient of the cost function of a neural network can be calculated iteratively, starting from the weights of the final layers of the network and backpropagating these values to the earlier layers.

For this explanation we will use Mean Squared Error (MSE)

$$L(W) =  \frac{1}{N} \sum_{i=1}^N (\hat{y_i} - y_i)^2$$

### Output Layer - $L$

By the chain rule, with respect to any weight $w^{(L)}_{ji}$ in the output layer, we have that

$$
\frac{\partial L}{\partial w^{(L)}_{ji}} = \frac{\partial L}{\partial z^{(L)}_j} \frac{\partial z^{(L)}_j}{\partial s^{(L)}_j} \frac{\partial s^{(L)}_j}{\partial w^{(L)} _{ji}} = \underbrace{- \frac{2}{N} (y_j - \hat{y}_j) \varphi'(s_j)} _{\Large{\delta^{(L)}_j}} x^{(L-1)}_j
$$

The value of each $\delta^{(L)}_j$ $: j=(0, 1, \dots,d^{(L)})$ is called **sensitivity**, and it will be backpropagated to the previous layer. 

$$\delta^{(L)}_j = \dfrac{\partial L}{\partial z^{(L)}_j} \dfrac{\partial z^{(L)}_j}{\partial s^{(L)}_j}$$



### Hidden Layers - $\lbrace 0,1,\dots,L-1 \rbrace$

Since we are beginning to calculate the gradients of the final layers and backpropagating them to earlier layers, with respect to any weight $w^{(l)}_{kj}$, where $l \in \lbrace 0,1,\dots,L-1 \rbrace$, we will utilize the previously calculated $\textcolor{red} {\delta^{(l+1)}_j}$

$$
\frac{\partial L}{\partial w^{(l)}_ {kj}} = \sum_{j=0}^{d^{(l+1)}} \Big{(} \frac{\partial L}{\partial z^{(l+1)}_j} \frac{\partial z^{(l+1)}_j}{\partial s^{(l+1)}_j} \frac{\partial s^{(l+1)}_j}{\partial z^{(l)}_j} \Big{)} \frac{\partial z^{(l)}_k}{\partial s^{(l)}_k} \frac{\partial s^{(l)}_k}{\partial w^{(l)} _{kj}} \underbrace{\sum _{j=0}^{d^{(l+1)}} \big{(} {\textcolor{red} {\delta^{(l+1)}_j}} w _{kj} \big{)} \varphi'(s^{(l)}_k)} _{\Large{\textcolor{blue} {\delta^{(l)}_k}}}  x^{(l)}_k
$$


In the process of computing derivatives for all hidden layers, we utilize the backpropagation algorithm to propagate the computed sensitivities back to the first layer of the network. This propagation through all layers yields derivatives with respect to all weights of the network, thus enabling the calculation of the gradient vector of the function. Subsequently, leveraging the gradient descent algorithm, we can minimize the cost function.



