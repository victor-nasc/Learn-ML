# Naive Bayes

Naive Bayes is a simple probabilistic classifier based on Bayes' theorem with the naive assumption of independence between features. It's widely used in text classification, spam filtering, and other tasks, offering fast training and prediction times with good performance, especially for relatively simple datasets.




## Bayes theorem

Bayes' theorem is stated mathematically as the following equation:

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

**Proof:**

Let A and B be events such that $0 \le P(A), P(B) \le 1$

By definition: 

$$P(A|B) = \frac{P(A \cap B)}{P(B)} \implies P(A \cap B) = P(A|B) P(B)$$

Likewise

$$P(B|A) = \frac{P(B \cap A)}{P(A)} \implies P(B \cap A) = P(B|A) P(A)$$

Since ${P(A \cap B)} = {P(B \cap A)}$, then 

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$




## Formulation 

In our case:

$$P(y_i = k|X_i) = \frac{P(X_i|y_i = k) P(y_i = k)}{P(X_i)}$$

$$i=1,2,\dots, N$$

We want an approach that estimates the K conditionals. Assuming examples are $i.i.d.$ (independent and identically distributed)

$$\hat{y_i} = P(y_i = k|X_i) = \frac{P(x_{i1}|y_i = k) P(x_{i2}|y_i = k) \dots P(x_{id}|y_i = k) P(y_i = k)}{P(X_i)}$$

$$k = 1,2,\dots,K$$

Note that $\hat{y_i} \in \mathcal{R}^K$ represents the <ins>probabilities for each class</ins>

Since probabilities range from $0 \leq P \leq 1$, underflow issues may arise when calculating $P(y_i = k|X_i)$. To address this, we can take the logarithm and omit the denominator $P(X)$.

$$\hat{y_i} = P(y_i = k|X_i) = \log\big{(} P(x_{i1}|y_i = k) \big{)} + \log\big{(} P(x_{i2}|y_i = k) \big{)} + \dots + \log\big{(} P(x_{id}|y_i = k) \big{)} + \log\big{(} P(y_i = k) \big{)}$$

Let's model the class probabilities $P(x_{ij}|y_i = k), \quad j=1,2,\dots,d \quad$ using a Gaussian distribution

$$P(x_{ij}|y_i = k) = \frac{1}{\sqrt{2\pi{\sigma}_{k}^2}} \mathrm{exp}\left(-\frac{(x-\mu_k)^2}{2{\sigma}_k^2}\right)$$

With all of that we can estimate $\hat{y_i}$ for each class

The predicted class of $X_i$ is $argmax(\hat{y_i})$

