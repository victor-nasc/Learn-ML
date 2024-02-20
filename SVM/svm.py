import numpy as np


# Victor Nascimento Ribeiro
# 02/2024

class SVM:
    def __init__(self):
        self.w = None
        self.loss_history = []
        


    # Trainig the model
    def fit(self, X, y, lr=0.001, epochs=500, C=1.0):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Convert labels to {1, -1}
        np.where(y == 0, -1, y)
        
        # Get the number of samples (N) and features (d)
        N, d = X.shape
        
        # Initialize weights
        self.w = np.random.rand(d)

        # Batch Gradient Descent
        for _ in range(epochs):
            # Compute hinge loss and gradient
            loss, gradient = self._compute_loss_gradient(X, y, C, N)
            
            # Update weights using gradient descent
            self.w -= lr * gradient
            
            # Store the loss for monitoring convergence
            self.loss_history.append(loss)
            


    # Compute hinge loss and gradient
    def _compute_loss_gradient(self, X, y, C, N):
        hinge_loss = 0.0
        gradient = np.zeros(len(self.w))

        for i in range(N):
            margin = y[i] * np.dot(self.w, X[i])
            if margin < 1:
                hinge_loss += 1 - margin
                gradient -= y[i] * X[i]

        # Add regularization term to loss and gradient
        hinge_loss = C * hinge_loss / N
        loss = 0.5 * np.dot(self.w, self.w) + hinge_loss
        
        gradient = self.w + C * gradient / N

        return loss, gradient


    # Make predictions
    def predict(self, X):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predicted values using learned weights
        y_pred = np.dot(X, self.w) > 0
        
        return y_pred
    
    
    # Evaluate the model
    def evaluate(self, X, y):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predicted values using learned weights
        y_pred = np.dot(X, self.w) > 0
        
        # Compute the accuracy
        accuracy = np.mean(y_pred == y)
        
        return accuracy
    
            




