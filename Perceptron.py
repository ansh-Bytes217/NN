import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation_function = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features) # initialisation of the weights
        self.bias = 0

        # Ensure labels are -1 or 1 for perceptron learning rule
        y_ = np.array([1 if i > 0 else -1 for i in y])

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Calculate the linear combination (net input)
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply activation function
                y_predicted = self.activation_function(linear_output)

                # Update weights and bias based on misclassification
                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, -1)

print("Perceptron class defined.")
