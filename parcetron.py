import numpy as np


class Praceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = np.random.rand()


    def step_function(self, x):
        return 1 if x >= 0 else 0


    def predict(self, x):
        weight_sum = np.dot(x, self.weights)
        return self.step_function(weight_sum)


    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1])
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                y_pred = self.predict(x)
                error = y[i] - y_pred
                
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error


X = np.array([
    [50, 30],  # Клиент 1 (не одобрить)
    [80, 70],  # Клиент 2 (одобрить)
    [20, 50],  # Клиент 3 (не одобрить)
    [90, 85],  # Клиент 4 (одобрить)
    [10, 25],  # Клиент 5 (не одобрить)
    [95, 10],  # Клиент 6 (не одобрить)
])
y = np.array([0, 1, 0, 1, 0, 0])

perceptron = Praceptron()


perceptron.fit(X, y) 


new_clients = np.array([
    [60, 50],  # Клиент 7
    [20, 50]   # Клиент 8
])
for person in new_clients:
    print(person)
    print(perceptron.predict(person))