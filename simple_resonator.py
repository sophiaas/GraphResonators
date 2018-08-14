import numpy as np

n = 50
d = 25
X = ((np.random.randn(n, d) < 0) * 2) - 1
Y = ((np.random.randn(n, d) < 0) * 2) - 1
i1 = np.random.randint(d)
i2 = np.random.randint(d)
b = X[:, i1] * Y[:, i2]
X_hat = 2*(np.random.randn(n) < 0) - 1 #for every node in the graph
Y_hat = 2*(np.random.randn(n) < 0) - 1
max_steps = 5
for i in range(max_steps):
    X_hat = np.dot(np.dot(b * Y_hat, X), X.T)/n
    X_hat = 2*(X_hat > 0) - 1
    #X_hat /= np.abs(X_hat)
    Y_hat = np.dot(np.dot(b * X_hat, Y), Y.T)/n
    #Y_hat /= np.abs(Y_hat)
    Y_hat = 2*(Y_hat > 0) - 1
i1_hat = np.argmax(np.abs(np.dot(X.T, X_hat)))
i2_hat = np.argmax(np.abs(np.dot(Y.T, Y_hat)))
print(i1, i1_hat, i2, i2_hat)
