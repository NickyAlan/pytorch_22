import numpy as np
import matplotlib.pyplot as plt


class linearResgression :
    def __init__(self, X, y):
        assert X.shape == y.shape
        self.ones = np.ones(shape=(X.shape[0], 1))
        self.X = np.append(self.ones, X, axis=1)
        self.y = y
        self.W = np.dot(np.linalg.pinv(np.dot(self.X.T, self.X)), np.dot(self.X.T, self.y))

    def predict(self, X_test) :
        y_hat = self.W[1]*X_test + self.W[0] # y = WX + b
        return y_hat

    def weight(self) :
        return self.W

    def plot_pred(self, X_test, y_test) :
        y_hat = self.predict(X_test)
        plt.scatter(X_test, y_hat, c='blue', s=5, label='predict')
        plt.scatter(X_test, y_test, c='red', s=5, label='true')
        plt.legend()
        plt.show()

if __name__ == '__main__' :
    n = 100
    X = np.random.uniform(0,5,size=(n,1))
    y  = 5*X + np.random.randn(n,1) * 0.2
    model = linearResgression(X, y)
    model.plot_pred(X, y)
