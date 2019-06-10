import numpy as np
import math
import matplotlib.pyplot as plt

class NN():
    def __init__(self, n_in, n_l1, n_l2, n_out=1):
        self.w1 = np.random.randn(n_l1, n_in) * np.sqrt(1/n_l1)
        self.b1 = np.zeros((n_l1, 1))

        self.w2 = np.random.randn(n_l2, n_l1) * np.sqrt(1/n_l2)
        self.b2 = np.zeros((n_l2, 1))

        self.w3 = np.random.randn(n_out, n_l2) * np.sqrt(1/n_out)
        self.b3 = np.zeros((n_out, 1))

    def feed_forward(self, x):
        self.x = x
        self.n = x.shape[1]
        self.z1 = np.matmul(self.w1, x) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.matmul(self.w3, self.a2) + self.b3
        self.yhat = self.sigmoid(self.z3)
        return self.yhat

    def sigmoid(self, z):
        try:
            return 1/(1 + np.exp(-z))
        except OverflowError:
            return 0

    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_prime(self, z):
        return np.where(z > 0., 1., 0.)

    def ce_loss(self, x, y):
        y = y.reshape(1, -1)
        yhat = self.feed_forward(x)
        return -(np.multiply(y, np.log(yhat)) + np.multiply((1 - y), np.log(1 - yhat)))

    def mse_loss(self, x, y):
        y = y.reshape(1, -1)
        yhat = self.feed_forward(x.T)
        return 2 * np.sum(np.power((y - yhat), 2)) / x.shape[0]

    def accuracy(self, x, y):
        y = y.reshape(1, -1)
        yhat = np.where(self.feed_forward(x.T) >= .5, 1, 0)
        pcc = 1 - (np.sum(np.abs(y-yhat)) / x.shape[0])
        return pcc

    def back_propagate(self, y):
        y = y.reshape(1, -1)
        z3_grad = (self.yhat - y) * self.sigmoid_prime(self.z3)
        self.w3_grad = np.matmul(z3_grad, self.a2.T) / self.n
        self.b3_grad = np.sum(z3_grad) / self.n

        z2_grad = np.matmul(self.w3.T, z3_grad) * self.relu_prime(self.z2)
        self.w2_grad = np.matmul(z2_grad, self.a1.T) / self.n
        self.b2_grad = np.sum(z2_grad, axis=1, keepdims=True)

        z1_grad = np.matmul(self.w2.T, z2_grad) * self.relu_prime(self.z1)
        self.w1_grad = np.matmul(z1_grad, self.x.T) / self.n
        self.b1_grad = np.sum(z1_grad, axis=1, keepdims=True)

    def update_weights(self, lr):
        self.w3 = self.w3 - self.w3_grad * lr
        self.b3 = self.b3 - self.b3_grad * lr

        self.w2 = self.w2 - self.w2_grad * lr
        self.b2 = self.b2 - self.b2_grad * lr

        self.w1 = self.w1 - self.w1_grad * lr
        self.b1 = self.b1 - self.b1_grad * lr

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size, log=True):
        train_losses = []
        val_losses = []
        for epoch in range(epochs):
            shuffle_order = np.random.permutation(X_train.shape[0])
            mb_X_train = np.array_split(X_train[shuffle_order], batch_size)
            mb_y_train = np.array_split(y_train[shuffle_order], batch_size)
            for x, y in list(zip(mb_X_train, mb_y_train)):
                self.feed_forward(x.T)
                self.back_propagate(y)
                self.update_weights(lr)
            train_loss = self.mse_loss(X_train, y_train)
            val_loss = self.mse_loss(X_val, y_val)
            if log:
                print('epoch #{}\n\ttrain loss: {:4f}\n\tvalidation loss: {:4f}'.format(epoch, train_loss, val_loss))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        return train_losses, val_losses

    def plot(self, train_losses, val_losses, h=20, w=10):
        x = list(range(len(train_losses)))
        plt.figure(figsize=(h, w))
        plt.xlabel('epoch', fontsize=18)
        plt.ylabel('loss', fontsize=18)
        plt.plot(x, train_losses, 'r', label='training data')
        plt.plot(x, val_losses, 'g', label='validation data')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    np.random.seed(3)
    X_train = np.array([[1,0],
                  [4,5],
                  [-1,4],
                  [6,5],
                  [1,1],
                  [-2,3]])
    y_train = np.array([1,0,1,0,1,0])

    X_val = np.array([[.99, 0.01],
                        [4, 5.1]])
    y_val = np.array([0,1])

    nn1 = NN(X_train.shape[1], 225, 200, n_out=1)
    train_losses, val_losses = nn1.fit(X_train, y_train, X_val, y_val, 10000, .0001, 3)