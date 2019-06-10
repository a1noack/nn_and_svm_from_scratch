import numpy as np
import matplotlib.pyplot as plt

class SVM():
    def __init__(self, n, C, kernel="rbf", sigma=3, p=2):
        self.C = C
        self.p = p
        self.sigma = sigma
        self.m = n
        self.alphas = np.array([0.]*self.m)
        self.b = 0
        if kernel == "rbf":
            self.kernel = self.rbf
        elif kernel == "polynomial":
            self.kernel = self.polynomial
        else:
            self.kernel = np.dot

    def rbf(self, x_i, x_j):
        k = np.exp(-np.power(np.linalg.norm(x_i - x_j), 2) / (2 * self.sigma ** 2))
        return k

    def polynomial(self, x_i, x_j):
        return np.power(1 + np.dot(x_i, x_j), self.p)

    def predict(self, X):
        total = 0
        for j in range(self.m):
            z = self.alphas[j] * self.y_train[j] * self.kernel(self.X_train[j], X)
            total += z
        return total + self.b

    def accuracy(self, X, y):
        wrong = 0
        for i in range(X.shape[0]):
            yhat = self.predict(X[i])
            if yhat > 0:
                yhat = 1
            else:
                yhat = -1
            if yhat != y[i]:
                wrong += 1
        return 1 - wrong / X.shape[0]

    def mse_loss(self, X, y):
        yhats = []
        for i in range(X.shape[0]):
            yhat = self.predict(X[i])
            yhats.append(yhat)
        return 2 * np.sum(np.power((y - yhats), 2)) / X.shape[0]

    def compute_lh(self, alpha_i, alpha_j, y_i, y_j):
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_j + alpha_i - self.C)
            H = min(self.C, alpha_j + alpha_i)
        return L,H

    def fit(self, X_train, y_train, X_val, y_val, tol, val_acc_break, max_epochs, log=True):
        self.tol = tol
        self.X_train = X_train
        self.y_train = y_train
        val_acc = 0

        train_losses = []
        val_losses = []

        z = 0
        while val_acc < val_acc_break and z < max_epochs:
            # alphas_changed = 0
            for i in range(self.m):
                x_i = self.X_train[i]
                y_i = self.y_train[i]
                alpha_i = self.alphas[i]
                E_i = self.predict(x_i) - y_i
                if (y_i * E_i < -self.tol and alpha_i < self.C) or (y_i * E_i > self.tol and alpha_i > 0):
                    j = i
                    while j == i:
                        j = np.random.randint(0, self.m)
                    x_j = self.X_train[j]
                    y_j = self.y_train[j]
                    alpha_j = self.alphas[j]
                    E_j = self.predict(x_j) - y_j

                    alpha_i_old = alpha_i
                    alpha_j_old = alpha_j

                    L,H = self.compute_lh(alpha_i, alpha_j, y_i, y_j)
                    if L == H:
                        continue

                    eta = 2 * np.dot(x_i, x_j) - np.dot(x_i, x_i) - np.dot(x_j, x_j)

                    alpha_j = alpha_j - (y_j * (E_i - E_j)) / eta
                    if alpha_j > H:
                        alpha_j = H
                    elif alpha_j < L:
                        alpha_j = L
                    self.alphas[j] = alpha_j

                    if abs(alpha_j - alpha_j_old) < self.tol:
                        continue

                    alpha_i = alpha_i + y_i*y_j*(alpha_j_old - alpha_j)
                    self.alphas[i] = alpha_i

                    b_1 = self.b - E_i - y_i * (alpha_i - alpha_i_old) * self.kernel(x_i, x_i) \
                          - y_j * (alpha_j - alpha_j_old) * self.kernel(x_i, x_j)
                    b_2 = self.b - E_j - y_i * (alpha_i - alpha_i_old) * self.kernel(x_i, x_j) \
                          - y_j * (alpha_j - alpha_j_old) * self.kernel(x_j, x_j)

                    if 0 < alpha_i < self.C:
                        if 0 < alpha_j < self.C:
                            self.b = (b_1 + b_2) / 2
                        else:
                            self.b = b_1
                    elif 0 < alpha_j < self.C:
                        self.b = b_2

                    # alphas_changed += 1
            train_loss = self.mse_loss(self.X_train, self.y_train)
            val_loss = self.mse_loss(X_val, y_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_acc = self.accuracy(self.X_train, self.y_train)
            val_acc = self.accuracy(X_val, y_val)

            if log:
                print("epoch {}\n\ttrain loss: {:4f}\n\tvalidation loss: {:4f}\n\ttrain accuracy: {:2f}%\n\tval accuracy: {:2f}%".format(z, train_loss, val_loss, train_acc*100, val_acc*100))

            z += 1

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


if __name__ == "__main__":
    import pandas as pd
    from sklearn.cross_validation import train_test_split

    x_mean = pd.read_pickle('clean_data/x_mean.pkl')
    x_minmax = pd.read_pickle('clean_data/x_minmax.pkl')
    y = pd.read_pickle('clean_data/y_negpos.pkl')

    X_train, X_test, y_train, y_test = train_test_split(x_minmax.values, y.values, test_size=0.15, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

    n_samples = X_train.shape[0]

    svm1 = SVM(n=X_train.shape[0], kernel='rbf')
    train_losses, val_losses = svm1.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, tol=.00001, val_acc_break=.84)

    svm1.plot(train_losses, val_losses)


