import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output

            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    # Probability of belonging to the positive class
    def predict_proba(self, X):
        return self.activation(self.net_input(X))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


class OneVsRestLogisticRegression(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.classes_ = None
        self.classifiers_ = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers_ = []

        # Train one binary classifier for each class
        for current_class in self.classes_:
            y_binary = np.where(y == current_class, 1, 0)

            classifier = LogisticRegressionGD(
                eta=self.eta,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            classifier.fit(X, y_binary)
            self.classifiers_.append(classifier)

        return self

    def predict_proba(self, X):
        probabilities = []

        # Collect probabilities from all binary classifiers
        for classifier in self.classifiers_:
            class_probability = classifier.predict_proba(X)
            probabilities.append(class_probability)

        probabilities = np.array(probabilities).T

        # Normalize probabilities so each row sums to 1
        probabilities_sum = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / probabilities_sum

        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=1,
        stratify=y
    )

    # Standardization based on training data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    model = OneVsRestLogisticRegression(eta=0.1, n_iter=2000, random_state=1)
    model.fit(X_train_std, y_train)

    y_pred = model.predict(X_test_std)
    accuracy = np.mean(y_pred == y_test)

    print("Accuracy:")
    print(accuracy)

    plot_decision_regions(X=X_train_std, y=y_train, classifier=model)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()