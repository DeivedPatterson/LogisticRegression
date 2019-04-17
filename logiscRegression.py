import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn import datasets

class RLTrainer(BaseEstimator):

    def __init__(self, learning_rate = 0.01, training_iters = 100):
        self.learning_rate = learning_rate
        self.training_iters = training_iters

    def __sigmoide(self, X):

        res = 1/(1 + np.exp(-np.dot(X, self.w)))

        return res

    def fit(self, X_train, y_train):

        #Se tiver apenas uma linha, transforma em um vetor coluna
        X = X_train.reshape(-1, 1) if len(X_train.shape) < 2 else X_train
        #insere uma coluna na posição 0 com valor na vertical
        X = np.insert(X, 0, 1, 1)

        #inicia os parâmetros com pequenos valores aleatórios
        self.w = np.random.normal(0, 1, size=X[0].shape)

        #loop de treinamento
        for epoch in range(self.training_iters):

            #inicia o gradiente
            gradient = np.zeros(self.w.shape)

            #atualiza o gradiente com informação de todos os pontos
            for var in range(len(gradient)):
                gradient[var] += np.dot((self.__sigmoide(X) - y_train), X[:, var])

            #multiplica o gradiente pela taxa de aprendizado
            gradient *= self.learning_rate
            #atualiza os parâmetros
            self.w -= gradient
            plt.plot(gradient)
        plt.show()

    def predict(self, X_test):
        #formata os dados
        if len(X_test.shape) < 2:
            X = X_test.reshape(-1, 1)

        X = np.insert(X, 0, 1, 1)

        #aplica função logística
        logit = self.__sigmoide(X)

        #aplica limiar
        return np.greater_equal(logit, 0.5).astype(int)

rl = RLTrainer(learning_rate=.0001, training_iters=100)
iris = datasets.load_iris()
X_train = iris.data[:, :2]  # we only take the first two features.
Y_train = iris.target
rl.fit(X_train, Y_train)