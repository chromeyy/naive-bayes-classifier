from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass

    @abstractmethod
    def clear(self):
        pass
