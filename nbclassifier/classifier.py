from abc import ABC, abstractmethod
import numpy as np


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
