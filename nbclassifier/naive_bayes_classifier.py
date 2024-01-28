import numpy as np
from nbclassifier.preprocessing import PreprocessedText
from nbclassifier.base_classifier import Classifier
from nbclassifier.exceptions import ClassifierUntrainedError, ClassifierAlreadyTrainedError, InvalidPreprocessingError, NotPreprocessedError


class NaiveBayesClassifier(Classifier):
    def __init__(self, alpha: float = 1):
        # Classes
        self.classes = None
        # Vocab, all tokens included in the text abstracts
        self.vocab = None
        # Class Probabilities, class distribution
        self.class_probabilities = {}
        # Word Probabilities, probability of a word given a class
        self.word_probabilities = {}
        # Laplace smoothing modifier based on chosen alpha hyperparameter
        self.alpha = alpha
        # Boolean value for whether this classifier been trained
        self.trained = False

    def train(self, x: PreprocessedText, y):
        # If the classifier has already been trained, raise an error
        if self.trained:
            raise ClassifierAlreadyTrainedError
        self.trained = True

        # Extract vector from Preprocessed Text
        try:
            x_vector = x.get_vector()
        except Exception:
            raise NotPreprocessedError

        # Get all classes from unique values in example classes
        self.classes = np.unique(y)

        # Extract vocab from Preprocessed Text
        self.vocab = x.get_feature_names()

        for c in self.classes:
            # get all texts for a class as the documents for a class
            docs = x_vector[y == c]
            # calculate class probabilities
            self.class_probabilities[c] = np.log(len(docs)) - np.log(len(x_vector))
            # calculate the total amount of tokens contained within the class's documents
            total_tokens = np.sum(docs)
            # calculate the occurrence of each token contained within a vector
            token_occurrence = np.sum(docs, axis=0)
            # calculate the probability of the token occurring given the class (with applied laplace smoothing)
            self.word_probabilities[c] = np.log(token_occurrence + self.alpha) - np.log(
                total_tokens + (len(self.vocab) * self.alpha))

    def predict(self, x: PreprocessedText):
        # If Classifier is not trained raise an exception
        if not self.trained:
            raise ClassifierUntrainedError

        # Checking if input is valid
        try:
            vocab = x.get_feature_names()
        except Exception:
            raise NotPreprocessedError

        if len(vocab) != len(self.vocab) or not vocab.all() == self.vocab.all():
            raise InvalidPreprocessingError

        # Vectorize the texts to be classified
        x_vector = x.get_vector()

        num_examples = len(x_vector)
        # Initialise vector for the classification of the text vector
        winning_classes = np.array([""] * num_examples)
        # Initialise vector for the probability of the current determined classes for the text vector
        winning_probabilities = np.array([-np.inf] * num_examples)
        for c in self.classes:
            # Calculate the probability of the classes given the text vector - Bayes theorem
            contesting_probabilities = np.sum(self.word_probabilities[c] * x_vector, axis=1)
            contesting_probabilities += self.class_probabilities[c]
            # Change the classification to the most probable class
            winning_classes = np.where(
                contesting_probabilities > winning_probabilities,
                c,
                winning_classes
            )
            # Track the probability for that classification
            winning_probabilities = np.maximum(winning_probabilities, contesting_probabilities)
        return winning_classes

    def clear(self):
        # Clearing classifier model
        self.classes = []
        self.vocab = []
        self.class_probabilities.clear()
        self.word_probabilities.clear()
        self.trained = False

    def clear_set_alpha(self, alpha: float = 1):
        # Clearing the model with function of changing the alpha hyperparameter
        self.__init__(alpha)
