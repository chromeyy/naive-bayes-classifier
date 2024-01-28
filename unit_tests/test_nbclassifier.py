import unittest
from nbclassifier.naive_bayes_classifier import NaiveBayesClassifier
from nbclassifier.base_classifier import Classifier
from nbclassifier.preprocessing import TextPreprocessor, PreprocessedText
from nbclassifier.exceptions import ClassifierUntrainedError, ClassifierAlreadyTrainedError, InvalidPreprocessingError, NotPreprocessedError
import pandas as pd
import numpy as np

df_train = pd.read_csv("../example_data/train.csv")
df_test = pd.read_csv("../example_data/test.csv")
x_train = df_train["Text"]
y_train = df_train["Category"]
x_test = df_test["Text"]
y_test = df_test["Category"]
preprocessor = TextPreprocessor()
x_train_vector = preprocessor.fit_transform(x_train)
x_test_vector = preprocessor.transform(x_test)


class TestNaiveBayesClassifier(unittest.TestCase):
    def test_training_classifier(self):
        # This tests that training a classifier works
        classifier = NaiveBayesClassifier()
        classifier.train(x_train_vector, y_train)
        # This tests that the classifier has been trained
        self.assertTrue(classifier.trained)
        # This tests that the classifier has the correct classes
        self.assertEqual(set(classifier.classes), set(y_train.unique()))
        # This tests that the classifier has the correct vocab
        self.assertEqual(classifier.vocab.all(), x_train_vector.get_feature_names().all())

    def test_predicting_classifier(self):
        # This tests that predicting with a classifier works
        classifier = NaiveBayesClassifier()
        classifier.train(x_train_vector, y_train)
        y_pred = classifier.predict(x_test_vector)
        # This tests that the output of predict is of type np.ndarray
        self.assertIsInstance(y_pred, np.ndarray)
        # This tests that the output of predict has the correct shape
        self.assertEqual(y_pred.shape, y_test.shape)
        # This tests that the output of predict mostly the correct values
        self.assertGreater(np.sum(y_pred == y_test) / len(y_test), .8)

    def test_predicting_untrained_classifier(self):
        # This tests that predicting with an untrained classifier raises a ClassifierUntrainedError
        classifier = NaiveBayesClassifier()
        with self.assertRaises(ClassifierUntrainedError):
            classifier.predict(x_test_vector)

    def test_training_classifier_twice(self):
        # This tests that training a classifier twice raises a ClassifierAlreadyTrainedError
        classifier = NaiveBayesClassifier()
        classifier.train(x_train_vector, y_train)
        with self.assertRaises(ClassifierAlreadyTrainedError):
            classifier.train(x_train_vector, y_train)

    def test_training_classifier_with_invalid_preprocessing(self):
        # This tests that training a classifier with invalid preprocessing raises an InvalidPreprocessingError
        classifier = NaiveBayesClassifier()
        arb_vector = PreprocessedText(np.array(["a", "b", "c"]), np.array([[1, 2, 3], [4, 5, 6]]))
        with self.assertRaises(InvalidPreprocessingError):
            classifier.train(x_train_vector, y_train)
            classifier.predict(arb_vector)

    def test_training_classifier_with_unpreprocessed_input(self):
        # This tests that training a classifier with unpreprocessed input raises a NotPreprocessedError
        classifier = NaiveBayesClassifier()
        with self.assertRaises(NotPreprocessedError):
            classifier.train(x_train, y_train)



if __name__ == '__main__':
    unittest.main()
