import unittest
import pandas as pd
from nbclassifier.preprocessing import PreprocessedText, TextPreprocessor
from nbclassifier.exceptions import UnfittedPreprocessorError, RefittingPreprocessorError

df_train = pd.read_csv("../example_data/train.csv")
x = df_train["Text"]


class PreprocessingTest(unittest.TestCase):

    def test_transforming_unfitted_Preprocessor(self):
        # This tests that transforming an unfitted Preprocessor raises an UnfittedPreprocessorError
        preprocessor = TextPreprocessor()
        with self.assertRaises(UnfittedPreprocessorError):
            preprocessor.transform(x)

    def test_fitting_twice_Preprocessor(self):
        # This tests that fitting a Preprocessor twice raises a RefittingPreprocessorError
        preprocessor = TextPreprocessor()
        preprocessor.fit_transform(x)
        with self.assertRaises(RefittingPreprocessorError):
            preprocessor.fit_transform(x)

    def test_fitting_Preprocessor(self):
        # This tests that fitting a Preprocessor works
        preprocessor = TextPreprocessor()
        transformed_x = preprocessor.fit_transform(x)
        # This tests that the output of fit_transform is of type PreprocessedText
        self.assertIsInstance(transformed_x, PreprocessedText)
        # This tests that the output of fit_transform has the correct feature names
        self.assertEqual(transformed_x.get_feature_names().all(), preprocessor.vectorizer.get_feature_names_out().all())
        # This tests that the output of fit_transform has the correct vector
        self.assertEqual(transformed_x.get_vector().shape, (len(x), len(transformed_x.get_feature_names())))


if __name__ == '__main__':
    unittest.main()
