from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import WordPunctTokenizer
import copy
from exceptions import UnfittedPreprocessorError, RefittingPreprocessorError


class PreprocessedText:
    def __init__(self, feature_names, vector):
        self.feature_names = feature_names
        self.vector = vector

    def get_feature_names(self):
        return self.feature_names

    def get_vector(self):
        return self.vector


class TextPreprocessor:
    def __init__(self, tfidf=False, ngram_range: tuple[int, int] = (0, 0), max_features=10000000):
        match tfidf:
            case False:
                self.vectorizer = CountVectorizer(
                    ngram_range=ngram_range,
                    tokenizer=WordPunctTokenizer().tokenize,
                    max_features=max_features
                )
            case True:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=ngram_range,
                    tokenizer=WordPunctTokenizer().tokenize,
                    max_features=max_features
                )

    def fit_transform(self, x):
        if self.vectorizer.vocabulary_:
            raise RefittingPreprocessorError
        vector = self.vectorizer.fit_transform(x).toarray()
        feature_names = self.vectorizer.get_feature_names_out()
        return PreprocessedText(feature_names, vector)

    def transform(self, x):
        if not self.vectorizer.vocabulary_:
            raise UnfittedPreprocessorError
        vector = self.vectorizer.transform(x).toarray()
        feature_names = self.vectorizer.get_feature_names_out()
        return PreprocessedText(feature_names, vector)
