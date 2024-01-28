# Naive Bayes Classifier
An importable Naive Bayes classifier

## Installation
When installing this library, first navigate your terminal to the root directory of this project.
Using ```cd <path to this project>```.
Then, run one of following commands:

**Installation via pip**
```shell
$ pip install .
```

**Installation via setup.py**
```shell
$ python setup.py install
```

## Usage
To use this library, first import both NaiveBayesClassifier and TextPreprocessor into your project.
```python
from nbclassifier.naive_bayes_classifier import NaiveBayesClassifier
from nbclassifier.preprocessing import TextPreprocessor
```

It's also important to note that the TextPreprocessor class takes Pandas dataframes as input.
As such, you'll need to import Pandas as well.
```python
import pandas as pd
```

Then, create a new instance of the NaiveBayesClassifier class.
```python
classifier = NaiveBayesClassifier()
```


Next, to preprocess the data necessary to train the classifier, create a new instance of the TextPreprocessor class.
```python
preprocessor = TextPreprocessor()
```

### Description of the two main classes below:

#### NaiveBayesClassifier
The NaiveBayesClassifier class is the main class of this library.
When creating a new instance of this class, you can pass in the following parameters:
1. alpha: The alpha value used in the Laplace smoothing equation. Defaults to 1.

It contains the following methods:

**_NaiveBayesClassifier_.train**
```python
def train(self, X: PreprocessedText, y):
    """
    Trains the classifier on the given data.

    :param X: A PreprocessedText object containing the preprocessed text data.
    :param y: An array or list of labels, where each label corresponds to a document in X.
    """
```

**_NaiveBayesClassifier_.predict**
```python
def predict(self, X: PreprocessedText) -> np.ndarray:
    """
    Predicts the labels of the given data.

    :param X: A PreprocessedText object containing the preprocessed text data.
    :return: A array type object of predicted labels.
    """
```

**_NaiveBayesClassifier_.clear**
```python
def clear(self):
    """
    Clears the classifier of all data.
    """
```

**_NaiveBayesClassifier_.clear_set_alpha**
```python
def clear_set_alpha(self, alpha: float):
    """
    Clears the classifier of all data and sets the alpha value to the given value.

    :param alpha: The alpha value used in the Laplace smoothing equation.
    """
```

#### TextPreprocessor
The TextPreprocessor class is used to preprocess text data.
When creating a new instance of this class, you can pass in the following parameters:
1. tdidf: A boolean value indicating whether or not to use tfidf vectorization. Defaults to False.
2. ngram_range: A tuple of integers indicating the range of ngrams to use. Defaults to (1, 1).
3. max_features: An integer indicating the maximum number of features to use. Defaults to 10000000.

It contains the following methods:

**_TextPreprocessor_.fit_transform**
```python
def fit_transform(self, x: pd.DataFrame) -> PreprocessedText:
    """
    Fits and transforms the given data.

    :param X: A Pandas dataframe containing the text data.
    :return: A PreprocessedText object containing the preprocessed text data.
    """
```

**_TextPreprocessor_.transform**
```python
def transform(self, x: pd.DataFrame) -> PreprocessedText:
    """
    Transforms the given data.

    :param X: A Pandas dataframe containing the text data.
    :return: A PreprocessedText object containing the preprocessed text data.
    """
```