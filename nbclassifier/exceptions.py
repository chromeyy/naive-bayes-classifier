class UnfittedPreprocessorError(Exception):
    def __init__(self):
        super().__init__()


class RefittingPreprocessorError(Exception):
    def __init__(self):
        super().__init__()


class ClassifierAlreadyTrainedError(Exception):
    def __init__(self):
        super().__init__()


class ClassifierUntrainedError(Exception):
    def __init__(self):
        super().__init__()


class InvalidPreprocessingError(Exception):
    def __init__(self):
        super().__init__("Input was preprocessed by the incorrect TextPreprocessor")


class NotPreprocessedError(Exception):
    def __init__(self):
        super().__init__("Input is not of form PreprocessedText returned by a TextPreprocessor")
