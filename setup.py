from setuptools import setup, find_packages

setup(
    name="nbclassifier",
    version="1.0",
    license="MIT",
    description="Naive Bayes Classifier",
    long_description=open('README.md').read(),
    url="https://github.com/chromeyy/naive-bayes-classifier",
    author="Kevin Jiang",
    author_email="kevinj1478@gmail.com",
    maintainer="Kevin Jiang",
    maintainer_email="kevinj1478@gmail.com",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'nltk'
    ],
    platforms="any"
)