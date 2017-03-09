import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class BaseClassifier(BaseEstimator, ClassifierMixin):
    """
    base class you can use for implementing a classifier

    You'll need to define the following methods:

    def __init__(self, param1=default_value1, ..., paramN=default_valueN):
        self.param1 = param1
        ...
        self.paramN = paramN
    """

    def fit(self, X, y):
        """An abstract method you should override to implement your classifier's fit method

        you should be able to treat this implementation as a template and simply
        replace the exception with your own code

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # this convienince method does some sanity checking on X and y
        self._pre_fit(X, y)
        # replace 'raise NotImplementedError' with your model training code
        raise NotImplementedError
        # make sure you return self
        return self

    def predict(self, X):
        """ A abstract method you need to override to implement your classifiers predict method

        you should be able to treat this implementation as a template and simply
        replace the exception with your own code

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # this convienince method performs some sanity checking
        X = self._pre_predict(X)
        # replace 'raise NotImplementedError' with your prediction code
        #
        # since 'predict' accepts an array of samples, you'll probably want
        # to implement some other method which does a prediction for an
        # individual sample and iterate over it, such as:
        #
        # return [self._predict(row) for row in X]
        #
        raise NotImplementedError

    def _pre_fit(self, X, y):
        """checks that X and y are valid array-line objects and have
        compatible shapes, etc. if the check passes sets up the attributes
        self.X_ and self.y_, self.classes_, and self.n_classes_
        """
        X, y = check_X_y(X, y)
        # setup some attributes we can determine automatically
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        self.n_classes = len(self.classes_)


    def _pre_predict(self, X):
        """verifies that fit() has been called on the model, and that X
        has the correct shape
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        return X
