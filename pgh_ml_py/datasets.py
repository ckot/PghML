
import os

import numpy as np
import pandas as pd

class Bunch(object):
    """
    A simple class which provides '.' access to dict variables, like a
    javascript object or python defaultdict, but unlike a defaultdict, allows
    you to pass kwargs to it instead of a fixed set of positional params

    As a convention, in sklearn, common field names are (at a minimum):
    -------------------------------------------------------------------
    data -> the features matrix
    target -> (if supervised learning) the vector or labels

    Additional common field names used are:
    ---------------------------------------
    feature_names -> labels for the columns in data (often extracted by the
                     csv header, but you may need to pass them in manually
                     if there is none)
    labels -> (if supervised learning) string values which map to the set
               of class values (typically for classifiers where the class
               values are integers)
    descr ->  string, possibly marked up describing a dataset
    """
    _STD_FLD_NAMES = ['data', 'target', 'feature_names', 'labels', 'descr']

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        fld_vals = []
        for std_var in self._STD_FLD_NAMES:
            if hasattr(self, std_var):
                fld_vals.append('%s=%s' % (std_var,
                                           type(getattr(self, std_var))))
        for var in sorted(vars(self)):
            if var not in self._STD_FLD_NAMES:
                fld_vals.append('%s: %s' % (var,
                                            type(getattr(self, var))))

        return 'Bunch(%s)' % ', '.join(fld_vals)


def load_banknote_authentication():
    """Loads the CSV file using pandas (which has the nice feature of
    automatically converting fields to the correct types), and
    returns the dataframe as a Bunch object.
    converting strings to floats and returns the data"""
    # print os.getcwd()
    datasets_path = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(datasets_path, "banknote_authentication.csv"))
    return Bunch(dataframe=df,
                 data=df[df.columns[:-1]].as_matrix(),
                 target=df[df.columns[-1]].as_matrix(),
                 feature_names=df.columns[:-1].tolist())
    # return {}
