
from os.path import dirname, join

import numpy as np
import pandas as pd

DATASETS_PATH = join(dirname(dirname(__file__)), "datasets")


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

    def _repr_fld(self, fld):
        val = getattr(self, fld)
        if isinstance(val, np.ndarray):
            return "%s=ndarray%s" % (fld, val.shape)
        else:
            return "%s=%s" % (fld, type(val))

    def __repr__(self):
        fld_vals = []
        for fld_name in self._STD_FLD_NAMES:
            if hasattr(self, fld_name):
                fld_vals.append(self._repr_fld(fld_name))
        for fld_name in sorted(vars(self)):
            if fld_name not in self._STD_FLD_NAMES:
                fld_vals.append(self._repr_fld(fld_name))
        return 'Bunch(%s)' % ', '.join(fld_vals)


def load_banknote_authentication():
    """Loads the CSV file using pandas (which has the nice feature of
    automatically converting fields to the correct types), and
    returns the dataframe as a Bunch object.
    converting strings to floats and returns the data"""
    # print os.getcwd()
    df = pd.read_csv(join(DATASETS_PATH, "banknote_authentication.csv"))
    return Bunch(dataframe=df,
                 data=df[df.columns[:-1]].as_matrix(),
                 target=df[df.columns[-1]].as_matrix(),
                 feature_names=df.columns[:-1].tolist())
    # return {}
