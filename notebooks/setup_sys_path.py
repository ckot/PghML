
import os
import sys

def setup_pgh_ml_path():
    curr_path = os.getcwd()
    while not curr_path.endswith('PghML'):
        curr_path = os.path.dirname(curr_path)
    if curr_path.endswith('PghML'):
        if curr_path not in sys.path:
            sys.path.append(curr_path)

setup_pgh_ml_path()
