# PghML
meetup group resources


Repo directory structure:
-------------------------
```
PghML/  (this directory)
__init__.py  (simply allows you to treat pgh_ml_py as a python pkg)
datasets/  (where we keep our .csv files, etc)
notebooks/ (where jupyter notebooks should be saved)
         /
pgh_ml_py/ (where any generic python code can be placed)
pgh_ml_py/sklearn_compat/ (where any scikit-learn compatible code should be placed)
README.md (this file)
requirements.txt (file which contains names of python packages necessary for our code/python notebooks)
```

This directory structure is intended to be both language and toolkit agnostic. It simply allows us to
save our code in an organized way which lets us both reuse it in general, and still make use of in Jupyter notebooks.  

Currently **pgh_ml_py** only contains a file named **datasets.py** which contains convenience functions
for loading files which reside in the PghML/datasets directory

**pgh_ml_py/sklearn_compat** contains a file **base.py** which currently contains some abstract base
classes you can use to create scikit-learn compatible classifiers, etc, minimizing the boilerplate you would otherwise need to implement.


We could also create a directory **pgh_ml_py/tensorflow_compat** if anyone implements algorithms
using portions of that toolkit. as well as **pgh_ml_julia**, **pgh_ml_r**, or **pgh_ml_java** if need be.

## Setting up a python environment

You should perform the following steps from you clone of the PghML repo:
* `virtualenv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`



## TODO
* Setting up a Julia environment
* Setting up an R environment
* Setup for java for various toolkits
