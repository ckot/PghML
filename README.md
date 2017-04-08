# PghML
meetup group resources


Repo directory structure:
-------------------------
```
PghML/  (this directory)
     README.md (this file)
     datasets/  (where we keep our .csv files, etc)
     notebooks/ (where jupyter notebooks should be saved)
              /setup_sys_path.py  (file which merely needs to be imported in
                                   a python jupyter notebook to be able
                                   make use of code in the pgh_ml_py subdir)
     pgh_ml_py/ (where any generic python code can be placed)
              /sklearn_compat/ (where any scikit-learn compatible code
                                should be placed)
     requirements.txt (file which contains names of python packages necessary for
                       python code/notebooks)
```

This directory structure is intended to be both language and toolkit agnostic. It simply allows us to
save our code in an organized way which lets us both reuse it in general, and still make use of in Jupyter notebooks.

Currently **pgh_ml_py** only contains a file named **datasets.py** which contains convenience functions
for loading files which reside in the PghML/datasets directory

**pgh_ml_py/sklearn_compat** contains a file **base.py** which currently contains some abstract base
classes you can use to create scikit-learn compatible classifiers, etc, minimizing the boilerplate you would otherwise need to implement.


We could also create a directory such as
    *pgh_ml_py/tensorflow_compat*, *pgh_ml_julia*, *pgh_ml_r*, etc.,
     if anyone implements algorithms using other languages and/or toolkits

Anyway, this is just a suggested directory organization, and also simple enough
to change at a later time, being a .git repo

## Setting up a python environment

You should perform the following steps from you clone of the PghML repo:
```
  virtualenv <some_directory>
  source <some_directory>/bin/activate
  pip install -r requirements.txt
```


## TODO
* Setting up a Julia environment
* Setting up an R environment
* Setup for other languages/toolkits
