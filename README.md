# cs231n Assignments

This is my attempts at the assignments for Stanford cs231n winter 2016 classes. 

The original course website is [here](http://cs231n.stanford.edu/2016/syllabus).

The original code works for Python 2.x, a few things are needed to port them to Python 3.x

First fix `print` statements:

```
2to3 --fix=print --write *.py
```

Secondly, `pickle` works differently in Python 3.x. Two changes are needed:

1. Change package `cPickle` to `_pickle` in Python 3.x,
2. To pickle a file written by Python 2.x in Python 3.x

```
import _pickle as pickle

pickle.load(f, encoding='latin1')
```

Still not sure how to fix `print` in notebooks, best I got is to use replace all...
