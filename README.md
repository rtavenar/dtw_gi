# Time Series Alignment with Global Invariances

This code is an attempt to help reproduce results from the paper "Time Series 
Alignment with Global Invariances" (<https://openreview.net/forum?id=JXCH5N4Ujy>).

## Pre-requisites

This code is Python3 code and relies on the following libraries:

```
tslearn>=0.3
numba
geoopt
torch
matplotlib
numpy
scipy
imageio
```

Dependencies can be installed via:

```bash
pip install -r requirements.txt
```

(maybe the `torch` dependency will raise an error and an alternative install 
method will be prompted).

Also, the base folder of this code (path to this `README.md` file) should be
added to your Python path for `dtw_gi` package to work properly:

```bash
export PYTHONPATH=$PYTHONPATH:path/to/base/folder
```

## Available scripts

For figures 1 to 5, notebooks are provided that generate the figures:

* [Fig. 1](fig1.ipynb)
* [Fig. 2](fig2.ipynb)
* [Fig. 3](fig3.ipynb) (note that this Figure reports timings and hence 
obtained results can be different from one execution environment to the other,
however, trends should be similar, since they reflect theoretical complexity)
* [Fig. 4](fig4.ipynb)
* [Fig. 5](fig5.ipynb)
* [Fig. 6](fig6.ipynb)
