# LambdaFRD
The `lambdaFRD` functions provide facilities for lambda class estimation and inference in fuzzy regression discontinuity (FRD) designs, currently available in R and python. The theory for this class is found in:

Lane (2025). The moment is here: a generalised class of estimators for fuzzy regression discontinuity designs. Working paper. https://arxiv.org/abs/2511.03424

## Installation and use

**Quick reference guides** are given for both R and python within the respective folders, to allow for immediate easy use. More detailed documentation is also given.

### R
Install the R development version from GitHub:
```r
# install.packages("devtools")
devtools::install_github("stuart-lane/lambdaFRD", subdir="R")
```
### Python
Install the python development version from GitHub:
```python
pip install git+https://github.com/stuart-lane/lambdaFRD.git#subdirectory=Python
```
Alternatively, simply download the functions, save them in to the appropriate directory and directly work with the functions themselves.

### Replication
Replication code for the empirical application is available in both R and python. Currently, replication code for simulations is only available in R.

## ⚠️ Development Status

Current version: 0.1.0 (pre-release)
 
This package is in the late stages of development, but has not undergone extensive testing. While I believe the implementation is correct, bugs may exist. Use with caution and please report any issues if you find any, you can email me at stuart.lane@bristol.ac.uk


