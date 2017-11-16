# fast_matched_filter
An efficient seismic matched-filter search for both CPU and GPU architectures.

Required software/hardware:
- A C compiler that supports OpenMP (default Mac OS compiler clang does not support OpenMP; gcc can be easily downloaded via homebrew)
- Either Python (v2.7 or 3) or Matlab
- BONUS: a discrete Nvidia graphics card that supports CUDA C with CUDA toolkit installed

Fast Matched Filter is available @ https://github.com/beridel/fast_matched_filter
and can be downloaded with:
$ git clone https://github.com/beridel/fast_matched_filter.git

A simple make + whichever implementation does the trick.  Possible make commands are:
$ make python_cpu
$ make python_gpu
$ make matlab

Can also be imported as a python module!

Installation on Python also possible, but not thoroughly tested, via setup tools or even pip (which supports clean uninstalling):
$ python setup.py build
$ python setup.py install
OR
$ python setup.py build
$ pip install .

Reference: Beaucé, Eric, W. B. Frank, and Alexey Romanenko, Fast matched-filter (FMF): an efficient seismic matched-filter search for both CPU and GPU architectures. _Seismological Research Letters_ (in press)
