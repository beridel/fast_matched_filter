# fast_matched_filter (FMF)
An efficient seismic matched-filter search for both CPU and GPU architectures.

If you use FMF in research to be published, please reference the following article: Beaucé, Eric, W. B. Frank, and Alexey Romanenko (2017). Fast matched-filter (FMF): an efficient seismic matched-filter search for both CPU and GPU architectures. _Seismological Research Letters_, doi: [10.1785/0220170181](https://doi.org/10.1785/0220170181)

FMF is available @ https://github.com/beridel/fast_matched_filter
and can be downloaded with:
$ git clone https://github.com/beridel/fast_matched_filter.git

_Required software/hardware:_
- A C compiler that supports OpenMP (default Mac OS compiler clang does not support OpenMP; gcc can be easily downloaded via homebrew)
- CPU version: either Python (v2.7 or 3.x) or Matlab
- GPU version: Python (v2.7 or 3.x) and a discrete Nvidia graphics card that supports CUDA C with CUDA toolkit installed

_Installation_
A simple make + whichever implementation does the trick.  Possible make commands are:
$ make python_cpu
$ make python_gpu
$ make matlab

NB: Matlab compiles via mex, which needs to be setup before running. Any compiler can be chosen during the setup of mex, because it will be bypassed by the CC environment variable in the Makefile. Therefore CC must be set to an OpenMP-compatible compiler.

Installation as a Python module is possible via setup tools or even pip (which supports clean uninstalling):
$ python setup.py build
$ python setup.py install
OR
$ python setup.py build
$ pip install .

_Running_
Python: Both CPU and GPU versions are called with the matched_filter function.
Matlab: The CPU version is called with the fast_matched_filter function
