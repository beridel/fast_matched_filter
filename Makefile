# DIRECTORIES
maindir=fast_matched_filter
srcdir=$(maindir)/src
libdir=$(maindir)/lib

# NB. If using Matlab on Mac, you MUST configure your mexopts.sh file to use an OpenMP-friendly
# compiler (like gcc)! The default clang compiler is not OpenMP-friendly. To easily install gcc
# with OpenMP enalbed via homebrew, try $brew install gcc --without-multilib
NVCC=nvcc
CC=gcc
MEX=/Applications/MATLAB_R2015b.app/bin/mex

# NB. If using Matlab on another platform than Mac, please change the file extensions for Matlab to
# the following MEX extension by platform:
# Mac: mexmaci64
# Linux: mexa64
# Windows: mexw64
mex_extension=mexa64

all: $(libdir)/matched_filter_GPU.so $(libdir)/matched_filter_CPU.so $(maindir)/matched_filter.$(mex_extension)
python_cpu: $(libdir)/matched_filter_CPU.so
python_gpu: $(libdir)/matched_filter_GPU.so 
matlab: $(maindir)/matched_filter.$(mex_extension)
.SUFFIXES: .c .cu

# GPU FLAGS
COPTIMFLAGS_GPU=-O3
CFLAGS_GPU=-Xcompiler "-fopenmp -fPIC -march=native -ftree-vectorize" -Xlinker -lgomp
CARDDEPENDENTFLAG=-arch=sm_35
LDFLAGS_GPU=--shared

# CPU FLAGS
COPTIMFLAGS_CPU=-O3
CFLAGS_CPU=-fopenmp -fPIC -ftree-vectorize -march=native
LDFLAGS_CPU=-shared

# MEX FLAGS
COPTIMFLAGS_MEX=-O3
CFLAGS_MEX=-fopenmp -fPIC -march=native
 # who knows why mex needs fopenmp again
LDFLAGS_MEX=-fopenmp -shared

# build for python
$(libdir)/matched_filter_GPU.so: $(srcdir)/matched_filter.cu
	$(NVCC) $(COPTIMFLAGS_GPU) $(CFLAGS_GPU) $(CARDDEPENDENTFLAG) $(LDFLAGS_GPU) $< -o $@

$(libdir)/matched_filter_CPU.so: $(srcdir)/matched_filter.c
	$(CC) $(COPTIMFLAGS_CPU) $(CFLAGS_CPU) $(LDFLAGS_CPU) $< -o $@

# build for Matlab
$(maindir)/matched_filter.$(mex_extension): $(srcdir)/matched_filter_mex.c $(srcdir)/matched_filter.c
	$(MEX) CC=$(CC) COPTIMFLAGS="$(COPTIMFLAGS_MEX)" CFLAGS="$(CFLAGS_MEX)" LDFLAGS="$(LDFLAGS_MEX)" -output $@ $^

clean:
	rm $(libdir)/*.so $(maindir)/*.mex*

