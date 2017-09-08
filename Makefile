all: matched_filter_GPU.so matched_filter_CPU.so matched_filter.mexa64
python_cpu: matched_filter_CPU.so
python_gpu: matched_filter_GPU.so 
matlab: matched_filter.mexa64
.SUFFIXES: .c .cu
# NB. If using Matlab on another platform than Mac, please change the file extensions for Matlab to
# the following MEX extension by platform:
# Mac: .mexmaci64
# Linux: .mexa64
# Windows: .mexw64

NVCC=nvcc
CC=gcc
MEX=mex
# NB. If using Matlab on Mac, you MUST configure your mexopts.sh file to use an OpenMP-friendly
# compiler (like gcc)! The default clang compiler is not OpenMP-friendly. To easily install gcc
# with OpenMP enalbed via homebrew, try $brew install gcc --without-multilib

# GPU FLAGS
COPTIMFLAGS_GPU=-O3
CFLAGS_GPU=-Xcompiler -fopenmp -Xcompiler -fPIC -Xlinker -lgomp
CARDDEPENDENTFLAG=-arch=sm_35
LDFLAGS_GPU=--shared

# CPU FLAGS
COPTIMFLAGS_CPU=-O3
CFLAGS_CPU=-fopenmp -fPIC
LDFLAGS_CPU=-shared

# MEX FLAGS
COPTIMFLAGS_MEX=-O3
CFLAGS_MEX=-fopenmp -fPIC
LDFLAGS_MEX=-fopenmp -shared # who knows why mex needs fopenmp again

# build for python
matched_filter_GPU.so: matched_filter.cu
	$(NVCC) $(COPTIMFLAGS_GPU) $(CFLAGS_GPU) $(CARDDEPENDENTFLAG) $(LDFLAGS_GPU) $< -o $@

matched_filter_CPU.so: matched_filter.c
	$(CC) $(COPTIMFLAGS_CPU) $(CFLAGS_CPU) $(LDFLAGS_CPU) $< -o $@

# build for Matlab
matched_filter.mexa64: matched_filter_mex.c matched_filter.c
	$(MEX) COPTIMFLAGS="$(COPTIMFLAGS_MEX)" CFLAGS="$(CFLAGS_MEX)" LDFLAGS="$(LDFLAGS_MEX)" -output $@ $^

clean:
	rm *.so *.mex* *.pyc
