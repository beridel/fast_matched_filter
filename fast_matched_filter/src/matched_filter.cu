/*
:copyright:
    William B. Frank and Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
*/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define BLOCKSIZE 512

extern "C" { // needed for C-style symbols in shared object compiled by nvcc
#include "matched_filter_GPU.h"

//-------------------------------------------------------------------------
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {

    if (code != cudaSuccess) 
    {
        fprintf(stderr, "An error occured in the kernel: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//-------------------------------------------------------------------------
__global__ void network_corr(float *templates, float *sum_square_template, int *moveout, float *data,
                             size_t step, size_t n_samples_template, size_t n_samples_data,
                             size_t n_stations, size_t n_components, size_t n_corr,
                             float *cc_mat) {
  
    // each thread matches the template to one time in the data
    int idx, first_sample_block; // sample's index
    int i, s, c; // counters
    int data_offset, templates_offset, sum_square_template_offset, cc_mat_offset;
    float numerator, sum_square_data;
    float data_sample, template_sample;
    int t_idx;

    //------------------------------------------------
    int count_template = (n_samples_template / 32 + 1) * 32;
    extern __shared__ float shared[];
    float *templates_s = &shared[0];
    float *data_s = &shared[count_template];

    // 1 block processes one channel to blockDim.x / step different positions in time
    idx = (blockIdx.x * blockDim.x) * step;
    first_sample_block = idx % n_samples_data;
    s = idx / (n_samples_data * n_components);
    c = (idx % (n_samples_data * n_components)) / n_samples_data;

    // compute offsets for input variables
    cc_mat_offset = (first_sample_block / step + threadIdx.x) * n_stations;
    templates_offset = s * n_samples_template * n_components + c * n_samples_template;
    sum_square_template_offset = s * n_components + c;
    data_offset = s * n_samples_data * n_components + c * n_samples_data + first_sample_block + moveout[s];

    // initialize sums
    sum_square_data = 0.0f;
    numerator = 0.0f;

    // load template and data into shared memory
    t_idx = threadIdx.x;
    while(t_idx < n_samples_template) {
        templates_s[t_idx] = __ldg(&templates[templates_offset + t_idx]);
        data_s[t_idx] = __ldg(&data[data_offset + t_idx]);
        t_idx += blockDim.x;
    }
    while(t_idx < (blockDim.x * step + n_samples_template)){
        data_s[t_idx] = __ldg(&data[data_offset + t_idx]);
        t_idx += blockDim.x;
    }

    __syncthreads(); // make sure the waveforms are read before keep going

    if ((first_sample_block / step + threadIdx.x) < n_corr) {
        // calculate correlation coefficient
        for(i = 0; i < n_samples_template; i++) {
            data_sample = data_s[i + threadIdx.x * step];
            template_sample = templates_s[i];
            numerator += data_sample * template_sample;
            sum_square_data += data_sample * data_sample; 
        }

        cc_mat[cc_mat_offset + s] += (numerator * rsqrtf(sum_square_data * sum_square_template[sum_square_template_offset])) / (float)n_components;
    }
    __syncthreads(); // wait for every thread to finish before leaving the kernel
}

//-------------------------------------------------------------------------
__global__ void sum_cc(float *cc_mat, float *cc_sum, float *weights,
        int n_stations, int n_corr) {

    int i, s;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_corr) {
        float *cc_mat_offset;

        cc_mat_offset = cc_mat + i * n_stations;
        for (s = 0; s < n_stations; s++) cc_sum[i] += cc_mat_offset[s] * weights[s];
    }
}

//-------------------------------------------------------------------------
void matched_filter(float *templates, float *sum_square_templates, 
                    int *moveouts, float *data, float *weights, size_t step,
                    size_t n_samples_template, size_t n_samples_data,
                    size_t n_templates, size_t n_stations, size_t n_components, size_t n_corr,
                    float *cc_sums) {

    int t_global = -1;
    int nGPUs;

    // find the number of available GPUs
    cudaGetDeviceCount(&nGPUs);
    omp_set_num_threads(nGPUs);

    // Size of variables to create on the device (GPU)
    size_t sizeof_templates = sizeof(float) * n_samples_template * n_stations * n_components * n_templates;
    size_t sizeof_moveouts = sizeof(int) * n_stations * n_templates;
    size_t sizeof_data = sizeof(float) * n_samples_data * n_stations * n_components;
    size_t sizeof_cc_mat = sizeof(float) * n_corr * n_stations; // cc matrix for one template
    size_t sizeof_cc_sum = sizeof(float) * n_corr; // cc sums for one template
    size_t sizeof_sum_square_templates = sizeof(float) * n_templates * n_stations * n_components;
    size_t sizeof_weigts = sizeof(float) * n_templates * n_stations;
    size_t sizeof_total = sizeof_templates + sizeof_moveouts + sizeof_data + sizeof_cc_mat + sizeof_cc_sum + sizeof_sum_square_templates + sizeof_weigts;

#pragma omp parallel shared(t_global, templates, moveouts, data, n_templates, cc_sums, weights, sum_square_templates) 
    {
        float *templates_d = NULL;
        float *data_d = NULL;
        int *moveouts_d = NULL;
        float *cc_mat_d = NULL;
        float *cc_sum_d = NULL;
        float *sum_square_templates_d = NULL;
        float *weights_d = NULL;
        int id;

        // assign thread to a GPU and get its properties
        id = omp_get_thread_num();
        cudaSetDevice(id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, id);

        // Card-dependent settings: prefer L1 cache or shared memory
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        // check if enough memory is available
        size_t freeMem = 0;
        size_t totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        if (sizeof_total > freeMem) {
            printf("%zu bytes are requested on GPU #%i whereas it has only %zu free bytes.\n", sizeof_total, id, freeMem);
            printf("Reduce the number of templates processed in one batch.\n");
            exit(0);
        }

        // allocate GPU memory
        cudaMalloc((void**)&templates_d, sizeof_templates);
        cudaMalloc((void**)&moveouts_d, sizeof_moveouts);
        cudaMalloc((void**)&data_d, sizeof_data);
        cudaMalloc((void**)&cc_mat_d, sizeof_cc_mat);
        cudaMalloc((void**)&cc_sum_d, sizeof_cc_sum);
        cudaMalloc((void**)&sum_square_templates_d, sizeof_sum_square_templates);
        cudaMalloc((void**)&weights_d, sizeof_weigts);

        // transfer the inputs from host to the GPU
        cudaMemcpy(templates_d, templates, sizeof_templates, cudaMemcpyHostToDevice);
        cudaMemcpy(moveouts_d, moveouts, sizeof_moveouts, cudaMemcpyHostToDevice);
        cudaMemcpy(data_d, data, sizeof_data, cudaMemcpyHostToDevice);
        cudaMemcpy(sum_square_templates_d, sum_square_templates, sizeof_sum_square_templates, cudaMemcpyHostToDevice);
        cudaMemcpy(weights_d, weights, sizeof_weigts, cudaMemcpyHostToDevice);

        // loop over templates
        while (t_global < (int)n_templates) {
            int t_thread;
            size_t n_corr_t;
            int max_moveout;
            float *templates_d_t = NULL;
            int *moveouts_t = NULL, *moveouts_d_t = NULL;
            float *cc_sums_t = NULL;
            float *sum_square_templates_d_t = NULL;
            float *weights_d_t = NULL;
            int maxSharedMem = props.sharedMemPerBlock; 

            // increment template loop
#pragma omp critical
            {
                t_global++;
                t_thread = t_global;
            }   
            if (t_thread >= (int)n_templates) break;

            // define block and grid sizes for kernels
            dim3 BS(BLOCKSIZE);
            dim3 GS(ceilf((n_samples_data * n_components * n_stations) / (float)(BS.x * step)));
            
            // calculate the space required in the shared memory
            int count_template = (n_samples_template / 32 + 1) * 32;
            int count_data = ((n_samples_template + BLOCKSIZE * step) / 32 + 1) * 32;
            int sharedMem = (count_template + count_data) * sizeof(float);
            if (sharedMem > maxSharedMem) {
                int new_step = (maxSharedMem/sizeof(float) - 2 * n_samples_template - 64) / BLOCKSIZE;
                int new_length = maxSharedMem/sizeof(float) - count_data - 32;
                if (new_length < 0) new_length = 0;
                printf("The maximum shared memory available on this card is %i bytes "\
                        "(%i bytes required). You should consider the different options:\n"\
                        "  - Change the temporal step to %i without changing the template length.\n"\
                        "  - Change the template length to %i without changing the temporal step.\n"\
                        "  - Try to decrease both of these parameters.\n",
                        maxSharedMem, sharedMem, new_step, new_length);
                exit(0);
            }
            
            // compute the number of correlation steps for this template
            moveouts_t = moveouts + t_thread * n_stations;
            max_moveout = 0;
            for (int i = 0; i < n_stations; i++) {
                max_moveout = (moveouts_t[i] > max_moveout) ? moveouts_t[i] : max_moveout;
            }
            n_corr_t = (n_samples_data - n_samples_template - max_moveout) / step + 1;

            // local pointers on the device
            templates_d_t = templates_d + t_thread * n_samples_template * n_stations * n_components;
            sum_square_templates_d_t = sum_square_templates_d + t_thread * n_stations * n_components;
            moveouts_d_t = moveouts_d + t_thread * n_stations;
            weights_d_t = weights_d + t_thread * n_stations;

            // process
            cudaMemset(cc_mat_d, 0, sizeof_cc_mat); // initiailize cc_mat to 0
            network_corr<<<GS, BS, sharedMem>>>(templates_d_t, 
                                                sum_square_templates_d_t, 
                                                moveouts_d_t, 
                                                data_d,
                                                step, 
                                                n_samples_template,
                                                n_samples_data, 
                                                n_stations,
                                                n_components,
                                                n_corr_t,
                                                cc_mat_d);

            // return an error if something happened in the kernel (and crash the program)
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // weighted sum of correlation coefficients
            cudaMemset(cc_sum_d, 0, sizeof_cc_sum);

            dim3 GS_sum(ceilf(n_corr_t / (float)BS.x));
            sum_cc<<<GS_sum, BS>>>(cc_mat_d, cc_sum_d, weights_d_t, n_stations, n_corr_t);

            // return an error if something happened in the kernel (and crash the program)
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // xfer cc_sum back to host
            cc_sums_t = cc_sums + t_thread * n_corr;
            cudaMemcpy(cc_sums_t, cc_sum_d, sizeof_cc_sum, cudaMemcpyDeviceToHost);
        } // while

        // free device memory
        cudaFree(templates_d);
        cudaFree(moveouts_d);
        cudaFree(data_d);
        cudaFree(cc_mat_d);
        cudaFree(cc_sum_d);
        cudaFree(sum_square_templates_d);
        cudaFree(weights_d);

    } // omp parallel
} //  matched_filter
} // extern C
