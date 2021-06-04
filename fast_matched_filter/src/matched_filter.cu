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
#define WARPSIZE 32
#define NCHUNKS 20
#define STABILITY_THRESHOLD 0.000001f

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
__global__ void network_corr(float *templates, float *sum_square_template, int *moveout, float *data, float *weights,
                             size_t step, size_t n_samples_template, size_t n_samples_data,
                             size_t n_stations, size_t n_components,
                             int chunk_offset, int chunk_size,
                             float *cc_mat, int normalize) {
  
    // each thread matches the template to one time in the data
    int idx, first_sample_block, first_sample_trace, last_sample_trace; // sample's index
    int i, s, c; // counters
    int data_offset, templates_offset, sum_square_template_offset, cc_mat_offset;
    float numerator, denominator, sum_square_data, mean_data;
    float data_sample;
    int t_idx;

    //------------------------------------------------
    int count_template = (n_samples_template / WARPSIZE + 1) * WARPSIZE;
    extern __shared__ float shared[];
    float *ss_template = &shared[0];
    float *templates_s = &shared[sizeof(float)];
    float *data_s = &shared[count_template+sizeof(float)];

    // 1 block processes one channel to blockDim.x / step different positions in time
    idx = blockIdx.x/n_stations * blockDim.x + chunk_offset;
    first_sample_block = idx * step;
    s = blockIdx.x % n_stations;

    for (c = 0; c < n_components; c++){
        if (weights[s * n_components + c] != 0.){
            // compute offsets for input variables
            cc_mat_offset = (first_sample_block / step + threadIdx.x - chunk_offset) * n_stations * n_components + s * n_components + c;
            templates_offset = s * n_samples_template * n_components + c * n_samples_template;
            sum_square_template_offset = s * n_components + c;
            first_sample_trace = first_sample_block + moveout[s * n_components + c];
            last_sample_trace = first_sample_trace + n_samples_template + threadIdx.x * step;
            data_offset = s * n_samples_data * n_components + c * n_samples_data + first_sample_trace;

            // initialize sums
            sum_square_data = 0.0f;
            mean_data = 0.0f;
            numerator = 0.0f;

            // load template and data into shared memory
            t_idx = threadIdx.x;
            if (t_idx == 0){
                ss_template[0] = sum_square_template[sum_square_template_offset];
            }
            while(t_idx < n_samples_template) {
                templates_s[t_idx] = templates[templates_offset + t_idx];
                if ((first_sample_trace + t_idx) < n_samples_data) data_s[t_idx] = data[data_offset + t_idx];
                t_idx += blockDim.x;
            }
            while(t_idx < (blockDim.x * step + n_samples_template)){
                if ((first_sample_trace + t_idx) < n_samples_data) data_s[t_idx] = data[data_offset + t_idx];
                t_idx += blockDim.x;
            }

            __syncthreads(); // make sure the waveforms are read before keep going

            // calculate correlation coefficient
            if (last_sample_trace <= n_samples_data){
                // if not, corresponds to an ill-defined CC with some samples out of the bounds
                // Calculate the mean if fully normalising
                if (normalize > 0){
                    for (i = 0; i < n_samples_template; i++){
                        mean_data += data_s[i + threadIdx.x * step];
                    }
                    mean_data /= n_samples_template;
                }

                for(i = 0; i < n_samples_template; i++) {
                    data_sample = data_s[i + threadIdx.x * step] - mean_data;
                    numerator += data_sample * templates_s[i];
                    sum_square_data += data_sample * data_sample;
                    
                }
                //denominator = sum_square_data * sum_square_template[sum_square_template_offset];
                denominator = sum_square_data * ss_template[0];
                if (cc_mat_offset < (chunk_size * n_stations * n_components)){
                    // check that this thread is not ouf of the chunk's bounds
                    if (denominator > STABILITY_THRESHOLD) {
                        cc_mat[cc_mat_offset] = numerator * rsqrtf(denominator);
                    }
                }
            }
        }
        __syncthreads(); // wait for every thread to finish before leaving the kernel
    }
}

//-------------------------------------------------------------------------
__global__ void sum_cc(float *cc_mat, float *cc_sum, float *weights,
        int n_stations, int n_components, int n_corr, int chunk_offset, int chunk_size) {

    int i, ch;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( ((i + chunk_offset) < n_corr) & (i < chunk_size) ){
        // first condition: check if we are not outside cc_sum's length
        // second condition: check if we are not outside the chunk's size
        float *cc_mat_offset;

        cc_mat_offset = cc_mat + i * n_stations * n_components;
        for (ch = 0; ch < (n_stations * n_components); ch++){
            cc_sum[i] += cc_mat_offset[ch] * weights[ch];
        }
    }
}

//-------------------------------------------------------------------------
void matched_filter(float *templates, float *sum_square_templates, 
                    int *moveouts, float *data, float *weights, size_t step,
                    size_t n_samples_template, size_t n_samples_data,
                    size_t n_templates, size_t n_stations, size_t n_components, size_t n_corr,
                    float *cc_sums, int normalize) {

    int t_global = -1;
    int nGPUs;

    // find the number of available GPUs
    cudaGetDeviceCount(&nGPUs);
    omp_set_num_threads(min(nGPUs, (int)n_templates));

    int chunk_size = n_corr/NCHUNKS + 1;

    // Size of variables to create on the device (GPU)
    size_t sizeof_templates = sizeof(float) * n_samples_template * n_stations * n_components * n_templates;
    size_t sizeof_moveouts = sizeof(int) * n_components * n_stations * n_templates;
    size_t sizeof_data = sizeof(float) * n_samples_data * n_stations * n_components;
    size_t sizeof_cc_mat = sizeof(float) * chunk_size * n_stations * n_components; // cc matrix for one template (and one chunk of data)
    size_t sizeof_cc_sum = sizeof(float) * chunk_size; // cc sums for one template (and one chunk of data)
    size_t sizeof_sum_square_templates = sizeof(float) * n_templates * n_stations * n_components;
    size_t sizeof_weights = sizeof(float) * n_templates * n_stations * n_components;
    size_t sizeof_total = sizeof_templates + sizeof_moveouts + sizeof_data + sizeof_cc_mat + sizeof_cc_sum + sizeof_sum_square_templates + sizeof_weights;

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
        cudaMalloc((void**)&weights_d, sizeof_weights);

        // transfer the inputs from host to the GPU
        cudaMemcpy(templates_d, templates, sizeof_templates, cudaMemcpyHostToDevice);
        cudaMemcpy(moveouts_d, moveouts, sizeof_moveouts, cudaMemcpyHostToDevice);
        cudaMemcpy(data_d, data, sizeof_data, cudaMemcpyHostToDevice);
        cudaMemcpy(sum_square_templates_d, sum_square_templates, sizeof_sum_square_templates, cudaMemcpyHostToDevice);
        cudaMemcpy(weights_d, weights, sizeof_weights, cudaMemcpyHostToDevice);

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

            // calculate the space required in the shared memory
            int count_template = (n_samples_template / WARPSIZE + 1) * WARPSIZE;
            int count_data = ((n_samples_template + BLOCKSIZE * step) / WARPSIZE + 1) * WARPSIZE;
            int sharedMem = (count_template + count_data + 1) * sizeof(float);
            if (sharedMem > maxSharedMem) {
                int new_step = (maxSharedMem/sizeof(float) - 2 * n_samples_template - 2 * WARPSIZE) / BLOCKSIZE;
                int new_length = maxSharedMem/sizeof(float) - count_data - WARPSIZE;
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
            moveouts_t = moveouts + t_thread * n_stations * n_components;
            max_moveout = 0;
            for (int i = 0; i < (n_stations * n_components); i++) {
                max_moveout = (moveouts_t[i] > max_moveout) ? moveouts_t[i] : max_moveout;
            }
            n_corr_t = (n_samples_data - n_samples_template - max_moveout) / step + 1;

            // local pointers on the device
            templates_d_t = templates_d + t_thread * n_samples_template * n_stations * n_components;
            sum_square_templates_d_t = sum_square_templates_d + t_thread * n_stations * n_components;
            moveouts_d_t = moveouts_d + t_thread * n_stations * n_components;
            weights_d_t = weights_d + t_thread * n_stations * n_components;
            
            for (int ch = 0; ch < NCHUNKS; ch++){
                int chunk_offset = ch * chunk_size;
                int cs;
                // make sure the chunk is not going out of bounds
                if (chunk_offset + chunk_size > n_corr_t){
                    cs = n_corr_t - chunk_offset;
                    if (cs <= 0) continue;
                }
                else{
                    cs = chunk_size;
                }
                //sizeof_cc_mat = sizeof(float) * cs * n_stations * n_components;
                size_t sizeof_cc_sum_chunk = sizeof(float) * cs;

                // define block and grid sizes for kernels
                dim3 BS(BLOCKSIZE);
                dim3 GS(ceilf(cs / (float)BS.x) * n_stations);

                // process
                cudaMemset(cc_mat_d, 0, sizeof_cc_mat); // initialize cc_mat to 0
                network_corr<<<GS, BS, sharedMem>>>(templates_d_t, 
                                                    sum_square_templates_d_t, 
                                                    moveouts_d_t, 
                                                    data_d,
                                                    weights_d_t,
                                                    step, 
                                                    n_samples_template,
                                                    n_samples_data, 
                                                    n_stations,
                                                    n_components,
                                                    chunk_offset,
                                                    cs,
                                                    cc_mat_d,
                                                    normalize);

                // return an error if something happened in the kernel (and crash the program)
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());

                // weighted sum of correlation coefficients
                cudaMemset(cc_sum_d, 0, sizeof_cc_sum);

                // using a small block size seems to improve the speed of sum_cc 
                dim3 BS_sum(32);
                dim3 GS_sum(ceilf(cs / (float)BS_sum.x));
                sum_cc<<<GS_sum, BS_sum>>>(cc_mat_d, cc_sum_d, weights_d_t, n_stations, n_components, n_corr_t, chunk_offset, cs);

                // return an error if something happened in the kernel (and crash the program)
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());

                // xfer cc_sum back to host
                cc_sums_t = cc_sums + t_thread * n_corr + chunk_offset;
                cudaMemcpy(cc_sums_t, cc_sum_d, sizeof_cc_sum_chunk, cudaMemcpyDeviceToHost);
            }
            cudaDeviceSynchronize();
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
