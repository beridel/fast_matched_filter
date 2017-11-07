/*
:copyright:
    William B. Frank and Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
*/

#include <mex.h>
#include <math.h>
#include <matrix.h>
#include "matched_filter_CPU.h"

void mexFunction(int nOutputs, mxArray *ptrOutputs[], int nInputs, const mxArray *ptrInputs[])
{
    float *templates = NULL, *sum_square_templates = NULL; // template input
    int *moveouts = NULL, max_moveout = 0; // template moveout
    float *data = NULL; // data input
    double *square_data, *csum_square_data = NULL;
    float *csum_square_data_f = NULL;
    float *weights = NULL; // weights for each CC
    int n_samples_template, n_samples_data, step;
    int n_templates, n_stations, n_components, n_corr; // size input
    float *cc_sum; // output
    int i, t, station_offset, s, c;
    int n_samples_out;
    int data_offset;
    
    /* check for good number of inputs/outputs */
    if (nInputs != 12)
        mexErrMsgIdAndTxt("Matlab:matched_filter.c", "Thirteen inputs required: \
                 templates (float*), \
                 sum_square_templates (float*), \
                 moveouts (int*), \
                 data (float*), \
                 csum_square_data (float*), \
                 weights (float*), \
                 step (int), \
                 n_samples_template (int*), \
                 n_samples_data (int), \
                 n_templates (int), \
                 n_stations (int), \
                 n_components (int), \
                 n_corr (int).");

    if (nOutputs != 1)
        mexErrMsgIdAndTxt("Matlab:matched_filter.c",
                "One output required.");
    
    /* read in inputs */
    templates = (float*)mxGetData(ptrInputs[0]);
    sum_square_templates = (float*)mxGetData(ptrInputs[1]);
    moveouts = (int*)mxGetData(ptrInputs[2]);
    data = (float*)mxGetData(ptrInputs[3]);
    //csum_square_data = (float*)mxGetData(ptrInputs[4]);
    weights = (float*)mxGetData(ptrInputs[4]);
    step = (int)mxGetScalar(ptrInputs[5]);
    n_samples_template = (int)mxGetScalar(ptrInputs[6]);
    n_samples_data = (int)mxGetScalar(ptrInputs[7]);
    n_templates = (int)mxGetScalar(ptrInputs[8]);
    n_stations = (int)mxGetScalar(ptrInputs[9]);
    n_components = (int)mxGetScalar(ptrInputs[10]);
    n_corr = (int)mxGetScalar(ptrInputs[11]);
   
    /* prepare outputs */
    n_samples_out = n_corr * n_templates;
    ptrOutputs[0] = mxCreateNumericArray(1, &(n_samples_out), mxSINGLE_CLASS, mxREAL);
    cc_sum = (float*)mxGetData(ptrOutputs[0]);

    /* precompute cumulative sum of squares for data */
    square_data = (double*)malloc(n_samples_data * n_stations * n_components * sizeof(double));
    csum_square_data = (double*)malloc(n_samples_data * n_stations * n_components * sizeof(double));
    csum_square_data_f = (float*)malloc(n_samples_data * n_stations * n_components * sizeof(float));
    
    for (s = 0; s < n_stations; s++) {
        for (c = 0; c < n_components; c++) {
            data_offset = n_samples_data * (s * n_components + c);
            
            for (i = 0; i < n_samples_data; i++) square_data[data_offset + i] = (double)(data[data_offset + i] * data[data_offset + i]);
            csum(square_data, n_samples_template, n_samples_data, n_stations, n_components, csum_square_data);
            for (i = 0; i < n_samples_data; i++) csum_square_data_f[data_offset + i] = (float)csum_square_data[data_offset + i];
        }
    }
    
    free(square_data);
    free(csum_square_data);
   
    /* and do the math */
    matched_filter(templates,
                   sum_square_templates,
                   moveouts,
                   data,
                   csum_square_data_f,
                   weights,
                   step,
                   n_samples_template,
                   n_samples_data,
                   n_templates,
                   n_stations,
                   n_components,
                   n_corr,
                   cc_sum); // output variable

    free(csum_square_data_f);
}

