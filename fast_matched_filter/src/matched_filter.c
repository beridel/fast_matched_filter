/*
:copyright:
    William B. Frank and Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
*/

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "matched_filter_CPU.h"

#define STABILITY_THRESHOLD 0.000001f

//-------------------------------------------------------------------------
void matched_filter(float *templates, float *sum_square_templates, int *moveouts,
                    float *data, 
                    float *weights, int step, int n_samples_template, int n_samples_data,
                    int n_templates, int n_stations, int n_components, int n_corr,
                    float *cc_sum) { // output variable

    int t, ch, i;
    int start_i, stop_i, cc_i;
    int min_moveout, max_moveout;
    int network_offset, station_offset, cc_sum_offset;
    int *moveouts_t = NULL;
    float *templates_t = NULL, *sum_square_templates_t = NULL, *weights_t = NULL;
    double *csum_square_data = NULL;


    // compute cumulative sum of squares of data
    csum_square_data = malloc(((n_samples_data + 1) * n_stations * n_components) * sizeof(double));
    memset(csum_square_data, 0., ((n_samples_data + 1) * n_stations * n_components) * sizeof(double));
    cumsum_square_data(data, n_samples_data, weights, n_stations, n_components, csum_square_data);

    // run matched filter template by template
    for (t = 0; t < n_templates; t++) {
        network_offset = t * n_stations * n_components;
        station_offset = t * n_stations;
        cc_sum_offset = t * n_corr;

        // find min/max moveout and template vector position
        min_moveout = 0;
        max_moveout = 0;
        for (ch = 0; ch < (n_stations * n_components); ch++) {
            if (moveouts[network_offset + ch] < min_moveout) min_moveout = moveouts[network_offset + ch];
            if (moveouts[network_offset + ch] > max_moveout) max_moveout = moveouts[network_offset + ch];
        }
    
        templates_t = templates + network_offset * n_samples_template;
        moveouts_t = moveouts + network_offset;
        weights_t = weights + network_offset;
        sum_square_templates_t = sum_square_templates + network_offset;

        start_i = (int)(ceilf(abs(min_moveout) / (float)step)) * step;
        stop_i = 1 + (n_samples_data - n_samples_template - max_moveout);

#pragma omp parallel for private(i, cc_i)
        for (i = start_i; i < stop_i; i += step) {
            cc_i = i / step;
            cc_sum[cc_sum_offset + cc_i] = network_corr(templates_t,
                                                        sum_square_templates_t,
                                                        moveouts_t,
                                                        data + i,
                                                        csum_square_data + i + 1,
                                                        weights_t,
                                                        n_samples_template,
                                                        n_samples_data,
                                                        n_stations,
                                                        n_components);
        }
    }

    free(csum_square_data);
}
 
//-------------------------------------------------------------------------
float network_corr(float *templates, float *sum_square_template, int *moveouts,
                   float *data, double *csum_square_data, float *weights,
                   int n_samples_template, int n_samples_data, int n_stations, int n_components) {

    int s, c, d, dd, t;
    int station_offset, component_offset;
    float cc, cc_sum = 0; // output
 
    for (s = 0; s < n_stations; s++) {
        
        station_offset = s * n_components;

        cc = 0;        
        for (c = 0; c < n_components; c++) {
            component_offset = station_offset + c;
            if (weights[component_offset] == 0) continue;

            // if ((i + moveouts[component_offset]) > n_samples_data - n_samples_template) continue;

            t  = component_offset * n_samples_template;
            d  = component_offset * n_samples_data + moveouts[component_offset];
            dd = component_offset * (n_samples_data + 1) + moveouts[component_offset];
            
            cc = corrc(templates + t,
                       sum_square_template[component_offset],
                       data + d,
                       csum_square_data + dd,
                       n_samples_template);
            cc_sum += cc * weights[component_offset];
        }
    }
    
    return cc_sum;
}
 
//-------------------------------------------------------------------------
float corrc(float *templates, float sum_square_template,
            float *data, double *csum_square_data,
            int n_samples_template) {

    int i;
    float numerator = 0, denominator = 0, cc = 0;
    
    for (i = 0; i < n_samples_template; i++){
        numerator += templates[i] * data[i];
    }
    denominator = sum_square_template * (float)(csum_square_data[n_samples_template - 1] - csum_square_data[-1]);

    if (denominator > STABILITY_THRESHOLD) cc = numerator / sqrt(denominator);
    return cc;
}

//-------------------------------------------------------------------------
void cumsum_square_data(float *data, int n_samples_data, 
                        float *weights, int n_stations, int n_components,
                        double *csum_square_data) {
    int ch, data_offset, csum_offset;

    // loop over channels
#pragma omp parallel for private(ch, data_offset, csum_offset)
    for (ch = 0; ch < n_stations * n_components; ch++) {

        data_offset = ch * n_samples_data;
        csum_offset = ch * (n_samples_data + 1) + 1;

        neumaier_cumsum_squared(data + data_offset, n_samples_data,
                                csum_square_data + csum_offset);
    }
}

//-------------------------------------------------------------------------
void neumaier_cumsum_squared(float *array, int length, double *cumsum) {
    int i;
    double running_sum, square, temporary;
    double correction = 0.0;

    running_sum = (double)array[0] * (double)array[0];
    cumsum[0] = running_sum;
    for (i = 1; i < length; i++) {
        square = (double)array[i] * (double)array[i];
        temporary = running_sum + square;

        if (fabsf(running_sum) >= fabsf(square)) {
            correction += (running_sum - temporary) + square;
        }
        else {
            correction += (square - temporary) + running_sum;
        }

        running_sum = temporary;
        cumsum[i] = temporary + correction;
    }
}


//-------------------------------------------------------------------------
void matched_filter_precise(float *templates, float *sum_square_templates, int *moveouts,
                            float *data,
                            float *weights, int step, int n_samples_template, int n_samples_data,
                            int n_templates, int n_stations, int n_components, int n_corr,
                            float *cc_sum, int normalize) { // output variable

    int t, ch, i;
    int start_i, stop_i, cc_i;
    int min_moveout, max_moveout;
    int network_offset, station_offset, cc_sum_offset;
    int *moveouts_t = NULL;
    float *templates_t = NULL, *sum_square_templates_t = NULL, *weights_t = NULL;

    // run matched filter template by template
    for (t = 0; t < n_templates; t++) {
        network_offset = t * n_stations * n_components;
        station_offset = t * n_stations;
        cc_sum_offset = t * n_corr;

        // find min/max moveout and template vector position
        min_moveout = 0;
        max_moveout = 0;
        for (ch = 0; ch < (n_stations * n_components); ch++) {
            if (moveouts[network_offset + ch] < min_moveout) min_moveout = moveouts[network_offset + ch];
            if (moveouts[network_offset + ch] > max_moveout) max_moveout = moveouts[network_offset + ch];
        }

        templates_t = templates + network_offset * n_samples_template;
        moveouts_t = moveouts + network_offset;
        weights_t = weights + network_offset;
        sum_square_templates_t = sum_square_templates + network_offset;

        start_i = (int)(ceilf(abs(min_moveout) / (float)step)) * step;
        stop_i = 1 + (n_samples_data - n_samples_template - max_moveout);

#pragma omp parallel for private(i, cc_i)
        for (i = start_i; i < stop_i; i += step) {
            cc_i = i / step;
            cc_sum[cc_sum_offset + cc_i] = network_corr_precise(templates_t,
                                                                sum_square_templates_t,
                                                                moveouts_t,
                                                                data + i,
                                                                weights_t,
                                                                n_samples_template,
                                                                n_samples_data,
                                                                n_stations,
                                                                n_components, 
                                                                normalize);
        }
    }
}

//-------------------------------------------------------------------------
float network_corr_precise(
    float *templates, float *sum_square_template, int *moveouts,
    float *data, float *weights, int n_samples_template, int n_samples_data, 
    int n_stations, int n_components, int normalize) {

    int s, c, d, dd, t;
    int station_offset, component_offset;
    float cc, cc_sum = 0; // output

    for (s = 0; s < n_stations; s++) {

        station_offset = s * n_components;

        cc = 0;
        for (c = 0; c < n_components; c++) {
            component_offset = station_offset + c;
            if (weights[component_offset] == 0) continue;

            // if ((i + moveouts[component_offset]) > n_samples_data - n_samples_template) continue;

            t  = component_offset * n_samples_template;
            d  = component_offset * n_samples_data + moveouts[component_offset];
            dd = component_offset * (n_samples_data + 1) + moveouts[component_offset];

            cc = corrc_precise(templates + t,
                               sum_square_template[component_offset],
                               data + d,
                               n_samples_template, normalize);
            cc_sum += cc * weights[component_offset];
        }
    }

    return cc_sum;
}

//-------------------------------------------------------------------------
float corrc_precise(float *templates, float sum_square_template,
            float *data, int n_samples_template, int normalize) {

    int i;
    float numerator = 0, sum_square_data = 0, denominator = 0, cc = 0, mean=0, sample;

    if (normalize > 0){
        for (i = 0; i < n_samples_template; i++){
            mean += data[i];
        }
        mean /= n_samples_template;
    }

    for (i = 0; i < n_samples_template; i++){
        sample = data[i] - mean;
        numerator += templates[i] * sample;
        sum_square_data += sample * sample;
    }
    denominator = sqrt(sum_square_template * sum_square_data);

    if (denominator > STABILITY_THRESHOLD){ cc = numerator / denominator; }
    return cc;
}
