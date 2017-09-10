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
#include "matched_filter_CPU.h"

//-------------------------------------------------------------------------
void matched_filter(float *templates, float *sum_square_templates, int *moveouts,
                    float *data, float *csum_square_data,
                    float *weights, int step, int n_samples_template, int n_samples_data,
                    int n_templates, int n_stations, int n_components, int n_corr,
                    float *cc_sum) { // output variable

    int t, s, i;
    int start_i, stop_i, cc_i;
    int min_moveout, max_moveout;
    int network_offset, station_offset, cc_sum_offset;
    int *moveouts_t = NULL;
    float *templates_t = NULL, *sum_square_templates_t = NULL;

    // run matched filter template by template
    for (t = 0; t < n_templates; t++) {
        network_offset = t * n_stations * n_components;
        station_offset = t * n_stations;
        cc_sum_offset = t * n_corr;

        // find min/max moveout and template vector position
        min_moveout = 0;
        max_moveout = 0;
        for (s = 0; s < n_stations; s++) {
            if (moveouts[station_offset + s] < min_moveout) min_moveout = moveouts[station_offset + s];
            if (moveouts[station_offset + s] > max_moveout) max_moveout = moveouts[station_offset + s];
        }
    
        templates_t = templates + network_offset * n_samples_template;
        moveouts_t = moveouts + station_offset;
        sum_square_templates_t = sum_square_templates + network_offset;

        start_i = (int)(ceilf(abs(min_moveout) / (float)step)) * step;
        stop_i = n_samples_data - n_samples_template - max_moveout - step;

#pragma omp parallel for private(i, cc_i)
        for (i = start_i; i < stop_i; i += step) {
            cc_i = i / step;
            cc_sum[cc_sum_offset + cc_i] = network_corr(templates_t,
                                                        sum_square_templates_t,
                                                        moveouts_t,
                                                        data + i,
                                                        csum_square_data + i,
                                                        weights,
                                                        n_samples_template,
                                                        n_samples_data,
                                                        n_stations,
                                                        n_components);
        }
    }
}
 
//-------------------------------------------------------------------------
float network_corr(float *templates, float *sum_square_template, int *moveouts,
                   float *data, float *csum_square_data, float *weights,
                   int n_samples_template, int n_samples_data, int n_stations, int n_components) {

    int s, c, d, dd, t;
    int station_offset, component_offset;
    float cc, cc_sum = 0; // output
 
    for (s = 0; s < n_stations; s++) {
        if (weights[s] == 0) continue;
        
        station_offset = s * n_components;

        cc = 0;        
        for (c = 0; c < n_components; c++) {
            component_offset = station_offset + c;

            t = component_offset * n_samples_template;
            d = component_offset * n_samples_data + moveouts[s];
            dd = component_offset * (n_samples_data + 1) + moveouts[s]; // take into account added leading zero

            
            cc += corrc(templates + t,
                        sum_square_template[component_offset],
                        data + d,
                        csum_square_data + dd,
                        n_samples_template);
        }
        cc_sum += cc / n_components * weights[s];
    }
    
    return cc_sum;
}
 
//-------------------------------------------------------------------------
float corrc(float *templates, float sum_square_template,
            float *data, float *csum_square_data,
            int n_samples_template) {

    int i;
    float numerator = 0, sum_square_data;
    
    for (i = 0; i < n_samples_template; i++) numerator += templates[i] * data[i];
    sum_square_data = csum_square_data[i] - csum_square_data[0]; // note i == n_samples_template
   
    return numerator / sqrtf(sum_square_template * sum_square_data);
}

