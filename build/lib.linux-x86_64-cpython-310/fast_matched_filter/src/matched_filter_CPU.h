void matched_filter(
        float*, float*, int*, float*, float*,
        size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, float*
        );
float network_corr(
        float*, float*, int*, float*, double*,
        float*, size_t, size_t, size_t, size_t
        );
float corrc(
        float*, float, float*, double*, size_t
        );

void matched_filter_precise(
        float*, float*, int*, float*, float*,
        size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, float*, int
        );
float network_corr_precise(
        float*, float*, int*, float*, float*,
        size_t, size_t, size_t, size_t, int
        );
float corrc_precise(
        float*, float, float*, size_t, int
        );

void matched_filter_no_sum(
        float*, float*, int*, float*, float*,
        size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, float*
        );
void network_corr_no_sum(
        float*, float*, int*, float*, double*,
        float*, size_t, size_t, size_t, size_t,
        float*
        );
float corrc_no_sum(
        float*, float, float*, double*, size_t
        );

void matched_filter_precise_no_sum(
        float*, float*, int*, float*, float*,
        size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, float*, int
        );
void network_corr_precise_no_sum(
        float*, float*, int*, float*, float*,
        size_t, size_t, size_t, size_t, int,
        float*
        );
float corrc_precise_no_sum(
        float*, float, float*, size_t, int
        );

void matched_filter_variable_precise(
    float*, float*, int*,
    float*, float*, size_t,
    int*, size_t, size_t, size_t,
    size_t, size_t, float*, int);

float network_corr_variable_precise(
    float*, float*, int*,
    float *, float*, float*, int*, size_t,
    size_t, size_t, size_t, int);


void cumsum_square_data(
        float*, size_t, float*, size_t, size_t, double*
        );
void neumaier_cumsum_squared(
        float*, size_t, double*
        );

