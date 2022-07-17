"""
Python bindings for the fast_matched_filter C libraries

:copyright:
    William B. Frank and Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import numpy as np
import ctypes as ct
import os

path = os.path.join(os.path.dirname(__file__), 'lib')
CPU_LOADED = False
GPU_LOADED = False

argtypes = [
    ct.POINTER(ct.c_float),    # templates
    ct.POINTER(ct.c_float),    # sum of squares of templates
    ct.POINTER(ct.c_int),      # moveouts
    ct.POINTER(ct.c_float),    # data
    ct.POINTER(ct.c_float),    # weights
    ct.c_size_t,               # step
    ct.c_size_t,               # n_samples_template
    ct.c_size_t,               # n_samples_data
    ct.c_size_t,               # n_templates
    ct.c_size_t,               # n_stations
    ct.c_size_t,               # n_components
    ct.c_size_t,               # n_corr
    ct.POINTER(ct.c_float)     # cc_sums or cc
    ]

try:
    _libCPU = ct.cdll.LoadLibrary(os.path.join(path, 'matched_filter_CPU.so'))
    _libCPU.matched_filter.argtypes = argtypes
    _libCPU.matched_filter_precise.argtypes = argtypes \
            + [ct.c_int] # normalize
    _libCPU.matched_filter_no_sum.argtypes = argtypes
    _libCPU.matched_filter_precise_no_sum.argtypes = argtypes \
            + [ct.c_int] # normalize
    CPU_LOADED = True

except OSError:
    print("Matched-filter CPU is not compiled! Should be here: {}".
          format(os.path.join(path, 'matched_filter_CPU.so')))
    CPU_LOADED = False

try:
    _libGPU = ct.cdll.LoadLibrary(os.path.join(path, 'matched_filter_GPU.so'))
    _libGPU.matched_filter.argtypes = argtypes
    GPU_LOADED = True

except OSError:
    print("Matched-filter GPU is not compiled! Should be here: {}".
          format(os.path.join(path, 'matched_filter_GPU.so')))
    GPU_LOADED = False


def matched_filter(templates, moveouts, weights, data, step, arch='cpu', 
                   check_zeros='first', normalize='short', network_sum=True):
    """Compute the correlation coefficients between `templates` and `data`.

    Scan the continuous waveforms `data` with the template waveforms
    `templates` given the relative propagation times `moveouts` and compute
    a time series of summed correlation coefficients. The weighted sum is
    defined by `weights`. Try `normalize='full'` and/or `arch='precise' or 'gpu'`
    to achieve better numerical precision.

    Parameters
    -----------
    templates: numpy.ndarray
        4D (n_templates, n_stations, n_channels, n_tp_samples) or 3D 
        (n_templates, n_traces, n_tp_samples) `numpy.ndarray` with the
        template waveforms.
    moveouts: numpy.ndarray, int
        3D (n_templates, n_stations, n_channels) or 2D (n_templates, n_stations)
        `numpy.ndarray` with the moveouts, in samples.
    weights: numpy.ndarray, float
        3D (n_templates, n_stations, n_channels) or 2D (n_stations, n_channels)
        `numpy.ndarray` with the channel weights. For a given template, the
        largest possible correlation coefficient is given by the sum of the
        weights. Make sure that the weights sum to one if you want CCs between
        1 and -1.
    data: numpy.ndarray
        3D (n_stations, n_channels, n_samples) or 2D (n_traces, n_samples)
        `numpy.ndarray` with the continuous waveforms.
    step: scalar, int
        Time interval, in samples, between consecutive correlations.
    arch: string, optional
        One `'cpu'`, `'precise'` or `'gpu'`. The `'precise'` implementation
        is a CPU implementation that slower but more accurate than `'cpu'`.
        The GPU implementation is used if `arch='gpu'`. Default is `'cpu'`.
    check_zeros: string, optional
        Controls the verbosity level at the end of this routine when
        checking zeros in the time series of correlation coefficients (CCs).
        - False: No messages.  
        - `'first'`: Check zeros on the first template's CCs (recommended).  
        - `'all'`: Check zeros on each template's CCs. It can be useful for
        troubleshooting but in general this would print too many messages.  

        Default is `'first'`.
    normalize: string, optional
        Either "short" or "full" - full is slower but removes the mean of the
        data at every correlation. Short is the original implementation.
        NB: When using normalize="short", the templates and the data sliding
        windows must have zero means (high-pass filter the data if necessary).
    network_sum: boolean, default to True
        If True, returns the weighted sum of correlation coefficients across
        the station network. If False, returns the correlation coefficients
        for each channel.

    Returns
    --------
    cc: numpy.ndarray, float
        If `network_sum=True`, 2D (n_templates, n_correlations) `numpy.ndarray`.
        If `network_sum=False`, 4D (n_templates, n_stations, n_components,
        n_correlations) `numpy.ndarray`. The number of correlations is
        controlled by `step`.
    """
    assert normalize in ("short", "full"), "Only know short or full normalization methods"
    if normalize == "full":
        normalize = 1
        assert arch != "cpu", "Full normalization not supported with cpu arch - try arch=precise"
    else:
        normalize = 0

    if arch.lower() == 'cpu' and CPU_LOADED is False:
        loaded = False
    if arch.lower() == 'precise' and CPU_LOADED is False:
        loaded = False
    elif arch.lower() == 'gpu' and GPU_LOADED is False:
        loaded = False
    else:
        loaded = True

    if not loaded:
        print("Compiled library for {} not loaded; exiting!".format(arch))
        return

    # figure out and check input formats
    impossible_dimensions = False
    if templates.ndim > data.ndim:
        n_templates = int(templates.shape[0])

        assert templates.shape[1] == data.shape[0] # check stations
        n_stations = int(templates.shape[1])

        if templates.ndim == 4:
            assert templates.shape[2] == data.shape[1] # check components
            n_components = int(templates.shape[2])
        elif templates.ndim == 3:
            n_components = int(1)
        else:
            impossible_dimensions = True

    elif templates.ndim == data.ndim:
        n_templates = int(1)
        
        assert templates.shape[0] == data.shape[0] # check stations
        n_stations = int(templates.shape[0])

        if templates.ndim == 3:
            assert templates.shape[1] == data.shape[1] # check components
            n_components = int(templates.shape[1])
        elif templates.ndim == 2:
            n_components = int(1)
        else:
            impossible_dimensions = True

    else:
        impossible_dimensions = True

    if impossible_dimensions:
        print("Template and data dimensions are not compatible!")
        return
   
    n_samples_template = templates.shape[-1]
    if templates.shape != (n_templates, n_stations, n_components, n_samples_template):
        templates = templates.reshape(n_templates, n_stations, n_components, n_samples_template)

    n_samples_data = data.shape[-1]
    if data.shape != (n_stations, n_components, n_samples_data):
        data = data.reshape(n_stations, n_components, n_samples_data)

    assert moveouts.shape == weights.shape

    if moveouts.shape != (n_templates, n_stations, n_components):
        if (n_templates * n_stations * n_components) / moveouts.size == n_components:
            moveouts = np.repeat(moveouts, n_components).reshape(n_templates, n_stations, n_components)
        elif (n_templates * n_stations * n_components) / moveouts.size == 1.:
            moveouts = moveouts.reshape(n_templates, n_stations, n_components)

    if weights.shape != (n_templates, n_stations, n_components):
        if (n_templates * n_stations * n_components) / weights.size == n_components:
            weights = np.repeat(weights, n_components).reshape(n_templates, n_stations, n_components)
        elif (n_templates * n_stations * n_components) / weights.size == 1.:
            weights = weights.reshape(n_templates, n_stations, n_components)

    n_corr = int((n_samples_data - n_samples_template) / step + 1)

    # compute sum of squares for templates
    sum_square_templates = np.sum(templates**2, axis=-1).astype(np.float32)

    templates = np.float32(templates.flatten())
    sum_square_templates = sum_square_templates.flatten()

    moveouts = np.int32(moveouts.flatten())
    weights = np.float32(weights.flatten())
    step = int(step)
    # Note: shouldn't need to enforce int here because they were np.int32 before

    data = np.float32(data.flatten())
    if network_sum:
        cc = np.zeros(n_templates * n_corr, dtype=np.float32)
    else:
        cc = np.zeros(n_templates * n_stations * n_components * n_corr, dtype=np.float32)

    # list of arguments
    args = (templates.ctypes.data_as(ct.POINTER(ct.c_float)),
            sum_square_templates.ctypes.data_as(ct.POINTER(ct.c_float)),
            moveouts.ctypes.data_as(ct.POINTER(ct.c_int)),
            data.ctypes.data_as(ct.POINTER(ct.c_float)),
            weights.ctypes.data_as(ct.POINTER(ct.c_float)),
            step,
            n_samples_template,
            n_samples_data,
            n_templates,
            n_stations,
            n_components,
            n_corr,
            cc.ctypes.data_as(ct.POINTER(ct.c_float))
            )

    if arch == 'cpu' and network_sum:
        _libCPU.matched_filter(*args)
    elif arch == 'cpu' and ~network_sum:
        _libCPU.matched_filter_no_sum(*args)
    elif arch == 'precise' and network_sum:
        args = args + (normalize,)
        _libCPU.matched_filter_precise(*args)
    elif arch == 'precise' and ~network_sum:
        args = args + (normalize,)
        _libCPU.matched_filter_precise_no_sum(*args)
    elif arch == 'gpu' and network_sum:
        args = args + (normalize,)
        _libGPU.matched_filter(*args)
    elif arch == 'gpu' and ~network_sum:
        print('no implementation yet!')
        return
        #args = args + (normalize,)
        #_libGPU.matched_filter_no_sum(*args)

    if network_sum:
        cc = cc.reshape(n_templates, n_corr)
    else:
        cc = cc.reshape(n_templates, n_corr, n_stations, n_components)
    # check for zeros in the CC time series more or less thoroughly
    # depending on the value of 'check_zeros'
    if (check_zeros != False) and (check_zeros != 'first') and (check_zeros != 'all'):
        print("check_zeros should be False, 'first', or 'all'. Set it to "
              "the default value: 'first'")
        check_zeros = 'first'
    if not check_zeros:
        pass
    elif check_zeros == 'first':
        # only check the first template
        if network_sum:
            zeros = np.sum(
                    cc[0:1, :int(n_corr - moveouts.max()/step)] == 0.,
                    axis=-1
                    )
        else:
            zeros = np.sum(
                    cc[0:1, :, :, :int(n_corr - moveouts.max()/step)] == 0.,
                    axis=(1, 2, 3)
                    )
    else:
        # check all templates
        zeros = np.sum(
                cc[..., :int(n_corr - moveouts.max()/step)].reshape(
                    n_templates, -1) == 0.,
                axis=-1
                )

    if check_zeros:
        for t in range(zeros.shape[0]):
            if zeros[t] > 10:
                print("{} correlation computations were skipped on the {:d}-th "
                      "template. Can be caused by zeros in data, or too low "
                      "amplitudes (try to increase the gain).".
                      format(zeros[t], t))
    return cc


def test_matched_filter(n_templates=1, n_stations=1, n_components=1,
                        template_duration=10, data_duration=86400,
                        sampling_rate=100, step=1, arch='cpu',
                        check_zeros='first', normalize='short',
                        network_sum=True):
    """Test the `matched_filter` function.  

    Generate random data, templates, and moveouts, and run a matched-filter
    search. The templates are sliced from the data, therefore the maximum
    correlation coefficient should always be one if the program ran normally.
    Try `normalize='full'` and/or `arch='precise' or 'gpu'` to achieve better
    numerical precision.

    Parameters
    ----------
    n_templates: scalar, int, optional
        Number of synthetic templates. Default to 1.
    n_stations: scalar, int, optional
        Number of stations. Default to 1.
    n_components: scalar, int, optional
        Number of components/channels. Default to 1.
    template_duration: scalar, float, optional
        Duration, in seconds, of the template waveforms. Default to 10s.
    data_duration: scalar, float, optional
        Duration, in seconds, of the data waveforms. Default to 86400s.
    sampling_rate: scalar, float, optional
        Sampling frequency (Hz) of the waveforms. Default to 100Hz.
    step: scalar, int
        Time interval, in samples, between consecutive correlations.
    arch: string, optional
        One `'cpu'`, `'precise'` or `'gpu'`. The `'precise'` implementation
        is a CPU implementation that slower but more accurate than `'cpu'`.
        The GPU implementation is used if `arch='gpu'`. Default is `'cpu'`.
    check_zeros: string, optional
        Controls the verbosity level at the end of this routine when
        checking zeros in the time series of correlation coefficients (CCs).  
        - False: No messages.  
        - `'first'`: Check zeros on the first template's CCs (recommended).  
        - `'all'`: Check zeros on each template's CCs. It can be useful for
        troubleshooting but in general this would print too many messages.  

        Default is `'first'`.
    normalize: string, optional
        Either "short" or "full" - full is slower but removes the mean of the
        data at every correlation. Short is the original implementation.
        NB: When using normalize="short", the templates and the data sliding
        windows must have zero means (high-pass filter the data if necessary).
    network_sum: boolean, default to True
        If True, returns the weighted sum of correlation coefficients across
        the station network. If False, returns the correlation coefficients
        for each channel.

    Returns
    --------
    templates: numpy.ndarray
        (n_templates, n_stations, n_components, n_tp_samples) `numpy.ndarray`
        with the random template waveforms generated by the function.
    moveouts: numpy.ndarray
        (n_templates, n_stations, n_components) `numpy.ndarray` with the random
        moveouts generated by the function.
    data: numpy.ndarray
        (n_stations, n_components, n_samples) `numpy.ndarray` with the random
        data generated by the function.
    step: scalar, int
        Time interval, in samples, between consecutive correlations.
    cc_sums: numpy.ndarray, float
        2D (n_templates, n_correlations) `numpy.ndarray`. The number of
        correlations is controlled by `step`.
    run_time: scalar, float
        Time spent by FMF to compute the correlation coefficients.
    """
    from time import time as give_time
    template_times = np.random.random_sample(n_templates) * (data_duration / 2)
    # if step is not 1, not very likely that random times will be found
    if step != 1:
        template_times = np.round(template_times / (step / sampling_rate)) * (step / sampling_rate)
    # determines how many templates there are

    min_moveout = 0
    max_moveout = 10
    moveouts = np.zeros((n_templates, n_stations, n_components))
    for t in range(n_templates):
        for s in range(n_stations):
            moveouts[t, s, :] = (np.random.random_sample(n_components)
                              * (max_moveout - min_moveout)) + min_moveout
    moveouts = np.int32(np.round(moveouts * sampling_rate))

    # generate data
    n_samples_data = data_duration * sampling_rate
    if float(int(n_samples_data)) == float(n_samples_data):
        n_samples_data = np.int32(n_samples_data)
    else:
        print('The data duration times the sampling rate yields a non-integer number of samples !')
        print('Adjust your input parameters so that this product is an integer.')
        return

    data = np.random.random_sample(
            (n_stations, n_components, n_samples_data))\
                    .astype('float32')
    for s in range(n_stations):
        for c in range(n_components):
            data[s, c, :] = data[s, c, :] - np.mean(data[s, c, :])

    # generate templates from data
    n_samples_template = template_duration * sampling_rate
    if float(int(n_samples_template)) == float(n_samples_template):
        n_samples_template = np.int32(n_samples_template)
    else:
        print('The template duration times the sampling rate yields a non-integer number of samples !')
        print('Adjust your input parameters so that this product is an integer.')
        return

    n_templates = template_times.size
    templates = np.zeros((n_templates,
                          n_stations,
                          n_components,
                          n_samples_template),
                          dtype=np.float32)
    for t in range(n_templates):
        start_t = template_times[t] * sampling_rate

        template = np.zeros((n_stations, n_components, n_samples_template))
        for s in range(n_stations):
            for c in range(n_components):
                start = int(start_t + np.round(moveouts[t, s, c]))
                stop = int(start_t + n_samples_template + np.round(moveouts[t, s, c]))
                template[s, c, :n_samples_template] = data[s, c, start:stop]

        templates[t, :, :, :n_samples_template] = template

    weights = np.ones(
            (n_templates, n_stations, n_components),
            dtype=np.float32) / (n_stations * n_components)

    start_time = give_time()
    cc_sum = matched_filter(templates,
                            moveouts,
                            weights,
                            data,
                            step,
                            arch=arch,
                            check_zeros=check_zeros,
                            normalize=normalize,
                            network_sum=network_sum)
    stop_time = give_time()

    print("Matched filter ({}) for {} templates on {} stations/{} "
            "components over {} samples ({} step) ran in {:.3f}s".
          format(arch, n_templates, n_stations, n_components, n_samples_data,
                 step, (stop_time - start_time)))

    return templates, moveouts, data, step, cc_sum, stop_time-start_time

