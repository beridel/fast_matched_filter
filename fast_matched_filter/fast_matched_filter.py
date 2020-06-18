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
import datetime as dt
import os

path = os.path.join(os.path.dirname(__file__), 'lib')
CPU_LOADED = False
GPU_LOADED = False

try:
    _libCPU = ct.cdll.LoadLibrary(os.path.join(path, 'matched_filter_CPU.so'))
    _libCPU.matched_filter.argtypes = [
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
        ct.POINTER(ct.c_float)]    # cc_sums
    _libCPU.matched_filter_precise.argtypes = [
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
        ct.POINTER(ct.c_float)]    # cc_sums
    CPU_LOADED = True

except OSError:
    print("Matched-filter CPU is not compiled! Should be here: {}".
          format(os.path.join(path, 'matched_filter_CPU.so')))
    CPU_LOADED = False

try:
    _libGPU = ct.cdll.LoadLibrary(os.path.join(path, 'matched_filter_GPU.so'))
    _libGPU.matched_filter.argtypes = [
        ct.POINTER(ct.c_float),    # templates
        ct.POINTER(ct.c_float),    # sum_square_templates
        ct.POINTER(ct.c_int),      # moveouts
        ct.POINTER(ct.c_float),    # data
        ct.POINTER(ct.c_float),    # stations' weights
        ct.c_size_t,               # step
        ct.c_size_t,               # n_samples_data
        ct.c_size_t,               # n_samples_template
        ct.c_size_t,               # n_templates
        ct.c_size_t,               # n_stations
        ct.c_size_t,               # n_components
        ct.c_size_t,               # n_corr
        ct.POINTER(ct.c_float)]    # cc_sums
    GPU_LOADED = True

except OSError:
    print("Matched-filter GPU is not compiled! Should be here: {}".
          format(os.path.join(path, 'matched_filter_GPU.so')))
    GPU_LOADED = False


def matched_filter(templates, moveouts, weights, data, step, arch='cpu'):
    """
    input:
    templates ---------- 4D numpy array [templates x stations x
                         components x time]
                         or 3D numpy array [templates x traces x time]
    moveouts ----------- 3D numpy array [templates x stations x components]
                         or 2D numpy array [templates x traces]
    weights ------------ 3D numpy array [templates x stations x components]
                         or 2D numpy array [templates x traces]
    data --------------- 3D numpy array [stations x components x
                         time]
                         or 2D numpy array [traces x time]
    step --------------- interval between correlations (in samples)
    arch --------------- 'cpu' or 'gpu' implementation
                         new: 'precise' for a more precise but slower
                         CPU implementation

    NB: Mean and trend MUST be removed from template and data traces before
        using this function

    output:
    2D numpy array (np.float32) [templates x time (at step defined interval)]
    """

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
        n_templates = np.int32(templates.shape[0])

        assert templates.shape[1] == data.shape[0] # check stations
        n_stations = np.int32(templates.shape[1])

        if templates.ndim == 4:
            assert templates.shape[2] == data.shape[1] # check components
            n_components = np.int32(templates.shape[2])
        elif templates.ndim == 3:
            n_components = np.int32(1)
        else:
            impossible_dimensions = True

    elif templates.ndim == data.ndim:
        n_templates = np.int32(1)
        
        assert templates.shape[0] == data.shape[0] # check stations
        n_stations = np.int32(templates.shape[0])

        if templates.ndim == 3:
            assert templates.shape[1] == data.shape[1] # check components
            n_components = np.int32(templates.shape[1])
        elif templates.ndim == 2:
            n_components = np.int32(1)
        else:
            impossible_dimensions = True

    else:
        impossible_dimensions = True
    
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

    if impossible_dimensions:
        print("Template and data dimensions are not compatible!")
        return

    n_corr = np.int32((n_samples_data - n_samples_template) / step + 1)

    # compute sum of squares for templates
    sum_square_templates = np.sum(templates**2, axis=-1).astype(np.float32)

    templates = np.float32(templates.flatten())
    sum_square_templates = sum_square_templates.flatten()

    moveouts = np.int32(moveouts.flatten())
    weights = np.float32(weights.flatten())
    step = np.int32(step)
    # Note: shouldn't need to enforce int here because they were np.int32 before

    data = np.float32(data.flatten())
    cc_sums = np.zeros(n_templates * n_corr, dtype=np.float32)

    if arch == 'cpu':
        _libCPU.matched_filter(
            templates.ctypes.data_as(ct.POINTER(ct.c_float)),
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
            cc_sums.ctypes.data_as(ct.POINTER(ct.c_float)))

    if arch == 'precise':
        _libCPU.matched_filter_precise(
            templates.ctypes.data_as(ct.POINTER(ct.c_float)),
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
            cc_sums.ctypes.data_as(ct.POINTER(ct.c_float)))
    
    elif arch == 'gpu':
        _libGPU.matched_filter(
                templates.ctypes.data_as(ct.POINTER(ct.c_float)),
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
                cc_sums.ctypes.data_as(ct.POINTER(ct.c_float)))

    cc_sums = cc_sums.reshape((n_templates, n_corr))
    zeros = np.sum(cc_sums[0, :int(n_corr - moveouts.max() / step)] == 0.)
    if zeros > 10:
        print("{} correlation computations were skipped. Can be caused by"
              " zeros in data, or too low amplitudes (try to increase the "
              "gain).".format(zeros))
    return cc_sums


def test_matched_filter(n_templates=1, n_stations=1, n_components=1,
                        template_duration=10, data_duration=86400,
                        sampling_rate=100, step=1, arch='cpu'):
    """
    output: templates, moveouts, data, step, cc_sum
    """

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
    moveouts = np.round(moveouts * sampling_rate)

    # generate data
    n_samples_data = data_duration * sampling_rate
    if float(int(n_samples_data)) == float(n_samples_data):
        n_samples_data = np.int32(n_samples_data)
    else:
        print('The data duration times the sampling rate yields a non-integer number of samples !')
        print('Adjust your input parameters so that this product is an integer.')
        return

    data = np.random.random_sample((n_stations, n_components, n_samples_data))
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
                          n_samples_template))
    for t in range(n_templates):
        start_t = template_times[t] * sampling_rate

        template = np.zeros((n_stations, n_components, n_samples_template))
        for s in range(n_stations):
            for c in range(n_components):
                start = int(start_t + np.round(moveouts[t, s, c]))
                stop = int(start_t + n_samples_template + np.round(moveouts[t, s, c]))
                template[s, c, :n_samples_template] = data[s, c, start:stop]

        templates[t, :, :, :n_samples_template] = template

    weights = np.ones((n_templates, n_stations, n_components)) / (n_stations * n_components)

    start_time = dt.datetime.now()
    cc_sum = matched_filter(templates,
                            moveouts,
                            weights,
                            data,
                            step,
                            arch=arch)
    stop_time = dt.datetime.now()

    print("Matched filter ({}) for {} templates on {} stations/{} "
          "components over {} samples ({} step) ran in {}s".
          format(arch, n_templates, n_stations, n_components, n_samples_data,
                 step, (stop_time - start_time).total_seconds()))

    return templates, moveouts, data, step, cc_sum

