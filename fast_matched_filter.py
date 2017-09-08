import numpy as np
import ctypes as C
import datetime as dt
import inspect, os 

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

try: 
    _libCPU = C.cdll.LoadLibrary(path + '/matched_filter_CPU.so')
    _libCPU.matched_filter.argtypes = [C.POINTER(C.c_float),    # templates
                                       C.POINTER(C.c_float),    # sum of squares of templates
                                       C.POINTER(C.c_int),      # moveouts
                                       C.POINTER(C.c_float),    # data
                                       C.POINTER(C.c_float),    # data csum squared
                                       C.POINTER(C.c_float),    # weights
                                       C.c_size_t,              # step
                                       C.c_size_t,              # n_samples_template
                                       C.c_size_t,              # n_samples_data
                                       C.c_size_t,              # n_templates
                                       C.c_size_t,              # n_stations
                                       C.c_size_t,              # n_components
                                       C.c_size_t,              # n_corr
                                       C.POINTER(C.c_float)]    # cc_sums
    cpu_loaded = True

except:
    print("Matched-filter CPU is not compiled! Should be here: {}".format(path + '/matched_filter_CPU.so'))
    cpu_loaded = False

try:
    _libGPU = C.cdll.LoadLibrary(path + '/matched_filter_GPU.so')
    _libGPU.matched_filter.argtypes = [C.POINTER(C.c_float),    # templates
                                       C.POINTER(C.c_float),    # sum_square_templates
                                       C.POINTER(C.c_int),      # moveouts
                                       C.POINTER(C.c_float),    # data
                                       C.POINTER(C.c_float),    # stations' weights
                                       C.c_size_t,              # step
                                       C.c_size_t,              # n_samples_data
                                       C.c_size_t,              # n_samples_template
                                       C.c_size_t,              # n_templates
                                       C.c_size_t,              # n_stations
                                       C.c_size_t,              # n_components
                                       C.c_size_t,              # n_corr
                                       C.POINTER(C.c_float)]    # cc_sums
    gpu_loaded = True

except:
    print("Matched-filter GPU is not compiled! Should be here: {}".format (path + '/matched_filter_GPU.so'))
    gpu_loaded = False


def matched_filter(templates, weights, moveouts, data, step, arch='cpu'):
    """
    input:
    templates - 4D numpy array (np.float32) [templates x stations x components x time]
    n_samples_template - 2D numpy array (np.int32) [templates x stations]
    moveouts - 2D numpy array (np.int32) [templates x stations]
    data - 3D numpy array (np.float32) [stations x components x time]
    step - np.int32 interval between correlations (in samples)

    NB: Mean and trend MUST be removed from template and data traces before using this function

    output:
    2D numpy array (np.float32) [templates x time (at step defined interval)]
    """
    
    if arch == 'cpu' and cpu_loaded == False:
        loaded = False
    elif arch == 'gpu' and gpu_loaded == False:
        loaded = False
    else:
        loaded = True

    if not loaded:
        print("Compiled library for {} not loaded; exiting!".format(arch))
        sys.exit()

    n_templates = np.int32(templates.shape[0])
    n_stations = np.int32(data.shape[0])
    n_components = np.int32(data.shape[1])
    n_samples_template = np.int32(templates.shape[-1])
    n_samples_data = np.int32(data.shape[-1])
    n_corr = np.int32((n_samples_data - n_samples_template) / step + 1)

    # compute sum of squares for templates
    sum_square_templates = np.zeros((n_templates, n_stations, n_components), dtype=np.float32)
    for t in range(n_templates):
        for s in range(n_stations):
            for c in range(n_components):
                sum_square_templates[t, s, c] = np.sum(templates[t, s, c, :n_samples_template] ** 2)

    templates = np.float32(templates.flatten())
    sum_square_templates = np.float32(sum_square_templates.flatten())
    moveouts = np.int32(moveouts.flatten())
    weights = np.float32(weights.flatten())
    step = np.int32(step)
    cc_sums = np.zeros(int(n_templates) * int(n_corr), dtype=np.float32)

    if arch == 'cpu':
        # compute square of data
        csum_square_data = np.cumsum(np.insert(data, 0, 0, axis=-1) ** 2, axis=-1)
        csum_square_data = np.float32(csum_square_data.flatten())
        data = np.float32(data.flatten())

        _libCPU.matched_filter(
            templates.ctypes.data_as(C.POINTER(C.c_float)),
            sum_square_templates.ctypes.data_as(C.POINTER(C.c_float)),
            moveouts.ctypes.data_as(C.POINTER(C.c_int)),
            data.ctypes.data_as(C.POINTER(C.c_float)),
            csum_square_data.ctypes.data_as(C.POINTER(C.c_float)),
            weights.ctypes.data_as(C.POINTER(C.c_float)),
            step,
            n_samples_template,
            n_samples_data,
            n_templates,
            n_stations,
            n_components,
            n_corr,
            cc_sums.ctypes.data_as(C.POINTER(C.c_float)))

    elif arch == 'gpu':
        data = np.float32(data.flatten())
        
        _libGPU.matched_filter(
                templates.ctypes.data_as(C.POINTER(C.c_float)),
                sum_square_templates.ctypes.data_as(C.POINTER(C.c_float)),
                moveouts.ctypes.data_as(C.POINTER(C.c_int)),
                data.ctypes.data_as(C.POINTER(C.c_float)),
                weights.ctypes.data_as(C.POINTER(C.c_float)),
                step,
                n_samples_template,
                n_samples_data,
                n_templates,
                n_stations,
                n_components,
                n_corr,
                cc_sums.ctypes.data_as(C.POINTER(C.c_float)))

    return cc_sums.reshape((n_templates, n_corr))


def test_matched_filter(n_templates=1, n_stations=1, n_components=1, template_duration=10, data_duration=86400, sampling_rate=100, step=1, arch='cpu'):
    """
    output: templates, moveouts, data, step, cc_sum
    """

    template_times = np.ones((n_templates)) * 5 # determines how many templates there are

    min_moveout = 0
    max_moveout = 10
    moveouts = np.zeros((n_templates, n_stations))
    for t in range(n_templates):
        moveouts[t, :] = np.random.random_sample(n_stations) * (max_moveout - min_moveout) + min_moveout
    moveouts = np.round(moveouts * sampling_rate)

    # generate data
    n_samples_data = data_duration * sampling_rate;
    data = np.random.random_sample((n_stations, n_components, n_samples_data))
    for s in range(n_stations):
        for c in range(n_components):
            data[s, c, :] = data[s, c, :] - np.mean(data[s, c, :])

    # generate templates from data
    n_samples_template = template_duration * sampling_rate
    n_templates = template_times.size
    templates = np.zeros((n_templates, n_stations, n_components, n_samples_template))
    for t in range(n_templates):
        start_t = template_times[t] * sampling_rate

        template = np.zeros((n_stations, n_components, n_samples_template))
        for s in range(n_stations):
            start = int(start_t + np.round(moveouts[t, s]))
            stop = int(start_t + n_samples_template + np.round(moveouts[t, s]))
            template[s, :, :n_samples_template] = data[s, :, start:stop]

        templates[t, :, :, :n_samples_template] = template

    weights = np.ones((n_templates, n_stations)) / n_stations
           
    start_time = dt.datetime.now()
    cc_sum = matched_filter(templates, weights, moveouts, data, step, arch=arch)
    stop_time = dt.datetime.now()

    print("Matched filter ({}) for {} templates on {} stations/{} components over {} samples ({} step) ran in {}s".format(
        arch, n_templates, n_stations, n_components, n_samples_data, step, (stop_time - start_time).total_seconds()))

    return templates, moveouts, data, step, cc_sum

