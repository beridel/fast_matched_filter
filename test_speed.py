import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import sys
from os import getcwd
sys.path.append(getcwd())
import fast_matched_filter as fmf

#-------- example's parameters ---------
ns = 12 # nb of stations
nc = 3 # nb of components
n_templates = 50 # nb of templates
step = 2 # temporal step (in nb of samples) used between each consecutive correlation
SampR = 50. # Sampling rate (nb of samples per second)
Tdata = 12.*60.*60. # Length in seconds of one data trace
Ttp = 8 # Length in seconds of one template trace
Nsamp_data = np.int32(Tdata * SampR)
Nsamp_tp = np.int32(Ttp * SampR)
max_moveout = 20. # maximum moveout (in seconds) in the randomly generated moveouts 
#---------------------------------------

#------------ Synthetic data and templates ---------------
# --- randomly generate the data ---
Data = np.zeros((ns, nc, Nsamp_data), dtype=np.float32)
for s in xrange(ns):
    for c in xrange(nc):
        Data[s,c,:] = np.random.normal(loc=0., scale=10.+np.random.uniform(-2., 2.), size=Nsamp_data)
# --- randomly generate the moveouts ---
moveouts = np.zeros((n_templates, ns, nc), dtype=np.int32)
for t in xrange(n_templates):
    for s in xrange(ns):
        moveouts[t,s,:] = np.int32(np.int32(np.random.uniform(0., max_moveout, size=nc)) * SampR)
# --- randomly pick the templates from the data ---
start_times = np.int32(np.int32(np.random.uniform(0, Tdata - Ttp - max_moveout, size=n_templates)) * SampR)
templates = np.zeros((n_templates, ns, nc, Nsamp_tp), dtype=np.float32)
for t in xrange(n_templates):
    for s in xrange(ns):
        for c in xrange(nc):
            idx1 = start_times[t] + moveouts[t,s,c]
            idx2 = idx1 + Nsamp_tp
            templates[t,s,c,:] = Data[s,c,idx1:idx2]
# --- equal weights for everybody ---
weights = np.ones((n_templates, ns, nc), dtype=np.float32)
weights /= (ns * nc)
#---------------------------------------------------------
#t1 = dt.datetime.now()
#CCs = matched_filter_GPU_python_wrapper.matched_filter_GPU(templates, Data, moveouts, weights, step, verbose=True)
#t2 = dt.datetime.now()
#
#print "%i correlations of %i template(s) (counting %i stations and %i channels each) with the data performed in %.2fsec on CPUs/GPUs" \
#      %(CCs.shape[-1], t, ns, nc, (t2-t1).total_seconds())
#
#for t in xrange(CCs.shape[0]):
#    # should corresponds
#    print "Template %i: Event start time = %i samples, Maximum coherency at %i samples" %(t, start_times[t], CCs[t,:].argmax())

#----------------------------------------------------------
#------------- FIRST TEST -------------------
#times = np.zeros(n_templates, dtype=np.float32)
#CCs = matched_filter_GPU_python_wrapper.matched_filter_GPU(templates[:1,:,:,:], Data, moveouts, weights, step, verbose=True)
#for nt in xrange(n_templates):
#    t1 = dt.datetime.now()
#    CCs = matched_filter_GPU_python_wrapper.matched_filter_GPU(templates[:nt+1,:,:,:], Data, moveouts, weights, step, verbose=True)
#    t2 = dt.datetime.now()
#    times[nt] = (t2-t1).total_seconds()
#with open('times_GPU_test1.dat', 'wb') as f:
#    times.tofile(f)

#times = np.zeros(n_templates, dtype=np.float32)
#CCs = correlate.fftw_multi_normxcorr(templates[:1,:,:,:], moveouts, Data)
#for nt in xrange(n_templates):
#    t1 = dt.datetime.now()
#    CCs = correlate.fftw_multi_normxcorr(templates[:nt+1,:,:,:], moveouts, Data)
#    t2 = dt.datetime.now()
#    times[nt] = (t2-t1).total_seconds()
#    print " -----------> %.2fsec for %i templates." %(times[nt], nt+1)
#with open('times_FREQ_test2.dat', 'wb') as f:
#    times.tofile(f)

#steps = [1,5,10]
steps = [step]

times = np.zeros((len(steps), n_templates), dtype=np.float32)
for i in xrange(len(steps)):
    #CCs = matched_filter_GPU_python_wrapper.matched_filter_GPU(templates[:1,:,:,:], Data, moveouts, weights, steps[i], verbose=True)
    #print "Cache operation"
    for nt in xrange(n_templates):
        t1 = dt.datetime.now()
        #CCs = matched_filter_GPU_python_wrapper.matched_filter_GPU(templates[:nt+1,:,:,:], Data, moveouts, weights, steps[i], verbose=True)
        CCs = fmf.matched_filter(templates[:nt+1,:,:,:], moveouts, weights, Data, steps[i], arch='cpu')
        t2 = dt.datetime.now()
        times[i,nt] = (t2-t1).total_seconds()
        print "%i correlations of %i template(s) (counting %i stations and %i channels each) with the data performed in %.2fsec on CPUs/GPUs" \
              %(CCs.shape[-1], CCs.shape[0], ns, nc, times[i,nt])
        
        for t in xrange(CCs.shape[0]):
            # should corresponds
            print "Template %i: Event start time = %i samples, Maximum coherency at %i samples" %(t, start_times[t], CCs[t,:].argmax()*steps[i])
#         
#        #t1 = dt.datetime.now()
#        ##CCs = func.matched_filter(templates[:nt+1,:,:,:], weights, moveouts, Data, steps[i])
#        #CCs = wrapper.matched_filter(templates[:nt+1,:,:,:], weights, moveouts, Data, steps[i])
#        #t2 = dt.datetime.now()
#        #times[i,nt] = (t2-t1).total_seconds()
#        #print "%i correlations of %i template(s) (counting %i stations and %i channels each) with the data performed in %.2fsec on CPUs" \
#        #      %(CCs.shape[-1], CCs.shape[0], ns, nc, times[i,nt])
#        #
#        #for t in xrange(CCs.shape[0]):
#        #    # should corresponds
#        #    print "Template %i: Event start time = %i samples, Maximum coherency at %i samples" %(t, start_times[t], CCs[t,:].argmax())
#
#
#with open('times_4GPU.dat', 'wb') as f:
#    times.tofile(f)


#Ndays = 4*7
#T1 = dt.datetime.now()
#for i in range(Ndays):
#    # 4 weeks
#    print "------ Day %i / %i --------" %(i+1, Ndays)
#    # --- randomly generate the data ---
#    Data = np.zeros((ns, nc, Nsamp_data), dtype=np.float32)
#    for s in xrange(ns):
#        for c in xrange(nc):
#            Data[s,c,:] = np.random.normal(loc=0., scale=10.+np.random.uniform(-2., 2.), size=Nsamp_data)
#    t1 = dt.datetime.now()
#    #CCs = matched_filter_GPU_python_wrapper.matched_filter_GPU(templates[:nt+1,:,:,:], Data, moveouts, weights, steps[i], verbose=True)
#    CCs = np.empty((templates.shape[0], 0), dtype=np.float32)
#    Nchunks = 3
#    L = Data.shape[-1]/Nchunks
#    for i in range(Nchunks):
#        i0 = i*L
#        i1 = (i+1)*L
#        if i1 > Data.shape[-1]:
#            i1 = Data.shape[-1]
#        CCs = np.hstack( (CCs, fmf.matched_filter(templates, weights, moveouts, Data[:,:,i0:i1], 1, arch='gpu')) )
#    t2 = dt.datetime.now()
#    print "%i correlations of %i template(s) (counting %i stations and %i channels each) with the data performed in %.2fsec on CPUs/GPUs" \
#          %(CCs.shape[-1], CCs.shape[0], ns, nc, (t2-t1).total_seconds())
#T2 = dt.datetime.now()
#print "Analysis of %i days in %.2f minutes." %(T2-T1).total_seconds()/60.
