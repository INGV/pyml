# Authors: Raffaele Di Stefano (raffaele.distefano@ingv.it), Barbara Castello (barbara.castello@ingv.it)
# Licence: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# PyAmp reads in SAC file(s) and does the following:
# - apply an adaptative filter based on the results of signal to noise analysis
# - converts the signal (by deconvolution of the instrument and convolution to Wood-Anderson)
# - estimates the maximum elongation (peak to peak) of the seismic signal with three different possible approaches
# - calculates channel ML on all the three components
# - calculates event ML when a set of waveform (of one single event) is given

# Amplitudes Methods
#    based on literature and on the approaches followed in several codes (open like earthworm/localmag or other proprietary ones)
#    we defined three possible approaches
#    - free: the minimum peak and the maximum peak are searched for through the 
#      whole piece of waveform defined by P onset and S onset with no regard for 
#      the time distance between the two; this approach does not force in any way 
#      the finding of close max and min and a later choice on how large this time 
#      distance can be to consider the measure reliable is applyied; advantage is 
#      that good results (close peaks) is really good with no a-priori condition; 
#      disadvantage is that a number of waveforms might not be used for ML calculation
#    - lmag: like in earthworm, the min and maximum peaks are searched for 
#      within a window of x seconds (tipically 0.8s) moving within the signal
#       search interval; after all the x secondos sub-windows are checked, 
#      the largest peak-to-peak is taken and the corresponding min and max are 
#      taken as the measure of min and max amplitude; advantage is that a measure 
#      will always come out; disadvantage: the quality of the result depends strictly 
#      on the moving window size and a value will always come out unless the waveform 
#      is discarded for other reasons
#    - swing: this approach is the one followed by Massimo Di Bona (INGV senior seismologist)
#      in its studies on #      the new attenuation law for Italy; it searches the highest 
#      peak-to-peak measure only between adjacent peaks, a negative and a positive or viceversa;
#      this is a pure "swing" approach; same advantages as lmag plus the fact that it is not 
#      forced by a sub-window lenght because the two peaks must be adjacent anyway; disadvantage 
#      is that it is not guaranteed that the two peaks are giving the maximum elongation 
#      of the signal because the frequency giving the maximum elongation might not be 
#      represented by the same swing
#    Note: the free approach allows to explore the whole waveform and lets the waveform 
#      itself giving hints on how reliable is the analysis at the cost of loosing some 
#      measure; the free approach, when a maximum distance between the peaks is applyied 
#      at the ML calculation stage is quite similar to lmag with the difference that two 
#      peaks will not necessarily be found within the given maximum distance
#      The advantage of PyAmp is that results of the three different approaches can be compared
#      
# Filters
#    PyAmp uses an adaptative approach to pre-filtering based on the results of a 
#    spectral analysis aimed to find the proper corner frequencies peculiar for the specific
#    waveform allowing to cut out those frequencies in which the signal to noise ratio is low
#    A fixed bandpass filter does not cope with different frquency content depending on the 
#    magnitude of an event or characteristics of the sensor at the specific site and the related
#    noise content; anyway this analysis for any reason can fail: in this case you can decide to
#    force a static bandpass (--forcebp) or to skip the waveform considering it not reliable for 
#    ML calculation; in some cases the spectral analysis finds the upper corner but not the lower
#    which comes out being zero: in this case a fixed lower corner frequency is used; at present
#    (2020/11/26) it is not possible to disable the adaptative filter unless it fails.
#    This approach to the pre-filtring had an important contribution by Marco Cattaneo (INGV senior
#    seismologist) in the framework of the EPSO/NFO-TABOO project and INGV-ANCONA/Marche Region 
#    Civil Protection agreement.
# 
# Attenuation Law
#    PyAmp implemets two different attenuation laws for the ML calculation:
#    - INGV Hutton & Boore which is mutuated by the one formerly used at USGS for the 
#      adapted to the Italian region with no stations corrections
#    - M. Di Bona attenuation law calculated specifically for the Italian region
#      with stations corrections: at present the stations corrections are not used
#      because the orignial aim of PyAmp was to calculate the amplitudes while the 
#      ML calculation runtime was considered to be only indicative to test the results
#      against the INGV seismic database, but in the end a file withe stations corrections
#      will be loaded and corrections applied
#
# Outliers removal
#    In ML calculation outliers are removed by a recursive calculation of the mean, median and 
#    standard deviation
#    
# ML calculation
#    Two different approaches are followed to calculate the station ML:
#    - the mean of the maximum amplitudes N and E is used in the attenuation law to get the station ML
#    - the mean of the N and E ML is performed to give the station ML
import argparse,sys,os,glob,copy,pwd,pathlib,itertools
import geographiclib
import pandas
import numpy
import math
from pyrocko import cake
from scipy import signal, stats
from scipy.signal import butter, lfilter, freqz,find_peaks,detrend,welch,hilbert
# Imports from obpsy
from obspy import read, UTCDateTime, Stream as st
from obspy import Trace as tr
from obspy import signal as obsi
from obspy.core.util import attribdict
from obspy.geodetics.base import gps2dist_azimuth as distaz
import obspy.io.sac as sac
from obspy.io.sac.util import SacInvalidContentError
from obspy.signal.quality_control import MSEEDMetadata 
from obspy.core.inventory import read_inventory as rinv
from decimal import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xml.etree import ElementTree as ET
from sklearn import preprocessing
from scipy.fft import fft, ifft, fftfreq

import matplotlib.pyplot as plt
import ray
#import datetime
#import time
#import getpass
#import socket
#
#from math import *
#from datetime import datetime
#import obspy.core

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def parseArguments():
        parser=MyParser()	
        parser.add_argument('--inpath',      default='.',            help='Full path to waveform file')
        parser.add_argument('--infile',      default=False,          help='Waveform File Name')
        parser.add_argument('--responsepath',default='.',            help='Full path to response files')
        parser.add_argument('--model',       default='model.tvel',   help='1D Velocity Model Path/Name')
        parser.add_argument('--plot',                                help='To plot or not to plot',action='store_true')
        parser.add_argument('--nopicks',                             help='Ignore the presence of both P and S picks if any',action='store_true')
        parser.add_argument('--forcebp',                             help='It forces bandpass if spectral analysis fails',action='store_true')
        parser.add_argument('--peakmethod',  default='all',          help='Choose between: swing (same wave swing), free (min and max at free distance), lmag (same as localmag within a moving window), all (output contains all)')
        parser.add_argument('--event_magnitude',                     help='It is used only if --infile is omitted; pyamp performs event ML calculation (based on the pyamp.conf related section parameters)',action='store_true')
        parser.add_argument('--dbona_corr',  default='dbcor.csv',    help='Input file with DiBona Stations corrections')
        parser.add_argument('--conf',        default='./pyamp.conf', help='A file containing sections and related parameters (see the example)')
        if len(sys.argv)==1:
            parser.print_help()
            sys.exit(1)
        args=parser.parse_args()
        return args

# Depending on the python version the configparser has different names so ... try
try:
    import ConfigParser as cp
except ImportError:
    import configparser as cp

# Build a dictionary from config file section
def get_config_dictionary(cfg, section):
    dict1 = {}
    options = cfg.options(section)
    for option in options:
        try:
            dict1[option] = cfg.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

def butter_bandpass(data,lowcut, highcut, fs, order):
    nyq = 0.5 * float(fs)
    low = float(lowcut) / nyq 
    high = float(highcut) / nyq 
    b, a = butter(order, [low, high], btype='bandpass',analog=False)
    y = lfilter(b, a, data)
    return y

def get_trace_time_info(t):
    try:
       t_start = UTCDateTime(t.reftime)+t.b
    except:
       t_start=False
    try:
       t_end = UTCDateTime(t.reftime)+t.e
    except:
       t_end=False
    try:
       oti = UTCDateTime(t.reftime)+t.o
    except:
       oti=False
    try:
       a_sec = Decimal(t.a)-Decimal(t.b)
       pti = UTCDateTime(t.reftime)+t.a
    except:
       a_sec = False
       pti=False
    try:
       t0_sec=Decimal(t.t0)-Decimal(t.b)
       sti=UTCDateTime(t.reftime)+t.t0
    except Exception as s:
       t0_sec = False
       sti=False
    d=1.0/t.delta
    return d,t_start,t_end,oti,pti,sti 


def rxml(f):
    xd=ET.parse('./AQU.xml').getroot()
    for k,v in enumerate(xd):
        if 'Network' in v.tag:
            for j,u in enumerate(xd[k]):
                if 'Station' in u.tag:
                    for i,m in enumerate(xd[k][j]):
                        if 'Channel' in m.tag:
                           try:
                              if m.attrib['code'] == 'HHN':
                                 for h,l in enumerate(xd[k][j][i]):
                                     if 'Sensor' in l.tag:
                                           for o,n in enumerate(xd[k][j][i][h]):
                                                if 'Description' in n.tag:
                                                   sens=xd[k][j][i][h][o].text
                                     if 'DataLogger' in l.tag:
                                           for o,n in enumerate(xd[k][j][i][h]):
                                                #print(n.tag)
                                                if 'Description' in n.tag:
                                                   dlog=xd[k][j][i][h][o].text
                           except:
                              sens=False
                              dlog=False
    return(sens,dlog)

def selewinidx(arrayofpositions):
    band_start=[]
    band_stop=[]
    arrayofpositions_sub=[]
    idsta=0
    idsto=0
    idx0 = 0
    idx1 = 1
    band_start.append(arrayofpositions[idx0]) # Store in the "good ones" the first element of the array of indexs of the over_snr frequencies
    while idx1 < len(arrayofpositions):  # but if the indexes are not sequential adds the idx0 to STOP and IDX1 to new start, creating parallel arrays with windows extremes
          if arrayofpositions[idx1] != arrayofpositions[idx0]+1:
             band_start.append(arrayofpositions[idx1])
             band_stop.append(arrayofpositions[idx0])
          idx0=idx1
          idx1=idx0+1
    band_stop.append(arrayofpositions[idx0])
    idmax=None
    idxBands = 0
    deltaidmax=0
    while idxBands < len(band_start):
          if (band_stop[idxBands]-band_start[idxBands]) >= deltaidmax:
             idmax = idxBands
             deltaidmax = band_stop[idxBands]-band_start[idxBands]
          idxBands = idxBands+1
    if idmax != None:
       idsta=arrayofpositions.index(band_start[idmax])
       idsto=arrayofpositions.index(band_stop[idmax])
       arrayofpositions_sub=arrayofpositions[idsta:idsto+1]
    return arrayofpositions_sub

def maxratio(r,f,snr):
    rati_val = None
    freq_val = None
    lo_co = None
    up_co = None

    over_idx=[]
    rr = r.tolist()
    for ratio in rr:
        if ratio >= snr:
#           print 'Ratio= ' + str(ratio)
           over_idx.append(rr.index(ratio))
    if len(over_idx) != 0:
           over_idx_sub=selewinidx(over_idx)
           rati_val = r[over_idx_sub]
           freq_val = f[over_idx_sub]
           lo_co = round(min(freq_val))
           up_co = round(max(freq_val))
    else:
           lo_co = False
           up_co = False
    return lo_co,up_co,len(over_idx)

def signal_to_noise(t,fs,ne1,ne2,se1,se2,snr):
    n=t.to_obspy_trace().slice(ne1,ne2)
    n.data=detrend(n.data,type='constant')
    n.taper(max_percentage=0.05, type='cosine')

    s=t.to_obspy_trace().slice(se1,se2)
    s.data=detrend(s.data,type='constant')
    s.taper(max_percentage=0.05, type='cosine')

    nperseg1 = n.data.shape[-1]
    nperseg2 = s.data.shape[-1]
    nperseg=int(round(min([nperseg1, nperseg2])/8))

    try:
        f, Pxx_denN = welch(n.data, fs, window='hanning', nperseg=nperseg)
        f, Pxx_denS = welch(s.data, fs, window='hanning', nperseg=nperseg)
        #print Pxx_denN
        #print Pxx_denS
        #print f
        Pxx_den = numpy.divide(Pxx_denS,Pxx_denN)
    except Exception as e:
        sys.stderr.write(' '.join(("Something Wrong in signal_to_noise",str(e),'\n')))

    return maxratio(Pxx_den,f,snr),Pxx_den

def to_woodanderson(trace,time,response_file,units):
    #pre_filter = (0.10, 1.0, 10.0, 40.0)
    paz_wa1 = {'sensitivity': 2800, 'zeros': [0j,0j], 'gain': 1, 'poles': [-6.2832 +4.7124j,-6.2832 -4.7124j]}
    paz_wa2 = {'sensitivity': 2080, 'zeros': [0j,0j], 'gain': 1, 'poles': [-5.4978 +5.6089j, -5.4978 -5.6089j]}
    date=UTCDateTime(time)
    response_parameters = {'filename': response_file,
                           'date': date,
                           'units': units # Units to return response in ('DIS', 'VEL' or ACC)
                          }
    #return trace.to_obspy_trace().simulate(paz_simulate=paz_wa1, seedresp=response_parameters,pre_filt=pre_filter)
    return trace.to_obspy_trace().simulate(paz_simulate=paz_wa1, seedresp=response_parameters)

def plot(figure,trace,color,secondary,a1id,a2id,a1,a2,label,b1,b2):
    # b1 should be a list of two X for picks bars
    # b2 should be a list of four X for two windows
    amps=trace.data
    i1=0
    i2=len(amps)-1
    index=list(range(i1,i2))
    samples=list(amps[i1:i2]) 
    pd_data_frame = pandas.DataFrame(samples, index , columns=['Amplitudes']) 
    time_series = pd_data_frame['Amplitudes'] 
    wave_trace = go.Scatter(
                 x = index,
                 y = time_series, 
                 mode = 'lines', 
                 name = label + ' ' + tr.kstnm + ' ' + tr.kcmpnm,
                 line_color=color)
    if a1id and a2id and a1 and a2:
       figure.add_trace(wave_trace,secondary_y=True)
       figure.add_annotation(
               x = a1id,
               y = a1,
               xref="x",
               yref="y2",
               text="Min Amp",
               showarrow=True,
               font=dict(family="Courier New, monospace", size=16, color="#ffffff"),
               align="center",
               arrowhead=2,
               arrowsize=1,
               arrowwidth=2,
               arrowcolor="#636363",
               ax=20,
               ay=-30,
               bordercolor="#c7c7c7",
               borderwidth=2,
               borderpad=4,
               bgcolor="#ff7f0e",
               opacity=0.8
       )
       figure.add_annotation(
               x = a2id,
               y = a2,
               xref="x",
               yref="y2",
               text="Max Amp",
               showarrow=True,
               font=dict(family="Courier New, monospace", size=16, color="#ffffff"),
               align="center",
               arrowhead=2,
               arrowsize=1,
               arrowwidth=2,
               arrowcolor="#636363",
               ax=20,
               ay=-30,
               bordercolor="#c7c7c7",
               borderwidth=2,
               borderpad=4,
               bgcolor="#ff7f0e",
               opacity=0.8
       )
       if b1:
          figure.add_vline(x=b1[0], line_width=2, line_dash="dash", line_color="blue")
          figure.add_vline(x=b1[1], line_width=2, line_dash="dash", line_color="red")
       if b2:
          figure.add_vrect(x0=b2[0], x1=b2[1], line_color="brown", line_width=1) # The noise box for spectral analysis
          figure.add_vrect(x0=b2[4], x1=b2[5], line_color="green", line_width=1) # The search box for min-max amplitude search
          figure.add_vrect(x0=b2[2], x1=b2[3], line_color="brown", line_width=1) # The signal box for spectral analysis
    else:
       figure.add_trace(wave_trace)
    return figure

def get_position(t,d,w):
    # t is the full trace
    # d is list of 5 timing data from trace
    # w is list of windowing data setup

    dx3 = int((d[3]-d[1])*d[0])+1
    dx4 = int((d[4]-d[1])*d[0])+1
    wx0 = int((w[0]-d[1])*d[0])+1
    wx1 = int((w[1]-d[1])*d[0])+1
    wx2 = int((w[2]-d[1])*d[0])+1
    wx3 = int((w[3]-d[1])*d[0])+1
    return wx0,wx1,wx2,wx3,dx3,dx4

def get_amps(t,i1,i2): # Free search method looking for maximum and minimum within the while window search at whatever distance
    i=list(range(0,len(t.data)-1))
    s=list(t.data[0:len(t.data)-1]) 
    df = pandas.DataFrame(s, i , columns=['amplitudes'])  
    idxmax = df.amplitudes[i1:i2].idxmax()
    idxmin = df.amplitudes[i1:i2].idxmin()
    amax = df.amplitudes.iloc[idxmax]
    amin = df.amplitudes.iloc[idxmin]
    return(idxmin,idxmax,amin,amax)

def get_amps_lmag(t,i1,i2,w): # LocalMag like method searching the maximum min2max over several consecutive windows moving of one sample in a win large w
    ws = int((w/t.stats.delta))+1 # the moving winodow from seconds to samples
    i=list(range(0,len(t.data)-1))
    s=list(t.data[0:len(t.data)-1]) 
    df = pandas.DataFrame(s, i , columns=['amplitudes'])  
    n=i1
    p2pmax=0
    while n <= (i2-ws) in df.index:
        n2=n+ws
        ia0 = df.amplitudes[n:n2].idxmax()
        ia1 = df.amplitudes[n:n2].idxmin()
        a0 = df.amplitudes.iloc[ia0]
        a1 = df.amplitudes.iloc[ia1]
        p2p=abs(a0-a1)
        if p2p >= p2pmax:
           p2pmax = p2p
           amax=a0
           amin=a1
           idxmax=ia0
           idxmin=ia1
        n+=1
    return(idxmin,idxmax,amin,amax)

def get_amps_swing(t,i1,i2): # Massimo Di Bona (INGV) like method looking for a min to max peak span on the same swing of the waveform: a min consecutive to a max or viceversa
    index=list(range(i1,i2))
    i=list(range(0,len(t.data)-1))
    s=list(t.data[0:len(t.data)-1]) 
    df = pandas.DataFrame(s, i , columns=['amplitudes'])  
    # Finding Positive Peaks with find_peaks on the waveform
    indices_plus  = find_peaks(df.amplitudes[i1:i2])[0]
    peaks_plus=[df.amplitudes[i1:i2][index[j]] for j in indices_plus]
    index_peaks_plus=[index[j] for j in indices_plus]
    # Finding Negative Peaks on the negative of the waveform
    indices_minus = find_peaks(-df.amplitudes[i1:i2])[0] # find_peaks finds max amps (peaks) of the negativized series corresponding to the negative peaks
    peaks_minus=[df.amplitudes[i1:i2][index[j]] for j in indices_minus]
    index_peaks_minus=[index[j] for j in indices_minus]
    ipeak = numpy.sort(numpy.concatenate((index_peaks_plus, index_peaks_minus)))
    maxp2p=0
    n=0
    lipeak=list(ipeak)
    ldata=list(t.data)
    while n < len(lipeak)-1:
        a0=ldata[lipeak[n]]
        a1=ldata[lipeak[n+1]]
        if a0 <= 0 and a1 >= 0 or a0 >= 0 and a1 <= 0:
           if abs(a0-a1) > maxp2p:
              maxp2p = abs(a0-a1)
              ia0=lipeak[n]
              ia1=lipeak[n+1]
        n+=1
    if ldata[ia0] < ldata[ia1]:
       amin=ldata[ia0]
       amax=ldata[ia1]
       idxmin=ia0
       idxmax=ia1
    else:
       amin=ldata[ia1]
       amax=ldata[ia0]
       idxmin=ia1
       idxmax=ia0
    return(idxmin,idxmax,amin,amax)
    #idxmax = df.amplitudes[i1:i2].idxmax()
    #idxmin = df.amplitudes[i1:i2].idxmin()
    #amax = df.amplitudes.iloc[idxmax]
    #amin = df.amplitudes.iloc[idxmin]
    #return(idxmin,idxmax,amin,amax)

def theoretical_phase_onset(mf,onset,d,z):
    a = numpy.asfarray([d]) * cake.m2d
    m = cake.load_model(mf)
    ph = cake.PhaseDef(onset)
    try:
       t = m.arrivals(a, phases=ph, zstart=z)[0].t
    except:
       t = 99999.
    return t


def reset(p):
    return make_subplots(specs=[[{"secondary_y": True}]]) if p else False

def ck(x):
    n=2
    cpd = pandas.DataFrame(x.astype(int),columns=['A'])
    gb = cpd.groupby((cpd.A != cpd.A.shift()).cumsum())
    df_target = gb.filter(lambda x: len(x) >= n)
    for i in df_target.iterrows():
        print(i)
    #for k, v in cpd.groupby((cpd['A'].shift() == cpd['A'])):
    #    print(f'[group {k}]')
    #    print(v)

def huttonboore(a,d,s,uc):
    # if uc (use_stcorr_hb) is True s is set to its entrance value and if this value is not False it is set to its value
    # if uc (use_stcorr_hb) is True s is set to its entrance value and if this value is False it is set to ZERO
    # if uc (use_stcorr_hb) is False s entrance value is overwritten and it is set to ZERO and then in the second condition it is ZERO (so it is True because it has a value) and remains ZERO
    s = 0 if not uc else s # station correction is set to 0 if use_stcorr_hb is False
    s = 0 if not s else s # station correction is set to 0 if it is False
    try:
       m = math.log10(a) + 1.110*math.log10(d / 100.) + 0.00189*(d - 100.) + 3.0 + s # Hutton & Boore with s added but not confirmed if it is correct
    except:
       m = False
    return m

def dibona(a,d,s,uc):
    # if uc (use_stcorr_db) is True s is set to its entrance value and if this value is not False it is set to its value
    # if uc (use_stcorr_db) is True s is set to its entrance value and if this value is False it is set to ZERO
    # if uc (use_stcorr_db) is False s entrance value is overwritten and it is set to ZERO and then in the second condition it is ZERO (so it is True because it has a value) and remains ZERO
    s = 0 if not uc else s # station correction is set to 0 if use_stcorr_hb is False
    s = 0 if not s else s # station correction is set to 0 if it is False
    try:
       m = math.log10(a) + 1.667*math.log10(d / 100.) + 0.001736*(d - 100.) + 3.0 + s # Massimo Di Bona
    except:
       m = False
    return m


#ml = math.log10(amp) + 1.792176*math.log10(tr.dist/100.) + 0.001428*(tr.dist-100.)  + 3.00 -s_cattaneo # Ancona/Cattaneo/TABOO Formula
#ml = math.log10(amp*1000) + 1.11*math.log10(tr.dist) + 0.00189*tr.dist - 2.09 # ISC Formula
#ml = math.log10(amp/1000.) + 1.110*math.log10(tr.dist) + 0.00189*(tr.dist) + 3.591

def create_sets(keys,cmpn,cmpe,mtd,mid,mad,dp,mty,whstc,stc):
    # mtd is the peakmethod
    # mty is the huttonboore 0, dibona 1
    # a channel cmpn is [ml,amp,tr.dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
    #   where ml is a list of two: [ml_hutton,ml_dibona]
    meanmag_ml_set=[]
    meanamp_ml_set=[]
    for k in keys:
        kk=k+'_'+mtd
        if kk in cmpn and kk in cmpe: # if both components are present in the set
           epidist = (cmpn[kk][2] + cmpe[kk][2])/2 # epicentral distance
           if epidist >= mid and epidist <= mad: # if epicentral distance is within the accepted range
              if mtd != 'free' or (mtd == 'free' and abs(cmpn[kk][6]-cmpn[kk][5]) <= dp and abs(cmpe[kk][6]-cmpe[kk][5]) <= dp): # if the method is not free the deltapeak has no meaning orherwise it is evaluated
                 ipodist = (cmpn[kk][3] + cmpe[kk][3])/2 # ipocentral distance
                 #Mean of channel magnitudes is calculated
                 if cmpn[kk][0][mty] and cmpe[kk][0][mty]:
                    mm = (cmpn[kk][0][mty] + cmpe[kk][0][mty])/2
                    meanmag_ml_set.append(mm)
                 # Magnitudes of Mean channel amplitudes is calculated if 
                 if not stc or (cmpn[kk][4][mty] and cmpe[kk][4][mty]) or whstc: 
                    #mean_amp = (cmpn[kk][1] + cmpe[kk][1])/2 # Artimetic mean
                    mean_amp_geo = math.sqrt((cmpn[kk][1] * cmpe[kk][1])) # Geometric mean that is the correct one according to Richter and Di Bona
                    corr = (cmpn[kk][4][mty] + cmpe[kk][4][mty])/2 if cmpn[kk][4][mty] and cmpe[kk][4][mty] else False
                    if mty == 0:
                       ma = huttonboore(mean_amp_geo,ipodist,corr,stc)
                    if mty == 1:
                       ma = dibona(mean_amp_geo,ipodist,corr,stc)
                    meanamp_ml_set.append(ma)
        #except Exception as e:
        #       sys.stderr.write(' '.join(("Error in magnitudes set building for ",str(kk),'\n')))
    return meanmag_ml_set,meanamp_ml_set
           
def calculate_event_ml(magnitudes,maxit,stop):
    m=numpy.array(magnitudes)
    finished = False
    N = 0
    Ml_Mean = numpy.mean(m)
    Ml_Std  = numpy.std(m)
    Ml_Medi = numpy.median(m)
    #print "--- Before Outliers Removal ---"
    #print Ml_Mean
    #print Ml_Std
    #print Ml_Medi
    Ml_ns_start = len(m)
    while not finished:
          N = N + 1
          Ml_Medi_old = Ml_Medi
          m = numpy.asfarray(list(filter((lambda x: abs(x - Ml_Medi) <= Ml_Std ), m))) # Questa riga contiene la sintassi lambda (funzione anonima) che sostituisce la funzione drop_outliers ora commentata
          if len(m) > 0:
             Ml_Mean = numpy.mean(m)
             Ml_Std  = numpy.std(m)
             Ml_Medi = numpy.median(m)
             deltaMean = abs(Ml_Medi-Ml_Medi_old)
             if deltaMean <= stop or N == maxit:
                finished = True
                Ml_ns = len(m)
                condition='deltaMean' if deltaMean <= stop else 'maxit'
          else:
             finished = True
             Ml_Std  = False
             Ml_Medi = False
             Ml_ns = False
             condition='emptyset'
    return Ml_Medi,Ml_Std,Ml_ns_start,Ml_ns,condition

###### End of Functions ##########
## Main ##
args = parseArguments()
inpath=args.inpath
infile=args.infile
model_file=args.model
responsepath=args.responsepath
method=args.peakmethod

if args.plot and method == 'all':
   sys.stderr.write("\n WARNING: '--plot' is incompatible with '--method all'\n\n")
   sys.exit(2)

# Now loading the configuration file
if os.path.exists(args.conf) and os.path.getsize(args.conf) > 0:
   paramfile=args.conf
else:
   sys.stderr.write("Config file " + args.conf + " not existing or empty\n\n")
   sys.exit(2)
confObj = cp.ConfigParser()
confObj.read(paramfile)

# Here we load some basic filenames where pyamp elaboration is written
# At present only outputs are stored here but this section is intended
# to also host input filenames if usefull or needed (after minor changes
# to pyamp
# Note:
#     when a filename is set to False the output is sent to sys.stdout
#     with the exception of log files which output is sent to sys.stderr
try:
    filenames_parameters=get_config_dictionary(confObj, 'iofilenames')
except Exception as e:
    sys.stderr.write(("\n"+str(e)+"\n\n"))
    sys.exit(1)
try:
    amplitudes_out=eval(filenames_parameters['amplitudes'])
    magnitudes_out=eval(filenames_parameters['magnitudes'])
    picks_out=eval(filenames_parameters['pickslog'])
    resp_out=eval(filenames_parameters['resplog'])
    log_out=eval(filenames_parameters['log'])
except Exception as e:
    sys.stderr.write(("\n"+str(e)+"\n\n"))
    sys.exit(1)

if amplitudes_out:
   amplitudes_out=open(amplitudes_out,'w')
else:
   amplitudes_out=sys.stdout

if magnitudes_out:
   magnitudes_out=open(magnitudes_out,'w')
else:
   magnitudes_out=sys.stdout

if picks_out:
   picks_out=open(picks_out,'w')
else:
   picks_out=sys.stderr

if resp_out:
   resp_out=open(resp_out,'w')
else:
   resp_out=sys.stderr

if log_out:
   log_out=open(log_out,'w')
else:
   log_out=sys.stderr

# Here we load the basic condition to decide if to proceed or not
# If there is no P do we proceed with theoretical or skip the channel?
# If there is no S do we proceed with theoretical or skip the channel?
preconditions_parameters=get_config_dictionary(confObj, 'preconditions')
theoP=eval(preconditions_parameters['theoretical_p'])
theoS=eval(preconditions_parameters['theoretical_s'])

# Loading from configuration file parameters for localmag-like approach
if method.lower() == 'lmag' or method.lower() == 'all':
   method_parameters=get_config_dictionary(confObj, 'ew_localmag')
   wvalue=float(method_parameters['moving_window'])
else:
   wvalue=False

# Loading from configuration file Frequency analysis and convolution parameters
analysis_parameters=get_config_dictionary(confObj, 'analysis')
snr=float(analysis_parameters['snr'])
negwin=float(analysis_parameters['noise_window'])
poswin=float(analysis_parameters['signal_window'])
backinnoise=float(analysis_parameters['backinnoise'])
forwardinsign=float(analysis_parameters['forwardinsign'])
out_units=str(analysis_parameters['out_units'])

if not infile:
   file_list = sorted([p for p in glob.glob(inpath) if pathlib.Path(p).is_file()])
else:
   file_list=[inpath + os.sep + infile]

try:
   station_magnitude_parameters=get_config_dictionary(confObj, 'station_magnitude')
except:
   log_out.write("No section 'station_magnitude' in config file")
   sys.exit()
try:
   delta_peaks=float(station_magnitude_parameters['delta_peaks'])
except:
   log_out.write("No parameter 'delta_peaks' in section 'station_magnitude' of config file")
   sys.exit()
try:
   use_stcorr_hb=eval(station_magnitude_parameters['use_stcorr_hb'])
except:
   log_out.write("No parameter 'use_stcorr_hb' in section 'station_magnitude' of config file")
   sys.exit()
try: 
   use_stcorr_db=eval(station_magnitude_parameters['use_stcorr_db'])
except:
   log_out.write("No parameter 'use_stcorr_db' in section 'station_magnitude' of config file")
   sys.exit()
try:
   when_no_stcorr_hb=eval(station_magnitude_parameters['when_no_stcorr_hb'])
except:
   log_out.write("No parameter 'when_no_stcorrhb' in section 'station_magnitude' of config file")
   sys.exit()
try:
   when_no_stcorr_db=eval(station_magnitude_parameters['when_no_stcorr_db'])
except:
   log_out.write("No parameter 'when_no_stcorrdb' in section 'station_magnitude' of config file")
   sys.exit()

mcalc=False
if args.event_magnitude:
   if not args.infile:
      components_N={}
      components_E={}
      components_Z={}
      try:
         event_magnitude_parameters=get_config_dictionary(confObj, 'event_magnitude')
      except:
         log_out.write("No section 'event_magnitude' in config file")
         sys.exit()
      try:
         mindist=float(event_magnitude_parameters['mindist'])
      except:
         log_out.write("No parameter 'mindist' in section 'event_magnitude' of config file")
         sys.exit()
      try: 
         maxdist=float(event_magnitude_parameters['maxdist'])
      except:
         log_out.write("No parameter 'maxdist' in section 'event_magnitude' of config file")
         sys.exit()
      try:
         outlayers_max_it=int(event_magnitude_parameters['outlayers_max_it'])
      except:
         log_out.write("No parameter 'outlayers_max_it' in section 'event_magnitude' of config file")
         sys.exit()
      try:
         outlayers_red_stop=float(event_magnitude_parameters['outlayers_red_stop'])
      except:
         log_out.write("No parameter 'outlayers_red_stop' in section 'event_magnitude' of config file")
         sys.exit()
      cmp_keys=set()
      mcalc=True
   else:
      log_out.write('Event ML calculation omitted due to single waveform use\n')
      mcalc=False


if args.dbona_corr:
   dbcorr=pandas.read_csv(args.dbona_corr,sep=';')
else:
   dbcorr=False

# Setup plot if option given
#fig = go.Figure() if args.plot else False


amplitudes_out.write("Net;Sta;Loc;Cha;Lat;Lon;Ele;EpiDistance(km);IpoDistance(km);MinAmp(m);MinAmpTime;MaxAmp(m);MaxAmpTime;DeltaPeaks;Method;NoiseWinMin;NoiseWinMax;SignalWinMin;SignalWinMax;P_Pick;Synth;S_Picks;Synth;LoCo;HiCo;LenOverSNRIn;SNRIn;ML_H;CORR_HB;CORR_USED_HB;ML_DB;CORR_DB;CORR_USED_DB\n")
km=1000.
for f in file_list:
    try:
        tr = sac.SACTrace.read(f)
    except Exception as e:
        log_out.write(' '.join(("Error opening",str(f),"Error Raised:",str(e)," Processing next file\n")))
        continue
    seed_id='.'.join((str(tr.knetwk),str(tr.kstnm),str(tr.khole),str(tr.kcmpnm)))
    log_out.write(' '.join(("Working on",seed_id,'\n')))
    # In this block all the possibile conditions not to use this waveforms are checked so to reduce useless computing time
    # First: get timing info from SAC to soon understand if this is a good cut or not for amplitude determination
    [df,start_time,end_time,o_time,p_time,s_time] = get_trace_time_info(tr)
    # Logging out when waveform has not otime, p-pick or s-pick
    if not o_time or not p_time or not s_time:
       txt_log='File: ' + str(f) + ' o_time: ' + str(o_time) + ' p_time: ' + str(p_time) + ' s_time: ' + str(s_time) + '\n'
       picks_out.write(txt_log)
    if o_time and p_time:
       txt_log='File: ' + str(f) + ' o_time: ' + str(o_time) + ' p_time: ' + str(p_time) + ' s_time: ' + str(s_time) + '\n'
       picks_out.write(txt_log)
    # Second: if otime is not present nothing can be done 
    if not o_time:
       log_out.write(' '.join(("No Otime",seed_id," Processing next file\n")))
       continue
    # Third: define the P and S picks situation
    if args.nopicks:
       p_time=False
       s_time=False
    # Fourth: if there are not the needed event coordinates and P and S picks are not present either amplitude cannot be calculated so skip immediatly
    if (not p_time or not s_time) and (not tr.evlo or not tr.evla or not tr.evdp):
          log_out.write(' '.join(("No P reading or no S reading, and not event coordinates to compute theoretical",seed_id," Processing next file\n")))
          continue
    # Fifth: this should be done anyway but it is done here to check the next condition
    [distmeters,azi,bazi] = distaz(tr.stla,tr.stlo,tr.evla,tr.evlo)
    tr.dist=distmeters/1000.
    # Sixt: this should be done anyway but it is done here before p_time because based on s_time and thus search_endtime the lenght of the cut is evaluated
    s_synth = 0
    if not s_time:
       if theoS:
          ts = []
          ts.append(theoretical_phase_onset(model_file,'S',distmeters,tr.evdp*km))
          ts.append(theoretical_phase_onset(model_file,'s',distmeters,tr.evdp*km))
          if len(ts) != 0:
             s_time=o_time+min(ts)
             s_synth=1
             log_out.write(' '.join(("No S Readings, gone for theoretical",seed_id,'\n')))
          else:
             log_out.write(' '.join(("No S Readings, theoretical failed",seed_id,'\n')))
             continue
       else:
          log_out.write(' '.join(("No S Readings, based on pyamp.conf I skip",seed_id,'\n')))
          continue
    # According to Di Bona and thus to modified INGV Earthworm Localmag:
    # start time = t0+tp 
    # end time = t0+ts +40s * [1 – exp(– ts /40s)
    end_amp_search = s_time + 40*(1-10**(-(s_time-o_time)/40)) # end of the window where to search max amp
    #print(df,start_time,end_time,o_time,p_time,s_time,end_amp_search)
    # Seventh: based on end_amp_search time and end_time of the cut the waveform is skipped or not
    good_cut=True
    good_cut=False if end_time <= end_amp_search else True
    if not good_cut:
       log_out.write(' '.join(("This is a bad cut for end_time <= end_amp_search",str(end_time),str(end_amp_search),seed_id,"  Processing next file\n")))
       continue
    # From now on the waveform is valid (if not clipped but this check should be done outside pyamp in a pre-analysis)
    p_synth = 0
    if not p_time:
       if theoP:
          tp = []
          tp.append(theoretical_phase_onset(model_file,'P',distmeters,tr.evdp))
          tp.append(theoretical_phase_onset(model_file,'p',distmeters,tr.evdp))
          if len(tp) != 0:
             p_time=o_time+min(tp)
             p_synth=1
             log_out.write(' '.join(("No P Readings, gone for theoretical",seed_id,"\n")))
          else:
             log_out.write(' '.join(("No P Readings, theoretical failed",seed_id,"\n")))
             continue
       else:
          log_out.write(' '.join(("No P Readings, based on pyamp.conf I skip",seed_id,"\n")))
          continue
    start_amp_search = p_time # start of the window where to search max amp

    if mcalc:
       components_key='_'.join((str(tr.knetwk),str(tr.kstnm),str(tr.khole),str(tr.kcmpnm[0:2])))
       cmp_keys.add(components_key)
       
    # Preparing the figure plot if requested
    fig = reset(args.plot)
    fig = plot(fig,tr,'blue',False,False,False,False,False,'Original',False,False) if fig else False
    # Setting up the frquency analysis parameters
    my_nyq = 0.5 * float(df)
    # windows for signal to noise spectral analsysis
    n1 = p_time - backinnoise - negwin # Start of noise window (in seconds)
    n2 = p_time - backinnoise # end of noise window (in seconds)
    s1 = p_time + forwardinsign # start of signal window (in seconds)
    s2 = s_time + poswin # end of signal window (in seconds)
    # Trying to run the spectral analysis
    lo_cof_default = 0.04
    up_cof_default = my_nyq
    try:
        [lo_cof, up_cof, lenoversnr], snr_array = signal_to_noise(tr,df,n1,n2,s1,s2,snr)
        log_out.write(' '.join(("Results of Signal to Noise Analysis",str(lo_cof), str(up_cof), str(lenoversnr),"\n")))
        if not lo_cof or lo_cof == 0:
           lo_cof = lo_cof_default
           log_out.write(' '.join(("Abnormal lo_cof value reset to",str(lo_cof_default),"\n")))
        if not up_cof or up_cof == 0:
           up_cof = up_cof_default
           log_out.write(' '.join(("Abnormal up_cof value reset to",str(up_cof_default),"\n")))
        if up_cof == lo_cof:
           lo_cof = lo_cof_default
           up_cof = up_cof_default
           log_out.write(' '.join(("Abnormal up_cof=lo_cof value reset to",str(up_cof_default),str(lo_cof_default),"\n")))
    except Exception as e:
        if args.forcebp:
           lo_cof=0.04
           lo_cof=2.00
           up_cof=my_nyq
           up_cof=20
           lenoversnr=0
           log_out.write(' '.join(("Signal to Noise Analysis failed (",str(e),": Using default filtering parameters",str(lo_cof),str(up_cof),"\n")))
        else:
           continue
    try:
        tr_bandpassed=butter_bandpass(detrend(tr.data,type='constant'),lo_cof,up_cof,df,4)
    except Exception as e:
        log_out.write(' '.join(("Error applying butter bandpass filter. The reason most likely is the waveform itself",str(e),"\n")))
        continue
    tr.data = tr_bandpassed
    fig = plot(fig,tr,'green',False,False,False,False,False,'Band-passed',False,False) if fig else False
    tr.khole='' if tr.khole == '00' else tr.khole
    tr.khole = '' if not tr.khole else tr.khole
    response_filename = responsepath + os.sep + 'RESP.' + tr.knetwk + '.' + tr.kstnm + '.' + tr.khole + '.' + tr.kcmpnm
    xml_filename='AQU.xml'
    if os.path.isfile(xml_filename):
       SENSOR,DATALOGGER = rxml(xml_filename)
    else:
       SENSOR=False
       DATALOGGER=False
    if os.path.isfile(response_filename):
       resp_out.write(' '.join(("RESP Found",str(tr.knetwk),str(tr.kstnm),str(tr.khole),str(tr.kcmpnm),"\n")))
       resp=rinv(response_filename)
       #print(resp.get_response(seed_id, UTCDateTime('2019-04-06T00:00:00')))
       #print(resp[0][0][0].response)
       #sys.exit()
    else: 
       resp_out.write(' '.join(("RESP NOT Found",str(tr.knetwk),str(tr.kstnm),str(tr.khole),str(tr.kcmpnm),"\n")))
       continue
    tr_wa = to_woodanderson(tr,o_time,response_filename,out_units)
    tr_obspy = tr.to_obspy_trace().slice(starttime=start_amp_search, endtime=end_amp_search)
    #ind = pandas.date_range(start_amp_search, periods = tr_obspy.npts, freq ='ms')
    wf = pandas.DataFrame(detrend(tr_obspy.data))

    # Here we get the indexs of start_amp_search (a1x) and end_amp_search (a2x)
    dum1,dum2,a1x,a2x,dum3,dum4 = get_position(tr_wa,[df,start_time,o_time,p_time,s_time],[n1,n2,start_amp_search,end_amp_search])
    #amp_methods_list=[[False]*8]*3
    amp_methods_list=[]
    if method.lower() == "free" or method.lower() == "all":
       try:
           minamp_id,maxamp_id,minamp,maxamp = get_amps(tr_wa,a1x,a2x)
           time_minamp = start_time+minamp_id*tr.delta
           time_maxamp = start_time+maxamp_id*tr.delta
           amp=(abs(minamp)*1000.+maxamp*1000.)/2
           #amp_methods_list[0]=[minamp_id,maxamp_id,minamp,maxamp,time_minamp,time_maxamp,amp,'free']
           amp_methods_list.append([minamp_id,maxamp_id,minamp,maxamp,time_minamp,time_maxamp,amp,'free'])
       except:
           #amp_methods_list[0]=[False,False,False,False,False,False,False,'free']
           amp_methods_list.append([False,False,False,False,False,False,False,'free'])
    if method.lower() == "swing" or method.lower() == "all":
       try:
           minamp_id,maxamp_id,minamp,maxamp = get_amps_swing(tr_wa,a1x,a2x)
           time_minamp = start_time+minamp_id*tr.delta
           time_maxamp = start_time+maxamp_id*tr.delta
           amp=(abs(minamp)*1000.+maxamp*1000.)/2
           #amp_methods_list[1]=[minamp_id,maxamp_id,minamp,maxamp,time_minamp,time_maxamp,amp,'swing']
           amp_methods_list.append([minamp_id,maxamp_id,minamp,maxamp,time_minamp,time_maxamp,amp,'swing'])
       except:
           #amp_methods_list[1]=[False,False,False,False,False,False,False,'swing']
           amp_methods_list.append([False,False,False,False,False,False,False,'swing'])
    if method.lower() == "lmag" or method.lower() == "all":
       try:
           minamp_id,maxamp_id,minamp,maxamp = get_amps_lmag(tr_wa,a1x,a2x,wvalue)
           time_minamp = start_time+minamp_id*tr.delta
           time_maxamp = start_time+maxamp_id*tr.delta
           amp=(abs(minamp)*1000.+maxamp*1000.)/2
           #amp_methods_list[2]=[minamp_id,maxamp_id,minamp,maxamp,time_minamp,time_maxamp,amp,'lmag']
           amp_methods_list.append([minamp_id,maxamp_id,minamp,maxamp,time_minamp,time_maxamp,amp,'lmag'])
       except:
           #amp_methods_list[2]=[False,False,False,False,False,False,False,'lmag']
           amp_methods_list.append([False,False,False,False,False,False,False,'lmag'])

    n1x,n2x,s1x,s2x,px,sx = get_position(tr_wa,[df,start_time,o_time,p_time,s_time],[n1,n2,s1,s2])
    fig = plot(fig,tr_wa,'red',True,minamp_id,maxamp_id,minamp,maxamp,'Wood-Anderson',[px,sx],[n1x,n2x,s1x,s2x,a1x,a2x]) if fig else False
    if fig and tr.kcmpnm[2] != 'Z':
       fig.show()

# Channel Magnitudes calculations and in case of event_magnitude argument on ... station magnitude calculation
    hypo_dist = math.sqrt(math.pow(tr.dist,2)+math.pow(tr.evdp,2))
    used_methods=[]
    ml = [False]*2
    for amp_method in amp_methods_list:
        minamp,maxamp,time_minamp,time_maxamp,amp,met = amp_method[2:]
        if mcalc:
           components_key_met=components_key+'_'+met
        used_methods.append(met)
        if amp:
           # Loading stations corrections
           dbsite=tr.kstnm
           try:
               s_dibona=float(dbcorr.loc[dbcorr['sta'] == dbsite, 'c'].iloc[0]) # Di Bona Station correction
           except:
               s_dibona=False
           s_hutton=False # Station correction for Hutton and Boore formula

           if s_hutton or (not s_hutton and when_no_stcorr_hb):
              try:
                  ml[0] = huttonboore(amp,hypo_dist,s_hutton,use_stcorr_hb)
              except:
                  ml[0] = False
           else:
              ml[0] = False

           if s_dibona or (not s_dibona and when_no_stcorr_db):
              try:
                  ml[1] = dibona(amp,hypo_dist,s_dibona,use_stcorr_db)
              except:
                  ml[1] = False
           else:
              ml[1] = False

           if mcalc:
              if tr.kcmpnm[2] == 'N':
                 components_N[components_key_met]=[ml,amp,tr.dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
              elif tr.kcmpnm[2] == 'E':
                 components_E[components_key_met]=[ml,amp,tr.dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
              elif tr.kcmpnm[2] == 'Z':
                 components_Z[components_key_met]=[ml,amp,tr.dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
              else:
                 log_out.write(' '.join(("Component not recognized for ",str(tr.knetwk),str(tr.kstnm),str(tr.khole),str(tr.kcmpnm),"\n")))
           amplitudes_out.write(';'.join((tr.knetwk,tr.kstnm,str(tr.khole),tr.kcmpnm,str(tr.stla),str(tr.stlo),str(tr.stel),str(tr.dist),str(hypo_dist),str(minamp),str(time_minamp),str(maxamp),str(time_maxamp),str(abs(time_maxamp-time_minamp)),str(met),str(n1),str(n2),str(s1),str(s2),str(p_time),str(p_synth),str(s_time),str(s_synth),str(lo_cof),str(up_cof),str(lenoversnr),str(snr),str(ml[0]),str(s_hutton),str(use_stcorr_hb),str(ml[1]),str(s_dibona),str(use_stcorr_db),'\n')))
        else:
           log_out.write(' '.join(("No Amps for this method ",str(met),' on ',str(tr.knetwk),str(tr.kstnm),str(tr.khole),str(tr.kcmpnm),"\n")))

if mcalc:
   magnitudes_out.write("ML_HB;Std_HB;TOTSTA_HB;USEDSTA_HB;ML_DB;Std_DB;TOTSTA_DB;USEDSTA_DB;ampmethod;magmethod;loopexitcondition\n")
   for method in used_methods:
          # Hutton and Boore
          meanmag_ml_sta,meanamp_ml_sta = create_sets(cmp_keys,components_N,components_E,method,mindist,maxdist,delta_peaks,0,when_no_stcorr_hb,use_stcorr_hb)
          ma_mlh,ma_stdh,ma_ns_s_h,ma_nsh,cond = calculate_event_ml(meanamp_ml_sta,outlayers_max_it,outlayers_red_stop)
          mm_mlh,mm_stdh,mm_ns_s_h,mm_nsh,cond = calculate_event_ml(meanmag_ml_sta,outlayers_max_it,outlayers_red_stop)
          # Di Bona
          meanmag_ml_sta,meanamp_ml_sta = create_sets(cmp_keys,components_N,components_E,method,mindist,maxdist,delta_peaks,1,when_no_stcorr_db,use_stcorr_db)
          ma_mld,ma_stdd,ma_ns_s_d,ma_nsd,cond = calculate_event_ml(meanamp_ml_sta,outlayers_max_it,outlayers_red_stop)
          mm_mld,mm_stdd,mm_ns_s_d,mm_nsd,cond = calculate_event_ml(meanmag_ml_sta,outlayers_max_it,outlayers_red_stop)
          magnitudes_out.write(';'.join((str(ma_mlh),str(ma_stdh),str(ma_ns_s_h),str(ma_nsh),str(ma_mld),str(ma_stdd),str(ma_ns_s_d),str(ma_nsd),method,'meanamp',cond,'\n')))
          magnitudes_out.write(';'.join((str(mm_mlh),str(mm_stdh),str(mm_ns_s_h),str(mm_nsh),str(mm_mld),str(mm_stdd),str(mm_ns_s_d),str(mm_nsd),method,'meanmag',cond,'\n')))
# Now closing all output files
amplitudes_out.close()
magnitudes_out.close()
picks_out.close()
resp_out.close()
log_out.close()
sys.exit()
