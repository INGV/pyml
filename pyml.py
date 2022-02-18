# Authors: Raffaele Di Stefano (raffaele.distefano@ingv.it), Barbara Castello (barbara.castello@ingv.it)
# Licence: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# PyML reads in pyamp_amplitudes.csv or the same data from a MySQL table and
# - calculates channel ML on all the three components
# - calculates event ML when a set of waveform (of one single event) is given
# Attenuation Law
#    PyML implemets two different attenuation laws for the ML calculation:
#    - INGV Hutton & Boore which is mutuated by the one formerly used at USGS for the 
#      adapted to the Italian region with no stations corrections
#    - M. Di Bona attenuation law calculated specifically for the Italian region
#      with stations corrections
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
import scipy
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

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def parseArguments():
        parser=MyParser()	
        parser.add_argument('--infile',      default=None,          help='pyamp-amplitudes.csv file full path')
        parser.add_argument('--eventid',     default='0',           help='Unique identifier of the event')
        parser.add_argument('--dbona_corr',  default='dbcor.csv',   help='Input file with DiBona Stations corrections')
        parser.add_argument('--conf',        default='./pyml.conf', help='A file containing sections and related parameters (see the example)')
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
           ipodist = (cmpn[kk][3] + cmpe[kk][3])/2 # ipocentral distance
           if ipodist >= mid and ipodist <= mad: # if ipocentral distance is within the accepted range
              if mtd != 'free' or (mtd == 'free' and abs(cmpn[kk][6]-cmpn[kk][5]) <= dp and abs(cmpe[kk][6]-cmpe[kk][5]) <= dp): # if the method is not free the deltapeak has no meaning orherwise it is evaluated
                 #Mean of channel magnitudes is calculated
                 if cmpn[kk][0][mty] and cmpe[kk][0][mty]:
                    mm = (cmpn[kk][0][mty] + cmpe[kk][0][mty])/2
                    meanmag_ml_set.append([kk,mm])
                 # Magnitudes of Mean channel amplitudes is calculated if 
                 if not stc or (cmpn[kk][4][mty] and cmpe[kk][4][mty]) or whstc: 
                    #mean_amp = (cmpn[kk][1] + cmpe[kk][1])/2 # Artimetic mean
                    mean_amp_geo = math.sqrt((cmpn[kk][1] * cmpe[kk][1])) # Geometric mean that is the correct one according to Richter and Di Bona
                    corr = (cmpn[kk][4][mty] + cmpe[kk][4][mty])/2 if cmpn[kk][4][mty] and cmpe[kk][4][mty] else False
                    if mty == 0:
                       ma = huttonboore(mean_amp_geo,ipodist,corr,stc)
                    if mty == 1:
                       ma = dibona(mean_amp_geo,ipodist,corr,stc)
                    meanamp_ml_set.append([kk,ma])
              else:
                 log_out.write(' '.join(("Station skipped due to minmax distance: ",str(kk),str(abs(cmpn[kk][6]-cmpn[kk][5])),"\n")))
           else:
              log_out.write(' '.join(("Station skipped due to epidist: ",str(k),str(epidist),"\n")))
        else:
           log_out.write(' '.join(("Station skipped due to missing channel ",str(kk),'\n')))
    return meanmag_ml_set,meanamp_ml_set

def calculate_event_ml(magnitudes,magnitudes_sta,maxit,stop,max_dev,hm_cutoff):
    m=numpy.array(magnitudes)
    s=magnitudes_sta
    finished = False
    N = 0
    Ml_Std  = scipy.stats.median_abs_deviation(m)
    Ml_Medi = numpy.median(m)
    Ml_ns_start = len(m)
    removed=[]
    while not finished:
          N = N + 1
          Ml_Medi_old = Ml_Medi
          distance_from_mean = abs(m - Ml_Medi)
          if hm_cutoff:
             #w = numpy.asarray(list(filter(lambda x: 1.0 if x <= float(hm_cutoff) else (float(hm_cutoff)/x),list(distance_from_mean)))) # Values beyond cutoff are downweighted
             w = numpy.where(distance_from_mean <= hm_cutoff,1.0,hm_cutoff/distance_from_mean)
             print("Stampeso su Media",str(Ml_Medi))
             print(distance_from_mean)
             print(w)
             Ml_Medi = numpy.sum(m * w)/numpy.sum(w)
             deltaMean = abs(Ml_Medi-Ml_Medi_old)
             Ml_Std=0.0
          print(N,Ml_Medi_old,Ml_Medi,deltaMean,stop)
          if not hm_cutoff:
             not_outlier = distance_from_mean < max_dev * Ml_Std
             yes_outlier = distance_from_mean >= max_dev * Ml_Std
             removed.append(list(zip(s[yes_outlier],m[yes_outlier])))
             m = m[not_outlier]
             s = s[not_outlier]
             if len(m) > 0:
                Ml_Std  = scipy.stats.median_abs_deviation(m)
                Ml_Medi = numpy.median(m)
                deltaMean = abs(Ml_Medi-Ml_Medi_old)
             else:
                finished = True
                Ml_Std  = False
                Ml_Medi = False
                Ml_ns = False
                condition='emptyset'
             w = numpy.ones(Ml_ns_start)
          if deltaMean <= stop or N == maxit:
             finished = True
             Ml_ns = len(m)
             condition='deltaMean:'+str(deltaMean)+':'+str(N) if deltaMean <= stop else 'maxit'
    return Ml_Medi,Ml_Std,Ml_ns_start,Ml_ns,condition,removed,w

###### End of Functions ##########
## Main ##
args = parseArguments()
infile=args.infile
eventid=args.eventid

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
    magnitudes_out=str(eventid)+'_'+eval(filenames_parameters['magnitudes'])
    log_out=str(eventid)+'_'+eval(filenames_parameters['log'])
except Exception as e:
    sys.stderr.write(("\n"+str(e)+"\n\n"))
    sys.exit(1)

#Net;Sta;Loc;Cha;Lat;Lon;Ele;EpiDistance(km);IpoDistance(km);MinAmp(m);MinAmpTime;MaxAmp(m);MaxAmpTime;DeltaPeaks;Method;NoiseWinMin;NoiseWinMax;SignalWinMin;SignalWinMax;P_Pick;Synth;S_Picks;Synth;Nyq;LoCo;HiCo;LenOverSNRIn;SNRIn;ML_H;CORR_HB;CORR_USED_HB;ML_DB;CORR_DB;CORR_USED_DB
dfa=pandas.read_csv(args.infile,sep=';',index_col=False)

if magnitudes_out:
   magnitudes_out=open(magnitudes_out,'w')
else:
   magnitudes_out=sys.stdout

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
delta_corner=float(preconditions_parameters['delta_corner'])
max_lowcorner=float(preconditions_parameters['max_lowcorner'])

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
   hm_cutoff=eval(event_magnitude_parameters['hm_cutoff'])
except:
   log_out.write("No parameter 'hm_cutoff' in section 'event_magnitude' of config file")
   sys.exit()
try:
   outliers_max_it=int(event_magnitude_parameters['outliers_max_it'])
except:
   log_out.write("No parameter 'outliers_max_it' in section 'event_magnitude' of config file")
   sys.exit()
try:
   outliers_red_stop=float(event_magnitude_parameters['outliers_red_stop'])
except:
   log_out.write("No parameter 'outliers_red_stop' in section 'event_magnitude' of config file")
   sys.exit()
try:
   outliers_nstd=float(event_magnitude_parameters['outliers_nstd'])
except:
   log_out.write("No parameter 'outliers_nstd' in section 'event_magnitude' of config file")
   sys.exit()
cmp_keys=set()
components_N={}
components_E={}
components_Z={}
mcalc=True

if args.dbona_corr:
   dbcorr=pandas.read_csv(args.dbona_corr,sep=';')
else:
   dbcorr=False

# Setup plot if option given
#fig = go.Figure() if args.plot else False


km=1000.
#Net;Sta;Loc;Cha;Lat;Lon;Ele;EpiDistance(km);IpoDistance(km);MinAmp(m);MinAmpTime;MaxAmp(m);MaxAmpTime;DeltaPeaks;Method;NoiseWinMin;NoiseWinMax;SignalWinMin;SignalWinMax;P_Pick;Synth;S_Picks;Synth;LoCo;HiCo;LenOverSNRIn;SNRIn;ML_H;CORR_HB;CORR_USED_HB;ML_DB;CORR_DB;CORR_USED_DB
for index, row in dfa.iterrows():
    corner_low=float(row['LoCo'])
    corner_high=float(row['HiCo'])
    if corner_low >= max_lowcorner or (corner_high-corner_low) < delta_corner:
       log_out.write(' '.join(("Component skipped:",str(row['Net']),str(row['Sta']),str(row['Loc']),str(row['Cha']),str(corner_low),str(corner_high),"\n")))
       continue
    seed_id='.'.join((str(row['Net']),str(row['Sta']),str(row['Loc']),str(row['Cha'])))
    log_out.write(' '.join(("Working on",seed_id,'\n')))
    # In this block all the possibile conditions not to use this waveforms are checked so to reduce useless computing time
    # First: get timing info from SAC to soon understand if this is a good cut or not for amplitude determination
    #[distmeters,azi,bazi] = distaz(tr.stla,tr.stlo,tr.evla,tr.evlo)
    #tr.dist=distmeters/1000.
    components_key='_'.join((str(row['Net']),str(row['Sta']),str(row['Loc']),str(row['Cha'][0:2])))
    cmp_keys.add(components_key)
# Channel Magnitudes calculations and in case of event_magnitude argument on ... station magnitude calculation
    hypo_dist = row['IpoDistance(km)'] #math.sqrt(math.pow(tr.dist,2)+math.pow(tr.evdp,2))
    epi_dist = row['EpiDistance(km)'] #math.sqrt(math.pow(tr.dist,2)+math.pow(tr.evdp,2))
    ml = [False]*2
    #minamp,maxamp,time_minamp,time_maxamp,amp,met = amp_method[2:]
    minamp=row['MinAmp(m)']*1000
    maxamp=row['MaxAmp(m)']*1000
    time_minamp=row['MinAmpTime']
    time_maxamp=row['MaxAmpTime']
    amp = abs(maxamp-minamp)/2
    met=row['Method']
    components_key_met=components_key+'_'+met
    # Loading stations corrections
    dbsite=row['Sta']
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

    if row['Cha'][2] == 'N':
       components_N[components_key_met]=[ml,amp,epi_dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
    elif row['Cha'][2] == 'E':
       components_E[components_key_met]=[ml,amp,epi_dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
    elif row['Cha'][2] == 'Z':
       components_Z[components_key_met]=[ml,amp,epi_dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
    else:
       log_out.write(' '.join(("Component not recognized for ",str(row['Net']),str(row['Sta']),str(row['Loc']),str(row['Cha']),"\n")))

magnitudes_out.write("EventID;ML_HB;Std_HB;TOTSTA_HB;USEDSTA_HB;ML_DB;Std_DB;TOTSTA_DB;USEDSTA_DB;ampmethod;magmethod;loopexitcondition\n")
# Hutton and Boore
meanmag_ml_sta,meanamp_hb_ml_sta = create_sets(cmp_keys,components_N,components_E,met,mindist,maxdist,delta_peaks,0,when_no_stcorr_hb,use_stcorr_hb)
meanmag_ml = list(list(zip(*meanmag_ml_sta))[1])
meanamp_ml = list(list(zip(*meanamp_hb_ml_sta))[1])
meanamp_ml_sta = numpy.asarray(list(list(zip(*meanamp_hb_ml_sta))[0]), dtype=object)
ma_mlh,ma_stdh,ma_ns_s_h,ma_nsh,cond_hb,outliers_hb,weights_hb = calculate_event_ml(meanamp_ml,meanamp_ml_sta,outliers_max_it,outliers_red_stop,outliers_nstd,hm_cutoff)
#mm_mlh,mm_stdh,mm_ns_s_h,mm_nsh,cond = calculate_event_ml(meanmag_ml_sta,outliers_max_it,outliers_red_stop)
# Di Bona
meanmag_ml_sta,meanamp_db_ml_sta = create_sets(cmp_keys,components_N,components_E,met,mindist,maxdist,delta_peaks,1,when_no_stcorr_db,use_stcorr_db)
meanmag_ml = list(list(zip(*meanmag_ml_sta))[1])
meanamp_ml = list(list(zip(*meanamp_db_ml_sta))[1])
meanamp_ml_sta = numpy.asarray(list(list(zip(*meanamp_db_ml_sta))[0]), dtype=object)
ma_mld,ma_stdd,ma_ns_s_d,ma_nsd,cond_db,outliers_db,weights_db = calculate_event_ml(meanamp_ml,meanamp_ml_sta,outliers_max_it,outliers_red_stop,outliers_nstd,hm_cutoff)
#mm_mld,mm_stdd,mm_ns_s_d,mm_nsd,cond = calculate_event_ml(meanmag_ml_sta,outliers_max_it,outliers_red_stop)
magnitudes_out.write(';'.join((str(eventid),str(ma_mlh),str(ma_stdh),str(ma_ns_s_h),str(ma_nsh),str(ma_mld),str(ma_stdd),str(ma_ns_s_d),str(ma_nsd),met,'meanamp',cond_hb+'-'+cond_db,'\n')))
#magnitudes_out.write(';'.join((str(eventid),str(mm_mlh),str(mm_stdh),str(mm_ns_s_h),str(mm_nsh),str(mm_mld),str(mm_stdd),str(mm_ns_s_d),str(mm_nsd),met,'meanmag',cond,'\n')))
# Now closing all output files
#print(weights_hb)
for x, y, wx, wy in zip(meanamp_hb_ml_sta, meanamp_db_ml_sta, weights_hb, weights_db):
    sth,mh = map(str,x)
    std,md = map(str,y)
    whb = str(wx)
    wdb = str(wy)
    magnitudes_out.write(' '.join(('MLSTA',sth,mh,whb,std,md,wdb,'\n')))
if not hm_cutoff:
   for x in outliers_hb[0]:
       sth,mh = map(str,list(x))
       magnitudes_out.write(' '.join(('OUTL_HB',sth,mh,'\n')))
   for y in outliers_db[0]:
       std,md = map(str,list(y))
       magnitudes_out.write(' '.join(('OUTL_DB',std,md,'\n')))
magnitudes_out.close()
log_out.close()
sys.exit()
