# Authors: Raffaele Di Stefano (raffaele.distefano@ingv.it), Barbara Castello (barbara.castello@ingv.it)
# Aknowledgements: Valentino Lauciani (valentino.lauciani@ingv.it) and Andrea Bono (andrea.bono@ingv.it) - adaptation to the monitoring system
# Licence: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# 
# Input files:
# - option 1) channels' amplitudes csv file, stations corrections csv files, pyml.conf
# - option 2) channels' amplitudes json file with conf included, stations corrections csv files
# Operations:
# - calculates channel ML on all the three components with several Attenuation Laws
# - calculates station ML on all the stations either as a mean of channel's one or as ML of the mean amplitude
# - calculates event ML for all the Attenuation Laws with different possibile statistical approaches
#    
# Details: Amplitude usage
#    Two different approaches are followed to calculate the station ML:
#    - the mean of the maximum amplitudes N and E is used in the attenuation law to get the station ML (mean can be aritmetic or geometric)
#    - the mean of the N and E ML is performed to give the station ML (mean can be aritmetic or geometric)
# Attenuation Law
#    PyML implemets two different attenuation laws for the ML calculation:
#    - INGV Hutton & Boore which is mutuated by the one formerly used at USGS for the 
#      adapted to the Italian region with no stations corrections
#    - M. Di Bona attenuation law calculated specifically for the Italian region
#      with stations corrections at 400 stations (Di Bona 1-199 and Mele-Quintiliani 200-400)
#    - Lolli Gssperini and Mele double attenuation law is coming soon
#
# Mean ML Value and Standard deviation calculation
#    Two alternative, both iterative, approaches can be used 
#    Both methods start from the median and not from the mean for robustness, then they diverge from each other
#    1) Strict Huber Mean / Outliers removal (weighting function is not applied or it can considered to be 0 for outlier and 1 for the rest)
#       Parameters are taken from the .conf file or from the json input:
#       - outliers_nstd: how many time (outliers_nstd*standarddev) must be used to cut out the outliers
#       - outliers_cutoff: a lower limit to the outliers_nstd*standarddev value
#       - outliers_red_stop: stops iterations when abs(mean_old-updated_mean)) is lower than outliers_red_stop
#       - outliers_max_it: after this value iterations it stops iterating anyway (no convergence); this value is also used in the other method
#    2) Weighted Huber Mean
#       after the preliminary mean calculation this approach calculates it's own weighted mean value and standard deviation
#       Parameters are taken from the .conf file or from the json input:
#       - outliers_max_it: same as the other
#       - hm_cutoff: this is the corner of the downweighting function; whatever is inside +/- hm_cutoff from the (updated) weighted mean has weight 1.0,
#         the rest is downweighted as hm_cutoff/(distance_from_the_mean)
#    In both cases stations with hypocentral distance lower than mindist (tipically 10km) and higher than maxdist (tipically 600) are excluded

import argparse,sys,os,glob,copy,pwd,pathlib,itertools,getpass,socket
from geographiclib.geodesic import Geodesic
import pandas
import time
import json

# Imports from obpsy
from obspy import read
from obspy.geodetics.base import gps2dist_azimuth as distaz
from scipy.stats import median_abs_deviation

from numpy import where as np_where
from numpy import sum as np_sum
from numpy import square as np_square
from numpy import errstate as np_errstate
from numpy import median as np_median
from numpy import ones as np_ones
from numpy import array as np_array
from numpy import asarray as np_asarray

from datetime import datetime

from math import sqrt as msqrt, log10 as mlog10, pow as mpow

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def parseArguments():
        parser=MyParser()	
        parser.add_argument('--in_file_name',       default=None,          help='Name (and path) of inpout amplitudes file')
        parser.add_argument('--in_file_format',     default='json',        help='options are json (data and config in one single file) or csv (pyamp format): this latter needs also the --conf argument')
        parser.add_argument('--conf',               default=None,          help='A file containing sections and related parameters (used only with --csv)')
        parser.add_argument('--out_format',         default='json',        help='options are json or txt')
        parser.add_argument('--eventid',            default='0',           help='Unique identifier of the event (only used with --csv_in and --conf)')
        parser.add_argument('--dbona_corr',         default='dbcor.csv',   help='Input file with DiBona Stations corrections')
        parser.add_argument('--clipped_info',help='Input file with information on clipped channels')
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
                sys.stderr.write(("skip: %s" % option))
        except:
            if log_out:
               log_out.write("exception on %s!\n" % option)
            dict1[option] = None
    return dict1

def huttonboore(a,d,s,uc):
    # if uc (use_stcorr_hb) is True s is set to its entrance value and if this value is not False it is set to its value
    # if uc (use_stcorr_hb) is True s is set to its entrance value and if this value is False it is set to ZERO
    # if uc (use_stcorr_hb) is False s entrance value is overwritten and it is set to ZERO and then in the second condition it is ZERO (so it is True because it has a value) and remains ZERO
    s = 0 if not uc else s # station correction is set to 0 if use_stcorr_hb is False
    s = 0 if not s else s # station correction is set to 0 if it is False
    try:
       m = mlog10(a) + 1.110*mlog10(d / 100.) + 0.00189*(d - 100.) + 3.0 + s # Hutton & Boore with s added but not confirmed if it is correct
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
       m = mlog10(a) + 1.667*mlog10(d / 100.) + 0.001736*(d - 100.) + 3.0 + s # Massimo Di Bona
    except:
       m = False
    return m


def create_sets(keys,cmpn_original,cmpe_original,mtd,mid,mad,dp,mty,whstc,stc,mmt,amt,logsms,jlog_stamag,jlog_stamag_cha):
    # mtd is the peakmethod
    # mty is the huttonboore 0, dibona 1
    # a channel cmpn is [ml,amp,tr.dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
    #   where ml is a list of two: [ml_hutton,ml_dibona]
    # mmt is the mag_mean_type parameter
    #     'meanmag"--> only the mean of the channels's ML is computed
    #     'meanamp"--> only the ML based on the mean of the channels's amplitudes is computed
    #     in the second case, amt is evaluated:
    #    'geo' --> the geometrical mean is computed
    #     'ari' --> the artihmetic mean is computed
    ml_set=[]
    mtytxt='HuttonBoore' if mty==0 else 'DiBona'
    midi = 99999.0 # Minimum distance is needed and here below calculated to be used in the mag quality definition
    cmpe = cmpe_original.copy()
    cmpn = cmpn_original.copy()
    for k in keys:
        log_condition=False
        n,s,l,c = k.split('_')
        kk=k+'_'+mtd
        logsm = copy.deepcopy(jlog_stamag)
        logsm['net'],logsm['sta'],logsm['loc'],logsm['band_inst'] = [n,s,l,c]
        #logsm['loc'] = "--" if logsm['loc'] == "None" else logsm['loc']
        why_missing_channel=""
        if kk in cmpe:
           if not cmpe[kk][1][1] or not cmpe[kk][1][0]:
              why_missing_channel="maxamp missing" if not cmpe[kk][1][1] else "minamp missing"
              remove_key = cmpe.pop(kk, None)
        if kk in cmpn:
           if not cmpn[kk][1][1] or not cmpn[kk][1][0]:
              why_missing_channel="maxamp missing" if not cmpn[kk][1][1] else "minamp missing"
              remove_key = cmpn.pop(kk, None)
        if kk in cmpn and kk in cmpe: # if both components are present in the set
           if not cmpn[kk][2] or not cmpe[kk][2]:
              epidist = False
              ipodist = False
              if log_out:
                 log_out.write(' '.join(("Station skipped due to stations coordinates missing: ",str(k),str(ipodist),"\n")))
              logsm['status'] = "critical"
              logsm['summary'] = "Station skipped due to stations coordinates missing"
              logsm['extended'] = mtytxt
              logsms.append(logsm)
              continue
           epidist = (cmpn[kk][2] + cmpe[kk][2])/2 # epicentral distance
           ipodist = (cmpn[kk][4] + cmpe[kk][4])/2 # ipocentral distance
           amp_e = abs(cmpe[kk][1][1]-cmpe[kk][1][0])/2 # now both the minamp and maxamp values are reported in the channels arrays so amplitude richter must be calculated here
           amp_n = abs(cmpn[kk][1][1]-cmpn[kk][1][0])/2 # now both the minamp and maxamp values are reported in the channels arrays so amplitude richter must be calculated here
           if ipodist >= mid and ipodist <= mad: # if ipocentral distance is within the accepted range we proceed
              if mtd != 'free' or (mtd == 'free' and abs(cmpn[kk][7]-cmpn[kk][6]) <= dp and abs(cmpe[kk][7]-cmpe[kk][6]) <= dp): # if the method is not free the deltapeak has no meaning orherwise it is evaluated
                 #Mean of channel magnitudes is calculated
                 if mmt == 'meanmag' and cmpn[kk][0][mty] and cmpe[kk][0][mty]:
                    mm = (cmpn[kk][0][mty] + cmpe[kk][0][mty])/2
                 elif mmt == 'meanamp' and (not stc or (cmpn[kk][5][mty] and cmpe[kk][5][mty]) or whstc): 
                 # Magnitudes of Mean channel amplitudes is calculated if 
                    if amt == 'ari':
                       mean_amp = (amp_n + amp_e)/2 # Artimetic mean
                    elif amt == 'geo':
                       mean_amp = msqrt(amp_n * amp_e) # Geometric mean that is the correct one according to Di Bona
                    corr = (cmpn[kk][5][mty] + cmpe[kk][5][mty])/2 if cmpn[kk][5][mty] and cmpe[kk][5][mty] else False
                    if mty == 0:
                       mm = huttonboore(mean_amp,ipodist,corr,stc)
                       mtytxt = 'HuttonBoore'
                    if mty == 1:
                       mm = dibona(mean_amp,ipodist,corr,stc)
                       mtytxt = 'DiBona'
                 ml_set.append([kk,mm])
                 midi = epidist if epidist < midi else midi
              else:
                 if log_out:
                    log_out.write(' '.join(("Station skipped due to time distance between min and max amp larger than",str(dp),":",str(kk),str(abs(cmpn[kk][7]-cmpn[kk][6])),"\n")))
                 logsm['status'] = 'warning'
                 logsm['summary'] = ' '.join(("In ML ",mtytxt,"Station skipped due to time distance between min and max amp larger than",str(dp)))
                 logsm['extended'] = ' '.join(("Time distance is",str(abs(cmpn[kk][7]-cmpn[kk][6])),"s"))
                 log_condition=True
                 #logsms.append(logsm)
           else:
              if log_out:
                 log_out.write(' '.join(("Station skipped due to ipodist: ",str(k),str(ipodist),"\n")))
              logsm['status'] = 'warning' 
              logsm['summary'] = ' '.join(("In ML ",mtytxt,"Station",str(kk),"skipped due to ipodist"))
              logsm['extended'] = ' '.join(("Distance is",str(ipodist),"km"))
              log_condition=True
              #logsms.append(logsm)
        elif kk not in cmpn or kk not in cmpe:
           if log_out:
              log_out.write(' '.join(("Station skipped due to missing channel",str(kk),'\n')))
           logsm['status'] = 'critical'
           logsm['summary'] = 'Station skipped due to missing channel'
           logsm['extended'] = 'ML ' + mtytxt
           logsmc = copy.deepcopy(jlog_stamag_cha)
           if kk not in cmpn:
              logsmc['orientation'] = 'N'
              logsmc['status'] = 'critical' 
              logsmc['summary'] = 'missing' 
              logsmc['extended'] = why_missing_channel
           else:
              logsmc['orientation'] = 'N'
              logsmc['status'] = 'ok' 
           logsm['channels'].append(logsmc)
           logsmc = copy.deepcopy(jlog_stamag_cha)
           if kk not in cmpe:
              logsmc['orientation'] = 'E'
              logsmc['status'] = 'critical' 
              logsmc['summary'] = 'missing' 
              logsmc['extended'] = why_missing_channel
           else:
              logsmc['orientation'] = 'E'
              logsmc['status'] = 'ok' 
           logsm['channels'].append(logsmc)
           log_condition=True
        if log_condition:
           logsms.append(logsm)
    return ml_set,midi,logsms

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Calculating the weighted standard deviation
#         from SUBROUTINE MEDIA_HUBER Programmed on 1 september 2003 by Franco Mele (INGV)
# Formula is: 
#           ___N
#           \                    2
#            \      Wi(Xi - Xmed)
#            /
#           /__i=1
# SQRT(  ------------------------------------- )
#
#              N'-1     __N
#              ----- (  \     Wi )
#                N'     /_i=1
#
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def wstd(v,wm,wf):
    eps2=5.0e-6
    if len(v) <= 1:
       wsd = False
    else:
       index_not_zero = np_where(wf > eps2)[0]
       wf_nz = wf[index_not_zero]
       #print("wf_nz: ",wf_nz)
       v_nz  = v[index_not_zero]
       wsd = np_sum(wf_nz*np_square((v_nz-wm)))
       fac = ((len(wf_nz)-1)/len(wf_nz))*np_sum(wf_nz)
       wsd = wsd / max(fac,eps2)
       wsd = msqrt(max(wsd,eps2))
    return wsd 

def whuber(v,w_mean,ruse,zero):
    res = abs(v - w_mean)
    with np_errstate(divide='ignore'):
         w = np_where(res <= ruse,1.0,np_where(res > zero,0.0,ruse/res))
    #print("W: ",w)
    try:
       w_mean = np_sum(v * w)/np_sum(w)
    except Exception as e:
       sys.stderr.write(("Error calculalting the Huber Mean",e))
       w_mean = False
    if w_mean:
       w_std = wstd(v,w_mean,w)
    else:
       w_std = False
    w_std = 0.0 if not w_std else w_std
    return w_mean,w_std,w

def rm_outliers(v,v_w,v_flag,v_mean,v_std,times_std,co,var_stop,it_max,skip):
    res = abs(v - v_mean)
    #w_fake = np_ones(len(v))
    v_mean_old = v_mean
    cut_limit = times_std * v_std if (times_std * v_std) > co else co
    not_outlier = res  < cut_limit
    yes_outlier = res >= cut_limit
    skip.append(list(zip(v_flag[yes_outlier],v[yes_outlier])))
    v_w[yes_outlier]=0
    v_w[not_outlier]=1
    #v = v[not_outlier]
    #v_flag = v_flag[not_outlier]
    if len(v[not_outlier]) > 0:
       v_std  = median_abs_deviation(v[not_outlier])
       v_mean = np_median(v[not_outlier])
       n_v_flag = len(v[not_outlier])
    else:
       v_std  = False
       v_mean = False
       n_v_flag = False
    return v_mean,v_std,v_w,skip

def calculate_event_ml(magnitudes,magnitudes_sta,it_max,var_stop,max_dev,out_cutoff,hm_cutoff):
    v = np_array(magnitudes)
    s = magnitudes_sta
    ruse = hm_cutoff
    # Calculate starting values from m (magnitudes) vector
    # We here use a median and mad instead of the mean and std as stating values because they are more robust
    # Names are taken from Huber Mean routine by Franco Mele for coherence
    xmd = np_median(v)
    xmd_std  = median_abs_deviation(v)
    vlen_start = len(v)
    n = 1
    finished = False
    if hm_cutoff:
       ruse,zero = [hm_cutoff,9999.0] if isinstance(hm_cutoff,float) else hm_cutoff
    weights = np_ones(len(v))
    removed=[]
    while not finished:
          amd = xmd
          if hm_cutoff:
             #print("Calling Whuber","ruse: ",ruse,"zero: ",zero,"V: ",v,"XMD: ",xmd)
             whuber_fail = False
             try:
                 xmd,xmd_std,weights = whuber(v,xmd,ruse,zero)
                 typemean = 'whuber'
             except Exception as e:
                 whuber_fail = True
          if not hm_cutoff or whuber_fail:
             xmd,xmd_std,weights,removed = rm_outliers(v,weights,s,xmd,xmd_std,max_dev,out_cutoff,var_stop,it_max,removed)
             if not whuber_fail:
                typemean = 'rmoutl'
             else:
                typemean = 'rowhfl'
          xmd_var = abs(amd-xmd)
          if xmd_var <= var_stop:
             finished = True
             whystop=typemean+'_varstop='+str(xmd_var)+'_n='+str(n)
          elif n == it_max:
             finished = True
             whystop=typemean+'_maxit='+str(n)+'_xmdvar='+str(xmd_var)
          n += 1
    vlen_stop = round(np_sum(weights),2)
    return xmd,xmd_std,vlen_start,vlen_stop,whystop,removed,weights,whuber_fail

def char_quality(nu,nw,d,m,r):
    #nu =  number of used stations
    #nw =  weighted number of used stations
    # d = minimum distance km
    # m = magnitude
    # defining position 1
    if d <= 50. and nu > (m*10.0+1):
       q1 = 'A'
    elif d > 50. and d <= 100.0 and nu > (m*7.5+1):
       q1 = 'B'
    elif d > 100. and d <= 150.0 and nu > (m*5.0+1):
       q1 = 'C'
    else:
       q1 = 'D'
    # defining position 2 (by Alberto Basili for INGV Locator code) 
    #A = rms<=0.2
    #B = 0.2<rms<=0.3
    #C = 0.3<rms<=0.4
    #D = 0.4<rms
    
    if r <= 0.2:
       q2 = 'A'
    elif r > 0.2 and r <= 0.3:
       q2 = 'B'
    elif r > 0.3 and r <= 0.4:
       q2 = 'C'
    elif r > 0.4:
       q2 = 'D'
    return ''.join((q1,q2))

def standard_pyml_load(infile,eventid,conf_file,log_out,f):
   # Now loading the configuration file
   if os.path.exists(conf_file) and os.path.getsize(conf_file) > 0:
      paramfile=conf_file
   else:
      sys.stderr.write("Config file " + args.conf + " not existing or empty\n\n")
      sys.exit(2)
   confObj = cp.ConfigParser()
   confObj.read(paramfile)

   #try:
   #    filenames_parameters=get_config_dictionary(confObj, 'iofilenames')
   #except Exception as e:
   #    sys.stderr.write(("\n"+str(e)+"\n\n"))
   #    sys.exit(1)
   #try:
   #    mf=eval(filenames_parameters['magnitudes'])
   #    lf=eval(filenames_parameters['log'])
   #    if mf:
   #       magnitudes_out=open(str(eventid)+'_'+str(mf),'w')
   #    else:
   #       magnitudes_out=open(str(eventid)+'_pyml_magnitudes.out','w')
   #    if lf:
   #       log_out=open(str(eventid)+'_'+str(lf),'w')
   #    else:
   #       log_out=open(str(eventid)+'_pyml_general.log','w')
   #except Exception as e:
   #    sys.stderr.write(("\n"+str(e)+"\n\n"))
   #    sys.exit(1)
    
   
   #Net;Sta;Loc;Cha;Lat;Lon;Ele;EpiDistance(km);IpoDistance(km);MinAmp(m);MinAmpTime;MaxAmp(m);MaxAmpTime;DeltaPeaks;Method;NoiseWinMin;NoiseWinMax;SignalWinMin;SignalWinMax;P_Pick;Synth;S_Picks;Synth;Nyq;LoCo;HiCo;LenOverSNRIn;SNRIn;ML_H;CORR_HB;CORR_USED_HB;ML_DB;CORR_DB;CORR_USED_DB
   dfa=pandas.read_csv(f,sep=';',index_col=False)
   dfa.columns = dfa.columns.str.lower()
   
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
      mag_mean_type=eval(station_magnitude_parameters['mag_mean_type'])
   except:
      log_out.write("No parameter 'mag_mean_type' in section 'station_magnitude' of config file")
      sys.exit()
   if mag_mean_type == 'meanamp':
      try:
         amp_mean_type=eval(station_magnitude_parameters['amp_mean_type'])
      except:
         log_out.write("No parameter 'amp_mean_type' in section 'station_magnitude' of config file")
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
   try:
      outliers_cutoff=float(event_magnitude_parameters['outliers_cutoff'])
   except:
      log_out.write("No parameter 'outliers_cutoff' in section 'event_magnitude' of config file")
      sys.exit()
   return dfa,theoP,theoS,delta_corner,max_lowcorner,delta_peaks,use_stcorr_hb,use_stcorr_db,when_no_stcorr_hb,when_no_stcorr_db,mindist,maxdist,hm_cutoff,outliers_max_it,outliers_red_stop,outliers_nstd,outliers_cutoff 

def json_pyml_load(json_in):
   try:
      config=json_in['data']['pyml_conf']
   except:
      config=False
   try:
      origin=json_in['data']['origin']
   except:
      origin=False
   try:
      dfa = pandas.DataFrame(json_in['data']['amplitudes'])
   except:
      dfa=pandas.DataFrame()
   return dfa,config,origin

# JSON ENCODER CLASS
class DataEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)

        if isinstance(o, datetime):
            return o.isoformat()

        return json.JSONEncoder.default(self, o)

def json_response_structure():
    null=''
    generic = {
               "type": null,
               "title": null,
               "status": null,
               "detail": null,
               "instance": null,
               "version": null,
               "request_submitted": null,
               "errors": {
                          "inputfile": [],
                          "arguments": []
                         }
              }
    response = {
                "random_string": null,
                "magnitudes": {},
                "stationmagnitudes": [],
                "log": {}
               }
    
    log = {
           "magnitude": {},
           "stationmagnitudes": []
          }
    log_magnitude = {
                     "hb": {
                            "status": null,   # eg: "warning"
                            "summary": null,  # eg: "In ML  HuttonBoore Station IV_FDMO_--_HH_ingv skipped due to ipodist"
                            "extended": null, # eg: "Distance is 6.045133888690926 km"
                           },
                     "db": {
                            "status": null,   # eg: "warning",
                            "summary": null,  # eg: "In ML  HuttonBoore Station IV_FDMO_--_HH_ingv skipped due to ipodist",
                            "extended": null, # eg: "Distance is 6.045133888690926 km"
                           }
                    }
    log_stationmagnitude = {
                            "net": null,      # eg: "IV"
                            "sta": null,      # eg: "FDMO",
                            "loc": null,      # eg: "--",
                            "band_inst": null,      # eg: "HH",
                            "status": null,   # eg: "warning",
                            "summary": null,  # eg: "In ML  HuttonBoore Station IV_FDMO_--_HH_ingv skipped due to ipodist",
                            "extended": null, # eg: "Distance is 6.045133888690926 km"
                            "channels": []
                           }
    log_stationmagnitude_channel = {
                                     "orientation": null,      # "N",
                                     "status": null,   # "ok",
                                     "summary": null,  # "In ML  HuttonBoore Station IV_FDMO_--_HH_ingv skipped due to ipodist",
                                     "extended": null, # "Distance is 6.045133888690926 km"
                                   }
    magnitudes = {
                  "hb": {},
                  "db": {},
                  "ampmethod": null,
                  "magmethod": null,
                  "loopexitcondition": null
                 }
    emag = {
            "ml": null,
            "std": null,
            "totsta": null,
            "usedsta": null,
            "quality": '--'
           }
            
    stationmagnitude = {
                         "net": null,
                         "sta": null,
                         "cha": null,
                         "loc": null,
                         "amp1": null,
                         "time1": null,
                         "amp2": null,
                         "time2": null,
#                        "lat": null,
#                        "lon": null,
#                        "elev": null,
                         "ep_distance_km": null,
                         "ep_distance_delta": null,
                         "orig_distance": null,
                         "hb": {
                             "ml": null,
                             "w": null 
                         },
                         "db": {
                             "ml": null,
                             "w": null
                         }
                        }
    return response,generic,log,log_magnitude,log_stationmagnitude,log_stationmagnitude_channel,magnitudes,stationmagnitude,emag

def json_pyml_response(r):
    x=json.dumps(r,cls=DataEncoder)
    return x 

###############################################################
###### End of Functions ##########
###### Main starts here ##########
pyml_version="1.0.0"
args = parseArguments()

# setting up geographiclib
geod = Geodesic.WGS84

exit_condition=False
jresponse,jbadrequest,jlog,jlog_mag,jlog_stamag,jlog_stamag_cha,jmagnitudes,jstationmagnitude,jemag = json_response_structure()
# JSON STRUCTURE DEEPCOPY
resp  = copy.deepcopy(jresponse)
log   = copy.deepcopy(jlog)
logm  = copy.deepcopy(jlog_mag)
gresp = copy.deepcopy(jbadrequest)
resp['random_string'] = 'github/ingv/pyml'

if args.out_format.lower() == 'txt':
   magnitudes_out=sys.stdout
   log_out=sys.stderr
elif args.out_format.lower() == 'json':
   magnitudes_out=False
   log_out=False
else:
#   if log_out:
#      sys.stderr.write("The given out format "+args.out_format.lower()+"/"+args.out_format+" is unknown\n")
#   else:
   log_out=False
   gresp['type'] = "https://tools.ietf.org/html/rfc4918#section-11.2"
   gresp['title'] = "Unprocessable entity"
   gresp['status'] = "422"
   gresp['detail'] = "The given data was invalid"
   gresp['instance'] = str(getpass.getuser())+'@'+str(socket.gethostname())
   gresp['version'] = str(pyml_version)
   gresp['request_submitted'] = str(datetime.now())
   gresp['errors']["arguments"].append("The given argument out format "+args.out_format.lower()+"/"+args.out_format+" is unknown")
   exit_condition=True

############## PYML works either in CSV in/out or JSON in/out) #############################################################
### IF input format is csv pyml works reading csv files, with configuration loaded from a dictionary file
if args.in_file_format.lower() == 'csv':
   infile=args.in_file_name
   conf_file = args.conf
   eventid=args.eventid
   dfa,theoP,theoS,delta_corner,max_lowcorner,delta_peaks,use_stcorr_hb,use_stcorr_db,when_no_stcorr_hb,when_no_stcorr_db,mindist,maxdist,hm_cutoff,outliers_max_it,outliers_red_stop,outliers_nstd,outliers_cutoff = standard_pyml_load(infile,eventid,conf_file,log_out,infile)
   if dfa.empty:
      if log_out:
         sys.stderr.write("The given input pyamp file "+infile+" was incomplete\n")
      else:
         gresp['type'] = "https://tools.ietf.org/html/rfc4918#section-11.2"
         gresp['title'] = "Unprocessable entity"
         gresp['status'] = "422"
         gresp['detail'] = "The given data was invalid"
         gresp['instance'] = str(getpass.getuser())+'@'+str(socket.gethostname())
         gresp['version'] = str(pyml_version)
         gresp['request_submitted'] = str(datetime.now())
         gresp['errors']["inputfile"].append("The given input file "+args.in_file_name+" format "+args.in_file_format.lower()+" was incomplete")
         exit_condition=True
elif args.in_file_format.lower() == 'json':
   ### IF JSON argument is given, pyml works reading both input data a configuration options from the same json file, and it writes out results and log ONLY in one single JSON file
   if os.path.exists(args.in_file_name):
      json_in=pandas.read_json(args.in_file_name)
      dfa,config,origin = json_pyml_load(json_in)
      eventid=0
      if dfa.empty or not config or not origin:
         if log_out:
            sys.stderr.write("The given input file "+args.in_file_name+" format "+args.in_file_format.lower()+" was incomplete")
         else:
            gresp['type'] = "https://tools.ietf.org/html/rfc4918#section-11.2"
            gresp['title'] = "Unprocessable entity"
            gresp['status'] = "422"
            gresp['detail'] = "The given data was invalid"
            gresp['instance'] = str(getpass.getuser())+'@'+str(socket.gethostname())
            gresp['version'] = str(pyml_version)
            gresp['request_submitted'] = str(datetime.now())
            gresp['errors']["inputfile"].append("The given input file "+args.in_file_name+" format "+args.in_file_format.lower()+" was incomplete")
            exit_condition=True
   else:
      if log_out:
         sys.stderr.write("No input file "+args.in_file_name+" format "+args.in_file_format.lower()+" found: exit")
      else:
         gresp['type'] = "https://tools.ietf.org/html/rfc4918#section-11.2"
         gresp['title'] = "Unprocessable entity"
         gresp['status'] = "422"
         gresp['detail'] = "The given data was invalid"
         gresp['instance'] = str(getpass.getuser())+'@'+str(socket.gethostname())
         gresp['version'] = str(pyml_version)
         gresp['request_submitted'] = str(datetime.now())
         gresp['errors']["inputfile"].append("No input file "+args.in_file_name+" format "+args.in_file_format.lower()+" found.")
         exit_condition=True

   if exit_condition:
      sys.stdout.write(json_pyml_response(gresp))
      sys.exit()
   # Preconditions
   theoP=config['preconditions']['theoretical_p']
   theoS=config['preconditions']['theoretical_s']
   delta_corner=config['preconditions']['delta_corner']
   max_lowcorner=config['preconditions']['max_lowcorner']
   # Stations magnitude
   mag_mean_type=config['station_magnitude']['mag_mean_type']
   amp_mean_type=config['station_magnitude']['amp_mean_type']
   delta_peaks=config['station_magnitude']['delta_peaks']
   use_stcorr_hb=config['station_magnitude']['use_stcorr_hb']
   use_stcorr_db=config['station_magnitude']['use_stcorr_db']
   when_no_stcorr_hb=config['station_magnitude']['when_no_stcorr_hb']
   when_no_stcorr_db=config['station_magnitude']['when_no_stcorr_db']
   # Event Magnitude
   mindist=config['event_magnitude']['mindist']
   maxdist=config['event_magnitude']['maxdist']
   hm_cutoff=config['event_magnitude']['hm_cutoff']
   outliers_max_it=config['event_magnitude']['outliers_max_it']
   outliers_red_stop=config['event_magnitude']['outliers_red_stop']
   outliers_nstd=config['event_magnitude']['outliers_nstd']
   outliers_cutoff=config['event_magnitude']['outliers_cutoff']
else:
   if log_out:
      sys.stderr.write("The given input format "+args.in_file_format.lower()+" is unknown")
   else:
      gresp['type'] = "https://tools.ietf.org/html/rfc4918#section-11.2"
      gresp['title'] = "Unprocessable entity"
      gresp['status'] = "422"
      gresp['detail'] = "The given data was invalid"
      gresp['instance'] = str(getpass.getuser())+'@'+str(socket.gethostname())
      gresp['version'] = str(pyml_version)
      gresp['request_submitted'] = str(datetime.now())
      gresp['errors']["arguments"].append("The given input format "+args.in_file_format.lower()+" is unknown")
      sys.stdout.write(json_pyml_response(gresp))
   sys.exit()

if args.clipped_info:
   clip=pandas.read_csv(args.clipped_info,sep=';',index_col=False)

cmp_keys=set()
components_N={}
components_E={}
components_Z={}
mcalc=True

if args.dbona_corr:
   try:
      dbcorr=pandas.read_csv(args.dbona_corr,sep=';')
   except:
      log_out.write("File "+args.dbona_corr+" not found\n")
      sys.exit()
else:
   dbcorr=False


start_time = time.perf_counter()
km=1000.
unit=1000 # pyamp units
if args.in_file_format == 'json':
   unit=1 #db units

for index, row in dfa.iterrows():
    #### Testing the basic data ####
    # Channel Info
    try:
       net = row['net']
       net = False if pandas.isna(row['net']) else net
    except:
       net = False
    try:
       sta = row['sta']
       sta = False if pandas.isna(row['sta']) else sta
    except:
       sta = False
    try:
       loc = row['loc']
       loc = "--" if not loc else loc
    except:
       loc = False
    try:
       cha = row['cha']
       cha = False if pandas.isna(row['cha']) else cha
    except:
       cha = False
    # Amplitudes
    try:
       minamp=row['minamp(m)']*unit
    except:
       try:
          minamp=row['amp1']*unit
          minamp = False if pandas.isna(row['amp1']) else minamp
       except:
          minamp = False
    try:
       maxamp=row['maxamp(m)']*unit
    except:
       try:
          maxamp=row['amp2']*unit
          maxamp = False if pandas.isna(row['amp2']) else maxamp
       except:
          maxamp = False

    # Amplitudes times 
    try:
       time_minamp=row['minamptime']
    except:
       try:
          time_minamp=row['time1']
          time_minamp = False if pandas.isna(row['time1']) else time_minamp
       except:
          time_minamp = False
    try:
       time_maxamp=row['maxamptime']
    except:
       try:
          time_maxamp=row['time2']
          time_maxamp = False if pandas.isna(row['time2']) else time_maxamp
       except:
          time_maxamp = False
    # Station's coordinates
    try:
       stla=False if pandas.isna(row['lat']) else float(row['lat'])
       stlo=False if pandas.isna(row['lon']) else float(row['lon'])
       stel=False if pandas.isna(row['elev']) else float(row['elev'])/km
    except:
       stla=False
       stlo=False
       stel=False

    check_list = {
                  'net': net,
                  'sta': sta,
                  'loc': loc,
                  'cha': cha,
                  'minamp': minamp,
                  'maxamp': maxamp,
                  'time_minamp': time_minamp,
                  'time_maxamp': time_maxamp,
                  'stla': stla,
                  'stlo': stlo
                 }
    false_keys = [key for key, value in check_list.items() if not value]
    if 'net' in false_keys or 'sta' in false_keys or 'loc' in false_keys or 'cha' in false_keys:
       continue
    #if 'minamp' in false_keys or 'maxamp' in false_keys:
    #   print(sta)

    try:
        corner_low=float(row['loco'])
    except:
        corner_low=False
    try:
        corner_high=float(row['hico'])
    except:
        corner_high=False
    try:
        met=row['method']

    except:
        met='ingv'
    if args.clipped_info:
       clp=clip.loc[(clip['net'] == net) & (clip['sta'] == sta) & (clip['cha'] == cha)]
       if not clp.empty:
          log_out.write(' '.join(("Component skipped:",str(net),str(sta),str(loc),str(cha)," due to cliping\n")))
          continue
    if corner_low and corner_high and (corner_low >= max_lowcorner or (corner_high-corner_low) < delta_corner):
       log_out.write(' '.join(("Component skipped:",str(net),str(sta),str(loc),str(cha),str(corner_low),str(corner_high),"\n")))
       continue
    seed_id='.'.join((str(net),str(sta),str(loc),str(cha)))
    if log_out:
       log_out.write(' '.join(("Working on",seed_id,'\n')))
    # In this block all the possibile conditions not to use this waveforms are checked so to reduce useless computing time
    # First: get timing info from SAC to soon understand if this is a good cut or not for amplitude determination
    #[distmeters,azi,bazi] = distaz(tr.stla,tr.stlo,tr.evla,tr.evlo)
    #tr.dist=distmeters/1000.
    components_key='_'.join((str(net),str(sta),str(loc),str(cha[0:2])))
    cmp_keys.add(components_key)

    try:
        hypo_dist = row['ipodistance(km)']
        epi_dist_km = row['epidistance(km)']
        epi_dist_deg = False
    except:
        #calcolo le distanze
        evla=float(origin['lat'])
        evlo=float(origin['lon'])
        evdp=float(origin['depth'])
        if not stla or not stlo:
           epi_dist_km = False
           epi_dist_deg = False
           hypo_dist = False
        else:
           geo_dict = geod.Inverse(evla,evlo,stla,stlo)
           #distmeters = geo_dict['s12']
           distdegrees = geo_dict['a12']
           #azi = geo_dict['azi1']
           #bazi = geo_dict['azi2']
           [distmeters,azi,bazi] = distaz(stla,stlo,evla,evlo)
           epi_dist_km = distmeters / km
           epi_dist_deg = distdegrees
           hypo_dist = msqrt(mpow(epi_dist_km,2)+mpow((evdp+stel),2))
        if log_out:
           log_out.write(' '.join(("Coordinates:",str(net),str(sta),str(loc),str(cha),str(stla),str(stlo),str(stel),str(evla),str(evlo),str(evdp),str(epi_dist_km),str(epi_dist_deg),str(hypo_dist),"\n")))
        
    ml = [False]*2
    amp = abs(maxamp-minamp)/2
    components_key_met=components_key+'_'+met
    # Loading stations corrections
    dbsite=sta
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

    if cha[2] == 'N':
       components_N[components_key_met]=[ml,[minamp,maxamp],epi_dist_km,epi_dist_deg,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp,stla,stlo,stel*km]
    elif cha[2] == 'E':
       components_E[components_key_met]=[ml,[minamp,maxamp],epi_dist_km,epi_dist_deg,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp,stla,stlo,stel*km]
    elif cha[2] == 'Z':
       components_Z[components_key_met]=[ml,[minamp,maxamp],epi_dist_km,epi_dist_deg,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp,stla,stlo,stel*km]
    else:
       if log_out:
          log_out.write(' '.join(("Component not recognized for ",str(net),str(sta),str(loc),str(cha),"\n")))

# Hutton and Boore
mean_hb_ml_sta,min_dist,log['stationmagnitudes'] = create_sets(cmp_keys,components_N,components_E,met,mindist,maxdist,delta_peaks,0,when_no_stcorr_hb,use_stcorr_hb,mag_mean_type,amp_mean_type,log['stationmagnitudes'],jlog_stamag,jlog_stamag_cha)
#if log_out:
#   log_out.write("create_sets: the execution time is: "+str(execution_time)+"\n")
if len(mean_hb_ml_sta) == 0:
   msg_sts = 'critical'
   msg_sum='HuttonBoore List is empty'
   msg_ext = 'All the stations missing due to only one channel is present, or out of minmax distance'
   if log_out:
      log_out.write(msg+"\n")
   mlhb = False
else:
   #meanmag_ml = list(list(zip(*meanmag_ml_sta))[1])
   mean_ml = list(list(zip(*mean_hb_ml_sta))[1])
   mean_ml_sta = np_asarray(list(list(zip(*mean_hb_ml_sta))[0]), dtype=object)
   ma_mlh,ma_stdh,ma_ns_s_h,ma_nsh,cond_hb,outliers_hb,weights_hb,wh_hb_fail = calculate_event_ml(mean_ml,mean_ml_sta,outliers_max_it,outliers_red_stop,outliers_nstd,outliers_cutoff,hm_cutoff)
   if ma_mlh:
      mlhb = True
      if wh_hb_fail:
         if log_out:
            log_out.write("Hutton&Boore: Weighted Huber Mean failed, Outliers Removal used instead\n")
         msg_sts='warning'
         msg_sum="Hutton&Boore: Weighted Huber Mean failed, Outliers Removal used instead"
         msg_ext=""
      else:
         msg_sts='ok'
         msg_sum=""
         msg_ext=""
   else:
      mlhb = False
      if log_out:
           log_out.write("Hutton&Boore: Both Weighted Huber Mean and Outliers Removal failed\n")
      msg_sts='critical'
      msg_sum="Hutton&Boore: Both Weighted Huber Mean and Outliers Removal failed"
      msg_ext=""
logm['hb']['status'] = msg_sts
logm['hb']['summary'] = msg_sum
logm['hb']['extended'] = msg_ext
#mm_mlh,mm_stdh,mm_ns_s_h,mm_nsh,cond = calculate_event_ml(meanmag_ml_sta,outliers_max_it,outliers_red_stop)
# Di Bona
mean_db_ml_sta,min_dist,log['stationmagnitudes'] = create_sets(cmp_keys,components_N,components_E,met,mindist,maxdist,delta_peaks,1,when_no_stcorr_db,use_stcorr_db,mag_mean_type,amp_mean_type,log['stationmagnitudes'],jlog_stamag,jlog_stamag_cha)
if len(mean_db_ml_sta) == 0:
   msg_sts = 'critical'
   msg_sum='Dibona List is empty'
   msg_ext = 'All the stations missing due to only one channel is present, or out of minmax distance'
   if log_out:
      log_out.write(msg+"\n")
   mldb = False
else:
   #meanmag_ml = list(list(zip(*meanmag_ml_sta))[1])
   mean_ml = list(list(zip(*mean_db_ml_sta))[1])
   mean_ml_sta = np_asarray(list(list(zip(*mean_db_ml_sta))[0]), dtype=object)
   ma_mld,ma_stdd,ma_ns_s_d,ma_nsd,cond_db,outliers_db,weights_db,wh_db_fail = calculate_event_ml(mean_ml,mean_ml_sta,outliers_max_it,outliers_red_stop,outliers_nstd,outliers_cutoff,hm_cutoff)
   if ma_mld:
      mldb = True
      if wh_db_fail:
         if log_out:
            log_out.write("Di Bona: Weighted Huber Mean failed, Outliers Removal used instead\n")
         msg_st ="warning"
         msg_sum="Di Bona: Weighted Huber Mean failed, Outliers Removal used instead"
         msg_ext=""
      else:
         msg_sts ="ok"
         msg_sum=""
         msg_ext=""
   else:
      mldb = False
      if log_out:
           log_out.write("Di Bona: Both Weighted Huber Mean and Outliers Removal failed\n")
      msg_sts='critical'
      msg_sum="Di Bona: Both Weighted Huber Mean and Outliers Removal failed"
      msg_ext=""

logm['db']['status'] = msg_sts
logm['db']['summary'] = msg_sum
logm['db']['extended'] = msg_ext

resp["log"]['magnitude'] = logm
resp["log"]['stationmagnitudes'] = log['stationmagnitudes']

if not mlhb and not mldb:
   if log_out:
      log_out.write("Neither Hutton&Boore nor Di Bona ML could be calculate\n")
   sys.stdout.write(json_pyml_response(resp))
   sys.exit()
#else:
#   if not mlhb:
#      if log_out:
#         log_out.write("Hutton&Boore ML was impossible to calculate\n")
#      sys.stdout.write(json_pyml_response(resp))
#   elif not mldb:
#      if log_out:
#         log_out.write("Di Bona ML was impossible to calculate\n")
#      sys.stdout.write(json_pyml_response(resp))
#mm_mld,mm_stdd,mm_ns_s_d,mm_nsd,cond = calculate_event_ml(meanmag_ml_sta,outliers_max_it,outliers_red_stop)

######################################################
#          Writing magnitudes.csv file               #
######################################################
#if magnitudes_out:
#   magnitudes_out=open(magnitudes_out,'w')
#else:
#   magnitudes_out=sys.stdout

if mag_mean_type == 'meanmag':
   amp_mean_type='False'

if magnitudes_out:
   magnitudes_out.write("EventID;ML_HB;Std_HB;TOTSTA_HB;USEDSTA_HB;ML_DB;Std_DB;TOTSTA_DB;USEDSTA_DB;ampmethod;amp_mean_type;mag_mean_type;loopexitcondition\n")
   magnitudes_out.write(';'.join((str(eventid),str(ma_mlh),str(ma_stdh),str(ma_ns_s_h),str(ma_nsh),str(ma_mld),str(ma_stdd),str(ma_ns_s_d),str(ma_nsd),met,amp_mean_type,mag_mean_type,cond_hb+'-'+cond_db,'\n')))

# JSON WRITE
jmags = copy.deepcopy(jmagnitudes)
empty_jemag = {}
if ma_mlh:
   jhb = copy.deepcopy(jemag)
   jhb['ml'] = ma_mlh
   jhb['std'] = ma_stdh
   jhb['totsta'] = ma_ns_s_h
   jhb['usedsta'] = ma_nsh
   jhb['quality'] = char_quality(ma_ns_s_h,ma_nsh,mindist,ma_mlh,ma_stdh)
else:
   jhb = copy.deepcopy(empty_jemag)
jmags["hb"].update(jhb) # push oggetto "magnitudo" HB in oggetto magnitudes
if ma_mld:
   jhb = copy.deepcopy(jemag)
   jhb['ml'] = ma_mld
   jhb['std'] = ma_stdd
   jhb['totsta'] = ma_ns_s_d
   jhb['usedsta'] = ma_nsd
   jhb['quality'] = char_quality(ma_ns_s_d,ma_nsd,mindist,ma_mld,ma_stdd)
else:
   jhb = copy.deepcopy(empty_jemag)
jmags["db"].update(jhb) # push oggetto "magnitudo" HB in oggetto magnitudes
jmags["ampmethod"] = met
jmags["magmethod"] = mag_mean_type
jmags["loopexitcondition"] = cond_hb+'-'+cond_db
resp["magnitudes"].update(jmags)

channels_dictionary = {}
for x, y, wx, wy in zip(mean_hb_ml_sta, mean_db_ml_sta, weights_hb, weights_db):
    sth,mh = map(str,x)
    std,md = map(str,y)
    whb = str(wx)
    wdb = str(wy)
    if components_N[sth] and components_E[sth]:
       nwr,swr,lwr,chwr,mwr = sth.split('_')
       ch_N_rewrite = nwr + "_" + swr + "_" + lwr + "_" + chwr + "N"
       ch_E_rewrite = nwr + "_" + swr + "_" + lwr + "_" + chwr + "E"
       ch_rewrite = nwr + "_" + swr + "_" + lwr + "_" + chwr + "_" + met
       if magnitudes_out:
          magnitudes_out.write(' '.join(('MLCHA',ch_N_rewrite,str(components_N[sth][0][0]),str(whb),ch_N_rewrite,str(components_N[sth][0][1]),str(wdb),'\n')))
          magnitudes_out.write(' '.join(("MLCHA",ch_E_rewrite,str(components_E[sth][0][0]),str(whb),ch_E_rewrite,str(components_E[sth][0][1]),str(wdb),'\n')))
       channels_dictionary[ch_rewrite] = [[components_N[sth][0][0],whb,components_N[sth][0][1],wdb],[components_E[sth][0][0],whb,components_E[sth][0][1],wdb]]
if not hm_cutoff or wh_hb_fail or wh_db_fail:
   for x in outliers_hb[0]:
       sth,mh = map(str,list(x))
       if magnitudes_out:
          magnitudes_out.write(' '.join(('OUTL_HB',sth,mh,'\n')))
   for y in outliers_db[0]:
       std,md = map(str,list(y))
       if magnitudes_out:
          magnitudes_out.write(' '.join(('OUTL_DB',std,md,'\n')))


for key in components_N:
    try:
        components_N[key]
    except:
        components_N[key] = False
    try:
        components_E[key]
    except:
        components_E[key] = False
    try:
        channels_dictionary[key]
    except:
        channels_dictionary[key] = False

    n,s,l,c,m = key.split('_')
    jstmag = copy.deepcopy(jstationmagnitude)
    if components_N[key]:
            #logm = copy.deepcopy(jlogmessage)
            #logmch = copy.deepcopy(jlogmessagech)
            jstmag["net"] = n
            jstmag["sta"] = s
            jstmag["cha"] = c + 'N'
            jstmag["loc"] =  "--" if l == 'None' else l
            jstmag["amp1"] = components_N[key][1][0]
            jstmag["time1"] = components_N[key][6]
            jstmag["amp2"] = components_N[key][1][1]
            jstmag["time2"] = components_N[key][7]
            jstmag["ep_distance_km"] = components_N[key][2]
            jstmag["ep_distance_delta"] = components_N[key][3]
            jstmag["orig_distance"] = components_N[key][4]
            if components_N[key][8]:
               jstmag["lat"] = components_N[key][8]
            if components_N[key][9]:
               jstmag["lon"] = components_N[key][9]
            if components_N[key][10]:
               jstmag["elev"] = components_N[key][10]
            if channels_dictionary[key]:
               jstmag["hb"] = {"ml": channels_dictionary[key][0][0], "w": float(channels_dictionary[key][0][1])}
               jstmag["db"] = {"ml": channels_dictionary[key][0][2], "w": float(channels_dictionary[key][0][3])}
            else:
               jstmag["hb"] = {}
               jstmag["db"] = {}
            #logmch["net"] = jstmag["net"] 
            #logmch["sta"] = jstmag["sta"]
            #logmch["loc"] = jstmag["loc"]
            #logmch["cha"] = jstmag["cha"]
            #logmch['status'] = 'ok'
            #logmch['level'] = 'channel'
            #logmch['info'] = {"summary": n+' '+s+' '+jstmag["loc"]+' '+c+'E', "extended": ''}
            if components_N[key][8] and components_N[key][9] and components_N[key][10]:
               resp["stationmagnitudes"].append(jstmag)
            #resp["log"].append(logmch)
    jstmag = copy.deepcopy(jstationmagnitude)
    if components_E[key]:
            #logm = copy.deepcopy(jlogmessage)
            #logmch = copy.deepcopy(jlogmessagech)
            jstmag["net"] = n
            jstmag["sta"] = s
            jstmag["cha"] = c + 'E'
            jstmag["loc"] =  "--" if l == 'None' else l
            jstmag["amp1"] = components_E[key][1][0]
            jstmag["time1"] = components_E[key][6]
            jstmag["amp2"] = components_E[key][1][1]
            jstmag["time2"] = components_E[key][7]
            jstmag["ep_distance_km"] = components_E[key][2]
            jstmag["ep_distance_delta"] = components_E[key][3]
            jstmag["orig_distance"] = components_E[key][4]
            if components_E[key][8]:
               jstmag["lat"] = components_E[key][8]
            if components_E[key][9]:
               jstmag["lon"] = components_E[key][9]
            if components_E[key][10]:
               jstmag["elev"] = components_E[key][10]
            if channels_dictionary[key]:
               jstmag["hb"] = {"ml": channels_dictionary[key][1][0], "w": float(channels_dictionary[key][1][1])}
               jstmag["db"] = {"ml": channels_dictionary[key][1][2], "w": float(channels_dictionary[key][1][3])}
            else:
               jstmag["hb"] = {}
               jstmag["db"] = {}
               
            #logmch["net"] = jstmag["net"] 
            #logmch["sta"] = jstmag["sta"]
            #logmch["loc"] = jstmag["loc"]
            #logmch["cha"] = jstmag["cha"]
            #logmch['status'] = 'ok'
            #logmch['level'] = 'channel'
            #logmch['info'] = {"summary": n+' '+s+' '+jstmag["loc"]+' '+c+'N', "extended": ''}
            if components_E[key][8] and components_E[key][9] and components_E[key][10]:
               resp["stationmagnitudes"].append(jstmag)
            #resp["log"].append(logmch)
            #resp["stationmagnitudes"].append(jstmag)
    
if args.out_format == 'json':
   sys.stdout.write(json_pyml_response(resp))

# Now closing all output files
if magnitudes_out:
   magnitudes_out.close()
#main_end_time = time.perf_counter()
#main_execution_time = main_end_time - main_start_time
if log_out:
   #log_out.write("MAIN: the execution time is: "+str(main_execution_time)+"\n")
   log_out.close()
sys.exit()
