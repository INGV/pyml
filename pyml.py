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

import argparse,sys,os,glob,copy,pwd,pathlib,itertools
import geographiclib
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

from math import sqrt as msqrt, log10 as mlog10, pow as mpow

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def parseArguments():
        parser=MyParser()	
        parser.add_argument('--json',        default=None,          help='json config and amplitudes file')
        parser.add_argument('--infile',      default=None,          help='pyamp-amplitudes.csv file full path')
        parser.add_argument('--eventid',     default='0',           help='Unique identifier of the event')
        parser.add_argument('--conf',        default='./pyml.conf', help='A file containing sections and related parameters (see the example)')
        parser.add_argument('--dbona_corr',  default='dbcor.csv',   help='Input file with DiBona Stations corrections')
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
                print("skip: %s" % option)
        except:
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


def create_sets(keys,cmpn,cmpe,mtd,mid,mad,dp,mty,whstc,stc,resp,jlogmessage):
    # mtd is the peakmethod
    # mty is the huttonboore 0, dibona 1
    # a channel cmpn is [ml,amp,tr.dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp]
    #   where ml is a list of two: [ml_hutton,ml_dibona]
    meanmag_ml_set=[]
    meanamp_ml_set=[]
    for k in keys:
        kk=k+'_'+mtd
        logm = copy.deepcopy(jlogmessage)
        logm['instance'] = 'function create_sets' 
        if kk in cmpn and kk in cmpe: # if both components are present in the set
           if not cmpn[kk][2] or not cmpe[kk][2]:
              epidist = False
              ipodist = False
              log_out.write(' '.join(("Station skipped due to stations coordinates missing: ",str(k),str(ipodist),"\n")))
              logm['status'] = '422' 
              logm['type'] = 'https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/422'
              logm['title'] = ' '.join(("In ML ",mtytxt,"Station",str(kk),"skipped due to stations coordinates missing"))
              logm['detail'] = ' '.join(("Distance is",str(ipodist),"km"))
              resp["log"].append(logm)
              continue
           epidist = (cmpn[kk][2] + cmpe[kk][2])/2 # epicentral distance
           ipodist = (cmpn[kk][3] + cmpe[kk][3])/2 # ipocentral distance
           amp_e = abs(cmpe[kk][1][1]-cmpe[kk][1][0])/2 # now both the minamp and maxamp values are reported in the channels arrays so amplitude richter must be calculated here
           amp_n = abs(cmpn[kk][1][1]-cmpn[kk][1][0])/2 # now both the minamp and maxamp values are reported in the channels arrays so amplitude richter must be calculated here
           if ipodist >= mid and ipodist <= mad: # if ipocentral distance is within the accepted range
              if mtd != 'free' or (mtd == 'free' and abs(cmpn[kk][6]-cmpn[kk][5]) <= dp and abs(cmpe[kk][6]-cmpe[kk][5]) <= dp): # if the method is not free the deltapeak has no meaning orherwise it is evaluated
                 #Mean of channel magnitudes is calculated
                 if cmpn[kk][0][mty] and cmpe[kk][0][mty]:
                    mm = (cmpn[kk][0][mty] + cmpe[kk][0][mty])/2
                    meanmag_ml_set.append([kk,mm])
                 # Magnitudes of Mean channel amplitudes is calculated if 
                 if not stc or (cmpn[kk][4][mty] and cmpe[kk][4][mty]) or whstc: 
                    mean_amp_ari = (amp_n + amp_e)/2 # Artimetic mean
                    mean_amp_geo = msqrt(amp_n * amp_e) # Geometric mean that is the correct one according to Di Bona
                    corr = (cmpn[kk][4][mty] + cmpe[kk][4][mty])/2 if cmpn[kk][4][mty] and cmpe[kk][4][mty] else False
                    if mty == 0:
                       ma = huttonboore(mean_amp_geo,ipodist,corr,stc)
                       mtytxt = 'HuttonBoore'
                    if mty == 1:
                       ma = dibona(mean_amp_geo,ipodist,corr,stc)
                       mtytxt = 'DiBona'
                    meanamp_ml_set.append([kk,ma])
              else:
                 log_out.write(' '.join(("Station skipped due to amp minmax distance: ",str(kk),str(abs(cmpn[kk][6]-cmpn[kk][5])),"\n")))
                 logm['status'] = '422' 
                 logm['type'] = 'https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/422'
                 logm['title'] = ' '.join(("In ML ",mtytxt,"Station",str(kk),"skipped due to amp minmax distance"))
                 logm['detail'] = ' '.join(("Distance is",str(abs(cmpn[kk][6]-cmpn[kk][5])),"s"))
                 resp["log"].append(logm)
           else:
              log_out.write(' '.join(("Station skipped due to ipodist: ",str(k),str(ipodist),"\n")))
              logm['status'] = '422' 
              logm['type'] = 'https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/422'
              logm['title'] = ' '.join(("In ML ",mtytxt,"Station",str(kk),"skipped due to ipodist"))
              logm['detail'] = ' '.join(("Distance is",str(ipodist),"km"))
              resp["log"].append(logm)
        else:
           log_out.write(' '.join(("Station skipped due to missing channel ",str(kk),'\n')))
           logm['status'] = '422' 
           logm['type'] = 'https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/422'
           logm['title'] = ' '.join(("In ML ",mtytxt,"Station",str(kk),"skipped due to missing channel"))
           missing="N" if kk not in cmpn else "E"
           logm['detail'] = ' '.join(("Distance is",missing,"km"))
           resp["log"].append(logm)
    return meanmag_ml_set,meanamp_ml_set


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
    w_mean = np_sum(v * w)/np_sum(w)
    w_std = wstd(v,w_mean,w)
    w_std = 0.0 if not w_std else w_std
    return w_mean,w_std,w

def rm_outliers(v,v_flag,v_mean,v_std,times_std,co,var_stop,it_max,skip):
    res = abs(v - v_mean)
    v_mean_old = v_mean
    cut_limit = times_std * v_std if (times_std * v_std) > co else co
    not_outlier = res  < cut_limit
    yes_outlier = res >= cut_limit
    skip.append(list(zip(v_flag[yes_outlier],v[yes_outlier])))
    v = v[not_outlier]
    v_flag = v_flag[not_outlier]
    if len(v) > 0:
       v_std  = median_abs_deviation(v)
       v_mean = np_median(v)
       n_v_flag = len(v)
    else:
       v_std  = False
       v_mean = False
       n_v_flag = False
    w_fake = np_ones(len(v))
    return v_mean,v_std,v,v_flag,w_fake,skip

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
    removed=[]
    if hm_cutoff:
       ruse,zero = [hm_cutoff,9999.0] if isinstance(hm_cutoff,float) else hm_cutoff
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
             xmd,xmd_std,v,s,weights,removed = rm_outliers(v,s,xmd,xmd_std,max_dev,out_cutoff,var_stop,it_max,removed)
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

def standard_pyml_load(infile,eventid,conf_file):
   # Now loading the configuration file
   if os.path.exists(conf_file) and os.path.getsize(conf_file) > 0:
      paramfile=conf_file
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
   return dfa,magnitudes_out,log_out,theoP,theoS,delta_corner,max_lowcorner,delta_peaks,use_stcorr_hb,use_stcorr_db,when_no_stcorr_hb,when_no_stcorr_db,mindist,maxdist,hm_cutoff,outliers_max_it,outliers_red_stop,outliers_nstd,outliers_cutoff 

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
    null=False
    response = {
                "random_string": null,
                "magnitudes": {},
                "stationmagnitudes": [],
                "log": []
               }
    logmessage = {
             "title": null,
             "instance": null,
             "detail": null,
             "status": null,
             "type": null
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
            "usedsta": null
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
                         "lat": null,
                         "lon": null,
                         "elev": null,
                         "hb": {
                             "ml": null,
                             "w": null 
                         },
                         "db": {
                             "ml": null,
                             "w": null
                         }
                        }
    return response,logmessage,magnitudes,stationmagnitude,emag

def json_pyml_response(r):
    x=json.dumps(r,cls=DataEncoder)
    return x 

###### End of Functions ##########
## Main ##
main_start_time = time.perf_counter()
args = parseArguments()
if not args.json:
   infile=args.infile
   conf_file = args.conf
   eventid=args.eventid
   dfa,magnitudes_out,log_out,theoP,theoS,delta_corner,max_lowcorner,delta_peaks,use_stcorr_hb,use_stcorr_db,when_no_stcorr_hb,when_no_stcorr_db,mindist,maxdist,hm_cutoff,outliers_max_it,outliers_red_stop,outliers_nstd,outliers_cutoff = standard_pyml_load(infile,eventid,conf_file)
   if dfa.empty:
      sys.stderr.write("The given input json file "+args.json+" was incomplete\n")
      sys.exit()
else:
   if os.path.exists(args.json):
      json_in=pandas.read_json(args.json)
      dfa,config,origin = json_pyml_load(json_in)
      eventid=0
      if dfa.empty or not config or not origin:
         sys.stderr.write("The given input json file "+args.json+" was incomplete\n")
         sys.exit()
   else:
      sys.stderr.write("No Json input file "+args.json+" found: exit\n")
      sys.exit()

   # Files out
   magnitudes_out=config['iofilenames']['magnitudes']
   log_out=config['iofilenames']['log']
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

if args.clipped_info:
   clip=pandas.read_csv(args.clipped_info,sep=';',index_col=False)

if log_out:
   log_out=open(log_out,'w')
else:
   log_out=sys.stderr
if args.json:
   log_out.write("Working on json input\n")
else:
   log_out.write("Working on standard pyamp input\n")
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


km=1000.
# Standard header
#Net;Sta;Loc;Cha;Lat;Lon;Ele;EpiDistance(km);IpoDistance(km);MinAmp(m);MinAmpTime;MaxAmp(m);MaxAmpTime;DeltaPeaks;Method;NoiseWinMin;NoiseWinMax;SignalWinMin;SignalWinMax;P_Pick;Synth;S_Picks;Synth;LoCo;HiCo;LenOverSNRIn;SNRIn;ML_H;CORR_HB;CORR_USED_HB;ML_DB;CORR_DB;CORR_USED_DB
start_time = time.perf_counter()
for index, row in dfa.iterrows():
    try:
        net = str(row['Net'])
    except:
        net = row['net']
    try:
        sta = str(row['Sta'])
    except:
        sta = row['sta']
    try:
        loc = str(row['Loc'])
    except:
        loc = row['loc']
    try:
        cha = str(row['Cha'])
    except:
        cha = row['cha']
    try:
        corner_low=float(row['LoCo'])
    except:
        corner_low=False
    try:
        corner_high=float(row['HiCo'])
    except:
        corner_high=False
    if args.clipped_info:
       clp=clip.loc[(clip['net'] == net) & (clip['sta'] == sta) & (clip['cha'] == cha)]
       if not clp.empty:
          log_out.write(' '.join(("Component skipped:",str(net),str(sta),str(loc),str(cha)," due to cliping\n")))
          continue
    if corner_low and corner_high and (corner_low >= max_lowcorner or (corner_high-corner_low) < delta_corner):
       log_out.write(' '.join(("Component skipped:",str(net),str(sta),str(loc),str(cha),str(corner_low),str(corner_high),"\n")))
       continue
    seed_id='.'.join((str(net),str(sta),str(loc),str(cha)))
    log_out.write(' '.join(("Working on",seed_id,'\n')))
    # In this block all the possibile conditions not to use this waveforms are checked so to reduce useless computing time
    # First: get timing info from SAC to soon understand if this is a good cut or not for amplitude determination
    #[distmeters,azi,bazi] = distaz(tr.stla,tr.stlo,tr.evla,tr.evlo)
    #tr.dist=distmeters/1000.
    components_key='_'.join((str(net),str(sta),str(loc),str(cha[0:2])))
    cmp_keys.add(components_key)
# Channel Magnitudes calculations and in case of event_magnitude argument on ... station magnitude calculation
#net   sta  cha loc        lat      lon  elev   amp1                     time1   amp2                     time2
    try:
        hypo_dist = row['IpoDistance(km)']
        epi_dist = row['EpiDistance(km)']
    except:
        #calcolo le distanze
        stla=False if pandas.isna(row['lat']) else float(row['lat'])
        stlo=False if pandas.isna(row['lon']) else float(row['lon'])
        stel=False if pandas.isna(row['elev']) else float(row['elev'])/km
        evla=float(origin['lat'])
        evlo=float(origin['lon'])
        evdp=float(origin['depth'])
        if not stla or not stlo:
           epi_dist = False
           hypo_dist = False
        else:
           [distmeters,azi,bazi] = distaz(stla,stlo,evla,evlo)
           epi_dist = distmeters / km
           hypo_dist = msqrt(mpow(epi_dist,2)+mpow((evdp+stel),2))
        log_out.write(' '.join(("Coordinates:",str(net),str(sta),str(loc),str(cha),str(stla),str(stlo),str(stel),str(evla),str(evlo),str(evdp),str(epi_dist),str(hypo_dist),"\n")))
        
    ml = [False]*2
    #minamp,maxamp,time_minamp,time_maxamp,amp,met = amp_method[2:]
    unit=1000 # pyamp units
    if args.json:
       unit=1 #db units
    try:
        minamp=row['MinAmp(m)']*unit
        time_minamp=row['MinAmpTime']
    except:
        minamp=row['amp1']*unit
        time_minamp=row['time1']
    try:
        maxamp=row['MaxAmp(m)']*unit
        time_maxamp=row['MaxAmpTime']
    except:
        maxamp=row['amp2']*unit
        time_maxamp=row['time2']
    amp = abs(maxamp-minamp)/2
    try:
        met=row['Method']
    except:
        met='ingv'
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
       components_N[components_key_met]=[ml,[minamp,maxamp],epi_dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp,stla,stlo,stel*km]
    elif cha[2] == 'E':
       components_E[components_key_met]=[ml,[minamp,maxamp],epi_dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp,stla,stlo,stel*km]
    elif cha[2] == 'Z':
       components_Z[components_key_met]=[ml,[minamp,maxamp],epi_dist,hypo_dist,[s_hutton,s_dibona],time_minamp,time_maxamp,stla,stlo,stel*km]
    else:
       log_out.write(' '.join(("Component not recognized for ",str(net),str(sta),str(loc),str(cha),"\n")))
end_time = time.perf_counter()
execution_time = end_time - start_time
log_out.write("dfa: the execution time is: "+str(execution_time)+"\n")

jresponse,jlogmessage,jmagnitudes,jstationmagnitude,jemag = json_response_structure()
resp = copy.deepcopy(jresponse)
resp['random_string'] = 'testraf'
# Hutton and Boore
start_time = time.perf_counter()
meanmag_ml_sta,meanamp_hb_ml_sta = create_sets(cmp_keys,components_N,components_E,met,mindist,maxdist,delta_peaks,0,when_no_stcorr_hb,use_stcorr_hb,resp,jlogmessage)
end_time = time.perf_counter()
execution_time = end_time - start_time
log_out.write("create_sets: the execution time is: "+str(execution_time)+"\n")
if len(meanmag_ml_sta) == 0 or meanamp_hb_ml_sta == 0:
   msg='HuttonBoore List is empty'
   log_out.write(msg+"\n")
   mlhb = False
   logm = copy.deepcopy(jlogmessage)
   logm['status'] = '204'
   logm['type'] = 'https://www.rfc-editor.org/rfc/rfc7231#section-6.3.5'
   logm['title'] = msg
   logm['detail'] = 'All the stations missing due to only one channel is present, or out of minmax distance'
   resp["log"].append(logm)
else:
   meanmag_ml = list(list(zip(*meanmag_ml_sta))[1])
   meanamp_ml = list(list(zip(*meanamp_hb_ml_sta))[1])
   meanamp_ml_sta = np_asarray(list(list(zip(*meanamp_hb_ml_sta))[0]), dtype=object)
   start_time = time.perf_counter()
   ma_mlh,ma_stdh,ma_ns_s_h,ma_nsh,cond_hb,outliers_hb,weights_hb,wh_hb_fail = calculate_event_ml(meanamp_ml,meanamp_ml_sta,outliers_max_it,outliers_red_stop,outliers_nstd,outliers_cutoff,hm_cutoff)
   end_time = time.perf_counter()
   execution_time = end_time - start_time
   log_out.write("calculate_event_ml HB: the execution time is: "+str(execution_time)+"\n")
   mlhb = True
   if wh_hb_fail:
      print("Hutto_Boore: whuber mean failed, rm_outl used")
      mlhb = False
#mm_mlh,mm_stdh,mm_ns_s_h,mm_nsh,cond = calculate_event_ml(meanmag_ml_sta,outliers_max_it,outliers_red_stop)
# Di Bona
meanmag_ml_sta,meanamp_db_ml_sta = create_sets(cmp_keys,components_N,components_E,met,mindist,maxdist,delta_peaks,1,when_no_stcorr_db,use_stcorr_db,resp,jlogmessage)
if len(meanmag_ml_sta) == 0 or meanamp_db_ml_sta == 0:
   msg='Dibona List is empty'
   log_out.write(msg+"\n")
   mldb = False
   logm = copy.deepcopy(jlogmessage)
   logm['status'] = '204'
   logm['type'] = 'https://www.rfc-editor.org/rfc/rfc7231#section-6.3.5'
   logm['title'] = msg
   logm['detail'] = 'All the stations missing due to only one channel is present, or out of minmax distance'
   resp["log"].append(logm)
else:
   meanmag_ml = list(list(zip(*meanmag_ml_sta))[1])
   meanamp_ml = list(list(zip(*meanamp_db_ml_sta))[1])
   meanamp_ml_sta = np_asarray(list(list(zip(*meanamp_db_ml_sta))[0]), dtype=object)
   start_time = time.perf_counter()
   ma_mld,ma_stdd,ma_ns_s_d,ma_nsd,cond_db,outliers_db,weights_db,wh_db_fail = calculate_event_ml(meanamp_ml,meanamp_ml_sta,outliers_max_it,outliers_red_stop,outliers_nstd,outliers_cutoff,hm_cutoff)
   end_time = time.perf_counter()
   execution_time = end_time - start_time
   log_out.write("calculate_event_ml DB: the execution time is: "+str(execution_time)+"\n")
   mldb = True
   if wh_db_fail:
      print("Di Bona: whuber mean failed, rm_outl used")
if not mlhb or not mldb:
   log_out.write("Either Hutton and Boore or Di Bona ML was impossible to calculate\n")
   sys.stderr.write(json_pyml_response(resp))
   sys.exit()
#mm_mld,mm_stdd,mm_ns_s_d,mm_nsd,cond = calculate_event_ml(meanmag_ml_sta,outliers_max_it,outliers_red_stop)

######################################################
#          Writing magnitudes.csv file               #
######################################################
if magnitudes_out:
   magnitudes_out=open(magnitudes_out,'w')
else:
   magnitudes_out=sys.stdout
magnitudes_out.write("EventID;ML_HB;Std_HB;TOTSTA_HB;USEDSTA_HB;ML_DB;Std_DB;TOTSTA_DB;USEDSTA_DB;ampmethod;magmethod;loopexitcondition\n")
magnitudes_out.write(';'.join((str(eventid),str(ma_mlh),str(ma_stdh),str(ma_ns_s_h),str(ma_nsh),str(ma_mld),str(ma_stdd),str(ma_ns_s_d),str(ma_nsd),met,mag_mean_type,cond_hb+'-'+cond_db,'\n')))

# JSON WRITE
jmags = copy.deepcopy(jmagnitudes)
jhb = copy.deepcopy(jemag)
jhb['ml'] = ma_mlh
jhb['std'] = ma_stdh
jhb['totsta'] = ma_ns_s_h
jhb['usedsta'] = ma_nsh
jmags["hb"].update(jhb) # push oggetto "magnitudo" HB in oggetto magnitudes
jhb = copy.deepcopy(jemag)
jhb['ml'] = ma_mld
jhb['std'] = ma_stdd
jhb['totsta'] = ma_ns_s_d
jhb['usedsta'] = ma_nsd
jmags["db"].update(jhb) # push oggetto "magnitudo" HB in oggetto magnitudes
jmags["ampmethod"] = met
jmags["magmethod"] = mag_mean_type
jmags["loopexitcondition"] = cond_hb+'-'+cond_db
resp["magnitudes"].update(jmags)

#json.dump(jmags,sys.stdout)

#magnitudes_out.write(';'.join((str(eventid),str(mm_mlh),str(mm_stdh),str(mm_ns_s_h),str(mm_nsh),str(mm_mld),str(mm_stdd),str(mm_ns_s_d),str(mm_nsd),met,'meanmag',cond,'\n')))
channels_dictionary = {}
for x, y, wx, wy in zip(meanamp_hb_ml_sta, meanamp_db_ml_sta, weights_hb, weights_db):
    sth,mh = map(str,x)
    std,md = map(str,y)
    whb = str(wx)
    wdb = str(wy)
    #magnitudes_out.write(' '.join(('MLSTA',sth,mh,whb,std,md,wdb,'\n')))
    #MLSTA IV_MIDA_None_HN_ingv 3.1782901276644764 1.0 IV_MIDA_None_HN_ingv 3.158132200624496 1.0
    if components_N[sth] and components_E[sth]:
       nwr,swr,lwr,chwr,mwr = sth.split('_')
       ch_N_rewrite = nwr + "_" + swr + "_" + lwr + "_" + chwr + "N"
       ch_E_rewrite = nwr + "_" + swr + "_" + lwr + "_" + chwr + "E"
       ch_rewrite = nwr + "_" + swr + "_" + lwr + "_" + chwr + "_" + met
       magnitudes_out.write(' '.join(('MLCHA',ch_N_rewrite,str(components_N[sth][0][0]),str(whb),ch_N_rewrite,str(components_N[sth][0][1]),str(wdb),'\n')))
       magnitudes_out.write(' '.join(("MLCHA",ch_E_rewrite,str(components_E[sth][0][0]),str(whb),ch_E_rewrite,str(components_E[sth][0][1]),str(wdb),'\n')))
       channels_dictionary[ch_rewrite] = [[components_N[sth][0][0],whb,components_N[sth][0][1],wdb],[components_E[sth][0][0],whb,components_E[sth][0][1],wdb]]
if not hm_cutoff or wh_hb_fail or wh_db_fail:
   for x in outliers_hb[0]:
       sth,mh = map(str,list(x))
       magnitudes_out.write(' '.join(('OUTL_HB',sth,mh,'\n')))
   for y in outliers_db[0]:
       std,md = map(str,list(y))
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
            jstmag["net"] = n
            jstmag["sta"] = s
            jstmag["cha"] = c + 'N'
            jstmag["loc"] =  "--" if l == 'None' else l
            jstmag["amp1"] = components_N[key][1][0]
            jstmag["time1"] = components_N[key][5]
            jstmag["amp2"] = components_N[key][1][1]
            jstmag["time2"] = components_N[key][6]
            if components_N[key][7]:
               jstmag["lat"] = components_N[key][7]
            if components_N[key][8]:
               jstmag["lon"] = components_N[key][8]
            if components_N[key][9]:
               jstmag["elev"] = components_N[key][9]
            if channels_dictionary[key]:
               jstmag["hb"] = {"ml": channels_dictionary[key][0][0], "w": channels_dictionary[key][0][1]}
               jstmag["db"] = {"ml": channels_dictionary[key][0][2], "w": channels_dictionary[key][0][3]}
            resp["stationmagnitudes"].append(jstmag)
    jstmag = copy.deepcopy(jstationmagnitude)
    if components_E[key]:
            jstmag["net"] = n
            jstmag["sta"] = s
            jstmag["cha"] = c + 'E'
            jstmag["loc"] =  "--" if l == 'None' else l
            jstmag["amp1"] = components_E[key][1][0]
            jstmag["time1"] = components_E[key][5]
            jstmag["amp2"] = components_E[key][1][1]
            jstmag["time2"] = components_E[key][6]
            if components_E[key][7]:
               jstmag["lat"] = components_E[key][7]
            if components_E[key][8]:
               jstmag["lon"] = components_E[key][8]
            if components_E[key][9]:
               jstmag["elev"] = components_E[key][9]
            if channels_dictionary[key]:
               jstmag["hb"] = {"ml": channels_dictionary[key][1][0], "w": channels_dictionary[key][1][1]}
               jstmag["db"] = {"ml": channels_dictionary[key][1][2], "w": channels_dictionary[key][1][3]}
            resp["stationmagnitudes"].append(jstmag)
    
sys.stdout.write(json_pyml_response(resp))

# Now closing all output files
magnitudes_out.close()
main_end_time = time.perf_counter()
main_execution_time = main_end_time - main_start_time
log_out.write("MAIN: the execution time is: "+str(main_execution_time)+"\n")
log_out.close()
sys.exit()
