# Authors:
#   Raffaele Di Stefano (raffaele.distefano@ingv.it),
#   Barbara Castello (barbara.castello@ingv.it)
#   Coding of present version powered by Gemini 3.1 Pro
# Aknowledgements: Valentino Lauciani (valentino.lauciani@ingv.it) and Andrea Bono (andrea.bono@ingv.it) - adaptation to the monitoring system
# Licence: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

# --- Libreria Standard Python ---
import argparse
import copy
import decimal
import json
import os
import socket
import sys
import logging
from datetime import datetime
import configparser as cp
from math import log2

# --- Librerie Scientifiche e di Analisi Dati ---
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

# --- Librerie Geodetiche e Sismologiche ---
from geographiclib.geodesic import Geodesic
from obspy.geodetics.base import gps2dist_azimuth


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)


def parse_arguments():
    parser = MyParser()
    parser.add_argument('--in_file_name', default=None, help='Name (and path) of input amplitudes file')
    parser.add_argument('--in_file_format', default='json',
                        help='options are json (data and config in one single file) or csv (pyamp format): this latter needs also the --conf argument')
    parser.add_argument('--conf', default=None,
                        help='A file containing sections and related parameters (used only with --csv)')
    parser.add_argument('--out_format', default='json', help='options are json or txt')
    parser.add_argument('--eventid', default='0',
                        help='Unique identifier of the event (only used with --csv_in and --conf)')
    parser.add_argument('--dbona_corr', default='dbcor.csv', help='Input file with DiBona Stations corrections')
    parser.add_argument('--clipped_info', help='Input file with information on clipped channels')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_config_dictionary(cfg, section):
    dict1 = {}
    options = cfg.options(section)
    for option in options:
        try:
            dict1[option] = cfg.get(section, option)
            if dict1[option] == -1:
                sys.stderr.write(f"skip: {option}")
        except (cp.NoOptionError, cp.NoSectionError) as ex:
            sys.stderr.write(f"Exception on {option}! {ex}\n")
            dict1[option] = None
    return dict1


def process_amplitudes_vectorized(df_amp, ev_origin, unit_val):
    """
    Pulisce i dati, gestisce le distanze (sia precalcolate CSV che da calcolare via Obspy/JSON)
    ed estrae il metodo operativo in logica Pandas.
    """
    df_amp['net'] = df_amp.get('net', pd.Series(dtype=object)).fillna(False)
    df_amp['sta'] = df_amp.get('sta', pd.Series(dtype=object)).fillna(False)
    df_amp['loc'] = df_amp.get('loc', pd.Series(dtype=object)).fillna('--')
    df_amp['cha'] = df_amp.get('cha', pd.Series(dtype=object)).fillna(False)

    method_type = str(df_amp['method'].iloc[0]) if 'method' in df_amp.columns else 'ingv'

    minamp_col = 'minamp(m)' if 'minamp(m)' in df_amp.columns else 'amp1'
    maxamp_col = 'maxamp(m)' if 'maxamp(m)' in df_amp.columns else 'amp2'
    time_min_col = 'minamptime' if 'minamptime' in df_amp.columns else 'time1'
    time_max_col = 'maxamptime' if 'maxamptime' in df_amp.columns else 'time2'

    df_amp['minamp'] = df_amp[minamp_col].astype(np.float64) * unit_val
    df_amp['maxamp'] = df_amp[maxamp_col].astype(np.float64) * unit_val
    df_amp['amp_calc'] = np.abs(df_amp['maxamp'] - df_amp['minamp']) / 2.0

    df_amp['time_minamp'] = df_amp[time_min_col]
    df_amp['time_maxamp'] = df_amp[time_max_col]

    valid_mask = (df_amp['net'] != False) & (df_amp['sta'] != False) & (df_amp['cha'] != False)
    df_valid = df_amp[valid_mask].copy()

    if 'ipodistance(km)' in df_valid.columns and 'epidistance(km)' in df_valid.columns:
        df_valid['epi_dist_km'] = df_valid['epidistance(km)'].astype(np.float64)
        df_valid['hypo_dist'] = df_valid['ipodistance(km)'].astype(np.float64)
        df_valid['epi_azimuth'] = 0.0
    else:
        if ev_origin and 'lat' in ev_origin and 'lon' in ev_origin:
            valid_coord_mask = df_valid['lat'].notna() & df_valid['lon'].notna()
            df_valid = df_valid[valid_coord_mask].copy()

            evla = float(ev_origin['lat'])
            evlo = float(ev_origin['lon'])
            evdp = float(ev_origin['depth'])

            def calc_dist_az(stla, stlo):
                distmeters, azi, _ = gps2dist_azimuth(stla, stlo, evla, evlo)
                return distmeters / 1000.0, azi

            vec_dist_az = np.vectorize(calc_dist_az, otypes=[np.float64, np.float64])
            df_valid['epi_dist_km'], df_valid['epi_azimuth'] = vec_dist_az(
                df_valid['lat'].to_numpy(),
                df_valid['lon'].to_numpy()
            )

            df_valid['stel_km'] = df_valid['elev'].astype(np.float64) / 1000.0
            df_valid['hypo_dist'] = np.sqrt(
                np.square(df_valid['epi_dist_km']) + np.square(evdp + df_valid['stel_km'])
            )
        else:
            sys.stderr.write("Dati di origine mancanti per il calcolo delle distanze.\n")
            sys.exit(1)

    return df_valid, method_type


def calc_huttonboore_vec(a, d, s, uc):
    s_arr = np.where(uc, s, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_val = np.where(a > 0, np.log10(a) + 1.110 * np.log10(d / 100.) + 0.00189 * (d - 100.) + 3.0 + s_arr, np.nan)
    return mag_val


def calc_dibona_vec(a, d, s, uc):
    s_arr = np.where(uc, s, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mag_val = np.where(a > 0, np.log10(a) + 1.667 * np.log10(d / 100.) + 0.001736 * (d - 100.) + 3.0 + s_arr,
                           np.nan)
    return mag_val


def apply_magnitudes_and_group(df_valid, db_corr, use_stcorr_h, use_stcorr_d, when_no_stcorr_h, when_no_stcorr_d):
    """
    Calcola ML in vettoriale e prepara i dati per l'aggregazione di stazione.
    """
    if db_corr is not False and not db_corr.empty:
        df_valid = pd.merge(df_valid, db_corr[['sta', 'c']], on='sta', how='left')
        df_valid.rename(columns={'c': 's_dibona'}, inplace=True)
        df_valid['s_dibona'] = df_valid['s_dibona'].fillna(False)
    else:
        df_valid['s_dibona'] = False

    df_valid['s_hutton'] = False

    calc_hb_mask = (df_valid['s_hutton'] != False) | ((df_valid['s_hutton'] == False) & when_no_stcorr_h)
    calc_db_mask = (df_valid['s_dibona'] != False) | ((df_valid['s_dibona'] == False) & when_no_stcorr_d)

    df_valid['ml_hb'] = np.nan
    df_valid.loc[calc_hb_mask, 'ml_hb'] = calc_huttonboore_vec(
        df_valid.loc[calc_hb_mask, 'amp_calc'].to_numpy(),
        df_valid.loc[calc_hb_mask, 'hypo_dist'].to_numpy(),
        df_valid.loc[calc_hb_mask, 's_hutton'].to_numpy(),
        use_stcorr_h
    )

    df_valid['ml_db'] = np.nan
    df_valid.loc[calc_db_mask, 'ml_db'] = calc_dibona_vec(
        df_valid.loc[calc_db_mask, 'amp_calc'].to_numpy(),
        df_valid.loc[calc_db_mask, 'hypo_dist'].to_numpy(),
        df_valid.loc[calc_db_mask, 's_dibona'].to_numpy(),
        use_stcorr_d
    )

    df_valid['cha_prefix'] = df_valid['cha'].str[0:2]
    df_valid['components_key'] = df_valid['net'] + '_' + df_valid['sta'] + '_' + df_valid['loc'] + '_' + df_valid[
        'cha_prefix']
    df_valid['orientation'] = df_valid['cha'].str[2]

    return df_valid


def create_sets_vectorized(dfa_proc, mtd, mid, mad, dp, mty, stc, mmt, amt):
    """
    Usa Pandas per allineare canali N ed E, filtrare vettorialmente e calcolare la ML di stazione.
    """
    df_ne = dfa_proc[dfa_proc['orientation'].isin(['N', 'E'])].copy()

    if df_ne.empty:
        return [], 99999.0, []

    df_p = df_ne.pivot(index='components_key', columns='orientation')
    df_p = df_p.dropna(subset=[('amp_calc', 'N'), ('amp_calc', 'E')])

    if df_p.empty:
        return [], 99999.0, []

    mean_hypo = (df_p[('hypo_dist', 'N')] + df_p[('hypo_dist', 'E')]) / 2.0
    mean_epi = (df_p[('epi_dist_km', 'N')] + df_p[('epi_dist_km', 'E')]) / 2.0

    valid_mask = (mean_hypo >= mid) & (mean_hypo <= mad)

    if mtd == 'free':
        t_max_n = pd.to_numeric(df_p[('time_maxamp', 'N')], errors='coerce')
        t_min_n = pd.to_numeric(df_p[('time_minamp', 'N')], errors='coerce')
        t_max_e = pd.to_numeric(df_p[('time_maxamp', 'E')], errors='coerce')
        t_min_e = pd.to_numeric(df_p[('time_minamp', 'E')], errors='coerce')

        dp_n = (t_max_n - t_min_n).abs()
        dp_e = (t_max_e - t_min_e).abs()

        valid_mask &= (dp_n <= dp) & (dp_e <= dp)

    df_v = df_p[valid_mask].copy()
    if df_v.empty:
        return [], 99999.0, []

    mty_str = 'ml_hb' if mty == 0 else 'ml_db'
    s_str = 's_hutton' if mty == 0 else 's_dibona'
    calc_func = calc_huttonboore_vec if mty == 0 else calc_dibona_vec

    sta_ml = None

    if mmt == 'meanmag':
        sta_ml = (df_v[(mty_str, 'N')] + df_v[(mty_str, 'E')]) / 2.0
    elif mmt == 'meanamp':
        amp_n, amp_e = df_v[('amp_calc', 'N')], df_v[('amp_calc', 'E')]
        mean_amp = (amp_n + amp_e) / 2.0 if amt == 'ari' else np.sqrt(amp_n * amp_e)

        corr_n = pd.to_numeric(df_v[(s_str, 'N')], errors='coerce').fillna(0)
        corr_e = pd.to_numeric(df_v[(s_str, 'E')], errors='coerce').fillna(0)
        mean_corr = (corr_n + corr_e) / 2.0

        sta_ml = calc_func(mean_amp.to_numpy(), mean_hypo[valid_mask].to_numpy(), mean_corr.to_numpy(), stc)

    valid_keys = df_v.index.astype(str) + '_' + mtd
    ml_set = [[k, m] for k, m in zip(valid_keys, sta_ml) if pd.notna(m)]

    midi = mean_epi[valid_mask].min() if not mean_epi[valid_mask].empty else 99999.0
    logsms = []

    return ml_set, midi, logsms


def wstd(v, wm, wf):
    eps2 = 5.0e-6
    if len(v) <= 1:
        return False
    index_not_zero = np.where(wf > eps2)[0]
    wf_nz = wf[index_not_zero]
    v_nz = v[index_not_zero]
    wsd = np.sum(wf_nz * np.square((v_nz - wm)))
    fac = ((len(wf_nz) - 1) / len(wf_nz)) * np.sum(wf_nz)
    wsd = wsd / max(fac, eps2)
    return np.sqrt(max(wsd, eps2))


def whuber(v, w_mean, ruse, zero_val):
    res = abs(v - w_mean)
    with np.errstate(divide='ignore'):
        w = np.where(res <= ruse, 1.0, np.where(res > zero_val, 0.0, ruse / res))
    try:
        w_mean = np.sum(v * w) / np.sum(w)
    except ZeroDivisionError as err:
        sys.stderr.write(f"Error calculating the Huber Mean: {err}\n")
        w_mean = False

    w_std = wstd(v, w_mean, w) if w_mean else False
    w_std = 0.0 if not w_std else w_std
    return w_mean, w_std, w


def rm_outliers(v, v_w, v_flag, v_mean, v_std, times_std, co, skip):
    res = abs(v - v_mean)
    cut_limit = times_std * v_std if (times_std * v_std) > co else co
    not_outlier = res < cut_limit
    yes_outlier = res >= cut_limit
    skip.append(list(zip(v_flag[yes_outlier], v[yes_outlier])))
    v_w[yes_outlier] = 0
    v_w[not_outlier] = 1

    if len(v[not_outlier]) > 0:
        v_std = median_abs_deviation(v[not_outlier])
        v_mean = np.median(v[not_outlier])
    else:
        v_std = False
        v_mean = False
    return v_mean, v_std, v_w, skip


def calculate_event_ml(magnitudes, magnitudes_sta, it_max, var_stop, max_dev, out_cutoff, hm_cutoff_val):
    v = np.array(magnitudes)
    s = magnitudes_sta
    ruse = hm_cutoff_val
    xmd = np.median(v)
    xmd_std = median_abs_deviation(v)
    vlen_start = len(v)
    n = 1
    finished = False

    if hm_cutoff_val:
        ruse, zero_val = [hm_cutoff_val, 9999.0] if isinstance(hm_cutoff_val, float) else hm_cutoff_val
    else:
        zero_val = 9999.0

    weights = np.ones(len(v))
    removed = []
    whuber_fail = False
    typemean = 'unknown'
    whystop = 'unknown'

    while not finished:
        amd = xmd
        if hm_cutoff_val:
            whuber_fail = False
            try:
                xmd, xmd_std, weights = whuber(v, xmd, ruse, zero_val)
                typemean = 'whuber'
            except (ZeroDivisionError, ValueError) as err:
                logging.warning(f"Calcolo della media di Huber fallito. Fallback in rm_outliers. Errore: {err}")
                whuber_fail = True

        if not hm_cutoff_val or whuber_fail:
            xmd, xmd_std, weights, removed = rm_outliers(v, weights, s, xmd, xmd_std, max_dev, out_cutoff, removed)
            typemean = 'rowhfl' if whuber_fail else 'rmoutl'

        xmd_var = abs(amd - xmd)
        if xmd_var <= var_stop:
            finished = True
            whystop = f"{typemean}_varstop={xmd_var}_n={n}"
        elif n == it_max:
            finished = True
            whystop = f"{typemean}_maxit={n}_xmdvar={xmd_var}"
        n += 1

    vlen_stop = round(np.sum(weights), 2)
    return xmd, xmd_std, vlen_start, vlen_stop, whystop, removed, weights, whuber_fail


def char_quality(nu, d, m, r):
    q1type = 'course'
    q1, q2 = 'D', 'D'
    if q1type == 'basili':
        if (log2(nu * 2) - 2) >= m:
            q1 = 'A'
        elif (log2(nu * 2) - 1) >= m:
            q1 = 'B'
        elif log2(nu * 2) >= m:
            q1 = 'C'
        elif log2(nu * 2) < m:
            q1 = 'D'
        else:
            q1 = 'E'
    elif q1type == 'course':
        if d <= 50. and nu * 2 > (m * 10.0 + 1):
            q1 = 'A'
        elif d <= 100.0 and nu * 2 > (m * 7.5 + 1):
            q1 = 'B'
        elif d <= 150.0 and nu * 2 > (m * 5.0 + 1):
            q1 = 'C'
        else:
            q1 = 'D'

    if r <= 0.2:
        q2 = 'A'
    elif 0.2 < r <= 0.3:
        q2 = 'B'
    elif 0.3 < r <= 0.4:
        q2 = 'C'
    elif r > 0.4:
        q2 = 'D'
    return ''.join((q1, q2))


def standard_pyml_load(infile_val, conf_file_val, log_out_val):
    if os.path.exists(conf_file_val) and os.path.getsize(conf_file_val) > 0:
        paramfile = conf_file_val
    else:
        sys.stderr.write(f"Config file {conf_file_val} not existing or empty\n\n")
        sys.exit(2)

    confObj = cp.ConfigParser()
    confObj.read(paramfile)
    df_amp = pd.read_csv(infile_val, sep=';', index_col=False)
    df_amp.columns = df_amp.columns.str.lower()

    pre_params = get_config_dictionary(confObj, 'preconditions')
    theo_p_val = eval(pre_params['theoretical_p'])
    theo_s_val = eval(pre_params['theoretical_s'])

    sta_params = get_config_dictionary(confObj, 'station_magnitude')
    mag_mean_type_val = eval(sta_params['mag_mean_type'])
    amp_mean_type_val = eval(sta_params['amp_mean_type']) if mag_mean_type_val == 'meanamp' else None
    delta_peaks_val = float(sta_params['delta_peaks'])
    use_stcorr_hb_val = eval(sta_params['use_stcorr_hb'])
    use_stcorr_db_val = eval(sta_params['use_stcorr_db'])
    when_no_stcorr_hb_val = eval(sta_params['when_no_stcorr_hb'])
    when_no_stcorr_db_val = eval(sta_params['when_no_stcorr_db'])

    ev_params = get_config_dictionary(confObj, 'event_magnitude')
    mindist_val = float(ev_params['mindist'])
    maxdist_val = float(ev_params['maxdist'])
    hm_cutoff_val = eval(ev_params['hm_cutoff'])
    outliers_max_it_val = int(ev_params['outliers_max_it'])
    outliers_red_stop_val = float(ev_params['outliers_red_stop'])
    outliers_nstd_val = float(ev_params['outliers_nstd'])
    outliers_cutoff_val = float(ev_params['outliers_cutoff'])

    return df_amp, theo_p_val, theo_s_val, delta_peaks_val, use_stcorr_hb_val, use_stcorr_db_val, when_no_stcorr_hb_val, when_no_stcorr_db_val, mindist_val, maxdist_val, hm_cutoff_val, outliers_max_it_val, outliers_red_stop_val, outliers_nstd_val, outliers_cutoff_val, mag_mean_type_val, amp_mean_type_val


def json_pyml_load(json_input):
    cfg = json_input.get('data', {}).get('pyml_conf', False)
    orig = json_input.get('data', {}).get('origin', False)
    amps = json_input.get('data', {}).get('amplitudes', [])
    df_amp = pd.DataFrame(amps)
    return df_amp, cfg, orig


class DataEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


def json_response_structure():
    null = ''
    generic = {
        "type": null, "title": null, "status": null, "detail": null,
        "instance": null, "version": null, "request_submitted": null,
        "errors": {"inputfile": [], "arguments": []}
    }
    response = {
        "random_string": null, "hostname": {}, "magnitudes": {},
        "stationmagnitudes": [], "log": {}
    }
    log_dict = {"magnitude": {}, "stationmagnitudes": []}
    log_magnitude = {
        "hb": {"status": null, "summary": null, "extended": null},
        "db": {"status": null, "summary": null, "extended": null}
    }
    log_stationmagnitude = {
        "net": null, "sta": null, "loc": null, "band_inst": null,
        "status": null, "summary": null, "extended": null, "channels": []
    }
    log_stationmagnitude_channel = {
        "orientation": null, "status": null, "summary": null, "extended": null
    }
    magnitudes = {
        "hb": {}, "db": {}, "ampmethod": null, "magmethod": null, "loopexitcondition": null
    }
    emag = {
        "ml": null, "std": null, "totsta": null, "usedsta": null, "quality": '--'
    }
    stationmagnitude = {
        "net": null, "sta": null, "cha": null, "loc": null,
        "amp1": null, "time1": null, "amp2": null, "time2": null,
        "ep_distance_km": null, "ep_distance_delta": null,
        "orig_distance": null, "azimuth": null,
        "hb": {"ml": null, "w": null},
        "db": {"ml": null, "w": null}
    }
    return response, generic, log_dict, log_magnitude, log_stationmagnitude, log_stationmagnitude_channel, magnitudes, stationmagnitude, emag


def json_pyml_response(r):
    return json.dumps(r, cls=DataEncoder)


###############################################################
###### Main starts here ##########
pyml_version = "1.0.0"
args = parse_arguments()
geod = Geodesic.WGS84

# Inizializzazioni sicure per il Linter e per la robustezza
dfa = pd.DataFrame()
origin = False
config = None
eventid = 0
magnitudes_out = False
log_out = False
exit_condition = False
met = 'ingv'
use_stcorr_hb = None
use_stcorr_db = None
when_no_stcorr_hb = None
when_no_stcorr_db = None
mindist = None
maxdist = None
delta_peaks = None
mag_mean_type = None
amp_mean_type = None
outliers_max_it = None
outliers_red_stop = None
outliers_nstd = None
outliers_cutoff = None
hm_cutoff = None
theo_p = None
theo_s = None

jresponse, jbadrequest, jlog, jlog_mag, jlog_stamag, jlog_stamag_cha, jmagnitudes, jstationmagnitude, jemag = json_response_structure()
resp = copy.deepcopy(jresponse)
log = copy.deepcopy(jlog)
logm = copy.deepcopy(jlog_mag)
gresp = copy.deepcopy(jbadrequest)
resp['random_string'] = 'github/ingv/pyml'
resp['hostname'] = socket.gethostname()

if args.out_format.lower() == 'txt':
    magnitudes_out = sys.stdout
    log_out = sys.stderr
elif args.out_format.lower() == 'json':
    magnitudes_out = False
    log_out = False
else:
    log_out = False
    gresp['type'] = "https://tools.ietf.org/html/rfc4918#section-11.2"
    gresp['status'] = "422"
    gresp['errors']["arguments"].append(f"The given argument out format {args.out_format} is unknown")
    exit_condition = True

if args.in_file_format.lower() == 'csv':
    infile = args.in_file_name
    conf_file = args.conf
    eventid = args.eventid

    dfa, theo_p, theo_s, delta_peaks, use_stcorr_hb, use_stcorr_db, when_no_stcorr_hb, when_no_stcorr_db, mindist, maxdist, hm_cutoff, outliers_max_it, outliers_red_stop, outliers_nstd, outliers_cutoff, mag_mean_type, amp_mean_type = standard_pyml_load(
        infile, conf_file, log_out)
    origin = False
    if dfa.empty:
        gresp['status'] = "422"
        gresp['errors']["inputfile"].append("The given input file was incomplete")
        exit_condition = True

elif args.in_file_format.lower() == 'json':
    if os.path.exists(args.in_file_name):
        try:
            with open(args.in_file_name, 'r') as f:
                json_in = json.load(f)
            dfa, config, origin = json_pyml_load(json_in)
        except json.JSONDecodeError:
            dfa = pd.DataFrame()

        if dfa.empty or not config or not origin:
            gresp['status'] = "422"
            gresp['errors']["inputfile"].append("The given input file was incomplete")
            exit_condition = True
    else:
        gresp['status'] = "422"
        gresp['errors']["inputfile"].append("No input file found.")
        exit_condition = True

    if not exit_condition:
        try:
            theo_p = config['preconditions']['theoretical_p']
            theo_s = config['preconditions']['theoretical_s']
            mag_mean_type = config['station_magnitude']['mag_mean_type']
            amp_mean_type = config['station_magnitude'].get('amp_mean_type', None)
            delta_peaks = config['station_magnitude']['delta_peaks']
            use_stcorr_hb = config['station_magnitude']['use_stcorr_hb']
            use_stcorr_db = config['station_magnitude']['use_stcorr_db']
            when_no_stcorr_hb = config['station_magnitude']['when_no_stcorr_hb']
            when_no_stcorr_db = config['station_magnitude']['when_no_stcorr_db']
            mindist = config['event_magnitude']['mindist']
            maxdist = config['event_magnitude']['maxdist']
            hm_cutoff = config['event_magnitude']['hm_cutoff']
            outliers_max_it = config['event_magnitude']['outliers_max_it']
            outliers_red_stop = config['event_magnitude']['outliers_red_stop']
            outliers_nstd = config['event_magnitude']['outliers_nstd']
            outliers_cutoff = config['event_magnitude']['outliers_cutoff']
        except KeyError as k_err:
            gresp['status'] = "422"
            gresp['errors']["inputfile"].append(f"Missing config key: {k_err}")
            exit_condition = True
else:
    gresp['status'] = "422"
    gresp['errors']["arguments"].append("The given input format is unknown")
    sys.stdout.write(json_pyml_response(gresp))
    sys.exit()

if exit_condition:
    sys.stdout.write(json_pyml_response(gresp))
    sys.exit()

if args.dbona_corr:
    try:
        dbcorr = pd.read_csv(args.dbona_corr, sep=';')
    except Exception as e:
        if log_out: log_out.write(f"File {args.dbona_corr} not found: {e}\n")
        sys.exit()
else:
    dbcorr = False

unit = 1000 if args.in_file_format.lower() == 'csv' else 1

# --- LOGICA VETTORIALIZZATA ---
dfa_valid, met = process_amplitudes_vectorized(dfa, origin, unit)
dfa_processed = apply_magnitudes_and_group(dfa_valid, dbcorr, use_stcorr_hb, use_stcorr_db, when_no_stcorr_hb,
                                           when_no_stcorr_db)

mean_hb_ml_sta, _, _ = create_sets_vectorized(dfa_processed, met, mindist, maxdist, delta_peaks, 0, use_stcorr_hb,
                                              mag_mean_type, amp_mean_type)
mean_db_ml_sta, _, _ = create_sets_vectorized(dfa_processed, met, mindist, maxdist, delta_peaks, 1, use_stcorr_db,
                                              mag_mean_type, amp_mean_type)

# Gestione calcoli statistici per HB
ma_mlh, ma_stdh, ma_ns_s_h, ma_nsh = False, False, 0, 0
cond_hb = "Failed"
weights_hb, outliers_hb = [], []
mlhb = False

if len(mean_hb_ml_sta) > 0:
    mean_ml = [x[1] for x in mean_hb_ml_sta]
    mean_ml_sta = np.asarray([x[0] for x in mean_hb_ml_sta], dtype=object)
    ma_mlh, ma_stdh, ma_ns_s_h, ma_nsh, cond_hb, outliers_hb, weights_hb, wh_hb_fail = calculate_event_ml(mean_ml,
                                                                                                          mean_ml_sta,
                                                                                                          outliers_max_it,
                                                                                                          outliers_red_stop,
                                                                                                          outliers_nstd,
                                                                                                          outliers_cutoff,
                                                                                                          hm_cutoff)
    if ma_mlh:
        mlhb = True
        logm['hb']['status'] = 'warning' if wh_hb_fail else 'ok'
    else:
        logm['hb']['status'] = 'critical'

# Gestione calcoli statistici per DB
ma_mld, ma_stdd, ma_ns_s_d, ma_nsd = False, False, 0, 0
cond_db = "Failed"
weights_db, outliers_db = [], []
mldb = False

if len(mean_db_ml_sta) > 0:
    mean_ml = [x[1] for x in mean_db_ml_sta]
    mean_ml_sta = np.asarray([x[0] for x in mean_db_ml_sta], dtype=object)
    ma_mld, ma_stdd, ma_ns_s_d, ma_nsd, cond_db, outliers_db, weights_db, wh_db_fail = calculate_event_ml(mean_ml,
                                                                                                          mean_ml_sta,
                                                                                                          outliers_max_it,
                                                                                                          outliers_red_stop,
                                                                                                          outliers_nstd,
                                                                                                          outliers_cutoff,
                                                                                                          hm_cutoff)
    if ma_mld:
        mldb = True
        logm['db']['status'] = 'warning' if wh_db_fail else 'ok'
    else:
        logm['db']['status'] = 'critical'

resp["log"]['magnitude'] = logm

if not mlhb and not mldb:
    if log_out: log_out.write("Neither Hutton&Boore nor Di Bona ML could be calculated\n")
    sys.stdout.write(json_pyml_response(resp))
    sys.exit()

if magnitudes_out:
    amp_mean_type_str = amp_mean_type if mag_mean_type == 'meanamp' else 'False'
    magnitudes_out.write(
        "EventID;ML_HB;Std_HB;TOTSTA_HB;USEDSTA_HB;ML_DB;Std_DB;TOTSTA_DB;USEDSTA_DB;ampmethod;amp_mean_type;mag_mean_type;loopexitcondition\n")
    magnitudes_out.write(
        f"{eventid};{ma_mlh};{ma_stdh};{ma_ns_s_h};{ma_nsh};{ma_mld};{ma_stdd};{ma_ns_s_d};{ma_nsd};{met};{amp_mean_type_str};{mag_mean_type};{cond_hb}-{cond_db}\n")

# Aggiornamento JSON per l'intero evento
jmags = copy.deepcopy(jmagnitudes)
if ma_mlh:
    jhb = copy.deepcopy(jemag)
    jhb.update({'ml': ma_mlh, 'std': ma_stdh, 'totsta': ma_ns_s_h, 'usedsta': ma_nsh,
                'quality': char_quality(ma_ns_s_h, mindist, ma_mlh, ma_stdh)})
    jmags["hb"].update(jhb)
if ma_mld:
    jdb = copy.deepcopy(jemag)
    jdb.update({'ml': ma_mld, 'std': ma_stdd, 'totsta': ma_ns_s_d, 'usedsta': ma_nsd,
                'quality': char_quality(ma_ns_s_d, mindist, ma_mld, ma_stdd)})
    jmags["db"].update(jdb)
jmags.update({"ampmethod": met, "magmethod": mag_mean_type, "loopexitcondition": f"{cond_hb}-{cond_db}"})
resp["magnitudes"].update(jmags)

# --- COSTRUZIONE OUTPUT STAZIONI DA DATAFRAME AGGREGATO ---
df_stations = dfa_processed[dfa_processed['orientation'].isin(['N', 'E'])].pivot(index='components_key',
                                                                                 columns='orientation')
df_stations = df_stations.dropna(subset=[('amp_calc', 'N'), ('amp_calc', 'E')])

w_hb_dict = dict(zip([x[0] for x in mean_hb_ml_sta], weights_hb)) if mlhb else {}
w_db_dict = dict(zip([x[0] for x in mean_db_ml_sta], weights_db)) if mldb else {}

if not df_stations.empty:
    for comp_key, row in df_stations.iterrows():
        st_key = f"{comp_key}_{met}"
        # Casting a stringa forzato per evitare il warning di PyCharm su Hashable split
        n_net, n_sta, n_loc, n_cha = str(comp_key).split('_')

        whb = w_hb_dict.get(st_key, 0.0)
        wdb = w_db_dict.get(st_key, 0.0)

        for ori in ['N', 'E']:
            jstmag: dict = copy.deepcopy(jstationmagnitude)
            jstmag.update({
                "net": n_net, "sta": n_sta, "cha": f"{n_cha}{ori}",
                "loc": "--" if n_loc == 'None' else n_loc,
                "amp1": row.loc[('minamp', ori)], "time1": row.loc[('time_minamp', ori)],
                "amp2": row.loc[('maxamp', ori)], "time2": row.loc[('time_maxamp', ori)],
                "ep_distance_km": row.loc[('epi_dist_km', ori)], "orig_distance": row.loc[('hypo_dist', ori)],
                "lat": row.loc[('lat', ori)], "lon": row.loc[('lon', ori)], "elev": row.loc[('elev', ori)],
                "azimuth": row.loc[('epi_azimuth', ori)],
                "hb": {"ml": row.loc[('ml_hb', ori)] if not pd.isna(row.loc[('ml_hb', ori)]) else None,"w": float(whb)},
                "db": {"ml": row.loc[('ml_db', ori)] if not pd.isna(row.loc[('ml_db', ori)]) else None, "w": float(wdb)}
            })
            if 'period' in row.index.levels[0] and pd.notna(row[('period', ori)]):
                jstmag["period"] = row[('period', ori)]

            resp["stationmagnitudes"].append(jstmag)

            if magnitudes_out:
                magnitudes_out.write(
                    f"MLCHA {comp_key}_{ori} {jstmag['hb']['ml']} {whb} {comp_key}_{ori} {jstmag['db']['ml']} {wdb}\n")

if args.out_format == 'json':
    sys.stdout.write(json_pyml_response(resp))

if magnitudes_out:
    magnitudes_out.close()
if log_out:
    log_out.close()
sys.exit()