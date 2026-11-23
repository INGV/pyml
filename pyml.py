# Authors of the original INGV pyml: Raffaele Di Stefano, Barbara Castello.
# Acknowledgements: Valentino Lauciani, Andrea Bono.
# License: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/
"""Local magnitude calculator; JSON core with legacy CLI adapters.

Input amplitudes: millimetres (JSON), metres (legacy CSV).
Depth/distances: kilometres; elevation: metres; coordinates: WGS84 degrees.
Missing numeric values are NaN internally and null in JSON, never zero.
"""
from __future__ import annotations

import argparse
import ast
import configparser
import json
import math
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from geographiclib.geodesic import Geodesic
from scipy.stats import median_abs_deviation

# Hardcoded because the Docker image ships pyml.py without the VERSION file.
# On every release bump it together with VERSION and publiccode.yml.
VERSION = "1.7.0"
KEY = ["net", "sta", "loc", "band_inst", "method"]
LAWS = {"hb": ("HuttonBoore", 1.110, 0.00189),
        "db": ("DiBona", 1.667, 0.001736)}


def is_bool(value):
    return isinstance(value, (bool, np.bool_))


def numeric(series):
    """Reject boolean sentinels without rejecting the numbers 0 or 1."""
    clean = series.mask(series.map(is_bool))
    values = pd.to_numeric(clean, errors="coerce").astype("float64")
    return values.where(np.isfinite(values))


def finite_number(value, name):
    if is_bool(value) or value is None:
        raise ValueError(f"{name}: expected a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name}: expected a finite number")
    return result


@dataclass(frozen=True)
class Config:
    mag_mean_type: str = "meanamp"
    amp_mean_type: str = "geo"
    delta_peaks: float = 1.0
    use_stcorr_hb: bool = True
    use_stcorr_db: bool = True
    when_no_stcorr_hb: bool = True
    when_no_stcorr_db: bool = True
    mindist: float = 10.0
    maxdist: float = 600.0
    hm_cutoff: object = (0.3, 1.0)
    outliers_max_it: int = 10
    outliers_red_stop: float = 0.05
    outliers_nstd: float = 1.0
    outliers_cutoff: float = 0.1
    delta_corner: float = 5.0
    max_lowcorner: float = 15.0

    @classmethod
    def parse(cls, document):
        if not isinstance(document, dict):
            raise ValueError("pyml_conf must be an object")
        sections = []
        for name in ("preconditions", "station_magnitude", "event_magnitude"):
            section = document.get(name)
            if not isinstance(section, dict):
                raise ValueError(f"Missing or invalid configuration section: {name}")
            sections.append(section)
        values = {}
        for section in sections:
            values.update({k: v for k, v in section.items()
                           if k in cls.__dataclass_fields__})
        for name in ("use_stcorr_hb", "use_stcorr_db",
                     "when_no_stcorr_hb", "when_no_stcorr_db"):
            if name in values and not isinstance(values[name], bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("delta_peaks", "mindist", "maxdist", "outliers_red_stop",
                     "outliers_nstd", "outliers_cutoff", "delta_corner",
                     "max_lowcorner"):
            if name in values:
                values[name] = finite_number(values[name], name)
        if "outliers_max_it" in values:
            n = finite_number(values["outliers_max_it"], "outliers_max_it")
            if n < 1 or not n.is_integer():
                raise ValueError("outliers_max_it must be a positive integer")
            values["outliers_max_it"] = int(n)
        cutoff = values.get("hm_cutoff", (0.3, 1.0))
        if cutoff is False or cutoff is None:
            values["hm_cutoff"] = None
        else:
            parts = cutoff if isinstance(cutoff, (list, tuple)) else (cutoff, 9999.0)
            if len(parts) != 2:
                raise ValueError("hm_cutoff must be a number or [corner, limit]")
            corner, limit = (finite_number(v, "hm_cutoff") for v in parts)
            if not 0 < corner <= limit:
                raise ValueError("hm_cutoff requires 0 < corner <= limit")
            values["hm_cutoff"] = (corner, limit)
        cfg = cls(**values)
        if cfg.mag_mean_type not in ("meanamp", "meanmag"):
            raise ValueError("mag_mean_type must be meanamp or meanmag")
        if cfg.mag_mean_type == "meanamp" and cfg.amp_mean_type not in ("ari", "geo"):
            raise ValueError("amp_mean_type must be ari or geo")
        if not 0 <= cfg.mindist <= cfg.maxdist or cfg.maxdist <= 0:
            raise ValueError("Invalid distance interval")
        if min(cfg.delta_peaks, cfg.outliers_red_stop, cfg.outliers_nstd,
               cfg.outliers_cutoff, cfg.delta_corner, cfg.max_lowcorner) < 0:
            raise ValueError("Thresholds must be nonnegative")
        return cfg


def station_log(row, status, summary, extended="", channels=None):
    return {**{k: row.get(k, "") for k in KEY[:4]},
            "status": status, "summary": summary, "extended": extended,
            "channels": [] if channels is None else channels}


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dumps(value):
    return json.dumps(clean_json(value), allow_nan=False)


def channel_corrections(frame, corrections):
    """Resolve rules on unique station/channel pairs, never duplicate readings.

    Accepted selectors have exactly three positions (letters/digits/*).
    !(...) negates a selector. More literal positions take precedence.
    Equal-priority matches with different corrections are an input error.
    """
    empty = pd.Series(np.nan, index=frame.index, dtype=float)
    if corrections is None or len(corrections) == 0:
        return empty
    table = pd.DataFrame(corrections, dtype=object).copy()
    if not {"sta", "c"}.issubset(table.columns):
        raise ValueError("Corrections require sta and c")
    table["c"] = numeric(table["c"])
    if table["c"].isna().any():
        raise ValueError("Corrections contain missing/invalid c")
    if not table["sta"].map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("Corrections contain invalid station identifiers")
    table["ch"] = table.get("ch", pd.Series("***", index=table.index)).fillna("***")
    pairs = frame[["sta", "cha"]].drop_duplicates().reset_index(drop=True)
    pairs["pair"] = pairs.index
    candidates = pairs.merge(table[["sta", "c", "ch"]], on="sta", how="inner")
    selected = []
    for pattern, group in candidates.groupby("ch", sort=False):
        if not isinstance(pattern, str):
            raise ValueError("Invalid correction selector")
        negate = pattern.startswith("!(") and pattern.endswith(")")
        body = pattern[2:-1] if negate else pattern
        if len(body) != 3 or any(not (c.isalnum() or c == "*") for c in body):
            raise ValueError(f"Unsupported correction selector: {pattern}")
        mask = np.ones(len(group), dtype=bool)
        for position, char in enumerate(body):
            if char != "*":
                mask &= group["cha"].str[position].eq(char).to_numpy()
        if negate:
            mask = ~mask
        match = group.loc[mask].copy()
        match["priority"] = sum(c != "*" for c in body)
        selected.append(match)
    if not selected:
        return empty
    hits = pd.concat(selected, ignore_index=True)
    if hits.empty:
        return empty
    best = hits.groupby("pair")["priority"].transform("max")
    hits = hits[hits["priority"].eq(best)]
    if hits.groupby("pair")["c"].nunique().gt(1).any():
        raise ValueError("Ambiguous station/channel corrections at equal priority")
    values = hits.drop_duplicates("pair").set_index("pair")["c"]
    pairs["correction"] = pairs["pair"].map(values)
    result = frame[["sta", "cha"]].merge(
        pairs[["sta", "cha", "correction"]], on=["sta", "cha"],
        how="left", sort=False, validate="many_to_one")
    return pd.Series(result["correction"].to_numpy(dtype=float), index=frame.index)


def elapsed_seconds(first, second):
    a, b = numeric(first), numeric(second)
    seconds = (b - a).abs()
    # Numeric times use seconds; ISO timestamps use timedeltas in seconds.
    mask = a.isna() & b.isna()
    ta = pd.to_datetime(first.where(mask & first.map(lambda x: isinstance(x, str))),
                        errors="coerce", utc=True, format="mixed")
    tb = pd.to_datetime(second.where(mask & second.map(lambda x: isinstance(x, str))),
                        errors="coerce", utc=True, format="mixed")
    return seconds.where(~mask, (tb - ta).dt.total_seconds().abs())


def prepare(amplitudes, origin, cfg, corrections, clips, unit, logs):
    if not isinstance(amplitudes, list) or not all(isinstance(a, dict) for a in amplitudes):
        raise ValueError("amplitudes must be an array of objects")
    raw = pd.DataFrame(amplitudes, dtype=object)
    f = pd.DataFrame(index=raw.index)

    def column(name, alternate=None):
        value = raw.get(name, pd.Series(None, index=raw.index, dtype=object))
        if alternate in raw:
            value = value.where(value.notna(), raw[alternate])
        return value

    for name in ("net", "sta", "cha"):
        f[name] = column(name)
    f["loc"] = column("loc").map(
        lambda v: "--" if v is None or v is False or v == "" or pd.isna(v) else v)
    f["method"] = column("method").map(
        lambda v: "ingv" if v is None or v is False or v == "" or pd.isna(v) else v)
    valid = pd.Series(True, index=f.index)
    for name in ("net", "sta", "loc", "method"):
        valid &= f[name].map(lambda v: isinstance(v, str) and bool(v) and "_" not in v)
    valid &= f["cha"].map(lambda v: isinstance(v, str) and len(v) == 3)
    f["band_inst"] = f["cha"].where(valid, "").str[:2]
    for row in f.loc[~valid].to_dict("records"):
        logs.append(station_log(row, "critical", "Invalid channel identifier"))
    for name, alternate in (("amp1", "minamp(m)"), ("amp2", "maxamp(m)"),
                            ("lat", None), ("lon", None), ("elev", None),
                            ("period", None), ("loco", None), ("hico", None),
                            ("epi", "epidistance(km)"), ("hypo", "ipodistance(km)")):
        f[name] = numeric(column(name, alternate))
    f[["amp1", "amp2"]] *= unit
    for name, alternate in (("time1", "minamptime"), ("time2", "maxamptime")):
        times = column(name, alternate)
        f[name] = times.mask(times.map(is_bool), None)
    f = f.loc[valid].copy()
    f["orientation"] = f["cha"].str[2]
    f["reason"] = ""
    if clips is not None and len(clips):
        clip = pd.DataFrame(clips)
        fields = ["net", "sta", "cha"]
        if not set(fields).issubset(clip.columns):
            raise ValueError("Clipping information requires net, sta, cha")
        if "loc" in clip:
            clip["loc"] = clip["loc"].fillna("--").replace("", "--")
            fields.append("loc")
        excluded = pd.MultiIndex.from_frame(f[fields]).isin(
            pd.MultiIndex.from_frame(clip[fields].drop_duplicates()))
        f.loc[excluded, "reason"] = "Component skipped due to clipping"
    bad_band = f["loco"].notna() & f["hico"].notna() & (
        f["loco"].ge(cfg.max_lowcorner) | (f["hico"] - f["loco"]).lt(cfg.delta_corner))
    f.loc[bad_band & f["reason"].eq(""), "reason"] = "Component skipped due to filter corners"
    for row in f.loc[f["reason"].ne("")].to_dict("records"):
        logs.append(station_log(row, "warning", row["reason"],
                                f"Channel {row['cha']}"))
    f = f.loc[f["reason"].eq("")].copy()
    duplicate = f.duplicated(KEY + ["orientation"], keep="last")
    for row in f.loc[duplicate].to_dict("records"):
        logs.append(station_log(row, "warning", "Duplicate channel: last accepted reading used",
                                row["cha"]))
    f = f.loc[~duplicate].reset_index(drop=True)
    if f["method"].nunique() > 1:
        raise ValueError("Mixed amplitude methods require separate requests")
    f["amp"] = (f["amp2"] / 2 - f["amp1"] / 2).abs()
    f["amp"] = f["amp"].where(np.isfinite(f["amp"]) & f["amp"].gt(0))
    f["azimuth"], f["delta"] = np.nan, np.nan
    coords = f["lat"].between(-90, 90) & f["lon"].between(-180, 180)
    f.loc[~coords, ["lat", "lon"]] = np.nan
    precomputed = f["epi"].notna() | f["hypo"].notna()
    valid_pre = f["epi"].ge(0) & f["hypo"].gt(0)
    f.loc[precomputed & ~valid_pre, ["epi", "hypo"]] = np.nan
    need = ~precomputed & coords & f["elev"].notna()
    if need.any():
        if not isinstance(origin, dict):
            raise ValueError("Missing origin for distance calculation")
        lat = finite_number(origin.get("lat"), "origin.lat")
        lon = finite_number(origin.get("lon"), "origin.lon")
        depth = finite_number(origin.get("depth"), "origin.depth")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Origin coordinates out of range")
        unique = f.loc[need, ["lat", "lon"]].drop_duplicates().copy()
        # One inverse per coordinate pair, shared by N/E and channel families.
        distances = [Geodesic.WGS84.Inverse(a, b, lat, lon)
                     for a, b in unique.itertuples(index=False, name=None)]
        unique["epi"] = [d["s12"] / 1000 for d in distances]
        unique["azimuth"] = [d["azi1"] % 360 for d in distances]
        unique["delta"] = [d["a12"] for d in distances]
        mapped = f.loc[need, ["lat", "lon"]].merge(
            unique, on=["lat", "lon"], how="left", sort=False, validate="many_to_one")
        f.loc[need, ["epi", "azimuth", "delta"]] = mapped[["epi", "azimuth", "delta"]].to_numpy()
        f.loc[need, "hypo"] = np.hypot(f.loc[need, "epi"],
                                      depth + f.loc[need, "elev"] / 1000)
    f["seconds"] = elapsed_seconds(f["time1"], f["time2"])
    f["corr_db"] = channel_corrections(f, corrections)
    f["corr_hb"] = np.nan
    for law in LAWS:
        use = getattr(cfg, "use_stcorr_" + law)
        allow = getattr(cfg, "when_no_stcorr_" + law)
        ok = ~pd.Series(use, index=f.index) | f["corr_" + law].notna() | allow
        correction = f["corr_" + law].fillna(0).to_numpy() if use else 0.0
        f["ml_" + law] = np.where(ok, magnitude(f["amp"], f["hypo"], correction, law), np.nan)
    return f


def magnitude(amplitude, distance, correction, law):
    _, alpha, beta = LAWS[law]
    a, d = np.asarray(amplitude, dtype=float), np.asarray(distance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.log10(a) + alpha * np.log10(d / 100) + beta * (d - 100) + 3 + correction
    return np.where((a > 0) & (d > 0) & np.isfinite(result), result, np.nan)


def quality(count, distance, ml, std):
    first = "D"
    for limit, factor, letter in ((50, 10, "A"), (100, 7.5, "B"), (150, 5, "C")):
        if distance <= limit and count * 2 > ml * factor + 1:
            first = letter
            break
    second = "A" if std <= .2 else "B" if std <= .3 else "C" if std <= .4 else "D"
    return first + second


def estimate(values, cfg):
    """Historical Huber/MAD approach, with finite and zero-weight checks."""
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.isfinite(v).all():
        raise ValueError("Estimator requires a nonempty finite vector")
    center = float(np.median(v))
    spread = float(median_abs_deviation(v))
    weights = np.ones(v.size)
    removed, fallback = [], False
    for iteration in range(1, cfg.outliers_max_it + 1):
        previous = center
        residual = np.abs(v - center)
        mode = "rmoutl"
        use_huber = cfg.hm_cutoff is not None
        if use_huber:
            corner, limit = cfg.hm_cutoff
            weights = np.ones(v.size)
            np.divide(corner, residual, out=weights, where=residual > corner)
            weights[residual > limit] = 0
            total = weights.sum()
            if total > 0 and np.isfinite(total):
                center = float(np.dot(v, weights) / total)
                active = weights > 5e-6
                count = int(active.sum())
                if v.size == 1:
                    spread = 0.0
                elif count:
                    denominator = (count - 1) / count * weights[active].sum()
                    variance = np.sum(weights[active] * (v[active] - center) ** 2)
                    spread = math.sqrt(max(variance / max(denominator, 5e-6), 5e-6))
                else:
                    use_huber = False
                mode = "whuber"
            else:
                use_huber = False
            if not use_huber:
                fallback = True
                center, spread = previous, float(median_abs_deviation(v))
        if not use_huber:
            cutoff = max(cfg.outliers_nstd * spread, cfg.outliers_cutoff)
            active = residual <= 0 if cutoff == 0 else residual < cutoff
            weights = active.astype(float)
            removed.append(np.flatnonzero(~active))
            if not active.any():
                return None, weights, "Failed", removed, fallback
            center = float(np.median(v[active]))
            spread = float(median_abs_deviation(v[active]))
            mode = "rowhfl" if fallback else "rmoutl"
        change = abs(previous - center)
        if change <= cfg.outliers_red_stop:
            reason = f"{mode}_varstop={change}_n={iteration}"
            break
        reason = f"{mode}_maxit={iteration}_xmdvar={change}"
    return (center, spread), weights, reason, removed, fallback


def calculate(request, corrections=None, *, clips=None, unit=1.0):
    """Return (response, legacy text details), without file access or exits."""
    if not isinstance(request, dict) or not isinstance(request.get("data"), dict):
        raise ValueError("Request requires a data object")
    data = request["data"]
    cfg = Config.parse(data.get("pyml_conf"))
    logs = []
    f = prepare(data.get("amplitudes", []), data.get("origin"), cfg,
                corrections, clips, unit, logs)
    response = {"random_string": "github/ingv/pyml", "hostname": socket.gethostname(),
                "magnitudes": {}, "stationmagnitudes": [],
                "log": {"magnitude": {}, "stationmagnitudes": logs}}
    if f.empty:
        keys = pd.MultiIndex.from_arrays([[] for _ in KEY], names=KEY)
    else:
        keys = pd.MultiIndex.from_frame(f[KEY].drop_duplicates())
    groups = keys.to_frame(index=False)
    components = {}
    for ori in ("N", "E"):
        components[ori] = f.loc[f["orientation"].eq(ori)].set_index(KEY).reindex(keys)
    n, e = components["N"], components["E"]
    have_n, have_e = n["amp"].notna(), e["amp"].notna()
    both = have_n & have_e
    hypo, epi = (n["hypo"] + e["hypo"]) / 2, (n["epi"] + e["epi"]) / 2
    coords_ok = hypo.notna() & epi.notna()
    distance_ok = hypo.between(cfg.mindist, cfg.maxdist)
    free = pd.Series(keys.get_level_values("method") == "free", index=keys)
    time_ok = ~free | (n["seconds"].le(cfg.delta_peaks) & e["seconds"].le(cfg.delta_peaks))
    groups["label"] = (groups[KEY].agg("_".join, axis=1) if len(groups)
                       else pd.Series(dtype=str))
    stats, weights_by_law, eligibility_by_law, reasons, outliers = {}, {}, {}, {}, {}
    for law, (title, _, _) in LAWS.items():
        use = getattr(cfg, "use_stcorr_" + law)
        allow = getattr(cfg, "when_no_stcorr_" + law)
        cn, ce = n["corr_" + law], e["corr_" + law]
        corr_ok = (~pd.Series(use, index=keys)) | (cn.notna() & ce.notna()) | allow
        if cfg.mag_mean_type == "meanmag":
            station_ml = (n["ml_" + law] + e["ml_" + law]) / 2
        else:
            if cfg.amp_mean_type == "ari":
                amp = n["amp"] / 2 + e["amp"] / 2
            else:
                amp = np.sqrt(n["amp"]) * np.sqrt(e["amp"])
            corr = (cn.fillna(0) + ce.fillna(0)) / 2 if use else 0.0
            station_ml = pd.Series(magnitude(amp, hypo, corr, law), index=keys)
        eligible = both & coords_ok & distance_ok & time_ok & corr_ok & station_ml.notna()
        reasons[law] = "Failed"
        full_weights = np.zeros(len(keys))
        maglog = {"status": "critical", "summary": f"{title} List is empty",
                  "extended": "All the stations missing due to only one channel is present, or out of minmax distance"}
        for i in np.flatnonzero(~eligible.to_numpy()):
            row = groups.iloc[i].to_dict()
            channels = []
            status = "warning"
            if not both.iloc[i]:
                status, summary, extended = "critical", "Station skipped due to missing channel", "ML " + title
                for ori, present, comp in (("N", have_n.iloc[i], n), ("E", have_e.iloc[i], e)):
                    detail = ""
                    if not present and pd.notna(comp["cha"].iloc[i]):
                        detail = "minamp missing" if pd.isna(comp["amp1"].iloc[i]) else (
                            "maxamp missing" if pd.isna(comp["amp2"].iloc[i]) else "invalid amplitude")
                    channels.append({"orientation": ori, "status": "ok" if present else "critical",
                                     "summary": "" if present else "missing", "extended": detail})
            elif not coords_ok.iloc[i]:
                status, summary, extended = "critical", "Station skipped due to stations coordinates missing", title
            elif not distance_ok.iloc[i]:
                summary = f"In ML  {title} Station {row['label']} skipped due to ipodist"
                extended = f"Distance is {hypo.iloc[i]} km"
            elif not time_ok.iloc[i]:
                summary = f"In ML  {title} Station skipped due to time distance between min and max amp larger than {cfg.delta_peaks}"
                seconds = n["seconds"].iloc[i]
                extended = (f"Time distance is {seconds} s" if pd.notna(seconds)
                            else "Missing or invalid peak times")
            else:
                summary, extended = "Station skipped due to missing correction or invalid magnitude", title
            logs.append(station_log(row, status, summary, extended, channels))
        selected = np.flatnonzero(eligible.to_numpy())
        outliers[law] = []
        if selected.size:
            result, w, reason, removed, fallback = estimate(station_ml.iloc[selected], cfg)
            full_weights[selected] = w
            reasons[law] = reason
            if removed:
                outliers[law] = [(groups["label"].iloc[selected[j]], float(station_ml.iloc[selected[j]]))
                                 for j in removed[0]]
            if result is not None:
                ml, std = result
                used = selected[w > 5e-6]
                dmin = float(epi.iloc[used].min()) if used.size else math.inf
                stats[law] = {"ml": ml, "std": std, "totsta": int(selected.size),
                              "usedsta": round(float(w.sum()), 2),
                              "quality": quality(int(selected.size), dmin, ml, std)}
                maglog = {"status": "warning" if fallback else "ok",
                          "summary": f"{title}: Weighted Huber Mean failed, Outliers Removal used instead" if fallback else "",
                          "extended": ""}
            else:
                maglog = {"status": "critical", "summary": f"{title}: magnitude estimation failed",
                          "extended": "No usable weighted observations"}
        weights_by_law[law] = full_weights
        eligibility_by_law[law] = eligible.to_numpy()
        response["log"]["magnitude"][law] = maglog
    if stats:
        methods = f["method"].drop_duplicates().tolist()
        response["magnitudes"] = {"hb": stats.get("hb", {}), "db": stats.get("db", {}),
                                 "ampmethod": methods[0] if len(methods) == 1 else "mixed",
                                 "magmethod": cfg.mag_mean_type,
                                 "loopexitcondition": reasons["hb"] + "-" + reasons["db"]}
        # Incomplete N/E pairs belong only in the station logs.
        out = f.loc[f["orientation"].isin(["N", "E"])]
        positions = keys.get_indexer(pd.MultiIndex.from_frame(out[KEY]))
        paired = both.to_numpy()[positions]
        out = out.loc[paired].copy()
        positions = positions[paired]
        for law in LAWS:
            out["w_" + law] = weights_by_law[law][positions]
            out["eligible_" + law] = eligibility_by_law[law][positions]
        out["order"] = positions * 2 + out["orientation"].eq("E").astype(int)
        out = out.sort_values("order", kind="stable")
        outliers["_channels"] = []
        for row in out.to_dict("records"):
            entry = {k: row[k] for k in ("net", "sta", "loc", "cha", "amp1", "amp2",
                                         "time1", "time2", "lat", "lon", "elev", "azimuth")}
            entry.update(ep_distance_km=row["epi"], ep_distance_delta=row["delta"],
                         orig_distance=row["hypo"])
            for law in LAWS:
                entry[law] = ({"ml": row["ml_" + law], "w": row["w_" + law]}
                              if row["eligible_" + law] and law in stats else {})
            if pd.notna(row["period"]):
                entry["period"] = row["period"]
            outliers["_channels"].append(entry)
            if all(pd.notna(row[k]) for k in ("lat", "lon", "elev")):
                response["stationmagnitudes"].append(entry)
    # Private details are returned separately to the legacy text adapter.
    return clean_json(response), outliers


def analyze(request, corrections=None, *, clips=None):
    """Public library API: request dict -> response dict."""
    return calculate(request, corrections, clips=clips)[0]


def read_table(path):
    if path is None:
        return None
    if Path(path).suffix.lower() == ".json":
        with open(path, encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, list):
            raise ValueError("Table JSON must be an array of records")
        return pd.DataFrame(result, dtype=object)
    return pd.read_csv(path, sep=";", dtype=object, keep_default_na=False, index_col=False)


def load_request(args):
    if args.in_file_format == "json":
        if args.in_file_name == "-":
            return json.load(sys.stdin), 1.0
        with open(args.in_file_name, encoding="utf-8") as stream:
            return json.load(stream), 1.0
    if not args.conf:
        raise ValueError("--conf is required for CSV input")
    cfg = configparser.ConfigParser()
    if not cfg.read(args.conf, encoding="utf-8"):
        raise ValueError("Configuration file not found")
    settings = {}
    for section in ("preconditions", "station_magnitude", "event_magnitude"):
        settings[section] = {}
        for name, value in cfg.items(section):
            try:
                settings[section][name] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                settings[section][name] = value
    table = pd.read_csv(args.in_file_name, sep=";", dtype=object, keep_default_na=False, index_col=False)
    table.columns = table.columns.str.lower()
    return {"data": {"pyml_conf": settings, "amplitudes": table.to_dict("records")}}, 1000.0


def render_text(response, outliers, cfg, eventid, stdout, stderr):
    mags = response["magnitudes"]
    if mags:
        fields = ["EventID", "ML_HB", "Std_HB", "TOTSTA_HB", "USEDSTA_HB",
                  "ML_DB", "Std_DB", "TOTSTA_DB", "USEDSTA_DB",
                  "ampmethod", "amp_mean_type", "mag_mean_type", "loopexitcondition"]
        print(";".join(fields), file=stdout)
        row = [eventid]
        for law in LAWS:
            value = mags[law]
            row.extend(value.get(k, False) for k in ("ml", "std", "totsta", "usedsta"))
        row.extend([mags["ampmethod"], cfg.amp_mean_type if cfg.mag_mean_type == "meanamp" else "False",
                    cfg.mag_mean_type, mags["loopexitcondition"]])
        print(";".join(map(str, row)) + ";", file=stdout)
        for channel in outliers.get("_channels", response["stationmagnitudes"]):
            if not channel["hb"] and not channel["db"]:
                continue
            key = "_".join(channel[k] for k in ("net", "sta", "loc", "cha"))
            values = [channel[law].get(k, False) for law in LAWS for k in ("ml", "w")]
            print("MLCHA", key, values[0], values[1], key, values[2], values[3], file=stdout)
        for law in LAWS:
            for key, ml in outliers[law]:
                print("OUTL_" + law.upper(), key, ml, file=stdout)
    else:
        print(dumps(response), file=stdout)
    for item in response["log"]["stationmagnitudes"]:
        print(f"{item['status']}:{item['summary']} --> {item['extended']}", file=stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_file_name", required=True)
    parser.add_argument("--in_file_format", type=str.lower, choices=("json", "csv"), default="json")
    parser.add_argument("--out_format", type=str.lower, choices=("json", "txt"), default="json")
    parser.add_argument("--conf")
    parser.add_argument("--eventid", default="0")
    parser.add_argument("--dbona_corr", default="dbcor.csv")
    parser.add_argument("--clipped_info")
    args = parser.parse_args(argv)
    try:
        request, unit = load_request(args)
        corrections = read_table(args.dbona_corr) if args.dbona_corr else None
        clips = read_table(args.clipped_info)
        response, outliers = calculate(request, corrections, clips=clips, unit=unit)
        if args.out_format == "json":
            print(dumps(response))
        else:
            render_text(response, outliers, Config.parse(request["data"]["pyml_conf"]),
                        args.eventid, sys.stdout, sys.stderr)
        return 0
    except (OSError, ValueError, TypeError, KeyError, configparser.Error) as exc:
        problem = {"type": "https://tools.ietf.org/html/rfc4918#section-11.2",
                   "title": "Unprocessable entity", "status": "422", "detail": str(exc),
                   "instance": socket.gethostname(), "version": VERSION,
                   "request_submitted": datetime.now(timezone.utc).isoformat(),
                   "errors": {"inputfile": [str(exc)], "arguments": []}}
        print(dumps(problem), file=sys.stdout if args.out_format == "json" else sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
