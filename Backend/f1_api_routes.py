"""
f1_api_routes.py  –  F1 Track Flask App
=========================================
Data sources:
  • FastF1                — telemetry, laps, results (primary)
  • Jolpica (Ergast clone)— silent fallback for schedule/circuits if FastF1 fails
  • OpenF1 (api.openf1.org)— LIVE data during active sessions

Endpoints:
  Core:
    GET /f1/seasons
    GET /f1/schedule/<year>
    GET /f1/race-results/<year>/<round>
    GET /f1/standings/drivers/<year>
    GET /f1/standings/constructors/<year>
    GET /f1/lap-times/<year>/<round>/<driver_code>
    GET /f1/circuits/<year>

  Replay (FastF1):
    GET /f1/replay/session-info/<year>/<round>
    GET /f1/replay/track-map/<year>/<round>
    GET /f1/replay/mini-sectors/<year>/<round>?n=25
    GET /f1/replay/telemetry/<year>/<round>/<driver_code>
    GET /f1/replay/sector-times/<year>/<round>
    GET /f1/replay/long-stints/<year>/<round>
    GET /f1/replay/speed-trap/<year>/<round>

  Live (OpenF1):
    GET /f1/live/current-session
    GET /f1/live/drivers/<session_key>
    GET /f1/live/positions/<session_key>
    GET /f1/live/car-data/<session_key>/<driver_number>
    GET /f1/live/intervals/<session_key>

Install:  pip install fastf1 requests numpy pandas
"""

import os
import time
import datetime
import threading
import logging

import requests
import fastf1
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

# ─── FastF1 cache ─────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "f1cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
logging.getLogger("fastf1").setLevel(logging.WARNING)

f1_bp = Blueprint("f1", __name__, url_prefix="/f1")

# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_YEAR        = datetime.date.today().year   # 2026
EARLIEST_YEAR       = 2018
STANDINGS_TTL_SEC   = 60 * 60
REPLAY_TTL_SEC      = 30 * 60
LIVE_TTL_SEC        = 4

JOLPICA = "https://api.jolpi.ca/ergast/f1"
OPENF1  = "https://api.openf1.org/v1"
HEADERS = {"User-Agent": "F1Track/1.0", "Accept": "application/json"}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _err(msg, code=500):
    return jsonify({"error": msg}), code


def _to_int(v):
    try:
        if pd.isna(v): return None
        return int(v)
    except Exception:
        return None


def _to_float(v, default=0.0):
    try:
        if pd.isna(v): return default
        return float(v)
    except Exception:
        return default


def _to_str(v, default=""):
    try:
        if pd.isna(v): return default
    except Exception:
        pass
    return str(v) if v is not None else default


def _normalize_team_color(c):
    s = _to_str(c).strip().lstrip("#")
    if not s or s.lower() == "nan" or len(s) not in (3, 6):
        return "#888888"
    return "#" + s


def _jolpica(path, timeout=10):
    try:
        r = requests.get(f"{JOLPICA}{path}", headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Jolpica] {path} → {e}")
        return None


def _openf1(endpoint, params=None, timeout=8):
    try:
        r = requests.get(f"{OPENF1}/{endpoint}", params=params or {},
                         headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[OpenF1] {endpoint} → {e}")
        return None


_cache_lock = threading.Lock()
_cache = {}

def _cached(key, ttl, compute):
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and (now - hit[0] < ttl):
            return hit[1]
    val = compute()
    with _cache_lock:
        _cache[key] = (now, val)
    return val


# ════════════════════════════════════════════════════════════════════════════
#   CORE ROUTES
# ════════════════════════════════════════════════════════════════════════════
@f1_bp.route("/seasons")
def get_seasons():
    return jsonify({
        "seasons": list(range(DEFAULT_YEAR, EARLIEST_YEAR - 1, -1)),
        "default": DEFAULT_YEAR,
    })


# ── Schedule (FastF1 → Jolpica fallback) ─────────────────────────────────────
def _schedule_fastf1(year):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if schedule is None or schedule.empty: return None
    schedule = schedule.sort_values("RoundNumber").reset_index(drop=True)
    today = datetime.date.today()
    next_found, races = False, []
    for _, ev in schedule.iterrows():
        rnd = _to_int(ev.get("RoundNumber"))
        if rnd is None or rnd <= 0: continue
        try:    ev_date = pd.Timestamp(ev.get("EventDate")).date()
        except: ev_date = None
        if ev_date is None:
            status = "future"
        elif ev_date < today:
            status = "done"
        elif not next_found:
            status, next_found = "next", True
        else:
            status = "future"
        ev_format = _to_str(ev.get("EventFormat"), "conventional").lower()
        races.append({
            "round":    rnd,
            "name":     _to_str(ev.get("EventName")),
            "country":  _to_str(ev.get("Country")),
            "location": _to_str(ev.get("Location")),
            "circuit":  _to_str(ev.get("OfficialEventName")) or _to_str(ev.get("EventName")),
            "date":     ev_date.isoformat() if ev_date else "",
            "time":     "",
            "format":   "sprint" if "sprint" in ev_format else "conventional",
            "status":   status,
        })
    return races


def _schedule_jolpica(year):
    data = _jolpica(f"/{year}/races.json?limit=100")
    if not data: return None
    raw = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not raw: return None
    today = datetime.date.today()
    next_found, races = False, []
    for r in raw:
        try:    d = datetime.date.fromisoformat(r["date"])
        except: d = None
        if d is None: status = "future"
        elif d < today: status = "done"
        elif not next_found: status, next_found = "next", True
        else: status = "future"
        circuit = r.get("Circuit", {})
        loc = circuit.get("Location", {})
        races.append({
            "round":    int(r["round"]),
            "name":     r.get("raceName", ""),
            "country":  loc.get("country", ""),
            "location": loc.get("locality", ""),
            "circuit":  circuit.get("circuitName", ""),
            "date":     r.get("date", ""),
            "time":     r.get("time", ""),
            "format":   "sprint" if r.get("Sprint") else "conventional",
            "status":   status,
        })
    return races


@f1_bp.route("/schedule/<int:year>")
def get_schedule(year):
    races, ff1_err = None, None
    try:
        races = _schedule_fastf1(year)
    except Exception as e:
        ff1_err = str(e)
        print(f"[schedule] FastF1 failed for {year}: {e}")
    if not races:
        races = _schedule_jolpica(year)
    if not races:
        msg = f"Could not load schedule for {year}"
        if ff1_err: msg += f" (FastF1: {ff1_err})"
        return _err(msg)
    return jsonify({"year": year, "races": races})


# ── Race Results (FastF1) ────────────────────────────────────────────────────
@f1_bp.route("/race-results/<int:year>/<int:round_num>")
def get_race_results(year, round_num):
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=False, weather=False, messages=False, laps=False)
    except Exception as e:
        return _err(f"Race results {year}/{round_num}: {e}")

    results = []
    for _, d in session.results.iterrows():
        results.append({
            "position":    _to_int(d.get("Position")),
            "driver_code": _to_str(d.get("Abbreviation")),
            "full_name":   _to_str(d.get("FullName")),
            "team":        _to_str(d.get("TeamName")),
            "grid_start":  _to_int(d.get("GridPosition")),
            "points":      _to_float(d.get("Points")),
            "status":      _to_str(d.get("Status")),
            "fastest_lap": bool(d.get("FastestLap", False)) if "FastestLap" in d else False,
        })

    try:    race_date = str(pd.Timestamp(session.event["EventDate"]).date())
    except: race_date = ""

    return jsonify({
        "year": year, "round": round_num,
        "race_name": _to_str(session.event.get("EventName")),
        "circuit":   _to_str(session.event.get("Location")),
        "date":      race_date,
        "results":   results,
    })


# ── Standings (computed) ─────────────────────────────────────────────────────
def _compute_standings(year):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule.sort_values("RoundNumber").reset_index(drop=True)
    today = pd.Timestamp(datetime.date.today())
    drivers, teams, last_round = {}, {}, 0

    for _, ev in schedule.iterrows():
        rnd = _to_int(ev.get("RoundNumber"))
        if rnd is None or rnd <= 0: continue
        try:    ev_date = pd.Timestamp(ev.get("EventDate"))
        except: continue
        if ev_date >= today: break
        ev_format = _to_str(ev.get("EventFormat"), "conventional").lower()
        sessions_to_load = ["R"] + (["S"] if "sprint" in ev_format else [])

        round_had_data = False
        for sess_id in sessions_to_load:
            try:
                sess = fastf1.get_session(year, rnd, sess_id)
                sess.load(telemetry=False, weather=False, messages=False, laps=False)
            except Exception as e:
                print(f"[standings] skip {year} R{rnd} {sess_id}: {e}")
                continue
            res = sess.results
            if res is None or res.empty: continue
            round_had_data = True

            for _, row in res.iterrows():
                code = _to_str(row.get("Abbreviation"))
                if not code: continue
                pts  = _to_float(row.get("Points"))
                pos  = _to_int(row.get("Position"))
                team = _to_str(row.get("TeamName"))
                full_name   = _to_str(row.get("FullName"))
                nationality = _to_str(row.get("CountryCode"))

                d = drivers.setdefault(code, {
                    "code": code, "driver": full_name,
                    "nationality": nationality, "team": team,
                    "points": 0.0, "wins": 0,
                })
                d["points"] += pts
                if team:        d["team"] = team
                if full_name:   d["driver"] = full_name
                if nationality: d["nationality"] = nationality
                if sess_id == "R" and pos == 1: d["wins"] += 1

                if team:
                    t = teams.setdefault(team, {
                        "team": team, "nationality": "",
                        "points": 0.0, "wins": 0,
                    })
                    t["points"] += pts
                    if sess_id == "R" and pos == 1: t["wins"] += 1

        if round_had_data: last_round = rnd

    driver_list = sorted(drivers.values(), key=lambda x: (-x["points"], -x["wins"], x["driver"]))
    for i, d in enumerate(driver_list, 1):
        d["position"], d["points"] = i, round(d["points"], 2)

    team_list = sorted(teams.values(), key=lambda x: (-x["points"], -x["wins"], x["team"]))
    for i, t in enumerate(team_list, 1):
        t["position"], t["points"] = i, round(t["points"], 2)

    return {"round": str(last_round), "drivers": driver_list, "constructors": team_list}


def _get_standings_cached(year):
    return _cached(("standings", year), STANDINGS_TTL_SEC, lambda: _compute_standings(year))


@f1_bp.route("/standings/drivers/<int:year>")
def get_driver_standings(year):
    try:    s = _get_standings_cached(year)
    except Exception as e: return _err(f"Could not compute driver standings for {year}: {e}")
    return jsonify({"year": year, "round": s["round"], "driver_standings": s["drivers"]})


@f1_bp.route("/standings/constructors/<int:year>")
def get_constructor_standings(year):
    try:    s = _get_standings_cached(year)
    except Exception as e: return _err(f"Could not compute constructor standings for {year}: {e}")
    return jsonify({"year": year, "round": s["round"], "constructor_standings": s["constructors"]})


# ── Lap Times (FastF1) ───────────────────────────────────────────────────────
@f1_bp.route("/lap-times/<int:year>/<int:round_num>/<driver_code>")
def get_lap_times(year, round_num, driver_code):
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=False, weather=False, messages=False)
    except Exception as e:
        return _err(f"Lap times {driver_code} {year}/{round_num}: {e}")

    try:
        driver_laps = session.laps.pick_drivers(driver_code.upper())
    except Exception as e:
        return _err(f"No laps for driver {driver_code}: {e}")

    laps = []
    for _, lap in driver_laps.iterlaps():
        lap_time = lap["LapTime"]
        if pd.isna(lap_time): continue
        try:    lap_secs = lap_time.total_seconds()
        except: continue
        if lap_secs < 60 or lap_secs > 300: continue
        laps.append({
            "lap_number":       _to_int(lap.get("LapNumber")),
            "lap_time_sec":     round(lap_secs, 3),
            "compound":         _to_str(lap.get("Compound"), "UNKNOWN") or "UNKNOWN",
            "tyre_life":        _to_int(lap.get("TyreLife")),
            "pit_in":           not pd.isna(lap.get("PitInTime")),
            "pit_out":          not pd.isna(lap.get("PitOutTime")),
            "is_personal_best": bool(lap.get("IsPersonalBest", False)) if not pd.isna(lap.get("IsPersonalBest")) else False,
        })

    return jsonify({
        "year": year, "round": round_num,
        "race_name": _to_str(session.event.get("EventName")),
        "driver":    driver_code.upper(),
        "laps":      laps, "total_laps": len(laps),
    })


# ── Circuits (FastF1 → Jolpica fallback) ─────────────────────────────────────
CIRCUIT_COORDS = {
    "Sakhir": (26.0325, 50.5106), "Jeddah": (21.6319, 39.1044),
    "Melbourne": (-37.8497, 144.9680), "Suzuka": (34.8431, 136.5410),
    "Shanghai": (31.3389, 121.2197), "Miami": (25.9581, -80.2389),
    "Imola": (44.3439, 11.7167), "Monaco": (43.7347, 7.4206),
    "Montréal": (45.5000, -73.5228), "Montreal": (45.5000, -73.5228),
    "Barcelona": (41.5700, 2.2611), "Madrid": (40.4710, -3.5610),
    "Spielberg": (47.2197, 14.7647), "Silverstone": (52.0786, -1.0169),
    "Budapest": (47.5789, 19.2486), "Spa-Francorchamps": (50.4372, 5.9714),
    "Zandvoort": (52.3888, 4.5408), "Monza": (45.6156, 9.2811),
    "Baku": (40.3725, 49.8533), "Marina Bay": (1.2914, 103.8642),
    "Singapore": (1.2914, 103.8642), "Austin": (30.1328, -97.6411),
    "Mexico City": (19.4042, -99.0907), "São Paulo": (-23.7036, -46.6997),
    "Sao Paulo": (-23.7036, -46.6997), "Las Vegas": (36.1147, -115.1728),
    "Lusail": (25.4900, 51.4542), "Yas Island": (24.4672, 54.6031),
    "Abu Dhabi": (24.4672, 54.6031),
}

def _circuits_fastf1(year):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if schedule is None or schedule.empty: return None
    schedule = schedule.sort_values("RoundNumber").reset_index(drop=True)
    seen, circuits = set(), []
    for _, ev in schedule.iterrows():
        rnd = _to_int(ev.get("RoundNumber"))
        if rnd is None or rnd <= 0: continue
        location = _to_str(ev.get("Location"))
        if not location or location in seen: continue
        seen.add(location)
        coords = CIRCUIT_COORDS.get(location, (None, None))
        circuits.append({
            "name":      _to_str(ev.get("OfficialEventName")) or _to_str(ev.get("EventName")),
            "country":   _to_str(ev.get("Country")),
            "locality":  location,
            "latitude":  coords[0], "longitude": coords[1],
            "url":       "",
        })
    return circuits

def _circuits_jolpica(year):
    data = _jolpica(f"/{year}/circuits.json?limit=100")
    if not data: return None
    raw = data.get("MRData", {}).get("CircuitTable", {}).get("Circuits", [])
    if not raw: return None
    out = []
    for c in raw:
        loc = c.get("Location", {})
        try:    lat, lng = float(loc.get("lat", 0)), float(loc.get("long", 0))
        except (ValueError, TypeError): lat, lng = None, None
        out.append({
            "name":      c.get("circuitName", ""),
            "country":   loc.get("country", ""),
            "locality":  loc.get("locality", ""),
            "latitude":  lat, "longitude": lng,
            "url":       c.get("url", ""),
        })
    return out

@f1_bp.route("/circuits/<int:year>")
def get_circuits(year):
    circuits = None
    try:    circuits = _circuits_fastf1(year)
    except Exception as e: print(f"[circuits] FastF1 failed: {e}")
    if not circuits:
        circuits = _circuits_jolpica(year)
    if not circuits:
        return _err(f"Could not load circuits for {year}")
    return jsonify({"year": year, "circuits": circuits})


# ════════════════════════════════════════════════════════════════════════════
#   REPLAY ENDPOINTS  (FastF1 telemetry — historical race replay)
# ════════════════════════════════════════════════════════════════════════════
def _load_race_session(year, round_num, want_laps=True, want_telemetry=False):
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=want_laps, telemetry=want_telemetry, weather=False, messages=False)
    return session


@f1_bp.route("/replay/session-info/<int:year>/<int:round_num>")
def replay_session_info(year, round_num):
    """Drivers list with team color + line-style hint, plus session metadata."""
    cache_key = ("replay-info", year, round_num)
    def compute():
        sess = _load_race_session(year, round_num, want_laps=True, want_telemetry=False)
        results = sess.results
        team_seen = {}
        drivers = []
        for _, d in results.iterrows():
            code = _to_str(d.get("Abbreviation"))
            if not code: continue
            team = _to_str(d.get("TeamName"))
            tcolor = _normalize_team_color(d.get("TeamColor"))
            seen_count = team_seen.get(team, 0)
            line_style = "solid" if seen_count == 0 else "dashed"
            team_seen[team] = seen_count + 1
            drivers.append({
                "code":       code,
                "number":     _to_int(d.get("DriverNumber")),
                "full_name":  _to_str(d.get("FullName")),
                "team":       team,
                "team_color": tcolor,
                "line_style": line_style,
                "position":   _to_int(d.get("Position")),
            })

        total_laps = 0
        try:
            total_laps = int(sess.laps["LapNumber"].max()) if not sess.laps.empty else 0
        except Exception:
            pass

        try:    race_date = str(pd.Timestamp(sess.event["EventDate"]).date())
        except: race_date = ""

        return {
            "year": year, "round": round_num,
            "race_name": _to_str(sess.event.get("EventName")),
            "circuit":   _to_str(sess.event.get("Location")),
            "country":   _to_str(sess.event.get("Country")),
            "date":      race_date,
            "total_laps": total_laps,
            "drivers":   drivers,
        }

    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Session info {year}/{round_num}: {e}")


@f1_bp.route("/replay/track-map/<int:year>/<int:round_num>")
def replay_track_map(year, round_num):
    cache_key = ("track-map", year, round_num)
    def compute():
        sess = fastf1.get_session(year, round_num, "R")
        sess.load(laps=True, telemetry=True, weather=False, messages=False)
        fastest = sess.laps.pick_fastest()
        if fastest is None or (hasattr(fastest, "empty") and fastest.empty):
            raise RuntimeError("No fastest lap available")
        tel = fastest.get_telemetry()
        if tel is None or tel.empty:
            raise RuntimeError("No telemetry on fastest lap")
        N = len(tel); step = max(1, N // 400)
        sub = tel.iloc[::step].copy()
        xs = sub["X"].astype(float).tolist()
        ys = sub["Y"].astype(float).tolist()
        dists = sub["Distance"].astype(float).tolist()
        return {
            "year": year, "round": round_num,
            "x": xs, "y": ys, "distance": dists,
            "bounds": {"x_min": float(min(xs)), "x_max": float(max(xs)),
                       "y_min": float(min(ys)), "y_max": float(max(ys))},
            "track_length": float(tel["Distance"].max()),
            "num_points": len(xs),
        }
    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Track map {year}/{round_num}: {e}")


@f1_bp.route("/replay/mini-sectors/<int:year>/<int:round_num>")
def replay_mini_sectors(year, round_num):
    """Per-driver fastest-lap time per mini-sector. Frontend computes 'fastest visible'."""
    n = max(5, min(60, request.args.get("n", default=25, type=int)))
    cache_key = ("mini-sectors", year, round_num, n)

    def compute():
        sess = fastf1.get_session(year, round_num, "R")
        sess.load(laps=True, telemetry=True, weather=False, messages=False)

        ref_lap = sess.laps.pick_fastest()
        ref_tel = ref_lap.get_telemetry()
        track_length = float(ref_tel["Distance"].max())
        boundaries = np.linspace(0, track_length, n + 1)

        N = len(ref_tel); step = max(1, N // 400)
        sub = ref_tel.iloc[::step]
        track_x = sub["X"].astype(float).tolist()
        track_y = sub["Y"].astype(float).tolist()
        track_d = sub["Distance"].astype(float).tolist()

        codes = [c for c in sess.laps["Driver"].dropna().unique().tolist() if c]
        sector_times_per_driver = {}
        for code in codes:
            try:
                lap = sess.laps.pick_drivers(code).pick_fastest()
                if lap is None or (hasattr(lap, "empty") and lap.empty): continue
                tel = lap.get_telemetry()
                if tel is None or tel.empty: continue
                t = pd.to_timedelta(tel["SessionTime"])
                t_secs = (t - t.iloc[0]).dt.total_seconds().to_numpy()
                d = tel["Distance"].astype(float).to_numpy()

                sector_times = []
                for i in range(n):
                    a, b = boundaries[i], boundaries[i+1]
                    mask = (d >= a) & (d <= b)
                    if mask.sum() < 2:
                        sector_times.append(None); continue
                    seg_t = t_secs[mask]
                    duration = float(seg_t.max() - seg_t.min())
                    sector_times.append(round(duration, 4) if duration > 0 else None)
                sector_times_per_driver[code] = sector_times
            except Exception as e:
                print(f"[mini-sectors] skip {code}: {e}")

        return {
            "year": year, "round": round_num,
            "num_sectors": n, "track_length": track_length,
            "boundaries": [float(b) for b in boundaries],
            "track": {"x": track_x, "y": track_y, "distance": track_d},
            "drivers": sector_times_per_driver,
        }

    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Mini-sectors {year}/{round_num}: {e}")


@f1_bp.route("/replay/telemetry/<int:year>/<int:round_num>/<driver_code>")
def replay_telemetry(year, round_num, driver_code):
    cache_key = ("telemetry", year, round_num, driver_code.upper())
    def compute():
        sess = fastf1.get_session(year, round_num, "R")
        sess.load(laps=True, telemetry=True, weather=False, messages=False)
        lap = sess.laps.pick_drivers(driver_code.upper()).pick_fastest()
        if lap is None or (hasattr(lap, "empty") and lap.empty):
            raise RuntimeError(f"No fastest lap for {driver_code}")
        tel = lap.get_telemetry()
        N = len(tel); step = max(1, N // 500)
        sub = tel.iloc[::step].copy()
        return {
            "year": year, "round": round_num,
            "driver": driver_code.upper(),
            "lap_number": _to_int(lap.get("LapNumber")),
            "lap_time_sec": float(lap["LapTime"].total_seconds()) if not pd.isna(lap.get("LapTime")) else None,
            "compound": _to_str(lap.get("Compound"), ""),
            "distance":  sub["Distance"].astype(float).tolist(),
            "speed":     sub["Speed"].astype(float).tolist(),
            "throttle":  sub["Throttle"].astype(float).tolist(),
            "brake":     [bool(b) for b in sub["Brake"].tolist()],
            "rpm":       sub["RPM"].astype(float).tolist() if "RPM" in sub.columns else [],
            "gear":      sub["nGear"].astype(int).tolist() if "nGear" in sub.columns else [],
            "drs":       sub["DRS"].astype(int).tolist() if "DRS" in sub.columns else [],
            "x":         sub["X"].astype(float).tolist(),
            "y":         sub["Y"].astype(float).tolist(),
        }
    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Telemetry {driver_code} {year}/{round_num}: {e}")


@f1_bp.route("/replay/sector-times/<int:year>/<int:round_num>")
def replay_sector_times(year, round_num):
    cache_key = ("sector-times", year, round_num)
    def compute():
        sess = _load_race_session(year, round_num, want_laps=True, want_telemetry=False)
        laps = sess.laps
        out = []
        codes = [c for c in laps["Driver"].dropna().unique().tolist() if c]
        for code in codes:
            dl = laps.pick_drivers(code)
            if dl.empty: continue
            row = dl.iloc[0]
            entry = {"driver": code, "team": _to_str(row.get("Team")),
                     "best_s1": None, "best_s2": None, "best_s3": None, "best_lap": None}
            for col, key in [("Sector1Time","best_s1"),("Sector2Time","best_s2"),
                             ("Sector3Time","best_s3"),("LapTime","best_lap")]:
                if col not in dl.columns: continue
                vals = dl[col].dropna()
                if vals.empty: continue
                t = vals.min()
                try:    entry[key] = round(t.total_seconds(), 3)
                except: pass
            out.append(entry)
        return {"year": year, "round": round_num, "drivers": out}
    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Sector times {year}/{round_num}: {e}")


@f1_bp.route("/replay/long-stints/<int:year>/<int:round_num>")
def replay_long_stints(year, round_num):
    cache_key = ("long-stints", year, round_num)
    def compute():
        sess = _load_race_session(year, round_num, want_laps=True, want_telemetry=False)
        laps = sess.laps
        out = {}
        codes = [c for c in laps["Driver"].dropna().unique().tolist() if c]
        for code in codes:
            dl = laps.pick_drivers(code)
            stints = []
            if "Stint" not in dl.columns or dl.empty:
                out[code] = stints; continue
            for stint_no, group in dl.groupby("Stint"):
                lap_nums = group["LapNumber"].dropna().astype(int).tolist()
                if not lap_nums: continue
                compound = _to_str(group["Compound"].iloc[0] if "Compound" in group.columns else "")
                times = group["LapTime"].dropna()
                avg_sec = round(times.mean().total_seconds(), 3) if not times.empty else None
                stints.append({
                    "stint":    int(stint_no) if not pd.isna(stint_no) else 0,
                    "compound": compound or "UNKNOWN",
                    "start_lap": min(lap_nums), "end_lap":   max(lap_nums),
                    "laps":      len(lap_nums), "avg_lap_time_sec": avg_sec,
                })
            out[code] = stints
        return {"year": year, "round": round_num, "stints": out}
    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Stints {year}/{round_num}: {e}")


@f1_bp.route("/replay/speed-trap/<int:year>/<int:round_num>")
def replay_speed_trap(year, round_num):
    cache_key = ("speed-trap", year, round_num)
    def compute():
        sess = fastf1.get_session(year, round_num, "R")
        sess.load(laps=True, telemetry=True, weather=False, messages=False)
        out = []
        codes = [c for c in sess.laps["Driver"].dropna().unique().tolist() if c]
        for code in codes:
            try:
                dl = sess.laps.pick_drivers(code)
                if dl.empty: continue
                top = None
                if "SpeedST" in dl.columns:
                    v = dl["SpeedST"].dropna()
                    if not v.empty: top = float(v.max())
                if top is None:
                    fastest = dl.pick_fastest()
                    if fastest is not None and not (hasattr(fastest,'empty') and fastest.empty):
                        tel = fastest.get_telemetry()
                        if tel is not None and not tel.empty:
                            top = float(tel["Speed"].max())
                if top is None: continue
                team = _to_str(dl["Team"].iloc[0]) if "Team" in dl.columns else ""
                out.append({"driver": code, "team": team, "top_speed_kph": round(top, 1)})
            except Exception as e:
                print(f"[speed-trap] {code}: {e}")
        out.sort(key=lambda r: -r["top_speed_kph"])
        return {"year": year, "round": round_num, "drivers": out}
    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Speed trap {year}/{round_num}: {e}")


# ════════════════════════════════════════════════════════════════════════════
#   LIVE ENDPOINTS  (OpenF1 — real-time during active sessions)
# ════════════════════════════════════════════════════════════════════════════
@f1_bp.route("/live/current-session")
def live_current_session():
    def compute():
        sessions = _openf1("sessions", {"year": DEFAULT_YEAR})
        if not sessions: return {"session": None}
        sessions.sort(key=lambda s: s.get("date_start", ""), reverse=True)
        return {"session": sessions[0]}
    try:    return jsonify(_cached("live-current", LIVE_TTL_SEC, compute))
    except Exception as e: return _err(f"Current session: {e}")


@f1_bp.route("/live/drivers/<int:session_key>")
def live_drivers(session_key):
    def compute():
        return {"session_key": session_key,
                "drivers": _openf1("drivers", {"session_key": session_key}) or []}
    try:    return jsonify(_cached(("live-drivers", session_key), 60, compute))
    except Exception as e: return _err(f"Live drivers: {e}")


@f1_bp.route("/live/positions/<int:session_key>")
def live_positions(session_key):
    def compute():
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        since = (now - timedelta(seconds=30)).isoformat().replace("+00:00", "Z")
        rows = _openf1("location", {"session_key": session_key, "date>": since}) or []
        latest = {}
        for r in rows:
            num = r.get("driver_number")
            if num is None: continue
            cur = latest.get(num)
            if cur is None or r.get("date", "") > cur.get("date", ""):
                latest[num] = r
        positions = list(latest.values())
        return {"session_key": session_key, "positions": positions, "count": len(positions)}
    try:    return jsonify(_cached(("live-pos", session_key), LIVE_TTL_SEC, compute))
    except Exception as e: return _err(f"Live positions: {e}")


@f1_bp.route("/live/car-data/<int:session_key>/<int:driver_number>")
def live_car_data(session_key, driver_number):
    def compute():
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        since = (now - timedelta(seconds=15)).isoformat().replace("+00:00", "Z")
        rows = _openf1("car_data", {
            "session_key": session_key,
            "driver_number": driver_number,
            "date>": since,
        }) or []
        if not rows:
            return {"session_key": session_key, "driver_number": driver_number, "sample": None}
        rows.sort(key=lambda r: r.get("date", ""), reverse=True)
        return {"session_key": session_key, "driver_number": driver_number, "sample": rows[0]}
    try:    return jsonify(_cached(("live-car", session_key, driver_number), LIVE_TTL_SEC, compute))
    except Exception as e: return _err(f"Live car data: {e}")


@f1_bp.route("/live/intervals/<int:session_key>")
def live_intervals(session_key):
    def compute():
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        since = (now - timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
        rows = _openf1("intervals", {"session_key": session_key, "date>": since}) or []
        latest = {}
        for r in rows:
            num = r.get("driver_number")
            if num is None: continue
            cur = latest.get(num)
            if cur is None or r.get("date", "") > cur.get("date", ""):
                latest[num] = r
        return {"session_key": session_key, "intervals": list(latest.values())}
    try:    return jsonify(_cached(("live-int", session_key), LIVE_TTL_SEC, compute))
    except Exception as e: return _err(f"Live intervals: {e}")
