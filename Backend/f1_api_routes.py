"""
f1_api_routes.py  –  F1 Track Flask App
=========================================
Data sources:
  • FastF1                — schedule and results (primary)
  • Jolpica (Ergast clone)— silent fallback for schedule/circuits if FastF1 fails
  • OpenF1 (api.openf1.org)— laps, stints and telemetry for replays, plus LIVE data

Endpoints:
  Core:
    GET /f1/seasons
    GET /f1/schedule/<year>
    GET /f1/race-results/<year>/<round>
    GET /f1/standings/drivers/<year>
    GET /f1/standings/constructors/<year>
    GET /f1/lap-times/<year>/<round>/<driver_code>
    GET /f1/circuits/<year>

  Replay + lap times (OpenF1):
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
# Vercel's filesystem is read-only except /tmp
CACHE_DIR = ("/tmp/f1cache" if os.environ.get("VERCEL")
             else os.path.join(os.path.dirname(__file__), "f1cache"))
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


def _openf1(endpoint, params=None, timeout=8, retries=3):
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{OPENF1}/{endpoint}", params=params or {},
                             headers=HEADERS, timeout=timeout)
            if r.status_code == 429 and attempt < retries:   # rate limited: back off
                time.sleep(float(r.headers.get("Retry-After") or 2 ** attempt))
                continue
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


# ── Lap Times (OpenF1) ───────────────────────────────────────────────────────
@f1_bp.route("/lap-times/<int:year>/<int:round_num>/<driver_code>")
def get_lap_times(year, round_num, driver_code):
    code = driver_code.upper()
    try:
        race = _race_data(year, round_num)
        num = _driver_number(race, code)
    except Exception as e:
        return _err(f"Lap times {code} {year}/{round_num}: {e}")

    pit_laps = {p["lap_number"] for p in race["pits"] if p["driver_number"] == num}
    laps, best = [], None
    for lap in sorted(_driver_laps(race, num), key=lambda l: l["lap_number"]):
        secs = lap["lap_duration"]
        if secs < 60 or secs > 300: continue
        stint = _stint_for_lap(race, num, lap["lap_number"])
        is_pb = best is None or secs < best
        if is_pb: best = secs
        tyre_life = None
        if stint.get("tyre_age_at_start") is not None:
            tyre_life = stint["tyre_age_at_start"] + lap["lap_number"] - stint["lap_start"] + 1
        laps.append({
            "lap_number":       lap["lap_number"],
            "lap_time_sec":     round(secs, 3),
            "compound":         _to_str(stint.get("compound"), "UNKNOWN") or "UNKNOWN",
            "tyre_life":        tyre_life,
            "pit_in":           lap["lap_number"] in pit_laps,
            "pit_out":          bool(lap.get("is_pit_out_lap")),
            "is_personal_best": is_pb,
        })

    return jsonify({
        "year": year, "round": round_num,
        "race_name": race["event"]["race_name"],
        "driver":    code,
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
#   REPLAY ENDPOINTS  (OpenF1 — historical race replay)
# ════════════════════════════════════════════════════════════════════════════
# F1's live-timing archive, where FastF1 gets laps and telemetry, now requires a
# paid F1 TV login, so race laps, stints and car telemetry come from OpenF1.

def _openf1_rows(endpoint, **params):
    rows = _openf1(endpoint, params, timeout=30)
    if rows is None:
        raise RuntimeError(f"OpenF1 {endpoint} unavailable")
    return rows


def _race_data(year, round_num):
    """OpenF1 race session (matched to the schedule's race day) with its laps,
    stints, pit stops, results and drivers."""
    def compute():
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        ev = schedule[schedule["RoundNumber"] == round_num]
        if ev.empty: raise RuntimeError(f"No round {round_num} in {year}")
        ev = ev.iloc[0]
        race_day = pd.Timestamp(ev["EventDate"]).date()

        sessions = _openf1_rows("sessions", year=year, session_name="Race")
        def days_off(s): return abs((datetime.date.fromisoformat(s["date_start"][:10]) - race_day).days)
        sess = min(sessions, key=days_off, default=None)
        if sess is None or days_off(sess) > 1:
            raise RuntimeError(f"OpenF1 has no race session for {year} round {round_num}")
        sk = sess["session_key"]

        return {
            "session_key": sk,
            "event": {"race_name": _to_str(ev.get("EventName")),
                      "circuit":   _to_str(ev.get("Location")),
                      "country":   _to_str(ev.get("Country")),
                      "date":      race_day.isoformat()},
            "drivers": {d["driver_number"]: d for d in _openf1_rows("drivers", session_key=sk)},
            "results": sorted(_openf1_rows("session_result", session_key=sk),
                              key=lambda r: r.get("position") or 99),
            "laps":    [l for l in _openf1_rows("laps", session_key=sk) if l.get("lap_duration")],
            "stints":  _openf1_rows("stints", session_key=sk),
            "pits":    _openf1_rows("pit", session_key=sk),
        }
    return _cached(("of1-race", year, round_num), REPLAY_TTL_SEC, compute)


def _race_driver_numbers(race):
    """Driver numbers in finishing order, then anyone without a result."""
    nums = [r["driver_number"] for r in race["results"]]
    return nums + [n for n in race["drivers"] if n not in nums]


def _code(race, num):
    return _to_str(race["drivers"].get(num, {}).get("name_acronym")) or str(num)


def _team(race, num):
    return _to_str(race["drivers"].get(num, {}).get("team_name"))


def _driver_number(race, code):
    for num, d in race["drivers"].items():
        if _to_str(d.get("name_acronym")).upper() == code.upper():
            return num
    raise RuntimeError(f"No driver {code} in this race")


def _driver_laps(race, num):
    return [l for l in race["laps"] if l["driver_number"] == num]


def _fastest_lap(laps):
    clean = [l for l in laps if not l.get("is_pit_out_lap")]
    return min(clean or laps, key=lambda l: l["lap_duration"], default=None)


def _stint_for_lap(race, num, lap_number):
    for s in race["stints"]:
        if (s["driver_number"] == num and s.get("lap_start") is not None
                and s["lap_start"] <= lap_number <= (s.get("lap_end") or lap_number)):
            return s
    return {}


def _lap_telemetry(race, lap, with_position=True):
    """Car data for one lap as a DataFrame with time (s) and integrated distance (m),
    plus x/y track position when with_position is set."""
    start = pd.Timestamp(lap["date_start"])
    end = start + pd.Timedelta(seconds=lap["lap_duration"])
    # requests sends {"date>": v} as "date>=v", which OpenF1 reads as date >= v
    window = {"session_key": race["session_key"], "driver_number": lap["driver_number"],
              "date>": start.isoformat(), "date<": end.isoformat()}

    car = pd.DataFrame(_openf1_rows("car_data", **window))
    if car.empty: raise RuntimeError("No car data for lap")
    car["date"] = pd.to_datetime(car["date"], format="ISO8601")
    car = car.sort_values("date").reset_index(drop=True)
    t = (car["date"] - start).dt.total_seconds()
    car["time"] = t
    car["distance"] = np.cumsum(car["speed"] / 3.6 * t.diff().fillna(t.iloc[0]).clip(lower=0))

    if with_position:
        loc = pd.DataFrame(_openf1_rows("location", **window))
        if loc.empty: raise RuntimeError("No position data for lap")
        loc["date"] = pd.to_datetime(loc["date"], format="ISO8601")
        car = pd.merge_asof(car, loc.sort_values("date")[["date", "x", "y"]],
                            on="date", direction="nearest")
    return car


@f1_bp.route("/replay/session-info/<int:year>/<int:round_num>")
def replay_session_info(year, round_num):
    """Drivers list with team color + line-style hint, plus session metadata."""
    cache_key = ("replay-info", year, round_num)
    def compute():
        race = _race_data(year, round_num)
        positions = {r["driver_number"]: r.get("position") for r in race["results"]}
        team_seen = {}
        drivers = []
        for num in _race_driver_numbers(race):
            d = race["drivers"].get(num)
            if not d: continue
            team = _team(race, num)
            seen_count = team_seen.get(team, 0)
            team_seen[team] = seen_count + 1
            drivers.append({
                "code":       _code(race, num),
                "number":     num,
                "full_name":  _to_str(d.get("full_name")),
                "team":       team,
                "team_color": _normalize_team_color(d.get("team_colour")),
                "line_style": "solid" if seen_count == 0 else "dashed",
                "position":   positions.get(num),
            })
        return {
            "year": year, "round": round_num, **race["event"],
            "total_laps": max((l["lap_number"] for l in race["laps"]), default=0),
            "drivers":   drivers,
        }

    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Session info {year}/{round_num}: {e}")


@f1_bp.route("/replay/track-map/<int:year>/<int:round_num>")
def replay_track_map(year, round_num):
    cache_key = ("track-map", year, round_num)
    def compute():
        race = _race_data(year, round_num)
        fastest = _fastest_lap(race["laps"])
        if fastest is None: raise RuntimeError("No fastest lap available")
        tel = _lap_telemetry(race, fastest)
        N = len(tel); step = max(1, N // 400)
        sub = tel.iloc[::step]
        xs = sub["x"].astype(float).tolist()
        ys = sub["y"].astype(float).tolist()
        return {
            "year": year, "round": round_num,
            "x": xs, "y": ys, "distance": sub["distance"].astype(float).tolist(),
            "bounds": {"x_min": float(min(xs)), "x_max": float(max(xs)),
                       "y_min": float(min(ys)), "y_max": float(max(ys))},
            "track_length": float(tel["distance"].max()),
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
        race = _race_data(year, round_num)
        fastest = _fastest_lap(race["laps"])
        if fastest is None: raise RuntimeError("No fastest lap available")
        ref_tel = _lap_telemetry(race, fastest)
        track_length = float(ref_tel["distance"].max())
        boundaries = np.linspace(0, track_length, n + 1)

        N = len(ref_tel); step = max(1, N // 400)
        sub = ref_tel.iloc[::step]

        sector_times_per_driver = {}
        for num in _race_driver_numbers(race):
            code = _code(race, num)
            try:
                lap = _fastest_lap(_driver_laps(race, num))
                if lap is None: continue
                tel = _lap_telemetry(race, lap, with_position=False)
                # scale integrated distance onto the reference lap so boundaries line up
                d = tel["distance"].to_numpy() * track_length / tel["distance"].max()
                t_at = np.interp(boundaries, d, tel["time"].to_numpy())
                sector_times_per_driver[code] = [
                    round(float(dt), 4) if dt > 0 else None for dt in np.diff(t_at)]
            except Exception as e:
                print(f"[mini-sectors] skip {code}: {e}")

        return {
            "year": year, "round": round_num,
            "num_sectors": n, "track_length": track_length,
            "boundaries": [float(b) for b in boundaries],
            "track": {"x": sub["x"].astype(float).tolist(),
                      "y": sub["y"].astype(float).tolist(),
                      "distance": sub["distance"].astype(float).tolist()},
            "drivers": sector_times_per_driver,
        }

    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Mini-sectors {year}/{round_num}: {e}")


@f1_bp.route("/replay/telemetry/<int:year>/<int:round_num>/<driver_code>")
def replay_telemetry(year, round_num, driver_code):
    cache_key = ("telemetry", year, round_num, driver_code.upper())
    def compute():
        race = _race_data(year, round_num)
        num = _driver_number(race, driver_code)
        lap = _fastest_lap(_driver_laps(race, num))
        if lap is None: raise RuntimeError(f"No fastest lap for {driver_code}")
        tel = _lap_telemetry(race, lap)
        N = len(tel); step = max(1, N // 500)
        sub = tel.iloc[::step]
        return {
            "year": year, "round": round_num,
            "driver": driver_code.upper(),
            "lap_number": lap["lap_number"],
            "lap_time_sec": float(lap["lap_duration"]),
            "compound": _to_str(_stint_for_lap(race, num, lap["lap_number"]).get("compound"), ""),
            "distance":  sub["distance"].astype(float).tolist(),
            "speed":     sub["speed"].astype(float).tolist(),
            "throttle":  sub["throttle"].astype(float).tolist(),
            "brake":     [bool(b) for b in sub["brake"].tolist()],
            "rpm":       sub["rpm"].astype(float).tolist(),
            "gear":      sub["n_gear"].astype(int).tolist(),
            "drs":       sub["drs"].astype(int).tolist(),
            "x":         sub["x"].astype(float).tolist(),
            "y":         sub["y"].astype(float).tolist(),
        }
    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Telemetry {driver_code} {year}/{round_num}: {e}")


@f1_bp.route("/replay/sector-times/<int:year>/<int:round_num>")
def replay_sector_times(year, round_num):
    cache_key = ("sector-times", year, round_num)
    def compute():
        race = _race_data(year, round_num)
        out = []
        for num in _race_driver_numbers(race):
            dl = _driver_laps(race, num)
            if not dl: continue
            entry = {"driver": _code(race, num), "team": _team(race, num)}
            for col, key in [("duration_sector_1", "best_s1"), ("duration_sector_2", "best_s2"),
                             ("duration_sector_3", "best_s3"), ("lap_duration", "best_lap")]:
                vals = [l[col] for l in dl if l.get(col)]
                entry[key] = round(min(vals), 3) if vals else None
            out.append(entry)
        return {"year": year, "round": round_num, "drivers": out}
    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Sector times {year}/{round_num}: {e}")


@f1_bp.route("/replay/long-stints/<int:year>/<int:round_num>")
def replay_long_stints(year, round_num):
    cache_key = ("long-stints", year, round_num)
    def compute():
        race = _race_data(year, round_num)
        out = {}
        for num in _race_driver_numbers(race):
            dl = _driver_laps(race, num)
            stints = []
            for s in sorted((s for s in race["stints"] if s["driver_number"] == num),
                            key=lambda s: s["stint_number"]):
                if s.get("lap_start") is None: continue
                end = s.get("lap_end") or s["lap_start"]
                times = [l["lap_duration"] for l in dl if s["lap_start"] <= l["lap_number"] <= end]
                stints.append({
                    "stint":     s["stint_number"],
                    "compound":  _to_str(s.get("compound")) or "UNKNOWN",
                    "start_lap": s["lap_start"], "end_lap": end,
                    "laps":      end - s["lap_start"] + 1,
                    "avg_lap_time_sec": round(sum(times) / len(times), 3) if times else None,
                })
            out[_code(race, num)] = stints
        return {"year": year, "round": round_num, "stints": out}
    try:    return jsonify(_cached(cache_key, REPLAY_TTL_SEC, compute))
    except Exception as e: return _err(f"Stints {year}/{round_num}: {e}")


@f1_bp.route("/replay/speed-trap/<int:year>/<int:round_num>")
def replay_speed_trap(year, round_num):
    cache_key = ("speed-trap", year, round_num)
    def compute():
        race = _race_data(year, round_num)
        out = []
        for num in _race_driver_numbers(race):
            speeds = [l["st_speed"] for l in _driver_laps(race, num) if l.get("st_speed")]
            if not speeds: continue
            out.append({"driver": _code(race, num), "team": _team(race, num),
                        "top_speed_kph": round(float(max(speeds)), 1)})
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
