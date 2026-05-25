import os
import httpx
import fastf1
from typing import Optional, Tuple
from state import AgentState, DataResult

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
FASTF1_CACHE = os.getenv("FASTF1_CACHE", "./fastf1_cache")

fastf1.Cache.enable_cache(FASTF1_CACHE)

# Full 2026 driver-team mapping (22 drivers, 11 constructors).
# Exported so retrieval and synthesis agents can import it for context.
DRIVERS_2026: dict[str, list[str]] = {
    "Mercedes":     ["Russell", "Antonelli"],
    "Ferrari":      ["Leclerc", "Hamilton"],
    "McLaren":      ["Norris", "Piastri"],
    "Red Bull":     ["Verstappen", "Hadjar"],
    "Aston Martin": ["Alonso", "Stroll"],
    "Alpine":       ["Gasly", "Colapinto"],
    "Williams":     ["Sainz", "Albon"],
    "Racing Bulls": ["Lawson", "Lindblad"],
    "Haas":         ["Bearman", "Ocon"],
    "Audi":         ["Hulkenberg", "Bortoleto"],
    "Cadillac":     ["Perez", "Bottas"],
}

# (season, round, is_sprint_weekend)
# Sprint weekends have only FP1 (no FP2/FP3); available sessions: FP1, SQ, S, Q, R.
_CIRCUIT_MAP: dict[str, tuple[int, int, bool]] = {
    # Round 1 — Australia
    "australia":    (2026, 1, False),
    "melbourne":    (2026, 1, False),
    "australian":   (2026, 1, False),
    # Round 2 — China (sprint)
    "china":        (2026, 2, True),
    "shanghai":     (2026, 2, True),
    "chinese":      (2026, 2, True),
    # Round 3 — Bahrain
    "bahrain":      (2026, 3, False),
    "sakhir":       (2026, 3, False),
    # Round 4 — Saudi Arabia
    "saudi":        (2026, 4, False),
    "jeddah":       (2026, 4, False),
    "saudi arabia": (2026, 4, False),
    # Round 5 — Canada (sprint)
    "montreal":     (2026, 5, True),
    "canada":       (2026, 5, True),
    "canadian":     (2026, 5, True),
    # Round 6 — Spain
    "spain":        (2026, 6, False),
    "barcelona":    (2026, 6, False),
    "spanish":      (2026, 6, False),
    # Round 7 — Austria (sprint)
    "austria":      (2026, 7, True),
    "spielberg":    (2026, 7, True),
    "austrian":     (2026, 7, True),
    # Round 8 — Britain
    "britain":      (2026, 8, False),
    "silverstone":  (2026, 8, False),
    "british":      (2026, 8, False),
    # Round 9 — Belgium
    "belgium":      (2026, 9, False),
    "spa":          (2026, 9, False),
    "belgian":      (2026, 9, False),
    # Round 10 — Hungary
    "hungary":      (2026, 10, False),
    "budapest":     (2026, 10, False),
    "hungarian":    (2026, 10, False),
    # Round 11 — Netherlands
    "netherlands":  (2026, 11, False),
    "zandvoort":    (2026, 11, False),
    "dutch":        (2026, 11, False),
    # Round 12 — Italy
    "italy":        (2026, 12, False),
    "monza":        (2026, 12, False),
    "italian":      (2026, 12, False),
    # Round 13 — Azerbaijan (sprint)
    "azerbaijan":   (2026, 13, True),
    "baku":         (2026, 13, True),
    # Round 14 — Singapore
    "singapore":    (2026, 14, False),
    "marina bay":   (2026, 14, False),
    # Round 15 — USA / Austin (sprint)
    "usa":          (2026, 15, True),
    "austin":       (2026, 15, True),
    "cota":         (2026, 15, True),
    "united states": (2026, 15, True),
    # Round 16 — Mexico
    "mexico":       (2026, 16, False),
    "mexico city":  (2026, 16, False),
    "mexican":      (2026, 16, False),
    # Round 17 — Brazil (sprint)
    "brazil":       (2026, 17, True),
    "sao paulo":    (2026, 17, True),
    "interlagos":   (2026, 17, True),
    "brazilian":    (2026, 17, True),
    # Round 18 — Las Vegas
    "las vegas":    (2026, 18, False),
    "vegas":        (2026, 18, False),
    # Round 19 — Qatar (sprint)
    "qatar":        (2026, 19, True),
    "lusail":       (2026, 19, True),
    "qatari":       (2026, 19, True),
    # Round 20 — Abu Dhabi
    "abu dhabi":    (2026, 20, False),
    "yas marina":   (2026, 20, False),
    "yas":          (2026, 20, False),
}

# Session preference order per weekend type. FastF1 raises for sessions that
# don't exist, so we try each in sequence and take the first that loads.
_SPRINT_SESSIONS   = ["FP1", "S", "Q", "R"]          # no FP2/FP3
_STANDARD_SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]


def _jolpica(path: str) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{JOLPICA_BASE}{path}.json")
        resp.raise_for_status()
        return resp.json()


def _race_results(season: int, round_num: Optional[int] = None) -> DataResult:
    path = f"/{season}/{round_num}/results" if round_num else f"/{season}/results"
    return DataResult(source="jolpica", endpoint=path, data=_jolpica(path))


def _qualifying_results(season: int, round_num: int) -> DataResult:
    path = f"/{season}/{round_num}/qualifying"
    return DataResult(source="jolpica", endpoint=path, data=_jolpica(path))


def _sprint_results(season: int, round_num: int) -> DataResult:
    path = f"/{season}/{round_num}/sprint"
    return DataResult(source="jolpica", endpoint=path, data=_jolpica(path))


def _fastest_laps(season: int, round_num: int) -> DataResult:
    path = f"/{season}/{round_num}/laps/fastest"
    return DataResult(source="jolpica", endpoint=path, data=_jolpica(path))


def _driver_standings(season: int) -> DataResult:
    path = f"/{season}/driverStandings"
    return DataResult(source="jolpica", endpoint=path, data=_jolpica(path))


def _constructor_standings(season: int) -> DataResult:
    path = f"/{season}/constructorStandings"
    return DataResult(source="jolpica", endpoint=path, data=_jolpica(path))


def _latest_completed_round(season: int) -> Optional[int]:
    """Return the highest round number with posted results for *season*, via Jolpica."""
    try:
        data = _jolpica(f"/{season}/results")
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if races:
            return int(races[-1]["round"])
    except Exception:
        pass
    return None


def _fastf1_session(season: int, round_num: int, session_type: str) -> DataResult:
    try:
        session = fastf1.get_session(season, round_num, session_type)
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps[
            ["Driver", "LapNumber", "LapTime", "Compound", "TyreLife", "Stint"]
        ].head(60)
        return DataResult(
            source="fastf1",
            endpoint=f"session/{season}/{round_num}/{session_type}",
            data={
                "event": session.event["EventName"],
                "date": str(session.date),
                "session_type": session_type,
                "laps": laps.to_dict(orient="records"),
            },
        )
    except Exception as exc:
        return DataResult(
            source="fastf1",
            endpoint=f"session/{season}/{round_num}/{session_type}",
            data={"error": str(exc)},
        )


def _fastf1_best_available(season: int, round_num: int, is_sprint: bool) -> DataResult:
    """Try sessions in weekend-appropriate order.

    If the requested round has no FastF1 data yet (future race), falls back to
    the most recent completed round in the *same* season using Jolpica — never
    crosses season boundaries so we never silently serve prior-season data.
    """
    session_order = _SPRINT_SESSIONS if is_sprint else _STANDARD_SESSIONS

    for stype in session_order:
        result = _fastf1_session(season, round_num, stype)
        if "error" not in result.data:
            return result

    # All sessions for the requested round failed — it is probably a future race.
    # Ask Jolpica which rounds have completed results and try the latest one.
    fallback_round = _latest_completed_round(season)
    if fallback_round is not None and fallback_round != round_num:
        for stype in _STANDARD_SESSIONS:
            fb = _fastf1_session(season, fallback_round, stype)
            if "error" not in fb.data:
                return fb

    # Nothing worked; surface the original failure so the caller can log it.
    return result  # noqa: F821 — loop above always assigns result


def _grid_context() -> DataResult:
    """Inject the static 2026 driver grid so the synthesis agent has it as context."""
    return DataResult(
        source="static",
        endpoint="2026_driver_grid",
        data={"season": 2026, "drivers_by_team": DRIVERS_2026},
    )


def _infer_season(text: str) -> int:
    for year in range(2020, 2027):
        if str(year) in text:
            return year
    return 2026


def _infer_circuit(text: str) -> Optional[Tuple[int, int, bool]]:
    lower = text.lower()
    for keyword, info in _CIRCUIT_MAP.items():
        if keyword in lower:
            return info
    return None


def data_agent(state: AgentState) -> dict:
    results: list[DataResult] = []

    tasks = [t for t in state.get("sub_tasks", []) if t.task_type in ("data", "both")]
    if not tasks:
        tasks = state.get("sub_tasks", [])

    season = _infer_season(state["query"])
    circuit_info = _infer_circuit(state["query"])

    # Always include the 2026 grid so synthesis knows all drivers and constructors.
    results.append(_grid_context())

    try:
        results.append(_driver_standings(season))
    except Exception:
        pass

    try:
        results.append(_constructor_standings(season))
    except Exception:
        pass

    if circuit_info:
        c_season, c_round, is_sprint = circuit_info
        try:
            results.append(_race_results(c_season, c_round))
        except Exception:
            pass
        try:
            results.append(_qualifying_results(c_season, c_round))
        except Exception:
            pass
        try:
            results.append(_fastest_laps(c_season, c_round))
        except Exception:
            pass
        if is_sprint:
            try:
                results.append(_sprint_results(c_season, c_round))
            except Exception:
                pass
        results.append(_fastf1_best_available(c_season, c_round, is_sprint))
    else:
        for task in tasks:
            task_season = _infer_season(task.query)
            task_circuit = _infer_circuit(task.query)

            try:
                results.append(_race_results(task_season))
            except Exception:
                pass

            if task_circuit:
                tc_season, tc_round, tc_sprint = task_circuit
                results.append(_fastf1_best_available(tc_season, tc_round, tc_sprint))
            else:
                results.append(_fastf1_best_available(task_season, 1, is_sprint=False))

    if not results:
        try:
            results.append(_race_results(season))
        except Exception:
            pass

    return {"data_results": results}
