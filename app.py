import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="Pro Match Analytics (Football)", layout="wide")
st.title("⚽ Pro Match Analytics (Football) — Season-independent")
st.caption(
    "Bu uygulama olasılık üretir; kesin tahmin değildir. Bahis/iddia yüksek risklidir. "
    "Eğlence ve araştırma amaçlı kullan."
)

# =========================================================
# SOURCES
# =========================================================
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
NEW_LEAGUE_URL = "https://www.football-data.co.uk/new/{div}.csv"  # Extra leagues often here

# =========================================================
# LEAGUE LIST (Manually named + auto-discovery from fixtures)
# =========================================================
LEAGUES = {
    # Turkey
    "Turkey - Süper Lig (T1)": "T1",

    # England
    "England - Premier League (E0)": "E0",
    "England - Championship (E1)": "E1",
    "England - League One (E2)": "E2",
    "England - League Two (E3)": "E3",
    "England - Conference (EC)": "EC",

    # Scotland
    "Scotland - Premiership (SC0)": "SC0",
    "Scotland - Championship (SC1)": "SC1",
    "Scotland - League One (SC2)": "SC2",
    "Scotland - League Two (SC3)": "SC3",

    # Spain / Italy / Germany / France
    "Spain - La Liga (SP1)": "SP1",
    "Spain - Segunda (SP2)": "SP2",
    "Italy - Serie A (I1)": "I1",
    "Italy - Serie B (I2)": "I2",
    "Germany - Bundesliga (D1)": "D1",
    "Germany - 2. Bundesliga (D2)": "D2",
    "France - Ligue 1 (F1)": "F1",
    "France - Ligue 2 (F2)": "F2",

    # Netherlands / Belgium / Portugal / Greece
    "Netherlands - Eredivisie (N1)": "N1",
    "Belgium - Jupiler Pro (B1)": "B1",
    "Portugal - Primeira Liga (P1)": "P1",
    "Greece - Super League (G1)": "G1",

    # Extra leagues (football-data 'Extra Leagues')
    "USA - MLS (USA)": "USA",
    "Argentina - Primera División (ARG)": "ARG",
    "Brazil - Serie A (BRA)": "BRA",
    "Mexico - Liga MX (MEX)": "MEX",
}

# football-data odds setleri (1X2)
ODDS_SETS = [
    ("Average", ("AvgH", "AvgD", "AvgA")),
    ("Bet365", ("B365H", "B365D", "B365A")),
    ("Pinnacle", ("PSH", "PSD", "PSA")),
    ("WilliamHill", ("WHH", "WHD", "WHA")),
    ("Bet&Win", ("BWH", "BWD", "BWA")),
    ("Interwetten", ("IWH", "IWD", "IWA")),
    ("VCBet", ("VCH", "VCD", "VCA")),
]

# =========================================================
# UTILS
# =========================================================
def season_code(start_year: int) -> str:
    y1 = start_year % 100
    y2 = (start_year + 1) % 100
    return f"{y1:02d}{y2:02d}"

def season_start_year_for_date(d: dt.date) -> int:
    # European season logic (Jul->Jun)
    return d.year if d.month >= 7 else d.year - 1

def hist_url(season: str, div: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df

def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return np.nan
        return float(s)
    except Exception:
        return np.nan

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    # Slightly more robust headers (some endpoints are picky)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)",
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
    }
    r = requests.get(url, timeout=45, headers=headers)
    r.raise_for_status()
    from io import StringIO
    txt = r.content.decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(txt))

@st.cache_data(ttl=60*60, show_spinner=False)
def discover_divs_from_fixtures() -> List[str]:
    fx = parse_dates(load_csv(FIXTURES_URL))
    if "Div" not in fx.columns:
        return []
    divs = (
        fx["Div"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )
    return sorted(divs)

@st.cache_data(show_spinner=False)
def load_fixtures() -> Optional[pd.DataFrame]:
    try:
        return parse_dates(load_csv(FIXTURES_URL))
    except Exception:
        return None

def _normalize_history(df: pd.DataFrame, anchor_date: dt.date) -> pd.DataFrame:
    d = parse_dates(df)

    need = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = need - set(d.columns)
    if missing:
        raise ValueError(f"Zorunlu kolonlar eksik: {sorted(list(missing))}")

    d = d.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce")
    d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
    d = d.dropna(subset=["FTHG", "FTAG"]).copy()

    # Optional numeric columns
    for col in ["HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d[d["Date"] <= pd.Timestamp(anchor_date)].copy()
    d = d.sort_values("Date").drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"], keep="last")
    return d

@st.cache_data(show_spinner=False)
def load_history(div: str, anchor_date: dt.date, back_seasons: int = 6) -> Tuple[pd.DataFrame, List[str], str]:
    """
    1) Main leagues: mmz4281/{season}/{div}.csv (multiple seasons)
    2) If nothing found: /new/{div}.csv (extra leagues)
    """
    start_year = season_start_year_for_date(anchor_date)
    dfs = []
    seasons_loaded = []

    # Try mmz season files first
    for y in range(start_year, start_year - back_seasons - 1, -1):
        s = season_code(y)
        url = hist_url(s, div)
        try:
            df = load_csv(url)
            df["Season"] = s
            dfs.append(df)
            seasons_loaded.append(s)
        except requests.HTTPError as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 404:
                continue
            # Some other HTTP error
            continue
        except Exception:
            continue

    if dfs:
        d = pd.concat(dfs, ignore_index=True)
        d = _normalize_history(d, anchor_date)
        return d, seasons_loaded, "mmz"

    # Fallback: /new/{DIV}.csv
    new_url = NEW_LEAGUE_URL.format(div=div)
    try:
        df = load_csv(new_url)
        # Season might not exist in /new files; keep placeholder
        df["Season"] = "NEW"
        d = _normalize_history(df, anchor_date)
        return d, ["NEW"], "new"
    except Exception as e:
        raise ValueError(
            f"Bu lig/div için data indirilemedi. (mmz sezonda yok + new endpoint başarısız)\n"
            f"Div: {div}\n"
            f"Denediğim: {new_url}\n"
            f"Hata: {e}"
        )

def find_available_odds_sets(df: pd.DataFrame) -> List[str]:
    cols = set(df.columns)
    avail = []
    for name, (h, d, a) in ODDS_SETS:
        if h in cols and d in cols and a in cols:
            avail.append(name)
    return avail

def get_odds_cols(odds_set_name: str) -> Optional[Tuple[str, str, str]]:
    for name, cols in ODDS_SETS:
        if name == odds_set_name:
            return cols
    return None

def implied_probs_1x2(o1: float, ox: float, o2: float) -> Tuple[float, float, float]:
    p1 = 1.0 / o1 if o1 and o1 > 0 else np.nan
    px = 1.0 / ox if ox and ox > 0 else np.nan
    p2 = 1.0 / o2 if o2 and o2 > 0 else np.nan
    s = p1 + px + p2
    if not np.isfinite(s) or s <= 0:
        return np.nan, np.nan, np.nan
    return p1 / s, px / s, p2 / s

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def score_matrix(lam_h: float, lam_a: float, max_goals: int = 6) -> np.ndarray:
    hp = np.array([poisson_pmf(i, lam_h) for i in range(max_goals + 1)], dtype=float)
    ap = np.array([poisson_pmf(i, lam_a) for i in range(max_goals + 1)], dtype=float)
    mat = np.outer(hp, ap)
    return mat / mat.sum()

def outcome_probs_from_mat(mat: np.ndarray) -> Tuple[float, float, float]:
    p_draw = float(np.trace(mat))
    p_home = float(np.tril(mat, -1).sum())
    p_away = float(np.triu(mat, 1).sum())
    return p_home, p_draw, p_away

def prob_over_from_mat(mat: np.ndarray, line: float) -> float:
    p = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (i + j) > line:
                p += mat[i, j]
    return float(p)

def prob_btts_from_mat(mat: np.ndarray) -> float:
    return float(mat[1:, 1:].sum())

def prob_team_over(mat: np.ndarray, team: str, line: float) -> float:
    p = 0.0
    if team == "H":
        for i in range(mat.shape[0]):
            if i > line:
                p += mat[i, :].sum()
    else:
        for j in range(mat.shape[1]):
            if j > line:
                p += mat[:, j].sum()
    return float(p)

def prob_win_to_nil(mat: np.ndarray, side: str) -> float:
    p = 0.0
    if side == "H":
        j = 0
        for i in range(mat.shape[0]):
            if i > j:
                p += mat[i, j]
    else:
        i = 0
        for j in range(mat.shape[1]):
            if j > i:
                p += mat[i, j]
    return float(p)

def prob_combo_btts_over25(mat: np.ndarray) -> float:
    p = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i >= 1 and j >= 1 and (i + j) >= 3:
                p += mat[i, j]
    return float(p)

def prob_handicap_home(mat: np.ndarray, line: float) -> float:
    p = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (i - j) > line:
                p += mat[i, j]
    return float(p)

def prob_handicap_away(mat: np.ndarray, line: float) -> float:
    p = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (j - i) > line:
                p += mat[i, j]
    return float(p)

def top_scores(mat: np.ndarray, topn: int = 10) -> pd.DataFrame:
    items = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            items.append((i, j, mat[i, j]))
    items.sort(key=lambda x: x[2], reverse=True)
    out = pd.DataFrame(items[:topn], columns=["HomeGoals", "AwayGoals", "Prob"])
    out["FairOdds"] = out["Prob"].apply(lambda p: (1.0 / p) if p > 0 else np.nan)
    return out

def exact_total_goals_probs(mat: np.ndarray, max_total: int = 6) -> pd.DataFrame:
    probs = []
    for t in range(max_total + 1):
        p = 0.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i + j == t:
                    p += mat[i, j]
        probs.append((t, p, (1/p if p > 0 else np.nan)))
    return pd.DataFrame(probs, columns=["TotalGoals", "Prob", "FairOdds"])

# xG proxy (shots-based)
def xg_proxy(shots: float, sot: float) -> float:
    if not np.isfinite(shots) and not np.isfinite(sot):
        return np.nan
    s = shots if np.isfinite(shots) else 0.0
    t = sot if np.isfinite(sot) else 0.0
    return 0.04 * s + 0.08 * t

# ELO
def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b) / 400.0))

def elo_update(r_a: float, r_b: float, s_a: float, k: float) -> Tuple[float, float]:
    e_a = elo_expected(r_a, r_b)
    e_b = 1.0 - e_a
    return r_a + k * (s_a - e_a), r_b + k * ((1.0 - s_a) - e_b)

# Glicko-1 (single opponent update)
def glicko_g(rd: float) -> float:
    q = math.log(10) / 400.0
    return 1.0 / math.sqrt(1.0 + (3.0 * q * q * rd * rd) / (math.pi * math.pi))

def glicko_E(r: float, rj: float, rdj: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-glicko_g(rdj) * (r - rj) / 400.0))

def glicko_update(r: float, rd: float, rj: float, rdj: float, s: float) -> Tuple[float, float]:
    q = math.log(10) / 400.0
    g = glicko_g(rdj)
    E = glicko_E(r, rj, rdj)
    d2 = 1.0 / (q * q * g * g * E * (1.0 - E) + 1e-12)
    pre = 1.0 / (1.0 / (rd * rd + 1e-12) + 1.0 / d2)
    r_new = r + q * pre * g * (s - E)
    rd_new = math.sqrt(max(pre, 1e-12))
    return r_new, rd_new

@st.cache_data(show_spinner=False)
def compute_ratings(df_all: pd.DataFrame, elo_k: float, elo_home_adv: float, glicko_home_adv: float):
    d = df_all.sort_values("Date").copy()
    teams = sorted(set(d["HomeTeam"]).union(set(d["AwayTeam"])))

    elo = {t: 1500.0 for t in teams}
    gl = {t: (1500.0, 350.0) for t in teams}

    elo_home_pre, elo_away_pre = [], []
    gl_home_pre, gl_away_pre = [], []
    gl_home_rd_pre, gl_away_rd_pre = [], []

    for _, row in d.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        fthg, ftag = int(row["FTHG"]), int(row["FTAG"])
        s_home = 1.0 if fthg > ftag else (0.0 if fthg < ftag else 0.5)

        elo_home_pre.append(elo[ht])
        elo_away_pre.append(elo[at])
        gl_home_pre.append(gl[ht][0])
        gl_away_pre.append(gl[at][0])
        gl_home_rd_pre.append(gl[ht][1])
        gl_away_rd_pre.append(gl[at][1])

        # ELO update
        rH = elo[ht] + elo_home_adv
        rA = elo[at]
        newH, newA = elo_update(rH, rA, s_home, elo_k)
        elo[ht] = newH - elo_home_adv
        elo[at] = newA

        # Glicko update
        r_h, rd_h = gl[ht]
        r_a, rd_a = gl[at]
        r_h_adj = r_h + glicko_home_adv
        r_a_adj = r_a
        r_h_new, rd_h_new = glicko_update(r_h_adj, rd_h, r_a_adj, rd_a, s_home)
        r_a_new, rd_a_new = glicko_update(r_a_adj, rd_a, r_h_adj, rd_h, 1.0 - s_home)
        gl[ht] = (r_h_new - glicko_home_adv, rd_h_new)
        gl[at] = (r_a_new, rd_a_new)

    d["EloHomePre"] = elo_home_pre
    d["EloAwayPre"] = elo_away_pre
    d["GlickoHomePre"] = gl_home_pre
    d["GlickoAwayPre"] = gl_away_pre
    d["GHomeRDPre"] = gl_home_rd_pre
    d["GAwayRDPre"] = gl_away_rd_pre
    return d, elo, gl

# Shrinkage helper
def shrink_mean(values: np.ndarray, prior_mean: float, prior_weight: float) -> float:
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return float(prior_mean)
    return float((v.sum() + prior_weight * prior_mean) / (n + prior_weight))

# ✅ FIXED weighted averages (this is what was crashing T1 etc.)
def league_avgs_weighted(df_all: pd.DataFrame, anchor_date: dt.date, half_life_days: int) -> Dict[str, float]:
    d = df_all.copy()
    last = pd.Timestamp(anchor_date)

    days_ago = (last - d["Date"]).dt.days.clip(lower=0).to_numpy()
    w = np.exp(-days_ago / float(max(10, half_life_days)))

    def wavg(series) -> float:
        vals = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
        m = np.isfinite(vals)
        if m.sum() == 0:
            return float("nan")
        return float(np.average(vals[m], weights=w[m]))

    out = {
        "home_goals": wavg(d["FTHG"]),
        "away_goals": wavg(d["FTAG"]),
    }

    if "HTHG" in d.columns and "HTAG" in d.columns:
        hthg = pd.to_numeric(d["HTHG"], errors="coerce")
        htag = pd.to_numeric(d["HTAG"], errors="coerce")
        if (
            hthg.notna().sum() > 50
            and np.isfinite(out["home_goals"]) and np.isfinite(out["away_goals"])
            and out["home_goals"] > 0 and out["away_goals"] > 0
        ):
            out["ht_ratio_home"] = float(np.clip(wavg(hthg) / out["home_goals"], 0.25, 0.65))
            out["ht_ratio_away"] = float(np.clip(wavg(htag) / out["away_goals"], 0.25, 0.65))
        else:
            out["ht_ratio_home"] = 0.45
            out["ht_ratio_away"] = 0.45
    else:
        out["ht_ratio_home"] = 0.45
        out["ht_ratio_away"] = 0.45

    out["has_corners"] = ("HC" in d.columns and "AC" in d.columns)
    if out["has_corners"]:
        out["home_corners"] = wavg(d["HC"])
        out["away_corners"] = wavg(d["AC"])
    else:
        out["home_corners"] = float("nan")
        out["away_corners"] = float("nan")

    out["has_shots"] = ("HS" in d.columns and "AS" in d.columns and "HST" in d.columns and "AST" in d.columns)
    if out["has_shots"]:
        hxg = d.apply(lambda r: xg_proxy(r["HS"], r["HST"]), axis=1)
        axg = d.apply(lambda r: xg_proxy(r["AS"], r["AST"]), axis=1)
        out["home_xg"] = wavg(hxg)
        out["away_xg"] = wavg(axg)
    else:
        out["home_xg"] = float("nan")
        out["away_xg"] = float("nan")

    out["raw_home_goals"] = float(pd.to_numeric(d["FTHG"], errors="coerce").mean())
    out["raw_away_goals"] = float(pd.to_numeric(d["FTAG"], errors="coerce").mean())
    out["raw_total_goals"] = out["raw_home_goals"] + out["raw_away_goals"]

    return out

@dataclass
class TeamRates:
    home_attack: float
    home_def: float
    away_attack: float
    away_def: float
    home_xg_att: Optional[float] = None
    home_xg_def: Optional[float] = None
    away_xg_att: Optional[float] = None
    away_xg_def: Optional[float] = None
    home_c_att: Optional[float] = None
    home_c_def: Optional[float] = None
    away_c_att: Optional[float] = None
    away_c_def: Optional[float] = None
    home_used: int = 0
    away_used: int = 0

def build_team_rates_recent(
    df_all: pd.DataFrame,
    league: Dict[str, float],
    lookback: int,
    prior_weight: float,
    min_matches: int,
) -> Dict[str, TeamRates]:
    teams = sorted(set(df_all["HomeTeam"]).union(set(df_all["AwayTeam"])))
    rates: Dict[str, TeamRates] = {}

    for t in teams:
        home = df_all[df_all["HomeTeam"] == t].tail(lookback)
        away = df_all[df_all["AwayTeam"] == t].tail(lookback)

        if (len(home) + len(away)) < min_matches:
            continue

        home_scored = shrink_mean(home["FTHG"].values, league["home_goals"], prior_weight)
        home_conceded = shrink_mean(home["FTAG"].values, league["away_goals"], prior_weight)
        away_scored = shrink_mean(away["FTAG"].values, league["away_goals"], prior_weight)
        away_conceded = shrink_mean(away["FTHG"].values, league["home_goals"], prior_weight)

        ha = home_scored / league["home_goals"] if league["home_goals"] > 0 else 1.0
        hd = home_conceded / league["away_goals"] if league["away_goals"] > 0 else 1.0
        aa = away_scored / league["away_goals"] if league["away_goals"] > 0 else 1.0
        ad = away_conceded / league["home_goals"] if league["home_goals"] > 0 else 1.0

        tr = TeamRates(
            home_attack=float(ha),
            home_def=float(hd),
            away_attack=float(aa),
            away_def=float(ad),
            home_used=int(len(home)),
            away_used=int(len(away)),
        )

        # xG proxy multipliers (if shots exist)
        if league.get("has_shots"):
            if all(c in df_all.columns for c in ["HS", "HST", "AS", "AST"]):
                home_xg_for = np.array([xg_proxy(s, t2) for s, t2 in zip(home["HS"].values, home["HST"].values)], dtype=float)
                home_xg_against = np.array([xg_proxy(s, t2) for s, t2 in zip(home["AS"].values, home["AST"].values)], dtype=float)
                away_xg_for = np.array([xg_proxy(s, t2) for s, t2 in zip(away["AS"].values, away["AST"].values)], dtype=float)
                away_xg_against = np.array([xg_proxy(s, t2) for s, t2 in zip(away["HS"].values, away["HST"].values)], dtype=float)

                if np.isfinite(league["home_xg"]) and np.isfinite(league["away_xg"]) and league["home_xg"] > 0 and league["away_xg"] > 0:
                    hxg = shrink_mean(home_xg_for, league["home_xg"], prior_weight)
                    hxa = shrink_mean(home_xg_against, league["away_xg"], prior_weight)
                    axg = shrink_mean(away_xg_for, league["away_xg"], prior_weight)
                    axa = shrink_mean(away_xg_against, league["home_xg"], prior_weight)

                    tr.home_xg_att = float(hxg / league["home_xg"])
                    tr.home_xg_def = float(hxa / league["away_xg"])
                    tr.away_xg_att = float(axg / league["away_xg"])
                    tr.away_xg_def = float(axa / league["home_xg"])

        # corners multipliers
        if league.get("has_corners"):
            if all(c in df_all.columns for c in ["HC", "AC"]):
                if np.isfinite(league["home_corners"]) and np.isfinite(league["away_corners"]) and league["home_corners"] > 0 and league["away_corners"] > 0:
                    home_cf = shrink_mean(pd.to_numeric(home["HC"], errors="coerce").values, league["home_corners"], prior_weight)
                    home_ca = shrink_mean(pd.to_numeric(home["AC"], errors="coerce").values, league["away_corners"], prior_weight)
                    away_cf = shrink_mean(pd.to_numeric(away["AC"], errors="coerce").values, league["away_corners"], prior_weight)
                    away_ca = shrink_mean(pd.to_numeric(away["HC"], errors="coerce").values, league["home_corners"], prior_weight)

                    tr.home_c_att = float(home_cf / league["home_corners"])
                    tr.home_c_def = float(home_ca / league["away_corners"])
                    tr.away_c_att = float(away_cf / league["away_corners"])
                    tr.away_c_def = float(away_ca / league["home_corners"])

        rates[t] = tr

    return rates

def expected_goals_from_rates(
    rates: Dict[str, TeamRates],
    league: Dict[str, float],
    home: str,
    away: str,
    xg_weight: float,
    elo_diff: float,
    elo_to_goal_k: float,
    manual_home_factor: float,
    manual_away_factor: float,
) -> Tuple[float, float]:
    h = rates.get(home)
    a = rates.get(away)
    if h is None or a is None:
        raise ValueError("Takım verisi yetersiz. min maç düşür veya daha fazla sezon/maç yükle.")

    lam_h_goal = league["home_goals"] * h.home_attack * a.away_def
    lam_a_goal = league["away_goals"] * a.away_attack * h.home_def

    lam_h, lam_a = lam_h_goal, lam_a_goal

    # xG blend (if available)
    if (
        xg_weight > 0
        and h.home_xg_att is not None and a.away_xg_def is not None
        and a.away_xg_att is not None and h.home_xg_def is not None
    ):
        lam_h_xg = league["home_goals"] * float(h.home_xg_att) * float(a.away_xg_def)
        lam_a_xg = league["away_goals"] * float(a.away_xg_att) * float(h.home_xg_def)
        lam_h = (1 - xg_weight) * lam_h_goal + xg_weight * lam_h_xg
        lam_a = (1 - xg_weight) * lam_a_goal + xg_weight * lam_a_xg

    # ELO -> goals scaling
    scale = math.exp((elo_to_goal_k * elo_diff) / 400.0)
    lam_h *= scale
    lam_a *= (1.0 / scale)

    # manual squad/injury factors
    lam_h *= manual_home_factor
    lam_a *= manual_away_factor

    lam_h = float(np.clip(lam_h, 0.05, 4.5))
    lam_a = float(np.clip(lam_a, 0.05, 4.5))
    return lam_h, lam_a

def expected_corners_from_rates(
    rates: Dict[str, TeamRates],
    league: Dict[str, float],
    home: str,
    away: str,
) -> Optional[Tuple[float, float]]:
    if not league.get("has_corners"):
        return None
    h = rates.get(home)
    a = rates.get(away)
    if h is None or a is None:
        return None
    if h.home_c_att is None or a.away_c_def is None or a.away_c_att is None or h.home_c_def is None:
        return None

    lam_hc = league["home_corners"] * float(h.home_c_att) * float(a.away_c_def)
    lam_ac = league["away_corners"] * float(a.away_c_att) * float(h.home_c_def)
    lam_hc = float(np.clip(lam_hc, 0.2, 18.0))
    lam_ac = float(np.clip(lam_ac, 0.2, 18.0))
    return lam_hc, lam_ac

def htft_probs(lam_h: float, lam_a: float, ht_ratio_home: float, ht_ratio_away: float, max_goals: int) -> Dict[str, float]:
    lam_h_ht = max(0.01, lam_h * ht_ratio_home)
    lam_a_ht = max(0.01, lam_a * ht_ratio_away)
    lam_h_2h = max(0.01, lam_h - lam_h_ht)
    lam_a_2h = max(0.01, lam_a - lam_a_ht)

    mat_ht = score_matrix(lam_h_ht, lam_a_ht, max_goals=max_goals)
    mat_2h = score_matrix(lam_h_2h, lam_a_2h, max_goals=max_goals)

    combos = {f"{h}/{f}": 0.0 for h in ["H", "D", "A"] for f in ["H", "D", "A"]}

    for i in range(mat_ht.shape[0]):
        for j in range(mat_ht.shape[1]):
            p_ht = mat_ht[i, j]
            ht_res = "H" if i > j else ("A" if i < j else "D")
            for a in range(mat_2h.shape[0]):
                for b in range(mat_2h.shape[1]):
                    p_2h = mat_2h[a, b]
                    ft_i, ft_j = i + a, j + b
                    ft_res = "H" if ft_i > ft_j else ("A" if ft_i < ft_j else "D")
                    combos[f"{ht_res}/{ft_res}"] += p_ht * p_2h

    return combos

def last_matches_table(df_all: pd.DataFrame, team: str, n: int = 20) -> pd.DataFrame:
    m = df_all[(df_all["HomeTeam"] == team) | (df_all["AwayTeam"] == team)].sort_values("Date").tail(n).copy()
    if m.empty:
        return m
    m["GF"] = np.where(m["HomeTeam"] == team, m["FTHG"], m["FTAG"])
    m["GA"] = np.where(m["HomeTeam"] == team, m["FTAG"], m["FTHG"])
    m["WDL"] = np.where(m["GF"] > m["GA"], "W", np.where(m["GF"] < m["GA"], "L", "D"))
    m["Score"] = m["FTHG"].astype(int).astype(str) + "-" + m["FTAG"].astype(int).astype(str)
    return m[["Date", "Season", "HomeTeam", "AwayTeam", "Score", "GF", "GA", "WDL"]].reset_index(drop=True)

# =========================================================
# SIDEBAR — Single list leagues + auto divs
# =========================================================
with st.sidebar:
    st.header("Ayarlar")

    # auto divs from fixtures
    try:
        auto_divs = discover_divs_from_fixtures()
    except Exception:
        auto_divs = []

    named = dict(LEAGUES)
    existing = set(v.upper() for v in named.values())
    for d in auto_divs:
        if d not in existing:
            named[f"[AUTO] {d}"] = d

    q = st.text_input("Lig ara (isim/div)", value="")
    keys = list(named.keys())
    if q.strip():
        qq = q.strip().lower()
        keys = [k for k in keys if qq in k.lower() or qq in str(named[k]).lower()]
        if not keys:
            keys = list(named.keys())

    league_key = st.selectbox("Lig (tek liste)", keys, index=0)
    div = named[league_key].strip().upper()

    with st.expander("Custom Div (isteğe bağlı)"):
        custom = st.text_input("Div code (örn: E0, SP1, T1, BRA, USA ...)", value="")
        if custom.strip():
            div = custom.strip().upper()

    anchor_date = st.date_input("Analiz tarihi", value=dt.date.today())
    back_seasons = st.slider("Geriye kaç sezon indirilsin (mmz ligler)", 1, 10, 6, 1)

    st.divider()
    lookback = st.slider("Form: son kaç maç (home/away ayrı)", 10, 30, 20, 1)
    min_matches = st.slider("Takım için min maç (home+away)", 4, 20, 8, 1)
    prior_weight = st.slider("Shrinkage (prior ağırlığı)", 0, 30, 10, 1)

    st.divider()
    half_life_days = st.slider("Lig ortalaması yarı-ömür (gün)", 15, 180, 60, 5)
    max_goals = st.slider("Skor matrisi max gol", 4, 10, 6, 1)

    st.divider()
    st.subheader("Model güçlendiriciler")
    xg_weight = st.slider("xG proxy ağırlığı (shots)", 0.0, 1.0, 0.25, 0.05)
    elo_k = st.slider("ELO K", 5.0, 60.0, 20.0, 1.0)
    elo_home_adv = st.slider("ELO home advantage", 0.0, 120.0, 60.0, 5.0)
    glicko_home_adv = st.slider("Glicko home advantage", 0.0, 120.0, 60.0, 5.0)
    elo_to_goal_k = st.slider("ELO->Goal mapping (k)", 0.0, 1.0, 0.25, 0.05)

    st.divider()
    st.subheader("Kadro / Sakatlık / Rotasyon (manuel)")
    manual_home_factor = st.slider("Home attack factor", 0.80, 1.10, 1.00, 0.01)
    manual_away_factor = st.slider("Away attack factor", 0.80, 1.10, 1.00, 0.01)

# =========================================================
# LOAD DATA
# =========================================================
try:
    df_all, seasons_loaded, source_used = load_history(div, anchor_date, back_seasons=back_seasons)
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

fixtures_df = load_fixtures()
league = league_avgs_weighted(df_all, anchor_date, half_life_days=half_life_days)
rates = build_team_rates_recent(df_all, league, lookback=lookback, prior_weight=prior_weight, min_matches=min_matches)
teams = sorted(rates.keys())

if len(teams) < 4:
    st.warning("Yeterli takım verisi oluşmadı. min_matches düşür veya daha fazla data yükle.")
    st.stop()

_, elo_final, glicko_final = compute_ratings(df_all, elo_k=elo_k, elo_home_adv=elo_home_adv, glicko_home_adv=glicko_home_adv)

# =========================================================
# LEAGUE GOALS PER MATCH
# =========================================================
weighted_home = float(league["home_goals"])
weighted_away = float(league["away_goals"])
weighted_total = weighted_home + weighted_away

raw_home = float(league["raw_home_goals"])
raw_away = float(league["raw_away_goals"])
raw_total = float(league["raw_total_goals"])

with st.sidebar:
    st.divider()
    st.subheader("Seçtiğin lig: Gol ortalaması")
    st.metric("Total goals / match (recent-weighted)", f"{weighted_total:.2f}")
    c1, c2 = st.columns(2)
    c1.metric("Home goals", f"{weighted_home:.2f}")
    c2.metric("Away goals", f"{weighted_away:.2f}")
    with st.expander("Raw mean (tüm yüklenen maçlar)"):
        st.write(f"Total goals/match: **{raw_total:.2f}**")
        st.write(f"Home: **{raw_home:.2f}** | Away: **{raw_away:.2f}**")
        st.write(f"Maç sayısı: **{len(df_all)}**")
        st.write(f"Kaynak: **{source_used}** | Seasons: **{', '.join(seasons_loaded[:8])}{'...' if len(seasons_loaded)>8 else ''}**")

st.info(
    f"**Lig ({div}) — Maç başına gol ortalaması (Total): {weighted_total:.2f}**  | "
    f"Home {weighted_home:.2f} / Away {weighted_away:.2f}  |  Source: {source_used}"
)

# =========================================================
# MATCH ANALYZER
# =========================================================
st.subheader("🎯 Match Analyzer")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home", teams, index=0)
with col2:
    away_team = st.selectbox("Away", teams, index=min(1, len(teams) - 1))

if home_team == away_team:
    st.warning("Home ve Away aynı olamaz.")
    st.stop()

elo_home = float(elo_final.get(home_team, 1500.0))
elo_away = float(elo_final.get(away_team, 1500.0))
g_home = float(glicko_final.get(home_team, (1500.0, 350.0))[0])
g_away = float(glicko_final.get(away_team, (1500.0, 350.0))[0])

elo_diff = (elo_home + elo_home_adv) - elo_away
g_diff = (g_home + glicko_home_adv) - g_away

lam_h, lam_a = expected_goals_from_rates(
    rates, league,
    home=home_team, away=away_team,
    xg_weight=xg_weight,
    elo_diff=elo_diff,
    elo_to_goal_k=elo_to_goal_k,
    manual_home_factor=manual_home_factor,
    manual_away_factor=manual_away_factor,
)

mat = score_matrix(lam_h, lam_a, max_goals=max_goals)
pH, pD, pA = outcome_probs_from_mat(mat)

ht_ratio_home = league.get("ht_ratio_home", 0.45)
ht_ratio_away = league.get("ht_ratio_away", 0.45)
mat_ht = score_matrix(max(0.01, lam_h * ht_ratio_home), max(0.01, lam_a * ht_ratio_away), max_goals=max_goals)
pH_ht, pD_ht, pA_ht = outcome_probs_from_mat(mat_ht)

p_over15 = prob_over_from_mat(mat, 1.5)
p_over25 = prob_over_from_mat(mat, 2.5)
p_over35 = prob_over_from_mat(mat, 3.5)
p_btts = prob_btts_from_mat(mat)
p_combo = prob_combo_btts_over25(mat)

p_h_over05 = prob_team_over(mat, "H", 0.5)
p_h_over15 = prob_team_over(mat, "H", 1.5)
p_a_over05 = prob_team_over(mat, "A", 0.5)
p_a_over15 = prob_team_over(mat, "A", 1.5)

ah_lines = [-0.5, +0.5, -1.5, +1.5]
ah_home = {line: prob_handicap_home(mat, line) for line in ah_lines}
ah_away = {line: prob_handicap_away(mat, line) for line in ah_lines}

p_h_wtn = prob_win_to_nil(mat, "H")
p_a_wtn = prob_win_to_nil(mat, "A")

htft = htft_probs(lam_h, lam_a, ht_ratio_home, ht_ratio_away, max_goals=max_goals)

corners = expected_corners_from_rates(rates, league, home_team, away_team)
if corners is not None:
    lam_hc, lam_ac = corners
    mat_c = score_matrix(lam_hc, lam_ac, max_goals=12)
    c_over95 = prob_over_from_mat(mat_c, 9.5)
    c_over105 = prob_over_from_mat(mat_c, 10.5)
else:
    lam_hc = lam_ac = None

with st.expander("📌 Son 20 maç (form)"):
    l, r = st.columns(2)
    with l:
        st.markdown(f"**{home_team}**")
        st.dataframe(last_matches_table(df_all, home_team, n=20), use_container_width=True, height=320)
    with r:
        st.markdown(f"**{away_team}**")
        st.dataframe(last_matches_table(df_all, away_team, n=20), use_container_width=True, height=320)

A, B, C = st.columns([1.15, 1.05, 1.25], vertical_alignment="top")

with A:
    st.markdown("### 1X2 (FT)")
    st.write(f"**H {pH*100:.1f}% | D {pD*100:.1f}% | A {pA*100:.1f}%**")
    st.caption(f"Expected goals: {lam_h:.2f} - {lam_a:.2f}")
    st.caption(f"ELO diff: {elo_diff:+.0f} | Glicko diff: {g_diff:+.0f}")

    st.markdown("### Over/Under")
    st.write(f"Over 1.5: **{p_over15*100:.1f}%** (Fair {1/p_over15:.2f})")
    st.write(f"Over 2.5: **{p_over25*100:.1f}%** (Fair {1/p_over25:.2f})")
    st.write(f"Over 3.5: **{p_over35*100:.1f}%** (Fair {1/p_over35:.2f})")

    st.markdown("### BTTS + Kombolar")
    st.write(f"BTTS (KG Var): **{p_btts*100:.1f}%** (Fair {1/p_btts:.2f})")
    st.write(f"BTTS & Over 2.5: **{p_combo*100:.1f}%** (Fair {1/p_combo:.2f})")

    st.markdown("### Team totals")
    st.write(f"Home over 0.5: **{p_h_over05*100:.1f}%** | Home over 1.5: **{p_h_over15*100:.1f}%**")
    st.write(f"Away over 0.5: **{p_a_over05*100:.1f}%** | Away over 1.5: **{p_a_over15*100:.1f}%**")

    st.markdown("### Win to Nil")
    st.write(f"Home win to nil: **{p_h_wtn*100:.1f}%**")
    st.write(f"Away win to nil: **{p_a_wtn*100:.1f}%**")

with B:
    st.markdown("### İlk Yarı")
    st.write(f"İY 1X2: **H {pH_ht*100:.1f}% | D {pD_ht*100:.1f}% | A {pA_ht*100:.1f}%**")
    st.write(f"İY Over 0.5: **{prob_over_from_mat(mat_ht, 0.5)*100:.1f}%**")
    st.write(f"İY Over 1.5: **{prob_over_from_mat(mat_ht, 1.5)*100:.1f}%**")

    st.markdown("### İY / MS (HT/FT)")
    htft_df = pd.DataFrame({"HT/FT": list(htft.keys()), "Prob": list(htft.values())}).sort_values("Prob", ascending=False)
    htft_df = htft_df.head(9).copy()
    htft_df["FairOdds"] = htft_df["Prob"].apply(lambda p: (1/p) if p > 0 else np.nan)
    st.dataframe(htft_df, use_container_width=True, height=320)

    st.markdown("### Asian Handicap")
    ah_tbl = []
    for line in ah_lines:
        ah_tbl.append(("Home", line, ah_home[line], (1/ah_home[line] if ah_home[line] > 0 else np.nan)))
    for line in ah_lines:
        ah_tbl.append(("Away", line, ah_away[line], (1/ah_away[line] if ah_away[line] > 0 else np.nan)))
    ahdf = pd.DataFrame(ah_tbl, columns=["Side", "Line", "Prob", "FairOdds"])
    ahdf["Prob"] = (ahdf["Prob"] * 100).round(2)
    ahdf["FairOdds"] = ahdf["FairOdds"].round(2)
    st.dataframe(ahdf, use_container_width=True, height=260)

with C:
    st.markdown("### Doğru skor & Total goals")
    st.dataframe(top_scores(mat, topn=10), use_container_width=True, height=300)
    st.dataframe(exact_total_goals_probs(mat, max_total=6), use_container_width=True, height=260)

    st.markdown("### Korner (varsa)")
    if lam_hc is None:
        st.info("Bu lig datasında korner kolonları yok (HC/AC). (Extra liglerde genelde yok.)")
    else:
        st.write(f"Expected corners: **{lam_hc:.2f} - {lam_ac:.2f}** (Total {lam_hc+lam_ac:.2f})")
        st.write(f"Total corners Over 9.5: **{c_over95*100:.1f}%** (Fair {1/c_over95:.2f})")
        st.write(f"Total corners Over 10.5: **{c_over105*100:.1f}%** (Fair {1/c_over105:.2f})")

st.divider()
st.subheader("🧩 Diagnostics")
st.write(f"Div: **{div}** | Source: **{source_used}** | Matches loaded: **{len(df_all)}**")
st.write(f"Seasons: **{', '.join(seasons_loaded[:8])}{'...' if len(seasons_loaded)>8 else ''}**")

if fixtures_df is not None and not fixtures_df.empty:
    with st.expander("Upcoming fixtures (opsiyonel)"):
        f = fixtures_df.copy()
        if "Div" in f.columns:
            f = f[f["Div"].astype(str).str.upper() == div]
        f = f.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date")
        f = f[f["Date"] >= pd.Timestamp(anchor_date)]
        st.dataframe(f.head(25), use_container_width=True, height=420)
