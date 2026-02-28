import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional (for calibration / multinomial logistic)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# =========================================================
# DISCLAIMER
# =========================================================
st.set_page_config(page_title="Pro Match Analytics (Football)", layout="wide")
st.title("⚽ Pro Match Analytics (Football) — Season-independent")
st.caption(
    "Bu uygulama olasılık üretir; kesin tahmin değildir. Bahis/iddia yüksek risklidir. "
    "Eğlence ve araştırma amaçlı kullan."
)

# =========================================================
# CONFIG
# =========================================================
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# football-data.co.uk div codes (çok lig + custom)
LEAGUES = {
    # Turkey
    "Turkey - T1": "T1",

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

    # Others (often available)
    "Austria - Bundesliga (A1)": "A1",
    "Switzerland - Super League (SW1)": "SW1",
    "Denmark - Superliga (DN1)": "DN1",
    "Norway - Eliteserien (NO1)": "NO1",
    "Sweden - Allsvenskan (S1)": "S1",
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
    # futbol sezonu genelde yazın başlar
    return d.year if d.month >= 7 else d.year - 1

def hist_url(season: str, div: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

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

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    from io import StringIO
    txt = r.content.decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(txt))

@st.cache_data(show_spinner=False)
def load_multi_season_history(div: str, anchor_date: dt.date, back_seasons: int = 6) -> Tuple[pd.DataFrame, List[str]]:
    start_year = season_start_year_for_date(anchor_date)
    dfs = []
    seasons_loaded = []
    for y in range(start_year, start_year - back_seasons - 1, -1):
        s = season_code(y)
        url = hist_url(s, div)
        try:
            df = parse_dates(load_csv(url))
            df["Season"] = s
            dfs.append(df)
            seasons_loaded.append(s)
        except requests.HTTPError as e:
            if getattr(e, "response", None) is not None and e.response.status_code == 404:
                continue
            raise

    if not dfs:
        raise ValueError("Bu lig/div için indirilebilir sezon bulunamadı (football-data üzerinde yok olabilir).")

    d = pd.concat(dfs, ignore_index=True)
    d = parse_dates(d)

    need = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = need - set(d.columns)
    if missing:
        raise ValueError(f"Zorunlu kolonlar eksik: {sorted(list(missing))}")

    d = d.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce")
    d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
    d = d.dropna(subset=["FTHG", "FTAG"]).copy()

    d = d[d["Date"] <= pd.Timestamp(anchor_date)].copy()
    d = d.sort_values("Date").drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"], keep="last")

    # numeric columns if exist
    for col in ["HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    return d, seasons_loaded

@st.cache_data(show_spinner=False)
def load_fixtures() -> Optional[pd.DataFrame]:
    try:
        f = parse_dates(load_csv(FIXTURES_URL))
        return f
    except Exception:
        return None

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
    p_home = float(np.tril(mat, -1).sum())  # i>j
    p_away = float(np.triu(mat, 1).sum())   # i<j
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
    # line: 0.5, 1.5 etc
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
    # side "H" or "A"
    p = 0.0
    if side == "H":
        # away goals = 0 and home>away
        j = 0
        for i in range(mat.shape[0]):
            if i > j:
                p += mat[i, j]
    else:
        # home goals = 0 and away>home
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
    # P(home goals - away goals > line)
    p = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (i - j) > line:
                p += mat[i, j]
    return float(p)

def prob_handicap_away(mat: np.ndarray, line: float) -> float:
    # P(away goals - home goals > line)  == P(home-away < -line)
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

# =========================================================
# xG proxy (shots-based)  — if HS/HST exist
# =========================================================
def xg_proxy(shots: float, sot: float) -> float:
    """
    Basit xG proxy: SoT daha değerli.
    0.04*shots + 0.08*sot  (yaklaşık)
    """
    if not np.isfinite(shots) and not np.isfinite(sot):
        return np.nan
    s = shots if np.isfinite(shots) else 0.0
    t = sot if np.isfinite(sot) else 0.0
    return 0.04 * s + 0.08 * t

# =========================================================
# ELO & GLICKO
# =========================================================
def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b) / 400.0))

def elo_update(r_a: float, r_b: float, s_a: float, k: float) -> Tuple[float, float]:
    e_a = elo_expected(r_a, r_b)
    e_b = 1.0 - e_a
    return r_a + k * (s_a - e_a), r_b + k * ((1.0 - s_a) - e_b)

def glicko_g(rd: float) -> float:
    q = math.log(10) / 400.0
    return 1.0 / math.sqrt(1.0 + (3.0 * q * q * rd * rd) / (math.pi * math.pi))

def glicko_E(r: float, rj: float, rdj: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-glicko_g(rdj) * (r - rj) / 400.0))

def glicko_update(r: float, rd: float, rj: float, rdj: float, s: float) -> Tuple[float, float]:
    # Glicko-1 single-opponent update
    q = math.log(10) / 400.0
    g = glicko_g(rdj)
    E = glicko_E(r, rj, rdj)
    d2 = 1.0 / (q * q * g * g * E * (1.0 - E) + 1e-12)
    pre = 1.0 / (1.0 / (rd * rd + 1e-12) + 1.0 / d2)
    r_new = r + q * pre * g * (s - E)
    rd_new = math.sqrt(pre)
    return r_new, rd_new

@st.cache_data(show_spinner=False)
def compute_ratings(df_all: pd.DataFrame, elo_k: float, elo_home_adv: float, glicko_home_adv: float) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Her maç için pre-match ELO/Glicko ratinglerini çıkarır.
    Returns:
      df_r: df_all with columns EloHomePre, EloAwayPre, GlickoHomePre, GlickoAwayPre, GHomeRDPre, GAwayRDPre
      elo_final: team->elo
      glicko_final: team->(r, rd)
    """
    d = df_all.sort_values("Date").copy()

    teams = sorted(set(d["HomeTeam"]).union(set(d["AwayTeam"])))

    # init
    elo = {t: 1500.0 for t in teams}
    gl = {t: (1500.0, 350.0) for t in teams}  # (r, rd)

    elo_home_pre = []
    elo_away_pre = []
    gl_home_pre = []
    gl_away_pre = []
    gl_home_rd_pre = []
    gl_away_rd_pre = []

    for _, row in d.iterrows():
        ht = row["HomeTeam"]
        at = row["AwayTeam"]
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])

        # store pre
        elo_home_pre.append(elo[ht])
        elo_away_pre.append(elo[at])

        gl_home_pre.append(gl[ht][0])
        gl_away_pre.append(gl[at][0])
        gl_home_rd_pre.append(gl[ht][1])
        gl_away_rd_pre.append(gl[at][1])

        # result score s for home: win=1 draw=0.5 lose=0
        s_home = 1.0 if fthg > ftag else (0.0 if fthg < ftag else 0.5)

        # ELO update with home advantage
        rH = elo[ht] + elo_home_adv
        rA = elo[at]
        newH, newA = elo_update(rH, rA, s_home, elo_k)
        # remove adv from stored value for home
        elo[ht] = newH - elo_home_adv
        elo[at] = newA

        # Glicko update (home adv by shifting home rating)
        r_h, rd_h = gl[ht]
        r_a, rd_a = gl[at]
        r_h_adj = r_h + glicko_home_adv
        r_a_adj = r_a

        # update both (each as single-opponent)
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

# =========================================================
# FORM + LEAGUE AVERAGES + SHRINKAGE
# =========================================================
def shrink_mean(values: np.ndarray, prior_mean: float, prior_weight: float) -> float:
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return float(prior_mean)
    return float((v.sum() + prior_weight * prior_mean) / (n + prior_weight))

def league_avgs_weighted(df_all: pd.DataFrame, anchor_date: dt.date, half_life_days: int) -> Dict[str, float]:
    d = df_all.copy()
    last = pd.Timestamp(anchor_date)
    days_ago = (last - d["Date"]).dt.days.clip(lower=0)
    w = np.exp(-days_ago / float(half_life_days))

    def wavg(series: pd.Series) -> float:
        vals = pd.to_numeric(series, errors="coerce")
        m = vals.notna()
        if m.sum() == 0:
            return float("nan")
        return float(np.average(vals[m], weights=w[m.index]))

    out = {
        "home_goals": wavg(d["FTHG"]),
        "away_goals": wavg(d["FTAG"]),
    }

    # HT ratios
    if "HTHG" in d.columns and "HTAG" in d.columns:
        hthg = pd.to_numeric(d["HTHG"], errors="coerce")
        htag = pd.to_numeric(d["HTAG"], errors="coerce")
        if hthg.notna().sum() > 50 and out["home_goals"] > 0 and out["away_goals"] > 0:
            out["ht_ratio_home"] = float(np.clip(wavg(hthg) / out["home_goals"], 0.25, 0.65))
            out["ht_ratio_away"] = float(np.clip(wavg(htag) / out["away_goals"], 0.25, 0.65))
        else:
            out["ht_ratio_home"] = 0.45
            out["ht_ratio_away"] = 0.45
    else:
        out["ht_ratio_home"] = 0.45
        out["ht_ratio_away"] = 0.45

    # corners
    out["has_corners"] = ("HC" in d.columns and "AC" in d.columns)
    if out["has_corners"]:
        out["home_corners"] = wavg(d["HC"])
        out["away_corners"] = wavg(d["AC"])
    else:
        out["home_corners"] = float("nan")
        out["away_corners"] = float("nan")

    # shots/xG proxy support
    out["has_shots"] = ("HS" in d.columns and "AS" in d.columns and "HST" in d.columns and "AST" in d.columns)
    if out["has_shots"]:
        # league avg xG proxy
        hxg = d.apply(lambda r: xg_proxy(r["HS"], r["HST"]), axis=1)
        axg = d.apply(lambda r: xg_proxy(r["AS"], r["AST"]), axis=1)
        out["home_xg"] = wavg(hxg)
        out["away_xg"] = wavg(axg)
    else:
        out["home_xg"] = float("nan")
        out["away_xg"] = float("nan")

    return out

@dataclass
class TeamRates:
    # goal-based
    home_attack: float
    home_def: float
    away_attack: float
    away_def: float
    # xg-proxy based (optional)
    home_xg_att: Optional[float] = None
    home_xg_def: Optional[float] = None
    away_xg_att: Optional[float] = None
    away_xg_def: Optional[float] = None
    # corners (optional)
    home_c_att: Optional[float] = None
    home_c_def: Optional[float] = None
    away_c_att: Optional[float] = None
    away_c_def: Optional[float] = None
    # counts
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

        # goals
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

        # shots -> xG proxy
        if league.get("has_shots") and ("HS" in home.columns) and ("HST" in home.columns):
            home_xg_for = np.array([xg_proxy(s, t) for s, t in zip(home["HS"].values, home["HST"].values)], dtype=float)
            home_xg_against = np.array([xg_proxy(s, t) for s, t in zip(home["AS"].values, home["AST"].values)], dtype=float)

            away_xg_for = np.array([xg_proxy(s, t) for s, t in zip(away["AS"].values, away["AST"].values)], dtype=float)
            away_xg_against = np.array([xg_proxy(s, t) for s, t in zip(away["HS"].values, away["HST"].values)], dtype=float)

            hxg = shrink_mean(home_xg_for, league["home_xg"], prior_weight) if np.isfinite(league["home_xg"]) else np.nan
            hxa = shrink_mean(home_xg_against, league["away_xg"], prior_weight) if np.isfinite(league["away_xg"]) else np.nan
            axg = shrink_mean(away_xg_for, league["away_xg"], prior_weight) if np.isfinite(league["away_xg"]) else np.nan
            axa = shrink_mean(away_xg_against, league["home_xg"], prior_weight) if np.isfinite(league["home_xg"]) else np.nan

            tr.home_xg_att = float(hxg / league["home_xg"]) if np.isfinite(hxg) and league["home_xg"] > 0 else None
            tr.home_xg_def = float(hxa / league["away_xg"]) if np.isfinite(hxa) and league["away_xg"] > 0 else None
            tr.away_xg_att = float(axg / league["away_xg"]) if np.isfinite(axg) and league["away_xg"] > 0 else None
            tr.away_xg_def = float(axa / league["home_xg"]) if np.isfinite(axa) and league["home_xg"] > 0 else None

        # corners
        if league.get("has_corners") and ("HC" in home.columns) and ("AC" in home.columns):
            home_cf = shrink_mean(pd.to_numeric(home["HC"], errors="coerce").values, league["home_corners"], prior_weight)
            home_ca = shrink_mean(pd.to_numeric(home["AC"], errors="coerce").values, league["away_corners"], prior_weight)
            away_cf = shrink_mean(pd.to_numeric(away["AC"], errors="coerce").values, league["away_corners"], prior_weight)
            away_ca = shrink_mean(pd.to_numeric(away["HC"], errors="coerce").values, league["home_corners"], prior_weight)

            tr.home_c_att = float(home_cf / league["home_corners"]) if league["home_corners"] > 0 else None
            tr.home_c_def = float(home_ca / league["away_corners"]) if league["away_corners"] > 0 else None
            tr.away_c_att = float(away_cf / league["away_corners"]) if league["away_corners"] > 0 else None
            tr.away_c_def = float(away_ca / league["home_corners"]) if league["home_corners"] > 0 else None

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
    """
    Combine:
      - goal-based multipliers
      - xG-proxy multipliers (if present)
      - Elo adjustment mapped to goal scaling
      - manual factors (injuries/rotation sliders)
    """
    h = rates.get(home)
    a = rates.get(away)
    if h is None or a is None:
        raise ValueError("Takım verisi yetersiz. min maç düşür veya back_seasons artır.")

    # base: goal model
    lam_h_goal = league["home_goals"] * h.home_attack * a.away_def
    lam_a_goal = league["away_goals"] * a.away_attack * h.home_def

    lam_h = lam_h_goal
    lam_a = lam_a_goal

    # xG-proxy blend (if available)
    if xg_weight > 0 and h.home_xg_att is not None and a.away_xg_def is not None and a.away_xg_att is not None and h.home_xg_def is not None:
        lam_h_xg = league["home_goals"] * float(h.home_xg_att) * float(a.away_xg_def)
        lam_a_xg = league["away_goals"] * float(a.away_xg_att) * float(h.home_xg_def)
        lam_h = (1 - xg_weight) * lam_h_goal + xg_weight * lam_h_xg
        lam_a = (1 - xg_weight) * lam_a_goal + xg_weight * lam_a_xg

    # Elo -> goals scaling (simple exponential)
    # elo_diff positive => home stronger
    scale = math.exp((elo_to_goal_k * elo_diff) / 400.0)
    lam_h *= scale
    lam_a *= (1.0 / scale)

    # manual squad factors
    lam_h *= manual_home_factor
    lam_a *= manual_away_factor

    # clamps
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
# MARKET ODDS HELPERS
# =========================================================
def match_market_odds_from_history(df_all: pd.DataFrame, home: str, away: str, odds_cols: Tuple[str, str, str]) -> Optional[Tuple[float, float, float]]:
    hcol, dcol, acol = odds_cols
    m = df_all[(df_all["HomeTeam"] == home) & (df_all["AwayTeam"] == away)].dropna(subset=[hcol, dcol, acol]).copy()
    if m.empty:
        return None
    row = m.sort_values("Date").iloc[-1]
    o1, ox, o2 = safe_float(row[hcol]), safe_float(row[dcol]), safe_float(row[acol])
    if np.isfinite(o1) and np.isfinite(ox) and np.isfinite(o2):
        return float(o1), float(ox), float(o2)
    return None

def try_fixture_odds(fixtures_df: Optional[pd.DataFrame], div: str, home: str, away: str, odds_cols: Tuple[str, str, str]) -> Optional[Tuple[float, float, float]]:
    if fixtures_df is None or fixtures_df.empty:
        return None
    hcol, dcol, acol = odds_cols
    cols = set(fixtures_df.columns)
    if not (hcol in cols and dcol in cols and acol in cols):
        return None
    f = fixtures_df.copy()
    if "Div" in f.columns:
        f = f[f["Div"].astype(str).str.upper() == div]
    f = f[(f["HomeTeam"] == home) & (f["AwayTeam"] == away)].dropna(subset=[hcol, dcol, acol])
    if f.empty:
        return None
    row = f.sort_values("Date").iloc[0]
    o1, ox, o2 = safe_float(row[hcol]), safe_float(row[dcol]), safe_float(row[acol])
    if np.isfinite(o1) and np.isfinite(ox) and np.isfinite(o2):
        return float(o1), float(ox), float(o2)
    return None

# =========================================================
# CALIBRATION (Platt / Isotonic)
# =========================================================
def multinomial_calibrator_fit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Multinomial logistic regression as calibrator.
    y: 0=H,1=D,2=A
    """
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0
    )
    clf.fit(X, y)
    return clf

def build_calib_features(p_model: np.ndarray, p_mkt: Optional[np.ndarray], elo_diff: float, glicko_diff: float) -> np.ndarray:
    """
    Feature engineering:
      - logits of model ratios
      - logits of market ratios if available
      - rating diffs
    """
    eps = 1e-9
    pH, pD, pA = np.clip(p_model, eps, 1.0)
    # ratios
    f1 = math.log(pH / pA)
    f2 = math.log(pD / (pH + pA))
    feats = [f1, f2, elo_diff / 400.0, glicko_diff / 400.0]

    if p_mkt is not None and np.all(np.isfinite(p_mkt)):
        mH, mD, mA = np.clip(p_mkt, eps, 1.0)
        feats += [math.log(mH / mA), math.log(mD / (mH + mA))]
    else:
        feats += [0.0, 0.0]

    return np.array(feats, dtype=float)

def reliability_bins(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.astype(int), "p": p_pred.astype(float)})
    df = df[df["p"].between(0, 1, inclusive="both")].copy()
    if df.empty:
        return pd.DataFrame(columns=["bin", "p_mean", "y_rate", "n"])
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(
        p_mean=("p", "mean"),
        y_rate=("y", "mean"),
        n=("y", "size"),
    ).reset_index()
    return g

def brier_multiclass(y_idx: np.ndarray, p3: np.ndarray) -> float:
    n = len(y_idx)
    y_onehot = np.zeros_like(p3)
    y_onehot[np.arange(n), y_idx] = 1.0
    return float(np.mean(np.sum((p3 - y_onehot) ** 2, axis=1)))

def logloss_multiclass(y_idx: np.ndarray, p3: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p3[np.arange(len(y_idx)), y_idx], eps, 1.0)
    return float(-np.mean(np.log(p)))

# =========================================================
# BACKTEST (incremental, leakage-free)
# =========================================================
@st.cache_data(show_spinner=False)
def run_backtest(
    df_all: pd.DataFrame,
    div: str,
    lookback: int,
    min_matches: int,
    prior_weight: float,
    half_life_days: int,
    max_goals: int,
    elo_k: float,
    elo_home_adv: float,
    glicko_home_adv: float,
    xg_weight: float,
    elo_to_goal_k: float,
    backtest_matches: int,
    odds_set_name: Optional[str],
) -> pd.DataFrame:
    """
    Incremental backtest over last N matches:
      - build rolling deques for team home/away goals, shots, corners
      - compute rolling league averages (simple EWMA-like using half-life days approx via alpha)
      - compute ELO/Glicko sequentially
      - compute model probabilities + market probs (if odds set exists)
      - store
    """
    d = df_all.sort_values("Date").tail(backtest_matches).copy()

    # odds cols if exist
    odds_cols = get_odds_cols(odds_set_name) if odds_set_name else None
    if odds_cols and not all(c in d.columns for c in odds_cols):
        odds_cols = None

    from collections import deque, defaultdict

    # team deques
    hf = defaultdict(lambda: deque(maxlen=lookback))
    ha = defaultdict(lambda: deque(maxlen=lookback))
    af = defaultdict(lambda: deque(maxlen=lookback))
    aa = defaultdict(lambda: deque(maxlen=lookback))

    # shots/xg proxy
    has_shots = ("HS" in df_all.columns and "AS" in df_all.columns and "HST" in df_all.columns and "AST" in df_all.columns)
    hxf = defaultdict(lambda: deque(maxlen=lookback))
    hxa = defaultdict(lambda: deque(maxlen=lookback))
    axf = defaultdict(lambda: deque(maxlen=lookback))
    axa = defaultdict(lambda: deque(maxlen=lookback))

    # corners
    has_corners = ("HC" in df_all.columns and "AC" in df_all.columns)
    hcf = defaultdict(lambda: deque(maxlen=lookback))
    hca = defaultdict(lambda: deque(maxlen=lookback))
    acf = defaultdict(lambda: deque(maxlen=lookback))
    aca = defaultdict(lambda: deque(maxlen=lookback))

    # league running (EWMA)
    # alpha roughly from half-life: alpha = 1 - exp(ln(0.5)/half_life_in_matches)
    # we don't have half-life in matches, approximate: 1 match/day for simplicity with clamp
    hl = max(10, half_life_days)
    alpha = float(np.clip(1.0 - math.exp(math.log(0.5) / hl), 0.005, 0.05))

    lhg = None
    lag = None
    lhxg = None
    laxg = None
    lhc = None
    lac = None

    # ratings
    teams = sorted(set(df_all["HomeTeam"]).union(set(df_all["AwayTeam"])))
    elo = {t: 1500.0 for t in teams}
    gl = {t: (1500.0, 350.0) for t in teams}

    rows = []

    for _, row in d.iterrows():
        date = row["Date"]
        ht = row["HomeTeam"]
        at = row["AwayTeam"]
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])

        # market probs if odds available
        p_mkt = None
        if odds_cols:
            o1 = safe_float(row.get(odds_cols[0]))
            ox = safe_float(row.get(odds_cols[1]))
            o2 = safe_float(row.get(odds_cols[2]))
            if np.isfinite(o1) and np.isfinite(ox) and np.isfinite(o2) and o1 > 1 and ox > 1 and o2 > 1:
                p_mkt = np.array(implied_probs_1x2(o1, ox, o2), dtype=float)

        # compute prediction using histories BEFORE update
        pred_ok = True

        # init league avgs if None
        if lhg is None:
            lhg = fthg
            lag = ftag
            if has_shots:
                lhxg = xg_proxy(safe_float(row.get("HS")), safe_float(row.get("HST")))
                laxg = xg_proxy(safe_float(row.get("AS")), safe_float(row.get("AST")))
            if has_corners:
                lhc = safe_float(row.get("HC"))
                lac = safe_float(row.get("AC"))
            pred_ok = False  # warm-up

        # check enough team history
        if (len(hf[ht]) + len(ha[ht]) + len(af[at]) + len(aa[at])) < min_matches:
            pred_ok = False

        if pred_ok:
            # build team multipliers (goals)
            home_scored = shrink_mean(np.array(hf[ht], dtype=float), float(lhg), prior_weight)
            home_conc = shrink_mean(np.array(ha[ht], dtype=float), float(lag), prior_weight)
            away_scored = shrink_mean(np.array(af[at], dtype=float), float(lag), prior_weight)
            away_conc = shrink_mean(np.array(aa[at], dtype=float), float(lhg), prior_weight)

            ha_mult = home_scored / float(lhg) if lhg and lhg > 0 else 1.0
            hd_mult = home_conc / float(lag) if lag and lag > 0 else 1.0
            aa_mult = away_scored / float(lag) if lag and lag > 0 else 1.0
            ad_mult = away_conc / float(lhg) if lhg and lhg > 0 else 1.0

            lam_h_goal = float(lhg) * ha_mult * ad_mult
            lam_a_goal = float(lag) * aa_mult * hd_mult

            lam_h = lam_h_goal
            lam_a = lam_a_goal

            # xg blend
            if has_shots and xg_weight > 0 and lhxg is not None and laxg is not None and np.isfinite(lhxg) and np.isfinite(laxg) and lhxg > 0 and laxg > 0:
                # team xg multipliers from deques
                hxg_for = np.array(hxf[ht], dtype=float)
                hxg_ag = np.array(hxa[ht], dtype=float)
                axg_for = np.array(axf[at], dtype=float)
                axg_ag = np.array(axa[at], dtype=float)
                if len(hxg_for) + len(hxg_ag) + len(axg_for) + len(axg_ag) >= min_matches:
                    hxf_m = shrink_mean(hxg_for, float(lhxg), prior_weight)
                    hxa_m = shrink_mean(hxg_ag, float(laxg), prior_weight)
                    axf_m = shrink_mean(axg_for, float(laxg), prior_weight)
                    axa_m = shrink_mean(axg_ag, float(lhxg), prior_weight)

                    lam_h_xg = float(lhg) * (hxf_m / float(lhxg)) * (axa_m / float(lhxg))
                    lam_a_xg = float(lag) * (axf_m / float(laxg)) * (hxa_m / float(laxg))

                    lam_h = (1 - xg_weight) * lam_h_goal + xg_weight * lam_h_xg
                    lam_a = (1 - xg_weight) * lam_a_goal + xg_weight * lam_a_xg

            # rating diffs (pre-match)
            elo_diff = (elo[ht] + elo_home_adv) - elo[at]
            g_diff = (gl[ht][0] + glicko_home_adv) - gl[at][0]

            # elo->goal scaling
            scale = math.exp((elo_to_goal_k * elo_diff) / 400.0)
            lam_h *= scale
            lam_a *= (1.0 / scale)

            lam_h = float(np.clip(lam_h, 0.05, 4.5))
            lam_a = float(np.clip(lam_a, 0.05, 4.5))

            mat = score_matrix(lam_h, lam_a, max_goals=max_goals)
            pH, pD, pA = outcome_probs_from_mat(mat)

            # store
            y = 0 if fthg > ftag else (2 if fthg < ftag else 1)
            rows.append({
                "Date": date,
                "HomeTeam": ht,
                "AwayTeam": at,
                "FTHG": fthg,
                "FTAG": ftag,
                "y": y,
                "pH": pH, "pD": pD, "pA": pA,
                "pOver25": prob_over_from_mat(mat, 2.5),
                "pBTTS": prob_btts_from_mat(mat),
                "elo_diff": elo_diff,
                "glicko_diff": g_diff,
                "mktH": float(p_mkt[0]) if p_mkt is not None and np.all(np.isfinite(p_mkt)) else np.nan,
                "mktD": float(p_mkt[1]) if p_mkt is not None and np.all(np.isfinite(p_mkt)) else np.nan,
                "mktA": float(p_mkt[2]) if p_mkt is not None and np.all(np.isfinite(p_mkt)) else np.nan,
            })

        # UPDATE histories AFTER prediction
        hf[ht].append(fthg)
        ha[ht].append(ftag)
        af[at].append(ftag)
        aa[at].append(fthg)

        # shots/xg proxy history update
        if has_shots:
            hxg_for = xg_proxy(safe_float(row.get("HS")), safe_float(row.get("HST")))
            hxg_ag = xg_proxy(safe_float(row.get("AS")), safe_float(row.get("AST")))
            axg_for_ = xg_proxy(safe_float(row.get("AS")), safe_float(row.get("AST")))
            axg_ag_ = xg_proxy(safe_float(row.get("HS")), safe_float(row.get("HST")))
            if np.isfinite(hxg_for): hxf[ht].append(hxg_for)
            if np.isfinite(hxg_ag):  hxa[ht].append(hxg_ag)
            if np.isfinite(axg_for_): axf[at].append(axg_for_)
            if np.isfinite(axg_ag_):  axa[at].append(axg_ag_)

        # corners update
        if has_corners:
            hc = safe_float(row.get("HC"))
            ac = safe_float(row.get("AC"))
            if np.isfinite(hc): hcf[ht].append(hc)
            if np.isfinite(ac): hca[ht].append(ac)
            if np.isfinite(ac): acf[at].append(ac)
            if np.isfinite(hc): aca[at].append(hc)

        # league EWMA update
        lhg = (1 - alpha) * float(lhg) + alpha * fthg
        lag = (1 - alpha) * float(lag) + alpha * ftag
        if has_shots:
            hxg = xg_proxy(safe_float(row.get("HS")), safe_float(row.get("HST")))
            axg = xg_proxy(safe_float(row.get("AS")), safe_float(row.get("AST")))
            if lhxg is None: lhxg = hxg
            if laxg is None: laxg = axg
            if np.isfinite(hxg): lhxg = (1 - alpha) * float(lhxg) + alpha * float(hxg)
            if np.isfinite(axg): laxg = (1 - alpha) * float(laxg) + alpha * float(axg)
        if has_corners:
            hc = safe_float(row.get("HC"))
            ac = safe_float(row.get("AC"))
            if lhc is None: lhc = hc
            if lac is None: lac = ac
            if np.isfinite(hc): lhc = (1 - alpha) * float(lhc) + alpha * float(hc)
            if np.isfinite(ac): lac = (1 - alpha) * float(lac) + alpha * float(ac)

        # rating updates
        s_home = 1.0 if fthg > ftag else (0.0 if fthg < ftag else 0.5)
        # ELO
        rH = elo[ht] + elo_home_adv
        rA = elo[at]
        newH, newA = elo_update(rH, rA, s_home, elo_k)
        elo[ht] = newH - elo_home_adv
        elo[at] = newA
        # Glicko
        r_h, rd_h = gl[ht]
        r_a, rd_a = gl[at]
        r_h_adj = r_h + glicko_home_adv
        r_a_adj = r_a
        r_h_new, rd_h_new = glicko_update(r_h_adj, rd_h, r_a_adj, rd_a, s_home)
        r_a_new, rd_a_new = glicko_update(r_a_adj, rd_a, r_h_adj, rd_h, 1.0 - s_home)
        gl[ht] = (r_h_new - glicko_home_adv, rd_h_new)
        gl[at] = (r_a_new, rd_a_new)

    return pd.DataFrame(rows)

# =========================================================
# UI - SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Ayarlar")

    league_mode = st.radio("Lig seçimi", ["Listeden", "Custom Div code"], horizontal=True)
    if league_mode == "Listeden":
        league_name = st.selectbox("Lig", list(LEAGUES.keys()), index=0)
        div = LEAGUES[league_name]
    else:
        div = st.text_input("Div code (örn: E0, SP1, I1, T1 ...)", value="E0").strip().upper()
        league_name = f"Custom ({div})"

    anchor_date = st.date_input("Analiz tarihi", value=dt.date.today())
    back_seasons = st.slider("Geriye kaç sezon indirilsin", 1, 10, 6, 1)

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
    st.caption("Örn: Home kadro eksikse 0.95, rakip eksikse 1.05 gibi.")

    st.divider()
    st.subheader("Market odds")
    odds_source = st.radio("Odds kaynağı", ["Dataset (football-data)", "CSV Upload", "Manual", "None"], index=0)
    blend_w = st.slider("Model vs Market blend (0=Model, 1=Market)", 0.0, 1.0, 0.25, 0.05)
    use_calibration = st.checkbox("Auto-calibration (multinomial LR)", value=True)
    st.caption("Market bilgisi güçlüdür; blend + calibration genelde performansı artırır.")

    st.divider()
    st.subheader("Backtest")
    bt_matches = st.slider("Backtest son kaç maç", 200, 2000, 800, 100)
    bt_run = st.button("Backtest çalıştır", use_container_width=True)

# =========================================================
# LOAD DATA
# =========================================================
try:
    df_all, seasons_loaded = load_multi_season_history(div, anchor_date, back_seasons=back_seasons)
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

fixtures_df = load_fixtures()

league = league_avgs_weighted(df_all, anchor_date, half_life_days=half_life_days)
rates = build_team_rates_recent(df_all, league, lookback=lookback, prior_weight=prior_weight, min_matches=min_matches)
teams = sorted(rates.keys())

if len(teams) < 4:
    st.warning("Yeterli takım verisi oluşmadı. min_matches düşür veya back_seasons artır.")
    st.stop()

df_ratings, elo_final, glicko_final = compute_ratings(df_all, elo_k=elo_k, elo_home_adv=elo_home_adv, glicko_home_adv=glicko_home_adv)

# =========================================================
# TABS
# =========================================================
tabs = st.tabs(["🎯 Match Analyzer", "📈 Backtest + Calibration", "🧩 Data & Odds"])

# =========================================================
# TAB: MATCH ANALYZER
# =========================================================
with tabs[0]:
    st.subheader("Maç Analizi")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home", teams, index=0)
    with col2:
        away_team = st.selectbox("Away", teams, index=min(1, len(teams) - 1))

    if home_team == away_team:
        st.warning("Home ve Away aynı olamaz.")
        st.stop()

    # ratings pre-match from last known state in df_ratings (approx using finals is OK for display),
    # but better: take latest pre ratings for teams from df_ratings
    last_h = df_ratings[df_ratings["HomeTeam"] == home_team].tail(1)
    last_a = df_ratings[df_ratings["AwayTeam"] == away_team].tail(1)

    # fallback to finals
    elo_home = float(elo_final.get(home_team, 1500.0))
    elo_away = float(elo_final.get(away_team, 1500.0))
    g_home = float(glicko_final.get(home_team, (1500.0, 350.0))[0])
    g_away = float(glicko_final.get(away_team, (1500.0, 350.0))[0])

    elo_diff = (elo_home + elo_home_adv) - elo_away
    g_diff = (g_home + glicko_home_adv) - g_away

    # expected goals
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

    # First half
    ht_ratio_home = league.get("ht_ratio_home", 0.45)
    ht_ratio_away = league.get("ht_ratio_away", 0.45)
    mat_ht = score_matrix(max(0.01, lam_h * ht_ratio_home), max(0.01, lam_a * ht_ratio_away), max_goals=max_goals)
    pH_ht, pD_ht, pA_ht = outcome_probs_from_mat(mat_ht)

    # markets
    p_over15 = prob_over_from_mat(mat, 1.5)
    p_over25 = prob_over_from_mat(mat, 2.5)
    p_over35 = prob_over_from_mat(mat, 3.5)
    p_btts = prob_btts_from_mat(mat)
    p_combo = prob_combo_btts_over25(mat)

    p_h_over05 = prob_team_over(mat, "H", 0.5)
    p_h_over15 = prob_team_over(mat, "H", 1.5)
    p_a_over05 = prob_team_over(mat, "A", 0.5)
    p_a_over15 = prob_team_over(mat, "A", 1.5)

    # asian handicap probabilities
    ah_lines = [-0.5, +0.5, -1.5, +1.5]
    ah_home = {line: prob_handicap_home(mat, line) for line in ah_lines}
    ah_away = {line: prob_handicap_away(mat, line) for line in ah_lines}

    # win to nil
    p_h_wtn = prob_win_to_nil(mat, "H")
    p_a_wtn = prob_win_to_nil(mat, "A")

    # ht/ft
    htft = htft_probs(lam_h, lam_a, ht_ratio_home, ht_ratio_away, max_goals=max_goals)

    # corners
    corners = expected_corners_from_rates(rates, league, home_team, away_team)
    if corners is not None:
        lam_hc, lam_ac = corners
        mat_c = score_matrix(lam_hc, lam_ac, max_goals=12)
        # team corners totals
        p_hc_over45 = prob_team_over(mat_c, "H", 4.5)
        p_ac_over35 = prob_team_over(mat_c, "A", 3.5)
        # corners handicap (home -1.5 etc)
        c_home_m15 = prob_handicap_home(mat_c, -1.5)
        c_away_m15 = prob_handicap_away(mat_c, -1.5)
        c_over95 = prob_over_from_mat(mat_c, 9.5)
        c_over105 = prob_over_from_mat(mat_c, 10.5)
    else:
        lam_hc = lam_ac = None

    # =========================================================
    # MARKET ODDS (1X2) + ENSEMBLE + CALIBRATION
    # =========================================================
    market_odds = None
    p_mkt = None
    chosen_set = None

    if odds_source == "Dataset (football-data)":
        avail_sets = find_available_odds_sets(df_all)
        if avail_sets:
            chosen_set = st.selectbox("Odds set", avail_sets, index=0)
            cols = get_odds_cols(chosen_set)
            market_odds = try_fixture_odds(fixtures_df, div, home_team, away_team, cols) if cols else None
            if market_odds is None and cols:
                market_odds = match_market_odds_from_history(df_all, home_team, away_team, cols)
            if market_odds is not None:
                p_mkt = np.array(implied_probs_1x2(*market_odds), dtype=float)
        else:
            st.info("Bu lig dosyasında odds kolonları yok. (CSV Upload / Manual deneyebilirsin.)")

    elif odds_source == "CSV Upload":
        st.caption("CSV format: Date,Div,HomeTeam,AwayTeam,H,D,A")
        up = st.file_uploader("Odds CSV yükle", type=["csv"])
        if up is not None:
            try:
                mdf = pd.read_csv(up)
                if "Date" in mdf.columns:
                    mdf["Date"] = pd.to_datetime(mdf["Date"], dayfirst=True, errors="coerce")
                for c in ["H", "D", "A"]:
                    mdf[c] = pd.to_numeric(mdf[c], errors="coerce")
                tmp = mdf.copy()
                if "Div" in tmp.columns:
                    tmp = tmp[tmp["Div"].astype(str).str.upper() == div]
                tmp = tmp[(tmp["HomeTeam"] == home_team) & (tmp["AwayTeam"] == away_team)].dropna(subset=["H", "D", "A"])
                if not tmp.empty:
                    row = tmp.sort_values("Date").iloc[-1]
                    market_odds = (float(row["H"]), float(row["D"]), float(row["A"]))
                    p_mkt = np.array(implied_probs_1x2(*market_odds), dtype=float)
            except Exception as e:
                st.error(f"CSV okunamadı: {e}")

    elif odds_source == "Manual":
        o1 = st.number_input("Home odds", min_value=1.01, value=2.10, step=0.01)
        ox = st.number_input("Draw odds", min_value=1.01, value=3.20, step=0.01)
        o2 = st.number_input("Away odds", min_value=1.01, value=3.50, step=0.01)
        market_odds = (float(o1), float(ox), float(o2))
        p_mkt = np.array(implied_probs_1x2(*market_odds), dtype=float)

    p_model = np.array([pH, pD, pA], dtype=float)
    if p_mkt is not None and np.all(np.isfinite(p_mkt)):
        p_blend = (1 - blend_w) * p_model + blend_w * p_mkt
        p_blend = p_blend / p_blend.sum()
    else:
        p_blend = p_model

    # show form tables
    with st.expander("📌 Son 20 maç (form)"):
        l, r = st.columns(2)
        with l:
            st.markdown(f"**{home_team}**")
            st.dataframe(last_matches_table(df_all, home_team, n=20), use_container_width=True, height=320)
        with r:
            st.markdown(f"**{away_team}**")
            st.dataframe(last_matches_table(df_all, away_team, n=20), use_container_width=True, height=320)

    # layout
    A, B, C = st.columns([1.15, 1.05, 1.25], vertical_alignment="top")

    with A:
        st.markdown("### 1X2 (FT)")
        st.write(f"Model: **H {p_model[0]*100:.1f}% | D {p_model[1]*100:.1f}% | A {p_model[2]*100:.1f}%**")
        st.write(f"Blend: **H {p_blend[0]*100:.1f}% | D {p_blend[1]*100:.1f}% | A {p_blend[2]*100:.1f}%**")
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

        st.markdown("### Market Odds → Edge / EV (1X2)")
        if market_odds is None or p_mkt is None or not np.all(np.isfinite(p_mkt)):
            st.info("Odds bağlı değil. Soldan Odds kaynağı seç.")
        else:
            st.write(f"Odds set: **{chosen_set or odds_source}**")
            st.write(f"Odds: **{market_odds[0]:.2f} / {market_odds[1]:.2f} / {market_odds[2]:.2f}**")
            st.write(f"Market prob (vig removed): **{p_mkt[0]*100:.1f}% / {p_mkt[1]*100:.1f}% / {p_mkt[2]*100:.1f}%**")

            edge = p_blend - p_mkt
            ev = p_blend * np.array(market_odds) - 1.0

            out = pd.DataFrame({
                "Outcome": ["Home", "Draw", "Away"],
                "P_blend": p_blend,
                "P_market": p_mkt,
                "Edge": edge,
                "Odds": market_odds,
                "EV(p*odds-1)": ev,
                "FairOdds(blend)": 1.0 / np.clip(p_blend, 1e-12, 1.0),
            })
            out["P_blend"] = (out["P_blend"] * 100).round(2)
            out["P_market"] = (out["P_market"] * 100).round(2)
            out["Edge"] = (out["Edge"] * 100).round(2)
            out["EV(p*odds-1)"] = out["EV(p*odds-1)"].round(3)
            out["FairOdds(blend)"] = out["FairOdds(blend)"].round(2)
            st.dataframe(out, use_container_width=True, height=240)

        st.markdown("### Korner (varsa)")
        if lam_hc is None:
            st.info("Bu lig datasında korner kolonları yok (HC/AC).")
        else:
            st.write(f"Expected corners: **{lam_hc:.2f} - {lam_ac:.2f}** (Total {lam_hc+lam_ac:.2f})")
            st.write(f"Total corners Over 9.5: **{c_over95*100:.1f}%** (Fair {1/c_over95:.2f})")
            st.write(f"Total corners Over 10.5: **{c_over105*100:.1f}%** (Fair {1/c_over105:.2f})")
            st.write(f"Home corners over 4.5: **{p_hc_over45*100:.1f}%**")
            st.write(f"Away corners over 3.5: **{p_ac_over35*100:.1f}%**")
            st.write(f"Corners Hcap Home -1.5 (P): **{c_home_m15*100:.1f}%**")
            st.write(f"Corners Hcap Away -1.5 (P): **{c_away_m15*100:.1f}%**")

# =========================================================
# TAB: BACKTEST + CALIBRATION
# =========================================================
with tabs[1]:
    st.subheader("Backtest + Calibration")

    # choose odds set for backtest if dataset odds used
    avail_sets = find_available_odds_sets(df_all)
    odds_set_for_bt = None
    if avail_sets:
        odds_set_for_bt = st.selectbox("Backtest odds set (opsiyonel)", ["None"] + avail_sets, index=0)
        if odds_set_for_bt == "None":
            odds_set_for_bt = None
    else:
        st.info("Bu lig datasında odds kolonları yoksa calibration 'market feature' kısmı sınırlı olur.")

    if bt_run:
        bt = run_backtest(
            df_all=df_all,
            div=div,
            lookback=lookback,
            min_matches=min_matches,
            prior_weight=prior_weight,
            half_life_days=half_life_days,
            max_goals=max_goals,
            elo_k=elo_k,
            elo_home_adv=elo_home_adv,
            glicko_home_adv=glicko_home_adv,
            xg_weight=xg_weight,
            elo_to_goal_k=elo_to_goal_k,
            backtest_matches=bt_matches,
            odds_set_name=odds_set_for_bt,
        )

        if bt.empty or len(bt) < 200:
            st.warning("Backtest sonucu az çıktı. bt_matches artır veya back_seasons artır.")
            st.stop()

        y = bt["y"].values.astype(int)
        p3 = bt[["pH", "pD", "pA"]].values.astype(float)

        brier = brier_multiclass(y, p3)
        ll = logloss_multiclass(y, p3)

        a, b, c = st.columns(3)
        a.metric("Matches evaluated", f"{len(bt)}")
        b.metric("Brier (1X2)", f"{brier:.4f}")
        c.metric("LogLoss (1X2)", f"{ll:.4f}")

        # binary brier for O2.5 and BTTS
        y_over = ((bt["FTHG"] + bt["FTAG"]) > 2.5).astype(int).values
        p_over = bt["pOver25"].values
        brier_over = float(np.mean((p_over - y_over) ** 2))

        y_btts = ((bt["FTHG"] > 0) & (bt["FTAG"] > 0)).astype(int).values
        p_btts = bt["pBTTS"].values
        brier_btts = float(np.mean((p_btts - y_btts) ** 2))

        d1, d2 = st.columns(2)
        d1.metric("Brier (Over2.5)", f"{brier_over:.4f}")
        d2.metric("Brier (BTTS)", f"{brier_btts:.4f}")

        st.divider()

        # Calibration / reliability
        st.markdown("### Calibration (Reliability)")
        cal_target = st.selectbox("Hangi olasılık için calibration?", ["Home Win (pH)", "Draw (pD)", "Away Win (pA)", "Over 2.5 (pOver25)", "BTTS (pBTTS)"])

        if cal_target.startswith("Home"):
            y_true = (y == 0).astype(int)
            p_pred = bt["pH"].values
        elif cal_target.startswith("Draw"):
            y_true = (y == 1).astype(int)
            p_pred = bt["pD"].values
        elif cal_target.startswith("Away"):
            y_true = (y == 2).astype(int)
            p_pred = bt["pA"].values
        elif cal_target.startswith("Over"):
            y_true = y_over
            p_pred = p_over
        else:
            y_true = y_btts
            p_pred = p_btts

        cal = reliability_bins(y_true, p_pred, n_bins=10)
        st.dataframe(cal, use_container_width=True)
        chart_df = cal[["p_mean", "y_rate"]].dropna().rename(columns={"p_mean": "Predicted", "y_rate": "Observed"})
        st.line_chart(chart_df, use_container_width=True)

        st.divider()

        # Auto-calibration model (multinomial LR) using features
        st.markdown("### Auto-calibration (multinomial LR)")
        st.caption("Amaç: Model olasılıklarını (ve varsa market olasılıklarını + rating farklarını) kalibre etmek.")

        # build feature matrix
        X = []
        for i in range(len(bt)):
            p_model = bt.loc[bt.index[i], ["pH", "pD", "pA"]].values.astype(float)
            mkt = bt.loc[bt.index[i], ["mktH", "mktD", "mktA"]].values.astype(float)
            p_mkt = mkt if np.all(np.isfinite(mkt)) else None
            X.append(build_calib_features(p_model, p_mkt, float(bt.iloc[i]["elo_diff"]), float(bt.iloc[i]["glicko_diff"])))
        X = np.vstack(X)
        y_mc = y

        # train/test split (time-based)
        split = int(len(bt) * 0.7)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y_mc[:split], y_mc[split:]

        try:
            clf = multinomial_calibrator_fit(X_tr, y_tr)
            p_cal = clf.predict_proba(X_te)
            brier_cal = brier_multiclass(y_te, p_cal)
            ll_cal = logloss_multiclass(y_te, p_cal)

            e1, e2 = st.columns(2)
            e1.metric("Calibrated Brier (test)", f"{brier_cal:.4f}")
            e2.metric("Calibrated LogLoss (test)", f"{ll_cal:.4f}")

            st.caption("Not: Market odds yoksa market feature'ları 0 olur; yine de rating+model ile kalibrasyon yapılır.")
        except Exception as e:
            st.warning(f"Calibration fit edilemedi: {e}")

    else:
        st.info("Backtest için soldan **Backtest çalıştır** butonuna bas.")

# =========================================================
# TAB: DATA & ODDS
# =========================================================
with tabs[2]:
    st.subheader("Data & Odds Diagnoser")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Div", div)
    c2.metric("Seasons loaded", str(len(seasons_loaded)))
    c3.metric("Has corners", "Yes" if league.get("has_corners") else "No")
    c4.metric("Has shots", "Yes" if league.get("has_shots") else "No")

    st.write(f"Loaded seasons: **{', '.join(seasons_loaded[:6])}{'...' if len(seasons_loaded) > 6 else ''}**")
    st.write(f"Lig avg goals: Home {league['home_goals']:.2f} | Away {league['away_goals']:.2f}")
    st.write(f"HT ratio: Home {league['ht_ratio_home']:.2f} | Away {league['ht_ratio_away']:.2f}")

    st.divider()
    st.markdown("### Mevcut 1X2 odds setleri (dataset)")
    avail = find_available_odds_sets(df_all)
    if avail:
        st.success("Bulunan odds setleri: " + ", ".join(avail))
    else:
        st.info("Bu lig datasında odds kolonları yok.")

    st.divider()
    st.markdown("### Fixtures (yaklaşan maçlar) (opsiyonel)")
    if fixtures_df is None or fixtures_df.empty:
        st.info("fixtures.csv çekilemedi.")
    else:
        f = fixtures_df.copy()
        if "Div" in f.columns:
            f = f[f["Div"].astype(str).str.upper() == div]
        f = f.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date")
        f = f[f["Date"] >= pd.Timestamp(anchor_date)]
        st.dataframe(f.head(25), use_container_width=True, height=420)

    st.divider()
    st.markdown("### Odds CSV Template (iddaa gibi manuel bağlamak için)")
    st.code(
        "Date,Div,HomeTeam,AwayTeam,H,D,A\n"
        "2026-02-28,T1,TeamA,TeamB,2.10,3.20,3.50\n",
        language="text"
    )
    st.caption("Team isimleri dataset ile birebir eşleşmeli.")
