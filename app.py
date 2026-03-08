import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="Pro Match Analytics (Football) - Enhanced", layout="wide")
st.title("⚽ Profesyonel Maç Analizi (Geliştirilmiş)")
st.caption(
    "Bu uygulama olasılık üretir; kesin tahmin değildir. Bahis/iddia yüksek risklidir. "
    "Eğlence ve araştırma amaçlı kullan."
)

# =========================================================
# SOURCES
# =========================================================
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
NEW_LEAGUE_URL = "https://www.football-data.co.uk/new/{div}.csv"

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
    # Extra leagues (football-data 'All new countries' /new/{div}.csv)
    "USA - MLS (USA)": "USA",
    "Argentina - Primera División (ARG)": "ARG",
    "Brazil - Serie A (BRA)": "BRA",
    "Mexico - Liga MX (MEX)": "MEX",
    "Austria - Bundesliga (AUT)": "AUT",
    "China - Super League (CHN)": "CHN",
    "Denmark - Superliga (DNK)": "DNK",
    "Finland - Veikkausliiga (FIN)": "FIN",
    "Ireland - Premier Division (IRL)": "IRL",
    "Japan - J-League (JPN)": "JPN",
    "Norway - Eliteserien (NOR)": "NOR",
    "Poland - Ekstraklasa (POL)": "POL",
    "Romania - Liga 1 (ROU)": "ROU",
    "Russia - Premier League (RUS)": "RUS",
    "Sweden - Allsvenskan (SWE)": "SWE",
    "Switzerland - Super League (SWZ)": "SWZ",
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
# UTILS (original functions, with some additions)
# =========================================================
def season_code(start_year: int) -> str:
    y1 = start_year % 100
    y2 = (start_year + 1) % 100
    return f"{y1:02d}{y2:02d}"

def season_start_year_for_date(d: dt.date) -> int:
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

def _normalize_history_v2(df: pd.DataFrame, anchor_date: dt.date) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).replace("\ufeff", "").strip() for c in d.columns]
    d = parse_dates(d)

    lower_map = {str(c).strip().lower(): c for c in d.columns}

    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            key = c.strip().lower()
            if key in lower_map:
                return lower_map[key]
        return None

    col_date = pick("Date", "MatchDate", "Match Date")
    col_home = pick("HomeTeam", "Home Team", "Home")
    col_away = pick("AwayTeam", "Away Team", "Away")
    col_fthg = pick("FTHG", "HG", "HomeGoals", "Home Goals", "FTHome")
    col_ftag = pick("FTAG", "AG", "AwayGoals", "Away Goals", "FTAway")

    if not col_date:
        raise ValueError("Zorunlu kolon eksik: Date")
    if not col_home:
        raise ValueError("Zorunlu kolon eksik: HomeTeam/Home")
    if not col_away:
        raise ValueError("Zorunlu kolon eksik: AwayTeam/Away")
    if not col_fthg:
        raise ValueError("Zorunlu kolon eksik: FTHG/HG")
    if not col_ftag:
        raise ValueError("Zorunlu kolon eksik: FTAG/AG")

    d = d.rename(columns={
        col_date: "Date",
        col_home: "HomeTeam",
        col_away: "AwayTeam",
        col_fthg: "FTHG",
        col_ftag: "FTAG",
    })

    d = d.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    d["FTHG"] = pd.to_numeric(d["FTHG"], errors="coerce")
    d["FTAG"] = pd.to_numeric(d["FTAG"], errors="coerce")
    d = d.dropna(subset=["FTHG", "FTAG"]).copy()

    for col in ["HTHG", "HTAG", "HS", "AS", "HST", "AST", "HC", "AC"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    d = d[d["Date"] <= pd.Timestamp(anchor_date)].copy()
    d = d.sort_values("Date").drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
        keep="last",
    )
    return d

@st.cache_data(show_spinner=False)
def load_history(div: str, anchor_date: dt.date, back_seasons: int = 6) -> Tuple[pd.DataFrame, List[str], str]:
    start_year = season_start_year_for_date(anchor_date)
    dfs = []
    seasons_loaded = []

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
            continue
        except Exception:
            continue

    if dfs:
        d = pd.concat(dfs, ignore_index=True)
        d = _normalize_history_v2(d, anchor_date)
        return d, seasons_loaded, "mmz"

    new_url = NEW_LEAGUE_URL.format(div=div)
    try:
        df = load_csv(new_url)
        df["Season"] = "NEW"
        d = _normalize_history_v2(df, anchor_date)
        return d, ["NEW"], "new"
    except Exception as e:
        raise ValueError(f"Data indirilemedi: {e}")

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

# xG proxy (shots-based) with learnable coefficients
def estimate_xg_coefficients(df: pd.DataFrame) -> Tuple[float, float]:
    df2 = df.dropna(subset=["HS", "HST", "FTHG", "AS", "AST", "FTAG"]).copy()
    if len(df2) < 50:
        return 0.04, 0.08
    X_h = np.column_stack([df2["HS"].values, df2["HST"].values])
    y_h = df2["FTHG"].values
    X_a = np.column_stack([df2["AS"].values, df2["AST"].values])
    y_a = df2["FTAG"].values
    X = np.vstack([X_h, X_a])
    y = np.concatenate([y_h, y_a])
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeff
    return float(np.clip(a, 0.01, 0.1)), float(np.clip(b, 0.01, 0.2))

def xg_proxy_enhanced(shots: float, sot: float, a: float, b: float) -> float:
    if not np.isfinite(shots) and not np.isfinite(sot):
        return np.nan
    s = shots if np.isfinite(shots) else 0.0
    t = sot if np.isfinite(sot) else 0.0
    return a * s + b * t

# ELO
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

        rH = elo[ht] + elo_home_adv
        rA = elo[at]
        newH, newA = elo_update(rH, rA, s_home, elo_k)
        elo[ht] = newH - elo_home_adv
        elo[at] = newA

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
    return out

# =========================================================
# NEW FEATURES: Rest days, head-to-head, Dixon-Coles, decay
# =========================================================

# Define TeamRates dataclass
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

def shrink_mean_with_prior(weighted_avg: float, prior_mean: float, prior_weight: float, n: int) -> float:
    if np.isnan(weighted_avg):
        return prior_mean
    return (weighted_avg * n + prior_weight * prior_mean) / (n + prior_weight)

def compute_rest_days(team: str, match_date: pd.Timestamp, all_matches: pd.DataFrame) -> int:
    prev = all_matches[
        (all_matches["Date"] < match_date) &
        ((all_matches["HomeTeam"] == team) | (all_matches["AwayTeam"] == team))
    ].sort_values("Date").tail(1)
    if prev.empty:
        return 7
    last_date = prev.iloc[0]["Date"]
    return (match_date - last_date).days

def h2h_features(home: str, away: str, all_matches: pd.DataFrame, n_last: int = 5) -> Tuple[float, float]:
    h2h = all_matches[
        ((all_matches["HomeTeam"] == home) & (all_matches["AwayTeam"] == away)) |
        ((all_matches["HomeTeam"] == away) & (all_matches["AwayTeam"] == home))
    ].sort_values("Date").tail(n_last)
    if h2h.empty:
        return np.nan, np.nan
    home_goals = []
    away_goals = []
    for _, row in h2h.iterrows():
        if row["HomeTeam"] == home:
            home_goals.append(row["FTHG"])
            away_goals.append(row["FTAG"])
        else:
            home_goals.append(row["FTAG"])
            away_goals.append(row["FTHG"])
    return float(np.mean(home_goals)), float(np.mean(away_goals))

def dixon_coles_rho(df: pd.DataFrame) -> float:
    if len(df) < 100:
        return -0.1
    total = len(df)
    low_draws = df[(df["FTHG"] == df["FTAG"]) & (df["FTHG"] <= 1)]
    observed = len(low_draws) / total
    avg_h = df["FTHG"].mean()
    avg_a = df["FTAG"].mean()
    p00 = math.exp(-avg_h) * math.exp(-avg_a)
    p11 = (avg_h * math.exp(-avg_h)) * (avg_a * math.exp(-avg_a))
    expected = p00 + p11
    rho = (observed / expected) - 1.0
    return float(np.clip(rho, -0.3, 0.3))

def dixon_coles_adjustment(mat: np.ndarray, lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    adj = np.ones_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == 0 and j == 0:
                adj[i, j] = 1 - rho * lam_h * lam_a
            elif i == 0 and j == 1:
                adj[i, j] = 1 + rho * lam_h
            elif i == 1 and j == 0:
                adj[i, j] = 1 + rho * lam_a
            elif i == 1 and j == 1:
                adj[i, j] = 1 - rho
            else:
                adj[i, j] = 1.0
    return mat * adj

def build_team_rates_recent_decayed(
    df_all: pd.DataFrame,
    league: Dict[str, float],
    lookback: int,
    prior_weight: float,
    min_matches: int,
    decay_halflife: float
) -> Dict[str, TeamRates]:
    teams = sorted(set(df_all["HomeTeam"]).union(set(df_all["AwayTeam"])))
    rates: Dict[str, TeamRates] = {}

    for t in teams:
        home = df_all[df_all["HomeTeam"] == t].tail(lookback).copy()
        away = df_all[df_all["AwayTeam"] == t].tail(lookback).copy()

        if (len(home) + len(away)) < min_matches:
            continue

        def decay_weights(n):
            if n == 0:
                return np.array([])
            positions = np.arange(n)
            w = np.exp(-positions / decay_halflife)
            return w / w.sum()

        w_home = decay_weights(len(home))
        w_away = decay_weights(len(away))

        def wavg(series, weights):
            vals = series.values.astype(float)
            if len(vals) == 0 or weights.sum() == 0:
                return np.nan
            return float(np.average(vals, weights=weights))

        home_scored = wavg(home["FTHG"], w_home) if len(home) > 0 else np.nan
        home_conceded = wavg(home["FTAG"], w_home) if len(home) > 0 else np.nan
        away_scored = wavg(away["FTAG"], w_away) if len(away) > 0 else np.nan
        away_conceded = wavg(away["FTHG"], w_away) if len(away) > 0 else np.nan

        home_scored = shrink_mean_with_prior(home_scored, league["home_goals"], prior_weight, len(home))
        home_conceded = shrink_mean_with_prior(home_conceded, league["away_goals"], prior_weight, len(home))
        away_scored = shrink_mean_with_prior(away_scored, league["away_goals"], prior_weight, len(away))
        away_conceded = shrink_mean_with_prior(away_conceded, league["home_goals"], prior_weight, len(away))

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
        rates[t] = tr
    return rates

def expected_goals_from_rates_enhanced(
    rates: Dict[str, TeamRates],
    league: Dict[str, float],
    home: str,
    away: str,
    xg_weight: float,
    elo_diff: float,
    elo_to_goal_k: float,
    manual_home_factor: float,
    manual_away_factor: float,
    rest_days_diff: float,
    h2h_home_avg: float,
    h2h_away_avg: float,
    xg_a: float,
    xg_b: float,
    rest_factor_strength: float = 0.05,
    h2h_weight: float = 0.15
) -> Tuple[float, float]:
    h = rates.get(home)
    a = rates.get(away)
    if h is None or a is None:
        raise ValueError("Takım verisi yetersiz.")

    lam_h_goal = league["home_goals"] * h.home_attack * a.away_def
    lam_a_goal = league["away_goals"] * a.away_attack * h.home_def

    lam_h, lam_a = lam_h_goal, lam_a_goal

    # xG blend (if shots data available and xg_weight>0) - not implemented here due to missing xg rates
    # You can extend rates to include xg_att/def if needed.

    # ELO scaling
    scale = math.exp((elo_to_goal_k * elo_diff) / 400.0)
    lam_h *= scale
    lam_a *= (1.0 / scale)

    # Rest days difference
    if rest_days_diff != 0:
        rest_factor = 1.0 + rest_factor_strength * rest_days_diff
        lam_h *= rest_factor
        lam_a /= rest_factor

    # Head-to-head
    if np.isfinite(h2h_home_avg) and np.isfinite(h2h_away_avg):
        h2h_home_ratio = h2h_home_avg / league["home_goals"] if league["home_goals"] > 0 else 1.0
        h2h_away_ratio = h2h_away_avg / league["away_goals"] if league["away_goals"] > 0 else 1.0
        lam_h = (1 - h2h_weight) * lam_h + h2h_weight * (league["home_goals"] * h2h_home_ratio)
        lam_a = (1 - h2h_weight) * lam_a + h2h_weight * (league["away_goals"] * h2h_away_ratio)

    lam_h *= manual_home_factor
    lam_a *= manual_away_factor

    lam_h = float(np.clip(lam_h, 0.05, 4.5))
    lam_a = float(np.clip(lam_a, 0.05, 4.5))
    return lam_h, lam_a

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

def kelly_fraction(p: float, odds: float) -> float:
    if not (np.isfinite(p) and np.isfinite(odds)) or odds <= 1.0 or p <= 0 or p >= 1:
        return 0.0
    b = odds - 1.0
    f = (p * odds - 1.0) / b
    return float(max(0.0, f))

def brier_1x2(pH_: float, pD_: float, pA_: float, y: int) -> float:
    yv = np.array([1.0 if y == 0 else 0.0, 1.0 if y == 1 else 0.0, 1.0 if y == 2 else 0.0])
    pv = np.array([pH_, pD_, pA_], dtype=float)
    pv = np.clip(pv, 1e-9, 1.0)
    pv = pv / pv.sum()
    return float(((pv - yv) ** 2).mean())

def logloss_1x2(pH_: float, pD_: float, pA_: float, y: int) -> float:
    pv = np.array([pH_, pD_, pA_], dtype=float)
    pv = np.clip(pv, 1e-12, 1.0)
    pv = pv / pv.sum()
    return float(-math.log(pv[y]))

# =========================================================
# BACKTEST with new features and calibration
# =========================================================
def backtest_1x2_value_enhanced(
    df_all_in: pd.DataFrame,
    df_pre_in: pd.DataFrame,
    anchor_date_in: dt.date,
    odds_cols: Tuple[str, str, str],
    lookback_in: int,
    prior_weight_in: float,
    min_matches_in: int,
    half_life_days_in: int,
    max_goals_in: int,
    xg_weight_in: float,
    elo_home_adv_in: float,
    elo_to_goal_k_in: float,
    edge_threshold_in: float,
    max_eval_matches_in: int,
    decay_halflife_in: float,
    rest_factor_strength_in: float,
    h2h_weight_in: float,
    use_dixon_coles_in: bool,
    xg_a: float,
    xg_b: float,
    league_tail_n: int = 800,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, Optional[List[LogisticRegression]]]:
    """
    Returns: results_df, summary_dict, calibration_df, calibration_models
    """
    hcol, dcol, acol = odds_cols

    d = df_all_in.copy()
    d = d[d["Date"].notna()].copy()
    d = d[d["Date"] <= pd.Timestamp(anchor_date_in)].copy()
    d = d.sort_values("Date").reset_index(drop=True)

    for c in [hcol, dcol, acol]:
        if c not in d.columns:
            return pd.DataFrame(), {"error": 1.0}, pd.DataFrame(), None

    d[hcol] = pd.to_numeric(d[hcol], errors="coerce")
    d[dcol] = pd.to_numeric(d[dcol], errors="coerce")
    d[acol] = pd.to_numeric(d[acol], errors="coerce")

    valid = (
        d[hcol].notna() & d[dcol].notna() & d[acol].notna() &
        (d[hcol] > 1.01) & (d[dcol] > 1.01) & (d[acol] > 1.01)
    )
    d = d[valid].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame(), {"error": 1.0}, pd.DataFrame(), None

    if max_eval_matches_in is not None and max_eval_matches_in > 0 and len(d) > max_eval_matches_in:
        d = d.iloc[-max_eval_matches_in:].copy().reset_index(drop=True)

    keys = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    pre = df_pre_in[keys + ["EloHomePre", "EloAwayPre"]].copy()
    pre["Date"] = pd.to_datetime(pre["Date"], errors="coerce")
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")

    merged = d.merge(pre, on=keys, how="left")
    merged["EloHomePre"] = merged["EloHomePre"].fillna(1500.0)
    merged["EloAwayPre"] = merged["EloAwayPre"].fillna(1500.0)

    full = df_all_in.copy()
    full = full[full["Date"].notna()].copy()
    full["Date"] = pd.to_datetime(full["Date"], errors="coerce")
    full = full[full["Date"] <= pd.Timestamp(anchor_date_in)].copy()
    full = full.sort_values("Date").reset_index(drop=True)

    home_scored_list: Dict[str, List[float]] = {}
    home_conceded_list: Dict[str, List[float]] = {}
    away_scored_list: Dict[str, List[float]] = {}
    away_conceded_list: Dict[str, List[float]] = {}
    counts: Dict[str, int] = {}

    pred_list = []
    outcome_list = []

    rows_out = []
    cal_bins = 10
    cal_counts = np.zeros(cal_bins, dtype=int)
    cal_sum_p = np.zeros(cal_bins, dtype=float)
    cal_sum_y = np.zeros(cal_bins, dtype=float)

    def league_fast(tail_df: pd.DataFrame, asof: pd.Timestamp) -> Dict[str, float]:
        dd = tail_df.copy()
        dd = dd[dd["Date"].notna()].copy()
        if dd.empty:
            return {
                "home_goals": 1.4, "away_goals": 1.1,
                "has_shots": False,
            }
        days_ago = (asof - dd["Date"]).dt.days.clip(lower=0).to_numpy()
        w = np.exp(-days_ago / float(max(10, half_life_days_in)))

        def wavg(arr) -> float:
            vals = np.asarray(pd.to_numeric(arr, errors="coerce"), dtype=float)
            m = np.isfinite(vals)
            if m.sum() == 0:
                return float("nan")
            return float(np.average(vals[m], weights=w[m]))

        out = {
            "home_goals": wavg(dd["FTHG"]),
            "away_goals": wavg(dd["FTAG"]),
            "has_shots": ("HS" in dd.columns and "AS" in dd.columns and "HST" in dd.columns and "AST" in dd.columns),
        }
        return out

    rho_global = dixon_coles_rho(full) if use_dixon_coles_in else 0.0

    eval_set = {(row.Date, row.HomeTeam, row.AwayTeam, row.FTHG, row.FTAG) for row in merged.itertuples()}

    for idx in range(len(full)):
        row = full.iloc[idx]
        date = row["Date"]
        ht = row["HomeTeam"]
        at = row["AwayTeam"]
        try:
            fthg = int(row["FTHG"])
            ftag = int(row["FTAG"])
        except Exception:
            continue

        is_eval = (date, ht, at, fthg, ftag) in eval_set

        if is_eval:
            odds_row = merged[(merged["Date"] == date) & (merged["HomeTeam"] == ht) & (merged["AwayTeam"] == at) & (merged["FTHG"] == fthg) & (merged["FTAG"] == ftag)]
            if odds_row.empty:
                continue
            oh = float(odds_row.iloc[0][hcol])
            od = float(odds_row.iloc[0][dcol])
            oa = float(odds_row.iloc[0][acol])
            elo_home_pre = float(odds_row.iloc[0]["EloHomePre"])
            elo_away_pre = float(odds_row.iloc[0]["EloAwayPre"])
        else:
            pass

        start_tail = max(0, idx - league_tail_n)
        tail_df = full.iloc[start_tail:idx].copy()
        lg = league_fast(tail_df, asof=date)

        prev_matches = full.iloc[:idx].copy()
        home_rest = compute_rest_days(ht, date, prev_matches)
        away_rest = compute_rest_days(at, date, prev_matches)
        rest_diff = home_rest - away_rest

        h2h_h, h2h_a = h2h_features(ht, at, prev_matches, n_last=5)

        def get_decayed_avg(lst: List[float], halflife: float) -> float:
            if not lst:
                return np.nan
            arr = np.array(lst[-lookback_in:], dtype=float)
            n = len(arr)
            if n == 0:
                return np.nan
            positions = np.arange(n)
            w = np.exp(-positions / halflife)
            w = w / w.sum()
            return float(np.average(arr, weights=w))

        home_scored = get_decayed_avg(home_scored_list.get(ht, []), decay_halflife_in)
        home_conceded = get_decayed_avg(home_conceded_list.get(ht, []), decay_halflife_in)
        away_scored = get_decayed_avg(away_scored_list.get(at, []), decay_halflife_in)
        away_conceded = get_decayed_avg(away_conceded_list.get(at, []), decay_halflife_in)

        def shrink_val(val, prior, weight, cnt):
            if np.isnan(val):
                return prior
            n_val = len(home_scored_list.get(ht, []))
            return (val * n_val + weight * prior) / (n_val + weight)

        home_scored_s = shrink_val(home_scored, lg["home_goals"], prior_weight_in, 1)
        home_conceded_s = shrink_val(home_conceded, lg["away_goals"], prior_weight_in, 1)
        away_scored_s = shrink_val(away_scored, lg["away_goals"], prior_weight_in, 1)
        away_conceded_s = shrink_val(away_conceded, lg["home_goals"], prior_weight_in, 1)

        ha = home_scored_s / lg["home_goals"] if lg["home_goals"] > 0 else 1.0
        hd = home_conceded_s / lg["away_goals"] if lg["away_goals"] > 0 else 1.0
        aa = away_scored_s / lg["away_goals"] if lg["away_goals"] > 0 else 1.0
        ad = away_conceded_s / lg["home_goals"] if lg["home_goals"] > 0 else 1.0

        class FakeRates:
            pass
        rates_fake = {
            ht: type('R', (), {'home_attack': ha, 'home_def': hd, 'away_attack': 1.0, 'away_def': 1.0})(),
            at: type('R', (), {'home_attack': 1.0, 'home_def': 1.0, 'away_attack': aa, 'away_def': ad})()
        }

        try:
            lam_h_pred, lam_a_pred = expected_goals_from_rates_enhanced(
                rates_fake, lg, ht, at,
                xg_weight=0.0,
                elo_diff=(elo_home_pre + elo_home_adv_in) - elo_away_pre,
                elo_to_goal_k=elo_to_goal_k_in,
                manual_home_factor=1.0,
                manual_away_factor=1.0,
                rest_days_diff=rest_diff,
                h2h_home_avg=h2h_h,
                h2h_away_avg=h2h_a,
                xg_a=xg_a,
                xg_b=xg_b,
                rest_factor_strength=rest_factor_strength_in,
                h2h_weight=h2h_weight_in
            )
        except ValueError:
            if not is_eval:
                pass
            continue

        mat_bt = score_matrix(lam_h_pred, lam_a_pred, max_goals=max_goals_in)
        if use_dixon_coles_in and rho_global != 0:
            mat_bt = dixon_coles_adjustment(mat_bt, lam_h_pred, lam_a_pred, rho_global)
            mat_bt = mat_bt / mat_bt.sum()
        pH_m, pD_m, pA_m = outcome_probs_from_mat(mat_bt)

        if is_eval:
            pH_mkt, pD_mkt, pA_mkt = implied_probs_1x2(oh, od, oa)
            y = 0 if fthg > ftag else (2 if fthg < ftag else 1)

            edges = {
                "H": pH_m - pH_mkt,
                "D": pD_m - pD_mkt,
                "A": pA_m - pA_mkt,
            }
            side = max(edges, key=lambda k: edges[k])
            best_edge = float(edges[side])

            bet = best_edge >= edge_threshold_in
            profit = 0.0
            odds_used = {"H": oh, "D": od, "A": oa}[side]
            if bet:
                win = (y == 0 and side == "H") or (y == 1 and side == "D") or (y == 2 and side == "A")
                profit = (odds_used - 1.0) if win else -1.0

            ph = float(np.clip(pH_m, 1e-9, 1.0))
            b = min(cal_bins - 1, int(ph * cal_bins))
            cal_counts[b] += 1
            cal_sum_p[b] += ph
            cal_sum_y[b] += 1.0 if y == 0 else 0.0

            pred_list.append([pH_m, pD_m, pA_m])
            outcome_list.append(y)

            rows_out.append({
                "Date": date.date() if hasattr(date, "date") else date,
                "Home": ht,
                "Away": at,
                "Score": f"{fthg}-{ftag}",
                "OddsH": float(oh),
                "OddsD": float(od),
                "OddsA": float(oa),
                "Mkt_H": float(pH_mkt),
                "Mkt_D": float(pD_mkt),
                "Mkt_A": float(pA_mkt),
                "Model_H": float(pH_m),
                "Model_D": float(pD_m),
                "Model_A": float(pA_m),
                "BestSide": side,
                "BestEdge": best_edge,
                "Bet": bool(bet),
                "Profit_1u": float(profit),
                "Brier_Model": brier_1x2(pH_m, pD_m, pA_m, y),
                "Brier_Mkt": brier_1x2(pH_mkt, pD_mkt, pA_mkt, y),
                "LogLoss_Model": logloss_1x2(pH_m, pD_m, pA_m, y),
                "LogLoss_Mkt": logloss_1x2(pH_mkt, pD_mkt, pA_mkt, y),
            })

        home_scored_list.setdefault(ht, []).append(float(fthg))
        home_conceded_list.setdefault(ht, []).append(float(ftag))
        away_scored_list.setdefault(at, []).append(float(ftag))
        away_conceded_list.setdefault(at, []).append(float(fthg))
        counts[ht] = counts.get(ht, 0) + 1
        counts[at] = counts.get(at, 0) + 1

    res = pd.DataFrame(rows_out)
    if res.empty:
        return res, {"error": 1.0}, pd.DataFrame(), None

    n = len(res)
    brier_model = float(res["Brier_Model"].mean())
    brier_mkt = float(res["Brier_Mkt"].mean())
    ll_model = float(res["LogLoss_Model"].mean())
    ll_mkt = float(res["LogLoss_Mkt"].mean())

    bets = res[res["Bet"]].copy()
    nb = len(bets)
    profit = float(bets["Profit_1u"].sum()) if nb > 0 else 0.0
    roi = float(profit / nb) if nb > 0 else 0.0
    hit = float((bets["Profit_1u"] > 0).mean()) if nb > 0 else 0.0
    avg_edge = float(bets["BestEdge"].mean()) if nb > 0 else 0.0
    avg_odds = float(bets.apply(lambda r: {"H": r["OddsH"], "D": r["OddsD"], "A": r["OddsA"]}[r["BestSide"]], axis=1).mean()) if nb > 0 else 0.0

    summary = {
        "matches": n,
        "brier_model": brier_model,
        "brier_market": brier_mkt,
        "logloss_model": ll_model,
        "logloss_market": ll_mkt,
        "bets": nb,
        "profit_1u": profit,
        "roi_per_bet": roi,
        "hit_rate": hit,
        "avg_edge": avg_edge,
        "avg_odds": avg_odds,
    }

    cal_rows = []
    for i in range(cal_bins):
        if cal_counts[i] == 0:
            continue
        cal_rows.append({
            "Bin": f"{i/cal_bins:.1f}-{(i+1)/cal_bins:.1f}",
            "Count": int(cal_counts[i]),
            "AvgPred(HomeWin)": float(cal_sum_p[i] / cal_counts[i]),
            "Empirical(HomeWin)": float(cal_sum_y[i] / cal_counts[i]),
        })
    cal_df = pd.DataFrame(cal_rows)

    calibration_models = None
    if len(pred_list) > 100:
        pred_arr = np.array(pred_list)
        outcome_arr = np.array(outcome_list)
        calibration_models = []
        for i in range(3):
            y = (outcome_arr == i).astype(int)
            X = np.log(pred_arr[:, i] / (1 - pred_arr[:, i] + 1e-12)).reshape(-1, 1)
            lr = LogisticRegression(C=1e5, solver='lbfgs')
            lr.fit(X, y)
            calibration_models.append(lr)

    return res.sort_values("Date"), summary, cal_df, calibration_models

# =========================================================
# SIDEBAR — Single list leagues + auto divs + new params
# =========================================================
with st.sidebar:
    st.header("Ayarlar")

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
    st.subheader("Gelişmiş Model Parametreleri")
    decay_halflife = st.slider("Form yarı ömrü (maç sayısı)", 3, 30, 10, 1,
                               help="Son maçların ağırlığının yarıya düştüğü maç sayısı")
    use_dixon_coles = st.checkbox("Dixon-Coles düzeltmesi uygula", value=True)
    rest_factor_strength = st.slider("Dinlenme günü etki katsayısı", 0.0, 0.2, 0.05, 0.01,
                                     help="Her gün farkı için % kaç etki")
    h2h_weight = st.slider("Head-to-head ağırlığı", 0.0, 0.5, 0.15, 0.05,
                           help="Geçmiş maç ortalamasının modele katkısı")
    calibrate_probs = st.checkbox("Olasılıkları kalibre et (Platt scaling)", value=True,
                                   help="Backtest sonuçlarına göre olasılıkları düzeltir")

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

# xG coefficients
xg_a, xg_b = estimate_xg_coefficients(df_all)
st.sidebar.write(f"xG proxy katsayıları: a={xg_a:.3f}, b={xg_b:.3f}")

league = league_avgs_weighted(df_all, anchor_date, half_life_days=half_life_days)
league["xg_a"] = xg_a
league["xg_b"] = xg_b
league["has_shots"] = ("HS" in df_all.columns and "AS" in df_all.columns and "HST" in df_all.columns and "AST" in df_all.columns)

rates = build_team_rates_recent_decayed(
    df_all, league, lookback=lookback, prior_weight=prior_weight,
    min_matches=min_matches, decay_halflife=decay_halflife
)
teams = sorted(rates.keys())

if len(teams) < 4:
    st.warning("Yeterli takım verisi oluşmadı. min_matches düşür veya daha fazla data yükle.")
    st.stop()

df_ratings_pre, elo_final, glicko_final = compute_ratings(
    df_all, elo_k=elo_k, elo_home_adv=elo_home_adv, glicko_home_adv=glicko_home_adv
)

# =========================================================
# LEAGUE GOALS PER MATCH
# =========================================================
weighted_home = float(league["home_goals"])
weighted_away = float(league["away_goals"])
weighted_total = weighted_home + weighted_away

raw_home = float(df_all["FTHG"].mean())
raw_away = float(df_all["FTAG"].mean())
raw_total = raw_home + raw_away

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
# MATCH ANALYZER (with enhanced features)
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

# Compute rest days difference
all_matches_sorted = df_all.sort_values("Date").copy()
home_rest = compute_rest_days(home_team, pd.Timestamp(anchor_date), all_matches_sorted)
away_rest = compute_rest_days(away_team, pd.Timestamp(anchor_date), all_matches_sorted)
rest_diff = home_rest - away_rest

# Head-to-head
h2h_h, h2h_a = h2h_features(home_team, away_team, df_all, n_last=5)

lam_h, lam_a = expected_goals_from_rates_enhanced(
    rates, league,
    home=home_team, away=away_team,
    xg_weight=xg_weight,
    elo_diff=elo_diff,
    elo_to_goal_k=elo_to_goal_k,
    manual_home_factor=manual_home_factor,
    manual_away_factor=manual_away_factor,
    rest_days_diff=rest_diff,
    h2h_home_avg=h2h_h,
    h2h_away_avg=h2h_a,
    xg_a=xg_a,
    xg_b=xg_b,
    rest_factor_strength=rest_factor_strength,
    h2h_weight=h2h_weight
)

# Dixon-Coles rho
rho = dixon_coles_rho(df_all) if use_dixon_coles else 0.0
mat = score_matrix(lam_h, lam_a, max_goals=max_goals)
if use_dixon_coles and rho != 0:
    mat = dixon_coles_adjustment(mat, lam_h, lam_a, rho)
    mat = mat / mat.sum()

pH, pD, pA = outcome_probs_from_mat(mat)

# If calibration models exist in session state, apply them
if calibrate_probs and 'calibration_models' in st.session_state and st.session_state.calibration_models is not None:
    probs_array = np.array([[pH, pD, pA]])
    cal_models = st.session_state.calibration_models
    cal_probs = np.zeros_like(probs_array)
    for i, lr in enumerate(cal_models):
        X = np.log(probs_array[:, i] / (1 - probs_array[:, i] + 1e-12)).reshape(-1, 1)
        cal_probs[:, i] = lr.predict_proba(X)[:, 1]
    cal_probs = cal_probs / cal_probs.sum(axis=1, keepdims=True)
    pH, pD, pA = cal_probs[0]

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
    st.caption(f"ELO diff: {elo_diff:+.0f} | Glicko diff: {g_diff:+.0f} | Rest diff: {rest_diff:+d}")

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
    st.info("Korner tahmini bu sürümde devre dışıdır.")

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

# =========================================================
# MANUAL ODDS EDGE
# =========================================================
st.divider()
st.subheader("📊 Market Odds (1X2) → Model vs Market (Edge/Value)")

with st.expander("Odds gir + edge/value hesapla (manual)"):
    o1, ox, o2 = st.columns(3)
    with o1:
        in_h = st.number_input("Home odds", min_value=1.01, value=2.20, step=0.01)
    with ox:
        in_d = st.number_input("Draw odds", min_value=1.01, value=3.30, step=0.01)
    with o2:
        in_a = st.number_input("Away odds", min_value=1.01, value=3.10, step=0.01)

    kelly_mult = st.slider("Kelly çarpanı (risk azaltma)", 0.0, 1.0, 0.50, 0.05)

    mktH, mktD, mktA = implied_probs_1x2(float(in_h), float(in_d), float(in_a))

    fairH, fairD, fairA = (1/pH if pH > 0 else np.nan), (1/pD if pD > 0 else np.nan), (1/pA if pA > 0 else np.nan)
    edgeH, edgeD, edgeA = (pH - mktH), (pD - mktD), (pA - mktA)

    st.write("**Market implied probs (normalized):**")
    st.write(f"H: **{mktH*100:.1f}%** | D: **{mktD*100:.1f}%** | A: **{mktA*100:.1f}%**")

    st.write("**Model probs:**")
    st.write(f"H: **{pH*100:.1f}%** | D: **{pD*100:.1f}%** | A: **{pA*100:.1f}%**")

    st.write("**Edge (Model - Market):**")
    st.write(f"H: **{edgeH*100:+.2f}%** | D: **{edgeD*100:+.2f}%** | A: **{edgeA*100:+.2f}%**")

    edges = {"H": edgeH, "D": edgeD, "A": edgeA}
    best_side = max(edges, key=lambda k: edges[k])
    best_edge = float(edges[best_side])
    odds_best = {"H": float(in_h), "D": float(in_d), "A": float(in_a)}[best_side]
    p_best = {"H": float(pH), "D": float(pD), "A": float(pA)}[best_side]
    fair_best = 1/p_best if p_best > 0 else np.nan

    kelly = kelly_fraction(p_best, odds_best)
    kelly_sized = kelly_mult * kelly

    st.divider()
    st.write(f"**En iyi value taraf:** `{best_side}` | Edge: **{best_edge*100:+.2f}%**")
    st.write(f"Model fair odds: **{fair_best:.2f}** | Market odds: **{odds_best:.2f}**")
    st.write(f"Kelly fraction: **{kelly*100:.2f}%** | Uygulanacak (çarpanlı): **{kelly_sized*100:.2f}%** (bankroll yüzdesi)")

    if best_edge <= 0:
        st.info("Model bu odds setinde net bir value görmüyor (edge <= 0).")

# =========================================================
# BACKTEST UI
# =========================================================
st.divider()
st.subheader("🧪 Odds Backtest (1X2) — Model vs Market + Value Betting")

with st.sidebar:
    st.divider()
    st.subheader("Backtest (1X2 odds)")
    avail_sets = find_available_odds_sets(df_all)
    if len(avail_sets) == 0:
        st.caption("Bu lig datasında 1X2 odds kolonları bulunamadı.")
        bt_set = None
    else:
        bt_set = st.selectbox("Odds set", avail_sets, index=0)
    bt_n = st.slider("Backtest: kaç maç (son)", 100, 1200, 400, 50)
    bt_edge = st.slider("Value threshold (edge)", 0.00, 0.10, 0.02, 0.005)
    bt_run = st.button("Backtest çalıştır")

with st.expander("Backtest sonuçları (aç)"):
    if bt_set is None:
        st.warning("Bu ligde odds kolonları yok.")
    elif not bt_run:
        st.info("Backtest için soldan **Backtest çalıştır** butonuna bas.")
    else:
        cols = get_odds_cols(bt_set)
        if cols is None:
            st.error("Odds set kolonları bulunamadı.")
        else:
            with st.spinner("Backtest çalışıyor..."):
                res_df, summary, cal_df, cal_models = backtest_1x2_value_enhanced(
                    df_all_in=df_all,
                    df_pre_in=df_ratings_pre,
                    anchor_date_in=anchor_date,
                    odds_cols=cols,
                    lookback_in=lookback,
                    prior_weight_in=prior_weight,
                    min_matches_in=min_matches,
                    half_life_days_in=half_life_days,
                    max_goals_in=max_goals,
                    xg_weight_in=xg_weight,
                    elo_home_adv_in=elo_home_adv,
                    elo_to_goal_k_in=elo_to_goal_k,
                    edge_threshold_in=bt_edge,
                    max_eval_matches_in=bt_n,
                    decay_halflife_in=decay_halflife,
                    rest_factor_strength_in=rest_factor_strength,
                    h2h_weight_in=h2h_weight,
                    use_dixon_coles_in=use_dixon_coles,
                    xg_a=xg_a,
                    xg_b=xg_b,
                )

            if "error" in summary and summary["error"] == 1.0:
                st.error("Backtest çalıştırılamadı (muhtemelen yeterli odds/maç yok).")
            else:
                if cal_models is not None:
                    st.session_state.calibration_models = cal_models
                    st.success("Kalibrasyon modeli oluşturuldu ve kaydedildi.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Matches", f"{summary['matches']}")
                m2.metric("Brier (Model vs Market)", f"{summary['brier_model']:.4f} | {summary['brier_market']:.4f}")
                m3.metric("LogLoss (Model vs Market)", f"{summary['logloss_model']:.4f} | {summary['logloss_market']:.4f}")
                m4.metric("Bets / ROI per bet", f"{summary['bets']} | {summary['roi_per_bet']*100:.2f}%")

                st.write(
                    f"Profit (1u staking): **{summary['profit_1u']:.2f}u** | "
                    f"Hit rate: **{summary['hit_rate']*100:.1f}%** | "
                    f"Avg edge: **{summary['avg_edge']*100:.2f}%** | "
                    f"Avg odds: **{summary['avg_odds']:.2f}**"
                )

                st.markdown("#### Kalibrasyon (Home win)")
                if cal_df is None or cal_df.empty:
                    st.info("Kalibrasyon tablosu üretilemedi.")
                else:
                    st.dataframe(cal_df, use_container_width=True, height=260)

                st.markdown("#### En iyi value bet’ler (edge’e göre)")
                if res_df is None or res_df.empty:
                    st.info("Sonuç yok.")
                else:
                    show = res_df.copy()
                    show["Edge%"] = (show["BestEdge"] * 100).round(2)
                    show = show.sort_values("BestEdge", ascending=False)
                    st.dataframe(show.head(50)[
                        ["Date", "Home", "Away", "Score", "BestSide", "Edge%", "Bet", "Profit_1u", "OddsH", "OddsD", "OddsA", "Model_H", "Model_D", "Model_A", "Mkt_H", "Mkt_D", "Mkt_A"]
                    ], use_container_width=True, height=420)

                with st.expander("Tüm backtest satırları (debug)"):
                    st.dataframe(res_df, use_container_width=True, height=520)
