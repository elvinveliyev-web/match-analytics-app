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

    # Extra leagues (worldwide)
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
    fx = load_csv(FIXTURES_URL)
    fx.columns = [str(c).replace("\ufeff", "").strip() for c in fx.columns]
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
        f = load_csv(FIXTURES_URL)
        f.columns = [str(c).replace("\ufeff", "").strip() for c in f.columns]
        if "Date" in f.columns:
            f["Date"] = pd.to_datetime(f["Date"], dayfirst=True, errors="coerce")
        return f
    except Exception:
        return None

def _normalize_history(df: pd.DataFrame, anchor_date: dt.date) -> pd.DataFrame:
    """
    Supports BOTH schemas:
    - mmz: Date, HomeTeam, AwayTeam, FTHG, FTAG
    - new/worldwide: Date, Home, Away, HG, AG  (or similar)
    """
    d = df.copy()
    d.columns = [str(c).replace("\ufeff", "").strip() for c in d.columns]

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

    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce")
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
        keep="last"
    )

    return d

@st.cache_data(show_spinner=False)
def load_history(div: str, anchor_date: dt.date, back_seasons: int = 6) -> Tuple[pd.DataFrame, List[str], str]:
    """
    1) Try mmz season files: /mmz4281/{season}/{div}.csv (multiple seasons)
    2) If nothing found: /new/{div}.csv (extra leagues)
    """
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
        d = _normalize_history(d, anchor_date)
        if "Season" not in d.columns:
            d["Season"] = "mmz"
        return d, seasons_loaded, "mmz"

    new_url = NEW_LEAGUE_URL.format(div=div)
    df = load_csv(new_url)
    df["Season"] = "NEW"
    d = _normalize_history(df, anchor_date)
    return d, ["NEW"], "new"

# Weighted averages (FIXED)
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
        def xg_proxy(shots: float, sot: float) -> float:
            if not np.isfinite(shots) and not np.isfinite(sot):
                return np.nan
            s = shots if np.isfinite(shots) else 0.0
            t = sot if np.isfinite(sot) else 0.0
            return 0.04 * s + 0.08 * t

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

def shrink_mean(values: np.ndarray, prior_mean: float, prior_weight: float) -> float:
    v = np.array(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return float(prior_mean)
    return float((v.sum() + prior_weight * prior_mean) / (n + prior_weight))

@dataclass
class TeamRates:
    home_attack: float
    home_def: float
    away_attack: float
    away_def: float
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

        rates[t] = TeamRates(
            home_attack=float(ha),
            home_def=float(hd),
            away_attack=float(aa),
            away_def=float(ad),
            home_used=int(len(home)),
            away_used=int(len(away)),
        )

    return rates

def expected_goals_from_rates(
    rates: Dict[str, TeamRates],
    league: Dict[str, float],
    home: str,
    away: str,
    manual_home_factor: float,
    manual_away_factor: float,
) -> Tuple[float, float]:
    h = rates.get(home)
    a = rates.get(away)
    if h is None or a is None:
        raise ValueError("Takım verisi yetersiz. min maç düşür veya daha fazla data yükle.")

    lam_h = league["home_goals"] * h.home_attack * a.away_def
    lam_a = league["away_goals"] * a.away_attack * h.home_def

    lam_h *= manual_home_factor
    lam_a *= manual_away_factor

    lam_h = float(np.clip(lam_h, 0.05, 4.5))
    lam_a = float(np.clip(lam_a, 0.05, 4.5))
    return lam_h, lam_a

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

def top_scores(mat: np.ndarray, topn: int = 10) -> pd.DataFrame:
    items = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            items.append((i, j, mat[i, j]))
    items.sort(key=lambda x: x[2], reverse=True)
    out = pd.DataFrame(items[:topn], columns=["HomeGoals", "AwayGoals", "Prob"])
    out["FairOdds"] = out["Prob"].apply(lambda p: (1.0 / p) if p > 0 else np.nan)
    return out

# =========================================================
# SIDEBAR
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

league = league_avgs_weighted(df_all, anchor_date, half_life_days=half_life_days)
rates = build_team_rates_recent(df_all, league, lookback=lookback, prior_weight=prior_weight, min_matches=min_matches)
teams = sorted(rates.keys())

if len(teams) < 4:
    st.warning("Yeterli takım verisi oluşmadı. min_matches düşür veya daha fazla data yükle.")
    st.stop()

weighted_home = float(league["home_goals"])
weighted_away = float(league["away_goals"])
weighted_total = weighted_home + weighted_away

raw_total = float(league["raw_total_goals"])

with st.sidebar:
    st.divider()
    st.subheader("Seçtiğin lig: Gol ortalaması")
    st.metric("Total goals / match (recent-weighted)", f"{weighted_total:.2f}")
    st.caption(f"Raw mean total: {raw_total:.2f}")
    st.caption(f"Kaynak: {source_used}")

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

try:
    lam_h, lam_a = expected_goals_from_rates(
        rates, league,
        home=home_team, away=away_team,
        manual_home_factor=manual_home_factor,
        manual_away_factor=manual_away_factor,
    )
except Exception as e:
    st.error(str(e))
    st.stop()

mat = score_matrix(lam_h, lam_a, max_goals=max_goals)
pH, pD, pA = outcome_probs_from_mat(mat)

p_over15 = prob_over_from_mat(mat, 1.5)
p_over25 = prob_over_from_mat(mat, 2.5)
p_over35 = prob_over_from_mat(mat, 3.5)
p_btts = prob_btts_from_mat(mat)

A, B = st.columns([1.2, 1.0], vertical_alignment="top")

with A:
    st.markdown("### 1X2 (FT)")
    st.write(f"**H {pH*100:.1f}% | D {pD*100:.1f}% | A {pA*100:.1f}%**")
    st.caption(f"Expected goals: {lam_h:.2f} - {lam_a:.2f}")

    st.markdown("### Over/Under")
    st.write(f"Over 1.5: **{p_over15*100:.1f}%** (Fair {1/p_over15:.2f})")
    st.write(f"Over 2.5: **{p_over25*100:.1f}%** (Fair {1/p_over25:.2f})")
    st.write(f"Over 3.5: **{p_over35*100:.1f}%** (Fair {1/p_over35:.2f})")

    st.markdown("### BTTS (KG Var)")
    st.write(f"BTTS: **{p_btts*100:.1f}%** (Fair {1/p_btts:.2f})")

with B:
    st.markdown("### Doğru skor (Top 10)")
    st.dataframe(top_scores(mat, topn=10), use_container_width=True, height=360)

st.divider()
st.subheader("🧩 Diagnostics")
st.write(f"Div: **{div}** | Source: **{source_used}** | Matches loaded: **{len(df_all)}**")
st.write(f"Seasons: **{', '.join(seasons_loaded[:8])}{'...' if len(seasons_loaded)>8 else ''}**")
with st.expander("Columns (debug)"):
    st.write(list(df_all.columns))
