import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import requests
import streamlit as st


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Pro Match Analytics (Football)", layout="wide")

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Çok lig + custom div
LEAGUES = {
    # Turkey
    "Turkey - T1": "T1",

    # England
    "England - Premier League (E0)": "E0",
    "England - Championship (E1)": "E1",
    "England - League One (E2)": "E2",
    "England - League Two (E3)": "E3",

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
}

# football-data 1X2 odds seçenekleri (varsa otomatik listeliyoruz)
ODDS_SETS = [
    ("Average", ("AvgH", "AvgD", "AvgA")),
    ("Bet365", ("B365H", "B365D", "B365A")),
    ("Pinnacle", ("PSH", "PSD", "PSA")),
    ("WilliamHill", ("WHH", "WHD", "WHA")),
    ("Bet&Win", ("BWH", "BWD", "BWA")),
    ("Interwetten", ("IWH", "IWD", "IWA")),
    ("VCBet", ("VCH", "VCD", "VCA")),
]


# =========================
# HELPERS
# =========================
def season_code(start_year: int) -> str:
    y1 = start_year % 100
    y2 = (start_year + 1) % 100
    return f"{y1:02d}{y2:02d}"


def season_start_year_for_date(d: dt.date) -> int:
    # Futbol sezonu çoğunlukla yazın başlar
    return d.year if d.month >= 7 else d.year - 1


def hist_url(season: str, div: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"


@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    from io import StringIO
    txt = r.content.decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(txt))


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_multi_season_history(div: str, anchor_date: dt.date, back_seasons: int = 5) -> Tuple[pd.DataFrame, List[str]]:
    """
    anchor_date'e göre doğru sezonu bulur, o sezon + önceki sezonları indirip birleştirir.
    """
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

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = parse_dates(df_all)

    need = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = need - set(df_all.columns)
    if missing:
        raise ValueError(f"Zorunlu kolonlar eksik: {sorted(list(missing))}")

    df_all = df_all.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df_all["FTHG"] = pd.to_numeric(df_all["FTHG"], errors="coerce")
    df_all["FTAG"] = pd.to_numeric(df_all["FTAG"], errors="coerce")
    df_all = df_all.dropna(subset=["FTHG", "FTAG"]).copy()

    df_all = df_all[df_all["Date"] <= pd.Timestamp(anchor_date)].copy()
    df_all = df_all.sort_values("Date")
    df_all = df_all.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"], keep="last")

    return df_all, seasons_loaded


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


def score_matrix(lam_home: float, lam_away: float, max_goals: int = 6) -> np.ndarray:
    hp = np.array([poisson_pmf(i, lam_home) for i in range(max_goals + 1)], dtype=float)
    ap = np.array([poisson_pmf(i, lam_away) for i in range(max_goals + 1)], dtype=float)
    mat = np.outer(hp, ap)
    return mat / mat.sum()


def outcome_probs_from_mat(mat: np.ndarray) -> Tuple[float, float, float]:
    # rows=home goals, cols=away goals
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


def prob_total_exact(mat: np.ndarray, total: int) -> float:
    p = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i + j == total:
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


def result_label(fthg: int, ftag: int) -> str:
    if fthg > ftag:
        return "H"
    if fthg < ftag:
        return "A"
    return "D"


def last_matches_table(df_all: pd.DataFrame, team: str, n: int = 20) -> pd.DataFrame:
    m = df_all[(df_all["HomeTeam"] == team) | (df_all["AwayTeam"] == team)].copy()
    m = m.sort_values("Date").tail(n).copy()
    if m.empty:
        return m
    m["GF"] = np.where(m["HomeTeam"] == team, m["FTHG"], m["FTAG"])
    m["GA"] = np.where(m["HomeTeam"] == team, m["FTAG"], m["FTHG"])
    m["WDL"] = np.where(m["GF"] > m["GA"], "W", np.where(m["GF"] < m["GA"], "L", "D"))
    m["Score"] = m["FTHG"].astype(int).astype(str) + "-" + m["FTAG"].astype(int).astype(str)
    cols = ["Date", "Season", "HomeTeam", "AwayTeam", "Score", "GF", "GA", "WDL"]
    return m[cols].reset_index(drop=True)


def league_weighted_avgs(df_all: pd.DataFrame, anchor_date: dt.date, half_life_days: int) -> Dict[str, float]:
    d = df_all.copy()
    last_ts = pd.Timestamp(anchor_date)
    days_ago = (last_ts - d["Date"]).dt.days.clip(lower=0)
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

    # HT ratios if exist
    if {"HTHG", "HTAG"}.issubset(d.columns):
        hthg = pd.to_numeric(d["HTHG"], errors="coerce")
        htag = pd.to_numeric(d["HTAG"], errors="coerce")
        if hthg.notna().sum() > 50 and out["home_goals"] > 0 and out["away_goals"] > 0:
            out["ht_ratio_home"] = max(0.25, min(0.65, wavg(hthg) / out["home_goals"]))
            out["ht_ratio_away"] = max(0.25, min(0.65, wavg(htag) / out["away_goals"]))
        else:
            out["ht_ratio_home"] = 0.45
            out["ht_ratio_away"] = 0.45
    else:
        out["ht_ratio_home"] = 0.45
        out["ht_ratio_away"] = 0.45

    # Corners if exist
    out["has_corners"] = {"HC", "AC"}.issubset(d.columns)
    if out["has_corners"]:
        out["home_corners"] = wavg(pd.to_numeric(d["HC"], errors="coerce"))
        out["away_corners"] = wavg(pd.to_numeric(d["AC"], errors="coerce"))
    else:
        out["home_corners"] = float("nan")
        out["away_corners"] = float("nan")

    return out


@dataclass
class TeamRates:
    home_attack: float
    home_def: float
    away_attack: float
    away_def: float
    home_used: int
    away_used: int
    # corners
    home_c_att: Optional[float] = None
    home_c_def: Optional[float] = None
    away_c_att: Optional[float] = None
    away_c_def: Optional[float] = None


def shrink_mean(values: np.ndarray, prior_mean: float, prior_weight: float) -> float:
    values = np.array(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float(prior_mean)
    return float((values.sum() + prior_weight * prior_mean) / (n + prior_weight))


def build_team_rates_recent(
    df_all: pd.DataFrame,
    league_avgs: Dict[str, float],
    lookback: int,
    prior_weight: float,
    min_matches: int,
) -> Dict[str, TeamRates]:
    teams = sorted(set(df_all["HomeTeam"]).union(set(df_all["AwayTeam"])))
    rates = {}

    for t in teams:
        home = df_all[df_all["HomeTeam"] == t].tail(lookback)
        away = df_all[df_all["AwayTeam"] == t].tail(lookback)

        if (len(home) + len(away)) < min_matches:
            continue

        # Goals (shrink towards league)
        home_scored = shrink_mean(home["FTHG"].values, league_avgs["home_goals"], prior_weight)
        home_conceded = shrink_mean(home["FTAG"].values, league_avgs["away_goals"], prior_weight)
        away_scored = shrink_mean(away["FTAG"].values, league_avgs["away_goals"], prior_weight)
        away_conceded = shrink_mean(away["FTHG"].values, league_avgs["home_goals"], prior_weight)

        ha = home_scored / league_avgs["home_goals"] if league_avgs["home_goals"] > 0 else np.nan
        hd = home_conceded / league_avgs["away_goals"] if league_avgs["away_goals"] > 0 else np.nan
        aa = away_scored / league_avgs["away_goals"] if league_avgs["away_goals"] > 0 else np.nan
        ad = away_conceded / league_avgs["home_goals"] if league_avgs["home_goals"] > 0 else np.nan

        tr = TeamRates(
            home_attack=float(ha),
            home_def=float(hd),
            away_attack=float(aa),
            away_def=float(ad),
            home_used=int(len(home)),
            away_used=int(len(away)),
        )

        # Corners
        if league_avgs.get("has_corners") and np.isfinite(league_avgs.get("home_corners", np.nan)) and np.isfinite(league_avgs.get("away_corners", np.nan)):
            # home corners for = HC, against = AC
            home_cf = shrink_mean(pd.to_numeric(home.get("HC"), errors="coerce").values, league_avgs["home_corners"], prior_weight)
            home_ca = shrink_mean(pd.to_numeric(home.get("AC"), errors="coerce").values, league_avgs["away_corners"], prior_weight)
            # away team corners for are AC, against are HC
            away_cf = shrink_mean(pd.to_numeric(away.get("AC"), errors="coerce").values, league_avgs["away_corners"], prior_weight)
            away_ca = shrink_mean(pd.to_numeric(away.get("HC"), errors="coerce").values, league_avgs["home_corners"], prior_weight)

            tr.home_c_att = float(home_cf / league_avgs["home_corners"]) if league_avgs["home_corners"] > 0 else np.nan
            tr.home_c_def = float(home_ca / league_avgs["away_corners"]) if league_avgs["away_corners"] > 0 else np.nan
            tr.away_c_att = float(away_cf / league_avgs["away_corners"]) if league_avgs["away_corners"] > 0 else np.nan
            tr.away_c_def = float(away_ca / league_avgs["home_corners"]) if league_avgs["home_corners"] > 0 else np.nan

        rates[t] = tr

    return rates


def expected_goals(team_rates: Dict[str, TeamRates], league_avgs: Dict[str, float], home: str, away: str) -> Tuple[float, float]:
    h = team_rates.get(home)
    a = team_rates.get(away)
    if h is None or a is None:
        raise ValueError("Takım verisi yetersiz: min maç düşür veya back seasons artır.")

    lam_h = league_avgs["home_goals"] * h.home_attack * a.away_def
    lam_a = league_avgs["away_goals"] * a.away_attack * h.home_def

    lam_h = float(np.clip(lam_h, 0.05, 4.5))
    lam_a = float(np.clip(lam_a, 0.05, 4.5))
    return lam_h, lam_a


def expected_corners(team_rates: Dict[str, TeamRates], league_avgs: Dict[str, float], home: str, away: str) -> Optional[Tuple[float, float]]:
    if not league_avgs.get("has_corners"):
        return None
    h = team_rates.get(home)
    a = team_rates.get(away)
    if h is None or a is None:
        return None
    if h.home_c_att is None or h.home_c_def is None or a.away_c_att is None or a.away_c_def is None:
        return None

    lam_hc = league_avgs["home_corners"] * h.home_c_att * a.away_c_def
    lam_ac = league_avgs["away_corners"] * a.away_c_att * h.home_c_def

    lam_hc = float(np.clip(lam_hc, 0.2, 18.0))
    lam_ac = float(np.clip(lam_ac, 0.2, 18.0))
    return lam_hc, lam_ac


def htft_probs(lam_h: float, lam_a: float, ht_ratio_home: float, ht_ratio_away: float, max_goals: int) -> Dict[str, float]:
    """
    İY/MS için basit ama işe yarayan yaklaşım:
    - 1. yarı golleri Poisson(lam_ht)
    - 2. yarı golleri Poisson(lam_2h = lam_ft - lam_ht) (clamp)
    - bağımsız varsayımıyla HT ve FT skorlarını birleştir
    """
    lam_h_ht = max(0.01, lam_h * ht_ratio_home)
    lam_a_ht = max(0.01, lam_a * ht_ratio_away)
    lam_h_2h = max(0.01, lam_h - lam_h_ht)
    lam_a_2h = max(0.01, lam_a - lam_a_ht)

    mat_ht = score_matrix(lam_h_ht, lam_a_ht, max_goals=max_goals)
    mat_2h = score_matrix(lam_h_2h, lam_a_2h, max_goals=max_goals)

    # HT result x FT result
    combos = {
        "H/H": 0.0, "H/D": 0.0, "H/A": 0.0,
        "D/H": 0.0, "D/D": 0.0, "D/A": 0.0,
        "A/H": 0.0, "A/D": 0.0, "A/A": 0.0,
    }

    for i in range(mat_ht.shape[0]):
        for j in range(mat_ht.shape[1]):
            p_ht = mat_ht[i, j]
            ht_res = "H" if i > j else ("A" if i < j else "D")

            for a in range(mat_2h.shape[0]):
                for b in range(mat_2h.shape[1]):
                    p_2h = mat_2h[a, b]
                    ft_i = i + a
                    ft_j = j + b
                    ft_res = "H" if ft_i > ft_j else ("A" if ft_i < ft_j else "D")
                    combos[f"{ht_res}/{ft_res}"] += p_ht * p_2h

    return combos


def find_available_odds_sets(df: pd.DataFrame) -> List[str]:
    avail = []
    cols = set(df.columns)
    for name, (h, d, a) in ODDS_SETS:
        if h in cols and d in cols and a in cols:
            avail.append(name)
    return avail


def get_odds_cols(odds_set_name: str) -> Optional[Tuple[str, str, str]]:
    for name, cols in ODDS_SETS:
        if name == odds_set_name:
            return cols
    return None


def match_market_odds_from_history(df_all: pd.DataFrame, home: str, away: str, odds_cols: Tuple[str, str, str]) -> Optional[Tuple[float, float, float]]:
    """
    Maç geçmişte varsa aynı fixture'ın en yakın tarihli odds'unu al.
    """
    hcol, dcol, acol = odds_cols
    m = df_all[(df_all["HomeTeam"] == home) & (df_all["AwayTeam"] == away)].dropna(subset=[hcol, dcol, acol]).copy()
    if m.empty:
        return None
    m = m.sort_values("Date")
    row = m.iloc[-1]
    o1, ox, o2 = safe_float(row[hcol]), safe_float(row[dcol]), safe_float(row[acol])
    if np.isfinite(o1) and np.isfinite(ox) and np.isfinite(o2):
        return o1, ox, o2
    return None


def try_fixture_odds(fixtures_df: pd.DataFrame, div: str, home: str, away: str, odds_cols: Tuple[str, str, str]) -> Optional[Tuple[float, float, float]]:
    """
    fixtures.csv içinden yaklaşan maç odds'u (varsa) yakalamaya çalış.
    """
    if fixtures_df is None or fixtures_df.empty:
        return None
    hcol, dcol, acol = odds_cols
    cols = set(fixtures_df.columns)
    if not (hcol in cols and dcol in cols and acol in cols):
        return None
    f = fixtures_df.copy()
    if "Div" in f.columns:
        f = f[f["Div"] == div]
    f = f[(f["HomeTeam"] == home) & (f["AwayTeam"] == away)].dropna(subset=[hcol, dcol, acol])
    if f.empty:
        return None
    f = f.sort_values("Date")
    row = f.iloc[0]
    o1, ox, o2 = safe_float(row[hcol]), safe_float(row[dcol]), safe_float(row[acol])
    if np.isfinite(o1) and np.isfinite(ox) and np.isfinite(o2):
        return o1, ox, o2
    return None


def reliability_bins(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Calibration / reliability table.
    """
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
    # y_idx: 0=H,1=D,2=A ; p3 shape (n,3)
    n = len(y_idx)
    y_onehot = np.zeros_like(p3)
    y_onehot[np.arange(n), y_idx] = 1.0
    return float(np.mean(np.sum((p3 - y_onehot) ** 2, axis=1)))


def logloss_multiclass(y_idx: np.ndarray, p3: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p3[np.arange(len(y_idx)), y_idx], eps, 1.0)
    return float(-np.mean(np.log(p)))


# =========================
# UI
# =========================
st.title("⚽ Pro Match Analytics (Season-independent)")
st.caption(
    "Sezon seçmeden çalışır: seçtiğin tarihe kadar olan maçlardan son 20 maç formu ile "
    "1X2, İY/MS, O/U, BTTS, korner gibi marketler için olasılık üretir; "
    "market odds bağlayıp edge/value gösterir; backtest + calibration yapar."
)

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
    lookback = st.slider("Form: son kaç maç (home/away ayrı)", 10, 30, 20, 1)
    min_matches = st.slider("Takım için min maç (home+away)", 4, 20, 8, 1)

    st.divider()
    half_life = st.slider("Lig ortalaması ağırlık yarı-ömür (gün)", 15, 180, 60, 5)
    prior_weight = st.slider("Shrinkage (prior ağırlığı)", 0, 30, 10, 1)
    back_seasons = st.slider("Geriye kaç sezon indirilsin", 1, 8, 5, 1)
    max_goals = st.slider("Skor matrisi max gol", 4, 10, 6, 1)

    st.divider()
    st.subheader("Odds / Market")
    odds_source = st.radio("Market odds kaynağı", ["Dataset (football-data)", "CSV Upload", "Manual"], index=0)
    blend_w = st.slider("Model vs Market blend (0=Model, 1=Market)", 0.0, 1.0, 0.0, 0.05)
    st.caption("Blend, tahminleri genelde iyileştirir: market bilgisi güçlüdür. 0.2–0.5 iyi başlangıç.")

    st.divider()
    st.subheader("Backtest")
    bt_max_matches = st.slider("Backtest: son kaç maçı değerlendir", 200, 2000, 800, 100)
    bt_run = st.button("Backtest çalıştır", use_container_width=True)


# Load history
try:
    df_all, seasons_loaded = load_multi_season_history(div, anchor_date, back_seasons=back_seasons)
    if df_all.empty:
        st.error("Analiz tarihine kadar maç bulunamadı.")
        st.stop()
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

league_avgs = league_weighted_avgs(df_all, anchor_date, half_life_days=half_life)
team_rates = build_team_rates_recent(df_all, league_avgs, lookback=lookback, prior_weight=prior_weight, min_matches=min_matches)

teams = sorted(team_rates.keys())
if len(teams) < 4:
    st.warning("Yeterli takım verisi oluşmadı. min maç düşür veya back seasons artır.")
    st.stop()

# Fixtures (optional)
fixtures_df = None
try:
    fixtures_df = parse_dates(load_csv(FIXTURES_URL))
except Exception:
    fixtures_df = None

# Summary row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lig/Div", div)
c2.metric("Yüklenen sezonlar", ", ".join(seasons_loaded[:4]) + ("..." if len(seasons_loaded) > 4 else ""))
c3.metric("Lig avg goller (H/A)", f"{league_avgs['home_goals']:.2f} / {league_avgs['away_goals']:.2f}")
c4.metric("Korner verisi", "Var" if league_avgs.get("has_corners") else "Yok")

st.divider()

tabs = st.tabs(["🎯 Match Analyzer", "📈 Backtest + Calibration", "🧩 Data & Odds"])


# =========================
# TAB: MATCH ANALYZER
# =========================
with tabs[0]:
    st.subheader("Maç Analizi")

    a, b = st.columns(2)
    with a:
        home_team = st.selectbox("Home", teams, index=0)
    with b:
        away_team = st.selectbox("Away", teams, index=min(1, len(teams) - 1))

    if home_team == away_team:
        st.warning("Home ve Away aynı olamaz.")
        st.stop()

    # show last 20
    with st.expander("📌 Form (son 20 maç)"):
        lcol, rcol = st.columns(2)
        with lcol:
            st.markdown(f"**{home_team} – Son {lookback} maç (genel tablo 20 gösterir)**")
            st.dataframe(last_matches_table(df_all, home_team, n=20), use_container_width=True, height=360)
        with rcol:
            st.markdown(f"**{away_team} – Son {lookback} maç (genel tablo 20 gösterir)**")
            st.dataframe(last_matches_table(df_all, away_team, n=20), use_container_width=True, height=360)

    # model
    lam_h, lam_a = expected_goals(team_rates, league_avgs, home_team, away_team)
    mat = score_matrix(lam_h, lam_a, max_goals=max_goals)
    pH, pD, pA = outcome_probs_from_mat(mat)

    p_over25 = prob_over_from_mat(mat, 2.5)
    p_btts = prob_btts_from_mat(mat)

    # First half
    ht_ratio_home = league_avgs.get("ht_ratio_home", 0.45)
    ht_ratio_away = league_avgs.get("ht_ratio_away", 0.45)
    mat_ht = score_matrix(max(0.01, lam_h * ht_ratio_home), max(0.01, lam_a * ht_ratio_away), max_goals=max_goals)
    pH_ht, pD_ht, pA_ht = outcome_probs_from_mat(mat_ht)
    p_over05_ht = prob_over_from_mat(mat_ht, 0.5)
    p_over15_ht = prob_over_from_mat(mat_ht, 1.5)

    # HT/FT
    htft = htft_probs(lam_h, lam_a, ht_ratio_home, ht_ratio_away, max_goals=max_goals)

    # Corners (if exist)
    corners = expected_corners(team_rates, league_avgs, home_team, away_team)
    if corners is not None:
        lam_hc, lam_ac = corners
        mat_c = score_matrix(lam_hc, lam_ac, max_goals=12)
        p_c_over95 = prob_over_from_mat(mat_c, 9.5)
        p_c_over105 = prob_over_from_mat(mat_c, 10.5)
    else:
        lam_hc = lam_ac = None
        p_c_over95 = p_c_over105 = None

    # derived markets
    p_1x = pH + pD
    p_x2 = pD + pA
    p_12 = pH + pA

    # DNB (draw no bet) fair probs
    pH_dnb = pH / (pH + pA) if (pH + pA) > 0 else np.nan
    pA_dnb = pA / (pH + pA) if (pH + pA) > 0 else np.nan

    # market odds integration
    market_odds = None
    market_probs = None
    chosen_odds_set = None

    if odds_source == "Dataset (football-data)":
        avail_sets = find_available_odds_sets(df_all)
        if not avail_sets:
            st.info("Bu lig datasında 1X2 odds kolonları bulunamadı. (CSV Upload veya Manual kullan)")
        else:
            chosen_odds_set = st.selectbox("Odds set", avail_sets, index=0)
            odds_cols = get_odds_cols(chosen_odds_set)
            market_odds = try_fixture_odds(fixtures_df, div, home_team, away_team, odds_cols) if odds_cols else None
            if market_odds is None and odds_cols:
                market_odds = match_market_odds_from_history(df_all, home_team, away_team, odds_cols)
            if market_odds is not None:
                market_probs = implied_probs_1x2(*market_odds)

    elif odds_source == "CSV Upload":
        st.caption("CSV format: Date, Div, HomeTeam, AwayTeam, H, D, A (odds). Tarih dayfirst olabilir.")
        up = st.file_uploader("Odds CSV yükle", type=["csv"])
        if up is not None:
            try:
                mdf = pd.read_csv(up)
                if "Date" in mdf.columns:
                    mdf["Date"] = pd.to_datetime(mdf["Date"], dayfirst=True, errors="coerce")
                for col in ["H", "D", "A"]:
                    mdf[col] = pd.to_numeric(mdf[col], errors="coerce")
                # match today/closest future
                tmp = mdf.copy()
                if "Div" in tmp.columns:
                    tmp = tmp[tmp["Div"].astype(str).str.upper() == div]
                tmp = tmp[(tmp["HomeTeam"] == home_team) & (tmp["AwayTeam"] == away_team)]
                tmp = tmp.dropna(subset=["H", "D", "A"])
                if not tmp.empty:
                    row = tmp.sort_values("Date").iloc[-1]
                    market_odds = (float(row["H"]), float(row["D"]), float(row["A"]))
                    market_probs = implied_probs_1x2(*market_odds)
            except Exception as e:
                st.error(f"CSV okunamadı: {e}")

    elif odds_source == "Manual":
        o1 = st.number_input("Home odds", min_value=1.01, value=2.10, step=0.01)
        ox = st.number_input("Draw odds", min_value=1.01, value=3.20, step=0.01)
        o2 = st.number_input("Away odds", min_value=1.01, value=3.50, step=0.01)
        market_odds = (float(o1), float(ox), float(o2))
        market_probs = implied_probs_1x2(*market_odds)

    # Blend
    p_model = np.array([pH, pD, pA], dtype=float)
    if market_probs is not None and all(np.isfinite(market_probs)):
        p_mkt = np.array(market_probs, dtype=float)
        p_final = (1 - blend_w) * p_model + blend_w * p_mkt
        p_final = p_final / p_final.sum()
    else:
        p_mkt = None
        p_final = p_model

    # display
    L, M, R = st.columns([1.1, 1.0, 1.3], vertical_alignment="top")

    with L:
        st.markdown("### 1X2 (FT)")
        st.write(f"Model: **H {pH*100:.1f}% | D {pD*100:.1f}% | A {pA*100:.1f}%**")
        st.write(f"Final (blend): **H {p_final[0]*100:.1f}% | D {p_final[1]*100:.1f}% | A {p_final[2]*100:.1f}%**")
        st.caption(f"Expected goals: {lam_h:.2f} - {lam_a:.2f}")

        st.markdown("### Diğer FT marketler")
        st.write(f"Over 2.5: **{p_over25*100:.1f}%** (Fair {1/p_over25:.2f})")
        st.write(f"BTTS (KG Var): **{p_btts*100:.1f}%** (Fair {1/p_btts:.2f})")
        st.write(f"Double chance 1X: **{p_1x*100:.1f}%** | X2: **{p_x2*100:.1f}%** | 12: **{p_12*100:.1f}%**")
        st.write(f"DNB (beraberlik iade): Home **{pH_dnb*100:.1f}%**, Away **{pA_dnb*100:.1f}%**")

        st.markdown("### Toplam gol exact (0–6)")
        exact = {g: prob_total_exact(mat, g) for g in range(0, 7)}
        exdf = pd.DataFrame({"TotalGoals": list(exact.keys()), "Prob": list(exact.values())})
        exdf["FairOdds"] = exdf["Prob"].apply(lambda p: (1 / p) if p > 0 else np.nan)
        st.dataframe(exdf, use_container_width=True, height=250)

    with M:
        st.markdown("### İlk Yarı")
        st.write(f"İY 1X2: **H {pH_ht*100:.1f}% | D {pD_ht*100:.1f}% | A {pA_ht*100:.1f}%**")
        st.write(f"İY Over 0.5: **{p_over05_ht*100:.1f}%** | İY Over 1.5: **{p_over15_ht*100:.1f}%**")

        st.markdown("### İY / MS (HT/FT)")
        htft_df = pd.DataFrame({"HT/FT": list(htft.keys()), "Prob": list(htft.values())}).sort_values("Prob", ascending=False).head(9)
        htft_df["FairOdds"] = htft_df["Prob"].apply(lambda p: (1 / p) if p > 0 else np.nan)
        st.dataframe(htft_df, use_container_width=True, height=320)

        st.markdown("### Doğru Skor (Top)")
        st.dataframe(top_scores(mat, topn=10), use_container_width=True, height=320)

    with R:
        st.markdown("### Market Odds → Edge / EV")
        if market_odds is None or market_probs is None or p_mkt is None:
            st.info("Market odds bağlı değil. Sidebar → Odds kaynağı seç.")
        else:
            st.write(f"Odds set: **{chosen_odds_set or odds_source}**")
            st.write(f"Odds: **{market_odds[0]:.2f} / {market_odds[1]:.2f} / {market_odds[2]:.2f}**")
            st.write(f"Market prob (vig removed): **{p_mkt[0]*100:.1f}% / {p_mkt[1]*100:.1f}% / {p_mkt[2]*100:.1f}%**")

            edge = p_final - p_mkt
            ev = p_final * np.array(market_odds) - 1.0  # expected return per 1 unit stake

            out = pd.DataFrame({
                "Outcome": ["Home", "Draw", "Away"],
                "P_final": p_final,
                "P_market": p_mkt,
                "Edge(P_final-P_mkt)": edge,
                "Odds": market_odds,
                "EV (p*odds-1)": ev,
                "FairOdds": 1.0 / np.clip(p_final, 1e-12, 1.0),
            })
            out["P_final"] = (out["P_final"] * 100).round(2)
            out["P_market"] = (out["P_market"] * 100).round(2)
            out["Edge(P_final-P_mkt)"] = (out["Edge(P_final-P_mkt)"] * 100).round(2)
            out["EV (p*odds-1)"] = out["EV (p*odds-1)"].round(3)
            out["FairOdds"] = out["FairOdds"].round(2)
            st.dataframe(out, use_container_width=True, height=260)

            st.caption("EV>0 olan outcome'lar (model/market farkı) teorik 'value' göstergesidir; garanti değildir.")

        st.markdown("### Korner")
        if corners is None:
            st.info("Bu lig datasında korner kolonları yok (HC/AC).")
        else:
            st.write(f"Expected corners: **{lam_hc:.2f} - {lam_ac:.2f}** (Total {lam_hc+lam_ac:.2f})")
            st.write(f"Total corners Over 9.5: **{p_c_over95*100:.1f}%** (Fair {1/p_c_over95:.2f})")
            st.write(f"Total corners Over 10.5: **{p_c_over105*100:.1f}%** (Fair {1/p_c_over105:.2f})")


# =========================
# TAB: BACKTEST
# =========================
with tabs[1]:
    st.subheader("Backtest + Calibration")

    st.caption(
        "Backtest mantığı: her maç için o maçtan ÖNCEKİ verilere göre (son 20 form) olasılık üretir. "
        "Sonra gerçekleşen sonuçla Brier/Logloss ve calibration hesaplar. (Leakage yok)"
    )

    # Prepare backtest slice (last N matches)
    d = df_all.sort_values("Date").copy()
    d = d.tail(bt_max_matches).copy()

    # Need enough data
    if len(d) < 200:
        st.info("Backtest için maç sayısı az görünüyor. back_seasons artır veya lig değiştir.")

    def backtest_compute(dslice: pd.DataFrame) -> pd.DataFrame:
        """
        Sequential backtest:
        - match i prediction uses history < i
        - stats built from rolling last N home/away matches (by slicing)
        Bu yöntem en doğru ama O(n^2) olabilir.
        O yüzden N=800 default + caching + optimize: her maçta sadece iki takımın son N home/away listesini kullanıyoruz.
        """
        # We'll keep per team home/away deques of recent goals
        from collections import deque

        look = lookback
        prior = prior_weight
        minm = min_matches
        maxg = max_goals
        hl = half_life

        # Histories
        home_for = {}
        home_against = {}
        away_for = {}
        away_against = {}

        # Corners histories if exist
        has_corners = league_avgs.get("has_corners")
        home_cf = {}
        home_ca = {}
        away_cf = {}
        away_ca = {}

        # expanding league avgs (simple, stable)
        lg_home_goals = []
        lg_away_goals = []
        lg_hthg = []
        lg_htag = []

        preds = []

        for idx, row in dslice.iterrows():
            date = row["Date"]
            ht = row["HomeTeam"]
            at = row["AwayTeam"]
            fthg = int(row["FTHG"])
            ftag = int(row["FTAG"])

            # league means so far
            if len(lg_home_goals) < 50:
                # not enough data, skip prediction (warm-up)
                pred = None
            else:
                lhg = float(np.mean(lg_home_goals))
                lag = float(np.mean(lg_away_goals))

                # ht ratio
                if len(lg_hthg) >= 50 and lhg > 0 and lag > 0:
                    ht_ratio_h = float(np.clip(np.mean(lg_hthg) / lhg, 0.25, 0.65))
                    ht_ratio_a = float(np.clip(np.mean(lg_htag) / lag, 0.25, 0.65))
                else:
                    ht_ratio_h = 0.45
                    ht_ratio_a = 0.45

                def get_deq(dct, key):
                    if key not in dct:
                        dct[key] = deque(maxlen=look)
                    return dct[key]

                hf = np.array(list(get_deq(home_for, ht)), dtype=float)
                ha = np.array(list(get_deq(home_against, ht)), dtype=float)
                af = np.array(list(get_deq(away_for, at)), dtype=float)
                aa = np.array(list(get_deq(away_against, at)), dtype=float)

                # if insufficient team matches, mark None
                if (len(hf) + len(ha) + len(af) + len(aa)) < minm:
                    pred = None
                else:
                    home_scored = shrink_mean(hf, lhg, prior)
                    home_conceded = shrink_mean(ha, lag, prior)
                    away_scored = shrink_mean(af, lag, prior)
                    away_conceded = shrink_mean(aa, lhg, prior)

                    ha_mult = home_scored / lhg if lhg > 0 else 1.0
                    hd_mult = home_conceded / lag if lag > 0 else 1.0
                    aa_mult = away_scored / lag if lag > 0 else 1.0
                    ad_mult = away_conceded / lhg if lhg > 0 else 1.0

                    lam_h = float(np.clip(lhg * ha_mult * ad_mult, 0.05, 4.5))
                    lam_a = float(np.clip(lag * aa_mult * hd_mult, 0.05, 4.5))

                    mat = score_matrix(lam_h, lam_a, max_goals=maxg)
                    pH, pD, pA = outcome_probs_from_mat(mat)
                    p_over25 = prob_over_from_mat(mat, 2.5)
                    p_btts = prob_btts_from_mat(mat)

                    # store
                    pred = {
                        "Date": date,
                        "HomeTeam": ht,
                        "AwayTeam": at,
                        "FTHG": fthg,
                        "FTAG": ftag,
                        "pH": pH,
                        "pD": pD,
                        "pA": pA,
                        "pOver25": p_over25,
                        "pBTTS": p_btts,
                        "ht_ratio_home": ht_ratio_h,
                        "ht_ratio_away": ht_ratio_a,
                    }

                    # add odds if available in row (for ROI later)
                    for name, (hcol, dcol, acol) in ODDS_SETS:
                        if hcol in dslice.columns and dcol in dslice.columns and acol in dslice.columns:
                            pred[f"{name}_H"] = safe_float(row.get(hcol))
                            pred[f"{name}_D"] = safe_float(row.get(dcol))
                            pred[f"{name}_A"] = safe_float(row.get(acol))

            # Update histories AFTER prediction (no leakage)
            from collections import deque

            def push(dct, key, val):
                if key not in dct:
                    dct[key] = deque(maxlen=look)
                dct[key].append(val)

            push(home_for, ht, fthg)
            push(home_against, ht, ftag)
            push(away_for, at, ftag)
            push(away_against, at, fthg)

            lg_home_goals.append(fthg)
            lg_away_goals.append(ftag)

            if "HTHG" in dslice.columns and "HTAG" in dslice.columns:
                hthg = safe_float(row.get("HTHG"))
                htag = safe_float(row.get("HTAG"))
                if np.isfinite(hthg) and np.isfinite(htag):
                    lg_hthg.append(hthg)
                    lg_htag.append(htag)

            if pred is not None:
                preds.append(pred)

        return pd.DataFrame(preds)

    @st.cache_data(show_spinner=False)
    def cached_backtest(dslice: pd.DataFrame) -> pd.DataFrame:
        return backtest_compute(dslice)

    if bt_run:
        bt = cached_backtest(d)
        if bt.empty:
            st.warning("Backtest sonucu üretilemedi (warm-up çok kısa olabilir). back_seasons veya bt_match sayısını artır.")
            st.stop()

        # metrics 1X2
        y = bt.apply(lambda r: 0 if r["FTHG"] > r["FTAG"] else (2 if r["FTHG"] < r["FTAG"] else 1), axis=1).values
        p3 = bt[["pH", "pD", "pA"]].values

        brier = brier_multiclass(y, p3)
        ll = logloss_multiclass(y, p3)

        c1, c2, c3 = st.columns(3)
        c1.metric("Matches evaluated", f"{len(bt)}")
        c2.metric("Brier (1X2)", f"{brier:.4f}")
        c3.metric("LogLoss (1X2)", f"{ll:.4f}")

        # Over2.5 & BTTS metrics
        y_over = (bt["FTHG"] + bt["FTAG"] > 2.5).astype(int).values
        p_over = bt["pOver25"].values
        brier_over = float(np.mean((p_over - y_over) ** 2))
        y_btts = ((bt["FTHG"] > 0) & (bt["FTAG"] > 0)).astype(int).values
        p_btts = bt["pBTTS"].values
        brier_btts = float(np.mean((p_btts - y_btts) ** 2))

        c4, c5 = st.columns(2)
        c4.metric("Brier (Over2.5)", f"{brier_over:.4f}")
        c5.metric("Brier (BTTS)", f"{brier_btts:.4f}")

        st.divider()

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
        if cal.empty:
            st.info("Calibration verisi üretilemedi.")
        else:
            st.dataframe(cal, use_container_width=True)

            # Simple chart (Streamlit line chart)
            chart_df = cal[["p_mean", "y_rate"]].dropna().copy()
            chart_df = chart_df.rename(columns={"p_mean": "Predicted", "y_rate": "Observed"})
            st.line_chart(chart_df, use_container_width=True)

        st.divider()

        st.markdown("### Basit Value/ROI simülasyonu (opsiyonel)")
        st.caption("Bu bölüm araştırma amaçlıdır. EV>0 filtresiyle basit kural tabanlı simülasyon.")

        avail_sets = []
        for name, (hcol, dcol, acol) in ODDS_SETS:
            if f"{name}_H" in bt.columns:
                avail_sets.append(name)

        if not avail_sets:
            st.info("Backtest datasında odds set yok. (Lig dosyasında odds kolonları olmayabilir.)")
        else:
            sim_set = st.selectbox("Odds set (sim)", avail_sets, index=0)
            ev_threshold = st.slider("EV eşik", 0.0, 0.2, 0.02, 0.01)
            stake_mode = st.selectbox("Stake", ["Flat 1 unit", "Kelly (scaled)"], index=0)
            kelly_scale = st.slider("Kelly scale", 0.0, 1.0, 0.25, 0.05)

            oH = bt[f"{sim_set}_H"].values.astype(float)
            oD = bt[f"{sim_set}_D"].values.astype(float)
            oA = bt[f"{sim_set}_A"].values.astype(float)

            # Use model probs (no market blending in backtest sim)
            pH = bt["pH"].values
            pD = bt["pD"].values
            pA = bt["pA"].values

            bankroll = 0.0
            bets = 0
            profit_series = []

            for i in range(len(bt)):
                if not (np.isfinite(oH[i]) and np.isfinite(oD[i]) and np.isfinite(oA[i])):
                    profit_series.append(bankroll)
                    continue

                probs = np.array([pH[i], pD[i], pA[i]], dtype=float)
                odds = np.array([oH[i], oD[i], oA[i]], dtype=float)
                ev = probs * odds - 1.0
                j = int(np.argmax(ev))
                if ev[j] <= ev_threshold:
                    profit_series.append(bankroll)
                    continue

                # stake
                stake = 1.0
                if stake_mode.startswith("Kelly"):
                    b = odds[j] - 1.0
                    p = probs[j]
                    q = 1.0 - p
                    f = (b * p - q) / b if b > 0 else 0.0
                    f = max(0.0, f) * kelly_scale
                    stake = f  # as "unit" fraction
                    if stake <= 0:
                        profit_series.append(bankroll)
                        continue

                bets += 1
                # outcome
                actual = result_label(int(bt.iloc[i]["FTHG"]), int(bt.iloc[i]["FTAG"]))
                pick = ["H", "D", "A"][j]
                if pick == actual:
                    bankroll += stake * (odds[j] - 1.0)
                else:
                    bankroll -= stake

                profit_series.append(bankroll)

            st.write(f"Bet count: **{bets}** | Net profit (units): **{bankroll:.2f}**")
            st.line_chart(pd.DataFrame({"bankroll_units": profit_series}), use_container_width=True)

    else:
        st.info("Backtest için soldan **Backtest çalıştır** butonuna bas.")


# =========================
# TAB: DATA & ODDS
# =========================
with tabs[2]:
    st.subheader("Data & Odds Diagnoser")

    st.markdown("### Dataset kolonları")
    st.write("Toplam kolon sayısı:", len(df_all.columns))
    st.code(", ".join(list(df_all.columns)[:60]) + (" ..." if len(df_all.columns) > 60 else ""))

    st.markdown("### Mevcut 1X2 odds setleri")
    avail = find_available_odds_sets(df_all)
    if not avail:
        st.info("Bu lig datasında 1X2 odds kolonları bulunamadı.")
    else:
        st.success("Bulunan odds setleri: " + ", ".join(avail))

    st.markdown("### Fixtures (yaklaşan maçlar)")
    if fixtures_df is None or fixtures_df.empty:
        st.info("fixtures.csv çekilemedi.")
    else:
        f = fixtures_df.copy()
        if "Div" in f.columns:
            f = f[f["Div"].astype(str).str.upper() == div]
        f = f.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date")
        f = f[f["Date"] >= pd.Timestamp(anchor_date)]
        st.dataframe(f.head(25), use_container_width=True, height=420)

    st.markdown("### Odds CSV template (iddaa gibi manuel bağlamak için)")
    st.code(
        "Date,Div,HomeTeam,AwayTeam,H,D,A\n"
        "2026-02-28,T1,TeamA,TeamB,2.10,3.20,3.50\n",
        language="text"
    )

    st.caption(
        "Team isimleri birebir eşleşmeli. Eğer farklı yazımlar varsa (kısaltma vs.), "
        "CSV tarafında dataset'teki isimle aynı yapman gerekir."
    )
