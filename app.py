import math
import datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Match Analytics (Football MVP)", layout="wide")

LEAGUES = {
    "Turkey - T1": "T1",
    "England - Premier League (E0)": "E0",
    "England - Championship (E1)": "E1",
    "Spain - La Liga (SP1)": "SP1",
    "Italy - Serie A (I1)": "I1",
    "Germany - Bundesliga (D1)": "D1",
    "France - Ligue 1 (F1)": "F1",
    "Netherlands - Eredivisie (N1)": "N1",
    "Portugal - Primeira Liga (P1)": "P1",
}

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

def season_code(start_year: int) -> str:
    y1 = start_year % 100
    y2 = (start_year + 1) % 100
    return f"{y1:02d}{y2:02d}"

def hist_url(season: str, div: str) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

@st.cache_data(show_spinner=False)
def load_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    txt = r.content.decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(txt))

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def score_matrix(lam_home: float, lam_away: float, max_goals: int = 6) -> np.ndarray:
    hp = np.array([poisson_pmf(i, lam_home) for i in range(max_goals + 1)], dtype=float)
    ap = np.array([poisson_pmf(i, lam_away) for i in range(max_goals + 1)], dtype=float)
    mat = np.outer(hp, ap)
    return mat / mat.sum()

def outcome_probs(mat: np.ndarray) -> tuple[float, float, float]:
    p_draw = float(np.trace(mat))
    p_home = float(np.tril(mat, -1).sum())  # i>j
    p_away = float(np.triu(mat,  1).sum())  # i<j
    return p_home, p_draw, p_away

def top_scores(mat: np.ndarray, topn: int = 7) -> pd.DataFrame:
    items = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            items.append((i, j, mat[i, j]))
    items.sort(key=lambda x: x[2], reverse=True)
    out = pd.DataFrame(items[:topn], columns=["HomeGoals", "AwayGoals", "Prob"])
    out["FairOdds"] = out["Prob"].apply(lambda p: (1.0 / p) if p > 0 else np.nan)
    return out

def prob_over(mat: np.ndarray, line: float) -> float:
    p = 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (i + j) > line:
                p += mat[i, j]
    return float(p)

def prob_btts(mat: np.ndarray) -> float:
    return float(mat[1:, 1:].sum())

def implied_probs_from_odds(o1: float, ox: float, o2: float) -> tuple[float, float, float]:
    p1 = 1.0 / o1 if o1 and o1 > 0 else np.nan
    px = 1.0 / ox if ox and ox > 0 else np.nan
    p2 = 1.0 / o2 if o2 and o2 > 0 else np.nan
    s = p1 + px + p2
    if not np.isfinite(s) or s <= 0:
        return (np.nan, np.nan, np.nan)
    return (p1 / s, px / s, p2 / s)

def build_team_strength(df: pd.DataFrame, half_life_days: int = 75, min_matches: int = 8):
    req_cols = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(list(req_cols - set(df.columns)))}")

    d = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    d = d.sort_values("Date")
    if d.empty:
        raise ValueError("No usable rows after cleaning (check Date parsing).")

    last_date = d["Date"].max()
    days_ago = (last_date - d["Date"]).dt.days.clip(lower=0)
    w = np.exp(-days_ago / float(half_life_days))

    def wavg(x):
        return float(np.average(x, weights=w.loc[x.index]))

    league_home_avg = wavg(d["FTHG"])
    league_away_avg = wavg(d["FTAG"])

    ht_ratio_home = 0.45
    ht_ratio_away = 0.45
    if {"HTHG", "HTAG"}.issubset(d.columns):
        hthg = d["HTHG"].astype(float)
        htag = d["HTAG"].astype(float)
        if (hthg.notna().sum() > 50) and league_home_avg > 0 and league_away_avg > 0:
            ht_ratio_home = max(0.25, min(0.65, wavg(hthg) / league_home_avg))
            ht_ratio_away = max(0.25, min(0.65, wavg(htag) / league_away_avg))

    teams = sorted(set(d["HomeTeam"]).union(set(d["AwayTeam"])))

    team_rates = {}
    for t in teams:
        home = d[d["HomeTeam"] == t]
        away = d[d["AwayTeam"] == t]
        if len(home) + len(away) < min_matches:
            continue

        def twavg(series, idx):
            ww = w.loc[idx]
            vals = pd.to_numeric(series, errors="coerce")
            m = vals.notna()
            if m.sum() == 0:
                return np.nan
            return float(np.average(vals[m], weights=ww[m.index]))

        home_scored = twavg(home["FTHG"], home.index)
        home_conceded = twavg(home["FTAG"], home.index)
        away_scored = twavg(away["FTAG"], away.index)
        away_conceded = twavg(away["FTHG"], away.index)

        ha = home_scored / league_home_avg if league_home_avg > 0 else np.nan
        hd = home_conceded / league_away_avg if league_away_avg > 0 else np.nan
        aa = away_scored / league_away_avg if league_away_avg > 0 else np.nan
        ad = away_conceded / league_home_avg if league_home_avg > 0 else np.nan

        team_rates[t] = {
            "home_attack": ha,
            "home_def": hd,
            "away_attack": aa,
            "away_def": ad,
        }

    league_avgs = {
        "home_goals": league_home_avg,
        "away_goals": league_away_avg,
    }
    ht_ratio = {"home": ht_ratio_home, "away": ht_ratio_away}
    return team_rates, league_avgs, ht_ratio, last_date

def expected_goals(team_rates, league_avgs, home_team: str, away_team: str) -> tuple[float, float]:
    lh = league_avgs["home_goals"]
    la = league_avgs["away_goals"]
    h = team_rates.get(home_team)
    a = team_rates.get(away_team)
    if h is None or a is None:
        raise ValueError("Selected teams don't have enough matches for stable estimates (min_matches).")

    lam_home = lh * h["home_attack"] * a["away_def"]
    lam_away = la * a["away_attack"] * h["home_def"]
    lam_home = float(np.clip(lam_home, 0.05, 4.5))
    lam_away = float(np.clip(lam_away, 0.05, 4.5))
    return lam_home, lam_away

st.title("⚽ Football Match Analytics (MVP)")
st.caption("Olasılık tabanlı analiz: 1X2, İY (yaklaşık), doğru skor, O/U, KG Var. Eğlence/araştırma amaçlıdır.")

with st.sidebar:
    st.header("Ayarlar")
    league_name = st.selectbox("Lig", list(LEAGUES.keys()), index=0)
    div = LEAGUES[league_name]
    year = st.selectbox("Sezon başlangıç yılı", list(range(2017, dt.datetime.now().year + 1))[::-1])
    season = season_code(year)
    half_life = st.slider("Form yarı-ömür (gün)", 30, 180, 75, 5)
    min_matches = st.slider("Takım için min maç", 4, 20, 8, 1)
    max_goals = st.slider("Skor matrisi max gol", 4, 10, 6, 1)
    st.divider()
    st.write("Kaynak URL:")
    st.code(hist_url(season, div), language="text")

try:
    df_hist = parse_dates(load_csv(hist_url(season, div)))
    team_rates, league_avgs, ht_ratio, last_date = build_team_strength(
        df_hist, half_life_days=half_life, min_matches=min_matches
    )
except Exception as e:
    st.error(f"Data load/build error: {e}")
    st.stop()

st.write(f"Sezon: **{season}**, Lig: **{div}**, Son maç tarihi: **{last_date.date()}**")
st.write(f"Weighted avg goller: Home {league_avgs['home_goals']:.2f} | Away {league_avgs['away_goals']:.2f}")
st.write(f"İY/FT oranları (yaklaşık): Home {ht_ratio['home']:.2f} | Away {ht_ratio['away']:.2f}")

st.subheader("Maç Analizi")
teams = sorted(team_rates.keys())
if len(teams) < 4:
    st.warning("Yeterli takım verisi oluşmadı. min_matches’i düşürmeyi deneyebilirsin.")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    home_team = st.selectbox("Home", teams, index=0)
with c2:
    away_team = st.selectbox("Away", teams, index=min(1, len(teams)-1))

if home_team == away_team:
    st.warning("Home ve Away aynı olamaz.")
    st.stop()

lam_h, lam_a = expected_goals(team_rates, league_avgs, home_team, away_team)
mat = score_matrix(lam_h, lam_a, max_goals=max_goals)
pH, pD, pA = outcome_probs(mat)

p_over25 = prob_over(mat, 2.5)
p_under25 = 1.0 - p_over25
p_btts = prob_btts(mat)

mat_ht = score_matrix(lam_h * ht_ratio["home"], lam_a * ht_ratio["away"], max_goals=max_goals)
pH_ht, pD_ht, pA_ht = outcome_probs(mat_ht)

l, r = st.columns([1, 1])

with l:
    st.markdown("### 1X2 (Maç Sonu)")
    st.write(f"Home Win: **{pH*100:.1f}%** (Fair {1/pH:.2f})")
    st.write(f"Draw: **{pD*100:.1f}%** (Fair {1/pD:.2f})")
    st.write(f"Away Win: **{pA*100:.1f}%** (Fair {1/pA:.2f})")
    st.caption(f"Beklenen goller: {lam_h:.2f} - {lam_a:.2f}")

    st.markdown("### Over/Under & BTTS")
    st.write(f"Over 2.5: **{p_over25*100:.1f}%** (Fair {1/p_over25:.2f})")
    st.write(f"Under 2.5: **{p_under25*100:.1f}%** (Fair {1/p_under25:.2f})")
    st.write(f"KG Var (BTTS): **{p_btts*100:.1f}%** (Fair {1/p_btts:.2f})")

with r:
    st.markdown("### İlk Yarı (Yaklaşık)")
    st.write(f"İY Home: **{pH_ht*100:.1f}%** | İY Beraberlik: **{pD_ht*100:.1f}%** | İY Away: **{pA_ht*100:.1f}%**")
    st.markdown("### Doğru Skor (Top)")
    st.dataframe(top_scores(mat, topn=10), use_container_width=True)
