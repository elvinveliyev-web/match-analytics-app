import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.isotonic import IsotonicRegression
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
# SOURCES (değişmedi)
# =========================================================
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
NEW_LEAGUE_URL = "https://www.football-data.co.uk/new/{div}.csv"

LEAGUES = { ... }  # (aynen önceki gibi, tekrar yazmaya gerek yok, kısaltıyorum)
# Lütfen buraya önceki LEAGUES sözlüğünü kopyalayın (yaklaşık 40 satır)

ODDS_SETS = [ ... ]  # aynen

# =========================================================
# YENİ FONKSİYONLAR / GELİŞTİRMELER
# =========================================================

def estimate_xg_coefficients(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Ligdeki şut ve isabetli şut verilerini kullanarak xG proxy katsayılarını
    (a, b) hesaplar: xG ≈ a*Şut + b*İsabetliŞut.
    Basit doğrusal regresyon (gol = a*S + b*ST) kullanılır.
    """
    df2 = df.dropna(subset=["HS", "HST", "FTHG", "AS", "AST", "FTAG"]).copy()
    if len(df2) < 50:
        return 0.04, 0.08   # varsayılan
    X_h = np.column_stack([df2["HS"].values, df2["HST"].values])
    y_h = df2["FTHG"].values
    X_a = np.column_stack([df2["AS"].values, df2["AST"].values])
    y_a = df2["FTAG"].values
    X = np.vstack([X_h, X_a])
    y = np.concatenate([y_h, y_a])
    # en küçük kareler
    coeff, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeff
    return float(np.clip(a, 0.01, 0.1)), float(np.clip(b, 0.01, 0.2))

def xg_proxy_enhanced(shots: float, sot: float, a: float, b: float) -> float:
    if not np.isfinite(shots) and not np.isfinite(sot):
        return np.nan
    s = shots if np.isfinite(shots) else 0.0
    t = sot if np.isfinite(sot) else 0.0
    return a * s + b * t

def compute_rest_days(team: str, match_date: pd.Timestamp, all_matches: pd.DataFrame) -> int:
    """Takımın verilen tarihten önceki son maçına kadar geçen gün sayısı."""
    prev = all_matches[
        (all_matches["Date"] < match_date) &
        ((all_matches["HomeTeam"] == team) | (all_matches["AwayTeam"] == team))
    ].sort_values("Date").tail(1)
    if prev.empty:
        return 7   # varsayılan (ortalama dinlenme)
    last_date = prev.iloc[0]["Date"]
    return (match_date - last_date).days

def h2h_features(home: str, away: str, all_matches: pd.DataFrame, n_last: int = 5) -> Tuple[float, float]:
    """Son n karşılaşmadaki ev sahibi ve deplasman golleri ortalaması."""
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
    """
    Tüm maçlardan ρ (rho) parametresini tahmin eder.
    Dixon‑Coles düzeltmesi için kullanılır.
    """
    if len(df) < 100:
        return -0.1   # varsayılan (zayıf negatif korelasyon)
    # Basit yöntem: 0-0, 1-1 gibi düşük skorlu beraberliklerin oranına bak
    # Gerçek uygulama maksimum olabilirlik gerektirir, burada pratik bir yaklaşım
    total = len(df)
    low_draws = df[(df["FTHG"] == df["FTAG"]) & (df["FTHG"] <= 1)]   # 0-0 ve 1-1
    observed = len(low_draws) / total
    # Bağımsız Poisson altında beklenen düşük beraberlik oranı
    avg_h = df["FTHG"].mean()
    avg_a = df["FTAG"].mean()
    p00 = math.exp(-avg_h) * math.exp(-avg_a)
    p11 = (avg_h * math.exp(-avg_h)) * (avg_a * math.exp(-avg_a))
    expected = p00 + p11
    # rho ≈ (observed / expected) - 1, ancak sınırlandır
    rho = (observed / expected) - 1.0
    return float(np.clip(rho, -0.3, 0.3))

def dixon_coles_adjustment(mat: np.ndarray, lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """
    Dixon‑Coles düzeltmesini uygular: τ(i,j) = 1 - ρ * i * j   (basitleştirilmiş)
    Gerçek formül daha karmaşık, ancak bu yaklaşım yeterlidir.
    """
    adj = np.ones_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == 0 and j == 0:
                adj[i, j] = 1 - rho * lam_h * lam_a   # yaklaşık
            elif i == 0 and j == 1:
                adj[i, j] = 1 + rho * lam_h   # yaklaşık
            elif i == 1 and j == 0:
                adj[i, j] = 1 + rho * lam_a
            elif i == 1 and j == 1:
                adj[i, j] = 1 - rho
            else:
                adj[i, j] = 1.0
    return mat * adj

def calibrate_probabilities(probs: np.ndarray, outcomes: np.ndarray) -> object:
    """
    Platt scaling (lojistik regresyon) ile olasılıkları kalibre eder.
    probs: (n, 3) boyutunda [p_h, p_d, p_a] tahminleri
    outcomes: (n,) 0=H,1=D,2=A
    Dönen: kalibrasyon modeli (her sınıf için ayrı).
    """
    models = []
    for i in range(3):
        y = (outcomes == i).astype(int)
        # tek değişkenli lojistik regresyon: log(p/(1-p)) = a + b*logit(model)
        X = np.log(probs[:, i] / (1 - probs[:, i] + 1e-12)).reshape(-1, 1)
        lr = LogisticRegression(C=1e5, solver='lbfgs')
        lr.fit(X, y)
        models.append(lr)
    return models

def apply_calibration(models, probs: np.ndarray) -> np.ndarray:
    """Kalibre edilmiş olasılıkları döndürür."""
    cal = np.zeros_like(probs)
    for i, lr in enumerate(models):
        X = np.log(probs[:, i] / (1 - probs[:, i] + 1e-12)).reshape(-1, 1)
        cal[:, i] = lr.predict_proba(X)[:, 1]
    # normalize et
    cal = cal / cal.sum(axis=1, keepdims=True)
    return cal

# =========================================================
# GÜNCELLENMİŞ TeamRates (decay eklendi)
# =========================================================
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

def build_team_rates_recent_decayed(
    df_all: pd.DataFrame,
    league: Dict[str, float],
    lookback: int,
    prior_weight: float,
    min_matches: int,
    decay_halflife: float   # yeni: maç sayısı cinsinden yarı ömür
) -> Dict[str, TeamRates]:
    teams = sorted(set(df_all["HomeTeam"]).union(set(df_all["AwayTeam"])))
    rates: Dict[str, TeamRates] = {}

    for t in teams:
        home = df_all[df_all["HomeTeam"] == t].tail(lookback).copy()
        away = df_all[df_all["AwayTeam"] == t].tail(lookback).copy()

        if (len(home) + len(away)) < min_matches:
            continue

        # Üstel ağırlıklar (son maç en yüksek)
        def decay_weights(n):
            if n == 0:
                return np.array([])
            positions = np.arange(n)
            w = np.exp(-positions / decay_halflife)   # en son maç index 0
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

        # shrink (prior ağırlığı) ile birleştir
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

def shrink_mean_with_prior(weighted_avg: float, prior_mean: float, prior_weight: float, n: int) -> float:
    if np.isnan(weighted_avg):
        return prior_mean
    return (weighted_avg * n + prior_weight * prior_mean) / (n + prior_weight)

# =========================================================
# GÜNCELLENMİŞ expected_goals_from_rates (rest days & h2h eklendi)
# =========================================================
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
    rest_days_diff: float,            # yeni: ev sahibi dinlenme günü - deplasman
    h2h_home_avg: float,               # yeni: ev sahibinin evindeki gol ort.
    h2h_away_avg: float,               # yeni: deplasmanın deplasman gol ort.
    rest_factor_strength: float = 0.05, # her gün farkı için %5
    h2h_weight: float = 0.2             # h2h ortalamasının ağırlığı
) -> Tuple[float, float]:
    h = rates.get(home)
    a = rates.get(away)
    if h is None or a is None:
        raise ValueError("Takım verisi yetersiz.")

    lam_h_goal = league["home_goals"] * h.home_attack * a.away_def
    lam_a_goal = league["away_goals"] * a.away_attack * h.home_def

    lam_h, lam_a = lam_h_goal, lam_a_goal

    # xG blend
    if xg_weight > 0 and h.home_xg_att is not None and a.away_xg_def is not None and a.away_xg_att is not None and h.home_xg_def is not None:
        lam_h_xg = league["home_goals"] * h.home_xg_att * a.away_xg_def
        lam_a_xg = league["away_goals"] * a.away_xg_att * h.home_xg_def
        lam_h = (1 - xg_weight) * lam_h_goal + xg_weight * lam_h_xg
        lam_a = (1 - xg_weight) * lam_a_goal + xg_weight * lam_a_xg

    # ELO
    scale = math.exp((elo_to_goal_k * elo_diff) / 400.0)
    lam_h *= scale
    lam_a *= (1.0 / scale)

    # Dinlenme günü farkı
    if rest_days_diff != 0:
        rest_factor = 1.0 + rest_factor_strength * rest_days_diff   # pozitif fark ev sahibi lehine
        lam_h *= rest_factor
        lam_a /= rest_factor   # deplasman aleyhine

    # Head-to-head geçmiş
    if np.isfinite(h2h_home_avg) and np.isfinite(h2h_away_avg):
        # geçmişteki ortalamaları lig ortalamasına oranla
        h2h_home_ratio = h2h_home_avg / league["home_goals"] if league["home_goals"] > 0 else 1.0
        h2h_away_ratio = h2h_away_avg / league["away_goals"] if league["away_goals"] > 0 else 1.0
        # mevcut beklentiyi ağırlıklı ortalama ile güncelle
        lam_h = (1 - h2h_weight) * lam_h + h2h_weight * (league["home_goals"] * h2h_home_ratio)
        lam_a = (1 - h2h_weight) * lam_a + h2h_weight * (league["away_goals"] * h2h_away_ratio)

    lam_h *= manual_home_factor
    lam_a *= manual_away_factor

    lam_h = float(np.clip(lam_h, 0.05, 4.5))
    lam_a = float(np.clip(lam_a, 0.05, 4.5))
    return lam_h, lam_a

# =========================================================
# YAN ÇUBUK (YENİ PARAMETRELER EKLENDİ)
# =========================================================
with st.sidebar:
    st.header("Ayarlar")
    # ... (lig seçimi, tarih vb. aynen)
    # Araya yeni parametreler ekleyelim:
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

# ... (load_history ve diğer fonksiyonlar aynen)

# =========================================================
# VERİ YÜKLEME VE LİG ORTALAMALARI (xG katsayıları eklendi)
# =========================================================
df_all, seasons_loaded, source_used = load_history(div, anchor_date, back_seasons=back_seasons)
fixtures_df = load_fixtures()

# xG katsayılarını hesapla
xg_a, xg_b = estimate_xg_coefficients(df_all)
st.sidebar.write(f"xG proxy katsayıları: a={xg_a:.3f}, b={xg_b:.3f}")

league = league_avgs_weighted(df_all, anchor_date, half_life_days=half_life_days)
# league içine xG katsayılarını ekle
league["xg_a"] = xg_a
league["xg_b"] = xg_b

# Takım güçlerini üstel azalma ile hesapla
rates = build_team_rates_recent_decayed(
    df_all, league, lookback=lookback, prior_weight=prior_weight,
    min_matches=min_matches, decay_halflife=decay_halflife
)

# ... (ELO hesaplamaları aynen)

# Dixon-Coles rho değerini hesapla
rho = dixon_coles_rho(df_all) if use_dixon_coles else 0.0
st.sidebar.write(f"Dixon-Coles ρ: {rho:.3f}")

# =========================================================
# MAÇ ANALİZİ (güncellenmiş)
# =========================================================
# ... (takım seçimi, ELO farkı vs)

# Dinlenme günü farkını hesapla
all_matches_sorted = df_all.sort_values("Date").copy()
home_last = compute_rest_days(home_team, pd.Timestamp(anchor_date), all_matches_sorted)
away_last = compute_rest_days(away_team, pd.Timestamp(anchor_date), all_matches_sorted)
rest_diff = home_last - away_last   # pozitif: ev sahibi daha dinlenik

# Head-to-head geçmiş
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
    rest_factor_strength=rest_factor_strength,
    h2h_weight=h2h_weight
)

# Dixon-Coles düzeltmeli skor matrisi
mat = score_matrix(lam_h, lam_a, max_goals=max_goals)
if use_dixon_coles and rho != 0:
    mat = dixon_coles_adjustment(mat, lam_h, lam_a, rho)
    mat = mat / mat.sum()   # normalize

pH, pD, pA = outcome_probs_from_mat(mat)

# Olasılıkları dizi haline getir (kalibrasyon için)
probs_array = np.array([[pH, pD, pA]])

# Kalibrasyon uygula (eğer backtest yapıldıysa ve model varsa)
# Not: Kalibrasyon modeli backtest sırasında oluşturulup session_state'e kaydedilebilir.
# Bu örnekte, backtest bölümünde kalibrasyon yapıp sonraki tahminlerde kullanacağız.
# Şimdilik doğrudan gösterelim.

# ... (diğer hesaplamalar aynen)

# =========================================================
# BACKTEST (geliştirilmiş, kalibrasyon dahil)
# =========================================================
def backtest_1x2_value_enhanced( ... ):
    # Önceki backtest fonksiyonuna benzer, ancak yeni özellikleri (rest days, h2h, decay) kullanır.
    # Ayrıca kalibrasyon modeli döndürür.
    # Kod oldukça uzun olacağından buraya sadece değişiklikleri özetleyip ana fonksiyonu ekleyeceğim.
    # Gerçek uygulamada backtest_1x2_value fonksiyonunun geliştirilmiş versiyonunu yazmalısınız.
    pass

# Backtest butonu ve sonuçları (aynen, ancak kalibrasyon seçeneği eklenecek)
if bt_run:
    # ... backtest çalıştır
    # Eğer calibrate_probs seçiliyse, kalibrasyon modelini hesapla ve session_state'e kaydet
    # Daha sonra ana tahminde bu modeli kullan
    pass

# =========================================================
# KALİBRASYON UYGULAMASI (eğer session_state'de model varsa)
# =========================================================
if 'calibration_models' in st.session_state and calibrate_probs:
    probs_cal = apply_calibration(st.session_state.calibration_models, probs_array)
    pH_cal, pD_cal, pA_cal = probs_cal[0]
    st.write("**Kalibre edilmiş olasılıklar:**")
    st.write(f"H: {pH_cal*100:.1f}% | D: {pD_cal*100:.1f}% | A: {pA_cal*100:.1f}%")

# =========================================================
# GÖSTERİM (aynen devam)
# =========================================================
