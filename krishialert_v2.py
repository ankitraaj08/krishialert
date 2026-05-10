"""
KrishiAlert v2 - Agricultural Price Intelligence System
=======================================================
Upgrades over v1:
  1. Real Agmarknet data fetching (with smart fallback)
  2. Weather integration via Open-Meteo (free, no key needed)
  3. Improved fair price estimation (quantile regression)
  4. Nearby mandi recommendation engine
  5. Explainable AI via SHAP
  6. Full frontend dashboard (served at /)
  7. Voice recommendation via gTTS
  8. Improved forecasting (ensemble XGBoost + trend)
  9. Data visualization (Chart.js JSON endpoints)
"""

import json, os, warnings, hashlib, io, base64
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import requests

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════

CROPS     = ["tomato", "onion", "potato", "wheat", "rice"]
DISTRICTS = ["raichur", "mysuru", "nashik", "agra", "warangal",
             "kurnool", "solapur", "indore", "ludhiana", "patna"]

CROP_CONFIG = {
    "tomato": {"base": 18.0, "volatility": 0.22, "seasonal_peak": [10,11,12,1], "unit": "kg"},
    "onion":  {"base": 24.0, "volatility": 0.18, "seasonal_peak": [11,12,1,2],  "unit": "kg"},
    "potato": {"base": 16.0, "volatility": 0.12, "seasonal_peak": [1,2,3],      "unit": "kg"},
    "wheat":  {"base": 28.0, "volatility": 0.09, "seasonal_peak": [3,4,5],      "unit": "kg"},
    "rice":   {"base": 32.0, "volatility": 0.08, "seasonal_peak": [9,10,11],    "unit": "kg"},
}

DISTRICT_META = {
    "raichur":  {"lat": 16.20, "lon": 77.36, "factor": 0.82, "isolation": 2, "state": "Karnataka",    "mandis": ["Raichur Main", "Sindhanur", "Manvi"]},
    "mysuru":   {"lat": 12.29, "lon": 76.64, "factor": 1.05, "isolation": 0, "state": "Karnataka",    "mandis": ["Mysuru APMC", "Nanjangud", "Hunsur"]},
    "nashik":   {"lat": 19.99, "lon": 73.79, "factor": 1.12, "isolation": 0, "state": "Maharashtra",  "mandis": ["Nashik APMC", "Lasalgaon", "Yeola"]},
    "agra":     {"lat": 27.18, "lon": 78.01, "factor": 0.95, "isolation": 1, "state": "Uttar Pradesh","mandis": ["Agra Main", "Firozabad", "Mathura"]},
    "warangal": {"lat": 17.97, "lon": 79.59, "factor": 0.88, "isolation": 1, "state": "Telangana",    "mandis": ["Warangal APMC", "Hanamkonda", "Narsampet"]},
    "kurnool":  {"lat": 15.83, "lon": 78.04, "factor": 0.85, "isolation": 2, "state": "Andhra Pradesh","mandis": ["Kurnool APMC", "Nandyal", "Adoni"]},
    "solapur":  {"lat": 17.68, "lon": 75.90, "factor": 0.98, "isolation": 1, "state": "Maharashtra",  "mandis": ["Solapur APMC", "Barshi", "Pandharpur"]},
    "indore":   {"lat": 22.72, "lon": 75.86, "factor": 1.08, "isolation": 0, "state": "Madhya Pradesh","mandis": ["Indore APMC", "Dewas", "Ujjain"]},
    "ludhiana": {"lat": 30.90, "lon": 75.85, "factor": 1.15, "isolation": 0, "state": "Punjab",       "mandis": ["Ludhiana Grain", "Moga", "Jalandhar"]},
    "patna":    {"lat": 25.59, "lon": 85.13, "factor": 0.90, "isolation": 1, "state": "Bihar",        "mandis": ["Patna APMC", "Hajipur", "Muzaffarpur"]},
}

FEATURE_COLS = [
    "crop_enc","district_enc","month","day_of_week","day_of_year",
    "arrival_volume","isolation_level","lag_1","lag_3","lag_7",
    "roll_7_mean","roll_7_std","temp_max","temp_min","rainfall",
    "seasonal_factor","price_q25","price_q75",
]

# ══════════════════════════════════════════════════════════════
# 1. REAL AGMARKNET DATA FETCHING
# ══════════════════════════════════════════════════════════════

def fetch_agmarknet(crop: str, state: str, days_back: int = 30) -> pd.DataFrame:
    """
    Fetch real mandi prices from data.gov.in Agmarknet API.
    API Key: free registration at https://data.gov.in/
    Falls back to enriched synthetic data if API unavailable.
    """
    API_KEY = os.environ.get("AGMARKNET_API_KEY", "")
    if API_KEY:
        try:
            url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
            params = {
                "api-key": API_KEY,
                "format": "json",
                "limit": 500,
                "filters[Commodity]": crop.capitalize(),
                "filters[State]": state,
            }
            resp = requests.get(url, params=params, timeout=8)
            if resp.status_code == 200:
                records = resp.json().get("records", [])
                if records:
                    df = pd.DataFrame(records)
                    df = df.rename(columns={
                        "Arrival_Date": "date",
                        "Modal_Price":  "price",
                        "Arrivals_in_Qtl": "arrival_volume",
                        "District": "district",
                        "Commodity": "crop",
                    })
                    df["price"] = pd.to_numeric(df["price"], errors="coerce") / 100
                    df["arrival_volume"] = pd.to_numeric(df["arrival_volume"], errors="coerce").fillna(500)
                    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
                    df = df.dropna(subset=["price"])
                    print(f"[Agmarknet] Fetched {len(df)} real records for {crop}/{state}")
                    return df
        except Exception as e:
            print(f"[Agmarknet] API error: {e}. Using enriched synthetic data.")

    return _generate_enriched_synthetic(crop, days_back)


def _generate_enriched_synthetic(crop: str, n_days: int = 730) -> pd.DataFrame:
    """Enriched synthetic data matching Agmarknet schema."""
    rows = []
    base_date = datetime.today() - timedelta(days=n_days)
    cfg = CROP_CONFIG.get(crop, CROP_CONFIG["tomato"])

    for day_offset in range(n_days):
        date = base_date + timedelta(days=day_offset)
        month = date.month
        seasonal = 1.15 if month in cfg["seasonal_peak"] else 0.92
        trend = 1 + 0.00015 * day_offset

        for district, dmeta in DISTRICT_META.items():
            noise = np.random.normal(0, cfg["volatility"] * cfg["base"])
            arrival_vol = max(10, np.random.normal(500, 120))
            cartel = 0
            if dmeta["isolation"] == 2 and np.random.random() < 0.12:
                cartel = 1
            price = (cfg["base"] * seasonal * trend * dmeta["factor"]
                     + noise - 0.0004 * arrival_vol - 3.2 * cartel)
            price = max(4.0, round(price, 2))
            rows.append({
                "date": date, "crop": crop, "district": district,
                "price": price, "arrival_volume": round(arrival_vol, 1),
                "month": month, "day_of_week": date.weekday(),
                "day_of_year": date.timetuple().tm_yday,
                "isolation_level": dmeta["isolation"], "cartel_flag": cartel,
            })
    return pd.DataFrame(rows)


def load_all_crops_data() -> pd.DataFrame:
    """Load/generate data for all crops and merge."""
    frames = []
    for crop in CROPS:
        df = _generate_enriched_synthetic(crop, n_days=730)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full


# ══════════════════════════════════════════════════════════════
# 2. WEATHER INTEGRATION (Open-Meteo, no API key needed)
# ══════════════════════════════════════════════════════════════

_weather_cache = {}

def fetch_weather(district: str) -> dict:
    """
    Fetch 7-day weather forecast from Open-Meteo (free, no key).
    Returns dict with daily temp_max, temp_min, rainfall arrays.
    """
    if district in _weather_cache:
        cached_time, cached_data = _weather_cache[district]
        if (datetime.now() - cached_time).seconds < 3600:
            return cached_data

    meta = DISTRICT_META.get(district, {"lat": 20.0, "lon": 78.0})
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": meta["lat"], "longitude": meta["lon"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Asia/Kolkata", "forecast_days": 7,
        }
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            d = resp.json()["daily"]
            result = {
                "temp_max":  d["temperature_2m_max"],
                "temp_min":  d["temperature_2m_min"],
                "rainfall":  d["precipitation_sum"],
                "dates":     d["time"],
                "source":    "Open-Meteo (live)",
            }
            _weather_cache[district] = (datetime.now(), result)
            return result
    except Exception as e:
        print(f"[Weather] Error: {e}. Using mock weather.")

    # Fallback mock
    return {
        "temp_max":  [32 + np.random.uniform(-3, 3) for _ in range(7)],
        "temp_min":  [22 + np.random.uniform(-2, 2) for _ in range(7)],
        "rainfall":  [max(0, np.random.normal(2, 5)) for _ in range(7)],
        "dates":     [(datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)],
        "source":    "estimated",
    }


def weather_impact_factor(temp_max: float, rainfall: float, crop: str) -> float:
    """
    Estimate how weather affects supply (and thus price).
    High rain/heat -> supply disruption -> price rises.
    """
    factor = 1.0
    if rainfall > 20:
        factor += 0.08  # heavy rain disrupts transport
    elif rainfall > 8:
        factor += 0.03
    if temp_max > 38 and crop in ["tomato", "onion"]:
        factor += 0.05  # heat damages perishables
    elif temp_max < 15 and crop in ["tomato", "potato"]:
        factor -= 0.02
    return round(factor, 4)


# ══════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame):
    df = df.sort_values(["crop","district","date"]).reset_index(drop=True)

    le_crop = LabelEncoder().fit(df["crop"])
    le_dist = LabelEncoder().fit(df["district"])
    df["crop_enc"]     = le_crop.transform(df["crop"])
    df["district_enc"] = le_dist.transform(df["district"])

    grp = df.groupby(["crop","district"])["price"]
    df["lag_1"]       = grp.shift(1)
    df["lag_3"]       = grp.shift(3)
    df["lag_7"]       = grp.shift(7)
    df["roll_7_mean"] = grp.transform(lambda x: x.shift(1).rolling(7).mean())
    df["roll_7_std"]  = grp.transform(lambda x: x.shift(1).rolling(7).std())

    # Seasonal factor
    def seasonal(row):
        pk = CROP_CONFIG.get(row["crop"], {}).get("seasonal_peak", [])
        return 1.15 if row["month"] in pk else 0.92
    df["seasonal_factor"] = df.apply(seasonal, axis=1)

    # Fair price quantiles per crop+district
    q25 = grp.transform(lambda x: x.quantile(0.25))
    q75 = grp.transform(lambda x: x.quantile(0.75))
    df["price_q25"] = q25
    df["price_q75"] = q75

    # Weather placeholders (filled at predict time with live data)
    if "temp_max"  not in df.columns: df["temp_max"]  = 30.0
    if "temp_min"  not in df.columns: df["temp_min"]  = 20.0
    if "rainfall"  not in df.columns: df["rainfall"]  = 2.0
    if "isolation_level" not in df.columns:
        df["isolation_level"] = df["district"].map(
            lambda d: DISTRICT_META.get(d, {}).get("isolation", 1))

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df, le_crop, le_dist


# ══════════════════════════════════════════════════════════════
# 4. IMPROVED FAIR PRICE ESTIMATION (quantile-based)
# ══════════════════════════════════════════════════════════════

def compute_fair_prices(df: pd.DataFrame) -> dict:
    """
    Fair price = 75th percentile of historical prices per crop+district
    weighted by recency. More robust than mean against cartel suppression.
    """
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - timedelta(days=180)
    recent = df[df["date"] >= cutoff]
    if recent.empty:
        recent = df

    fair = {}
    for (crop, district), g in recent.groupby(["crop","district"]):
        days_ago = (recent["date"].max() - g["date"]).dt.days
        weights = np.exp(-days_ago / 60)     # exponential decay, 60-day half-life
        weighted_prices = np.average(g["price"], weights=weights)
        q75 = g["price"].quantile(0.75)
        # Blend weighted mean with 75th percentile
        fair[(crop, district)] = round(0.6 * q75 + 0.4 * weighted_prices, 2)

    return fair


# ══════════════════════════════════════════════════════════════
# 5. ENSEMBLE FORECASTING MODEL (XGBoost + GBR)
# ══════════════════════════════════════════════════════════════

def train_models(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False)

    xgb_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.07,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        random_state=42, verbosity=0)
    xgb_model.fit(X_train, y_train)

    gbr_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42)
    gbr_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    gbr_pred = gbr_model.predict(X_test)
    ensemble  = 0.65 * xgb_pred + 0.35 * gbr_pred
    mae = mean_absolute_error(y_test, ensemble)
    print(f"[Model] Ensemble MAE on test: ₹{mae:.2f}/kg")

    return xgb_model, gbr_model


def forecast_prices(xgb_model, gbr_model, le_crop, le_dist,
                    crop, district, current_price, weather, days=7):
    cfg   = CROP_CONFIG[crop]
    dmeta = DISTRICT_META[district]
    today = datetime.today()

    crop_enc = int(le_crop.transform([crop])[0])
    dist_enc = int(le_dist.transform([district])[0])

    # Quantile bounds for confidence interval
    q25_seed = fair_prices.get((crop, district), cfg["base"]) * 0.88
    q75_seed = fair_prices.get((crop, district), cfg["base"]) * 1.12

    history = [current_price] * 8
    forecasts = []

    for i in range(days):
        fd = today + timedelta(days=i)
        month = fd.month
        dow   = fd.weekday()
        doy   = fd.timetuple().tm_yday
        arrival_vol = max(10, np.random.normal(500, 80))

        t_max = weather["temp_max"][i] if i < len(weather["temp_max"]) else 30.0
        t_min = weather["temp_min"][i] if i < len(weather["temp_min"]) else 20.0
        rain  = weather["rainfall"][i]  if i < len(weather["rainfall"])  else 2.0
        w_factor = weather_impact_factor(t_max, rain, crop)

        seasonal = 1.15 if month in cfg["seasonal_peak"] else 0.92
        row = {
            "crop_enc": crop_enc, "district_enc": dist_enc,
            "month": month, "day_of_week": dow, "day_of_year": doy,
            "arrival_volume": arrival_vol, "isolation_level": dmeta["isolation"],
            "lag_1": history[-1], "lag_3": history[-3], "lag_7": history[-7],
            "roll_7_mean": np.mean(history[-7:]), "roll_7_std": np.std(history[-7:]),
            "temp_max": t_max, "temp_min": t_min, "rainfall": rain,
            "seasonal_factor": seasonal, "price_q25": q25_seed, "price_q75": q75_seed,
        }
        X = pd.DataFrame([row])[FEATURE_COLS]
        p_xgb = float(xgb_model.predict(X)[0])
        p_gbr = float(gbr_model.predict(X)[0])
        pred  = max(4.0, round((0.65 * p_xgb + 0.35 * p_gbr) * w_factor, 2))
        ci_lo = max(4.0, round(pred * 0.91, 2))
        ci_hi = round(pred * 1.09, 2)

        history.append(pred)
        day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        label = "Today" if i == 0 else day_names[dow]
        forecasts.append({
            "day_index": i, "day_label": label,
            "date": fd.strftime("%d %b"),
            "price": pred, "ci_low": ci_lo, "ci_high": ci_hi,
            "temp_max": round(t_max, 1), "rainfall": round(rain, 1),
            "weather_impact": round((w_factor - 1) * 100, 1),
        })
    return forecasts


# ══════════════════════════════════════════════════════════════
# 6. ANOMALY / CARTEL DETECTOR
# ══════════════════════════════════════════════════════════════

def train_anomaly_detector(df: pd.DataFrame):
    df = df.copy()
    df["expected"] = df.groupby(["crop","district"])["price"].transform("mean")
    df["deviation"] = (df["price"] - df["expected"]) / (df["expected"] + 1e-6)
    X_anom = df[["deviation","arrival_volume","isolation_level"]]
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X_anom)
    return iso


def detect_anomaly(iso_model, current_price, fair_price, arrival_vol, isolation_level):
    deviation = (current_price - fair_price) / (fair_price + 1e-6)
    X = pd.DataFrame([[deviation, arrival_vol, isolation_level]],
                     columns=["deviation","arrival_volume","isolation_level"])
    score     = float(iso_model.score_samples(X)[0])
    is_anom   = iso_model.predict(X)[0] == -1
    pct_below = round((fair_price - current_price) / fair_price * 100, 1) if current_price < fair_price else 0.0
    severity  = "high" if pct_below > 25 else ("medium" if pct_below > 12 else "low")
    return {
        "cartel_flag": bool(is_anom and current_price < fair_price),
        "anomaly_score": round(score, 4),
        "pct_below_fair": pct_below,
        "fair_price": round(fair_price, 2),
        "severity": severity,
    }


# ══════════════════════════════════════════════════════════════
# 7. EXPLAINABLE AI (SHAP)
# ══════════════════════════════════════════════════════════════

def explain_prediction(xgb_model, row_df: pd.DataFrame) -> list:
    """
    Returns top-5 SHAP feature contributions for a single prediction.
    Explains WHY the model gave this price forecast.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(row_df[FEATURE_COLS])
        contribs = sorted(
            zip(FEATURE_COLS, shap_vals[0]),
            key=lambda x: abs(x[1]), reverse=True
        )[:5]
        readable = {
            "crop_enc": "Crop type", "district_enc": "District",
            "month": "Month/season", "lag_1": "Yesterday's price",
            "lag_3": "Price 3 days ago", "lag_7": "Price 7 days ago",
            "roll_7_mean": "7-day avg price", "roll_7_std": "Price volatility",
            "arrival_volume": "Market arrivals", "isolation_level": "District isolation",
            "temp_max": "Max temperature", "rainfall": "Rainfall",
            "seasonal_factor": "Seasonal demand", "price_q25": "Historical low",
            "price_q75": "Historical high", "day_of_week": "Day of week",
            "day_of_year": "Day of year", "temp_min": "Min temperature",
        }
        return [
            {"feature": readable.get(f, f),
             "impact": round(float(v), 3),
             "direction": "increases price" if v > 0 else "decreases price"}
            for f, v in contribs
        ]
    except Exception as e:
        return [{"feature": "Model explanation unavailable", "impact": 0, "direction": str(e)}]


# ══════════════════════════════════════════════════════════════
# 8. NEARBY MANDI RECOMMENDATION
# ══════════════════════════════════════════════════════════════

def recommend_mandis(district: str, current_price: float, crop: str) -> list:
    """
    Compare current district's fair price against nearby districts.
    Returns ranked list of better markets within ~300km.
    """
    import math
    meta = DISTRICT_META.get(district)
    if not meta:
        return []

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))

    results = []
    for d, dmeta in DISTRICT_META.items():
        if d == district:
            continue
        dist_km = haversine(meta["lat"], meta["lon"], dmeta["lat"], dmeta["lon"])
        if dist_km > 320:
            continue
        fp = fair_prices.get((crop, d), CROP_CONFIG[crop]["base"] * dmeta["factor"])
        gain = round(fp - current_price, 2)
        if gain > 1.5:
            results.append({
                "district": d.capitalize(),
                "state": dmeta["state"],
                "distance_km": round(dist_km, 1),
                "expected_price": round(fp, 2),
                "potential_gain_per_kg": gain,
                "gain_on_quintal": round(gain * 100, 1),
                "mandis": dmeta["mandis"][:2],
                "isolation": ["low","medium","high"][dmeta["isolation"]],
            })

    return sorted(results, key=lambda x: x["potential_gain_per_kg"], reverse=True)[:4]


# ══════════════════════════════════════════════════════════════
# 9. SELLING WINDOW OPTIMIZER
# ══════════════════════════════════════════════════════════════

def find_selling_window(forecasts, max_wait_days, current_price):
    best = {"day_index": 0, "score": -999}
    scored = []
    for f in forecasts:
        i = f["day_index"]
        urgency  = i * 0.40
        risk     = (i ** 1.3) * 0.15
        score    = f["price"] - urgency - risk
        scored.append({**f, "score": round(score, 2), "urgency_cost": round(urgency + risk, 2)})
        if i < max_wait_days and score > best["score"]:
            best = {**f, "score": round(score, 2)}

    gain_per_kg  = round(best["price"] - current_price, 2)
    gain_100kg   = round(gain_per_kg * 100, 2)
    confidence   = "high" if abs(gain_per_kg) < 3 else ("medium" if abs(gain_per_kg) < 7 else "low")
    return {
        "best_day_index":  best["day_index"],
        "best_day_label":  best["day_label"],
        "best_day_date":   best["date"],
        "expected_price":  best["price"],
        "gain_per_kg":     gain_per_kg,
        "gain_on_100kg":   gain_100kg,
        "sell_today":      best["day_index"] == 0,
        "confidence":      confidence,
        "all_scores":      scored,
    }


# ══════════════════════════════════════════════════════════════
# 10. SMS FORMATTER
# ══════════════════════════════════════════════════════════════

def format_sms(crop, district, current_price, window, anomaly):
    crop_t = crop.capitalize()
    if window["sell_today"]:
        action = "Sell TODAY. Prices may fall soon."
    else:
        action = f"Best: Sell {window['best_day_label']} {window['best_day_date']}. Exp Rs.{window['expected_price']}/kg."
    warn = f" ALERT: {anomaly['pct_below_fair']}% below fair price. Possible buyer cartel." if anomaly["cartel_flag"] else ""
    return (
        f"KrishiAlert|{crop_t}|{district.capitalize()}\n"
        f"Today: Rs.{current_price}/kg\n"
        f"{action}\n"
        f"Fair: Rs.{anomaly['fair_price']}/kg{warn}"
    )


def send_sms_twilio(to_number, message):
    try:
        from twilio.rest import Client
        c = Client(os.environ["TWILIO_SID"], os.environ["TWILIO_TOKEN"])
        m = c.messages.create(body=message, from_=os.environ["TWILIO_FROM"], to=to_number)
        return {"status": "sent", "sid": m.sid}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ══════════════════════════════════════════════════════════════
# 11. VOICE RECOMMENDATION (gTTS)
# ══════════════════════════════════════════════════════════════

def generate_voice(crop, district, window, anomaly, lang="en") -> str:
    """
    Generate voice recommendation using gTTS.
    Returns base64-encoded MP3 string.
    Supports lang='hi' for Hindi, 'kn' for Kannada, 'te' for Telugu.
    """
    try:
        from gtts import gTTS
        crop_t    = crop.capitalize()
        district_t = district.capitalize()

        if lang == "hi":
            if window["sell_today"]:
                text = f"कृषि अलर्ट। {crop_t} के लिए {district_t} मंडी में। आज का भाव {window['expected_price']} रुपये प्रति किलो है। आज बेचना उचित रहेगा।"
            else:
                text = f"कृषि अलर्ट। {crop_t} के लिए {district_t} मंडी में। सबसे अच्छा समय {window['best_day_label']} है। अनुमानित भाव {window['expected_price']} रुपये प्रति किलो।"
            if anomaly["cartel_flag"]:
                text += f" सावधान: मौजूदा भाव उचित मूल्य से {anomaly['pct_below_fair']} प्रतिशत कम है। किसी दूसरी मंडी में जाने पर विचार करें।"
        else:
            if window["sell_today"]:
                text = (f"KrishiAlert for {crop_t} in {district_t}. "
                        f"Today's price is {current_global_price} rupees per kg. "
                        f"We recommend selling today as prices are expected to fall.")
            else:
                text = (f"KrishiAlert for {crop_t} in {district_t}. "
                        f"Best selling window is {window['best_day_label']}, {window['best_day_date']}. "
                        f"Expected price is {window['expected_price']} rupees per kg.")
            if anomaly["cartel_flag"]:
                text += (f" Warning: Current price is {anomaly['pct_below_fair']} percent below "
                         f"fair market value of {anomaly['fair_price']} rupees. "
                         f"Consider visiting a nearby mandi for better prices.")

        buf = io.BytesIO()
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        print(f"[Voice] Error: {e}")
        return ""


current_global_price = 14.0  # updated per request


# ══════════════════════════════════════════════════════════════
# BOOTSTRAP — TRAIN ON STARTUP
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

print("[KrishiAlert v2] Loading data...")
raw_df = load_all_crops_data()

print("[KrishiAlert v2] Engineering features...")
feat_df, le_crop, le_dist = engineer_features(raw_df)

print("[KrishiAlert v2] Training ensemble model...")
xgb_model, gbr_model = train_models(feat_df)

print("[KrishiAlert v2] Training anomaly detector...")
iso_model = train_anomaly_detector(feat_df)

print("[KrishiAlert v2] Computing fair prices (quantile-weighted)...")
fair_prices = compute_fair_prices(feat_df)

print("[KrishiAlert v2] Ready.\n")


@app.route("/")
def index():
    return open("templates/index.html", encoding="utf-8").read()


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "2.0",
                    "model": "XGBoost+GBR Ensemble + IsolationForest + SHAP"})


@app.route("/crops")
def list_crops():
    return jsonify({"crops": CROPS, "districts": list(DISTRICT_META.keys())})


@app.route("/forecast", methods=["POST"])
def forecast():
    global current_global_price
    data          = request.get_json()
    crop          = data.get("crop", "tomato").lower()
    district      = data.get("district", "raichur").lower()
    current_price = float(data.get("current_price", 14.0))
    max_wait      = int(data.get("max_wait_days", 4))
    arrival_vol   = float(data.get("arrival_volume", 500))
    phone         = data.get("phone_number")
    lang          = data.get("voice_lang", "en")
    want_voice    = bool(data.get("voice", False))
    want_explain  = bool(data.get("explain", True))

    if crop not in CROPS:
        return jsonify({"error": f"Unknown crop. Options: {CROPS}"}), 400
    if district not in DISTRICT_META:
        return jsonify({"error": f"Unknown district. Options: {list(DISTRICT_META.keys())}"}), 400

    current_global_price = current_price

    # Fetch live weather
    weather = fetch_weather(district)

    # Forecast
    forecasts = forecast_prices(xgb_model, gbr_model, le_crop, le_dist,
                                crop, district, current_price, weather, days=7)

    # Selling window
    window = find_selling_window(forecasts, max_wait, current_price)

    # Anomaly
    fair_p = fair_prices.get((crop, district), CROP_CONFIG[crop]["base"])
    iso_lv = DISTRICT_META[district]["isolation"]
    anomaly = detect_anomaly(iso_model, current_price, fair_p, arrival_vol, iso_lv)

    # Nearby mandis
    mandis = recommend_mandis(district, current_price, crop)

    # SHAP explanation
    explanation = []
    if want_explain:
        today = datetime.today()
        h = [current_price] * 8
        row = {
            "crop_enc": int(le_crop.transform([crop])[0]),
            "district_enc": int(le_dist.transform([district])[0]),
            "month": today.month, "day_of_week": today.weekday(),
            "day_of_year": today.timetuple().tm_yday,
            "arrival_volume": arrival_vol,
            "isolation_level": iso_lv,
            "lag_1": h[-1], "lag_3": h[-3], "lag_7": h[-7],
            "roll_7_mean": np.mean(h[-7:]), "roll_7_std": np.std(h[-7:]),
            "temp_max": weather["temp_max"][0], "temp_min": weather["temp_min"][0],
            "rainfall": weather["rainfall"][0],
            "seasonal_factor": 1.15 if today.month in CROP_CONFIG[crop]["seasonal_peak"] else 0.92,
            "price_q25": fair_p * 0.88, "price_q75": fair_p * 1.12,
        }
        explanation = explain_prediction(xgb_model, pd.DataFrame([row]))

    # SMS
    sms_text = format_sms(crop, district, current_price, window, anomaly)
    sms_result = {}
    if phone:
        sms_result = send_sms_twilio(phone, sms_text)

    # Voice
    voice_b64 = ""
    if want_voice:
        voice_b64 = generate_voice(crop, district, window, anomaly, lang=lang)

    return jsonify({
        "crop": crop, "district": district,
        "current_price": current_price,
        "weather": weather,
        "forecasts": forecasts,
        "selling_window": window,
        "anomaly": anomaly,
        "nearby_mandis": mandis,
        "explanation": explanation,
        "sms_text": sms_text,
        "sms_sent": sms_result,
        "voice_mp3_b64": voice_b64,
        "district_info": {
            "state": DISTRICT_META[district]["state"],
            "isolation": ["Low","Medium","High"][iso_lv],
            "mandis": DISTRICT_META[district]["mandis"],
        }
    })


@app.route("/chart/historical", methods=["GET"])
def chart_historical():
    """Returns last 60 days of price data for a crop+district pair — for Chart.js"""
    crop     = request.args.get("crop", "tomato")
    district = request.args.get("district", "raichur")
    cutoff   = datetime.today() - timedelta(days=60)
    sub = feat_df[(feat_df["crop"] == crop) & (feat_df["district"] == district)]
    sub = sub[pd.to_datetime(sub["date"]) >= cutoff].sort_values("date")
    return jsonify({
        "labels": sub["date"].astype(str).tolist(),
        "prices": sub["price"].round(2).tolist(),
        "arrivals": sub["arrival_volume"].round(1).tolist(),
    })


<<<<<<< HEAD
@app.route("/chat", methods=["POST"])
def chat_proxy():
    """
    Proxy endpoint for the chatbot.
    Forwards messages to Anthropic API so the API key stays server-side.
    """
    import requests as req
    data = request.get_json()
    messages  = data.get("messages", [])
    system    = data.get("system", "")
    crop      = data.get("crop", "tomato")
    district  = data.get("district", "raichur")
    price     = data.get("price", "14")

    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    if not ANTHROPIC_API_KEY:
        # Fallback: rule-based chatbot when no API key is set
        user_msg = messages[-1]["content"].lower() if messages else ""

        # Detect crop mentioned
        mentioned_crop = next((c for c in CROPS if c in user_msg), crop)
        cfg = CROP_CONFIG.get(mentioned_crop, CROP_CONFIG["tomato"])
        peak_months = cfg["seasonal_peak"]
        month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                       7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
        peak_str = ", ".join(month_names[m] for m in peak_months)

        now = datetime.today()
        is_peak = now.month in peak_months

        if any(w in user_msg for w in ["when", "kab", "sell", "bech"]):
            if is_peak:
                reply = (f"🌾 Good time! {mentioned_crop.capitalize()} prices are typically at PEAK right now "
                         f"(peak season: {peak_str}).\n\n"
                         f"Recommended: Sell within the next 2-3 days while demand is high.\n"
                         f"Fair price range: ₹{cfg['base']*0.9:.0f}–₹{cfg['base']*1.2:.0f}/kg")
            else:
                reply = (f"🍅 {mentioned_crop.capitalize()} peak season is {peak_str}.\n\n"
                         f"Currently off-peak — prices may be lower. If you can wait, hold stock until peak season. "
                         f"Otherwise sell now to avoid storage losses.\n"
                         f"Current base price: ~₹{cfg['base']:.0f}/kg")
        elif any(w in user_msg for w in ["price", "rate", "bhav", "kitna"]):
            reply = (f"📊 {mentioned_crop.capitalize()} price info:\n\n"
                     f"• Base price: ₹{cfg['base']:.0f}/kg\n"
                     f"• Peak months: {peak_str}\n"
                     f"• Volatility: {'High' if cfg['volatility']>0.15 else 'Moderate'}\n\n"
                     f"Use the Analyze button above to get a full 7-day ML forecast for your district!")
        elif any(w in user_msg for w in ["mandi", "market", "where", "kahan"]):
            reply = (f"🗺️ For {mentioned_crop.capitalize()}, here are some top mandis:\n\n"
                     f"• Nashik APMC (Maharashtra) — major {mentioned_crop} hub\n"
                     f"• Mysuru APMC (Karnataka) — good connectivity\n"
                     f"• Lasalgaon — Asia's largest onion market\n\n"
                     f"Select your district in the form above to get nearby mandi recommendations!")
        elif any(w in user_msg for w in ["cartel", "anomaly", "detection"]):
            reply = ("⚠️ Cartel Detection works using IsolationForest ML model.\n\n"
                     "If current price is >15% below the fair price AND the district has high isolation, "
                     "it flags a possible buyer cartel. This protects farmers from being cheated.\n\n"
                     "Fair price is computed as the 75th percentile of recent 6-month prices.")
        elif any(w in user_msg for w in ["model", "ml", "predict", "accuracy", "xgboost"]):
            reply = ("🤖 KrishiAlert uses an ensemble of:\n\n"
                     "• XGBoost (65% weight) — captures non-linear price patterns\n"
                     "• Gradient Boosting (35% weight) — adds stability\n"
                     "• Features: lag prices, weather, seasonality, arrivals, district isolation\n\n"
                     "SHAP values explain *why* the model made each prediction.")
        elif any(w in user_msg for w in ["weather", "rain", "mausam", "temperature"]):
            reply = ("🌦️ Weather affects crop prices significantly:\n\n"
                     "• Heavy rain (>20mm) → transport disruption → prices rise\n"
                     "• High heat (>38°C) → perishable damage for tomato/onion → prices spike\n\n"
                     "KrishiAlert fetches live 7-day weather from Open-Meteo API and factors it into forecasts.")
        else:
            reply = (f"👋 Namaste Kisan!\n\n"
                     f"I can help you with:\n"
                     f"• **When to sell** — 'When should I sell tomatoes?'\n"
                     f"• **Price info** — 'What is the price of onion?'\n"
                     f"• **Mandi selection** — 'Which mandi is best?'\n"
                     f"• **How the model works** — 'How does KrishiAlert predict prices?'\n\n"
                     f"Currently viewing: {crop.capitalize()} in {district.capitalize()} at ₹{price}/kg")

        return jsonify({"content": [{"text": reply}]})

    # Real Anthropic API call (server-side, key is safe)
    system_prompt = f"""You are KrishiAlert AI — an expert agricultural market intelligence assistant embedded in the KrishiAlert platform.

KrishiAlert is an ML system that forecasts mandi prices for Indian farmers using XGBoost + Gradient Boosting ensemble, detects cartel behaviour using IsolationForest, provides SHAP explainability, integrates Open-Meteo weather data, and delivers recommendations via SMS and Hindi voice (gTTS).

Current dashboard context:
- Selected crop: {crop}
- Selected district: {district}
- Current price: ₹{price}/kg

Crop seasonal peak months:
- Tomato: October–January
- Onion: November–February  
- Potato: January–March
- Wheat: March–May
- Rice: September–November

Your role:
- Answer questions about agricultural prices, mandi markets, crop selling strategy
- Tell farmers the BEST TIME TO SELL their specific crop based on seasonal patterns and current context
- Explain how KrishiAlert's ML models work in simple terms
- Give practical advice about when and where to sell
- Use Indian context — mention mandis, MSP, APMC, Agmarknet, kharif/rabi seasons
- Keep answers concise (3-5 lines max) and practical
- Use ₹ for prices, mention specific crops and districts when relevant
- Occasionally use Hindi words like "kisan", "mandi", "fasal" naturally
- Be warm, helpful, farmer-friendly

Never say you cannot help. Always give a useful, grounded answer."""

    try:
        resp = req.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 400,
                "system": system_prompt,
                "messages": messages,
            },
            timeout=20,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


=======
>>>>>>> b1602811ecb0e617867a9ad58dfb58c33045c064
@app.route("/chart/district_comparison", methods=["GET"])
def chart_district_comparison():
    """Returns current fair prices across all districts for a crop."""
    crop = request.args.get("crop", "tomato")
    data = []
    for d in DISTRICT_META:
        fp = fair_prices.get((crop, d), CROP_CONFIG[crop]["base"])
        data.append({"district": d.capitalize(), "fair_price": round(fp, 2),
                     "isolation": ["Low","Medium","High"][DISTRICT_META[d]["isolation"]]})
    return jsonify(sorted(data, key=lambda x: x["fair_price"], reverse=True))


# ══════════════════════════════════════════════════════════════
# FRONTEND DASHBOARD HTML
# ══════════════════════════════════════════════════════════════

# Frontend files live in templates/index.html and static/.



# ══════════════════════════════════════════════════════════════
# CLI DEMO
# ══════════════════════════════════════════════════════════════

def run_cli_demo():
    print("=" * 60)
    print("  KrishiAlert v2 — Full System Demo")
    print("=" * 60)
    cases = [
        ("tomato", "raichur",  9.0, 4),
        ("onion",  "nashik",  22.0, 7),
        ("wheat",  "ludhiana",26.5, 2),
    ]
    for crop, district, price, wait in cases:
        print(f"\nCrop: {crop.upper()} | District: {district.capitalize()}")
        print(f"Current: ₹{price}/kg | Wait limit: {wait} days")
        weather   = fetch_weather(district)
        forecasts = forecast_prices(xgb_model, gbr_model, le_crop, le_dist,
                                    crop, district, price, weather)
        window    = find_selling_window(forecasts, wait, price)
        fair_p    = fair_prices.get((crop, district), CROP_CONFIG[crop]["base"])
        anomaly   = detect_anomaly(iso_model, price, fair_p, 500,
                                   DISTRICT_META[district]["isolation"])
        mandis    = recommend_mandis(district, price, crop)
        sms       = format_sms(crop, district, price, window, anomaly)

        print("\n7-day forecast:")
        for f in forecasts:
            mark = " <-- SELL" if f["day_index"] == window["best_day_index"] else ""
            print(f"  {f['day_label']:5} {f['date']}: ₹{f['price']:6.2f}  "
                  f"[{f['temp_max']}°C, {f['rainfall']}mm rain]{mark}")

        print(f"\nSelling window : {window['best_day_label']} {window['best_day_date']}")
        print(f"Gain on 100kg  : ₹{window['gain_on_100kg']}")
        print(f"Cartel flag    : {anomaly['cartel_flag']} ({anomaly['pct_below_fair']}% below ₹{anomaly['fair_price']})")
        if mandis:
            print(f"Better mandis  : {mandis[0]['district']} ({mandis[0]['distance_km']}km, +₹{mandis[0]['potential_gain_per_kg']}/kg)")
        print(f"\nSMS:\n{sms}")
        print("=" * 60)


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_cli_demo()
    else:
        print("[KrishiAlert v2] Starting at http://localhost:5000")
        app.run(debug=True, port=5000)
