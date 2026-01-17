import os
# Force le mode CPU pour éviter l'erreur "CUDA invalid argument"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from datetime import timedelta
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from vmdpy import VMD

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(layout="wide", page_title="Desk Pro BTC TFT Local", page_icon="₿")

# --- 2. CHARGEMENT DU MODÈLE ---
TFT_CHECKPOINT_PATH = "TFT_model.ckpt"
SEQ_LEN = 24

@st.cache_resource
def load_tft_model():
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(TFT_CHECKPOINT_PATH, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None

tft_model = load_tft_model()

# --- 3. LOGIQUE TECHNIQUE ---
def apply_vmd_causal(df_subset):
    signal = df_subset["close"].values.astype(np.float32)
    try:
        imfs, _, _ = VMD(signal, 2000, 0, 3, 0, 1, 1e-7)
        for i in range(3):
            val = imfs[i]
            if len(val) < len(df_subset): 
                val = np.pad(val, (len(df_subset) - len(val), 0), 'edge')
            else: 
                val = val[-len(df_subset):]
            df_subset[f"IMF{i + 1}"] = val
    except Exception:
        df_subset["IMF1"], df_subset["IMF2"], df_subset["IMF3"] = 0.0, 0.0, 0.0
    return df_subset

def run_prediction_local(input_data):
    try:
        df = pd.DataFrame(input_data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_return"].rolling(SEQ_LEN).std()
        df["momentum"] = df["close"].pct_change(SEQ_LEN)
        delta = df["close"].diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))
        df = apply_vmd_causal(df).dropna()
        df.reset_index(drop=True, inplace=True)
        df["time_idx"] = np.arange(len(df))
        df["group_id"], df["month"], df["day_of_week"] = "BTC", df.datetime.dt.month.astype(str), df.datetime.dt.dayofweek.astype(str)
        
        ds = TimeSeriesDataSet.from_parameters(tft_model.dataset_parameters, df, predict=True, stop_randomization=True)
        with torch.no_grad():
            raw_preds = tft_model.predict(ds.to_dataloader(train=False, batch_size=1), mode="raw")
            preds = raw_preds["prediction"][0].cpu().numpy()
            
        return {
            "median": preds[:, 3].tolist(), "low": preds[:, 1].tolist(), "high": preds[:, 5].tolist(),
            "last_close": float(df.iloc[-1]["close"]), "last_date": str(df.iloc[-1]["datetime"])
        }
    except Exception as e:
        return None

# --- 4. DESIGN CSS ---
st.markdown("""
<style>
    .main-title { text-align: center; font-size: 2.2rem; font-weight: 800; color: white; margin-bottom: 20px; }
    .header-container { position: relative; display: flex; flex-direction: column; align-items: center; margin-top: 40px; margin-bottom: 60px; width: 100%; }
    .top-line { display: flex; justify-content: flex-start; width: 100%; border-bottom: 2px solid #333; padding-bottom: 40px; margin-bottom: 50px; }
    .last-price-box { display: flex; align-items: center; gap: 25px; }
    .last-price-label { color: #00CCFF; font-size: 2.6rem; font-weight: 800; margin: 0; }
    .last-price-value { color: white; font-size: 2.6rem; font-weight: 800; margin: 0; }
    .section-subtitle { text-align: center; font-size: 3.5rem; font-weight: 900; color: #DA70D6; text-transform: uppercase; letter-spacing: 5px; margin: 0; }
    .footer-signature { text-align: center; margin-top: 80px; padding: 40px; font-size: 1.2rem; color: white; border-top: 1px solid #333; background-color: rgba(0,0,0,0.2); }
    .footer-signature a { color: white; text-decoration: none; font-weight: bold; margin: 0 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Desk pro for multi-horizons Bitcoin price forecasting and trading using a Temporal Fusion Transformer (TFT) model</div>', unsafe_allow_html=True)

# --- 5. CHARGEMENT DONNÉES ---
st.sidebar.header("📁 Données")
file = st.sidebar.file_uploader("Fichier CSV", type="csv")

if file is not None:
    df_raw = pd.read_csv(file)
else:
    try:
        df_raw = pd.read_csv("Val_dec_2025_Binance.csv")
    except:
        df_raw = None

# --- 6. DASHBOARD ---
if df_raw is not None and tft_model:
    c_t = next((x for x in ["timestamp", "date", "datetime"] if x in df_raw.columns.str.lower()), None)
    df_raw["datetime"] = pd.to_datetime(df_raw[c_t], unit="s" if pd.api.types.is_numeric_dtype(df_raw[c_t]) else None)
    df_h = df_raw.sort_values("datetime").set_index("datetime").resample("1H").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

    tr = st.sidebar.slider("Période", df_h.index[0].to_pydatetime(), df_h.index[-1].to_pydatetime(), (df_h.index[0].to_pydatetime(), df_h.index[-1].to_pydatetime()), format="DD/MM HH:mm")
    hor_sel = st.sidebar.multiselect("Horizons", [1,2,4,6,8,12,24], default=[1,2,4,6,8,12,24])
    hor_sel.sort()

    df_calc = df_h[df_h.index <= tr[1]].tail(350).reset_index()
    res = run_prediction_local(df_calc.to_dict(orient="records"))

    if res:
        # --- GRAPH (Rétablissement interactif complet) ---
        fut_dates = [pd.to_datetime(res["last_date"]) + timedelta(hours=h) for h in range(1, 25)]
        graph_labels = [f"H+{h}" if h in hor_sel else "" for h in range(1, 25)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_h.index, y=df_h["close"], name="Prix Réel", line=dict(color="rgba(255,255,255,0.4)", width=1), hovertemplate="Prix Réel: $%{y:,.2f}<extra></extra>"))
        df_v = df_h.loc[(df_h.index >= tr[0]) & (df_h.index <= tr[1])]
        fig.add_trace(go.Scatter(x=df_v.index, y=df_v["close"], name="Observation", line=dict(color="#00CCFF", width=2.5), hovertemplate="Observation: $%{y:,.2f}<extra></extra>"))
        
        fig.add_trace(go.Scatter(
            x=fut_dates, y=res["median"], mode='lines+markers+text', name="TFT", 
            line=dict(color="#DA70D6", width=4),
            marker=dict(size=[12 if h in hor_sel else 0 for h in range(1, 25)], symbol="diamond", color="#DA70D6", line=dict(width=1, color="white")),
            text=graph_labels, textposition="top center", 
            textfont=dict(color="#DA70D6", size=12, family="Arial Black"),
            hovertemplate="Prédiction TFT: $%{y:,.2f}<extra></extra>"
        ))
        
        # Rétablissement du Zoom, Pan et Scroll
        fig.update_layout(template="plotly_dark", height=650, dragmode="pan", margin=dict(l=0,r=0,t=20,b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})

        # --- HEADER ---
        dt_str = pd.to_datetime(res["last_date"]).strftime("%d/%m/%Y %H:%M")
        st.markdown(f"""
            <div class="header-container">
                <div class="top-line">
                    <div class="last-price-box">
                        <span class="last-price-label">Last known BTC price as of {dt_str} :</span>
                        <span class="last-price-value">${res['last_close']:,.2f}</span>
                    </div>
                </div>
                <h1 class="section-subtitle">MODEL PREDICTIONS</h1>
            </div>
        """, unsafe_allow_html=True)

        # --- CARTES (Rétablissement des dates sous horizons) ---
        if hor_sel:
            cols = st.columns(len(hor_sel))
            for i, h in enumerate(hor_sel):
                idx = h-1
                p_h, p_l, p_hi = res["median"][idx], res["low"][idx], res["high"][idx]
                perf = (p_h - res["last_close"]) / res["last_close"]
                conf = max(0, min(100, 100 * (1 - (p_hi - p_l) / p_h * 5)))
                # Calcul de la date exacte pour H+h
                date_card = (pd.to_datetime(res["last_date"]) + timedelta(hours=h)).strftime("%d/%m %H:00")
                
                color_trend = "#00FF88" if perf > 0.0015 else "#FF4B4B" if perf < -0.0015 else "#FFA500"
                decision = "BUY" if (conf > 40 and perf > 0.0015) else "SELL" if (conf > 40 and perf < -0.0015) else "HOLD"
                bg_btn = "#238636" if decision == "BUY" else "#DA3633" if decision == "SELL" else "#444"

                with cols[i]:
                    st.markdown(f"""
                        <div style='text-align:center;'>
                            <p style='font-size:3.0rem; font-weight:bold; color:#DA70D6; margin:0;'>H+{h}</p>
                            <p style='font-size:2.0rem; color:white; font-weight:bold; margin-bottom:15px;'>{date_card}</p>
                            <h2 style='color:{color_trend}; font-size:2.6rem; margin:0;'>${p_h:,.0f}</h2>
                            <p style='font-size:1.6rem; color:white; margin:5px 0;'>{perf:+.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    fig_i = go.Figure(go.Indicator(mode="gauge+number", value=conf, number={'suffix':"%",'font':{'size':30}}, gauge={'bar':{'color':color_trend},'bgcolor':"#222",'axis':{'range':[0,100],'showticklabels':False}}))
                    fig_i.update_layout(height=130, margin=dict(t=0,b=0,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)", font={'color':"white"})
                    st.plotly_chart(fig_i, use_container_width=True, key=f"g_{h}")
                    st.markdown(f"<div style='background:{bg_btn}; color:white; text-align:center; padding:10px; border-radius:8px; font-weight:bold; font-size:1.4rem;'>{decision}</div>", unsafe_allow_html=True)

# --- 7. SIGNATURE (Rétablissement des liens Email et LinkedIn) ---
st.markdown(f"""
    <div class="footer-signature">
       <b>Developed by Mahmoud TOURKI</b><br><br>
        <span>📧 <a href="mailto:mahmoud.tourki24@gmail.com">mahmoud.tourki24@gmail.com</a></span> | 
        <span>🔗 <a href="https://www.linkedin.com/in/mahmoud-tourki-b228b9147/" target="_blank">LinkedIn Profile</a></span>
    </div>
""", unsafe_allow_html=True)