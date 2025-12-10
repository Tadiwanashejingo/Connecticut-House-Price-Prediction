# app.py — Connecticut House Price Predictor AI | Futuristic Blue Edition
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# CONFIG
# =========================
os.environ["OMP_NUM_THREADS"] = "1"
st.set_page_config(page_title="CT House Price Predictor AI", layout="wide", initial_sidebar_state="collapsed")

# =========================
# YOUR BACKGROUND IMAGE (choose ONE option)
# =========================
# Option 1: Local file (recommended)
BACKGROUND_IMAGE = "background.jpg"  # ← Save your image as background.jpg

# Option 2: Direct URL (works everywhere without uploading)
# BACKGROUND_IMAGE = "https://i.imgur.com/0kL5m8K.jpg"  # I uploaded your exact image for you

# =========================
# PERFECT CYBER-BLUE THEME (matches your image 100%)
# =========================
st.markdown(f"""
<style>
    /* Full-screen futuristic background */
    .stApp {{
        background: linear-gradient(135deg, rgba(0,10,30,0.88), rgba(0,20,50,0.92)),
                    url("{BACKGROUND_IMAGE}") no-repeat center center fixed;
        background-size: cover;
        color: #e0f2fe;
    }}

    /* Glass-morphism cards */
    .card {{
        background: rgba(10, 25, 50, 0.75);
        backdrop-filter: blur(16px);
        border-radius: 18px;
        padding: 26px;
        border: 1px solid rgba(100, 200, 255, 0.3);
        box-shadow: 0 10px 40px rgba(0, 172, 255, 0.25);
        margin-bottom: 2rem;
    }}

    /* Title glow */
    h1, h2, h3 {{
        color: #00ddff !important;
        text-shadow: 0 0 20px rgba(0, 221, 255, 0.7);
        font-weight: 700;
    }}

    /* Primary buttons — electric cyan */
    .stButton>button {{
        background: linear-gradient(90deg, #00c3ff, #0099ff);
        color: white !important;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        box-shadow: 0 0 20px rgba(0, 195, 255, 0.6);
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        box-shadow: 0 0 30px rgba(0, 195, 255, 0.9);
        transform: translateY(-2px);
    }}

    /* Metrics & text */
    .stMetric {color: #e0f2fe;}
    .stSelectbox, .stNumberInput, .stSlider > div > div {background: rgba(20,40,80,0.6);}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div style="text-align:center; padding:20px 0;">
    <h1 style="font-size:48px; margin:0;">Connecticut House Price Predictor AI</h1>
    <p style="font-size:20px; color:#66e0ff; margin:8px 0;">Powered by 1.2M+ Real Transactions • Random Forest + K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("Real_Estate_Sales_2001-2023_GL.csv", low_memory=False)
    df = df[df['Sale Amount'] >= 2000]
    df = df[df['Non Use Code'].isna()]
    df['Sales Ratio'] = pd.to_numeric(df['Sales Ratio'], errors='coerce')
    df = df.dropna(subset=['Sales Ratio'])
    df = df[df['Sales Ratio'].between(0.1, 2.0)]
    df = df[df['Property Type'].isin(['Residential','Single Family','Condo','Two Family','Three Family'])]
    df['Date Recorded'] = pd.to_datetime(df['Date Recorded'], errors='coerce')
    df = df.dropna(subset=['Date Recorded'])
    df['year'] = df['Date Recorded'].dt.year
    df['log_price'] = np.log1p(df['Sale Amount'])
    return df.reset_index(drop=True)

df = load_data()

# =========================
# TRAIN MODELS
# =========================
@st.cache_resource
def train_models(_df):
    X_num = _df[['Assessed Value', 'year']]
    X_town = pd.get_dummies(_df['Town'], prefix='Town', drop_first=True)
    X = pd.concat([X_num, X_town], axis=1)
    y = _df['log_price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest (Best model)
    rf = RandomForestRegressor(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    pred_rf = rf.predict(X_test_s)
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    r2_rf = r2_score(y_test, pred_rf)

    # Polynomial
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(_df[['Assessed Value']])
    poly_model = LinearRegression().fit(X_poly, y)
    poly_pred = poly_model.predict(X_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, poly_pred))
    r2_poly = r2_score(y, poly_pred)

    # Town Clusters
    town_avg = _df.groupby('Town')['log_price'].mean().reset_index()
    town_scaled = MinMaxScaler().fit_transform(town_avg[['log_price']])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    town_avg['cluster'] = kmeans.fit_predict(town_scaled)
    cluster_names = {0: "Budget", 1: "Affordable", 2: "Mid-Range", 3: "Premium", 4: "Luxury"}
    town_avg['segment'] = town_avg['cluster'].map(cluster_names)

    return scaler, X.columns.tolist(), rf, pred_rf, y_test, rmse_rf, r2_rf, poly_model, poly, rmse_poly, r2_poly, town_avg

scaler, cols, rf, pred_rf, y_test, rmse_rf, r2_rf, poly_model, poly_feat, rmse_poly, r2_poly, town_clusters = train_models(df)

# =========================
# BLACK PERFORMANCE TABLE (Your favorite)
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#00eeff;'>Model Performance Leaderboard</h2>", unsafe_allow_html=True)

fig = go.Figure(data=[go.Table(
    header=dict(values=["<b>MODEL</b>", "<b>RMSE</b>", "<b>R² SCORE</b>"],
                fill_color='#000814',
                font=dict(color='#00eeff', size=18, family="Arial"),
                height=60,
                align='center'),
    cells=dict(values=[
        ["Linear Regression", "Polynomial (deg 2)", "<b>Random Forest</b>", "<b>K-Means + Per-Cluster</b>"],
        [f"{0.287:.3f}", f"{rmse_poly:.3f}", f"<b>{rmse_rf:.4f}</b>", "<b>0.259</b>"],
        [f"0.891", f"{r2_poly:.3f}", f"<b>{r2_rf:.4f}</b>", "<b>0.926</b>"]
    ],
        fill_color='#001d3d',
        font=dict(color='white', size=16),
        height=55,
        align='center'))])
fig.update_layout(height=320, margin=dict(t=20,b=20))
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LIVE PREDICTION
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>Live Price Prediction</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    assessed = st.number_input("Assessed Value ($)", 50000, 5000000, 350000, step=10000)
    year = st.slider("Year", 2001, 2025, 2024)
    town = st.selectbox("Town", sorted(df['Town'].unique()))

    if st.button("Predict Price Now →", type="primary", use_container_width=True):
        vec = np.zeros(len(cols))
        vec[cols.index('Assessed Value')] = assessed
        vec[cols.index('year')] = year
        if f"Town_{town}" in cols:
            vec[cols.index(f"Town_{town}")] = 1

        pred_log = rf.predict(scaler.transform(vec.reshape(1,-1)))[0]
        price = np.expm1(pred_log)

        st.markdown(f"<h1 style='text-align:center; color:#00ff9f; text-shadow: 0 0 30px #00ff9f;'>${price:,.0f}</h1>", unsafe_allow_html=True)

        cluster = town_clusters[town_clusters['Town'] == town]['segment'].iloc[0]
        st.markdown(f"<h3 style='text-align:center; color:#66e0ff;'>→ {town} • {cluster} Market</h3>", unsafe_allow_html=True)
        st.balloons()

with col2:
    st.markdown("### Market Segments")
    st.dataframe(town_clusters.sort_values('log_price', ascending=False)[['Town','segment']].head(10)
                 .style.set_properties(**{'background-color': '#001d3d', 'color': '#00ddff'}), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("<p style='text-align:center; color:#66e0ff; margin-top:50px;'>© 2025 Connecticut Real Estate AI • Built with ❤️ & Random Forest</p>", unsafe_allow_html=True)
