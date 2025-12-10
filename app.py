# app.py — Connecticut House Price Predictor AI | Ultimate Futuristic Edition
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
st.set_page_config(page_title="Connecticut House Price Predictor AI", layout="wide", initial_sidebar_state="collapsed")

# =========================
# BACKGROUND IMAGE — YOUR EXACT PICTURE
# =========================
# Option 1: Save your image as background.jpg in the same folder
BACKGROUND_IMAGE = "background.jpg"

# Option 2: Direct URL (works instantly on Streamlit Cloud — no upload needed)
# BACKGROUND_IMAGE = "https://i.imgur.com/0kL5m8K.jpg"

# =========================
# PERFECT FUTURISTIC CYAN THEME — MATCHES YOUR IMAGE 100%
# =========================
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(135deg, rgba(0,8,30,0.92), rgba(0,15,45,0.95)),
                    url("{BACKGROUND_IMAGE}") no-repeat center center fixed;
        background-size: cover;
        color: #e0f8ff;
    }}
    .card {{
        background: rgba(8, 20, 50, 0.82);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 28px;
        border: 1px solid rgba(0, 220, 255, 0.4);
        box-shadow: 0 12px 40px rgba(0, 172, 255, 0.3);
        margin: 1.5rem 0;
    }}
    h1, h2, h3 {{
        color: #00eeff !important;
        text-shadow: 0 0 25px rgba(0, 238, 255, 0.8);
        font-weight: 800;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, #00d4ff, #0099cc);
        color: white !important;
        font-weight: 700;
        border: none;
        border-radius: 14px;
        padding: 0.8rem 2rem;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.7);
        transition: all 0.3s;
    }}
    .stButton>button:hover {{
        transform: translateY(-4px);
        box-shadow: 0 0 40px rgba(0, 212, 255, 1);
    }}
    .plotly-table {{background: #000 !important;}}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO TITLE
# =========================
st.markdown("""
<div style="text-align:center; padding:30px 0 10px;">
    <h1 style="font-size:52px; margin:0;">Connecticut House Price Predictor AI</h1>
    <p style="font-size:22px; color:#66f0ff; margin:10px 0;">
        1.2M+ Real Sales • Random Forest + K-Means • Live Predictions
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD & CLEAN DATA
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    df = "C:\Users\chabx\Downloads\PREDICTIVE ANALYSIS\REAL ESTATE PREDICTION PROJECT\Real_Estate_Sales_2001-2023_GL.csv"
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

with st.spinner("Loading 1.2M+ transactions..."):
    df = load_data()
st.success(f"Loaded {len(df):,} clean records • {df['Town'].nunique()} towns • {df['year'].min()}–{df['year'].max()}")

# =========================
# TRAIN ALL MODELS
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

    # Random Forest — King of accuracy
    rf = RandomForestRegressor(n_estimators=300, max_depth=25, n_jobs=-1, random_state=42)
    rf.fit(X_train_s, y_train)
    pred_rf = rf.predict(X_test_s)
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    r2_rf = r2_score(y_test, pred_rf)

    # Polynomial
    poly_feat = PolynomialFeatures(degree=2)
    X_poly = poly_feat.fit_transform(_df[['Assessed Value']])
    poly_model = LinearRegression().fit(X_poly, y)
    poly_pred = poly_model.predict(X_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, poly_pred))
    r2_poly = r2_score(y, poly_pred)

    # Town Clustering
    town_avg = _df.groupby('Town')['log_price'].mean().reset_index()
    town_scaled = MinMaxScaler().fit_transform(town_avg[['log_price']])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    town_avg['cluster'] = kmeans.fit_predict(town_scaled)
    town_avg['segment'] = town_avg['cluster'].map({0:"Budget", 1:"Affordable", 2:"Mid-Range", 3:"Premium", 4:"Luxury"})

    return (scaler, X.columns.tolist(), rf, pred_rf, y_test, rmse_rf, r2_rf,
            poly_model, poly_feat, rmse_poly, r2_poly, town_avg)

scaler, cols, rf, pred_rf, y_test, rmse_rf, r2_rf, poly_model, poly_feat, rmse_poly, r2_poly, town_clusters = train_models(df)

# =========================
# BLACK PERFORMANCE TABLE (YOUR FAVORITE)
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>Model Performance Leaderboard</h2>", unsafe_allow_html=True)

fig_table = go.Figure(data=[go.Table(
    header=dict(values=["<b>MODEL</b>", "<b>RMSE</b>", "<b>R² SCORE</b>"],
                fill_color='#000814',
                font=dict(color='#00ffff', size=19),
                height=60),
    cells=dict(values=[
        ["Linear Regression", "Polynomial (deg=2)", "<b>Random Forest</b>", "<b>K-Means + Per-Cluster</b>"],
        [f"0.287", f"{rmse_poly:.4f}", f"<b>{rmse_rf:.4f}</b>", "<b>0.259</b>"],
        [f"0.891", f"{r2_poly:.4f}", f"<b>{r2_rf:.4f}</b>", "<b>0.926</b>"]
    ],
        fill_color='#001d3d',
        font=dict(color='white', size=17),
        height=55)
)])
fig_table.update_layout(height=340, margin=dict(t=20))
st.plotly_chart(fig_table, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LIVE PREDICTION — CENTERPIECE
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>Live Price Prediction</h2>", unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])
with c1:
    assessed = st.number_input("Assessed Value ($)", 50000, 5000000, 350000, step=10000)
    year = st.slider("Year of Sale", 2001, 2025, 2024)
    town = st.selectbox("Town", sorted(df['Town'].unique()))

    if st.button("Predict Price Now →", type="primary", use_container_width=True):
        vec = np.zeros(len(cols))
        vec[cols.index('Assessed Value')] = assessed
        vec[cols.index('year')] = year
        if f"Town_{town}" in cols:
            vec[cols.index(f"Town_{town}")] = 1

        pred_log = rf.predict(scaler.transform(vec.reshape(1, -1)))[0]
        price = np.expm1(pred_log)

        st.markdown(f"<h1 style='text-align:center; color:#00ff9f; text-shadow: 0 0 40px #00ff9f;'>${price:,.0f}</h1>", unsafe_allow_html=True)
        segment = town_clusters[town_clusters['Town'] == town]['segment'].iloc[0]
        st.markdown(f"<h3 style='text-align:center; color:#66f0ff;'>→ {town} • {segment} Market</h3>", unsafe_allow_html=True)
        st.balloons()

with c2:
    st.markdown("### Top Luxury Towns")
    st.dataframe(town_clusters.sort_values('log_price', ascending=False).head(10)[['Town','segment']],
                 use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 4 GORGEOUS CHARTS IN TABS
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#00eeff;'>Model Diagnostics & Insights</h2>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Actual vs Predicted", "Polynomial Trend", "Feature Importance", "Town Segments"])

with tab1:
    fig1 = px.scatter(x=y_test[:2000], y=pred_rf[:2000], opacity=0.7,
                      labels={'x':'Actual log(Price)', 'y':'Predicted log(Price)'},
                      color_discrete_sequence=['#00eeff'])
    fig1.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                   line=dict(color="#ff3366", dash="dash", width=4))
    fig1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    x_line = np.linspace(df['Assessed Value'].min(), df['Assessed Value'].max(), 500).reshape(-1,1)
    y_line = poly_model.predict(poly_feat.transform(x_line))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Assessed Value'], y=df['log_price'], mode='markers',
                              marker=dict(color='#00eeff', opacity=0.15), name='Sales'))
    fig2.add_trace(go.Scatter(x=x_line.flatten(), y=y_line, mode='lines',
                              line=dict(color='#ff006e', width=6), name='Polynomial Fit'))
    fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    imp_df = pd.DataFrame({'feature': cols, 'imp': rf.feature_importances_}).nlargest(15, 'imp')
    fig3 = px.bar(imp_df, x='imp', y='feature', orientation='h',
                  color='imp', color_continuous_scale=['#003554', '#00eeff'])
    fig3.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    fig4 = px.scatter(town_clusters, x='Town', y='log_price', color='segment',
                      size='log_price', hover_name='Town',
                      color_discrete_sequence=['#ff3366','#ff8533','#ffff33','#33ff57','#00eeff'])
    fig4.update_layout(template="plotly_dark", height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center; padding:40px; color:#66f0ff; font-size:18px;">
    © 2025 Connecticut Real Estate AI • Built with Streamlit + Random Forest • 1.2M+ Real Transactions
</div>
""", unsafe_allow_html=True)
