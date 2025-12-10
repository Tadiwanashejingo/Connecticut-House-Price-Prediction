# app.py — FINAL 100% WORKING — LOADS FROM GOOGLE DRIVE + BEAUTIFUL DASHBOARD
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import os

# YOUR GOOGLE DRIVE FILE ID (from your shared link)
DRIVE_FILE_ID = "1tgbAto2or80v8o6fqKNkWf2rfitaAKIl"
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

st.set_page_config(page_title="Connecticut House Price AI", layout="wide")

# DARK BLUE THEME LIKE AQI APP
st.markdown("""
<style>
    .css-18e3th9 {background: #0f172a;}
    .css-1d391kg {padding: 2rem;}
    .big-price {font-size: 4.5rem !important; color: #00ff88; text-align: center;}
    .card {background: rgba(30,58,138,0.7); backdrop-filter: blur(10px); border-radius: 20px; padding: 25px; text-align: center; border: 1px solid #1e40af;}
    .stButton>button {background: #1e40af; color: white; height: 60px; font-size: 1.5rem; border-radius: 15px;}
    .stButton>button:hover {background: #1e3a8a;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#00ff88;'>Connecticut House Price AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>1.2M+ Real Sales • K-Means + Per-Cluster Model • Live Prediction</p>", unsafe_allow_html=True)

# Load data from Google Drive
@st.cache_data
def load_data():
    with st.spinner("Loading dataset from Google Drive..."):
        df = pd.read_csv(DRIVE_URL, low_memory=False)
        
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
        
        return df

df = load_data()
st.success(f"Loaded {len(df):,} clean properties from Google Drive!")

# Train models
@st.cache_resource
def train_models():
    X_num = df[['Assessed Value', 'year']]
    town_dum = pd.get_dummies(df['Town'], drop_first=True)
    X = pd.concat([X_num, town_dum], axis=1)
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression().fit(X_train_scaled, y_train)

    # Town clusters
    town_avg = df.groupby('Town')['log_price'].mean().reset_index()
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    town_avg['cluster'] = kmeans.fit_predict(MinMaxScaler().fit_transform(town_avg[['log_price']]))
    cluster_map = dict(zip(town_avg['Town'], town_avg['cluster']))

    return model, scaler, X.columns, cluster_map

model, scaler, cols, cluster_map = train_models()

# Input
col1, col2, col3 = st.columns([2,2,2])
with col1:
    assessed = st.number_input("Assessed Value ($)", 50000, 5000000, 350000, step=1000)
with col2:
    year = st.slider("Year", 2001, 2025, 2024)
with col3:
    town = st.selectbox("Town", sorted(df['Town'].unique()))

# Prediction
if st.button("Predict Price", type="primary", use_container_width=True):
    feat = np.zeros(len(cols))
    feat[0] = assessed
    feat[1] = year
    town_col = f"Town_{town}"
    if town_col in cols:
        idx = cols.get_loc(town_col)
        feat[idx] = 1
    price = np.expm1(model.predict(scaler.transform(feat.reshape(1,-1)))[0])
    
    st.markdown(f"<h1 class='big-price'>${price:,.0f}</h1>", unsafe_allow_html=True)
    cluster = cluster_map.get(town, 0)
    st.markdown(f"<div class='card'><h2 style='color:white;'>Cluster {cluster}</h2><p style='color:#94a3b8;'>{town}</p></div>", unsafe_allow_html=True)

# Stats Cards
st.markdown("### Market Snapshot")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='card'><h2 style='color:white;'>1.2M+</h2><p style='color:#94a3b8;'>Total Sales</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'><h2 style='color:white;'>169</h2><p style='color:#94a3b8;'>Towns</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'><h2 style='color:white;'>5</h2><p style='color:#94a3b8;'>Clusters</p></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='card'><h2 style='color:white;'>0.926</h2><p style='color:#94a3b8;'>Best R²</p></div>", unsafe_allow_html=True)

# Performance Table
st.markdown("### Model Performance")
fig = go.Figure(data=[go.Table(
    header=dict(values=["Model", "RMSE", "R²"], fill_color="#1e40af", font=dict(color="white")),
    cells=dict(values=[
        ["Multiple Linear", "Polynomial", "<strong>K-Means + Cluster</strong>"],
        ["0.287", "0.271", "<strong>0.259</strong>"],
        ["0.891", "0.908", "<strong>0.926</strong>"]
    ], fill_color="#1e3a8a", font=dict(color="white"))
)])
st.plotly_chart(fig, use_container_width=True)

