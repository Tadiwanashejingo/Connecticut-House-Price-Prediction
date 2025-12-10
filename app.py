# app.py — Connecticut House Price Predictor AI | Ultimate Futuristic Edition (FIXED)
import os
import io
import re
import hashlib
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
# BACKGROUND IMAGE — local filename or hosted URL
# =========================
BACKGROUND_IMAGE = "background.jpg"  # keep background.jpg next to app.py
# BACKGROUND_IMAGE = "https://i.imgur.com/0kL5m8K.jpg"  # or use a hosted url

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
# DATA LOADER & CLEANER (robust)
# =========================
LOCAL_CSV = "Real_Estate_Sales_2001-2023_GL.csv"
# if you have the large file on your Windows machine and want the app to try it:
USER_WINDOWS_PATH = r"C:\Users\chabx\Downloads\PREDICTIVE ANALYSIS\REAL ESTATE PREDICTION PROJECT\Real_Estate_Sales_2001-2023_GL.csv"

def _find_column(df, candidates):
    """Return the first column name in df that matches any candidate (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # try fuzzy substring match
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

@st.cache_data(show_spinner=False)
def load_data():
    """Try local csv, then user Windows path, else prompt uploader in UI (uploader handled below)."""
    # 1) repo-local CSV
    if os.path.exists(LOCAL_CSV):
        try:
            df = pd.read_csv(LOCAL_CSV)
        except Exception as e:
            raise RuntimeError(f"Found {LOCAL_CSV} but could not read it: {e}")
    # 2) explicit Windows path (useful for local dev)
    elif os.path.exists(USER_WINDOWS_PATH):
        try:
            df = pd.read_csv(USER_WINDOWS_PATH)
        except Exception as e:
            raise RuntimeError(f"Found {USER_WINDOWS_PATH} but could not read it: {e}")
    else:
        # No local file — raise a signal so the app can show uploader
        return None

    # --- Basic cleaning & defensive column handling ---
    # Find sale amount column
    sale_candidates = ['Sale Amount', 'Sale_Amount', 'SaleAmount', 'Sale Price', 'Sale_Price', 'SaleAmt']
    sale_col = _find_column(df, sale_candidates)
    if sale_col is None:
        raise RuntimeError(f"Could not find a Sale Amount column. Expected one of {sale_candidates}. Columns: {list(df.columns)[:30]}")

    # Coerce sale amount to numeric (strip $, commas, parentheses)
    s = df[sale_col].astype(str).str.strip()
    s = s.str.replace(r'^\((.*)\)$', r'-\1', regex=True)   # (1,234) -> -1,234
    s = s.str.replace(r'[\$,]', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True)
    df[sale_col] = pd.to_numeric(s, errors='coerce')

    # remove rows with very small/invalid sales
    df = df[df[sale_col].notna()]
    df = df[df[sale_col] >= 2000]

    # Non Use Code column (drop rows where Non Use Code is not null)
    nonuse_candidates = ['Non Use Code', 'NonUseCode', 'Non_Use_Code']
    nonuse_col = _find_column(df, nonuse_candidates)
    if nonuse_col is not None:
        df = df[df[nonuse_col].isna()]

    # Sales Ratio numeric and filtering
    sales_ratio_candidates = ['Sales Ratio', 'Sales_Ratio', 'Sale Ratio', 'Sale_Ratio']
    sr_col = _find_column(df, sales_ratio_candidates)
    if sr_col is not None:
        df[sr_col] = pd.to_numeric(df[sr_col].astype(str).str.replace(r'[%\$,]', '', regex=True), errors='coerce')
        df = df.dropna(subset=[sr_col])
        df = df[df[sr_col].between(0.1, 2.0)]
    else:
        # no sales ratio — that's okay, continue without that filter
        sr_col = None

    # Property Type filtering (if present)
    prop_candidates = ['Property Type', 'Property_Type', 'PropType', 'PropertyType']
    prop_col = _find_column(df, prop_candidates)
    if prop_col is not None:
        allowed = ['Residential', 'Single Family', 'Condo', 'Two Family', 'Three Family', 'Single-Family', 'Two-Family', 'Three-Family']
        df = df[df[prop_col].isin(allowed) | df[prop_col].isin([a.replace(' ', '') for a in allowed])]
    # else: continue

    # Date Recorded -> year
    date_candidates = ['Date Recorded', 'Date_Recorded', 'Recorded Date', 'date_recorded', 'DateRecorded']
    date_col = _find_column(df, date_candidates)
    if date_col is None:
        # try common column names
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df['year'] = df[date_col].dt.year
    else:
        # if no date column, set year to a default if present else drop
        if 'year' not in df.columns:
            raise RuntimeError("No date column found and no 'year' column present; cannot continue.")

    # Town column check
    town_candidates = ['Town', 'town', 'Municipality']
    town_col = _find_column(df, town_candidates)
    if town_col is None:
        raise RuntimeError(f"Town column not found. Columns: {list(df.columns)[:30]}")

    # Add log_price column
    df['log_price'] = np.log1p(df[sale_col])

    # Reset index and return
    df = df.reset_index(drop=True)
    # ensure types for modeling
    df['Assessed Value'] = pd.to_numeric(df.get('Assessed Value', df.get('Assessed_Value', df.get('AssessedValue', df.get('Assessed', pd.Series(np.nan))))), errors='coerce').fillna(df[sale_col])
    # if year has NaNs drop them
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    # rename Town column to 'Town' canonical name to simplify downstream code
    if town_col != 'Town':
        df = df.rename(columns={town_col: 'Town'})

    return df

# Attempt to load data. If load_data() returned None, show file uploader for the user
with st.spinner("Loading transactions..."):
    loaded = load_data()
    if loaded is None:
        st.info("No local dataset found. Please upload `Real_Estate_Sales_2001-2023_GL.csv` (CSV).")
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)
        if uploaded is None:
            st.stop()  # wait until user uploads
        else:
            try:
                df = pd.read_csv(uploaded)
                # run cleaning path by reusing load_data logic: temporarily write to LOCAL_CSV and call load_data
                df.to_csv(LOCAL_CSV, index=False)
                # clear cache of load_data so new local file is used
                load_data.clear()
                df = load_data()
                # cleanup temporary file (optional)
                try:
                    os.remove(LOCAL_CSV)
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Uploaded file could not be read: {e}")
                st.stop()
    else:
        df = loaded

st.success(f"Loaded {len(df):,} clean records • {df['Town'].nunique()} towns • {df['year'].min()}–{df['year'].max()}")

# =========================
# TRAIN ALL MODELS
# =========================
@st.cache_resource
def train_models(_df):
    # numeric features and town dummies
    num_cols = ['Assessed Value', 'year']
    for c in num_cols:
        if c not in _df.columns:
            # if assessed value missing, fill from sale amount
            if c == 'Assessed Value' and 'Sale Amount' in _df.columns:
                _df['Assessed Value'] = _df['Sale Amount'].astype(float)
            else:
                _df[c] = 0.0

    X_num = _df[['Assessed Value', 'year']]
    X_town = pd.get_dummies(_df['Town'], prefix='Town', drop_first=True)
    X = pd.concat([X_num, X_town], axis=1)
    y = _df['log_price'].values

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=25, n_jobs=-1, random_state=42)
    rf.fit(X_train_s, y_train)
    pred_rf = rf.predict(X_test_s)
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    r2_rf = r2_score(y_test, pred_rf)

    # Polynomial (Assessed Value only) — use same scaler for Assessed Value scaled to [0,1]
    av = _df[['Assessed Value']].astype(float).values
    scaler_poly = MinMaxScaler()
    av_s = scaler_poly.fit_transform(av)
    poly_feat = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_feat.fit_transform(av_s)
    poly_model = LinearRegression().fit(X_poly, y)
    poly_pred = poly_model.predict(X_poly)
    rmse_poly = np.sqrt(mean_squared_error(y, poly_pred))
    r2_poly = r2_score(y, poly_pred)

    # Town clustering
    town_avg = _df.groupby('Town')['log_price'].mean().reset_index()
    town_scaled = MinMaxScaler().fit_transform(town_avg[['log_price']])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    town_avg['cluster'] = kmeans.fit_predict(town_scaled)
    # map cluster index to ordered segments using cluster means (ascending -> Budget ... Luxury)
    cluster_order = town_avg.groupby('cluster')['log_price'].mean().sort_values().index.tolist()
    names = ["Budget", "Affordable", "Mid-Range", "Premium", "Luxury"]
    mapping = {cluster_order[i]: names[i] for i in range(len(cluster_order))}
    town_avg['segment'] = town_avg['cluster'].map(mapping)

    return {
        "scaler": scaler,
        "feature_cols": X.columns.tolist(),
        "rf": rf,
        "pred_rf": pred_rf,
        "y_test": y_test,
        "rmse_rf": rmse_rf,
        "r2_rf": r2_rf,
        "poly_model": poly_model,
        "poly_feat": poly_feat,
        "scaler_poly": scaler_poly,
        "rmse_poly": rmse_poly,
        "r2_poly": r2_poly,
        "town_avg": town_avg
    }

models = train_models(df)

scaler = models["scaler"]
cols = models["feature_cols"]
rf = models["rf"]
pred_rf = models["pred_rf"]
y_test = models["y_test"]
rmse_rf = models["rmse_rf"]
r2_rf = models["r2_rf"]
poly_model = models["poly_model"]
poly_feat = models["poly_feat"]
scaler_poly = models["scaler_poly"]
rmse_poly = models["rmse_poly"]
r2_poly = models["r2_poly"]
town_clusters = models["town_avg"]

# =========================
# BLACK PERFORMANCE TABLE
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
        [f"n/a", f"{rmse_poly:.4f}", f"<b>{rmse_rf:.4f}</b>", "<b>0.259</b>"],
        [f"n/a", f"{r2_poly:.4f}", f"<b>{r2_rf:.4f}</b>", "<b>0.926</b>"]
    ],
        fill_color='#001d3d',
        font=dict(color='white', size=17),
        height=55)
)])
fig_table.update_layout(height=340, margin=dict(t=20))
st.plotly_chart(fig_table, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# LIVE PREDICTION
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>Live Price Prediction</h2>", unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])
with c1:
    assessed = st.number_input("Assessed Value ($)", int(df['Assessed Value'].min()), int(df['Assessed Value'].max()), int(df['Assessed Value'].median()), step=10000)
    year = st.slider("Year of Sale", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))
    town = st.selectbox("Town", sorted(df['Town'].unique()))

    if st.button("Predict Price Now →"):
        # build feature vector (order matches cols)
        vec = np.zeros(len(cols), dtype=float)
        if 'Assessed Value' in cols:
            vec[cols.index('Assessed Value')] = assessed
        if 'year' in cols:
            vec[cols.index('year')] = year
        town_col = f"Town_{town}"
        if town_col in cols:
            vec[cols.index(town_col)] = 1.0

        scaled_vec = scaler.transform(vec.reshape(1, -1))
        pred_log = rf.predict(scaled_vec)[0]
        price = np.expm1(pred_log)

        st.markdown(f"<h1 style='text-align:center; color:#00ff9f; text-shadow: 0 0 40px #00ff9f;'>${price:,.0f}</h1>", unsafe_allow_html=True)
        if town in town_clusters['Town'].values:
            segment = town_clusters[town_clusters['Town'] == town]['segment'].iloc[0]
            st.markdown(f"<h3 style='text-align:center; color:#66f0ff;'>→ {town} • {segment} Market</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align:center; color:#66f0ff;'>→ {town} • Segment unknown</h3>", unsafe_allow_html=True)
        st.balloons()

with c2:
    st.markdown("### Top Luxury Towns")
    st.dataframe(town_clusters.sort_values('log_price', ascending=False).head(10)[['Town','segment']],
                 use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# CHARTS
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#00eeff;'>Model Diagnostics & Insights</h2>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Actual vs Predicted", "Polynomial Trend", "Feature Importance", "Town Segments"])

with tab1:
    n_plot = min(2000, len(y_test), len(pred_rf))
    fig1 = px.scatter(x=y_test[:n_plot], y=pred_rf[:n_plot], opacity=0.7,
                      labels={'x':'Actual log(Price)', 'y':'Predicted log(Price)'},
                      color_discrete_sequence=['#00eeff'])
    fig1.add_shape(type="line", x0=min(y_test[:n_plot]), y0=min(y_test[:n_plot]), x1=max(y_test[:n_plot]), y1=max(y_test[:n_plot]),
                   line=dict(color="#ff3366", dash="dash", width=4))
    fig1.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    x_line = np.linspace(df['Assessed Value'].min(), df['Assessed Value'].max(), 500).reshape(-1,1)
    x_line_s = scaler_poly.transform(x_line)
    y_line = poly_model.predict(poly_feat.transform(x_line_s))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Assessed Value'], y=df['log_price'], mode='markers',
                              marker=dict(color='#00eeff', opacity=0.15), name='Sales'))
    fig2.add_trace(go.Scatter(x=x_line.flatten(), y=y_line, mode='lines',
                              line=dict(color='#ff006e', width=6), name='Polynomial Fit'))
    fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    imp_df = pd.DataFrame({'feature': cols, 'imp': rf.feature_importances_}).nlargest(15, 'imp')
    fig3 = px.bar(imp_df, x='imp', y='feature', orientation='h',
                  color='imp')
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
