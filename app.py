# app.py — Connecticut House Price ML Dashboard (Full feature) with robust price diagnostics
import os
import io
import base64
import joblib
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="CT House Price — Full ML Dashboard", layout="wide", initial_sidebar_state="expanded")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Default Drive ID (your provided file)
DRIVE_FILE_ID_DEFAULT = "1tgbAto2or80v8o6fqKNkWf2rfitaAKIl"
DRIVE_DOWNLOAD_URL = lambda fid: f"https://drive.google.com/uc?export=download&id={fid}"

# ---------------------------
# Styles (dark theme & cards)
# ---------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#041023 0%, #00101a 100%); color: #e6f7ff; }
    .hero { background: linear-gradient(90deg,#07263b,#041c2b); padding: 28px; border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
    .hero h1 { color: #7fe0f5; margin:0; }
    .card { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; border: 1px solid rgba(127,224,245,0.06); }
    .muted { color: #bcdfe8; font-size:14px }
    .kpi { font-size:20px; font-weight:700; color:#bfefff }
    .small { font-size:12px; color:#bcdfe8 }
    .download-btn { background: #0ea5b7; color: white; padding: 8px 12px; border-radius:6px; }
    </style>
    """, unsafe_allow_html=True
)

# ---------------------------
# Utilities
# ---------------------------
def safe_read_csv(path_or_buf):
    return pd.read_csv(path_or_buf, low_memory=False)

@st.cache_data(show_spinner=False)
def load_from_drive_url(url: str) -> pd.DataFrame:
    return safe_read_csv(url)

@st.cache_data(show_spinner=False)
def load_from_filelike(uploaded) -> pd.DataFrame:
    return safe_read_csv(uploaded)

def ensure_cols(df):
    # normalize common column names
    if 'SalePrice' in df.columns and 'Sale Amount' not in df.columns:
        df['Sale Amount'] = df['SalePrice']
    return df

def smoothed_target_encoding(train_series, target_series, min_samples_leaf=20, smoothing=10):
    df = pd.DataFrame({'cat': train_series, 'y': target_series})
    averages = df.groupby('cat')['y'].agg(['mean','count'])
    global_mean = target_series.mean()
    averages['smooth'] = (averages['count'] * averages['mean'] + smoothing * global_mean) / (averages['count'] + smoothing)
    mapping = averages['smooth'].to_dict()
    return mapping, global_mean

def apply_target_encoding(series, mapping, global_mean):
    return series.map(mapping).fillna(global_mean)

def model_save_bytes(obj):
    bio = io.BytesIO()
    joblib.dump(obj, bio)
    bio.seek(0)
    return bio.read()

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ---------------------------
# Sidebar: Data source + options
# ---------------------------
st.sidebar.title("Data & Model Settings")
st.sidebar.markdown("Load dataset from Google Drive (direct link) or upload CSV as fallback.")

drive_id = st.sidebar.text_input("Google Drive FILE_ID", value=DRIVE_FILE_ID_DEFAULT)
use_drive = st.sidebar.checkbox("Attempt Drive load first", value=True)
uploaded = st.sidebar.file_uploader("Upload CSV fallback", type=["csv"])

# modeling options
st.sidebar.markdown("---")
st.sidebar.markdown("### Modeling options")
use_target_enc = st.sidebar.checkbox("Use Target Encoding for Town (smoothed mean)", value=False)
target_enc_min_samples = st.sidebar.number_input("Min samples for Town smoothing", value=20, step=5)
target_enc_smoothing = st.sidebar.number_input("Town smoothing factor", value=10, step=1)

k_clusters = st.sidebar.slider("K-Means town clusters (when running KMeans)", 2, 12, 5)
random_state = int(st.sidebar.number_input("Random seed", value=42, step=1))

# ---------------------------
# Load Data
# ---------------------------
df = None
load_error = None
if use_drive and drive_id.strip():
    try:
        with st.spinner("Loading CSV from Google Drive..."):
            df = load_from_drive_url(DRIVE_DOWNLOAD_URL(drive_id.strip()))
            st.sidebar.success("Loaded from Drive")
    except Exception as e:
        load_error = str(e)
        st.sidebar.error(f"Drive load failed: {e}")
        df = None

if df is None and uploaded is not None:
    try:
        df = load_from_filelike(uploaded)
        st.sidebar.success("Loaded uploaded CSV")
    except Exception as e:
        load_error = str(e)
        st.sidebar.error(f"Upload failed: {e}")

if df is None:
    st.sidebar.info("No dataset loaded. Provide Drive ID or upload CSV.")
    st.stop()

# Normalize columns
df = ensure_cols(df)

# ---------------------------
# DIAGNOSTIC + Robust price-column detection
# This block ensures a usable 'Sale Amount' and 'log_price' column exists
# ---------------------------
st.markdown("## Dataset diagnostics")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))
with st.expander("Show first 10 rows"):
    st.dataframe(df.head(10), use_container_width=True)

# If DataFrame empty, show helpful hints
if df.shape[0] == 0:
    st.error("DataFrame has 0 rows. Common causes:\n"
             "- Wrong Drive ID or file not shared publicly\n"
             "- The file uses a non-standard delimiter (e.g., ';')\n"
             "- File has header rows to skip\n             )
    st.markdown("### Try reloading with different CSV options")
    alt_sep = st.selectbox("Try delimiter", options=[",", ";", "\t", "|"], index=0)
    alt_header = st.number_input("Header row index (0-based) if first rows are garbage", min_value=0, max_value=10, value=0)
    alt_enc = st.text_input("Encoding (leave blank to auto-detect)", value="")
    # Provide instructions rather than attempting an automatic re-read here (complex with Drive/file paths)
    st.info("If re-reading is needed, re-upload the CSV with the appropriate delimiter/encoding or correct your Drive file.")
# Attempt to auto-detect price column
common_price_names = [
    'Sale Amount', 'Sale_Amount', 'sale_amount', 'SaleAmount',
    'SalePrice', 'Sale Price', 'saleprice', 'price', 'Price',
    'SALE AMOUNT', 'SALEPRICE'
]
# Exact-normalized matches
found_exact = [c for c in df.columns if any(c.strip().lower().replace('_',' ').replace('-',' ') == name.lower().replace('_',' ').replace('-',' ') for name in common_price_names)]
# Keyword matches
found_kw = [c for c in df.columns if ('price' in c.lower() or 'amount' in c.lower() or 'sale' in c.lower())]

found = found_exact or found_kw
if found:
    st.success(f"Detected possible price columns: {found}")
    price_col = st.selectbox("Select confirmed price column", options=found, index=0)
    if st.button("Use this column as price"):
        # strip non-numeric characters (commas, $) then convert
        df['Sale Amount'] = pd.to_numeric(df[price_col].astype(str).str.replace('[^0-9.-]','', regex=True), errors='coerce')
        df['log_price'] = np.log1p(df['Sale Amount'])
        if df['Sale Amount'].isnull().all():
            st.error("Conversion produced only NaN values — check the column contains numeric values (remove $/commas).")
        else:
            st.success(f"Created 'Sale Amount' and 'log_price' from '{price_col}'.")
            st.write(df[['Sale Amount','log_price']].head())
else:
    st.warning("No likely price column found automatically.")
    pick = st.selectbox("Pick a column to use as price (or re-upload corrected CSV)", options=list(df.columns) + ["None"])
    if pick != "None":
        if st.button(f"Use '{pick}' as price column"):
            df['Sale Amount'] = pd.to_numeric(df[pick].astype(str).str.replace('[^0-9.-]','', regex=True), errors='coerce')
            df['log_price'] = np.log1p(df['Sale Amount'])
            if df['Sale Amount'].isnull().all():
                st.error("Conversion produced only NaN values — check the column contains numeric values (remove $/commas).")
            else:
                st.success(f"Created 'Sale Amount' and 'log_price' from '{pick}'.")
                st.write(df[['Sale Amount','log_price']].head())

# Final check: proceed only if log_price exists and not all NaN
if 'log_price' not in df.columns or df['log_price'].isnull().all():
    st.error("No price column present or conversion failed. Resolve the price column (see diagnostics above).")
    st.stop()

# ---------------------------
# Continue normal processing (cleaning & filtering)
# ---------------------------
# cleaning & engineered columns (like your script)
if 'Date Recorded' in df.columns:
    df['Date Recorded'] = pd.to_datetime(df['Date Recorded'], errors='coerce')
    df = df.dropna(subset=['Date Recorded'])
    df['year'] = df['Date Recorded'].dt.year
    df['month'] = df['Date Recorded'].dt.month
else:
    if 'year' not in df.columns:
        df['year'] = df.get('year', 2000)
    if 'month' not in df.columns:
        df['month'] = df.get('month', 1)

# Ensure log_price numeric
df['log_price'] = pd.to_numeric(df['log_price'], errors='coerce')

# simple filters
if 'Sales Ratio' in df.columns:
    df = df[df['Sales Ratio'].between(0.05, 5.0, inclusive="both")]
if 'Non Use Code' in df.columns:
    df = df[df['Non Use Code'].isna()]
if 'Sale Amount' in df.columns:
    df = df[df['Sale Amount'] >= 2000]
if 'Property Type' in df.columns:
    df = df[df['Property Type'].isin(['Residential', 'Single Family', 'Condo', 'Two Family', 'Three Family'])]

st.markdown(f"**After lightweight cleaning:** {df.shape[0]:,} rows")

# ---------------------------
# Feature selection UI
# ---------------------------
st.markdown("---")
st.subheader("Feature selection & temporal split")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_base = [c for c in ['Assessed Value', 'year', 'month'] if c in numeric_cols]
features = st.multiselect("Numeric features (choose features to include)", options=numeric_cols, default=default_base)
if 'log_price' in features:
    features = [f for f in features if f != 'log_price']

has_town = 'Town' in df.columns
if has_town:
    st.write("Town column found and will be used for Town encoding (OHE or TargetEnc).")

min_year = int(df['year'].min())
max_year = int(df['year'].max())
train_until = st.slider("Train years ≤", min_value=min_year, max_value=max_year-1, value=min(2020, max_year-1))
test_from   = st.slider("Test years ≥", min_value=train_until+1, max_value=max_year, value=max(train_until+1, min_year+1))

train_df = df[df['year'] <= train_until].copy()
test_df  = df[df['year'] >= test_from].copy()
st.write(f"Train rows: {train_df.shape[0]:,} — Test rows: {test_df.shape[0]:,}")

if len(features) < 1:
    st.error("Select at least one numeric feature.")
    st.stop()

town_enc_map = None
global_town_mean = None
if use_target_enc and has_town:
    town_enc_map, global_town_mean = smoothed_target_encoding(train_df['Town'], train_df['log_price'],
                                                             min_samples_leaf=target_enc_min_samples,
                                                             smoothing=target_enc_smoothing)
    st.sidebar.success("Target encoding mapping computed from training data")

def prepare_design(df_local, features_local, town_enc_map=None, global_mean=None, one_hot=False):
    X_num = df_local[features_local].reset_index(drop=True)
    if has_town:
        if one_hot:
            dummies = pd.get_dummies(df_local['Town'], prefix='town', drop_first=True).reset_index(drop=True)
            X = pd.concat([X_num, dummies], axis=1)
        else:
            if town_enc_map is not None:
                X_num['town_te'] = apply_target_encoding(df_local['Town'], town_enc_map, global_mean)
            else:
                X_num['town_te'] = apply_target_encoding(df_local['Town'], {}, df_local['log_price'].mean())
            X = X_num
    else:
        X = X_num
    X = X.fillna(0)
    return X

one_hot_mode = not use_target_enc

X_train_base = prepare_design(train_df, features, town_enc_map if use_target_enc else None,
                              global_town_mean, one_hot=one_hot_mode)
X_test_base  = prepare_design(test_df,  features, town_enc_map if use_target_enc else None,
                              global_town_mean, one_hot=one_hot_mode)

y_train = train_df['log_price'].values
y_test  = test_df['log_price'].values

# ---------------------------
# Modeling controls & run
# ---------------------------
st.markdown("---")
st.subheader("Train models / Run experiments")
col1, col2, col3 = st.columns([2,1,1])
with col1:
    model_choice = st.selectbox("Model to train", options=["All", "Linear Regression", "Polynomial Regression", "KMeans + Per-Cluster Regression"])
with col2:
    poly_degree = st.number_input("Polynomial degree (applied to year)", min_value=2, max_value=6, value=3)
with col3:
    run_button = st.button("Train / Run")

results = {}

def train_linear(Xtr, ytr, Xte):
    model = LinearRegression()
    model.fit(Xtr, ytr)
    return model, model.predict(Xte)

def train_polynomial(train_df_local, test_df_local, Xtr_base, Xte_base, deg=3):
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Yt = poly.fit_transform(train_df_local[['year']])
    Yte = poly.transform(test_df_local[['year']])
    Xtr_poly = np.hstack([Xtr_base.values, Yt])
    Xte_poly = np.hstack([Xte_base.values, Yte])
    model = LinearRegression()
    model.fit(Xtr_poly, train_df_local['log_price'].values)
    return model, model.predict(Xte_poly)

def kmeans_per_cluster_regression(train_df_local, test_df_local, Xtr_base, Xte_base, k=k_clusters):
    if 'Town' not in train_df_local.columns:
        raise ValueError("Town column required for KMeans approach.")
    town_stats = train_df_local.groupby('Town')['log_price'].agg(['mean','std','count']).fillna(0)
    town_stats = town_stats[town_stats['count'] >= 30]
    if town_stats.shape[0] < 2:
        raise ValueError("Not enough towns with >=30 records to run KMeans reliably.")
    scaler = StandardScaler()
    ts = scaler.fit_transform(town_stats[['mean','std']])
    k_use = min(k, max(2, town_stats.shape[0]))
    kmeans = KMeans(n_clusters=k_use, random_state=random_state, n_init=10)
    town_stats['cluster'] = kmeans.fit_predict(ts)
    train_local = train_df_local.merge(town_stats[['cluster']], left_on='Town', right_index=True, how='left')
    test_local  = test_df_local.merge(town_stats[['cluster']], left_on='Town', right_index=True, how='left')
    train_local['cluster'] = train_local['cluster'].fillna(-1).astype(int)
    test_local['cluster']  = test_local['cluster'].fillna(-1).astype(int)
    preds = np.full(len(test_local), np.nan)
    cluster_models = {}
    for cl in sorted(train_local['cluster'].unique()):
        if cl == -1:
            continue
        mask_tr = train_local['cluster'] == cl
        mask_te = test_local['cluster'] == cl
        if mask_tr.sum() < 50:
            continue
        Xc_tr = Xtr_base[mask_tr.values]
        yc_tr = train_local.loc[mask_tr, 'log_price'].values
        Xc_te = Xte_base[mask_te.values]
        model = LinearRegression()
        model.fit(Xc_tr, yc_tr)
        preds[mask_te.values] = model.predict(Xc_te)
        cluster_models[cl] = model
    nan_mask = np.isnan(preds)
    if nan_mask.any():
        glr = LinearRegression()
        glr.fit(Xtr_base, train_df_local['log_price'].values)
        preds[nan_mask] = glr.predict(Xte_base[nan_mask])
    return preds, town_stats.reset_index(), cluster_models

if run_button:
    st.info("Running selected models — this may take a moment.")
    if model_choice in ("Linear Regression", "All"):
        try:
            lr_model, pred_lr = train_linear(X_train_base.values, y_train, X_test_base.values)
            mae1, rmse1, r2_1 = eval_metrics(y_test, pred_lr)
            results['Linear Regression'] = {'model': lr_model, 'pred': pred_lr, 'mae': mae1, 'rmse': rmse1, 'r2': r2_1}
            st.success(f"Linear regression done — R²: {r2_1:.4f}")
            fn = os.path.join(MODELS_DIR, "linear_reg.joblib")
            joblib.dump(lr_model, fn)
            st.sidebar.success(f"Linear model saved to {fn}")
        except Exception as e:
            st.error(f"Linear regression failed: {e}")

    if model_choice in ("Polynomial Regression", "All"):
        try:
            poly_model, pred_poly = train_polynomial(train_df, test_df, X_train_base, X_test_base, deg=poly_degree)
            mae2, rmse2, r2_2 = eval_metrics(y_test, pred_poly)
            results['Polynomial Regression'] = {'model': poly_model, 'pred': pred_poly, 'mae': mae2, 'rmse': rmse2, 'r2': r2_2}
            st.success(f"Polynomial (deg={poly_degree}) done — R²: {r2_2:.4f}")
            fn = os.path.join(MODELS_DIR, f"poly_reg_deg{poly_degree}.joblib")
            joblib.dump(poly_model, fn)
            st.sidebar.success(f"Polynomial model saved to {fn}")
        except Exception as e:
            st.error(f"Polynomial regression failed: {e}")

    if model_choice in ("KMeans + Per-Cluster Regression", "All"):
        try:
            preds_cluster, town_stats_df, cluster_models = kmeans_per_cluster_regression(train_df, test_df, X_train_base, X_test_base, k=k_clusters)
            mae3, rmse3, r2_3 = eval_metrics(y_test, preds_cluster)
            results['KMeans+PerCluster'] = {'model': cluster_models, 'pred': preds_cluster, 'mae': mae3, 'rmse': rmse3, 'r2': r2_3, 'town_stats': town_stats_df}
            st.success(f"KMeans+PerCluster done — R²: {r2_3:.4f}")
            fn = os.path.join(MODELS_DIR, "kmeans_percluster.joblib")
            joblib.dump({'town_stats': town_stats_df, 'cluster_models': cluster_models}, fn)
            st.sidebar.success(f"KMeans-related artifacts saved to {fn}")
        except Exception as e:
            st.error(f"KMeans per-cluster failed: {e}")

# ---------------------------
# Show summary cards + plots if results exist
# ---------------------------
if results:
    st.markdown("---")
    st.subheader("Model performance summary")
    perf_rows = []
    for name, val in results.items():
        perf_rows.append({'Model': name, 'MAE': val['mae'], 'RMSE': val['rmse'], 'R²': val['r2']})
    perf_df = pd.DataFrame(perf_rows).round(4)
    st.dataframe(perf_df, use_container_width=True)

    st.markdown("### Actual vs Predicted (test set)")
    cols = st.columns(len(results))
    i = 0
    for name, info in results.items():
        fig = px.scatter(x=y_test, y=info['pred'], labels={'x':'Actual log_price','y':'Predicted log_price'}, title=name)
        minv = min(float(np.nanmin(y_test)), float(np.nanmin(info['pred'])))
        maxv = max(float(np.nanmax(y_test)), float(np.nanmax(info['pred'])))
        fig.add_shape(type="line", x0=minv, x1=maxv, y0=minv, y1=maxv, line=dict(color="red", dash="dash"))
        cols[i].plotly_chart(fig, use_container_width=True)
        i += 1

    st.markdown("### Sample predictions (first 20 test rows)")
    sample = test_df.reset_index(drop=True).head(20).copy()
    for name, info in results.items():
        sample[f"pred_{name}"] = info['pred'][:len(sample)]
    st.dataframe(sample, use_container_width=True)

    st.markdown("### Download predictions")
    for name, info in results.items():
        preds_df = test_df.reset_index(drop=True).head(len(info['pred'])).copy()
        preds_df[f"pred_{name}"] = info['pred']
        csv_bytes = df_to_csv_bytes(preds_df)
        btn_label = f"Download predictions — {name}"
        st.download_button(btn_label, data=csv_bytes, file_name=f"predictions_{name.replace(' ','_')}.csv", mime="text/csv")

    st.markdown("### Download saved model artifacts")
    saved_files = os.listdir(MODELS_DIR)
    if saved_files:
        for f in saved_files:
            path = os.path.join(MODELS_DIR, f)
            with open(path, "rb") as fh:
                b = fh.read()
            st.download_button(f"Download {f}", data=b, file_name=f, mime="application/octet-stream")
    else:
        st.write("No saved model artifacts found yet.")

    if 'KMeans+PerCluster' in results:
        ts = results['KMeans+PerCluster'].get('town_stats')
        if ts is not None and not ts.empty:
            fig = px.bar(ts.groupby('cluster')['Town'].count().reset_index().rename(columns={'Town':'count'}), x='cluster', y='count', title='Towns per cluster')
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Extra visualizations and elbow
# ---------------------------
st.markdown("---")
st.subheader("Extra visualizations")
try:
    yearly = df.groupby('year')['log_price'].mean().dropna()
    fig_year = px.line(x=yearly.index, y=np.expm1(yearly.values), labels={'x':'Year', 'y':'Avg Sale Price'}, title='Average Sale Price Over Time (expm1)')
    st.plotly_chart(fig_year, use_container_width=True)
except Exception:
    st.write("Yearly trend not available.")

if st.checkbox("Show KMeans elbow (train towns)"):
    if 'Town' in train_df.columns:
        town_stats_full = train_df.groupby('Town')['log_price'].agg(['mean','std','count']).fillna(0)
        town_stats_full = town_stats_full[town_stats_full['count'] >= 30]
        if len(town_stats_full) >= 2:
            scaler = StandardScaler()
            ts = scaler.fit_transform(town_stats_full[['mean','std']])
            inertias = []
            Ks = list(range(2, min(12, max(3, len(town_stats_full)))))
            for k in Ks:
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                km.fit(ts)
                inertias.append(km.inertia_)
            elbow_fig = go.Figure()
            elbow_fig.add_trace(go.Scatter(x=Ks, y=inertias, mode='lines+markers'))
            elbow_fig.update_layout(title="Elbow method for town clusters", xaxis_title="k", yaxis_title="inertia")
            st.plotly_chart(elbow_fig, use_container_width=True)
        else:
            st.write("Not enough towns with >=30 records for elbow.")
    else:
        st.write("Town column missing; elbow unavailable.")

# ---------------------------
# Live prediction UI
# ---------------------------
st.markdown("---")
st.subheader("Live prediction — try the models interactively")

with st.form("live_pred_form"):
    st.write("Provide input values to get a prediction from a trained model (choose model). If a model isn't trained in this session, you can upload a saved model artifact (joblib).")
    colA, colB = st.columns(2)
    inputs = {}
    for f in features:
        inputs[f] = colA.number_input(f"{f}", value=float(train_df[f].median()) if f in train_df.columns else 0.0)
    town_input = colB.text_input("Town (exact name from dataset)", value=str(train_df['Town'].iloc[0]) if 'Town' in train_df.columns else "")
    model_source = st.selectbox("Which model to use for live predict?", options=["Linear Regression", "Polynomial Regression", "KMeans + Per-Cluster Regression", "Upload model file"])
    uploaded_model_file = None
    if model_source == "Upload model file":
        uploaded_model_file = st.file_uploader("Upload joblib model artifact (optional)", type=["joblib","pkl"])
    submitted = st.form_submit_button("Predict now")

if submitted:
    x_df = pd.DataFrame([inputs])
    if has_town:
        x_df['Town'] = town_input
        if use_target_enc and town_enc_map is not None:
            x_df = prepare_design(x_df, features, town_enc_map if use_target_enc else None, global_town_mean, one_hot=one_hot_mode)
        else:
            x_df = prepare_design(x_df, features, None, None, one_hot=one_hot_mode)
    else:
        x_df = prepare_design(x_df, features, None, None, one_hot=True)
    X_live = x_df.values

    predict_val = None
    model_used = None
    try:
        if model_source == "Upload model file" and uploaded_model_file is not None:
            uploaded_bytes = uploaded_model_file.read()
            model_tmp = joblib.load(io.BytesIO(uploaded_bytes))
            if hasattr(model_tmp, "predict"):
                pred = model_tmp.predict(X_live)
                predict_val = pred[0]
                model_used = "Uploaded model"
        else:
            if model_source in results:
                info = results[model_source]
                mod = info.get('model')
                if model_source == "KMeans + Per-Cluster Regression":
                    town_stats_art = info.get('town_stats')
                    if town_stats_art is None:
                        st.warning("KMeans artifacts not available in memory — run KMeans model first.")
                    else:
                        town_row = town_stats_art[town_stats_art['Town'] == town_input]
                        if not town_row.empty:
                            cl = int(town_row['cluster'].iloc[0])
                            cl_model = info['model'].get(cl)
                            if cl_model is not None:
                                predict_val = cl_model.predict(X_live)[0]
                                model_used = f"KMeans cluster {cl} model"
                            else:
                                gl = LinearRegression().fit(X_train_base.values, y_train)
                                predict_val = gl.predict(X_live)[0]
                                model_used = "KMeans fallback -> global linear"
                        else:
                            gl = LinearRegression().fit(X_train_base.values, y_train)
                            predict_val = gl.predict(X_live)[0]
                            model_used = "KMeans fallback -> global linear"
                else:
                    model_obj = info.get('model')
                    try:
                        predict_val = model_obj.predict(X_live)[0]
                        model_used = model_source
                    except Exception:
                        st.warning("Direct predict failed — ensure the model was trained with the same feature ordering/encoding.")
    except Exception as e:
        st.error(f"Live prediction failed: {e}")

    if predict_val is not None:
        price_pred = np.expm1(predict_val)
        st.success(f"Predicted log_price = {predict_val:.4f} → Predicted Sale Amount ≈ ${price_pred:,.2f} (model: {model_used})")
    else:
        st.info("No prediction produced. Train models or upload an artifact.")

# ---------------------------
# Final tips & next steps
# ---------------------------
st.markdown("---")
st.markdown("### Next steps & extras you can enable")
st.markdown("""
- Add K-fold out-of-fold target encoding to avoid leakage.  
- Wrap polynomial transform into a pipeline and save that pipeline for safe live predictions.  
- Add an HTTP endpoint (FastAPI) to serve saved joblib models.  
- Polish UI with custom assets to match your screenshots.
""")

st.info("Done — diagnostics integrated. If your CSV still fails to produce a price column, open the CSV in a text editor and tell me the exact header names (or paste df.columns output here) and I'll auto-map it for you.")
