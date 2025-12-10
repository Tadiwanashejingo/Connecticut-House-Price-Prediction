# app.py — Connecticut House Price ML Dashboard (fixed diagnostics + full features)
import os
import io
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

# ---------------------------
# Basic config
# ---------------------------
st.set_page_config(page_title="CT House Price — ML Dashboard", layout="wide")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Default Drive ID you provided earlier (change if needed)
DRIVE_FILE_ID_DEFAULT = "1tgbAto2or80v8o6fqKNkWf2rfitaAKIl"
DRIVE_DOWNLOAD_URL = lambda fid: f"https://drive.google.com/uc?export=download&id={fid}"

# ---------------------------
# CSS (simple dark look)
# ---------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#041023 0%, #00101a 100%); color: #e6f7ff; }
    .hero { background: linear-gradient(90deg,#07263b,#041c2b); padding: 20px; border-radius: 10px; }
    .hero h1 { color: #7fe0f5; margin:0; }
    </style>
    """, unsafe_allow_html=True
)

# ---------------------------
# Utilities
# ---------------------------
def safe_read_csv(src):
    return pd.read_csv(src, low_memory=False)

@st.cache_data(show_spinner=False)
def load_from_drive(url: str):
    return safe_read_csv(url)

@st.cache_data(show_spinner=False)
def load_from_filelike(uploaded):
    return safe_read_csv(uploaded)

def smoothed_target_encoding(train_series, target_series, smoothing=10):
    df = pd.DataFrame({"cat": train_series, "y": target_series})
    agg = df.groupby("cat")["y"].agg(["mean", "count"])
    global_mean = target_series.mean()
    agg["smooth"] = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
    return agg["smooth"].to_dict(), global_mean

def apply_target_encoding(series, mapping, global_mean):
    return series.map(mapping).fillna(global_mean)

def eval_metrics(yt, yp):
    mae = mean_absolute_error(yt, yp)
    rmse = mean_squared_error(yt, yp, squared=False)
    r2 = r2_score(yt, yp)
    return mae, rmse, r2

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# Sidebar: data & model options
# ---------------------------
st.sidebar.title("Data & Options")
st.sidebar.markdown("Load CSV from Google Drive (file id) or upload a file as fallback.")

drive_id = st.sidebar.text_input("Google Drive FILE_ID", value=DRIVE_FILE_ID_DEFAULT)
attempt_drive = st.sidebar.checkbox("Try Drive first", value=True)
uploaded = st.sidebar.file_uploader("Upload CSV (fallback)", type=["csv"])

st.sidebar.markdown("---")
use_target_enc = st.sidebar.checkbox("Use target encoding (Town)", value=False)
target_enc_smoothing = st.sidebar.number_input("Target-enc smoothing", min_value=1, value=10, step=1)
k_clusters = st.sidebar.slider("KMeans clusters (town-level)", 2, 8, 5)
random_state = int(st.sidebar.number_input("Random seed", value=42))

# ---------------------------
# Load dataset (Drive -> uploader)
# ---------------------------
df = None
if attempt_drive and drive_id.strip():
    try:
        with st.spinner("Loading from Google Drive..."):
            df = load_from_drive(DRIVE_DOWNLOAD_URL(drive_id.strip()))
            st.sidebar.success("Loaded dataset from Drive")
    except Exception as e:
        st.sidebar.error(f"Drive load failed: {e}")
        df = None

if df is None and uploaded is not None:
    try:
        df = load_from_filelike(uploaded)
        st.sidebar.success("Loaded uploaded CSV")
    except Exception as e:
        st.sidebar.error(f"Upload failed: {e}")
        df = None

if df is None:
    st.sidebar.info("No dataset loaded. Provide Drive ID or upload CSV.")
    st.stop()

# ---------------------------
# Diagnostics & robust price detection
# ---------------------------
st.title("Dataset & quick checks")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))
with st.expander("Preview (first 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# If no rows: give simple guidance (use triple-quoted string here to avoid unterminated issues)
if df.shape[0] == 0:
    st.error(
        """DataFrame has 0 rows. Common causes:
- Drive file not shared publicly (set to 'Anyone with the link').
- Wrong Drive ID or not a CSV.
- CSV uses a different delimiter (e.g., ';').
- First rows are header/notes (need header row index).
""")
    st.info("If the file is empty when loaded from Drive, please verify the Drive link or upload the CSV manually.")
    st.stop()

# Normalize common price column names
if "SalePrice" in df.columns and "Sale Amount" not in df.columns:
    df["Sale Amount"] = df["SalePrice"]

# Try detect price column automatically
common_price_names = [
    "Sale Amount", "Sale_Amount", "SaleAmount", "SalePrice", "Sale Price", "price", "Price", "SALE AMOUNT"
]
found_exact = [c for c in df.columns if c.strip().lower().replace("_"," ") in [n.lower().replace("_"," ") for n in common_price_names]]
found_keyword = [c for c in df.columns if any(k in c.lower() for k in ["price", "amount", "sale"])]

candidates = found_exact or found_keyword
price_col = None

if candidates:
    st.success(f"Detected price-like columns: {candidates}")
    price_col = st.selectbox("Select a column to use as 'Sale Amount'", options=candidates, index=0)
    if st.button("Use selected column as price"):
        df["Sale Amount"] = pd.to_numeric(df[price_col].astype(str).str.replace("[^0-9.-]", "", regex=True), errors="coerce")
        df["log_price"] = np.log1p(df["Sale Amount"])
        if df["Sale Amount"].isnull().all():
            st.error("Conversion failed (all NaN). Check the chosen column contains numeric values.")
        else:
            st.success(f"Created 'Sale Amount' and 'log_price' from '{price_col}'.")
else:
    st.warning("No price-like column detected automatically.")
    pick = st.selectbox("Pick any column to convert to price (or re-upload corrected CSV)", options=list(df.columns) + ["None"])
    if pick != "None" and st.button(f"Use '{pick}' as price column"):
        df["Sale Amount"] = pd.to_numeric(df[pick].astype(str).str.replace("[^0-9.-]", "", regex=True), errors="coerce")
        df["log_price"] = np.log1p(df["Sale Amount"])
        if df["Sale Amount"].isnull().all():
            st.error("Conversion produced only NaN values — column likely non-numeric.")
        else:
            st.success(f"Created 'Sale Amount' and 'log_price' from '{pick}'.")

# Final check for log_price
if "log_price" not in df.columns or df["log_price"].isnull().all():
    st.error("No usable price column found. Please upload a CSV that contains a numeric price column (e.g., 'Sale Amount').")
    st.stop()

# ---------------------------
# Minimal cleaning & engineerd features
# ---------------------------
if "Date Recorded" in df.columns:
    df["Date Recorded"] = pd.to_datetime(df["Date Recorded"], errors="coerce")
    df = df.dropna(subset=["Date Recorded"])
    df["year"] = df["Date Recorded"].dt.year
    df["month"] = df["Date Recorded"].dt.month
else:
    if "year" not in df.columns:
        df["year"] = 2000
    if "month" not in df.columns:
        df["month"] = 1

# apply some safe filters if columns exist
if "Sales Ratio" in df.columns:
    df = df[df["Sales Ratio"].between(0.05, 5.0)]
if "Non Use Code" in df.columns:
    df = df[df["Non Use Code"].isna()]
if "Sale Amount" in df.columns:
    df = df[df["Sale Amount"] >= 2000]
if "Property Type" in df.columns:
    df = df[df["Property Type"].isin(["Residential", "Single Family", "Condo", "Two Family", "Three Family"])]

st.write(f"After lightweight cleaning: {df.shape[0]:,} rows")

# ---------------------------
# Feature selection & temporal split
# ---------------------------
st.markdown("---")
st.subheader("Feature selection")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_features = [c for c in ["Assessed Value", "year", "month"] if c in numeric_cols]
features = st.multiselect("Numeric features to include", options=numeric_cols, default=default_features)
if "log_price" in features:
    features = [f for f in features if f != "log_price"]

if not features:
    st.error("Please select at least one numeric feature.")
    st.stop()

has_town = "Town" in df.columns
if has_town:
    st.write("Town column present — can be OHE or target-encoded")

min_year = int(df["year"].min())
max_year = int(df["year"].max())
train_until = st.slider("Train years ≤", min_value=min_year, max_value=max_year-1, value=min(2020, max_year-1))
test_from = st.slider("Test years ≥", min_value=train_until+1, max_value=max_year, value=max(train_until+1, min_year+1))

train_df = df[df["year"] <= train_until].copy()
test_df = df[df["year"] >= test_from].copy()
st.write(f"Train rows: {train_df.shape[0]:,} — Test rows: {test_df.shape[0]:,}")

# ---------------------------
# Encoding & design matrices
# ---------------------------
town_enc_map, global_town_mean = None, None
if use_target_enc and has_town:
    town_enc_map, global_town_mean = smoothed_target_encoding(train_df["Town"], train_df["log_price"], smoothing=target_enc_smoothing)
    st.sidebar.success("Computed smoothed target encoding from train set")

def prepare_design(df_local, features_local, use_te=False):
    X = df_local[features_local].reset_index(drop=True)
    if has_town:
        if use_te:
            X["town_te"] = apply_target_encoding(df_local["Town"], town_enc_map, global_town_mean)
        else:
            dummies = pd.get_dummies(df_local["Town"], prefix="town", drop_first=True).reset_index(drop=True)
            X = pd.concat([X, dummies], axis=1)
    X = X.fillna(0)
    return X

X_train = prepare_design(train_df, features, use_te=use_target_enc)
X_test = prepare_design(test_df, features, use_te=use_target_enc)
y_train = train_df["log_price"].values
y_test = test_df["log_price"].values

# ---------------------------
# Modeling UI & training
# ---------------------------
st.markdown("---")
st.subheader("Models")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    model_choice = st.selectbox("Model to run", options=["All", "Linear Regression", "Polynomial Regression", "KMeans + Per-Cluster Regression"])
with col2:
    poly_degree = st.number_input("Polynomial degree (year)", min_value=2, max_value=5, value=3)
with col3:
    run_models = st.button("Train / Run")

results = {}

def train_linear(Xtr, ytr, Xte):
    m = LinearRegression()
    m.fit(Xtr, ytr)
    return m, m.predict(Xte)

def train_poly(train_df_local, test_df_local, Xtr_base, Xte_base, deg=3):
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    Ytr = poly.fit_transform(train_df_local[["year"]])
    Yte = poly.transform(test_df_local[["year"]])
    Xtr = np.hstack([Xtr_base.values, Ytr])
    Xte = np.hstack([Xte_base.values, Yte])
    m = LinearRegression()
    m.fit(Xtr, train_df_local["log_price"].values)
    return m, m.predict(Xte)

def kmeans_per_cluster(train_df_local, test_df_local, Xtr_base, Xte_base, k=5):
    if "Town" not in train_df_local.columns:
        raise ValueError("Town column required for KMeans approach.")
    town_stats = train_df_local.groupby("Town")["log_price"].agg(["mean","std","count"]).fillna(0)
    town_stats = town_stats[town_stats["count"] >= 30]
    if town_stats.shape[0] < 2:
        raise ValueError("Not enough towns with >=30 records for clustering.")
    scaler = StandardScaler()
    ts = scaler.fit_transform(town_stats[["mean","std"]])
    k_use = min(k, max(2, town_stats.shape[0]))
    kmeans = KMeans(n_clusters=k_use, random_state=random_state, n_init=10)
    town_stats["cluster"] = kmeans.fit_predict(ts)
    train_local = train_df_local.merge(town_stats[["cluster"]], left_on="Town", right_index=True, how="left")
    test_local = test_df_local.merge(town_stats[["cluster"]], left_on="Town", right_index=True, how="left")
    train_local["cluster"] = train_local["cluster"].fillna(-1).astype(int)
    test_local["cluster"] = test_local["cluster"].fillna(-1).astype(int)
    preds = np.full(len(test_local), np.nan)
    cluster_models = {}
    for cl in sorted(train_local["cluster"].unique()):
        if cl == -1:
            continue
        mask_tr = train_local["cluster"] == cl
        mask_te = test_local["cluster"] == cl
        if mask_tr.sum() < 50:
            continue
        Xc_tr = Xtr_base[mask_tr.values]
        yc_tr = train_local.loc[mask_tr, "log_price"].values
        Xc_te = Xte_base[mask_te.values]
        m = LinearRegression()
        m.fit(Xc_tr, yc_tr)
        preds[mask_te.values] = m.predict(Xc_te)
        cluster_models[cl] = m
    nan_mask = np.isnan(preds)
    if nan_mask.any():
        gl = LinearRegression()
        gl.fit(Xtr_base, train_df_local["log_price"].values)
        preds[nan_mask] = gl.predict(Xte_base[nan_mask])
    return preds, town_stats.reset_index(), cluster_models

if run_models:
    st.info("Running...")

    if model_choice in ("Linear Regression", "All"):
        try:
            model_lr, pred_lr = train_linear(X_train.values, y_train, X_test.values)
            mae1, rmse1, r21 = eval_metrics(y_test, pred_lr)
            results["Linear Regression"] = {"model": model_lr, "pred": pred_lr, "mae": mae1, "rmse": rmse1, "r2": r21}
            joblib.dump(model_lr, os.path.join(MODELS_DIR, "linear_reg.joblib"))
            st.success(f"Linear regression done — R²: {r21:.4f}")
        except Exception as e:
            st.error(f"Linear failed: {e}")

    if model_choice in ("Polynomial Regression", "All"):
        try:
            model_poly, pred_poly = train_poly(train_df, test_df, X_train, X_test, deg=poly_degree)
            mae2, rmse2, r22 = eval_metrics(y_test, pred_poly)
            results["Polynomial Regression"] = {"model": model_poly, "pred": pred_poly, "mae": mae2, "rmse": rmse2, "r2": r22}
            joblib.dump(model_poly, os.path.join(MODELS_DIR, f"poly_deg{poly_degree}.joblib"))
            st.success(f"Polynomial regression done — R²: {r22:.4f}")
        except Exception as e:
            st.error(f"Polynomial failed: {e}")

    if model_choice in ("KMeans + Per-Cluster Regression", "All"):
        try:
            pred_cluster, town_stats, cluster_models = kmeans_per_cluster(train_df, test_df, X_train, X_test, k=k_clusters)
            mae3, rmse3, r23 = eval_metrics(y_test, pred_cluster)
            results["KMeans+PerCluster"] = {"model": cluster_models, "pred": pred_cluster, "mae": mae3, "rmse": rmse3, "r2": r23, "town_stats": town_stats}
            joblib.dump({"town_stats": town_stats, "cluster_models": cluster_models}, os.path.join(MODELS_DIR, "kmeans_artifacts.joblib"))
            st.success(f"KMeans+PerCluster done — R²: {r23:.4f}")
        except Exception as e:
            st.error(f"KMeans failed: {e}")

# ---------------------------
# Show results if present
# ---------------------------
if results:
    st.markdown("---")
    st.subheader("Model performance summary")
    rows = []
    for name, v in results.items():
        rows.append({"Model": name, "MAE": v["mae"], "RMSE": v["rmse"], "R2": v["r2"]})
    st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)

    st.markdown("### Actual vs Predicted (test set)")
    cols = st.columns(len(results))
    for i, (name, v) in enumerate(results.items()):
        fig = px.scatter(x=y_test, y=v["pred"], labels={"x": "Actual log_price", "y": "Predicted log_price"}, title=name)
        mn = float(min(np.nanmin(y_test), np.nanmin(v["pred"])))
        mx = float(max(np.nanmax(y_test), np.nanmax(v["pred"])))
        fig.add_shape(type="line", x0=mn, x1=mx, y0=mn, y1=mx, line=dict(color="red", dash="dash"))
        cols[i].plotly_chart(fig, use_container_width=True)

    st.markdown("### Download predictions & artifacts")
    for name, v in results.items():
        preds_df = test_df.reset_index(drop=True).head(len(v["pred"])).copy()
        preds_df[f"pred_{name.replace(' ','_')}"] = v["pred"]
        st.download_button(f"Download predictions ({name})", data=df_to_csv_bytes(preds_df), file_name=f"predictions_{name.replace(' ','_')}.csv", mime="text/csv")
    # saved artifacts
    saved = os.listdir(MODELS_DIR)
    for f in saved:
        path = os.path.join(MODELS_DIR, f)
        with open(path, "rb") as fh:
            st.download_button(f"Download {f}", data=fh.read(), file_name=f, mime="application/octet-stream")

# ---------------------------
# Extra visualizations
# ---------------------------
st.markdown("---")
st.subheader("Extra visuals")
try:
    yearly = df.groupby("year")["log_price"].mean().dropna()
    fig_year = px.line(x=yearly.index, y=np.expm1(yearly.values), labels={"x": "Year", "y": "Avg Sale Price"}, title="Average Sale Price Over Time")
    st.plotly_chart(fig_year, use_container_width=True)
except Exception:
    st.write("Yearly trend unavailable.")

if st.checkbox("Show KMeans elbow (train towns)"):
    if "Town" in train_df.columns:
        town_stats_full = train_df.groupby("Town")["log_price"].agg(["mean","std","count"]).fillna(0)
        town_stats_full = town_stats_full[town_stats_full["count"] >= 30]
        if len(town_stats_full) >= 2:
            scaler = StandardScaler()
            ts = scaler.fit_transform(town_stats_full[["mean","std"]])
            inertias = []
            Ks = list(range(2, min(12, max(3, len(town_stats_full)))))
            for k in Ks:
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                km.fit(ts)
                inertias.append(km.inertia_)
            elbow_fig = go.Figure()
            elbow_fig.add_trace(go.Scatter(x=Ks, y=inertias, mode="lines+markers"))
            elbow_fig.update_layout(title="Elbow method for town clusters", xaxis_title="k", yaxis_title="inertia")
            st.plotly_chart(elbow_fig, use_container_width=True)
        else:
            st.write("Not enough towns with >=30 rows for elbow.")
    else:
        st.write("Town column missing; elbow unavailable.")

# ---------------------------
# Live prediction UI
# ---------------------------
st.markdown("---")
st.subheader("Live prediction (use a trained model or upload a joblib artifact)")

with st.form("live"):
    st.write("Enter numeric inputs for features below.")
    input_vals = {}
    cols = st.columns(2)
    for f in features:
        input_vals[f] = cols[0].number_input(f, value=float(train_df[f].median()) if f in train_df.columns else 0.0)
    town_input = cols[1].text_input("Town (exact)", value=str(train_df["Town"].iloc[0]) if "Town" in train_df.columns else "")
    model_choice_live = st.selectbox("Model to use", options=["Linear Regression", "Polynomial Regression", "KMeans + Per-Cluster Regression", "Upload model (joblib)"])
    upload_model = None
    if model_choice_live == "Upload model (joblib)":
        upload_model = st.file_uploader("Upload joblib model", type=["joblib", "pkl"])
    submitted = st.form_submit_button("Predict")

if submitted:
    X_live_df = pd.DataFrame([input_vals])
    if has_town:
        X_live_df["Town"] = town_input
        X_live = prepare_design = None  # placeholder to avoid linter error
        # reuse prepare logic from earlier by replicating small part:
        if use_target_enc and town_enc_map is not None:
            X_live_df = X_live_df.copy()
            X_live_df = X_live_df[features]
            X_live_df["town_te"] = apply_target_encoding(pd.Series([town_input]), town_enc_map, global_town_mean)
            X_live = X_live_df.fillna(0).values
        else:
            # one-hot: create same dummies as train (align by columns)
            X0 = X_live_df[features].copy()
            d = pd.get_dummies(pd.Series([town_input]), prefix="town", drop_first=True)
            # align to test/train columns
            train_dummy_cols = [c for c in X_train.columns if c.startswith("town_")]
            for col in train_dummy_cols:
                X0[col] = 1 if col in d.columns else 0
            X_live = X0.fillna(0).values
    else:
        X_live = X_live_df[features].fillna(0).values

    pred_val = None
    model_used = None
    try:
        if model_choice_live == "Upload model (joblib)" and upload_model is not None:
            mod = joblib.load(upload_model)
            pred_val = mod.predict(X_live)[0]
            model_used = "uploaded model"
        else:
            if model_choice_live in results:
                info = results[model_choice_live]
                mod = info.get("model")
                # For KMeans we stored cluster models dict
                if model_choice_live == "KMeans + Per-Cluster Regression":
                    ts = info.get("town_stats")
                    if ts is None:
                        st.warning("KMeans artifacts not available in memory — run KMeans first.")
                    else:
                        row = ts[ts["Town"] == town_input]
                        if not row.empty:
                            cl = int(row["cluster"].iloc[0])
                            cl_model = info["model"].get(cl)
                            if cl_model is not None:
                                pred_val = cl_model.predict(X_live)[0]
                                model_used = f"KMeans cluster {cl}"
                            else:
                                gl = LinearRegression().fit(X_train.values, y_train)
                                pred_val = gl.predict(X_live)[0]
                                model_used = "KMeans fallback -> global linear"
                        else:
                            gl = LinearRegression().fit(X_train.values, y_train)
                            pred_val = gl.predict(X_live)[0]
                            model_used = "KMeans fallback -> global linear"
                else:
                    pred_val = info["model"].predict(X_live)[0]
                    model_used = model_choice_live
            else:
                st.warning("Selected model not trained in this session. Upload a model file or train first.")
    except Exception as e:
        st.error(f"Live prediction error: {e}")

    if pred_val is not None:
        sale_pred = np.expm1(pred_val)
        st.success(f"Predicted log_price = {pred_val:.4f} → Predicted Sale Amount ≈ ${sale_pred:,.2f} (model: {model_used})")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.info("If your dataset still doesn't provide a price column, open it in a text editor and tell me the exact header names shown in 'Columns:' above. I can then add an auto-mapping line to convert it to 'Sale Amount'.")
