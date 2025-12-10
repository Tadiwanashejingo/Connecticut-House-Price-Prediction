# app.py — Streamlit app that loads CSV from Google Drive (no upload required)
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

st.set_page_config(layout="wide", page_title="Real Estate Sales — Drive Load")

# ---------------------------
# CONFIG: Put your Google Drive file id here
# ---------------------------
# Your Drive file id (from the link you provided)
DRIVE_FILE_ID = "1tgbAto2or80v8o6fqKNkWf2rfitaAKIl"
DRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

# ---------------------------
# DATA LOADING
# ---------------------------
@st.cache_data(show_spinner=False)
def load_from_drive(url: str) -> pd.DataFrame:
    """Load CSV directly from Google Drive url. Returns a pandas DataFrame."""
    try:
        df = pd.read_csv(url, low_memory=False)
        return df
    except Exception as e:
        # bubble up exception to app
        raise RuntimeError(f"Failed to load from Google Drive: {e}")

@st.cache_data(show_spinner=False)
def load_from_filelike(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, low_memory=False)

# ---------------------------
# HEADER / INFO
# ---------------------------
st.markdown("<h1 style='text-align:center;color:#7fe0f5'>1.2M+ Real Sales • Random Forest + K-Means • Live Predictions</h1>", unsafe_allow_html=True)
st.write("")
st.write("This app will attempt to load your dataset directly from Google Drive (no upload required).")

# ---------------------------
# Try to load directly from Drive
# ---------------------------
data_load_success = False
df = None
with st.spinner("Loading dataset from Google Drive..."):
    try:
        df = load_from_drive(DRIVE_DOWNLOAD_URL)
        data_load_success = True
        st.success("Dataset loaded from Google Drive successfully!")
    except Exception as e:
        st.error(f"Could not load dataset from Google Drive: {e}")
        st.info("You can optionally upload the file manually using the uploader below (fallback).")

# ---------------------------
# Optional fallback uploader
# ---------------------------
st.markdown("### Upload fallback (optional)")
uploaded_file = st.file_uploader("If Drive fails you can upload the CSV here", type=["csv"])
if uploaded_file is not None and not data_load_success:
    try:
        df = load_from_filelike(uploaded_file)
        data_load_success = True
        st.success("Dataset loaded from uploaded file.")
    except Exception as e:
        st.error(f"Failed to read the uploaded CSV: {e}")

if not data_load_success:
    st.stop()  # nothing more we can do

# ---------------------------
# QUICK EDA
# ---------------------------
st.markdown("## Quick preview & checks")
st.write("Shape:", df.shape)
st.dataframe(df.head(10))

# show missing value summary
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if not missing.empty:
    st.markdown("**Missing values (columns with any missing entries):**")
    st.write(missing)
else:
    st.markdown("No missing values detected (based on simple check).")

# show dtypes
st.markdown("**Column types:**")
st.write(df.dtypes)

# ---------------------------
# SIMPLE FEATURE SELECTOR FOR MODELING
# ---------------------------
st.markdown("---")
st.markdown("## Quick model & clustering demo")

# autopick numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.warning("No numeric columns found for modeling. Please provide numeric columns.")
    st.stop()

st.markdown("### Select features and target")
with st.container():
    col1, col2 = st.columns([2,1])
    with col1:
        features = st.multiselect("Features (numeric)", numeric_cols, default=numeric_cols[:5])
    with col2:
        target = st.selectbox("Target (numeric)", numeric_cols, index=0)

if target in features:
    st.warning("Target should not be among features — it will be removed automatically from features.")
    features = [f for f in features if f != target]

if len(features) < 1:
    st.error("Select at least one feature.")
    st.stop()

# drop rows with missing values in used cols
model_df = df[features + [target]].dropna()
st.write("After dropping rows with NA in chosen columns:", model_df.shape)

# ---------------------------
# SPLIT and TRAIN controls
# ---------------------------
st.sidebar.markdown("## Model controls")
test_size = st.sidebar.slider("Test set size (%)", 5, 50, 20)
n_estimators = st.sidebar.slider("RandomForest n_estimators", 10, 200, 50)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

train_now = st.sidebar.button("Train RandomForest")

if train_now:
    X = model_df[features].values
    y = model_df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))
    st.info(f"Training RandomForestRegressor on {X_train.shape[0]} rows...")
    rf = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=-1)
    rf.fit(X_train, y_train)
    score = rf.score(X_test, y_test)
    st.success(f"Model trained. R² on test set: {score:.4f}")
    # show feature importances
    try:
        importances = rf.feature_importances_
        fi = pd.Series(importances, index=features).sort_values(ascending=False)
        st.write("Feature importances:")
        st.bar_chart(fi)
    except Exception:
        pass

# ---------------------------
# K-MEANS CLUSTERING demo
# ---------------------------
st.markdown("---")
st.markdown("## K-Means clustering demo (choose numeric columns)")

k_cols = st.multiselect("Columns to cluster on (numeric)", numeric_cols, default=numeric_cols[:3])
n_clusters = st.slider("Number of clusters", 2, 12, 4)
run_kmeans = st.button("Run KMeans")

if run_kmeans:
    if len(k_cols) < 1:
        st.error("Select at least one numeric column for clustering.")
    else:
        km_df = df[k_cols].dropna()
        kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_state))
        labels = kmeans.fit_predict(km_df.values)
        km_df_ = km_df.copy()
        km_df_["cluster"] = labels.astype(int)
        st.write("Cluster counts:")
        st.write(km_df_["cluster"].value_counts().sort_index())
        # if at least 2 columns, show scatter
        if len(k_cols) >= 2:
            fig = px.scatter(km_df_.reset_index(), x=k_cols[0], y=k_cols[1], color=km_df_["cluster"].astype(str),
                             title="KMeans clusters (colored)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Not enough columns for scatter plot. See cluster counts above.")

# ---------------------------
# Footer / tips
# ---------------------------
st.markdown("---")
st.info("This app loads data directly from your Google Drive file. If you want to change file, update DRIVE_FILE_ID at top of this script.")
st.write("If you'd like, I can further adapt this to match your original layout (charts, performance table, advanced preprocessing).")

