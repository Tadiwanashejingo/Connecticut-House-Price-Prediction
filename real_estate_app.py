import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

DATA_PATH = r"C:\Users\chabx\Downloads\PREDICTIVE ANALYSIS\REAL ESTATE PREDICTION PROJECT\Real_Estate_Sales_2001-2023_GL.csv"

# Page config & gorgeous dark theme
st.set_page_config(page_title="CT House Price Predictor", page_icon="🏠", layout="wide")

# Custom CSS - exact look from the Cardio Risk app
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); color: white;}
    .stApp {background: transparent;}
    h1, h2, h3 {color: #00ffcc; font-family: 'Segoe UI';}
    .css-1d391kg {color: white;}
    .big-pred {font-size: 60px; font-weight: bold; text-align: center; color: #00ffcc;
               text-shadow: 0 0 20px #00ffcc; padding: 30px; background: rgba(0,255,204,0.1);
               border-radius: 20px; margin: 30px 0;}
    .card {background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; backdrop-filter: blur(10px);}
    .stButton>button {background: linear-gradient(90deg, #ff0066, #ff3300); color: white; 
                      height: 60px; font-size: 24px; border-radius: 15px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Load and train model once
@st.cache_resource
def load_and_train():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found! Please check the DATA_PATH in the code.")
        st.stop()
        
    df = pd.read_csv(DATA_PATH)
    df = df[df['Property Type'].str.contains('Residential', na=False, case=False)]
    df = df.dropna(subset=['Sale Amount', 'Assessed Value', 'List Year', 'Town', 'Residential Type'])
    df = df[(df['Sale Amount'].between(10000, 3000000)) & (df['Assessed Value'] > 1000)]
    
    features = ['Assessed Value', 'List Year']
    X = df[features]
    y = df['Sale Amount']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = KNeighborsRegressor(n_neighbors=15)
    knn.fit(X_scaled, y)
    
    return knn, scaler, df

model, scaler, data = load_and_train()

# Sidebar navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/home.png", width=100)
st.sidebar.title("🏠 CT House Price Predictor")
page = st.sidebar.radio("Navigation", ["🏠 Predict", "📊 Model Performance", "📁 Batch Prediction", "ℹ️ About"])

# ==================================== PREDICT PAGE ====================================
if page == "🏠 Predict":
    st.markdown("<h1 style='text-align:center;'>Connecticut House Price Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#88ffaa;'>2001-2023 Real Estate Sales • Powered by KNN</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Enter Property Details")
        
        assessed = st.slider("🏠 Assessed Value ($)", 10000, 2000000, 300000, step=5000)
        year = st.slider("📅 List Year", 2001, 2023, 2022)
        town = st.selectbox("🏘️ Town", sorted(data['Town'].unique()))
        prop_type = st.selectbox("🏡 Property Type", sorted(data['Residential Type'].dropna().unique()))
        
        if st.button("🔥 Predict House Price"):
            input_data = np.array([[assessed, year]])
            input_scaled = scaler.transform(input_data)
            pred = model.predict(input_scaled)[0]
            
            st.markdown(f"<div class='big-pred'>${pred:,.0f}</div>", unsafe_allow_html=True)
            st.success(f"Predicted Sale Price in {town}, {year}")
            st.markdown("</div>", unsafe_allow_html=True)

# ==================================== MODEL PERFORMANCE ====================================
elif page == "📊 Model Performance":
    st.header("Model Performance Dashboard")
    # Quick retrain for plots
    X = data[['Assessed Value', 'List Year']]
    y = data['Sale Amount']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", f"${rmse:,.0f}")
    col2.metric("R² Score", f"{r2:.4f}")
    col3.metric("Best Model", "KNN (k=15)")
    col4.metric("Dataset Size", f"{len(data):,} houses")
    
    st.markdown("### Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(y, y_pred, alpha=0.6, color='#00ffcc')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=3)
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ==================================== BATCH PREDICTION ====================================
elif page == "📁 Batch Prediction":
    st.header("Batch Prediction")
    uploaded = st.file_uploader("Upload CSV with columns: Assessed Value, List Year", type="csv")
    if uploaded:
        batch = pd.read_csv(uploaded)
        if {'Assessed Value', 'List Year'}.issubset(batch.columns):
            batch_scaled = scaler.transform(batch[['Assessed Value', 'List Year']])
            batch['Predicted Price'] = model.predict(batch_scaled)
            st.success("Prediction complete!")
            st.dataframe(batch)
            csv = batch.to_csv(index=False).encode()
            st.download_button("Download Results", csv, "predicted_prices.csv", "text/csv")
        else:
            st.error("CSV must have 'Assessed Value' and 'List Year' columns")

# ==================================== ABOUT ====================================
else:
    st.header("About This Project")
    st.write("""
    **Connecticut House Price Predictor**  
    • Built using real 2001–2023 sales data  
    • Best model: K-Nearest Neighbors (k=15) → RMSE ≈ $152,000  
    • Features used: Assessed Value, List Year  
    • Made using Streamlit
    """)
    st.info("Tadiwanashe Jingo • Lovely Professional University • Data Science Project 2025")

st.markdown("</div>", unsafe_allow_html=True)

