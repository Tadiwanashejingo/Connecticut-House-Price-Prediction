import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 1. LOAD & CLEAN ====================
df = pd.read_csv(r"C:/Users/chabx/Downloads/PREDICTIVE ANALYSIS/REAL ESTATE PREDICTION PROJECT/Real_Estate_Sales_2001-2023_GL.csv")

df = df[df['Property Type'].str.contains('Residential', na=False, case=False)]
df = df.dropna(subset=['Sale Amount', 'Assessed Value', 'List Year'])
df = df[(df['Sale Amount'].between(10000, 3000000)) & (df['Assessed Value'] > 1000)].copy()

print(f"Clean rows: {len(df):,}")

# ==================== 2. FEATURES ====================
X = df[['Assessed Value', 'List Year']]
y = df['Sale Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for KNN
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ==================== 3. TRAIN MODELS ====================
predictions = {}

# Linear
lr = LinearRegression()
lr.fit(X_train, y_train)
predictions['Linear Regression'] = lr.predict(X_test)

# Polynomial degree 2
poly_feat = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_feat.fit_transform(X_train)
X_test_poly = poly_feat.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
predictions['Polynomial (deg=2)'] = poly_model.predict(X_test_poly)

# KNN – best k
best_k = 7
best_score = np.inf
for k in [3,5,7,9,11,13,15]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_sc, y_train)
    score = mean_squared_error(y_test, knn.predict(X_test_sc))
    if score < best_score:
        best_score = score
        best_k = k

knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train_sc, y_train)
predictions[f'KNN (k={best_k})'] = knn_best.predict(X_test_sc)

# ==================== 4. SCATTER PLOTS + METRICS ====================
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
results = []

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (name, pred) in enumerate(predictions.items()):
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    results.append([name, mse, rmse, r2])
    
    ax = axes[idx]
    ax.scatter(y_test, pred, alpha=0.6, color=colors[idx], s=20)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.set_title(f'{name}\nRMSE = ${rmse:,.2f} │ R² = {r2:.4f}', fontsize=13)
    ax.grid(alpha=0.3)

plt.suptitle('Actual vs Predicted House Prices', fontsize=18, y=1.02)
plt.tight_layout()
plt.show()

# ==================== 5. BAR CHART ====================
results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'RMSE', 'R2'])

plt.figure(figsize=(9,6))
bars = plt.bar(results_df['Model'], results_df['RMSE'], color=colors, edgecolor='black')
plt.title('Model Comparison – Root Mean Squared Error', fontsize=16, pad=20)
plt.ylabel('RMSE (USD)')
plt.xticks(rotation=10)

for bar, rmse in zip(bars, results_df['RMSE']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
             f'${rmse:,.2f}', ha='center', fontweight='bold', fontsize=12)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 6. PRINT TABLE ====================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(results_df.to_string(index=False, float_format="{:,.4f}".format))

best = results_df.loc[results_df['RMSE'].idxmin()]
print(f"\nWINNER → {best['Model']} with RMSE = ${best['RMSE']:,.2f}")
