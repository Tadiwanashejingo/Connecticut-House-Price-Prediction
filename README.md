Predicting residential property prices in Connecticut (2001–2023) using only allowed classic ML models.

Features:
• Multiple Linear Regression  
• Polynomial Regression (degree 2)  
• K-Means Clustering + Per-Cluster Linear Regression (Best performing model)

Key findings:
- Simple linear regression → good baseline
- Polynomial features capture non-linear trends (2008 crash, COVID boom)
- Clustering towns into 5 market types gives the highest accuracy (R² ≈ 0.92–0.94)

Dataset: Official Connecticut OPM Real Estate Sales 2001–2023 GL (~1.2M rows)  
Target: log(Sale Amount)  
Best model: K-Means + Per-Cluster Linear Regression

Includes:
- Full Jupyter/Spyder-ready Python script
- Auto-generated beautiful HTML report with plots and results
- Cleaned dataset preprocessing steps

Perfect final project for Machine Learning / Data Science courses using only basic allowed models.
