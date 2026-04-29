# SalesForecastAI

A production-ready full-stack sales forecasting and customer analytics system built with Python, Flask, Pandas, Scikit-learn, Bootstrap, and SQLite.

## 📌 Project Overview
This project converts historical customer purchase data from `Customer-Purchase-History.xlsx` into a clean CSV dataset and SQLite database, performs data engineering and exploratory analytics, trains regression and classification models for sales prediction, and exposes a modern analytics dashboard with REST API endpoints.

## ✅ Tech Stack
- Python 3
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- SQLite
- HTML / CSS / JavaScript
- Bootstrap
- Chart.js
- Gunicorn

## 📁 Repository Structure
```
project/
├── app.py
├── preprocess.py
├── train_models.py
├── requirements.txt
├── Procfile
├── README.md
├── Customer-Purchase-History.xlsx
├── data/
│   ├── customer_purchase_history.csv
│   ├── sales_data.db
│   ├── analytics_summary.json
│   ├── monthly_trend.json
│   ├── top_products.json
│   ├── category_revenue.json
│   ├── payment_methods.json
│   ├── rfm_segments.json
│   ├── churn_risk.json
│   ├── sales_forecast.csv
│   ├── sales_forecast.json
│   └── rfm_analysis.csv
├── model/
│   ├── sales_prediction_model.pkl
│   ├── classification_model.pkl
│   ├── le_product.pkl
│   ├── le_category.pkl
│   ├── le_payment.pkl
│   └── feature_names.pkl
├── static/
│   ├── style.css
│   ├── script.js
│   └── charts/
│       ├── monthly_sales_trend.png
│       ├── top_products.png
│       ├── category_revenue_pie.png
│       ├── payment_methods.png
│       ├── review_distribution.png
│       ├── revenue_heatmap.png
│       ├── top_customers.png
│       ├── weekday_sales.png
│       ├── quarterly_revenue.png
│       ├── feature_importance.png
│       ├── actual_vs_predicted.png
│       ├── model_comparison_regression.png
│       ├── model_comparison_classification.png
│       ├── residuals.png
│       ├── rfm_segments.png
│       ├── churn_risk.png
│       ├── kmeans_elbow.png
│       ├── kmeans_clusters.png
│       ├── sales_forecast.png
│       └── peak_sales.png
└── templates/
    ├── index.html
    ├── dashboard.html
    └── prediction.html
```

## 🚀 Setup Instructions
1. Create a virtual environment and activate it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run preprocessing to convert Excel to CSV and generate analytics assets:
   ```bash
   python preprocess.py
   ```
4. Train the machine learning models:
   ```bash
   python train_models.py
   ```
5. Start the Flask app locally:
   ```bash
   python app.py
   ```
6. Open your browser at `http://127.0.0.1:5000`

## 🔧 Available Scripts
- `python preprocess.py`
  - Converts `Customer-Purchase-History.xlsx` to `data/customer_purchase_history.csv`
  - Creates SQLite database `data/sales_data.db`
  - Generates charts and analytics JSON files

- `python train_models.py`
  - Trains regression and classification models
  - Saves the best estimator to `model/sales_prediction_model.pkl`
  - Saves model metrics to `data/model_results.json`

- `python app.py`
  - Runs the Flask web server with dashboard, prediction pages, and REST API endpoints

## 📡 REST API Endpoints
- `GET /dashboard` — Dashboard page
- `GET /analytics` — Full analytics JSON payload
- `GET /top-products?n=10&category=...` — Top-selling products
- `GET /customer-insights?search=...&page=1&limit=20` — Customer insight data
- `POST /predict-sales` — Predict revenue for a transaction
- `POST /predict-demand` — Predict product demand for a future month
- `GET /model-metrics` — ML evaluation metrics JSON
- `GET /health` — Health check endpoint

## 📈 Features Included
- Excel → CSV data conversion
- Data cleaning and validation
- Feature engineering: year, month, weekday, quarter, is_weekend, average order value, customer lifetime value, purchase frequency, days since last purchase
- SQL database persistence with SQLite
- Exploratory charts: line, bar, pie, heatmap, histogram, correlation matrix
- Regression models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, optional XGBoost
- Classification models: Logistic Regression, Decision Tree Classifier, Random Forest Classifier
- Model evaluation: MAE, MSE, RMSE, R², accuracy, precision, recall, F1
- RFM segmentation and churn risk analysis
- Interactive dashboard with Bootstrap and Chart.js

## 📌 Deployment
- Use `Procfile` for deployment with Gunicorn
- App ready for platforms such as Render, Railway, and PythonAnywhere

## 💡 Notes
- The primary dataset source is `data/customer_purchase_history.csv`
- SQLite is used for fast analytical queries via `data/sales_data.db`
- Charts and JSON summaries are regenerated each time `python preprocess.py` runs

## 📚 Future Improvements
- Add user authentication and role-based access
- Implement real-time filtering and date range selection on the dashboard
- Add product recommendation engine using collaborative filtering
- Expand demand forecasting with seasonal ARIMA or Prophet
- Add export to Excel / PDF for dashboard reports
