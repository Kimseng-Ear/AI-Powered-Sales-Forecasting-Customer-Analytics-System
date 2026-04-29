"""
Phase 1 & 5: Data Engineering, Preprocessing, and Advanced Analytics
Converts Excel to CSV, cleans data, engineers features, generates EDA charts,
and performs advanced analytics (KMeans, RFM, Churn, Forecasting).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import sqlite3
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ─── Directories ────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
os.makedirs('static/charts', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('model', exist_ok=True)

DATA_DB = 'data/sales_data.db'

# helper utilities

def save_json(filename, data):
    path = os.path.join('data', filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"   Saved → {path}")


def save_sqlite(df):
    conn = sqlite3.connect(DATA_DB)
    df.to_sql('customer_purchase_history', conn, if_exists='replace', index=False)
    conn.close()
    print(f"✅  Saved → {DATA_DB}")

# ─── Palette ─────────────────────────────────────────────────────────────────
PALETTE = ['#6C63FF', '#FF6584', '#43B89C', '#F7C59F', '#4A90D9',
           '#E17055', '#00CEC9', '#FDCB6E', '#A29BFE', '#55EFC4']
sns.set_style("whitegrid")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 14,
                     'axes.labelsize': 12})

# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CONVERT EXCEL → CSV
# ════════════════════════════════════════════════════════════════════════════
print("📥  Loading Excel file …")
df_raw = pd.read_excel('Customer-Purchase-History.xlsx', engine='openpyxl')
print(f"   Raw shape: {df_raw.shape}")
print(f"   Columns  : {list(df_raw.columns)}")

# ════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════
df = df_raw.copy()

# Standardise column names
df.columns = [c.strip().replace(' ', '') for c in df.columns]

# Required columns (flexible name mapping)
col_map = {}
for col in df.columns:
    cl = col.lower().replace('_', '').replace(' ', '')
    if 'customerid'   in cl: col_map[col] = 'CustomerID'
    elif 'customername' in cl: col_map[col] = 'CustomerName'
    elif 'product' == cl:      col_map[col] = 'Product'
    elif 'purchasedate' in cl: col_map[col] = 'PurchaseDate'
    elif 'quantity'    in cl:  col_map[col] = 'Quantity'
    elif 'unitprice'   in cl:  col_map[col] = 'UnitPrice'
    elif 'productcategory' in cl: col_map[col] = 'ProductCategory'
    elif 'paymentmethod'   in cl: col_map[col] = 'PaymentMethod'
    elif 'reviewrating'    in cl: col_map[col] = 'ReviewRating'
    elif 'totalprice'      in cl: col_map[col] = 'TotalPrice'

df.rename(columns=col_map, inplace=True)

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"   Dropped {before - len(df)} duplicate rows")

# Fix dtypes
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
for col in ['Quantity', 'UnitPrice', 'ReviewRating', 'TotalPrice']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill / drop missing
df['TotalPrice'].fillna(df['Quantity'] * df['UnitPrice'], inplace=True)
df.dropna(subset=['PurchaseDate', 'CustomerID', 'TotalPrice'], inplace=True)

# Standardise text
for col in ['Product', 'ProductCategory', 'PaymentMethod', 'CustomerName']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# Validate numerics
df = df[df['TotalPrice'] > 0]
df = df[df['Quantity']   > 0]
df = df[df['UnitPrice']  > 0]

print(f"   Clean shape: {df.shape}")

# ════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════
df['Year']      = df['PurchaseDate'].dt.year
df['Month']     = df['PurchaseDate'].dt.month
df['Day']       = df['PurchaseDate'].dt.day
df['Weekday']   = df['PurchaseDate'].dt.dayofweek          # 0=Mon
df['Quarter']   = df['PurchaseDate'].dt.quarter
df['IsWeekend'] = (df['Weekday'] >= 5).astype(int)
df['MonthName'] = df['PurchaseDate'].dt.strftime('%b')
df['YearMonth'] = df['PurchaseDate'].dt.to_period('M').astype(str)

# Customer-level aggregations
cust_agg = df.groupby('CustomerID').agg(
    CustomerLifetimeValue=('TotalPrice', 'sum'),
    PurchaseFrequency=('CustomerID', 'count'),
    LastPurchaseDate=('PurchaseDate', 'max')
).reset_index()
cust_agg['DaysSinceLastPurchase'] = (
    df['PurchaseDate'].max() - cust_agg['LastPurchaseDate']
).dt.days
cust_agg['AverageOrderValue'] = (
    cust_agg['CustomerLifetimeValue'] / cust_agg['PurchaseFrequency']
)

df = df.merge(cust_agg[['CustomerID', 'CustomerLifetimeValue',
                          'PurchaseFrequency', 'DaysSinceLastPurchase',
                          'AverageOrderValue']], on='CustomerID', how='left')

# Revenue per category / per customer
rev_cat  = df.groupby('ProductCategory')['TotalPrice'].sum().rename('RevenuePerCategory')
rev_cust = df.groupby('CustomerID')['TotalPrice'].sum().rename('RevenuePerCustomer')
df = df.join(rev_cat,  on='ProductCategory')
df = df.join(rev_cust, on='CustomerID', rsuffix='_x')
if 'RevenuePerCustomer_x' in df.columns:
    df.drop(columns=['RevenuePerCustomer_x'], inplace=True)

# Save clean CSV
df.to_csv('data/customer_purchase_history.csv', index=False)
print("✅  Saved → data/customer_purchase_history.csv")

# Save SQLite database for analytics queries
save_sqlite(df)

# ════════════════════════════════════════════════════════════════════════════
# 4. EDA CHARTS
# ════════════════════════════════════════════════════════════════════════════
print("📊  Generating EDA charts …")

def save_chart(name):
    path = f'static/charts/{name}.png'
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    return path

# 4-1  Monthly Sales Trend
monthly = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
monthly.columns = ['YearMonth', 'Revenue']
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly['YearMonth'], monthly['Revenue'], marker='o',
        color=PALETTE[0], linewidth=2)
ax.fill_between(range(len(monthly)), monthly['Revenue'],
                alpha=0.15, color=PALETTE[0])
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly['YearMonth'], rotation=45, ha='right', fontsize=8)
ax.set_title('Monthly Sales Revenue Trend')
ax.set_ylabel('Total Revenue ($)')
save_chart('monthly_sales_trend')

# 4-2  Top 10 Products
top_prod = df.groupby('Product')['TotalPrice'].sum().nlargest(10).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top_prod['Product'], top_prod['TotalPrice'],
               color=PALETTE[:len(top_prod)])
ax.set_title('Top 10 Products by Revenue')
ax.set_xlabel('Total Revenue ($)')
ax.invert_yaxis()
save_chart('top_products')

# 4-3  Revenue by Category (pie)
cat_rev = df.groupby('ProductCategory')['TotalPrice'].sum()
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    cat_rev.values, labels=cat_rev.index,
    autopct='%1.1f%%', colors=PALETTE[:len(cat_rev)],
    startangle=140, pctdistance=0.82)
ax.set_title('Revenue Distribution by Category')
save_chart('category_revenue_pie')

# 4-4  Payment Methods
pay = df['PaymentMethod'].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(pay.index, pay.values, color=PALETTE[:len(pay)])
ax.set_title('Payment Method Usage')
ax.set_ylabel('Transactions')
plt.xticks(rotation=30)
save_chart('payment_methods')

# 4-5  Review Rating Distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['ReviewRating'].dropna(), bins=20, color=PALETTE[2], edgecolor='white')
ax.set_title('Customer Review Rating Distribution')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')
save_chart('review_distribution')

# 4-6  Heatmap – Monthly Revenue per Year
try:
    pivot = df.pivot_table(values='TotalPrice', index='Year',
                           columns='Month', aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot))))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                linewidths=0.5)
    ax.set_title('Revenue Heatmap (Year × Month)')
    save_chart('revenue_heatmap')
except Exception as e:
    print(f"   Heatmap skipped: {e}")

# 4-7  Top 10 Customers
top_cust = df.groupby('CustomerName')['TotalPrice'].sum().nlargest(10).reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top_cust['CustomerName'], top_cust['TotalPrice'],
        color=PALETTE[:len(top_cust)])
ax.set_title('Top 10 Customers by Revenue')
ax.set_xlabel('Total Revenue ($)')
ax.invert_yaxis()
save_chart('top_customers')

# 4-8  Weekday Sales
wd_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
wd_rev = df.groupby('Weekday')['TotalPrice'].sum()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([wd_names[i] for i in wd_rev.index], wd_rev.values,
       color=PALETTE[:len(wd_rev)])
ax.set_title('Sales Revenue by Day of Week')
ax.set_ylabel('Total Revenue ($)')
save_chart('weekday_sales')

# 4-9  Quarterly Revenue
q_rev = df.groupby(['Year', 'Quarter'])['TotalPrice'].sum().reset_index()
q_rev['Period'] = q_rev['Year'].astype(str) + '-Q' + q_rev['Quarter'].astype(str)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(q_rev['Period'], q_rev['TotalPrice'], color=PALETTE[4])
ax.set_title('Quarterly Revenue')
ax.set_ylabel('Total Revenue ($)')
plt.xticks(rotation=45)
save_chart('quarterly_revenue')

# 4-10 Correlation Matrix
num_cols = ['Quantity', 'UnitPrice', 'TotalPrice', 'ReviewRating',
            'CustomerLifetimeValue', 'PurchaseFrequency',
            'DaysSinceLastPurchase', 'AverageOrderValue']
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, square=True)
ax.set_title('Feature Correlation Matrix')
save_chart('correlation_matrix')

print("✅  All EDA charts saved to static/charts/")

# ════════════════════════════════════════════════════════════════════════════
# 5. ADVANCED ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
print("🔬  Running advanced analytics …")

# ── 5a. RFM Analysis ─────────────────────────────────────────────────────────
ref_date = df['PurchaseDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg(
    Recency   =('PurchaseDate', lambda x: (ref_date - x.max()).days),
    Frequency =('CustomerID',  'count'),
    Monetary  =('TotalPrice',  'sum')
).reset_index()

rfm['R_Score'] = pd.qcut(rfm['Recency'],   4, labels=[4, 3, 2, 1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4,
                          labels=[1, 2, 3, 4]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'],  4, labels=[1, 2, 3, 4]).astype(int)
rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

def rfm_segment(score):
    if score >= 10: return 'Champions'
    elif score >= 8: return 'Loyal Customers'
    elif score >= 6: return 'Potential Loyalists'
    elif score >= 4: return 'At Risk'
    else:           return 'Lost'

rfm['Segment'] = rfm['RFM_Score'].apply(rfm_segment)
rfm.to_csv('data/rfm_analysis.csv', index=False)

seg_counts = rfm['Segment'].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(seg_counts.values, labels=seg_counts.index,
       autopct='%1.1f%%', colors=PALETTE[:len(seg_counts)], startangle=140)
ax.set_title('Customer RFM Segments')
save_chart('rfm_segments')

# ── 5b. KMeans Customer Segmentation ────────────────────────────────────────
scaler = StandardScaler()
rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
rfm_scaled = scaler.fit_transform(rfm_features)

# Elbow method
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(K_range), inertias, marker='o', color=PALETTE[0])
ax.set_title('KMeans Elbow Method')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
save_chart('kmeans_elbow')

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
rfm.to_csv('data/rfm_analysis.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(rfm['Frequency'], rfm['Monetary'],
                     c=rfm['Cluster'], cmap='Set1',
                     alpha=0.6, s=30)
ax.set_title('Customer Segments (KMeans Clustering)')
ax.set_xlabel('Purchase Frequency')
ax.set_ylabel('Monetary Value ($)')
plt.colorbar(scatter, label='Cluster')
save_chart('kmeans_clusters')

# ── 5c. Churn Risk Analysis ──────────────────────────────────────────────────
max_days = rfm['Recency'].max()
rfm['ChurnRisk'] = pd.cut(rfm['Recency'],
                           bins=[0, 30, 90, 180, max_days+1],
                           labels=['Active', 'Warm', 'Cold', 'Churned'])
churn_counts = rfm['ChurnRisk'].value_counts()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(churn_counts.index, churn_counts.values,
       color=['#43B89C', '#FDCB6E', '#F7C59F', '#FF6584'])
ax.set_title('Customer Churn Risk Distribution')
ax.set_ylabel('Number of Customers')
save_chart('churn_risk')

# ── 5d. 30/60/90-day Sales Forecasting (Linear Regression) ──────────────────
daily = df.groupby('PurchaseDate')['TotalPrice'].sum().reset_index()
daily.sort_values('PurchaseDate', inplace=True)
daily['DayIndex'] = (daily['PurchaseDate'] - daily['PurchaseDate'].min()).dt.days

X_fc = daily[['DayIndex']].values
y_fc = daily['TotalPrice'].values
lr_fc = LinearRegression()
lr_fc.fit(X_fc, y_fc)

last_day = daily['DayIndex'].max()
future_days = [last_day + d for d in [1, 30, 60, 90]]
future_dates = [daily['PurchaseDate'].max() + pd.Timedelta(days=d)
                for d in [1, 30, 60, 90]]
future_preds = lr_fc.predict([[d] for d in future_days])

forecast_df = pd.DataFrame({
    'Days': ['Day+1', 'Day+30', 'Day+60', 'Day+90'],
    'Date': future_dates,
    'ForecastedRevenue': future_preds
})
forecast_df.to_csv('data/sales_forecast.csv', index=False)

# Forecast chart
plot_daily = daily.tail(90)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(plot_daily['PurchaseDate'], plot_daily['TotalPrice'],
        color=PALETTE[0], label='Actual Sales', linewidth=1.5)
ax.plot(future_dates, future_preds, 'o--', color=PALETTE[1],
        label='Forecast', linewidth=2, markersize=8)
ax.set_title('Sales Forecast (30/60/90 Days)')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue ($)')
ax.legend()
save_chart('sales_forecast')

# ── 5e. Peak Sales Time ──────────────────────────────────────────────────────
hour_rev = df.groupby('Weekday')['TotalPrice'].mean()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar([wd_names[i] for i in hour_rev.index], hour_rev.values,
       color=PALETTE[3])
ax.set_title('Average Revenue by Day of Week (Peak Analysis)')
ax.set_ylabel('Avg Revenue ($)')
save_chart('peak_sales')

# ════════════════════════════════════════════════════════════════════════════
# 6. EXPORT ANALYTICS SUMMARY (for Flask API)
# ════════════════════════════════════════════════════════════════════════════
summary = {
    'total_revenue'    : float(df['TotalPrice'].sum()),
    'total_orders'     : int(len(df)),
    'total_customers'  : int(df['CustomerID'].nunique()),
    'total_products'   : int(df['Product'].nunique()),
    'avg_rating'       : float(df['ReviewRating'].mean()),
    'avg_order_value'  : float(df['AverageOrderValue'].mean()),
    'best_product'     : str(df.groupby('Product')['TotalPrice'].sum().idxmax()),
    'best_category'    : str(df.groupby('ProductCategory')['TotalPrice'].sum().idxmax()),
    'monthly_growth'   : float(
        (monthly['Revenue'].iloc[-1] - monthly['Revenue'].iloc[-2])
        / monthly['Revenue'].iloc[-2] * 100
        if len(monthly) >= 2 else 0),
    'forecast_30d'     : float(future_preds[1]),
    'forecast_60d'     : float(future_preds[2]),
    'forecast_90d'     : float(future_preds[3]),
}

with open('data/analytics_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Monthly trend JSON
monthly_json = monthly.to_dict(orient='records')
with open('data/monthly_trend.json', 'w') as f:
    json.dump(monthly_json, f, indent=2)

# Top products JSON
top_prod_json = df.groupby('Product')['TotalPrice'].sum().nlargest(10).reset_index()
top_prod_json.columns = ['Product', 'Revenue']
with open('data/top_products.json', 'w') as f:
    json.dump(top_prod_json.to_dict(orient='records'), f, indent=2)

# Category revenue JSON
cat_rev_json = df.groupby('ProductCategory')['TotalPrice'].sum().reset_index()
cat_rev_json.columns = ['Category', 'Revenue']
with open('data/category_revenue.json', 'w') as f:
    json.dump(cat_rev_json.to_dict(orient='records'), f, indent=2)

# RFM segment JSON
rfm_seg_json = rfm['Segment'].value_counts().reset_index()
rfm_seg_json.columns = ['Segment', 'Count']
with open('data/rfm_segments.json', 'w') as f:
    json.dump(rfm_seg_json.to_dict(orient='records'), f, indent=2)

# Churn JSON
churn_json = rfm['ChurnRisk'].value_counts().reset_index()
churn_json.columns = ['Risk', 'Count']
with open('data/churn_risk.json', 'w') as f:
    json.dump(churn_json.to_dict(orient='records'), f, indent=2)

# Payment method JSON
pay_json = df['PaymentMethod'].value_counts().reset_index()
pay_json.columns = ['Method', 'Count']
with open('data/payment_methods.json', 'w') as f:
    json.dump(pay_json.to_dict(orient='records'), f, indent=2)

# Forecast JSON
forecast_json = forecast_df.copy()
forecast_json['Date'] = forecast_json['Date'].dt.strftime('%Y-%m-%d')
with open('data/sales_forecast.json', 'w') as f:
    json.dump(forecast_json.to_dict(orient='records'), f, indent=2)

print("✅  Analytics summary JSON files saved to data/")
print("🎉  Preprocessing complete!\n")
print(f"   Total Revenue   : ${summary['total_revenue']:,.2f}")
print(f"   Total Orders    : {summary['total_orders']:,}")
print(f"   Total Customers : {summary['total_customers']:,}")
print(f"   Avg Rating      : {summary['avg_rating']:.2f}")
print(f"   Best Product    : {summary['best_product']}")
print(f"   30-day Forecast : ${summary['forecast_30d']:,.2f}")
