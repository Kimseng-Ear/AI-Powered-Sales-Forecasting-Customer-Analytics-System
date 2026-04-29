"""
Phase 3: Flask Backend — AI-Powered Sales Forecasting & Customer Analytics System
REST API + HTML template serving
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import sqlite3
import joblib, json, os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder=os.path.join(os.getcwd(), 'static'))

# ─── Load artifacts ──────────────────────────────────────────────────────────
DATA_DIR  = 'data'
MODEL_DIR = 'model'
DATABASE_PATH = os.path.join(DATA_DIR, 'sales_data.db')

def load_json(fname):
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_model(fname):
    path = os.path.join(MODEL_DIR, fname)
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Lazy-load heavy objects
_df = None
_model = None
_clf_model = None
_le_product = None
_le_category = None
_le_payment = None
_features = None

def get_db_connection():
    if not os.path.exists(DATABASE_PATH):
        return None
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_df():
    global _df
    if _df is None:
        csv_path = os.path.join(DATA_DIR, 'customer_purchase_history.csv')
        if os.path.exists(csv_path):
            _df = pd.read_csv(csv_path, parse_dates=['PurchaseDate'])
    return _df

def get_model():
    global _model
    if _model is None:
        _model = load_model('sales_prediction_model.pkl')
    return _model

def get_clf_model():
    global _clf_model
    if _clf_model is None:
        _clf_model = load_model('classification_model.pkl')
    return _clf_model

def get_encoders():
    global _le_product, _le_category, _le_payment, _features
    if _le_product is None:
        _le_product  = load_model('le_product.pkl')
        _le_category = load_model('le_category.pkl')
        _le_payment  = load_model('le_payment.pkl')
        _features    = load_model('feature_names.pkl')
    return _le_product, _le_category, _le_payment, _features

# ════════════════════════════════════════════════════════════════════════════
# HTML PAGE ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    print("Index route called")
    summary = load_json('analytics_summary.json')
    return render_template('index.html', summary=summary)

@app.route('/dashboard')
def dashboard():
    summary  = load_json('analytics_summary.json')
    monthly  = load_json('monthly_trend.json')
    top_prod = load_json('top_products.json')
    cat_rev  = load_json('category_revenue.json')
    payments = load_json('payment_methods.json')
    rfm_seg  = load_json('rfm_segments.json')
    churn    = load_json('churn_risk.json')
    return render_template('dashboard.html',
                           summary=summary,
                           monthly=monthly,
                           top_products=top_prod,
                           category_revenue=cat_rev,
                           payments=payments,
                           rfm_segments=rfm_seg,
                           churn=churn)

@app.route('/prediction')
def prediction_page():
    df = get_df()
    products   = sorted(df['Product'].dropna().unique().tolist())   if df is not None else []
    categories = sorted(df['ProductCategory'].dropna().unique().tolist()) if df is not None else []
    payments   = sorted(df['PaymentMethod'].dropna().unique().tolist())   if df is not None else []
    model_results = load_json('model_results.json')
    return render_template('prediction.html',
                           products=products,
                           categories=categories,
                           payments=payments,
                           model_results=model_results)

# ════════════════════════════════════════════════════════════════════════════
# REST API ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.route('/analytics', methods=['GET'])
def analytics():
    """GET /analytics — Returns full analytics summary"""
    summary   = load_json('analytics_summary.json')
    monthly   = load_json('monthly_trend.json')
    top_prod  = load_json('top_products.json')
    cat_rev   = load_json('category_revenue.json')
    rfm_seg   = load_json('rfm_segments.json')
    churn     = load_json('churn_risk.json')
    payments  = load_json('payment_methods.json')
    forecast  = load_json('sales_forecast.json') if os.path.exists(
                  os.path.join(DATA_DIR, 'sales_forecast.csv')) else {}

    return jsonify({
        'status'          : 'success',
        'summary'         : summary,
        'monthly_trend'   : monthly,
        'top_products'    : top_prod,
        'category_revenue': cat_rev,
        'rfm_segments'    : rfm_seg,
        'churn_risk'      : churn,
        'payment_methods' : payments,
    })

@app.route('/top-products', methods=['GET'])
def top_products():
    """GET /top-products — Returns top-selling products"""
    n        = request.args.get('n', 10, type=int)
    category = request.args.get('category', None)
    df       = get_df()
    if df is None:
        return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

    conn = get_db_connection()
    if conn:
        if category:
            rows = conn.execute(
                "SELECT Product, SUM(TotalPrice) AS Revenue, SUM(Quantity) AS Quantity "
                "FROM customer_purchase_history "
                "WHERE LOWER(ProductCategory)=? "
                "GROUP BY Product "
                "ORDER BY Revenue DESC LIMIT ?",
                (category.lower(), n)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT Product, SUM(TotalPrice) AS Revenue, SUM(Quantity) AS Quantity "
                "FROM customer_purchase_history "
                "GROUP BY Product "
                "ORDER BY Revenue DESC LIMIT ?",
                (n,)
            ).fetchall()
        conn.close()
        data = [dict(row) for row in rows]
        return jsonify({'status': 'success', 'data': data})

    if category:
        df = df[df['ProductCategory'].str.lower() == category.lower()]

    top = (df.groupby('Product')['TotalPrice']
             .sum()
             .nlargest(n)
             .reset_index())
    top.columns = ['Product', 'Revenue']

    qty_top = (df.groupby('Product')['Quantity']
                 .sum()
                 .reset_index())
    top = top.merge(qty_top, on='Product', how='left')

    return jsonify({'status': 'success', 'data': top.to_dict(orient='records')})

@app.route('/customer-insights', methods=['GET'])
def customer_insights():
    """GET /customer-insights — Returns customer analytics"""
    search = request.args.get('search', '').lower()
    page   = request.args.get('page', 1, type=int)
    limit  = request.args.get('limit', 20, type=int)
    df     = get_df()
    conn   = get_db_connection()
    if conn:
        query = (
            "SELECT CustomerID, CustomerName, "
            "SUM(TotalPrice) AS TotalRevenue, "
            "COUNT(*) AS TotalOrders, "
            "AVG(TotalPrice) AS AvgOrderValue, "
            "AVG(ReviewRating) AS AvgRating, "
            "MAX(PurchaseDate) AS LastPurchase, "
            "MAX(CustomerLifetimeValue) AS LifetimeValue "
            "FROM customer_purchase_history "
        )
        params = []
        if search:
            query += "WHERE LOWER(CustomerName) LIKE ? OR CustomerID LIKE ? "
            params.extend([f"%{search}%", f"%{search}%"])
        query += "GROUP BY CustomerID, CustomerName "
        query += "ORDER BY TotalRevenue DESC "
        query += "LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        data = [dict(row) for row in rows]
        return jsonify({
            'status': 'success',
            'total' : len(data),
            'page'  : page,
            'limit' : limit,
            'data'  : data,
        })

    if df is None:
        return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

    cust = df.groupby(['CustomerID', 'CustomerName']).agg(
        TotalRevenue       =('TotalPrice',  'sum'),
        TotalOrders        =('CustomerID',  'count'),
        AvgOrderValue      =('TotalPrice',  'mean'),
        AvgRating          =('ReviewRating','mean'),
        LastPurchase       =('PurchaseDate','max'),
        LifetimeValue      =('CustomerLifetimeValue', 'first'),
    ).reset_index()

    cust['LastPurchase'] = cust['LastPurchase'].astype(str)

    if search:
        mask = (cust['CustomerName'].str.lower().str.contains(search, na=False) |
                cust['CustomerID'].astype(str).str.contains(search, na=False))
        cust = cust[mask]

    total = len(cust)
    cust  = cust.sort_values('TotalRevenue', ascending=False)
    cust  = cust.iloc[(page-1)*limit : page*limit]

    return jsonify({
        'status' : 'success',
        'total'  : total,
        'page'   : page,
        'limit'  : limit,
        'data'   : cust.round(2).to_dict(orient='records'),
    })

@app.route('/predict-sales', methods=['POST'])
def predict_sales():
    """POST /predict-sales — Predicts total sale amount for a transaction"""
    try:
        body = request.get_json(force=True)
        le_prod, le_cat, le_pay, features = get_encoders()
        model = get_model()
        if model is None:
            return jsonify({'status': 'error',
                            'message': 'Model not trained yet. Run train_models.py first.'}), 503

        # Safe encode
        def safe_encode(le, val, default=0):
            try:
                return int(le.transform([str(val)])[0])
            except Exception:
                return default

        quantity    = float(body.get('quantity',    1))
        unit_price  = float(body.get('unit_price',  0))
        month       = int(body.get('month',         datetime.now().month))
        year        = int(body.get('year',          datetime.now().year))
        quarter     = (month - 1) // 3 + 1
        weekday     = int(body.get('weekday',       0))
        is_weekend  = 1 if weekday >= 5 else 0
        product_enc = safe_encode(le_prod, body.get('product',  ''))
        cat_enc     = safe_encode(le_cat,  body.get('category', ''))
        pay_enc     = safe_encode(le_pay,  body.get('payment_method', ''))
        rating      = float(body.get('review_rating', 3.0))
        freq        = float(body.get('purchase_frequency', 5))
        avg_order   = float(body.get('average_order_value', unit_price * quantity))

        row = {
            'Quantity'            : quantity,
            'UnitPrice'           : unit_price,
            'Month'               : month,
            'Year'                : year,
            'Quarter'             : quarter,
            'IsWeekend'           : is_weekend,
            'Weekday'             : weekday,
            'Product_enc'         : product_enc,
            'ProductCategory_enc' : cat_enc,
            'PaymentMethod_enc'   : pay_enc,
            'ReviewRating'        : rating,
            'PurchaseFrequency'   : freq,
            'AverageOrderValue'   : avg_order,
        }

        input_df = pd.DataFrame([[row.get(f, 0) for f in features]],
                                 columns=features)
        prediction = float(model.predict(input_df)[0])

        # High-value classification
        clf    = get_clf_model()
        is_hv  = bool(clf.predict(input_df)[0]) if clf else False
        hv_prob = float(clf.predict_proba(input_df)[0][1]) if clf and hasattr(clf, 'predict_proba') else 0.0

        return jsonify({
            'status'              : 'success',
            'predicted_revenue'   : round(prediction, 2),
            'is_high_value'       : is_hv,
            'high_value_probability': round(hv_prob * 100, 1),
            'input_summary'       : {
                'quantity'   : quantity,
                'unit_price' : unit_price,
                'product'    : body.get('product', ''),
                'category'   : body.get('category', ''),
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/predict-demand', methods=['POST'])
def predict_demand():
    """POST /predict-demand — Predicts demand for a product in a future month"""
    try:
        body     = request.get_json(force=True)
        product  = body.get('product',  '').strip()
        month    = int(body.get('month',  datetime.now().month))
        year     = int(body.get('year',   datetime.now().year))

        df = get_df()
        if df is None:
            return jsonify({'status': 'error', 'message': 'Data not loaded'}), 500

        prod_df = df[df['Product'].str.lower() == product.lower()]
        if prod_df.empty:
            return jsonify({'status': 'error', 'message': f'Product "{product}" not found'}), 404

        monthly_qty = prod_df.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()
        monthly_qty['DayIndex'] = (monthly_qty['Year'] - monthly_qty['Year'].min()) * 12 + \
                                   monthly_qty['Month']

        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(monthly_qty[['DayIndex']], monthly_qty['Quantity'])

        future_idx = (year - monthly_qty['Year'].min()) * 12 + month
        predicted_qty = float(lr.predict([[future_idx]])[0])

        avg_unit_price = float(prod_df['UnitPrice'].mean())
        predicted_rev  = max(0, predicted_qty) * avg_unit_price

        hist = (monthly_qty[['Month', 'Quantity']]
                .groupby('Month')['Quantity'].mean()
                .reset_index()
                .to_dict(orient='records'))

        return jsonify({
            'status'               : 'success',
            'product'              : product,
            'target_month'         : f'{year}-{month:02d}',
            'predicted_quantity'   : round(max(0, predicted_qty), 0),
            'predicted_revenue'    : round(predicted_rev, 2),
            'avg_unit_price'       : round(avg_unit_price, 2),
            'monthly_avg_history'  : hist,
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/forecast', methods=['GET'])
def forecast():
    """GET /forecast — Returns 30/60/90-day revenue forecasts"""
    try:
        path = os.path.join(DATA_DIR, 'sales_forecast.csv')
        if not os.path.exists(path):
            return jsonify({'status': 'error', 'message': 'Forecast not generated yet'}), 503
        fc = pd.read_csv(path)
        fc['Date'] = fc['Date'].astype(str)
        return jsonify({'status': 'success', 'data': fc.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/rfm-analysis', methods=['GET'])
def rfm_analysis():
    """GET /rfm-analysis — Returns RFM segmentation data"""
    try:
        rfm_path = os.path.join(DATA_DIR, 'rfm_analysis.csv')
        if not os.path.exists(rfm_path):
            return jsonify({'status': 'error', 'message': 'RFM analysis not run yet'}), 503
        rfm = pd.read_csv(rfm_path)
        rfm['LastPurchaseDate'] = rfm['LastPurchaseDate'].astype(str)

        seg_summary = rfm['Segment'].value_counts().reset_index()
        seg_summary.columns = ['Segment', 'Count']
        seg_summary['AvgMonetary'] = [
            float(rfm[rfm['Segment']==s]['Monetary'].mean())
            for s in seg_summary['Segment']
        ]

        top_champions = rfm[rfm['Segment']=='Champions'].nlargest(10, 'Monetary')

        return jsonify({
            'status'       : 'success',
            'segment_summary': seg_summary.to_dict(orient='records'),
            'churn_risk'   : rfm['ChurnRisk'].value_counts().reset_index()
                                             .rename(columns={'index':'Risk',
                                                              'ChurnRisk':'Count'})
                                             .to_dict(orient='records'),
            'top_champions': top_champions[['CustomerID', 'Recency',
                                             'Frequency', 'Monetary',
                                             'RFM_Score']].to_dict(orient='records'),
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/model-metrics', methods=['GET'])
def model_metrics():
    """GET /model-metrics — Returns ML model evaluation metrics"""
    results = load_json('model_results.json')
    if not results:
        return jsonify({'status': 'error', 'message': 'Model results not found'}), 503
    return jsonify({'status': 'success', 'data': results})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/logo.png')
def logo_file():
    return send_from_directory(os.getcwd(), 'logo.png')

@app.route('/static/<path:filename>')
def static_files(filename):
    print(f"Serving static file: {filename}")
    return send_from_directory('static', filename)

# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
