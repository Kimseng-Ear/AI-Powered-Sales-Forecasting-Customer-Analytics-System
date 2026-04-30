"""
Phase 2: Machine Learning Sales Prediction Models
Trains multiple regression and classification models, evaluates them,
generates feature-importance charts, and saves the best model.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, json, os, warnings

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing       import LabelEncoder, StandardScaler
from sklearn.linear_model        import LinearRegression, LogisticRegression
from sklearn.tree                import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble            import (RandomForestRegressor,
                                         RandomForestClassifier,
                                         GradientBoostingRegressor)
from sklearn.metrics             import (mean_absolute_error, mean_squared_error,
                                         r2_score, accuracy_score, precision_score,
                                         recall_score, f1_score, classification_report)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings('ignore')
os.makedirs('model',         exist_ok=True)
os.makedirs('static/charts', exist_ok=True)

PALETTE = ['#6C63FF', '#FF6584', '#43B89C', '#F7C59F', '#4A90D9',
           '#E17055', '#00CEC9', '#FDCB6E']

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
print("Loading preprocessed data...")
df = pd.read_csv('data/customer_purchase_history.csv', parse_dates=['PurchaseDate'])
print(f"   Shape: {df.shape}")

# ════════════════════════════════════════════════════════════════════════════
# FEATURE PREPARATION
# ════════════════════════════════════════════════════════════════════════════
le_product  = LabelEncoder()
le_category = LabelEncoder()
le_payment  = LabelEncoder()

df['Product_enc']         = le_product.fit_transform(df['Product'].astype(str))
df['ProductCategory_enc'] = le_category.fit_transform(df['ProductCategory'].astype(str))
df['PaymentMethod_enc']   = le_payment.fit_transform(df['PaymentMethod'].astype(str))

# Save encoders
joblib.dump(le_product,  'model/le_product.pkl')
joblib.dump(le_category, 'model/le_category.pkl')
joblib.dump(le_payment,  'model/le_payment.pkl')

FEATURES = ['Quantity', 'UnitPrice', 'Month', 'Year', 'Quarter', 'IsWeekend',
            'Weekday', 'Product_enc', 'ProductCategory_enc', 'PaymentMethod_enc',
            'ReviewRating', 'PurchaseFrequency', 'AverageOrderValue']

FEATURES = [f for f in FEATURES if f in df.columns]
TARGET   = 'TotalPrice'

df_model = df[FEATURES + [TARGET]].dropna()
X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

# ── REGRESSION MODELS (Tuned for Accuracy) ──────────────────
reg_models = {
    'Linear Regression'   : LinearRegression(),
    'Decision Tree'       : DecisionTreeRegressor(random_state=42, max_depth=12),
    'Random Forest'       : RandomForestRegressor(n_estimators=200, max_depth=15, 
                                                  random_state=42, n_jobs=-1),
    'Gradient Boosting'   : GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=5, random_state=42),
}
if HAS_XGB:
    reg_models['XGBoost'] = XGBRegressor(n_estimators=300, learning_rate=0.05, 
                                          max_depth=6, random_state=42, verbosity=0)

reg_results = {}
print("\nTraining Regression Models ...")
for name, model in reg_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    mse   = mean_squared_error(y_test, preds)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_test, preds)
    reg_results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    print(f"   {name:<25} MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

# Best regression model by R²
best_reg_name  = max(reg_results, key=lambda k: reg_results[k]['R2'])
best_reg_model = reg_models[best_reg_name]
print(f"\nBest regression model: {best_reg_name}  (R2={reg_results[best_reg_name]['R2']:.4f})")

joblib.dump(best_reg_model, 'model/sales_prediction_model.pkl')
joblib.dump(FEATURES,       'model/feature_names.pkl')
print("Saved best model -> model/sales_prediction_model.pkl")

# ════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION — High-Value Customer (top 25%)
# ════════════════════════════════════════════════════════════════════════════
threshold = df_model[TARGET].quantile(0.75)
df_model['HighValue'] = (df_model[TARGET] >= threshold).astype(int)

X_c = df_model[FEATURES]
y_c = df_model['HighValue']
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42)

clf_models = {
    'Logistic Regression'     : LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree Classifier': DecisionTreeClassifier(random_state=42, max_depth=8),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=200, max_depth=15,
                                                        random_state=42, n_jobs=-1),
}

clf_results = {}
print("\nTraining Classification Models ...")
for name, model in clf_models.items():
    model.fit(Xc_train, yc_train)
    preds  = model.predict(Xc_test)
    acc    = accuracy_score(yc_test, preds)
    prec   = precision_score(yc_test, preds, zero_division=0)
    rec    = recall_score(yc_test, preds, zero_division=0)
    f1     = f1_score(yc_test, preds, zero_division=0)
    clf_results[name] = {'Accuracy': acc, 'Precision': prec,
                         'Recall': rec, 'F1': f1}
    print(f"   {name:<30} Acc={acc:.4f}  F1={f1:.4f}")

best_clf_name  = max(clf_results, key=lambda k: clf_results[k]['F1'])
best_clf_model = clf_models[best_clf_name]
joblib.dump(best_clf_model, 'model/classification_model.pkl')

# ════════════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════════════
def save_chart(name):
    plt.tight_layout()
    plt.savefig(f'static/charts/{name}.png', dpi=110, bbox_inches='tight')
    plt.close()

# Model comparison – Regression
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['MAE', 'RMSE', 'R2']
for i, metric in enumerate(metrics):
    vals  = [reg_results[m][metric] for m in reg_results]
    names = list(reg_results.keys())
    axes[i].barh(names, vals, color=PALETTE[:len(names)])
    axes[i].set_title(metric)
    axes[i].invert_yaxis()
fig.suptitle('Regression Model Comparison', fontsize=14, fontweight='bold')
save_chart('model_comparison_regression')

# Model comparison – Classification
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
metrics_c = ['Accuracy', 'Precision', 'Recall', 'F1']
for i, metric in enumerate(metrics_c):
    vals  = [clf_results[m][metric] for m in clf_results]
    names = list(clf_results.keys())
    axes[i].barh(names, vals, color=PALETTE[:len(names)])
    axes[i].set_xlim(0, 1)
    axes[i].set_title(metric)
    axes[i].invert_yaxis()
fig.suptitle('Classification Model Comparison', fontsize=14, fontweight='bold')
save_chart('model_comparison_classification')

# Feature Importance (Random Forest)
if 'Random Forest' in reg_models:
    rf_model = reg_models['Random Forest']
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
    fi_df.sort_values('Importance', ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(FEATURES)*0.4)))
    ax.barh(fi_df['Feature'], fi_df['Importance'], color=PALETTE[0])
    ax.set_title('Feature Importance (Random Forest)')
    ax.set_xlabel('Importance Score')
    save_chart('feature_importance')

# Actual vs Predicted
rf_preds = reg_models[best_reg_name].predict(X_test)
fig, ax  = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, rf_preds, alpha=0.4, color=PALETTE[0], s=15)
mn, mx = min(y_test.min(), rf_preds.min()), max(y_test.max(), rf_preds.max())
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
ax.set_xlabel('Actual Revenue ($)')
ax.set_ylabel('Predicted Revenue ($)')
ax.set_title(f'Actual vs Predicted — {best_reg_name}')
save_chart('actual_vs_predicted')

# Residuals
residuals = y_test.values - rf_preds
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(residuals, bins=40, color=PALETTE[2], edgecolor='white')
ax.axvline(0, color='red', linestyle='--')
ax.set_title('Residuals Distribution')
ax.set_xlabel('Residual')
save_chart('residuals')

# ════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS JSON
# ════════════════════════════════════════════════════════════════════════════
model_results = {
    'regression'    : {k: {m: float(v) for m, v in vv.items()}
                       for k, vv in reg_results.items()},
    'classification': {k: {m: float(v) for m, v in vv.items()}
                       for k, vv in clf_results.items()},
    'best_regression_model'    : best_reg_name,
    'best_classification_model': best_clf_name,
    'features'                 : FEATURES,
    'threshold_high_value'     : float(threshold),
}
with open('data/model_results.json', 'w') as f:
    json.dump(model_results, f, indent=2)

print("\nModel results saved -> data/model_results.json")
print("Training complete!")
