
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="LATTICE | PWR Intelligence", layout="wide", page_icon="⚛️")



st.markdown("""
<style>
    html, body, [class*="st-"] { color: white; }
   .stApp { background: linear-gradient(180deg, #0E1117 0%, #161A25 100%); }
    [data-testid="stSidebar"] { background-color: #111827; color: white; }

   .kpi-card { background: #1F2937; padding: 25px; border-radius: 15px; border-left: 5px solid #00FF88; color: white; }
   .kpi-value { font-size: 32px; font-weight: 700; color: #00FF88; }
   .kpi-label { font-size: 15px; color: white !important; }

    h1, h2, h3 { color: white !important; }

    /* تنسيق الـ number_input كامل */
    div[data-testid="stNumberInput"] label { color: white !important; font-weight: 600; } 
    div[data-testid="stNumberInput"] input { color: black !important; background-color: white !important; border: 1px solid #4A5568; border-radius: 8px; } 
    div[data-testid="stNumberInput"] button { color: black !important; background-color: #4A5568 !important; } 
    div[data-testid="stNumberInput"] button:hover { background-color: #CBD5E0 !important; }

   .stButton>button { background-color: #4A5568 !important; color: #0E1117; font-weight: bold; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.title("⚛️ LATTICE")
    page = st.radio("Navigation", ["🏠 Home", "🎯 Prediction"])

@st.cache_data
def load_data():
   
    df = pd.read_csv("raw.csv", sep=r'\s+', header=None)
    feature_names = [f'feature_{i}' for i in range(1,40)]
    df.columns = ['k_inf', 'PPPF'] + feature_names
    return df

df = load_data()


x = df.drop(columns=['k_inf', 'PPPF'])
y = df[['k_inf', 'PPPF']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)

@st.cache_resource
def train_models():
    model_k =  XGBRegressor(n_estimators=3000, random_state=42 , learning_rate=0.05,
    max_depth=3,).fit(x_train_scaled, y_train['k_inf'])
    model_p =  XGBRegressor(n_estimators=4000, random_state=42, learning_rate=0.05,
    max_depth=3,).fit(x_train_scaled, y_train['PPPF'])
    return model_k, model_p

model_k, model_p = train_models()

if page == "🏠 Home":
    st.title("LATTICE | PWR Physics Intelligence")

    yk_pred = model_k.predict(x_test_scaled)
    yp_pred = model_p.predict(x_test_scaled)

    r2_k = r2_score(y_test['k_inf'], yk_pred)
    rmse_k = np.sqrt(mean_squared_error(y_test['k_inf'], yk_pred))
    r2_p = r2_score(y_test['PPPF'], yp_pred)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{x.shape[1]}</div><div class="kpi-label">Features</div></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{df.shape[0]:,}</div><div class="kpi-label">Simulations</div></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{r2_k:.4f}</div><div class="kpi-label">R² k_inf</div></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="kpi-card"><div class="kpi-value">{rmse_k:.5f}</div><div class="kpi-label">RMSE k_inf</div></div>', unsafe_allow_html=True)

    st.subheader("📊 Top 10 Important Features for k_inf")
    importance = pd.DataFrame({'Feature': x.columns, 'Importance': model_k.feature_importances_})
    importance = importance.sort_values('Importance', ascending=False).head(10)
    st.bar_chart(importance.set_index('Feature'))

elif page == "🎯 Prediction":
    st.title("Real-Time k_inf & PPPF Prediction")

    inputs = {}
    cols = st.columns(3)
    for i, col_name in enumerate(x.columns[:15]): 
        with cols[i % 3]:
            inputs[col_name] = st.number_input(col_name, value=float(x[col_name].mean()))

    for col_name in x.columns[15:]:
        inputs[col_name] = x[col_name].mean()

    if st.button(" Predict Now"):
        input_df = pd.DataFrame([inputs])
        input_df = input_df[x.columns]

    
        input_scaled = scaler_x.transform(input_df)

        pred_k = model_k.predict(input_scaled)[0]
        pred_p = model_p.predict(input_scaled)[0]

        st.success(f"### Predicted k_inf: {pred_k:.4f}")
        st.success(f"### Predicted PPPF: {pred_p:.4f}")


