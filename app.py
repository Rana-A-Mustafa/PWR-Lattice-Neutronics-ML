import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="LATTICE | PWR Physics Intelligence", layout="wide", page_icon="⚛️")

# CSS Nuclear Control Room
st.markdown("""
<style>
.stApp { background: #05070D; }
[data-testid="stSidebar"] { background-color: #0A0F1A; }
.kpi-card { background: #0F1422; padding: 18px; border-radius: 10px; border-left: 4px solid; }
.kpi-value { font-size: 26px; font-weight: 800; }
.kpi-label { font-size: 12px; color: #64748B!important; text-transform: uppercase; }
.alert-box { padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; }
.green-alert { background: #022C22; border: 1px solid #00FF88; color: #00FF88; }
.red-alert { background: #450A0A; border: 1px solid #EF4444; color: #EF4444; }
.orange-alert { background: #431407; border: 1px solid #F97316; color: #F97316; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚛️ LATTICE") 
    st.markdown("### PWR Physics Intelligence")
    page = st.radio("Navigation", ["📊 Control Room", "🎯 Predictor", "📈 Analytics"])

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
    model_k = XGBRegressor(n_estimators=3000, random_state=42, learning_rate=0.05, max_depth=3).fit(x_train_scaled, y_train['k_inf'])
    model_p = XGBRegressor(n_estimators=4000, random_state=42, learning_rate=0.05, max_depth=3).fit(x_train_scaled, y_train['PPPF'])
    return model_k, model_p

model_k, model_p = train_models()

def plot_gauge(value, title):
    color = "#00FF88" if 1.0 <= value <= 1.3 else "#F97316" if value > 1.3 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'color': 'white'}},
        gauge = {'axis': {'range': [0.8, 1.4]}, 'bar': {'color': color},
        'steps': [{'range': [0.8, 1.0], 'color': "#EF4444"}, {'range': [1.0, 1.3], 'color': "#00FF88"}, {'range': [1.3, 1.4], 'color': "#F97316"}]}))
    fig.update_layout(paper_bgcolor = "#0F1422", height=250)
    return fig


yk_pred = model_k.predict(x_test_scaled)
yp_pred = model_p.predict(x_test_scaled)
r2_k = r2_score(y_test['k_inf'], yk_pred)
rmse_k = np.sqrt(mean_squared_error(y_test['k_inf'], yk_pred))

if page == "📊 Control Room":
    st.title("LATTICE | PWR Physics Intelligence") # العنوان القديم

    # صف 1: 4 KPI Cards
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card" style="border-color:#3B82F6"><div class="kpi-label">R² Score k_inf</div><div class="kpi-value" style="color:#3B82F6">{r2_k:.4f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card" style="border-color:#00FF88"><div class="kpi-label">RMSE</div><div class="kpi-value" style="color:#00FF88">{rmse_k:.5f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card" style="border-color:#8B5CF6"><div class="kpi-label">Dataset Size</div><div class="kpi-value" style="color:#8B5CF6">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card" style="border-color:#F97316"><div class="kpi-label">Features</div><div class="kpi-value" style="color:#F97316">{x.shape[1]}</div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.plotly_chart(plot_gauge(1.0, "Core k_inf Status"), use_container_width=True)
        st.markdown('<div class="alert-box green-alert"> REACTOR CRITICAL</div>', unsafe_allow_html=True)
    with col2:
        core_data = np.random.rand(10,10) * 2 + 1.0
        fig = px.imshow(core_data, color_continuous_scale='RdYlGn_r', aspect="equal")
        fig.update_layout(title="Core Power Distribution Map", paper_bgcolor="#0F1422", height=250)
        st.plotly_chart(fig, use_container_width=True)

    # صف 3: 2 شارت
    c1,c2 = st.columns(2)
    with c1:
        importance = pd.DataFrame({'Feature': x.columns[:15], 'Importance': model_k.feature_importances_[:15]})
        fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title="Top 15 Features Impact")
        fig.update_layout(paper_bgcolor="#0F1422")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(x=yk_pred, y=y_test['k_inf'], title="Predicted vs Actual k_inf")
        fig.update_layout(paper_bgcolor="#0F1422")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Predictor":
    st.title("Real-Time Prediction")

    st.subheader("Enter Reactor Parameters") 
    cols = st.columns(3) 
    inputs = {}
    for i, col_name in enumerate(x.columns):
        with cols[i % 3]:
            inputs[col_name] = st.number_input(col_name, value=1.30, min_value=0.0, step=0.01, format="%.2f")

    if st.button("⚛️ Run Prediction"):
        input_scaled = scaler_x.transform(pd.DataFrame([inputs])[x.columns])
        pred_k, pred_p = model_k.predict(input_scaled)[0], model_p.predict(input_scaled)[0]

        # Alert
        if pred_k < 1.0: st.markdown(f'<div class="alert-box red-alert">🚨 SUBCRITICAL: {pred_k:.5f}</div>', unsafe_allow_html=True)
        elif pred_k > 1.3: st.markdown(f'<div class="alert-box orange-alert">⚠️ HIGH POWER: {pred_k:.5f}</div>', unsafe_allow_html=True)
        else: st.markdown(f'<div class="alert-box green-alert">✅ SAFE: {pred_k:.5f}</div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("k_inf", f"{pred_k:.5f}")
        c2.metric("PPPF", f"{pred_p:.5f}")
        c3.metric("Safety Margin", f"{1.3-pred_k:.4f}")
        st.plotly_chart(plot_gauge(pred_k, "Live k_inf"), use_container_width=True)

elif page == "📈 Analytics":
    st.title("Model Analytics")
    tab1, tab2 = st.tabs(["Feature Importance", "Error Distribution"])
    with tab1:
        imp = pd.DataFrame({'Feature': x.columns, 'Importance': model_k.feature_importances_}).sort_values('Importance', ascending=False).head(20)
        st.bar_chart(imp.set_index('Feature'))
    with tab2:
        errors = y_test['k_inf'] - yk_pred
        fig = px.histogram(errors, nbins=100, title="Prediction Error Distribution")
        fig.update_layout(paper_bgcolor="#0F1422")
        st.plotly_chart(fig)

