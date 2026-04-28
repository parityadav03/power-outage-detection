import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from tensorflow import keras

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title = "Power Outage Detection System",
    page_icon  = "⚡",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .metric-card {
      background: #1e2130;
      border: 1px solid #2e3250;
      border-radius: 10px;
      padding: 20px;
      text-align: center;
  }
  .metric-value {
      font-size: 2rem;
      font-weight: 700;
      color: #00d4aa;
  }
  .metric-label {
      font-size: 0.85rem;
      color: #8892b0;
      margin-top: 4px;
  }
  .outage-alert {
      background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
      border: 1px solid #ff4b4b;
      border-radius: 10px;
      padding: 16px;
      text-align: center;
      font-size: 1.2rem;
      font-weight: 600;
      color: #ff4b4b;
  }
  .normal-alert {
      background: linear-gradient(135deg, #00d4aa22, #00d4aa44);
      border: 1px solid #00d4aa;
      border-radius: 10px;
      padding: 16px;
      text-align: center;
      font-size: 1.2rem;
      font-weight: 600;
      color: #00d4aa;
  }
  .section-header {
      font-size: 1.1rem;
      font-weight: 600;
      color: #ccd6f6;
      padding: 8px 0;
      border-bottom: 1px solid #2e3250;
      margin-bottom: 12px;
  }
  div[data-testid="stSidebar"] {
      background-color: #1e2130;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD ALL ARTIFACTS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    from xgboost import XGBClassifier

    # XGBoost — native JSON, no pickle issues
    xgb = XGBClassifier()
    xgb.load_model('xgb_model_native.json')

    # Scaler and Isolation Forest
    scaler = joblib.load('scaler_hard.pkl')
    iso    = joblib.load('model_isolation_forest.pkl')

    # Neural Network
    try:
        nn = keras.models.load_model('best_final_model.keras')
    except Exception as e:
        nn = None

     # ADD THESE 3 LINES
    lstm_data   = joblib.load('lstm_ae_results.pkl')
    lstm_ae = tf.keras.models.load_model('best_lstm_ae.h5')
    lstm_thresh = lstm_data['threshold']

    return xgb, scaler, iso, nn, lstm_ae, lstm_thresh

@st.cache_data
def load_data():
    df = pd.read_csv('features_day2.csv',
                     index_col='DateTime', parse_dates=True)
    feature_cols = [c for c in df.columns if c != 'outage']
    X = df[feature_cols].copy()
    y = df['outage'].copy()

    # Rebuild hardened features
    X['z1_z2_ratio']      = X['zone1'] / (X['zone2'] + 1e-6)
    X['z1_z3_ratio']      = X['zone1'] / (X['zone3'] + 1e-6)
    X['all_zones_sum']    = X['zone1'] + X['zone2'] + X['zone3']
    X['zone_std']         = X[['zone1','zone2','zone3']].std(axis=1)
    X['temp_x_hour']      = X['temp'] * X['hour']
    X['wind_x_z1']        = X['wind_speed'] * X['zone1']
    X['humidity_x_z1']    = X['humidity'] * X['zone1']
    X['volatility_ratio'] = (X['z1_rolling_std_w12'] /
                              (X['z1_rolling_std_w144'] + 1e-6))
    X['zscore_momentum']  = X['z1_zscore'].diff(6).fillna(0)

    for col in ['z1_z2_ratio','z1_z3_ratio','all_zones_sum','zone_std',
                'temp_x_hour','wind_x_z1','humidity_x_z1',
                'volatility_ratio','zscore_momentum']:
        p1  = X[col].quantile(0.01)
        p99 = X[col].quantile(0.99)
        X[col] = X[col].clip(p1, p99)

    return df, X, y, feature_cols

xgb_model, scaler, iso_model, nn_model, lstm_ae, lstm_thresh = load_models()
df_raw, X_hard, y_true, feature_cols   = load_data()
X_scaled = scaler.transform(X_hard)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Power Outage Detection")
    st.markdown("**BTech Final Year Project**")
    st.markdown("*Data Mining Course*")
    st.markdown("---")

    st.markdown("### 🔧 Settings")
    selected_model = st.selectbox(
    "Detection Model",
    ["XGBoost (Champion)", "Isolation Forest", 
     "Neural Network", "LSTM Autoencoder"]
)

    threshold = st.slider(
        "Detection Threshold",
        min_value = 0.1,
        max_value = 0.9,
        value     = 0.5,
        step      = 0.05,
        help      = "Lower = more sensitive (more detections, more false alarms)"
    )

    st.markdown("---")
    st.markdown("### 📅 Date Range")
    date_options = {
        "Full Dataset"    : (df_raw.index.min(), df_raw.index.max()),
        "Jan 2017"        : (pd.Timestamp('2017-01-01'), pd.Timestamp('2017-01-31')),
        "Q1 2017"         : (pd.Timestamp('2017-01-01'), pd.Timestamp('2017-03-31')),
        "Q2 2017"         : (pd.Timestamp('2017-04-01'), pd.Timestamp('2017-06-30')),
        "Q3 2017"         : (pd.Timestamp('2017-07-01'), pd.Timestamp('2017-09-30')),
    }
    selected_range = st.selectbox("Time Window", list(date_options.keys()))
    start_dt, end_dt = date_options[selected_range]

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown("""
    | Model | F1 | AUC |
    |---|---|---|
    | XGBoost | **0.9935** | **1.0000** |
    | Neural Net | 0.6609 | 0.9996 |
    | Iso Forest | — | — |
    | LSTM AE | 0.0406 | — |
    """)

    st.markdown("---")
    st.markdown("**Dataset:** Tétouan City, Morocco")
    st.markdown("**Records:** 52,416")
    st.markdown("**Features:** 39 engineered")
    st.markdown("**Outage rate:** 0.727%")


# ══════════════════════════════════════════════════════════════
# GENERATE PREDICTIONS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def get_predictions(model_name, thresh):
    if model_name == "XGBoost (Champion)":
        probs = xgb_model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= thresh).astype(int)

    elif model_name == "Isolation Forest":
        raw   = iso_model.predict(X_scaled)
        preds = (raw == -1).astype(int)
        probs = (-iso_model.score_samples(X_scaled) -
                 (-iso_model.score_samples(X_scaled)).min())
        probs = probs / probs.max()

    elif model_name == "Neural Network":
        if nn_model is not None:
            probs = nn_model.predict(X_scaled, verbose=0).flatten()
            preds = (probs >= thresh).astype(int)
        else:
            probs = np.zeros(len(X_scaled))
            preds = np.zeros(len(X_scaled), dtype=int)

    # ADD THIS BLOCK
    elif model_name == "LSTM Autoencoder":
        from sklearn.preprocessing import MinMaxScaler
        lstm_info    = joblib.load('lstm_ae_results.pkl')
        lstm_scaler  = lstm_info['scaler']
        window       = lstm_info['window']
        features     = lstm_info['features']

        # Build sequences
        X_lstm_raw  = df_raw[features].values
        X_lstm_norm = lstm_scaler.transform(X_lstm_raw)

        # Compute reconstruction error for each point
        recon_errors_full = np.zeros(len(X_scaled))
        X_seq_all = np.array([
            X_lstm_norm[i:i+window]
            for i in range(len(X_lstm_norm) - window)
        ])
        X_pred_all   = lstm_ae.predict(X_seq_all, verbose=0)
        errors       = np.mean(
            np.square(X_seq_all - X_pred_all), axis=(1, 2)
        )
        # Map errors back to original indices
        recon_errors_full[window:] = errors
        recon_errors_full[:window] = errors[0]

        # Normalize to 0-1 for probability display
        probs = (recon_errors_full - recon_errors_full.min()) / \
                (recon_errors_full.max() - recon_errors_full.min())
        preds = (recon_errors_full >= lstm_thresh).astype(int)

    return preds, probs

preds, probs = get_predictions(selected_model, threshold)

# Filter to selected date range
mask        = (df_raw.index >= start_dt) & (df_raw.index <= end_dt)
df_window   = df_raw[mask]
y_window    = y_true[mask].values
preds_window= preds[mask]
probs_window= probs[mask]
dates_window= df_raw.index[mask]


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# ⚡ Power Outage Detection System")
st.markdown(
    f"**Model:** `{selected_model}` &nbsp;|&nbsp; "
    f"**Window:** `{selected_range}` &nbsp;|&nbsp; "
    f"**Threshold:** `{threshold}`"
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════
# ROW 1 — KPI CARDS
# ══════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)

total_records  = len(df_window)
detected       = int(preds_window.sum())
true_outages   = int(y_window.sum())
correct        = int(((preds_window == 1) & (y_window == 1)).sum())
false_alarms   = int(((preds_window == 1) & (y_window == 0)).sum())

with c1:
    st.metric("📋 Total Records",  f"{total_records:,}")
with c2:
    st.metric("🔴 Detected Outages", f"{detected:,}")
with c3:
    st.metric("✅ True Outages",    f"{true_outages:,}")
with c4:
    st.metric("🎯 Correct Detections", f"{correct:,}")
with c5:
    st.metric("⚠️ False Alarms",   f"{false_alarms:,}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════
# ROW 2 — MAIN TIME SERIES CHART
# ══════════════════════════════════════════════════════════════
st.markdown("### 📈 Power Consumption — Live Detection View")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    subplot_titles=("Zone 1 Power Consumption with Detected Outages",
                    "Outage Probability Score")
)

# Zone 1 line
fig.add_trace(go.Scatter(
    x    = dates_window,
    y    = df_window['zone1'],
    mode = 'lines',
    name = 'Zone 1 Power',
    line = dict(color='#4fc3f7', width=0.8),
    opacity = 0.8,
), row=1, col=1)

# True outages (ground truth)
true_mask = y_window == 1
if true_mask.sum() > 0:
    fig.add_trace(go.Scatter(
        x    = dates_window[true_mask],
        y    = df_window['zone1'].values[true_mask],
        mode = 'markers',
        name = 'True Outage',
        marker = dict(color='red', size=6, symbol='circle'),
    ), row=1, col=1)

# Detected outages
det_mask = preds_window == 1
if det_mask.sum() > 0:
    fig.add_trace(go.Scatter(
        x    = dates_window[det_mask],
        y    = df_window['zone1'].values[det_mask],
        mode = 'markers',
        name = 'Detected',
        marker = dict(color='orange', size=5,
                      symbol='triangle-up'),
    ), row=1, col=1)

# Probability score
fig.add_trace(go.Scatter(
    x    = dates_window,
    y    = probs_window,
    mode = 'lines',
    name = 'Outage Probability',
    line = dict(color='#ff9800', width=1),
    fill = 'tozeroy',
    fillcolor = 'rgba(255,152,0,0.15)',
), row=2, col=1)

# Threshold line
fig.add_hline(
    y         = threshold,
    line_dash = "dash",
    line_color= "red",
    annotation_text = f"Threshold: {threshold}",
    row=2, col=1
)

fig.update_layout(
    height      = 550,
    template    = 'plotly_dark',
    showlegend  = True,
    legend      = dict(orientation='h', y=1.05),
    margin      = dict(l=0, r=0, t=40, b=0),
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
)
fig.update_xaxes(showgrid=True, gridcolor='#2e3250')
fig.update_yaxes(showgrid=True, gridcolor='#2e3250')

st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ROW 3 — THREE ZONE COMPARISON
# ══════════════════════════════════════════════════════════════
st.markdown("### 🏭 All Three Zones")

fig2 = go.Figure()
zone_colors = {'zone1': '#4fc3f7', 'zone2': '#ff9800', 'zone3': '#66bb6a'}

for zone, color in zone_colors.items():
    fig2.add_trace(go.Scatter(
        x    = dates_window,
        y    = df_window[zone],
        mode = 'lines',
        name = zone.replace('zone', 'Zone '),
        line = dict(color=color, width=0.8),
        opacity = 0.85,
    ))

# Shade detected outage regions
if det_mask.sum() > 0:
    for i, (is_det, date) in enumerate(
            zip(det_mask, dates_window)):
        if is_det:
            fig2.add_vrect(
                x0          = date,
                x1          = date + pd.Timedelta(minutes=20),
                fillcolor   = "red",
                opacity     = 0.15,
                line_width  = 0,
            )

fig2.update_layout(
    height        = 300,
    template      = 'plotly_dark',
    showlegend    = True,
    legend        = dict(orientation='h'),
    margin        = dict(l=0, r=0, t=20, b=0),
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    xaxis         = dict(showgrid=True, gridcolor='#2e3250'),
    yaxis         = dict(showgrid=True, gridcolor='#2e3250'),
)
st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ROW 4 — ANALYSIS CHARTS
# ══════════════════════════════════════════════════════════════
st.markdown("### 📊 Detection Analysis")
col1, col2, col3 = st.columns(3)

with col1:
    # Confusion breakdown
    tp = correct
    fp = false_alarms
    fn = int(((preds_window == 0) & (y_window == 1)).sum())
    tn = int(((preds_window == 0) & (y_window == 0)).sum())

    fig3 = go.Figure(go.Pie(
        labels = ['True Positive', 'False Positive',
                  'False Negative', 'True Negative'],
        values = [tp, fp, fn, tn],
        marker = dict(colors=['#00d4aa','#ff9800','#ff4b4b','#4fc3f7']),
        hole   = 0.5,
        textinfo = 'percent+label',
        textfont_size = 10,
    ))
    fig3.update_layout(
        title         = 'Detection Breakdown',
        height        = 280,
        template      = 'plotly_dark',
        showlegend    = False,
        margin        = dict(l=0, r=0, t=35, b=0),
        paper_bgcolor = 'rgba(0,0,0,0)',
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Outages by hour in window
    df_window_copy         = df_window.copy()
    df_window_copy['hour'] = df_window_copy.index.hour
    df_window_copy['det']  = preds_window

    hourly_det = df_window_copy.groupby('hour')['det'].sum()

    fig4 = go.Figure(go.Bar(
        x           = hourly_det.index,
        y           = hourly_det.values,
        marker_color= '#ff9800',
        opacity     = 0.85,
    ))
    fig4.update_layout(
        title         = 'Detections by Hour',
        height        = 280,
        template      = 'plotly_dark',
        margin        = dict(l=0, r=0, t=35, b=0),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        xaxis         = dict(title='Hour', showgrid=False),
        yaxis         = dict(title='Count', showgrid=True,
                             gridcolor='#2e3250'),
    )
    st.plotly_chart(fig4, use_container_width=True)

with col3:
    # Probability distribution
    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(
        x       = probs_window[y_window == 0],
        name    = 'Normal',
        opacity = 0.7,
        marker_color = '#4fc3f7',
        nbinsx  = 50,
        histnorm= 'probability density',
    ))
    fig5.add_trace(go.Histogram(
        x       = probs_window[y_window == 1],
        name    = 'Outage',
        opacity = 0.8,
        marker_color = '#ff4b4b',
        nbinsx  = 30,
        histnorm= 'probability density',
    ))
    fig5.add_vline(
        x           = threshold,
        line_dash   = "dash",
        line_color  = "yellow",
        annotation_text = f"Threshold"
    )
    fig5.update_layout(
        title         = 'Probability Distribution',
        height        = 280,
        template      = 'plotly_dark',
        barmode       = 'overlay',
        margin        = dict(l=0, r=0, t=35, b=0),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        showlegend    = True,
        legend        = dict(x=0.7, y=0.95),
        xaxis         = dict(showgrid=True, gridcolor='#2e3250'),
        yaxis         = dict(showgrid=True, gridcolor='#2e3250'),
    )
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# ROW 5 — LIVE PREDICTION PANEL
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔮 Live Prediction — Enter Your Own Reading")
st.markdown("Manually input sensor readings and get an instant outage prediction.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**⚡ Power Readings (kW)**")
        zone1_in = st.number_input("Zone 1", value=30000.0,
                                    min_value=0.0, max_value=100000.0)
        zone2_in = st.number_input("Zone 2", value=20000.0,
                                    min_value=0.0, max_value=100000.0)
        zone3_in = st.number_input("Zone 3", value=15000.0,
                                    min_value=0.0, max_value=100000.0)

    with c2:
        st.markdown("**🌤️ Weather Conditions**")
        temp_in     = st.number_input("Temperature (°C)", value=20.0,
                                       min_value=-10.0, max_value=50.0)
        humidity_in = st.number_input("Humidity (%)",    value=60.0,
                                       min_value=0.0, max_value=100.0)
        wind_in     = st.number_input("Wind Speed (km/h)", value=10.0,
                                       min_value=0.0, max_value=100.0)

    with c3:
        st.markdown("**🕐 Time Info**")
        hour_in = st.slider("Hour of Day", 0, 23, 12)
        day_in  = st.selectbox("Day of Week",
                                ['Monday','Tuesday','Wednesday',
                                 'Thursday','Friday','Saturday','Sunday'])
        month_in= st.selectbox("Month",
                                ['Jan','Feb','Mar','Apr','May','Jun',
                                 'Jul','Aug','Sep','Oct','Nov','Dec'])

    submitted = st.form_submit_button("⚡ Predict Outage",
                                       use_container_width=True)

if submitted:
    day_map   = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
                 'Friday':4,'Saturday':5,'Sunday':6}
    month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                 'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

    # Build a feature row matching training features
    # Use dataset means for rolling/lag features
    means = X_hard.mean()

    input_dict = {}
    for col in feature_cols:
        input_dict[col] = float(means[col])

    # Override with actual inputs
    input_dict['zone1']      = zone1_in
    input_dict['zone2']      = zone2_in
    input_dict['zone3']      = zone3_in
    input_dict['temp']       = temp_in
    input_dict['humidity']   = humidity_in
    input_dict['wind_speed'] = wind_in
    input_dict['hour']       = hour_in
    input_dict['dayofweek']  = day_map[day_in]
    input_dict['month']      = month_map[month_in]
    input_dict['is_weekend'] = 1 if day_map[day_in] >= 5 else 0
    input_dict['is_night']   = 1 if (hour_in < 6 or hour_in >= 22) else 0

    # Rebuild interaction features
    input_dict['z1_z2_ratio']   = zone1_in / (zone2_in + 1e-6)
    input_dict['z1_z3_ratio']   = zone1_in / (zone3_in + 1e-6)
    input_dict['all_zones_sum'] = zone1_in + zone2_in + zone3_in
    input_dict['zone_std']      = np.std([zone1_in, zone2_in, zone3_in])
    input_dict['temp_x_hour']   = temp_in * hour_in
    input_dict['wind_x_z1']     = wind_in * zone1_in
    input_dict['humidity_x_z1'] = humidity_in * zone1_in

    # Build row in correct feature order
    all_cols   = list(X_hard.columns)
    input_row  = np.array([[input_dict.get(c, float(means.get(c, 0)))
                            for c in all_cols]])
    input_scaled = scaler.transform(input_row)

    # Predict
    prob = float(xgb_model.predict_proba(input_scaled)[0][1])
    pred = int(prob >= threshold)

    st.markdown("---")
    r1, r2, r3 = st.columns([1, 2, 1])

    with r2:
        if pred == 1:
            st.markdown(f"""
            <div class="outage-alert">
                🚨 OUTAGE DETECTED<br>
                <span style="font-size:2.5rem;">{prob*100:.1f}%</span><br>
                <span style="font-size:0.9rem;">outage probability</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="normal-alert">
                ✅ NORMAL OPERATION<br>
                <span style="font-size:2.5rem;">{prob*100:.1f}%</span><br>
                <span style="font-size:0.9rem;">outage probability</span>
            </div>
            """, unsafe_allow_html=True)

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = prob * 100,
            title = {'text': "Outage Probability (%)"},
            gauge = {
                'axis'  : {'range': [0, 100]},
                'bar'   : {'color': '#ff4b4b' if pred else '#00d4aa'},
                'steps' : [
                    {'range': [0,   30],  'color': '#1e3a2f'},
                    {'range': [30,  70],  'color': '#3a2e1e'},
                    {'range': [70, 100],  'color': '#3a1e1e'},
                ],
                'threshold': {
                    'line' : {'color': 'yellow', 'width': 3},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            },
            number = {'suffix': '%', 'font': {'size': 40}}
        ))
        fig_gauge.update_layout(
            height        = 280,
            template      = 'plotly_dark',
            margin        = dict(l=20, r=20, t=40, b=20),
            paper_bgcolor = 'rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Feature contribution table
    st.markdown("**Input Summary:**")
    summary = pd.DataFrame({
        'Feature'  : ['Zone 1', 'Zone 2', 'Zone 3',
                      'Temp', 'Humidity', 'Wind', 'Hour'],
        'Value'    : [zone1_in, zone2_in, zone3_in,
                      temp_in,  humidity_in, wind_in, hour_in],
        'Unit'     : ['kW', 'kW', 'kW', '°C', '%', 'km/h', 'h']
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# ROW 6 — MODEL PERFORMANCE CARDS
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🏆 Model Performance Summary")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.9935</div>
        <div class="metric-label">XGBoost F1 Score</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">1.0000</div>
        <div class="metric-label">XGBoost ROC-AUC</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">39</div>
        <div class="metric-label">Engineered Features</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">6</div>
        <div class="metric-label">Models Evaluated</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════
# ROW 7 — DATASET STATS TABLE
# ══════════════════════════════════════════════════════════════
st.markdown("### 📋 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    stats = pd.DataFrame({
        'Metric'   : ['Total Records', 'Date Range',
                      'Outage Records', 'Outage Rate',
                      'Zones', 'Features'],
        'Value'    : [
            f"{len(df_raw):,}",
            f"{df_raw.index.min().date()} → {df_raw.index.max().date()}",
            f"{int(y_true.sum()):,}",
            f"{y_true.mean()*100:.3f}%",
            "3 (Zone 1, 2, 3)",
            "39 engineered"
        ]
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)

with col2:
    # Zone correlation
    corr = df_raw[['zone1','zone2','zone3']].corr()
    fig6 = px.imshow(
        corr,
        text_auto = '.3f',
        color_continuous_scale = 'RdBu_r',
        title = "Zone Correlation Matrix",
        template= 'plotly_dark',
    )
    fig6.update_layout(
        height        = 220,
        margin        = dict(l=0, r=0, t=35, b=0),
        paper_bgcolor = 'rgba(0,0,0,0)',
    )
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8892b0; font-size:0.85rem;'>"
    "⚡ Power Outage Detection System &nbsp;|&nbsp; "
    "Data Mining Project &nbsp;|&nbsp; "
    "BTech Final Year &nbsp;|&nbsp; "
    "Models: XGBoost · Neural Network · Isolation Forest"
    "</div>",
    unsafe_allow_html=True
)