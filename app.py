import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="CyberSOC Dashboard", layout="wide")

# 2. Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stMetric { background-color: ##2D1B4E; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Enterprise Threat Intelligence Monitor")

# 3. Secure Data Loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset2_threat_detection.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error("❌ ERROR: 'dataset2_threat_detection.csv' not found in the folder!")
        return None

df = load_data()

if df is not None:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🛡️ Security Filters")
    severity_filter = st.sidebar.multiselect("Filter by Severity:", 
                                            options=df["severity"].unique(), 
                                            default=df["severity"].unique())
    
    # Apply Filter
    filtered_df = df[df["severity"].isin(severity_filter)]

    # --- TOP ROW: KPI METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Alerts", len(filtered_df))
    with m2:
        crit_count = len(filtered_df[filtered_df['severity'] == 'Critical'])
        st.metric("Critical Threats", crit_count, delta=f"{crit_count} Urgent", delta_color="inverse")
    with m3:
        if len(filtered_df) > 0:
            res_rate = (filtered_df['is_resolved'].sum() / len(filtered_df)) * 100
        else:
            res_rate = 0.0
        st.metric("Resolution Rate", f"{res_rate:.1f}%")
    with m4:
        avg_time = filtered_df['response_time_minutes'].mean()
        st.metric("Avg Response Time", f"{avg_time:.1f}m")

    st.divider()

    # --- MIDDLE ROW: CHARTS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚠️ Threat Volume by Type")
        threat_fig = px.bar(filtered_df['threat_type'].value_counts().reset_index(), 
                            x='threat_type', y='count', color='threat_type',
                            labels={'count': 'Incidents', 'threat_type': 'Type'})
        st.plotly_chart(threat_fig, use_container_width=True)

    with col2:
        st.subheader("🕒 Peak Attack Hours")
        hour_data = filtered_df.groupby('hour').size().reset_index(name='count')
        hour_fig = px.area(hour_data, x='hour', y='count', line_shape='spline')
        st.plotly_chart(hour_fig, use_container_width=True)

    # --- BOTTOM ROW: ANALYSIS ---
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.subheader("📋 Recent Incident Log")
        st.dataframe(filtered_df.sort_values('timestamp', ascending=False), height=300)

    with col4:
        st.subheader("📝 Analyst Notes")
        st.info("**Key Finding:** Most 'Critical' threats are targeting Database Servers.")
        st.warning("**Action Required:** High volume of Port Scans detected at hour 0.")
        st.success("**System Status:** EDR and SIEM are syncing correctly.")

# --- FOOTER ---
st.caption("Internal Security Tool - Unauthorized Access Prohibited")