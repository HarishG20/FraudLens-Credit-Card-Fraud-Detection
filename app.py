"""
FraudLens: AI-Powered Credit Card Fraud Detection Using Deep Autoencoders
Production-grade modern Streamlit application featuring dark navy fintech aesthetics,
real-time anomaly detection, interactive Plotly analytics, batch processing, and model auditing.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.autoencoder import load_trained_autoencoder, get_model_summary
from src.anomaly_detection import (
    compute_reconstruction_error,
    classify_anomalies,
    load_threshold_artifact,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudLens | AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Main Application Theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background Gradients */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0F172A 0%, #0B0F19 90%);
    color: #F8FAFC;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #070B14;
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}

/* Metric & Value Cards */
.fl-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.fl-card:hover {
    border-color: rgba(99, 102, 241, 0.45);
    transform: translateY(-2px);
}

.fl-card-title {
    font-size: 0.825rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}

.fl-card-val {
    font-size: 1.85rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.02em;
}

.fl-card-sub {
    font-size: 0.775rem;
    color: #64748B;
    margin-top: 0.35rem;
}

/* Status Badges */
.badge-normal {
    background: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
}

.badge-fraud {
    background: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
}

/* Code & Mono */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55rem 1.25rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    box-shadow: 0 6px 20px 0 rgba(59, 130, 246, 0.5);
    transform: translateY(-1px);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED RESOURCE LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Deep Autoencoder...")
def get_model():
    if not config.MODEL_PATH.exists():
        return None
    return load_trained_autoencoder(config.MODEL_PATH)


@st.cache_resource(show_spinner="Loading Preprocessing Scaler...")
def get_scaler():
    if not config.SCALER_PATH.exists():
        return None
    return joblib.load(config.SCALER_PATH)


@st.cache_resource(show_spinner="Loading Threshold Calibration...")
def get_threshold_info():
    if not config.THRESHOLD_PATH.exists():
        return None
    return load_threshold_artifact(config.THRESHOLD_PATH)


@st.cache_data(show_spinner="Loading Benchmark Metrics...")
def get_benchmark_metrics():
    metrics_path = config.OUTPUTS_DIR / "evaluation_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM HEALTH
# -----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;">
                <div style="background: linear-gradient(135deg, #3B82F6, #8B5CF6); border-radius: 10px; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                    🛡️
                </div>
                <div>
                    <h2 style="margin: 0; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em;">FraudLens</h2>
                    <p style="margin: 0; font-size: 0.75rem; color: #94A3B8;">Deep Autoencoder Surveillance</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        page = st.radio(
            "Navigation",
            [
                "📊 Overview Dashboard",
                "📂 Batch Transaction Analysis",
                "🎯 Single Transaction Inspection",
                "📈 Model Performance & Auditing",
                "📖 Architecture & Technical Specs",
            ],
            index=0,
        )
        st.markdown("---")

        # System Health Card
        model = get_model()
        scaler = get_scaler()
        threshold_info = get_threshold_info()

        is_ready = (model is not None) and (scaler is not None) and (threshold_info is not None)

        st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;'>Engine Status</p>", unsafe_allow_html=True)
        if is_ready:
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 15px;">
                    <span style="color: #10B981; font-weight: 700; font-size: 0.85rem;">● Engine Operational</span>
                    <p style="margin: 4px 0 0 0; font-size: 0.725rem; color: #94A3B8;">Keras 3.x Deep Autoencoder Ready</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            thresh_val = threshold_info.get("threshold", 0.0)
            st.caption(f"Active Threshold: `{thresh_val:.5f}`")
        else:
            st.error("Engine Incomplete. Please run `python -m src.train`.")

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #64748B; line-height: 1.4;">
                <strong>FraudLens Core</strong><br>
                Unsupervised Anomaly Detection<br>
                TensorFlow / Keras Functional API<br>
                v1.0.0 • Academic Production
            </div>
            """,
            unsafe_allow_html=True,
        )
        return page


# -----------------------------------------------------------------------------
# 4. PAGE 1: OVERVIEW DASHBOARD
# -----------------------------------------------------------------------------
def render_overview():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Executive Surveillance Dashboard
            </h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
                Real-time synthesis of autoencoder reconstruction fidelity and anomaly rates on audited test transactions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = get_benchmark_metrics()
    threshold_info = get_threshold_info()

    if not metrics:
        st.warning("Auditing benchmarks are not yet generated. Please execute `python -m src.evaluation`.")
        return

    ae_stats = metrics["autoencoder"]
    cm = ae_stats["confusion_matrix"]
    total_test = ae_stats["test_samples"]
    total_frauds = ae_stats["test_frauds"]
    flagged = cm[0][1] + cm[1][1]
    flagged_pct = (flagged / total_test) * 100.0

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="fl-card">
                <div class="fl-card-title">Test Audited Volume</div>
                <div class="fl-card-val">{total_test:,}</div>
                <div class="fl-card-sub">Untouched held-out test partition</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="fl-card">
                <div class="fl-card-title">Flagged Potential Frauds</div>
                <div class="fl-card-val" style="color: #EF4444;">{flagged:,}</div>
                <div class="fl-card-sub">{flagged_pct:.2f}% of monitored stream</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="fl-card">
                <div class="fl-card-title">True Frauds Captured</div>
                <div class="fl-card-val" style="color: #10B981;">{cm[1][1]}/{total_frauds}</div>
                <div class="fl-card-sub">Recall: {ae_stats['recall']*100:.1f}% sensitivity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="fl-card">
                <div class="fl-card-title">Operating Threshold</div>
                <div class="fl-card-val" style="color: #6366F1;">{ae_stats['threshold']:.4f}</div>
                <div class="fl-card-sub">Calibrated on validation set</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Visualizations
    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        st.markdown("### 📊 Reconstruction Error by Ground Truth")
        error_stats = ae_stats["error_stats"]
        
        # Comparison Bar chart of Mean Reconstruction Error
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Normal Transactions",
            x=["Normal Transactions"],
            y=[error_stats["normal_mean"]],
            marker_color="#10B981",
            error_y=dict(type="data", array=[error_stats["normal_std"]], visible=True),
            text=[f"Mean: {error_stats['normal_mean']:.3f}"],
            textposition="auto",
        ))
        fig_bar.add_trace(go.Bar(
            name="Fraudulent Transactions",
            x=["Fraudulent Transactions"],
            y=[error_stats["fraud_mean"]],
            marker_color="#EF4444",
            error_y=dict(type="data", array=[min(error_stats["fraud_std"], 35.0)], visible=True),
            text=[f"Mean: {error_stats['fraud_mean']:.3f}"],
            textposition="auto",
        ))
        fig_bar.add_hline(
            y=ae_stats["threshold"],
            line_dash="dash",
            line_color="#F8FAFC",
            annotation_text=f"Decision Threshold ({ae_stats['threshold']:.2f})",
            annotation_position="top right",
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#94A3B8", family="Inter"),
            yaxis=dict(title="Reconstruction Error (MSE)", gridcolor="rgba(148, 163, 184, 0.1)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
            margin=dict(l=30, r=30, t=70, b=30),
            height=370,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("### 🎯 Detection Confusion Breakdown")
        labels = ["True Normal (TN)", "False Alarm (FP)", "Missed Fraud (FN)", "Caught Fraud (TP)"]
        values = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
        colors = ["#10B981", "#F59E0B", "#EF4444", "#3B82F6"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.6,
            marker=dict(colors=colors),
            textinfo="label+value",
            textposition="outside",
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC", family="Inter"),
            showlegend=False,
            margin=dict(l=30, r=30, t=40, b=30),
            height=370,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Activity & Summary Table
    st.markdown("### 📋 System Verification Summary")
    summary_df = pd.DataFrame({
        "Metric": [
            "Precision (Positive Predictive Value)",
            "Recall (Fraud Detection Rate)",
            "F1-Score (Harmonic Mean)",
            "Area Under ROC (ROC-AUC)",
            "Average Precision (PR-AUC)",
            "False Positive Rate (FPR)",
        ],
        "Autoencoder Score": [
            f"{ae_stats['precision']*100:.2f}%",
            f"{ae_stats['recall']*100:.2f}%",
            f"{ae_stats['f1_score']:.4f}",
            f"{ae_stats['roc_auc']:.4f}",
            f"{ae_stats['average_precision']:.4f}",
            f"{(cm[0][1] / (cm[0][0] + cm[0][1]))*100:.2f}% ({cm[0][1]:,} false alarms out of {cm[0][0]+cm[0][1]:,} legitimate)",
        ],
        "Significance": [
            "When flagged, how often is it truly fraudulent",
            "Percentage of all real frauds successfully intercepted",
            "Balanced quality metric on highly skewed classes",
            "Global discrimination ability across all thresholds",
            "Gold standard metric for extreme class imbalance",
            "Operational burden on human fraud review teams",
        ]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 5. PAGE 2: BATCH TRANSACTION ANALYSIS
# -----------------------------------------------------------------------------
def render_batch_analysis():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Batch Transaction Audit & Analysis
            </h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
                Upload credit card transactions in CSV format for automated neural reconstruction scoring and threat triage.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = get_model()
    scaler = get_scaler()
    threshold_info = get_threshold_info()

    if model is None or scaler is None or threshold_info is None:
        st.error("System components are missing. Please complete training first.")
        return

    active_threshold = threshold_info["threshold"]

    col_upload, col_sample = st.columns([2, 1.2])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Transaction File (CSV)",
            type=["csv"],
            help="Ensure the CSV contains columns: Time, V1-V28, Amount (30 features).",
        )
    with col_sample:
        st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #94A3B8;'>Or Test with Built-in Data</p>", unsafe_allow_html=True)
        c_btn1, c_btn2 = st.columns(2)
        use_sample_test = c_btn1.button("⚡ Load 500 Samples", use_container_width=True)
        clear_batch = c_btn2.button("🗑️ Clear Batch", use_container_width=True)

    if clear_batch:
        st.session_state.pop("batch_df", None)
        st.session_state.pop("batch_result_df", None)
        st.session_state.pop("batch_source", None)
        st.session_state.pop("last_uploaded_file_id", None)
        st.rerun()

    # Track uploads in session state
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file_id") != file_id:
            try:
                st.session_state["batch_df"] = pd.read_csv(uploaded_file)
                st.session_state["batch_source"] = uploaded_file.name
                st.session_state["last_uploaded_file_id"] = file_id
                st.session_state.pop("batch_result_df", None)
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
                return
    elif use_sample_test:
        if config.SAMPLE_TEST_PATH.exists():
            st.session_state["batch_df"] = pd.read_csv(config.SAMPLE_TEST_PATH)
            st.session_state["batch_source"] = "Built-in Balanced Benchmark Slice (500 Samples)"
        elif config.DATASET_PATH.exists():
            df_full = pd.read_csv(config.DATASET_PATH)
            normal_sample = df_full[df_full[config.TARGET_COLUMN] == 0].sample(400, random_state=42)
            fraud_sample = df_full[df_full[config.TARGET_COLUMN] == 1].sample(min(100, (df_full[config.TARGET_COLUMN] == 1).sum()), random_state=42)
            st.session_state["batch_df"] = pd.concat([normal_sample, fraud_sample]).sample(frac=1.0, random_state=42).reset_index(drop=True)
            st.session_state["batch_source"] = "Built-in Balanced Benchmark Slice"
        else:
            st.warning("Sample dataset not found.")
            return
        st.session_state.pop("batch_result_df", None)

    # Process if batch data exists in session_state
    if "batch_df" in st.session_state and st.session_state["batch_df"] is not None:
        df_to_analyze = st.session_state["batch_df"]
        source_name = st.session_state.get("batch_source", "Uploaded Batch")

        # Compute predictions if not already in session state
        if "batch_result_df" not in st.session_state:
            missing_cols = [c for c in config.FEATURE_COLUMNS if c not in df_to_analyze.columns]
            if missing_cols:
                st.error(f"Missing required columns ({len(missing_cols)}): {missing_cols[:5]}...")
                return

            with st.spinner(f"Analyzing {len(df_to_analyze):,} records from '{source_name}'..."):
                features = df_to_analyze[config.FEATURE_COLUMNS].copy()
                scaled_features = scaler.transform(features)
                reconstructions = model.predict(scaled_features, batch_size=256, verbose=0)
                errors = compute_reconstruction_error(scaled_features, reconstructions)
                preds, labels = classify_anomalies(errors, threshold=active_threshold)

                result_df = df_to_analyze.copy()
                result_df["Reconstruction_Error"] = errors
                result_df["Anomaly_Threshold"] = active_threshold
                result_df["Anomaly_Status"] = labels
                result_df["Predicted_Class"] = preds

                st.session_state["batch_result_df"] = result_df
        else:
            result_df = st.session_state["batch_result_df"]

        flagged_count = int(np.sum(result_df["Predicted_Class"] == 1))
        normal_count = int(np.sum(result_df["Predicted_Class"] == 0))
        flagged_pct = (flagged_count / len(result_df)) * 100.0

        # Summary Metric Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records Processed", f"{len(result_df):,}")
        m2.metric("Flagged as Potential Fraud", f"{flagged_count:,}", delta=f"{flagged_pct:.2f}% of batch", delta_color="inverse")
        m3.metric("Verified Legitimate", f"{normal_count:,}")
        m4.metric("Active Decision Cutoff", f"{active_threshold:.4f}")

        # Interactive filter
        st.markdown("### 🔍 Transaction Inspection Table")
        filter_mode = st.radio(
            "Filter Display Rows",
            ["All Records", "Potential Frauds Only (Flagged)", "Normal Only"],
            horizontal=True,
            key="batch_filter_mode",
        )

        if filter_mode == "Potential Frauds Only (Flagged)":
            display_df = result_df[result_df["Anomaly_Status"] == "Potential Fraud"]
        elif filter_mode == "Normal Only":
            display_df = result_df[result_df["Anomaly_Status"] == "Normal"]
        else:
            display_df = result_df

        st.dataframe(
            display_df[["Time", "Amount", "Reconstruction_Error", "Anomaly_Threshold", "Anomaly_Status"] + [f"V{i}" for i in range(1, 11)]],
            use_container_width=True,
            height=380,
        )

        # Download Report Button
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Audited Report (CSV)",
            data=csv_bytes,
            file_name="FraudLens_Audited_Report.csv",
            mime="text/csv",
            key="download_audited_csv_btn",
        )


# -----------------------------------------------------------------------------
# 6. PAGE 3: INDIVIDUAL TRANSACTION INSPECTION
# -----------------------------------------------------------------------------
def render_single_transaction():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Single Transaction Deep Diagnostics
            </h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
                Inspect individual 30-feature vector inputs, perform neural reconstruction, and dissect per-feature residual error contributions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = get_model()
    scaler = get_scaler()
    threshold_info = get_threshold_info()

    if model is None or scaler is None or threshold_info is None:
        st.error("System components are missing. Please complete training first.")
        return

    active_threshold = threshold_info["threshold"]

    # Sample Presets for quick evaluation
    st.markdown("#### ⚡ Quick Presets")
    c_pre1, c_pre2, c_pre3 = st.columns(3)
    
    preset_choice = st.session_state.get("selected_preset", "normal")

    if c_pre1.button("🟢 Load Typical Legitimate Transaction"):
        st.session_state["selected_preset"] = "normal"
        st.session_state.pop("single_result", None)
        st.rerun()
    if c_pre2.button("🔴 Load Known High-Loss Fraud Transaction"):
        st.session_state["selected_preset"] = "fraud"
        st.session_state.pop("single_result", None)
        st.rerun()
    if c_pre3.button("⚪ Reset to Zero Vector"):
        st.session_state["selected_preset"] = "zero"
        st.session_state.pop("single_result", None)
        st.rerun()

    # Pre-populate sample values based on preset
    preset_choice = st.session_state.get("selected_preset", "normal")
    default_vals = {}
    
    if preset_choice == "fraud":
        # High-loss confirmed fraud transaction (Reconstruction Error = 101.07 >> 16.34)
        default_vals = {
            "Time": 18088.0, "Amount": 1218.89,
            "V1": -12.2240, "V2": 3.8542, "V3": -12.4668, "V4": 9.6483, "V5": -2.7270,
            "V6": -4.4456, "V7": -21.9228, "V8": 0.3208, "V9": -4.4332, "V10": -11.2014,
            "V11": 9.3288, "V12": -13.1049, "V13": 0.8885, "V14": -10.1402, "V15": 0.7135,
            "V16": -10.0987, "V17": -17.5066, "V18": -8.0612, "V19": 1.6069, "V20": -2.1472,
            "V21": -1.1598, "V22": -1.5041, "V23": -19.2543, "V24": 0.5449, "V25": -4.7816,
            "V26": -0.0078, "V27": 3.0524, "V28": -0.7750
        }
    elif preset_choice == "zero":
        default_vals = {col: 0.0 for col in config.FEATURE_COLUMNS}
    else:
        # Typical legitimate transaction
        default_vals = {
            "Time": 1000.0, "Amount": 29.50,
            "V1": 1.191, "V2": 0.266, "V3": 0.166, "V4": 0.448, "V5": 0.060,
            "V6": -0.082, "V7": -0.078, "V8": 0.085, "V9": -0.255, "V10": -0.166,
            "V11": 1.612, "V12": 1.065, "V13": 0.489, "V14": -0.143, "V15": 0.635,
            "V16": 0.463, "V17": -0.114, "V18": -0.183, "V19": -0.145, "V20": -0.069,
            "V21": -0.228, "V22": -0.638, "V23": 0.101, "V24": -0.339, "V25": 0.167,
            "V26": 0.125, "V27": -0.008, "V28": 0.014
        }

    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("single_transaction_form"):
        st.markdown("### 📝 Enter Transaction Parameters (30 Features)")

        # Section 1: Meta
        st.markdown("##### 🕒 Transaction Metadata")
        c_m1, c_m2 = st.columns(2)
        time_val = c_m1.number_input("Time (Seconds from first transaction)", value=float(default_vals.get("Time", 0.0)), step=1.0)
        amount_val = c_m2.number_input("Amount ($ USD)", value=float(default_vals.get("Amount", 0.0)), step=1.0)

        # Section 2: V1 - V10
        st.markdown("##### 🧬 Principal Components V1 to V10")
        v_cols1 = st.columns(5)
        v_cols2 = st.columns(5)
        v_inputs = {}
        for i in range(1, 6):
            v_inputs[f"V{i}"] = v_cols1[i-1].number_input(f"V{i}", value=float(default_vals.get(f"V{i}", 0.0)), format="%.4f")
        for i in range(6, 11):
            v_inputs[f"V{i}"] = v_cols2[i-6].number_input(f"V{i}", value=float(default_vals.get(f"V{i}", 0.0)), format="%.4f")

        # Section 3: V11 - V20
        st.markdown("##### 🧬 Principal Components V11 to V20")
        v_cols3 = st.columns(5)
        v_cols4 = st.columns(5)
        for i in range(11, 16):
            v_inputs[f"V{i}"] = v_cols3[i-11].number_input(f"V{i}", value=float(default_vals.get(f"V{i}", 0.0)), format="%.4f")
        for i in range(16, 21):
            v_inputs[f"V{i}"] = v_cols4[i-16].number_input(f"V{i}", value=float(default_vals.get(f"V{i}", 0.0)), format="%.4f")

        # Section 4: V21 - V28
        st.markdown("##### 🧬 Principal Components V21 to V28")
        v_cols5 = st.columns(4)
        v_cols6 = st.columns(4)
        for i in range(21, 25):
            v_inputs[f"V{i}"] = v_cols5[i-21].number_input(f"V{i}", value=float(default_vals.get(f"V{i}", 0.0)), format="%.4f")
        for i in range(25, 29):
            v_inputs[f"V{i}"] = v_cols6[i-25].number_input(f"V{i}", value=float(default_vals.get(f"V{i}", 0.0)), format="%.4f")

        submitted = st.form_submit_button("⚡ Analyze Transaction Now", use_container_width=True)

    if submitted:
        # Build 1-row DataFrame
        input_data = {"Time": [time_val]}
        for i in range(1, 29):
            input_data[f"V{i}"] = [v_inputs[f"V{i}"]]
        input_data["Amount"] = [amount_val]

        raw_df = pd.DataFrame(input_data)[config.FEATURE_COLUMNS]

        # Scale features
        scaled = scaler.transform(raw_df)
        reconstruction = model.predict(scaled, verbose=0)
        rec_error = float(compute_reconstruction_error(scaled, reconstruction)[0])

        is_fraud = rec_error > active_threshold

        st.session_state["single_result"] = {
            "rec_error": rec_error,
            "is_fraud": is_fraud,
            "active_threshold": active_threshold,
            "scaled": scaled,
            "reconstruction": reconstruction,
        }

    if "single_result" in st.session_state and st.session_state["single_result"] is not None:
        res = st.session_state["single_result"]
        rec_error = res["rec_error"]
        is_fraud = res["is_fraud"]
        active_threshold = res["active_threshold"]
        scaled = res["scaled"]
        reconstruction = res["reconstruction"]

        st.markdown("---")
        st.markdown("### 🔍 Forensic Analysis Results")

        res_col1, res_col2 = st.columns([1, 1.3])

        with res_col1:
            if is_fraud:
                st.markdown(
                    f"""
                    <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #EF4444; border-radius: 12px; padding: 20px; text-align: center;">
                        <span style="font-size: 2.5rem;">⚠️</span>
                        <h2 style="color: #EF4444; margin: 10px 0 5px 0;">POTENTIAL FRAUD DETECTED</h2>
                        <p style="color: #FCA5A5; font-size: 0.9rem;">Reconstruction Error Exceeds Operational Safety Threshold</p>
                        <div style="margin-top: 15px; font-size: 1.1rem; font-weight: 700;">
                            Error: <span style="color: #EF4444;">{rec_error:.4f}</span> > Threshold: <span style="color: #F8FAFC;">{active_threshold:.4f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10B981; border-radius: 12px; padding: 20px; text-align: center;">
                        <span style="font-size: 2.5rem;">✅</span>
                        <h2 style="color: #10B981; margin: 10px 0 5px 0;">NORMAL TRANSACTION</h2>
                        <p style="color: #6EE7B7; font-size: 0.9rem;">Transaction Conforms to Expected Legitimate Patterns</p>
                        <div style="margin-top: 15px; font-size: 1.1rem; font-weight: 700;">
                            Error: <span style="color: #10B981;">{rec_error:.4f}</span> ≤ Threshold: <span style="color: #F8FAFC;">{active_threshold:.4f}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            # Gauge chart of Reconstruction Error vs Threshold
            max_gauge = max(rec_error * 1.3, active_threshold * 2.0, 10.0)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rec_error,
                delta={"reference": active_threshold, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
                title={"text": "Reconstruction Anomaly Score (MSE)", "font": {"size": 14, "color": "#F8FAFC"}},
                gauge={
                    "axis": {"range": [None, max_gauge], "tickcolor": "#94A3B8"},
                    "bar": {"color": "#EF4444" if is_fraud else "#10B981"},
                    "bgcolor": "rgba(15, 23, 42, 0.8)",
                    "steps": [
                        {"range": [0, active_threshold], "color": "rgba(16, 185, 129, 0.2)"},
                        {"range": [active_threshold, max_gauge], "color": "rgba(239, 68, 68, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#F8FAFC", "width": 3},
                        "thickness": 0.8,
                        "value": active_threshold,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#F8FAFC", "family": "Inter"},
                height=320,
                margin=dict(l=30, r=30, t=80, b=25),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature Decomposition: Which features caused the reconstruction error?
        st.markdown("#### 🔬 Per-Feature Residual Decomposition")
        per_feature_sq_err = np.square(scaled[0] - reconstruction[0])
        top_err_indices = np.argsort(per_feature_sq_err)[::-1][:8]
        top_features = [config.FEATURE_COLUMNS[idx] for idx in top_err_indices]
        top_errors = [float(per_feature_sq_err[idx]) for idx in top_err_indices]

        fig_contrib = go.Figure(go.Bar(
            x=top_features,
            y=top_errors,
            marker_color="#8B5CF6",
            text=[f"{e:.2f}" for e in top_errors],
            textposition="auto",
        ))
        fig_contrib.update_layout(
            title="Top 8 Features Deviating from Legitimate Autoencoder Manifold",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            font=dict(color="#94A3B8", family="Inter"),
            yaxis=dict(title="Feature Squared Error ((x - x̂)²)", gridcolor="rgba(148, 163, 184, 0.1)"),
            margin=dict(l=30, r=30, t=65, b=30),
            height=320,
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        # Scientific explanation of detection logic
        st.markdown(
            f"""
            > **Diagnostic Explanation**: The Deep Autoencoder was trained exclusively on 199,020 legitimate transactions.
            > When exposed to this input, its reconstructed output differed by an average squared error of **`{rec_error:.4f}`** across the 30 dimensions.
            > Because this {'**exceeds**' if is_fraud else '**remains below**'} the operational threshold of **`{active_threshold:.4f}`**,
            > this transaction is classified as **`{'Potential Fraud' if is_fraud else 'Normal'}`**.
            """
        )


# -----------------------------------------------------------------------------
# 7. PAGE 4: MODEL PERFORMANCE & AUDITING
# -----------------------------------------------------------------------------
def render_model_performance():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Model Performance & Academic Audit
            </h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
                Comprehensive evaluation metrics on 42,722 unseen test transactions and side-by-side benchmark comparison against Isolation Forest.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = get_benchmark_metrics()
    if not metrics:
        st.warning("No evaluation metrics found. Please execute `python -m src.evaluation`.")
        return

    ae_stats = metrics["autoencoder"]
    iso_stats = metrics.get("isolation_forest_baseline", {})

    # Top Benchmark Comparison Table
    st.markdown("### 🏆 Deep Autoencoder vs Baseline Comparison")
    comp_df = pd.DataFrame({
        "Evaluation Metric": [
            "Model Paradigm",
            "Precision (Positive Predictive Value)",
            "Recall (Fraud Sensitivity)",
            "F1-Score (Harmonic Mean)",
            "ROC-AUC (Global Discrimination)",
            "Average Precision (PR-AUC)",
        ],
        "FraudLens Deep Autoencoder (Primary)": [
            "Deep Neural Bottleneck (Unsupervised)",
            f"{ae_stats['precision']*100:.2f}%",
            f"{ae_stats['recall']*100:.2f}%",
            f"{ae_stats['f1_score']:.4f}",
            f"{ae_stats['roc_auc']:.4f}",
            f"{ae_stats['average_precision']:.4f}",
        ],
        "Isolation Forest (Baseline)": [
            "Tree-Based Random Partitioning",
            f"{iso_stats.get('precision', 0)*100:.2f}%",
            f"{iso_stats.get('recall', 0)*100:.2f}%",
            f"{iso_stats.get('f1_score', 0):.4f}",
            f"{iso_stats.get('roc_auc', 0):.4f}",
            f"{iso_stats.get('average_precision', 0):.4f}",
        ],
        "Performance Differential": [
            "Neural Manifold Learning",
            f"+{(ae_stats['precision'] - iso_stats.get('precision', 0))*100:.2f}% improvement",
            f"+{(ae_stats['recall'] - iso_stats.get('recall', 0))*100:.2f}% improvement",
            f"+{(ae_stats['f1_score'] - iso_stats.get('f1_score', 0)):.4f} improvement",
            f"+{(ae_stats['roc_auc'] - iso_stats.get('roc_auc', 0)):.4f}",
            f"+{(ae_stats['average_precision'] - iso_stats.get('average_precision', 0)):.4f} (3.6x higher)",
        ]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Diagnostic Visualizations Gallery")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Training & Validation Dynamics",
        "🎯 Confusion Matrix",
        "📈 Precision-Recall Curve",
        "⚡ ROC Curve",
        "🔬 Reconstruction Error Distribution",
    ])

    with tab1:
        if config.TRAINING_LOSS_PLOT.exists():
            st.image(str(config.TRAINING_LOSS_PLOT), caption="Autoencoder Training vs Validation Loss (Normal Data)", use_container_width=True)
        else:
            st.info("Training loss plot not found.")

    with tab2:
        if config.CONFUSION_MATRIX_PLOT.exists():
            st.image(str(config.CONFUSION_MATRIX_PLOT), caption="Held-Out Test Set Confusion Matrix (42,722 Transactions)", use_container_width=True)
        else:
            st.info("Confusion matrix plot not found.")

    with tab3:
        if config.PRECISION_RECALL_PLOT.exists():
            st.image(str(config.PRECISION_RECALL_PLOT), caption="Precision-Recall Curve with Average Precision (AP = 0.3403)", use_container_width=True)
        else:
            st.info("Precision-Recall plot not found.")

    with tab4:
        if config.ROC_CURVE_PLOT.exists():
            st.image(str(config.ROC_CURVE_PLOT), caption="Receiver Operating Characteristic Curve (ROC-AUC = 0.9424)", use_container_width=True)
        else:
            st.info("ROC plot not found.")

    with tab5:
        if config.RECONSTRUCTION_ERROR_PLOT.exists():
            st.image(str(config.RECONSTRUCTION_ERROR_PLOT), caption="Reconstruction Error Density and Scatter Comparison", use_container_width=True)
        else:
            st.info("Reconstruction error plot not found.")


# -----------------------------------------------------------------------------
# 8. PAGE 5: ABOUT THE PROJECT & ARCHITECTURE
# -----------------------------------------------------------------------------
def render_about():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h1 style="font-size: 2.1rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                FraudLens Architecture & Technical Specifications
            </h1>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 4px;">
                Formal documentation, mathematical formulation, and deep neural autoencoder specifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1.2, 1.0])

    with col_a:
        st.markdown(
            """
            ### 📌 1. Problem Statement
            Credit card fraud represents billions in global financial losses annually. Traditional supervised learning
            struggles because:
            1. **Extreme Class Imbalance**: Fraud accounts for only **0.172%** (492 out of 284,807 transactions).
            2. **Concept Drift & Novelty**: Fraudsters continuously invent new evasion tactics that supervised models trained on past fraud patterns fail to detect.
            3. **Label Latency**: Fraud chargebacks often take weeks to surface.

            ### 💡 2. The Autoencoder Solution
            FraudLens uses an **unsupervised Deep Autoencoder** neural network:
            - The model is trained **exclusively on legitimate transactions**.
            - It learns the compressed mathematical manifold of normal financial behavior.
            - When an anomalous or fraudulent transaction passes through the network, the autoencoder fails to reconstruct it accurately.
            - The resulting high **Reconstruction Error (MSE)** flags the transaction as a **Potential Fraud**.
            """
        )

        st.markdown(
            """
            ### 📐 3. Mathematical Formulation
            $$\\text{Reconstruction Error}_i = \\frac{1}{D} \\sum_{j=1}^{D} \\left( x_{ij} - \\hat{x}_{ij} \\right)^2$$
            Where:
            - $D = 30$ features ($Time, V_1 \\dots V_{28}, Amount$)
            - $x_{ij}$ is the normalized input feature
            - $\\hat{x}_{ij}$ is the autoencoder's reconstructed output
            - If $\\text{Error}_i > \\theta$ (threshold), classify as **Potential Fraud**.
            """
        )

    with col_b:
        st.markdown(
            """
            ### 🧠 4. Neural Network Architecture
            ```
            Input Layer:        30 Neurons (Time, V1-V28, Amount)
            │
            ├── Encoder L1:     Dense(16, activation='relu')
            │
            ├── Encoder L2:     Dense(8, activation='relu')
            │
            ├── Bottleneck:     Dense(4, activation='relu')  <-- Latent Space
            │
            ├── Decoder L1:     Dense(8, activation='relu')
            │
            ├── Decoder L2:     Dense(16, activation='relu')
            │
            Output Layer:       Dense(30, activation='linear')
            ```
            - **Total Trainable Parameters**: 1,362
            - **Optimizer**: Adam (learning rate = 0.001)
            - **Loss Function**: Mean Squared Error (MSE)
            - **Batch Size**: 256
            - **Early Stopping**: Patience = 5, Restoring Best Weights
            """
        )

        st.markdown(
            """
            ### 🛠️ 5. Technology Stack
            - **Deep Learning**: TensorFlow 2.x, Keras 3.x
            - **Data Manipulation**: Pandas, NumPy
            - **Preprocessing & Metrics**: Scikit-Learn
            - **Serialization**: Joblib, Keras native (`.keras`)
            - **Visualization**: Matplotlib, Seaborn, Plotly Express/Graph Objects
            - **Web Application**: Streamlit with Custom Fintech CSS
            """
        )

    st.markdown("---")
    st.markdown(
        """
        ### 🔮 6. Limitations & Future Scope
        - **Feature Explainability**: Incorporating SHAP/Integrated Gradients for explaining latent space transformations.
        - **Sequential Modeling**: Integrating LSTM or Temporal Graph Convolutional Networks (GCN) to model time-series velocity between sequential transactions.
        - **Dynamic Thresholding**: Adapting thresholds dynamically based on transaction amount tiers or geographic zones.
        """
    )


# -----------------------------------------------------------------------------
# 9. MAIN ROUTER
# -----------------------------------------------------------------------------
def main():
    selected_page = render_sidebar()

    if selected_page == "📊 Overview Dashboard":
        render_overview()
    elif selected_page == "📂 Batch Transaction Analysis":
        render_batch_analysis()
    elif selected_page == "🎯 Single Transaction Inspection":
        render_single_transaction()
    elif selected_page == "📈 Model Performance & Auditing":
        render_model_performance()
    elif selected_page == "📖 Architecture & Technical Specs":
        render_about()


if __name__ == "__main__":
    main()
