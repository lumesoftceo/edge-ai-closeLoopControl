import streamlit as st
import psutil
import pandas as pd
import numpy as np
import time
import pickle
import os
import base64
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Cross-platform sound support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# --- CONFIGURATION ---
MODEL_FILE = 'sentinai_model.pkl'
SCALER_FILE = 'sentinai_scaler.pkl'
WINDOW_SIZE = 5  # Seconds to wait before confirming anomaly
CONTAMINATION = 0.05 

st.set_page_config(page_title="SentinAI: Advanced Guardian", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'history_cpu' not in st.session_state:
    st.session_state['history_cpu'] = deque(maxlen=60)
if 'anomaly_buffer' not in st.session_state:
    st.session_state['anomaly_buffer'] = deque(maxlen=WINDOW_SIZE)
# Session-based model storage (unique per user session)
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'sound_enabled' not in st.session_state:
    st.session_state['sound_enabled'] = True
# Baseline normal usage values
if 'baseline_cpu' not in st.session_state:
    st.session_state['baseline_cpu'] = None
if 'baseline_ram' not in st.session_state:
    st.session_state['baseline_ram'] = None
if 'baseline_disk' not in st.session_state:
    st.session_state['baseline_disk'] = None

# --- HELPER: IDENTIFY TOP PROCESS ---
def get_top_process():
    try:
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                pinfo = proc.info
                processes.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        sorted_procs = sorted(processes, key=lambda p: p['cpu_percent'], reverse=True)
        if sorted_procs:
            return sorted_procs[0]
        return None
    except Exception:
        return None

# --- HELPER: GET METRICS ---
def get_metrics():
    cpu = psutil.cpu_percent(interval=0.0)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_io_counters()
    disk_activity = (disk.read_bytes + disk.write_bytes) / 1024 / 1024 # MB
    return [cpu, ram, disk_activity]

# --- UI HEADER ---
st.title("🛡️ SentinAI: Process-Aware Resource Governor")
st.markdown("Advanced Edge AI that detects resource anomalies and identifies the **Process ID** responsible.")
st.info("ℹ️ This system learns your normal PC behavior and alerts you when unusual resource spikes occur, pinpointing the exact process causing the issue.")

col_nav1, col_nav2 = st.columns([1, 3])

with col_nav1:
    st.subheader("Control System")
    st.caption("Switch between monitoring your system or training the AI on your usage patterns.")
    mode = st.radio("System Mode:", ["Monitor (Active)", "Retrain System"])
    
    st.divider()
    st.subheader("Alert Settings")
    sound_toggle = st.checkbox("🔊 Sound Alerts", value=st.session_state['sound_enabled'], help="Play alarm sound when anomalies are detected")
    st.session_state['sound_enabled'] = sound_toggle
    
    st.divider()
    if st.session_state['model_trained']:
        st.success("✅ Your Personal AI Model Loaded")
        if st.button("🗑️ Reset Model", help="Clear your trained model and start fresh"):
            st.session_state['model'] = None
            st.session_state['scaler'] = None
            st.session_state['model_trained'] = False
            st.success("Model reset! Please retrain.")
            st.rerun()
    else:
        st.warning("⚠️ No Model Found. Please Train First.")

# --- MODE: RETRAIN SYSTEM ---
if mode == "Retrain System":
    st.header("🧠 Context Learning Phase")
    st.info("Click Start. For 30 seconds, vary your usage (Idle vs Active).")
    st.caption("The AI will observe your CPU, RAM, and Disk activity to understand what's normal for your system.")
    
    if st.button("Start Context Learning"):
        progress = st.progress(0)
        status = st.empty()
        training_data = []
        
        for i in range(30):
            m = get_metrics()
            training_data.append(m)
            status.text(f"Learning Context... CPU: {m[0]}% | RAM: {m[1]}% | Disk: {m[2]:.1f}MB")
            progress.progress((i+1)/30)
            time.sleep(1)
            
        # Data Science Pipeline
        # We explicitly name columns here
        df = pd.DataFrame(training_data, columns=['CPU', 'RAM', 'Disk'])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        
        model = IsolationForest(contamination=CONTAMINATION, random_state=42)
        model.fit(X_scaled)
        
        # Calculate baseline normal usage (mean of training data)
        st.session_state['baseline_cpu'] = df['CPU'].mean()
        st.session_state['baseline_ram'] = df['RAM'].mean()
        st.session_state['baseline_disk'] = df['Disk'].mean()
        
        # Store in session state (user-specific, in-memory)
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['model_trained'] = True
            
        st.success(f"✅ Training Complete! Your personal model is ready. Switch to 'Monitor' mode.")
        st.info(f"📊 Baseline Normal Usage - CPU: {st.session_state['baseline_cpu']:.1f}% | RAM: {st.session_state['baseline_ram']:.1f}% | Disk: {st.session_state['baseline_disk']:.1f}MB")

# --- MODE: MONITOR (ACTIVE) ---
elif mode == "Monitor (Active)":
    if not st.session_state['model_trained']:
        st.error("Please train your personal model first.")
        st.stop()
        
    # Load from session state (user-specific model)
    model = st.session_state['model']
    scaler = st.session_state['scaler']
        
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1_metric = kpi1.empty()
    kpi2_metric = kpi2.empty()
    status_metric = kpi3.empty()
    
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Live Tensor Stream")
        st.caption("Real-time CPU usage over the last 60 seconds - spikes indicate potential anomalies.")
        chart_place = st.empty()
    with c2:
        st.subheader("Forensics Log")
        st.caption("AI analysis results - shows which process is causing resource anomalies.")
        log_place = st.empty()
        
    start_monitoring = st.checkbox("Enable Real-Time Watchdog", value=True)
    st.caption("Toggle to start/stop continuous monitoring. The AI scans your system every second.")
    
    if start_monitoring:
        while True:
            # 1. ACQUIRE DATA
            raw_metrics = get_metrics() 
            
            # --- THE FIX IS HERE ---
            # We convert the list to a DataFrame with NAMES to satisfy Sklearn warning
            input_df = pd.DataFrame([raw_metrics], columns=['CPU', 'RAM', 'Disk'])
            
            # 2. PREPROCESS
            vector_scaled = scaler.transform(input_df)
            
            # 3. INFERENCE
            prediction = model.predict(vector_scaled)[0]
            score = model.decision_function(vector_scaled)[0]
            
            # 4. THRESHOLD CHECK: Only consider it anomaly if usage exceeds baseline + 10%
            cpu_threshold = st.session_state['baseline_cpu'] * 1.10
            ram_threshold = st.session_state['baseline_ram'] * 1.10
            disk_threshold = st.session_state['baseline_disk'] * 1.10
            
            # Check if current usage exceeds baseline + 10%
            exceeds_threshold = (raw_metrics[0] > cpu_threshold or 
                               raw_metrics[1] > ram_threshold or 
                               raw_metrics[2] > disk_threshold)
            
            # Only register as anomaly if both ML model detects it AND it exceeds threshold
            final_prediction = prediction if exceeds_threshold else 1
            
            # 5. SLIDING WINDOW LOGIC
            st.session_state['anomaly_buffer'].append(final_prediction)
            anomaly_count = st.session_state['anomaly_buffer'].count(-1)
            # Require majority (3 out of 5) for confirmed anomaly
            is_confirmed_anomaly = anomaly_count >= 3
            
            # 6. FORENSICS
            culprit_name = "System"
            culprit_pid = "N/A"
            
            if is_confirmed_anomaly:
                top_proc = get_top_process()
                if top_proc:
                    culprit_name = top_proc['name']
                    culprit_pid = top_proc['pid']
            
            # 7. UI UPDATE
            kpi1_metric.metric("CPU Load", f"{raw_metrics[0]}%", delta=f"Threshold: {cpu_threshold:.1f}%")
            kpi2_metric.metric("RAM Usage", f"{raw_metrics[1]}%", delta=f"Threshold: {ram_threshold:.1f}%")
            
            if is_confirmed_anomaly:
                status_metric.metric("Status", "CRITICAL ANOMALY", delta="-ALERT", delta_color="inverse")
                log_place.error(f"🚨 **ANOMALY DETECTED**\n\n**Culprit:** {culprit_name}\n**PID:** {culprit_pid}\n**CPU:** {raw_metrics[0]}% (Threshold: {cpu_threshold:.1f}%)\n**RAM:** {raw_metrics[1]}% (Threshold: {ram_threshold:.1f}%)\n**Confidence:** {abs(score):.2f}")
                
                # Play sound alarm if enabled
                if st.session_state['sound_enabled']:
                    if HAS_WINSOUND:
                        try:
                            # Windows: Play 3 beeps: 1000Hz for 200ms
                            winsound.Beep(1000, 200)
                            time.sleep(0.1)
                            winsound.Beep(1000, 200)
                            time.sleep(0.1)
                            winsound.Beep(1000, 200)
                        except Exception:
                            pass
                    else:
                        # Non-Windows: Use browser-based audio
                        try:
                            # Generate a simple beep sound using HTML5 audio
                            audio_html = """
                            <audio autoplay>
                                <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBTGH0fPTgjMGHm7A7+OZRA0PVa3n5bJeGwc9mNr0v3IdBSuBzvLaiTkIGGa67OmVSw0NUqbj8bllHQY2kNbx0IMpBSh5yPDblkEKE1y06+qnVhQKRp/g8r5sIQUxh9Hz04IzBh1twO/kmUQND1Wt5+WyXhsHPZja9L9yHQUrgc7y2ok5CBhmuuzplUsNDVKm4/G5ZR0GNo/W8dCDKQUoecjw25ZBChNctOvqp1YUCkWf4PK+bCEFMYfR89OCMwYdbcDv5JlEDQ9Vrufmsl4bBz2Y2vS/ch0FK4HO8tqJOQgYZrrs6ZVLDAxSpuPxuWUdBjaP1vHQgykFKHnI8NuWQQoTXLTr6qdWFApFn+DyvmwhBTGH0fPTgjMGHW3A7+SZRA0PVa7n5rJeGwc9mNr0v3IdBSuBzvLaiTkIGGa67OmVSw0MUqbj8bllHQY2j9bx0IMpBSh5yPDblkEKE1y06+qnVhQKRZ/g8r5sIQUxh9Hz04IzBh1twO/kmUQND1Wu5+ayXhsHPZja9L9yHQUrgc7y2ok5CBhmuuzplUsNDFKm4/G5ZR0GNo/W8dCDKQUoecjw25ZBChNctOvqp1YUCkWf4PK+bCEFMYfR89OCMwYdbcDv5JlEDQ9Vrufmsl4bBz2Y2vS/ch0FK4HO8tqJOQgYZrrs6ZVLDQxSpuPxuWUdBjaP1vHQgykFKHnI8NuWQQoTXLTr6qdWFApFn+DyvmwhBTGH0fPTgjMGHW3A7+SZRA0PVa7n5rJeGwc9mNr0v3IdBSuBzvLaiTkIGGa67OmVSw0MUqbj8bllHQY2j9bx0IMpBSh5yPDblkEKE1y06+qnVhQKRZ/g8r5sIQUxh9Hz04IzBh1twO/kmUQND1Wu5+ayXhsHPZja9L9yHQUrgc7y2ok5CBhmuuzplUsNDFKm4/G5ZR0GNo/W8dCDKQUoecjw25ZBChNctOvqp1YUCkWf4PK+bCEFMYfR89OCMwYdbcDv5JlEDQ9Vrufmsl4bBz2Y2vS/cA==">
                            </audio>
                            """
                            st.markdown(audio_html, unsafe_allow_html=True)
                        except Exception:
                            pass
            elif prediction == -1 and not exceeds_threshold:
                status_metric.metric("Status", "Within Threshold", delta="✓", delta_color="normal")
                log_place.info("ℹ️ Spike detected but within 10% threshold (considered normal)")
            elif final_prediction == -1:
                status_metric.metric("Status", "Analyzing Spike...", delta="⚠️", delta_color="off")
                log_place.warning("⚠️ Spike above threshold detected (confirming...)")
            else:
                status_metric.metric("Status", "System Nominal", delta="✓", delta_color="normal")
                log_place.success("✅ System running normally")
            
            st.session_state['history_cpu'].append(raw_metrics[0])
            
            # Create DataFrame with proper labels for chart
            cpu_data = pd.DataFrame(
                list(st.session_state['history_cpu']),
                columns=["CPU Usage (%)"]
            )
            chart_place.line_chart(cpu_data, x_label="Time (seconds)", y_label="CPU (%)")
            
            time.sleep(1)