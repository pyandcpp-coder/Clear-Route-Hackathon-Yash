import streamlit as st
import requests
import json
import time 
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random # Added for mock data randomization
from scipy import stats 
import re # Import regex for parsing LLM output

# --- Configuration ---
BACKEND_API_BASE_URL = "http://localhost:8000"
AI_API_BASE_URL = "http://localhost:8001"

# --- Streamlit Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="United Autosports RaceBrain AI Pro",
    initial_sidebar_state="expanded",
    page_icon="🏁"
)

# Professional Racing Theme CSS
st.markdown("""
<style>
    :root {
        --primary: #0A2463; /* Dark Blue */
        --secondary: #1E88E5; /* Bright Blue */
        --accent: #FF6B35; /* Orange-Red */
        --success: #4CAF50; /* Green */
        --warning: #FFC107; /* Amber */
        --danger: #F44336; /* Red */
        --dark: #121212; /* Very Dark Gray */
        --light: #F8F9FA; /* Off-White */
    }
    
    body {
        background: linear-gradient(135deg, var(--dark) 0%, #1a1a2e 100%);
        color: var(--light);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.5rem;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-weight: 800;
        font-size: 2.5rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--accent);
    }
    
    .metric-card {
        background: rgba(30, 30, 46, 0.7);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 107, 53, 0.3);
    }
    
    .status-good { 
        background: linear-gradient(135deg, var(--success), #2E7D32) !important; 
        border-left: 4px solid #4CAF50;
    }
    .status-warning { 
        background: linear-gradient(135deg, var(--warning), #FF8F00) !important; 
        border-left: 4px solid #FFC107;
    }
    .status-critical { 
        background: linear-gradient(135deg, var(--danger), #C62828) !important; 
        border-left: 4px solid #F44336;
    }
    
    .tire-visual {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .tire-visual:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    
    .ai-response {
        background: rgba(30, 30, 46, 0.8);
        border-radius: 15px;
        padding: 1.8rem;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .race-position {
        display: flex;
        align-items: center;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: rgba(30, 30, 46, 0.7);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }
    
    .race-position:hover {
        transform: translateX(5px);
        background: rgba(42, 42, 66, 0.8);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--primary) 0%, #0d3b66 100%);
        box-shadow: 5px 0 15px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 12px;
        padding: 0.5rem;
        gap: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--accent), #F7931E) !important;
        color: white !important;
        box-shadow: 0 4px 10px rgba(255, 107, 53, 0.3);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, var(--accent), #F7931E);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 107, 53, 0.4);
    }
    
    .alert-critical {
        animation: pulse 1.5s infinite;
        border-left: 4px solid var(--danger) !important;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
    }
    
    .alert-high {
        border-left: 4px solid var(--warning) !important;
    }
    
    .alert-medium {
        border-left: 4px solid #FF9800 !important;
    }
    
    .driver-card {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .position-badge {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent), #F7931E);
        color: white;
        font-weight: bold;
        margin-right: 15px;
        flex-shrink: 0;
    }
    
    .position-1 { background: linear-gradient(135deg, #FFD700, #D4AF37) !important; }
    .position-2 { background: linear-gradient(135deg, #C0C0C0, #A9A9A9) !important; }
    .position-3 { background: linear-gradient(135deg, #CD7F32, #A97142) !important; }
    
    .driver-name {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .driver-stats {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: rgba(0, 0, 0, 0.2);
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--accent);
    }
    
    .stat-label {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    
    .tire-indicator {
        height: 10px;
        width: 100%;
        background: #333;
        border-radius: 5px;
        margin-top: 8px;
        overflow: hidden;
        position: relative;
    }
    
    .tire-wear-bar {
        height: 100%;
        background: linear-gradient(90deg, var(--success), var(--danger));
    }
    
    .technical-panel {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2); /* Corrected from box_shadow */
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .analytics-card {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2); /* Corrected from box_shadow */
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 100%;
    }
    
    .race-control-panel {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2); /* Corrected from box_shadow */
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_response_cache" not in st.session_state:
    st.session_state.ai_response_cache = {
        "strategy_recommendation": "AI could not generate a recommendation.", 
        "confidence_score": 0.0, 
        "priority_actions": [], 
        "anomaly_report": {}
    }
if "ai_query_input_value" not in st.session_state: 
    st.session_state.ai_query_input_value = ""
if "ai_query_submitted" not in st.session_state: 
    st.session_state.ai_query_submitted = False
if "performance_history" not in st.session_state:
    st.session_state.performance_history = []
if "prev_position" not in st.session_state:
    st.session_state.prev_position = None
if "selected_driver" not in st.session_state:
    st.session_state.selected_driver = "Phil Hanson" # Default driver for initial display
if "driver_stint_start" not in st.session_state:
    st.session_state.driver_stint_start = time.time() # Real time for UI stint duration
if "driver_stint_laps_start" not in st.session_state: # Lap number at which current driver's stint began
    st.session_state.driver_stint_laps_start = 0
if 'chart_counter' not in st.session_state: # NEW: Counter for unique chart keys
    st.session_state.chart_counter = 0

# --- API Functions ---
@st.cache_data(ttl=1)
def get_live_data():
    try:
        response = requests.get(f"{BACKEND_API_BASE_URL}/live_data")
        response.raise_for_status()
        data = response.json()
        
        if data and 'car' in data:
            current_timestamp = data.get('timestamp_simulated_sec', time.time())
            
            # Check for driver change to reset stint info
            current_driver_from_sim = data['car'].get('current_driver', 'N/A')
            # Only reset if the driver actually changed from the last known driver
            if st.session_state.selected_driver != current_driver_from_sim:
                st.session_state.selected_driver = current_driver_from_sim # Update the session state driver
                st.session_state.driver_stint_start = time.time() # Reset real time for UI display
                st.session_state.driver_stint_laps_start = data['car'].get('lap_number', 0)
            
            performance_point = {
                'timestamp_sec': current_timestamp,
                'lap_time': data['car'].get('last_lap_time_sec', 0),
                'speed': data['car'].get('speed_kmh', 0),
                'fuel': data['car'].get('fuel_level_liters', 0),
                'fuel_consumption': data['car'].get('fuel_consumption_current_L_per_lap', 0),
                'tire_temp_FL_C': data['car'].get('tire_temp_FL_C', 0),
                'tire_temp_FR_C': data['car'].get('tire_temp_FR_C', 0),
                'tire_temp_RL_C': data['car'].get('tire_temp_RL_C', 0),
                'tire_temp_RR_C': data['car'].get('tire_temp_RR_C', 0),
                'tire_wear_FL_percent': data['car'].get('tire_wear_FL_percent', 0),
                'tire_wear_FR_percent': data['car'].get('tire_wear_FR_percent', 0),
                'tire_wear_RL_percent': data['car'].get('tire_wear_RL_percent', 0),
                'tire_wear_RR_percent': data['car'].get('tire_wear_RR_percent', 0),
                'oil_temp_C': data['car'].get('oil_temp_C', 0),
                'water_temp_C': data['car'].get('water_temp_C', 0),
                'brake_percent': data['car'].get('brake_percent', 0),
                'suspension_travel_FL_mm': data['car'].get('suspension_travel_FL_mm', 0),
                'suspension_travel_FR_mm': data['car'].get('suspension_travel_FR_mm', 0),
                'suspension_travel_RL_mm': data['car'].get('suspension_travel_RL_mm', 0),
                'suspension_travel_RR_mm': data['car'].get('suspension_travel_RR_mm', 0),
                'throttle_percent': data['car'].get('throttle_percent', 0)
            }
            st.session_state.performance_history.append(performance_point)
            
            # Keep only last N points to prevent memory issues for long runs
            max_history_points = st.session_state.get('telemetry_limit', 100) # Use slider value if available
            if len(st.session_state.performance_history) > max_history_points:
                st.session_state.performance_history.pop(0)
        
        return data
    except requests.exceptions.RequestException:
        return None

@st.cache_data(ttl=5)
def get_telemetry_history_ui(limit=50): 
    try:
        response = requests.get(f"{BACKEND_API_BASE_URL}/telemetry_history?limit={limit}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def query_race_brain_ai(user_query: str):
    try:
        openai.api_key = st.secrets["GROQ_API_KEY"]
        openai.base_url = "https://api.groq.com/openai/v1"
        response = requests.post(
            f"{AI_API_BASE_URL}/query_race_brain_ai",
            json={"user_input": user_query},
            timeout=60
        )
        response.raise_for_status()
        ai_response = response.json()
        ai_response['timestamp'] = datetime.now().isoformat()
        return ai_response 
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to query RaceBrain AI: {e}")
        return {
            "strategy_recommendation": f"Error: AI service unavailable: {e}", 
            "confidence_score": 0.0, 
            "priority_actions": [], 
            "anomaly_report": {},
            "timestamp": datetime.now().isoformat()
        }

# --- Helper Functions ---
def get_tire_status(temp, wear):
    if temp > 110 or wear > 80:
        return "status-critical", "❌"
    elif temp > 90 or wear > 60:
        return "status-warning", "⚠️"
    else:
        return "status-good", "✅"

def create_enhanced_gauge(value, title, min_val=0, max_val=100, unit="", color_ranges=None):
    if color_ranges is None:
        color_ranges = [(0, 50, "#4CAF50"), (50, 80, "#FFC107"), (80, 100, "#F44336")]
    
    if value is None or np.isnan(value):
        value = min_val

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{title} ({unit})", 'font': {'size': 16}},
        delta = {'reference': (max_val - min_val) * 0.5, 'relative': True},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#1E88E5", 'thickness': 0.3},
            'steps': [
                {'range': [r[0], r[1]], 'color': r[2]} for r in color_ranges
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def handle_query_submit():
    st.session_state.ai_query_submitted = True

def get_position_change_ui(current_position):
    # This function is for UI display, prev_position is stored in session_state
    # This logic assumes '1' is the best position.
    
    if st.session_state.prev_position is None:
        st.session_state.prev_position = current_position
        return "→", "#FFC107" # No change yet

    change = st.session_state.prev_position - current_position
    
    if change > 0: # Position improved (e.g., from 3 to 2, change is 1)
        delta_text = f"↑{change}"
        color = "#4CAF50" # Green for improvement
    elif change < 0: # Position worsened (e.g., from 2 to 3, change is -1)
        delta_text = f"↓{abs(change)}"
        color = "#F44336" # Red for worsening
    else: # No change
        delta_text = "→"
        color = "#FFC107" # Yellow for stable
    
    st.session_state.prev_position = current_position # Update for next cycle
    return delta_text, color

def parse_llm_response(raw_response: str) -> (str, str):
    """
    Parses the raw LLM response to separate the internal thought process
    from the final strategic recommendation.
    """
    think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
    think_process = think_match.group(1).strip() if think_match else "No internal thought process provided."

    strategy_match = re.search(r"STRATEGY RECOMMENDATION:\s*(.*)", raw_response, re.DOTALL)
    strategy_recommendation = strategy_match.group(1).strip() if strategy_match else raw_response.strip()

    return think_process, strategy_recommendation

# Mock data generation functions for sections not directly from simulator
def generate_technical_data_mock(live_data_car):
    """Generates mock technical data, incorporating some live data for consistency."""
    return {
        "oil_temp_C": live_data_car.get('oil_temp_C', np.random.uniform(85, 115)),
        "water_temp_C": live_data_car.get('water_temp_C', np.random.uniform(75, 100)),
        "engine_rpm": live_data_car.get('engine_rpm', np.random.randint(8000, 12000)),
        "hybrid_power_output_kw": live_data_car.get('hybrid_power_output_kw', np.random.uniform(120, 180)),
        "hybrid_battery_percent": live_data_car.get('hybrid_battery_percent', np.random.uniform(30, 80)),
        "throttle_percent": live_data_car.get('throttle_percent', np.random.uniform(0, 100)),
        "brake_percent": live_data_car.get('brake_percent', np.random.uniform(0, 100)),
        "gear": live_data_car.get('gear', np.random.randint(1, 8)),
        "data_link": "🟢 Connected",
        "gps_accuracy": "🟢 High",
        "sensors": "🟢 All Operational",
        "downforce_setting": "Medium", # Could be dynamic in a real sim
        "brake_balance": "58% Front", # Could be dynamic
        "suspension_travel_FL_mm": live_data_car.get('suspension_travel_FL_mm', np.random.uniform(20, 50)),
        "suspension_travel_FR_mm": live_data_car.get('suspension_travel_FR_mm', np.random.uniform(20, 50)),
        "suspension_travel_RL_mm": live_data_car.get('suspension_travel_RL_mm', np.random.uniform(20, 50)),
        "suspension_travel_RR_mm": live_data_car.get('suspension_travel_RR_mm', np.random.uniform(20, 50)),
    }

def generate_analytics_data_mock():
    # This is static mock data for the benchmarking and predictive sections
    return pd.DataFrame({
        'metric': ['Speed', 'Fuel Efficiency', 'Tire Wear', 'Braking', 'Acceleration', 'Cornering'],
        'current': [92 + np.random.uniform(-2,2), 85 + np.random.uniform(-2,2), 78 + np.random.uniform(-2,2), 88 + np.random.uniform(-2,2), 90 + np.random.uniform(-2,2), 86 + np.random.uniform(-2,2)],
        'target': [95, 90, 85, 90, 92, 90],
        'variance': [-3, -5, -7, -2, -2, -4]
    })

def generate_race_control_data_mock(live_data):
    # Combines some live data with mock for other competitors
    
    competitors = []
    # Include our car as the first competitor, with our current position (from simulator)
    our_car_position = live_data.get('car', {}).get('position_in_class', 1)
    competitors.append({
        "current_position": our_car_position,
        "name": live_data.get('car', {}).get('name', "Our Car #22"),
        "gap_to_leader_sec": 0.0, # We are the leader in our class
        "last_lap_time_sec": live_data.get('car', {}).get('last_lap_time_sec', np.random.uniform(210, 230)),
        "fuel_level_liters": live_data.get('car', {}).get('fuel_level_liters', np.random.uniform(30, 70)),
        "tire_age_laps": live_data.get('car', {}).get('tire_age_laps', np.random.randint(10, 30)),
        "pit_status": "On Track"
    })

    # Add other mock competitors
    mock_names = ["Toyota #8", "Porsche #6", "Cadillac #2", "Ferrari #51", "Peugeot #93", 
                  "Glickenhaus #708", "BMW #15", "Alpine #36", "Lamborghini #63", "United Autosports #23"]
    # Adjust gaps relative to our car's position
    base_gaps = [12.4, 25.7, 42.1, 59.3, 77.8, 95.2, 113.6, 132.9, 152.3, 170.0]
    
    # Ensure competitor positions are relative to our car's simulated position
    for i, name in enumerate(mock_names):
        if name == live_data.get('car', {}).get('name', "Our Car #22"): # Skip our car if already processed
            continue
        
        # Simple assignment of subsequent positions
        comp_position = our_car_position + (i + 1) 
        
        competitors.append({
            "current_position": comp_position,
            "name": name,
            "gap_to_leader_sec": base_gaps[i] + np.random.uniform(-5, 5), # Add some randomness
            "last_lap_time_sec": np.random.uniform(210, 230),
            "fuel_level_liters": np.random.uniform(30, 70),
            "tire_age_laps": np.random.randint(10, 30),
            "pit_status": random.choice(["On Track", "In Pit", "Next Lap"])
        })
    
    # Sort by position to ensure correct order
    competitors.sort(key=lambda x: x['current_position'])

    return {
        "competitors": competitors,
        "strategy": live_data.get('strategy', { # Use actual strategy from simulator
            "current_strategy": "Normal Pace",
            "fuel_target_laps": 35,
            "tire_change_recommended": False,
            "driver_change_due": False,
            "next_pit_recommendation": "Lap 30"
        }),
        "race_info": live_data.get('race_info', { # Use actual race info from simulator
            "session_type": "Race",
            "current_hour": 18,
            "time_of_day": "Evening",
            "is_night": False,
            "race_time_elapsed_sec": 18 * 3600,
            "race_time_remaining_sec": 6 * 3600,
            "safety_car_active": False
        }),
        "environmental": live_data.get('environmental', { # Use actual env info from simulator
            "current_weather": "Clear",
            "rain_intensity": 0,
            "ambient_temp_C": 25.0,
            "track_temp_C": 35.0,
            "track_grip_level": 1.0,
            "visibility_level": 1.0,
            "current_sector": "Sector 1",
            "wind_speed_kmh": 15.0,
            "wind_direction_deg": 180
        })
    }


# --- UI Layout ---
st.markdown('<div class="main-header">🏁 UNITED AUTOSPORTS RACEBRAIN AI PRO 🏁</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🏎️ Race Control Center")
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/United_Autosports.svg/1200px-United_Autosports.svg.png", width=200)
    st.markdown("---")
    
    # Driver Information Card (dynamic based on simulator or default)
    live_data_for_sidebar = get_live_data() or {}
    current_driver_from_sim = live_data_for_sidebar.get('car', {}).get('current_driver', st.session_state.selected_driver)
    current_lap_number = live_data_for_sidebar.get('car', {}).get('lap_number', 0)
    laps_in_stint = current_lap_number - st.session_state.driver_stint_laps_start
    
    st.subheader("👤 Current Driver")
    driver_card = st.container()
    driver_card.markdown(f"""
        <div class="driver-card">
            <div class="driver-name">{current_driver_from_sim}</div>
            <div>Stint Time: {timedelta(seconds=int(time.time() - st.session_state.driver_stint_start))}</div>
            <div>Laps in Stint: {laps_in_stint}</div>
            <div class="driver-stats">
                <div class="stat-item">
                    <div class="stat-value">{live_data_for_sidebar.get('car', {}).get('average_lap_time_sec', 0):.2f}s</div>
                    <div class="stat-label">AVG LAP</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{live_data_for_sidebar.get('car', {}).get('last_lap_time_sec', 0):.2f}s</div>
                    <div class="stat-label">LAST LAP</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Advanced Settings
    with st.expander("⚙️ Dashboard Settings"):
        refresh_interval = st.slider("Refresh Interval (sec)", 1, 10, 2)
        show_debug = st.checkbox("Show Debug Info", False)
        st.session_state.telemetry_limit = st.slider("Telemetry History Points", 20, 200, 50)
    
    # Driver Selection (This allows a visual selection, but current_driver from sim overrides actual active driver display)
    available_drivers_list_for_selectbox = live_data_for_sidebar.get('car', {}).get('drivers', ["Phil Hanson", "Filipe Albuquerque", "Will Owen"])
    
    # Set default index for selectbox to reflect current driver from sim
    default_index = 0
    if current_driver_from_sim in available_drivers_list_for_selectbox:
        default_index = available_drivers_list_for_selectbox.index(current_driver_from_sim)

    # Note: This selectbox is primarily for UI presentation. 
    # If you wanted to *control* the simulated driver from here, you'd need a backend API endpoint.
    st.selectbox("👤 Filter by Driver", available_drivers_list_for_selectbox, 
                 index=default_index, key="sidebar_driver_select")
    
    # System Status
    st.markdown("---")
    st.subheader("🔌 System Status")
    status_placeholder = st.empty() # Define the placeholder here, before its content is written
    
    # Position Tracker (Simplified as per the initial image, showing just current)
    st.markdown("---")
    st.subheader("📈 Position Tracker")
    current_position_val = live_data_for_sidebar.get('car', {}).get('position_in_class', 1)
    position_change_text, position_change_color = get_position_change_ui(current_position_val)
    st.markdown(f"""
        <div class="metric-card" style="background: rgba(0,0,0,0.2); text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 700;">{current_position_val} <span style="color: {position_change_color}; font-size: 1rem;">{position_change_text}</span></div>
            <div style="font-size: 0.8rem; opacity: 0.8;">CURRENT POSITION</div>
        </div>
    """, unsafe_allow_html=True)


# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚦 LIVE RACE", "📊 PERFORMANCE", "🧠 AI STRATEGIST", 
    "🔧 TECHNICAL", "📈 ANALYTICS", "🏆 RACE CONTROL"
])

# Main update block - this entire block will be run once per Streamlit rerun
# NEW: Increment chart counter at the start of each run (once per Streamlit rerun)
st.session_state.chart_counter += 1

live_data = get_live_data() or {}
telemetry_history = get_telemetry_history_ui(st.session_state.telemetry_limit) or []

# Generate simulated data for missing sections, passing live_data.car for consistency
tech_data = generate_technical_data_mock(live_data.get('car', {}))
analytics_data = generate_analytics_data_mock()
race_control_data = generate_race_control_data_mock(live_data) # Pass live_data to this mock

with st.container(): # Using st.container() instead of a placeholder allows direct content rendering
    current_car = live_data.get('car', {})
    race_info = live_data.get('race_info', {})
    env_info = live_data.get('environmental', {})
    
    # Tab 1: Live Race
    with tab1:
        st.subheader("🏁 LIVE RACE COMMAND CENTER")
        
        # Position Tracking
        current_position = current_car.get('position_in_class', 1) # Ensure this is always 1 for our car in LMP2
        position_change_text, position_change_color = get_position_change_ui(current_position) # Using the UI function
        
        col_pos, col_lap, col_speed, col_fuel, col_lap_time = st.columns(5)
        with col_pos:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 1.2rem; margin-bottom: 5px;">🏎️ POSITION</div>
                    <div style="display: flex; align-items: baseline; gap: 10px;">
                        <span style="font-size: 2.5rem; font-weight: 800;">{current_position}</span>
                        <span style="color: {position_change_color}; font-size: 1.5rem; font-weight: 700;">{position_change_text}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        with col_lap:
            st.metric("⏱️ LAP", current_car.get('lap_number', 'N/A'))
        with col_speed:
            speed = current_car.get('speed_kmh', 0)
            st.metric("🚀 SPEED", f"{speed:.1f} km/h", delta=f"{speed - 250:.1f}" if speed else None) # Delta vs 250km/h baseline
        with col_fuel:
            fuel = current_car.get('fuel_level_liters', 0)
            st.metric("⛽ FUEL", f"{fuel:.1f} L", delta=f"{fuel - 75:.1f}" if fuel else None, delta_color="inverse") # Delta vs 75L tank
        with col_lap_time:
            lap_time = current_car.get('last_lap_time_sec', 0)
            avg_lap_time = current_car.get('average_lap_time_sec', 0)
            delta_lap_time = lap_time - avg_lap_time if avg_lap_time else 0
            st.metric("⏰ LAST LAP", f"{lap_time:.2f}s" if lap_time else "N/A", delta=f"{delta_lap_time:.2f}s" if lap_time else None, delta_color="inverse")
        
        # Live Gauges
        st.markdown("---")
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        
        with gauge_col1:
            speed_gauge = create_enhanced_gauge(
                current_car.get('speed_kmh', 0), "SPEED", 0, 350, "km/h", 
                [(0, 100, "#616161"), (100, 250, "#4CAF50"), (250, 350, "#F44336")]
            )
            st.plotly_chart(speed_gauge, use_container_width=True, key=f"live_speed_gauge_{st.session_state.chart_counter}")
        
        with gauge_col2:
            fuel_gauge = create_enhanced_gauge(
                current_car.get('fuel_level_liters', 0), "FUEL LEVEL", 0, 75, "L", 
                [(0, 15, "#F44336"), (15, 30, "#FFC107"), (30, 75, "#4CAF50")]
            )
            st.plotly_chart(fuel_gauge, use_container_width=True, key=f"live_fuel_gauge_{st.session_state.chart_counter}")
        
        with gauge_col3:
            tire_temps = [
                current_car.get(f'tire_temp_{t}_C', 0) 
                for t in ['FL', 'FR', 'RL', 'RR']
            ]
            avg_tire_temp = np.mean(tire_temps) if all(t is not None for t in tire_temps) else 0 
            temp_gauge = create_enhanced_gauge(
                avg_tire_temp, "AVG TIRE TEMP", 0, 150, "°C",
                [(0, 80, "#4CAF50"), (80, 120, "#FFC107"), (120, 150, "#F44336")]
            )
            st.plotly_chart(temp_gauge, use_container_width=True, key=f"live_avg_tire_temp_gauge_{st.session_state.chart_counter}")
        
        # Tire Status
        st.markdown("---")
        st.subheader("🏎️ TIRE STATUS MATRIX")
        tire_cols = st.columns(4)
        tires = ["FL", "FR", "RL", "RR"]
        tire_positions = ["FRONT LEFT", "FRONT RIGHT", "REAR LEFT", "REAR RIGHT"]
        
        for i, (tire, position) in enumerate(zip(tires, tire_positions)):
            with tire_cols[i]:
                temp = current_car.get(f'tire_temp_{tire}_C', np.random.uniform(70, 110))
                pressure = current_car.get(f'tire_pressure_{tire}_bar', np.random.uniform(1.8, 2.2))
                wear = current_car.get(f'tire_wear_{tire}_percent', np.random.uniform(20, 70))
                
                status_class, status_icon = get_tire_status(temp, wear)
                
                st.markdown(f"""
                <div class="tire-visual {status_class}">
                    <h4>{position} {status_icon}</h4>
                    <div>🌡️ {temp:.1f}°C</div>
                    <div>⚡ {pressure:.1f} bar</div>
                    <div>WEAR: {wear:.1f}%</div>
                    <div class="tire-indicator">
                        <div class="tire-wear-bar" style="width: {wear}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Alerts (from simulator)
        st.markdown("---")
        st.subheader("🚨 LIVE ALERTS & ANOMALIES")
        active_sim_anomalies = live_data.get('active_anomalies', [])
        if active_sim_anomalies:
            for anomaly in active_sim_anomalies:
                severity = anomaly.get('severity', 'UNKNOWN').upper()
                message = anomaly.get('message', 'Unknown issue')
                
                alert_class = ""
                if severity == 'CRITICAL':
                    alert_class = "alert-critical"
                elif severity == 'HIGH': 
                    alert_class = "alert-high"
                elif severity == 'MEDIUM':
                    alert_class = "alert-medium"
                
                st.markdown(f"""
                    <div class="metric-card {alert_class}">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="font-size: 1.5rem;">
                                {'🔴' if severity == 'CRITICAL' else '🟠' if severity == 'HIGH' else '🟡' if severity == 'MEDIUM' else '🔵'}
                            </div>
                            <div>
                                <div style="font-weight: 700; font-size: 1.1rem; text-transform: uppercase;">{severity}</div>
                                <div>{message}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🟢 All systems nominal - No active alerts from simulator.")
    
    # Tab 2: Performance
    with tab2:
        st.subheader("📊 PERFORMANCE ANALYTICS DASHBOARD")
        
        if st.session_state.performance_history:
            df_perf = pd.DataFrame(st.session_state.performance_history)
            
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                fig_lap = px.line(df_perf, x='timestamp_sec', y='lap_time', 
                                 title='🏁 LAP TIME EVOLUTION',
                                 labels={'timestamp_sec': 'TIME (S)', 'lap_time': 'LAP TIME (S)'},
                                 color_discrete_sequence=['#FF6B35'])
                fig_lap.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_lap, use_container_width=True, key=f"perf_lap_time_trend_{st.session_state.chart_counter}")
                
                fig_scatter = px.scatter(df_perf, x='fuel', y='speed',
                                       title='⛽ FUEL LEVEL VS SPEED ANALYSIS',
                                       labels={'fuel': 'FUEL (L)', 'speed': 'SPEED (KM/H)'},
                                       color='tire_temp_FL_C', 
                                       color_continuous_scale='Viridis',
                                       hover_data=['lap_time', 'fuel_consumption'])
                fig_scatter.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_scatter, use_container_width=True, key=f"perf_fuel_speed_scatter_{st.session_state.chart_counter}")
            
            with perf_col2:
                fig_multi = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('SPEED OVER TIME', 'FUEL CONSUMPTION OVER TIME'), 
                    vertical_spacing=0.15
                )
                
                fig_multi.add_trace(
                    go.Scatter(x=df_perf['timestamp_sec'], y=df_perf['speed'],
                             name='SPEED (KM/H)', line=dict(color='#F7931E')),
                    row=1, col=1
                )
                
                fig_multi.add_trace(
                    go.Scatter(x=df_perf['timestamp_sec'], y=df_perf['fuel_consumption'], 
                             name='FUEL CONSUMPTION (L/LAP)', line=dict(color='#FFD23F')),
                    row=2, col=1
                )
                
                fig_multi.update_layout(
                    height=500, 
                    showlegend=True,
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_multi, use_container_width=True, key=f"perf_multi_metric_trend_{st.session_state.chart_counter}")
        
        # Performance Statistics
        st.markdown("---")
        st.subheader("📈 SESSION STATISTICS")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        # Use actual data from telemetry_history (which contains live_data points)
        lap_times_history = [d.get('car', {}).get('last_lap_time_sec', 0) for d in telemetry_history if d.get('car', {}).get('last_lap_time_sec', 0) > 0]
        speeds_history = [d.get('car', {}).get('speed_kmh', 0) for d in telemetry_history if d.get('car', {}).get('speed_kmh', 0) > 0]
        fuel_consumptions_history = [d.get('car', {}).get('fuel_consumption_current_L_per_lap', 0) for d in telemetry_history if d.get('car', {}).get('fuel_consumption_current_L_per_lap', 0) > 0]
        
        with stat_col1:
            best_lap = min(lap_times_history) if lap_times_history else 0
            st.metric("🏆 BEST LAP TIME", f"{best_lap:.3f}s" if best_lap > 0 else "N/A")
        with stat_col2:
            avg_speed = np.mean(speeds_history) if speeds_history else 0
            st.metric("📊 AVG SPEED", f"{avg_speed:.1f} km/h" if not np.isnan(avg_speed) else "N/A")
        with stat_col3:
            if telemetry_history and len(telemetry_history) > 1:
                initial_fuel = telemetry_history[0].get('car', {}).get('fuel_level_liters', 0)
                final_fuel = telemetry_history[-1].get('car', {}).get('fuel_level_liters', 0)
                fuel_used = initial_fuel - final_fuel
                st.metric("⛽ TOTAL FUEL USED", f"{fuel_used:.1f}L" if fuel_used > 0 else "N/A")
            else:
                st.metric("⛽ TOTAL FUEL USED", "N/A")
        with stat_col4:
            total_laps = max([d.get('car', {}).get('lap_number', 0) for d in telemetry_history]) if telemetry_history else 0
            st.metric("🔄 TOTAL LAPS", str(int(total_laps)) if not np.isnan(total_laps) else "N/A")
    
    # Tab 3: AI Strategist
    with tab3:
        st.subheader("🧠 RACEBRAIN AI STRATEGIC COMMAND")
        
        if st.session_state.ai_response_cache and st.session_state.ai_response_cache.get('strategy_recommendation') != "AI could not generate a recommendation.":
            ai_res = st.session_state.ai_response_cache
            
            confidence = ai_res.get('confidence_score', 0)
            conf_color = "#4CAF50" if confidence > 0.8 else "#FFC107" if confidence > 0.5 else "#F44336"
            conf_text = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.5 else "LOW"
            
            # Parse the raw LLM response
            think_process, strategy_text = parse_llm_response(ai_res['strategy_recommendation'])

            st.markdown(f"""
            <div class="ai-response">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                    <h3>🎯 STRATEGIC RECOMMENDATION</h3>
                    <div style="background: {conf_color}; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 700;">
                        CONFIDENCE: {conf_text} ({confidence:.0%})
                    </div>
                </div>
                <div style="font-size: 1.1rem; line-height: 1.6;">
                    {strategy_text}
                </div>
                <div style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
                    Generated: {datetime.fromisoformat(ai_res['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if ai_res.get('priority_actions'):
                st.subheader("⚡ IMMEDIATE ACTION ITEMS")
                for i, action in enumerate(ai_res['priority_actions'], 1):
                    st.markdown(f"""
                    <div class="metric-card" style="padding: 1rem; margin-bottom: 0.8rem;">
                        <div style="display: flex; align-items: flex-start; gap: 10px;">
                            <div style="font-size: 1.5rem; color: #FF6B35;">{i}.</div>
                            <div style="font-size: 1.1rem;">{action}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if ai_res.get('anomaly_report') and ai_res['anomaly_report'].get('priority_level') != 'NONE':
                with st.expander("🔍 DETAILED TECHNICAL ANALYSIS", expanded=True):
                    priority = ai_res['anomaly_report'].get('priority_level', 'UNKNOWN')
                    st.markdown(f"**Priority Level:** {priority}")
                    st.json(ai_res['anomaly_report'])
        
            if show_debug:
                with st.expander("💡 LLM Internal Thought Process (Debug)", expanded=False):
                    st.markdown(f"```\n{think_process}\n```")

        else:
            st.info("No strategic recommendation generated yet. Ask RaceBrain AI a question below!")
        
        st.markdown("---")
        
        col_query1, col_query2 = st.columns([3, 1])
        with col_query1:
            user_question_input = st.text_input( # Renamed variable to avoid confusion
                "💬 ASK RACEBRAIN AI:",
                value=st.session_state.ai_query_input_value,
                placeholder="e.g., 'What's our optimal pit strategy?' or 'Analyze tire degradation'",
                key="main_ai_query_input", # UNIQUE KEY
                on_change=handle_query_submit # This will set ai_query_submitted=True when user types and presses enter
            )
        
        with col_query2:
            # Use a specific key for this button to avoid conflicts
            ask_button = st.button("🚀 QUERY AI", type="primary", use_container_width=True, key="ask_ai_button") 
        
        st.markdown("**QUICK STRATEGY QUERIES:**")
        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        
        # --- Quick Query Button Logic ---
        # Define a function to handle quick query clicks
        def handle_quick_query(query_text):
            st.session_state.ai_query_input_value = query_text # Set the input box value
            with st.spinner("🧠 RACEBRAIN AI ANALYZING TELEMETRY DATA..."):
                ai_output = query_race_brain_ai(query_text)
                if ai_output:
                    st.session_state.ai_response_cache = ai_output 
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": query_text,
                        "timestamp": datetime.now()
                    })
                    _, parsed_strategy = parse_llm_response(ai_output['strategy_recommendation'])
                    st.session_state.chat_history.append({
                        "role": "ai", 
                        "content": parsed_strategy,
                        "timestamp": datetime.now()
                    })
                    st.rerun() # Force rerun to update UI with new state
                else:
                    st.error("Failed to get response from AI.")

        with quick_col1:
            if st.button("🏁 PIT STRATEGY", key="quick_pit", use_container_width=True):
                handle_quick_query("What's our optimal pit window and strategy?")
        with quick_col2:
            if st.button("🏎️ TIRE ANALYSIS", key="quick_tire", use_container_width=True):
                handle_quick_query("Analyze current tire performance and degradation")
        with quick_col3:
            if st.button("⛽ FUEL MANAGEMENT", key="quick_fuel", use_container_width=True):
                handle_quick_query("Evaluate fuel consumption and stint length")
        with quick_col4:
            if st.button("🌦️ WEATHER IMPACT", key="quick_weather", use_container_width=True):
                handle_quick_query("How will weather conditions affect our strategy?")
        # --- End Quick Query Button Logic ---
        
        # Process AI Query from main text input or 'Ask AI' button
        if ask_button or st.session_state.ai_query_submitted:
            query_to_process = user_question_input # Use the current value of the text input
            if query_to_process:
                st.session_state.ai_query_submitted = False # Reset the flag
                st.session_state.ai_query_input_value = "" # Clear input box

                with st.spinner("🧠 RACEBRAIN AI ANALYZING TELEMETRY DATA..."):
                    ai_output = query_race_brain_ai(query_to_process)
                    if ai_output:
                        st.session_state.ai_response_cache = ai_output 
                        st.session_state.chat_history.append({
                            "role": "user", 
                            "content": query_to_process,
                            "timestamp": datetime.now()
                        })
                        _, parsed_strategy = parse_llm_response(ai_output['strategy_recommendation'])
                        st.session_state.chat_history.append({
                            "role": "ai", 
                            "content": parsed_strategy,
                            "timestamp": datetime.now()
                        })
                        st.rerun() 
                    else:
                        st.error("Failed to get response from AI.")
            else:
                st.warning("Please enter a query for RaceBrain AI")
                st.session_state.ai_query_submitted = False # Reset even if no query
        
        st.markdown("---")
        st.subheader("💬 STRATEGY DISCUSSION HISTORY")
        if st.session_state.chat_history:
            for message in reversed(st.session_state.chat_history): 
                timestamp = message.get('timestamp', datetime.now()).strftime("%H:%M:%S")
                if message["role"] == "user":
                    st.markdown(f"""
                        <div class="metric-card" style="background: rgba(30, 30, 46, 0.7);">
                            <div style="display: flex; align-items: flex-start; gap: 10px;">
                                <div style="font-size: 1.5rem;">👤</div>
                                <div>
                                    <div style="font-weight: 700; color: #1E88E5;">ENGINEER [{timestamp}]</div>
                                    <div style="margin-top: 5px;">{message['content']}</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="metric-card" style="background: rgba(30, 46, 46, 0.7);">
                            <div style="display: flex; align-items: flex-start; gap: 10px;">
                                <div style="font-size: 1.5rem;">🤖</div>
                                <div>
                                    <div style="font-weight: 700; color: #FF6B35;">RACEBRAIN AI [{timestamp}]</div>
                                    <div style="margin-top: 5px;">{message['content']}</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True) 
        else:
            st.info("Start a conversation with RaceBrain AI to see chat history")
    
    # Tab 4: Technical Analysis
    with tab4:
        st.subheader("🔧 TECHNICAL SYSTEMS MONITOR")
        
        # System Health Matrix
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("### 🔋 POWER UNIT STATUS")
            with st.container():
                st.markdown(f"""
                    <div class="technical-panel">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Oil Temperature</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['oil_temp_C']:.1f}°C</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Water Temperature</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['water_temp_C']:.1f}°C</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Engine RPM</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['engine_rpm']}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Hybrid Power</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['hybrid_power_output_kw']:.1f} kW</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Hybrid Battery</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['hybrid_battery_percent']:.1f}%</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🛠️ MECHANICAL SYSTEMS")
            with st.container():
                st.markdown(f"""
                    <div class="technical-panel">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Throttle</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['throttle_percent']:.1f}%</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Brake</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['brake_percent']:.1f}%</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Gear</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['gear']}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Downforce</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['downforce_setting']}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Brake Balance</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['brake_balance']}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        with tech_col2:
            st.markdown("### 📡 TELEMETRY SYSTEMS")
            with st.container():
                st.markdown(f"""
                    <div class="technical-panel">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Data Link</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: #4CAF50;">{tech_data['data_link']}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">GPS Accuracy</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: #4CAF50;">{tech_data['gps_accuracy']}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Sensors</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: #4CAF50;">{tech_data['sensors']}</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🔄 VEHICLE DYNAMICS")
            with st.container():
                st.markdown(f"""
                    <div class="technical-panel">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Suspension FL</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['suspension_travel_FL_mm']:.1f} mm</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Suspension FR</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['suspension_travel_FR_mm']:.1f} mm</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Suspension RL</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['suspension_travel_RL_mm']:.1f} mm</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Suspension RR</div>
                                <div style="font-size: 1.5rem; font-weight: 700;">{tech_data['suspension_travel_RR_mm']:.1f} mm</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Detailed Telemetry Charts
        st.markdown("---")
        st.subheader("📊 TELEMETRY ANALYSIS")
        
        tech_chart_col1, tech_chart_col2 = st.columns(2)
        
        with tech_chart_col1:
            df_hist_current = pd.DataFrame(st.session_state.performance_history) # Use actual history for charts if possible

            fig = go.Figure()

            # Oil Temp trace (using actual data if available, else mock)
            if not df_hist_current.empty and 'oil_temp_C' in df_hist_current.columns and df_hist_current['oil_temp_C'].any(): # Check if column exists and has non-zero values
                fig.add_trace(go.Scatter(
                    x=df_hist_current['timestamp_sec'], 
                    y=df_hist_current['oil_temp_C'],
                    name="Oil Temp",
                    line=dict(color='#FF6B35')
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=np.arange(st.session_state.telemetry_limit), # Use history limit for mock x-axis
                    y=np.random.normal(95, 3, st.session_state.telemetry_limit),
                    name="Oil Temp",
                    line=dict(color='#FF6B35')
                ))


            # Water Temp trace
            if not df_hist_current.empty and 'water_temp_C' in df_hist_current.columns and df_hist_current['water_temp_C'].any(): # Check if column exists and has non-zero values
                fig.add_trace(go.Scatter(
                    x=df_hist_current['timestamp_sec'], 
                    y=df_hist_current['water_temp_C'],
                    name="Water Temp",
                    line=dict(color='#1E88E5')
                ))
            else:
                 fig.add_trace(go.Scatter(
                    x=np.arange(st.session_state.telemetry_limit), 
                    y=np.random.normal(85, 2, st.session_state.telemetry_limit),
                    name="Water Temp",
                    line=dict(color='#1E88E5')
                ))


            # Layout updates
            fig.update_layout(
                title="ENGINE TEMPERATURES OVER TIME",
                xaxis_title="Time (s)",
                yaxis_title="Temperature (°C)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig, use_container_width=True, key=f"tech_engine_temps_chart_{st.session_state.chart_counter}")
                
        with tech_chart_col2:
            fig = go.Figure()

            # Suspension traces (using actual data if available, else mock)
            suspension_cols = ['suspension_travel_FL_mm', 'suspension_travel_FR_mm', 'suspension_travel_RL_mm', 'suspension_travel_RR_mm']
            if not df_hist_current.empty and all(col in df_hist_current.columns and df_hist_current[col].any() for col in suspension_cols):
                fig.add_trace(go.Scatter(x=df_hist_current['timestamp_sec'], y=df_hist_current['suspension_travel_FL_mm'], name="Front Left", line=dict(color='#FF6B35')))
                fig.add_trace(go.Scatter(x=df_hist_current['timestamp_sec'], y=df_hist_current['suspension_travel_FR_mm'], name="Front Right", line=dict(color='#1E88E5')))
                fig.add_trace(go.Scatter(x=df_hist_current['timestamp_sec'], y=df_hist_current['suspension_travel_RL_mm'], name="Rear Left", line=dict(color='#4CAF50')))
                fig.add_trace(go.Scatter(x=df_hist_current['timestamp_sec'], y=df_hist_current['suspension_travel_RR_mm'], name="Rear Right", line=dict(color='#FFC107')))
            else:
                # Mock data for demonstration if history is empty or columns are missing/all zero
                fig.add_trace(go.Scatter(x=np.arange(st.session_state.telemetry_limit), y=np.random.normal(45, 5, st.session_state.telemetry_limit), name="Front Left", line=dict(color='#FF6B35')))
                fig.add_trace(go.Scatter(x=np.arange(st.session_state.telemetry_limit), y=np.random.normal(42, 4, st.session_state.telemetry_limit), name="Front Right", line=dict(color='#1E88E5')))
                fig.add_trace(go.Scatter(x=np.arange(st.session_state.telemetry_limit), y=np.random.normal(38, 3, st.session_state.telemetry_limit), name="Rear Left", line=dict(color='#4CAF50')))
                fig.add_trace(go.Scatter(x=np.arange(st.session_state.telemetry_limit), y=np.random.normal(40, 4, st.session_state.telemetry_limit), name="Rear Right", line=dict(color='#FFC107')))

            # Layout
            fig.update_layout(
                title="SUSPENSION TRAVEL ANALYSIS",
                xaxis_title="Time (s)",
                yaxis_title="Travel (mm)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig, use_container_width=True, key=f"tech_suspension_travel_chart_{st.session_state.chart_counter}")

     # Brake & Throttle Analysis
            st.markdown("---")
            st.subheader("🛑 BRAKE & THROTTLE ANALYSIS")
                    
            fig = go.Figure()
            if not df_hist_current.empty and 'throttle_percent' in df_hist_current.columns and 'brake_percent' in df_hist_current.columns and df_hist_current['throttle_percent'].any() and df_hist_current['brake_percent'].any():
                fig.add_trace(go.Scatter(
                    x=df_hist_current['timestamp_sec'], 
                    y=df_hist_current['throttle_percent'],
                    name="Throttle",
                    line=dict(color='#4CAF50')
                ))
                fig.add_trace(go.Scatter(
                    x=df_hist_current['timestamp_sec'], 
                    y=df_hist_current['brake_percent'],
                    name="Brake",
                    line=dict(color='#F44336')
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=np.arange(st.session_state.telemetry_limit), 
                    y=np.random.uniform(0, 100, st.session_state.telemetry_limit),
                    name="Throttle",
                    line=dict(color='#4CAF50')
                ))
                fig.add_trace(go.Scatter(
                    x=np.arange(st.session_state.telemetry_limit), 
                    y=np.random.uniform(0, 100, st.session_state.telemetry_limit),
                    name="Brake",
                    line=dict(color='#F44336')
                    ))

            fig.update_layout(
                title="THROTTLE & BRAKE APPLICATION",
                xaxis_title="Time (s)",
                yaxis_title="Percentage (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
                )
            st.plotly_chart(fig, use_container_width=True, key=f"tech_throttle_brake_chart_{st.session_state.chart_counter}")
                    
                   
    
    # Tab 5: Analytics
    with tab5:
        st.subheader("📈 ADVANCED RACE ANALYTICS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔗 PERFORMANCE CORRELATION MATRIX")
            # Using live performance history for correlation
            if not st.session_state.performance_history:
                st.info("No historical data yet for correlation analysis.")
            else:
                df_analytics = pd.DataFrame(st.session_state.performance_history)
                numeric_cols = df_analytics.select_dtypes(include=[np.number]).columns
                key_metrics = ['speed', 'lap_time', 'fuel', 
                             'tire_temp_FL_C', 'tire_temp_FR_C', 'fuel_consumption',
                             'oil_temp_C', 'water_temp_C', 'brake_percent', 
                             'suspension_travel_FL_mm', 'throttle_percent'] 
                available_metrics = [col for col in key_metrics if col in numeric_cols and not df_analytics[col].isnull().all() and df_analytics[col].any()]
                
                if len(available_metrics) > 1:
                    corr_matrix = df_analytics[available_metrics].corr()
                    fig_corr = px.imshow(corr_matrix, 
                                       text_auto=True, 
                                       aspect="auto",
                                       title="📊 PERFORMANCE METRICS CORRELATION HEATMAP",
                                       color_continuous_scale='RdBu')
                    fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig_corr, use_container_width=True, key=f"analytics_correlation_heatmap_{st.session_state.chart_counter}")
                else:
                    st.info("Not enough numeric data for correlation analysis. Need at least 2 non-empty numeric columns.")
            
            st.markdown("### 🔮 PREDICTIVE ANALYTICS")
            # Predictive Analytics now uses simulator's historical data
            
            # Initialize with default values
            tire_pred_info = "N/A (no data)"
            fuel_pred_info = "N/A (no data)"
            
            if st.session_state.performance_history and len(st.session_state.performance_history) > 10:
                df_analytics = pd.DataFrame(st.session_state.performance_history)
                
                # Tire Prediction
                tire_wear_data = df_analytics[['timestamp_sec', 'tire_wear_FL_percent']].dropna()
                if len(tire_wear_data) >= 2 and tire_wear_data['tire_wear_FL_percent'].any(): # stats.linregress needs at least 2 points and non-zero data
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        tire_wear_data['timestamp_sec'], tire_wear_data['tire_wear_FL_percent']
                    )
                    if slope > 0: # Only predict if wear is increasing
                        time_to_90_percent_wear = (90 - intercept) / slope
                        current_time = df_analytics['timestamp_sec'].iloc[-1]
                        if time_to_90_percent_wear > current_time:
                            remaining_time_sec = max(0, time_to_90_percent_wear - current_time)
                            tire_pred_info = f"{remaining_time_sec/60:.1f} mins"
                        else:
                            tire_pred_info = f"Already > 90% ({tire_wear_data['tire_wear_FL_percent'].iloc[-1]:.1f}%)"
                    else:
                        tire_pred_info = "Stable/Decreasing wear"
                else:
                    tire_pred_info = "Not enough wear data"

                # Fuel Prediction
                if 'fuel' in df_analytics.columns and 'fuel_consumption' in df_analytics.columns and len(df_analytics) > 10 and df_analytics['fuel'].any():
                    recent_data = df_analytics.tail(10) # Look at recent 10 points
                    fuel_consumed_recent_period = recent_data['fuel'].iloc[0] - recent_data['fuel'].iloc[-1]
                    time_elapsed_recent_period = recent_data['timestamp_sec'].iloc[-1] - recent_data['timestamp_sec'].iloc[0]
                    if time_elapsed_recent_period > 0 and fuel_consumed_recent_period > 0:
                        avg_consumption_L_per_sec = fuel_consumed_recent_period / time_elapsed_recent_period
                        current_fuel = df_analytics['fuel'].iloc[-1]
                        if avg_consumption_L_per_sec > 0:
                            time_to_empty_sec = current_fuel / avg_consumption_L_per_sec
                            fuel_pred_info = f"{time_to_empty_sec/60:.1f} mins ({time_to_empty_sec/210:.1f} laps)" 
                        else:
                            fuel_pred_info = "Stable/Zero consumption"
                    else:
                        fuel_pred_info = "Not enough recent fuel data"
                else:
                    fuel_pred_info = "Missing fuel data"
            else: # Fallback if no history or not enough points
                tire_pred_info = "N/A (no history)"
                fuel_pred_info = "N/A (no history)"

            st.markdown(f"""
                <div class="analytics-card">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                        <div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Tire Wear Prediction</div>
                            <div style="font-size: 1.2rem; font-weight: 700;">{tire_pred_info}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">Fuel Prediction</div>
                            <div style="font-size: 1.2rem; font-weight: 700;">{fuel_pred_info}</div>
                        </div>
                    </div>
                    <div style="background: rgba(0,0,0,0.2); border-radius: 8px; padding: 1rem;">
                        <div style="font-size: 0.9rem; opacity: 0.8;">Optimal Pit Window (Sim. Rec)</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #FF6B35;">{live_data.get('strategy', {}).get('next_pit_recommendation', 'N/A')} laps</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🏁 STINT PERFORMANCE ANALYSIS")
            
            if not st.session_state.performance_history:
                st.info("No historical data yet for stint analysis.")
            else:
                df_analytics = pd.DataFrame(st.session_state.performance_history)
                # Check for fuel column before attempting diff
                if 'fuel' in df_analytics.columns:
                    df_analytics['fuel_diff'] = df_analytics['fuel'].diff()
                    # A pit stop is typically a large, sudden increase in fuel level.
                    # We look for a positive change greater than a threshold (e.g., 10L, assuming a minimum pit stop refill).
                    pit_stops_indices = df_analytics[(df_analytics['fuel_diff'] > 10) & (df_analytics['fuel_diff'].notna())].index.tolist()
                else:
                    pit_stops_indices = []

                if pit_stops_indices:
                    st.write(f"**Detected Pit Stops:** {len(pit_stops_indices)}")
                    
                    stint_analysis = []
                    current_stint_start_idx = 0
                    
                    # Add the end of the data as a "virtual" pit stop to analyze the last stint
                    all_pit_indices = pit_stops_indices + [len(df_analytics) - 1]
                    all_pit_indices = sorted(list(set(all_pit_indices))) # Remove duplicates
                    
                    for i, pit_idx in enumerate(all_pit_indices):
                        if pit_idx < current_stint_start_idx: # Skip if this pit stop is before the current stint start
                            continue
                        
                        stint_data = df_analytics.iloc[current_stint_start_idx:pit_idx+1]
                        
                        if len(stint_data) > 5: # Require at least a few data points for meaningful analysis
                            avg_lap_time_stint = stint_data['lap_time'].mean() if 'lap_time' in stint_data.columns and stint_data['lap_time'].any() else np.nan
                            
                            if 'fuel' in stint_data.columns:
                                fuel_at_start = stint_data['fuel'].iloc[0]
                                fuel_at_end = stint_data['fuel'].iloc[-1]
                                fuel_used_in_stint = fuel_at_start - fuel_at_end
                                if fuel_used_in_stint < 0: # If fuel increased, it was a pit-in lap, exclude from fuel used
                                    fuel_used_in_stint = 0
                            else:
                                fuel_used_in_stint = np.nan

                            stint_analysis.append({
                                'Stint': i + 1,
                                'Start Time (s)': df_analytics['timestamp_sec'].iloc[current_stint_start_idx],
                                'End Time (s)': df_analytics['timestamp_sec'].iloc[pit_idx],
                                'Data Points': len(stint_data), 
                                'Avg Lap Time': f"{avg_lap_time_stint:.3f}s" if not np.isnan(avg_lap_time_stint) and avg_lap_time_stint > 0 else "N/A",
                                'Fuel Used (L)': f"{fuel_used_in_stint:.1f}" if not np.isnan(fuel_used_in_stint) else "N/A",
                            })
                        current_stint_start_idx = pit_idx + 1 # Start next stint from next data point
                    
                    if stint_analysis:
                        st.dataframe(pd.DataFrame(stint_analysis), use_container_width=True)
                    else:
                        st.info("Not enough data to analyze stints between detected pit stops.")
                else:
                    st.info("No pit stops detected yet in history for stint analysis. (Need fuel level changes > 10L)")
            
            st.markdown("### 📊 PERFORMANCE BENCHMARKING")
            with st.container():
                fig = px.bar(analytics_data, x='metric', y=['current', 'target'], 
                            barmode='group', title='PERFORMANCE VS TARGET',
                            color_discrete_map={'current': '#1E88E5', 'target': '#FF6B35'},
                            labels={'value': 'Score', 'metric': 'Metric'})
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig, use_container_width=True, key=f"analytics_performance_benchmark_{st.session_state.chart_counter}")
    
    # Tab 6: Race Control
    with tab6:
        st.subheader("🏆 RACE CONTROL CENTER")
        
        # Race Status Overview (using data from generate_race_control_data_mock, which takes live_data)
        race_col1, race_col2 = st.columns(2)
        
        with race_col1:
            st.markdown("### 🏁 RACE INFORMATION")
            with st.container():
                st.markdown(f"""
                    <div class="race-control-panel">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Session</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['race_info'].get('session_type', 'N/A').title()}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Time of Day</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['race_info'].get('time_of_day', 'N/A')}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Session Hour</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['race_info'].get('current_hour', 'N/A')}/24</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Is Night</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{'Yes' if race_control_data['race_info'].get('is_night') else 'No'}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Elapsed</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{timedelta(seconds=int(race_control_data['race_info'].get('race_time_elapsed_sec', 0)))}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Remaining</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{timedelta(seconds=int(race_control_data['race_info'].get('race_time_remaining_sec', 0)))}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Safety Car</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: {'#F44336' if race_control_data['race_info'].get('safety_car_active') else '#4CAF50'}">
                                    {'Deployed' if race_control_data['race_info'].get('safety_car_active') else 'Not Deployed'}
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        with race_col2:
            st.markdown("### 🌡️ ENVIRONMENTAL CONDITIONS")
            with st.container():
                st.markdown(f"""
                    <div class="race-control-panel">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Weather</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['environmental'].get('current_weather', 'N/A').replace('_', ' ').title()}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Rain Intensity</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['environmental'].get('rain_intensity', 'N/A')}/3</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Ambient Temp</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['environmental'].get('ambient_temp_C', 'N/A')}°C</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Track Temp</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['environmental'].get('track_temp_C', 'N/A')}°C</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Track Grip</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['environmental'].get('track_grip_level', 'N/A') * 100:.0f}%</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Visibility</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['environmental'].get('visibility_level', 'N/A') * 100:.0f}%</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Current Sector</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['environmental'].get('current_track_segment', 'N/A').replace('_', ' ').title()}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Wind Speed</div>
                                <div style="font-size: 1.2rem; font-weight: 700;">{race_control_data['environmental'].get('wind_speed_kmh', 'N/A')} km/h</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Competitor Analysis
        st.markdown("---")
        st.subheader("🏎️ COMPETITOR ANALYSIS")
        
        competitors = race_control_data.get('competitors', [])
        if competitors:
            df_competitors = pd.DataFrame(competitors)
            
            # Highlight "our car"
            def highlight_our_car(s):
                if s['name'] == live_data.get('car', {}).get('name', "Our Car #22"):
                    return ['background-color: rgba(255, 107, 53, 0.2); font-weight: bold; color: white;'] * len(s)
                return [''] * len(s)

            st.dataframe(
                df_competitors[['current_position', 'name', 'gap_to_leader_sec', 
                              'last_lap_time_sec', 'fuel_level_liters', 'tire_age_laps', 'pit_status']].style.format({
                    'gap_to_leader_sec': '{:.1f}s',
                    'last_lap_time_sec': '{:.1f}s',
                    'fuel_level_liters': '{:.1f}L',
                    'tire_age_laps': '{:.0f}'
                }).apply(highlight_our_car, axis=1),
                use_container_width=True
            )
            
            if len(df_competitors) > 1:
                # Filter top 10 competitors for the bar chart
                df_top_10_competitors = df_competitors.head(10)
                fig_gaps = px.bar(df_top_10_competitors, 
                                x='name', 
                                y='gap_to_leader_sec',
                                title='🏁 GAP TO LEADER ANALYSIS (TOP 10)',
                                color='gap_to_leader_sec',
                                color_continuous_scale='Viridis')
                fig_gaps.update_layout(xaxis_tickangle=-45,
                                        plot_bgcolor='rgba(0,0,0,0)', 
                                        paper_bgcolor='rgba(0,0,0,0)', 
                                        font=dict(color='white'))
                st.plotly_chart(fig_gaps, use_container_width=True, key=f"race_control_gap_chart_{st.session_state.chart_counter}")
        else:
            st.info("No competitor data available yet.")
        
        # Strategy Insights (from mock data, incorporating live sim strategy)
        st.markdown("---")
        st.subheader("🎯 STRATEGY INSIGHTS")
        
        sim_strategy = race_control_data.get('strategy', {})
        if sim_strategy:
            col_strat1, col_strat2, col_strat3 = st.columns(3)
            
            with col_strat1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; opacity: 0.8;">Current Strategy</div>
                        <div style="font-size: 1.5rem; font-weight: 700;">{sim_strategy.get('current_strategy', 'N/A').title()}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_strat2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; opacity: 0.8;">Fuel Target Laps</div>
                        <div style="font-size: 1.5rem; font-weight: 700;">{sim_strategy.get('fuel_target_laps', 'N/A')}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_strat3:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; opacity: 0.8;">Next Pit Recommendation</div>
                        <div style="font-size: 1.5rem; font-weight: 700;">{sim_strategy.get('next_pit_recommendation', 'N/A')}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            col_strat4, col_strat5 = st.columns(2)
            
            with col_strat4:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; opacity: 0.8;">Tire Change Recommended</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {'#4CAF50' if sim_strategy.get('tire_change_recommended', False) else '#F44336'}">
                            {'Yes' if sim_strategy.get('tire_change_recommended', False) else 'No'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_strat5:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 0.9rem; opacity: 0.8;">Driver Change Due</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: {'#F44336' if sim_strategy.get('driver_change_due', False) else '#4CAF50'}">
                            {'Yes' if sim_strategy.get('driver_change_due', False) else 'No'}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No strategy insights available from simulator.")
        
        # Performance Insights (using actual history from simulator)
        st.markdown("---")
        st.subheader("📊 CURRENT SESSION PERFORMANCE SUMMARY")
        
        if telemetry_history and len(telemetry_history) > 5:
            df_summary = pd.json_normalize(telemetry_history)
            df_summary.columns = [col.replace('car.', '').replace('environmental.', '') 
                                for col in df_summary.columns]
            
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                best_lap = df_summary['last_lap_time_sec'].min() if 'last_lap_time_sec' in df_summary.columns and df_summary['last_lap_time_sec'].any() else 0
                st.metric("🏆 BEST LAP", f"{best_lap:.3f}s" if best_lap > 0 else "N/A")
            
            with summary_col2:
                avg_speed = df_summary['speed_kmh'].mean() if 'speed_kmh' in df_summary.columns and df_summary['speed_kmh'].any() else 0
                st.metric("💨 AVG SPEED", f"{avg_speed:.1f} km/h" if not np.isnan(avg_speed) and avg_speed > 0 else "N/A")
            
            with summary_col3:
                if 'fuel_consumption_current_L_per_lap' in df_summary.columns and df_summary['fuel_consumption_current_L_per_lap'].any():
                    avg_fuel_economy = df_summary['fuel_consumption_current_L_per_lap'].mean()
                    st.metric("⛽ AVG FUEL ECON", f"{avg_fuel_economy:.2f} L/lap" if not np.isnan(avg_fuel_economy) and avg_fuel_economy > 0 else "N/A")
                else:
                    st.metric("⛽ AVG FUEL ECON", "N/A")
                
            with summary_col4:
                tire_cols_exist = [f'tire_temp_{t}_C' for t in ['FL', 'FR', 'RL', 'RR']]
                available_tire_cols = [col for col in tire_cols_exist if col in df_summary.columns and df_summary[col].any()]
                if available_tire_cols:
                    avg_tire_temp_hist = df_summary[available_tire_cols].mean().mean()
                    st.metric("🌡️ AVG TIRE TEMP", f"{avg_tire_temp_hist:.1f}°C" if not np.isnan(avg_tire_temp_hist) and avg_tire_temp_hist > 0 else "N/A")
                else:
                    st.metric("🌡️ AVG TIRE TEMP", "N/A")

        else:
            st.info("Summary statistics will be displayed when more history is available.")

# Show system status - this part correctly uses the placeholder defined earlier
if live_data:
    status_placeholder.success(
        f"🟢 SYSTEM OPERATIONAL | Last refresh: {datetime.now().strftime('%H:%M:%S')}"
    )
else:
    status_placeholder.error("🔴 WAITING FOR TELEMETRY DATA...")
    st.info("Ensure your simulator (data_simulator.py) and telemetry API (backend.py) are running.")

# Auto-refresh loop
time.sleep(refresh_interval)
st.rerun()
