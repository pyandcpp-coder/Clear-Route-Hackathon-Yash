# import streamlit as st
# import requests
# import json
# import time 
# import pandas as pd 
# import plotly.express as px 

# # --- Configuration ---
# BACKEND_API_BASE_URL = "http://localhost:8000" # Your FastAPI backend for telemetry
# AI_API_BASE_URL = "http://localhost:8001"    # Your new FastAPI backend for AI queries

# # --- Streamlit Page Configuration ---
# st.set_page_config(
#     layout="wide", 
#     page_title="United Autosports RaceBrain AI",
#     initial_sidebar_state="expanded"
# )

# # --- Initialize session state ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "ai_response_cache" not in st.session_state:
#     st.session_state.ai_response_cache = {"strategy_recommendation": "AI could not generate a recommendation.", "confidence_score": 0.0, "priority_actions": [], "anomaly_report": {}} # Initialize with fallback
# if "ai_query_input_value" not in st.session_state: 
#     st.session_state.ai_query_input_value = ""
# if "ai_query_submitted" not in st.session_state: 
#     st.session_state.ai_query_submitted = False

# # --- Custom Functions to Interact with Backends ---
# @st.cache_data(ttl=1) # Cache data for 1 second to reduce API calls on reruns
# def get_live_data():
#     try:
#         response = requests.get(f"{BACKEND_API_BASE_URL}/live_data")
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         # st.error(f"Failed to fetch live data from telemetry backend: {e}") # Suppress frequent error messages
#         return None

# @st.cache_data(ttl=5) # Cache history for 5 seconds
# def get_telemetry_history_ui(limit=50): 
#     try:
#         response = requests.get(f"{BACKEND_API_BASE_URL}/telemetry_history?limit={limit}")
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         # st.error(f"Failed to fetch telemetry history from telemetry backend: {e}") # Suppress frequent error messages
#         return None

# def query_race_brain_ai(user_query: str):
#     """Sends user query to the AI API and returns the structured response."""
#     try:
#         response = requests.post(
#             f"{AI_API_BASE_URL}/query_race_brain_ai",
#             json={"user_input": user_query} 
#         )
#         response.raise_for_status()
#         ai_response = response.json() 
#         # --- DEBUG PRINT ---
#         st.sidebar.markdown("---")
#         st.sidebar.write("DEBUG: AI API Raw Response Received:")
#         st.sidebar.json(ai_response) # Display raw JSON for debugging in Streamlit sidebar
#         st.sidebar.markdown("---")
#         # --- END DEBUG PRINT ---
#         return ai_response 
#     except requests.exceptions.RequestException as e:
#         st.error(f"Failed to query RaceBrain AI: {e}")
#         return {"strategy_recommendation": f"Error: AI service unavailable or returned an error: {e}", "confidence_score": 0.0, "priority_actions": [], "anomaly_report": {}}

# # --- Callback for text input submission ---
# def handle_query_submit():
#     st.session_state.ai_query_submitted = True

# # --- Streamlit UI Layout ---

# st.sidebar.title("🏁 Race Control Panel")
# st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/United_Autosports.svg/1200px-United_Autosports.svg.png", width=200) 
# st.sidebar.markdown("---")
# refresh_interval = st.sidebar.slider("UI Refresh Interval (seconds)", 1, 10, 2)
# st.sidebar.markdown(f"Next refresh in: {refresh_interval} seconds")

# # --- Main Content Area with Tabs ---
# tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Dashboard", "📈 Telemetry Trends", "💬 AI Strategist", "🚦 Race Overview"])

# # --- Real-time Data Update Loop ---
# main_placeholder = st.empty() 

# # This outer loop ensures the Streamlit UI continuously updates
# while True:
#     # Fetch data (will be cached for short periods)
#     live_data = get_live_data()
#     telemetry_history = get_telemetry_history_ui()

#     with main_placeholder.container(): # All content within this container will refresh
#         if live_data:
#             # === Tab 1: Live Dashboard ===
#             with tab1:
#                 st.header("📊 Current Race Status")
#                 current_car = live_data.get('car', {})
#                 race_info = live_data.get('race_info', {})
#                 env_info = live_data.get('environmental', {})

#                 col_info1, col_info2, col_info3 = st.columns(3)
#                 with col_info1:
#                     st.metric("Lap Number", current_car.get('lap_number', 'N/A'))
#                     st.metric("Current Driver", current_car.get('current_driver', 'N/A'))
#                     st.metric("Fuel Level", f"{current_car.get('fuel_level_liters', 'N/A')} L")
#                 with col_info2:
#                     st.metric("Race Time", f"{race_info.get('time_of_day', 'N/A')} ({race_info.get('current_hour', 'N/A')}h)")
#                     st.metric("Speed", f"{current_car.get('speed_kmh', 'N/A')} km/h")
#                     st.metric("Last Lap Time", f"{current_car.get('last_lap_time_sec', 'N/A')} s")
#                 with col_info3:
#                     st.metric("Ambient Temp", f"{env_info.get('ambient_temp_C', 'N/A')} °C")
#                     st.metric("Track Temp", f"{env_info.get('track_temp_C', 'N/A')} °C")
#                     st.metric("Weather", env_info.get('current_weather', 'N/A').replace("_", " ").title())
                
#                 st.markdown("---")
#                 st.subheader("Tire Status")
#                 tire_cols = st.columns(4)
#                 tires = ["FL", "FR", "RL", "RR"]
#                 for i, tire in enumerate(tires):
#                     with tire_cols[i]:
#                         st.markdown(f"**{tire}**")
#                         st.write(f"Temp: {current_car.get(f'tire_temp_{tire}_C', 'N/A')}°C")
#                         st.write(f"Pressure: {current_car.get(f'tire_pressure_{tire}_bar', 'N/A')} bar")
#                         st.write(f"Wear: {current_car.get(f'tire_wear_{tire}_percent', 'N/A')}%")
                
#                 st.markdown("---")
#                 st.subheader("Active Anomalies from Simulator")
#                 active_sim_anomalies = live_data.get('active_anomalies', [])
#                 if active_sim_anomalies:
#                     for anomaly in active_sim_anomalies:
#                         st.error(f"🚨 {anomaly.get('severity', 'UNKNOWN').upper()} ANOMALY: {anomaly.get('message', 'No message')}")
#                 else:
#                     st.info("🟢 No active anomalies reported by simulator.")

#             # === Tab 2: Detailed Telemetry Trends ===
#             with tab2:
#                 st.header("📈 Telemetry Trends Over Time")
#                 if telemetry_history:
#                     # Normalize the 'car' data for easier plotting
#                     df_history_normalized = pd.json_normalize(telemetry_history)
                    
#                     # Flatten keys like 'car.fuel_level_liters'
#                     df_history_normalized.columns = [col.replace('car.', '').replace('environmental.', '') for col in df_history_normalized.columns]

#                     st.subheader("Fuel Level (Liters)")
#                     fig_fuel = px.line(df_history_normalized, x='timestamp_simulated_sec', y='fuel_level_liters', title="Fuel Level Over Time")
#                     st.plotly_chart(fig_fuel, use_container_width=True)

#                     st.subheader("Tire Temperatures (°C)")
#                     fig_tires = px.line(df_history_normalized, x='timestamp_simulated_sec', 
#                                          y=[f'tire_temp_{t}_C' for t in tires], 
#                                          title="Tire Temperatures Over Time")
#                     st.plotly_chart(fig_tires, use_container_width=True)
                    
#                     st.subheader("Lap Time (Seconds)")
#                     fig_lap_time = px.line(df_history_normalized, x='timestamp_simulated_sec', y='last_lap_time_sec', title="Last Lap Time Over Time")
#                     st.plotly_chart(fig_lap_time, use_container_width=True)
#                 else:
#                     st.info("No historical telemetry data available yet for trends.")

#             # === Tab 3: AI Strategist ===
#             with tab3:
#                 st.header("💬 Race Strategist AI")
                
#                 # Display current AI response prominently
#                 if st.session_state.ai_response_cache:
#                     ai_res = st.session_state.ai_response_cache
#                     st.markdown("---")
#                     st.subheader("Latest AI Strategic Recommendation:")
#                     # --- CRITICAL: Display the full string as-is with st.write or st.markdown ---
#                     # Using st.text_area with value= and disabled=True is good for long, pre-formatted text
#                     st.text_area(
#                         "Full Recommendation:", 
#                         value=ai_res['strategy_recommendation'], 
#                         height=500, # Increased height to accommodate full output
#                         disabled=True,
#                         key="full_recommendation_display" # Unique key
#                     )
#                     # --- END CRITICAL FIX ---

#                     st.write(f"Confidence: **{ai_res['confidence_score']:.2f}/1.0**")
#                     if ai_res['priority_actions']:
#                         st.write("Priority Actions: " + ", ".join(ai_res['priority_actions']))
                    
#                     # Display detailed anomaly report if it exists
#                     if ai_res['anomaly_report'] and ai_res['anomaly_report'].get('priority_level') != 'NONE':
#                         with st.expander(f"Detailed Anomaly Report ({ai_res['anomaly_report'].get('priority_level', 'UNKNOWN')} Priority)"):
#                             st.json(ai_res['anomaly_report'])
#                     st.markdown("---")

#                 # Input for new query
#                 user_question = st.text_input(
#                     "Engineer's Query:", 
#                     value=st.session_state.ai_query_input_value, 
#                     key="ai_query_text_input", 
#                     on_change=handle_query_submit 
#                 )
                
#                 # Button to trigger AI
#                 if st.button("Ask RaceBrain AI") or st.session_state.ai_query_submitted:
#                     if user_question:
#                         st.session_state.ai_query_input_value = "" # Clear input before rerun
#                         st.session_state.ai_query_submitted = False 

#                         with st.spinner("RaceBrain AI is thinking..."):
#                             ai_output = query_race_brain_ai(user_question)
#                             if ai_output:
#                                 st.session_state.ai_response_cache = ai_output 
#                                 st.session_state.chat_history.append({"role": "user", "content": user_question})
#                                 # Store the full string in chat history as well
#                                 st.session_state.chat_history.append({"role": "ai", "content": ai_output['strategy_recommendation']})
#                                 st.rerun() # Force a rerun to update UI with new state
#                             else:
#                                 st.error("Failed to get response from AI.")
#                     else:
#                         st.warning("Please enter a query for the AI.")
#                         st.session_state.ai_query_submitted = False 
                
#                 st.subheader("Chat History")
#                 for message in reversed(st.session_state.chat_history):
#                     if message["role"] == "user":
#                         st.markdown(f"**Engineer:** {message['content']}")
#                     else:
#                         st.markdown(f"**RaceBrain AI:** {message['content']}")

#             # === Tab 4: Race Overview ===
#             with tab4:
#                 st.header("🚦 Race Overview & Competitors")
#                 competitors = live_data.get('competitors', [])
#                 st.subheader("Competitor Positions")
#                 df_competitors = pd.DataFrame(competitors)
#                 if not df_competitors.empty:
#                     # Ensure columns exist before sorting/selecting
#                     required_cols = ['current_position', 'name', 'gap_to_leader_sec', 'last_lap_time_sec', 'fuel_level_liters', 'tire_age_laps', 'pit_status']
#                     for col in required_cols:
#                         if col not in df_competitors.columns:
#                             df_competitors[col] = 'N/A' # Add missing columns with default value

#                     df_competitors = df_competitors[required_cols].sort_values(by='current_position')
#                     st.dataframe(df_competitors, use_container_width=True)
#                 else:
#                     st.info("No competitor data available yet.")
                
#                 st.subheader("Current Weather & Track")
#                 env_data = live_data.get('environmental', {})
#                 track_data = live_data.get('track_info', {})
#                 st.write(f"**Weather:** {env_data.get('current_weather', 'N/A').replace('_', ' ').title()}")
#                 st.write(f"**Ambient Temp:** {env_data.get('ambient_temp_C', 'N/A')}°C")
#                 st.write(f"**Track Temp:** {env_data.get('track_temp_C', 'N/A')}°C")
#                 st.write(f"**Track Grip:** {env_data.get('track_grip_level', 'N/A')}")
#                 st.write(f"**Visibility:** {env_data.get('visibility_level', 'N/A')}")
#                 st.write(f"**Current Sector:** {current_car.get('current_track_segment', 'N/A').replace('_', ' ').title()}")
#                 st.write(f"**Safety Car Active:** {'Yes' if live_data.get('race_control', {}).get('safety_car_active', False) else 'No'}")


#             status_message_placeholder = st.sidebar.empty()
#             status_message_placeholder.success(f"UI refreshed: {time.strftime('%H:%M:%S')}")
#         else:
#             status_message_placeholder = st.sidebar.empty()
#             status_message_placeholder.error("Waiting for telemetry data...")
#             st.info("Ensure your simulator (data_simulator.py) and telemetry API (backend_api.py) are running.")

#     # Pause before next refresh and force rerun for continuous updates
#     time.sleep(refresh_interval)
#     st.rerun() 


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
import altair as alt 
from scipy import stats 

# --- Configuration ---
BACKEND_API_BASE_URL = "http://localhost:8000" # Your FastAPI backend for telemetry
AI_API_BASE_URL = "http://localhost:8001"    # Your new FastAPI backend for AI queries

# --- Streamlit Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="United Autosports RaceBrain AI Pro",
    initial_sidebar_state="expanded",
    page_icon="🏁"
)

# Custom CSS for professional racing theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E, #FFD23F);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .status-good { 
        background: linear-gradient(135deg, #4CAF50, #45a049) !important; 
    }
    .status-warning { 
        background: linear-gradient(135deg, #FF9800, #F57C00) !important; 
    }
    .status-critical { 
        background: linear-gradient(135deg, #F44336, #D32F2F) !important; 
    }
    
    .tire-visual {
        border: 3px solid #333;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        background: linear-gradient(45deg, #2C3E50, #4A6741);
        color: white;
    }
    
    .ai-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .race-position {
        display: flex;
        align-items: center;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50, #3498DB);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Enhanced Session State ---
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
if "alerts_history" not in st.session_state:
    st.session_state.alerts_history = []
if "selected_driver" not in st.session_state:
    st.session_state.selected_driver = "All Drivers"

# --- Enhanced API Functions ---
@st.cache_data(ttl=1) # Cache data for 1 second to reduce API calls on reruns
def get_live_data():
    try:
        response = requests.get(f"{BACKEND_API_BASE_URL}/live_data")
        response.raise_for_status()
        data = response.json()
        
        # Store performance metrics for trending
        if data and 'car' in data:
            # Ensure timestamp exists before appending
            current_timestamp = data.get('timestamp_simulated_sec', time.time()) 
            
            performance_point = {
                'timestamp_sec': current_timestamp, # Use simulated timestamp for x-axis
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
                'brake_percent': data['car'].get('brake_percent', 0), # Add brake_percent
                'suspension_travel_FL_mm': data['car'].get('suspension_travel_FL_mm', 0), # Add suspension data
                'suspension_travel_FR_mm': data['car'].get('suspension_travel_FR_mm', 0),
                'suspension_travel_RL_mm': data['car'].get('suspension_travel_RL_mm', 0),
                'suspension_travel_RR_mm': data['car'].get('suspension_travel_RR_mm', 0),
            }
            st.session_state.performance_history.append(performance_point)
            
            # Keep only last 100 points
            if len(st.session_state.performance_history) > 100:
                st.session_state.performance_history.pop(0)
        
        return data
    except requests.exceptions.RequestException as e:
        # st.error(f"Failed to fetch live data from telemetry backend: {e}") # Suppress frequent error messages
        return None

@st.cache_data(ttl=5) # Cache history for 5 seconds
def get_telemetry_history_ui(limit=50): 
    try:
        response = requests.get(f"{BACKEND_API_BASE_URL}/telemetry_history?limit={limit}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # st.error(f"Failed to fetch telemetry history from telemetry backend: {e}") # Suppress frequent error messages
        return None

def query_race_brain_ai(user_query: str):
    """Sends user query to the AI API and returns the structured response."""
    try:
        response = requests.post(
            f"{AI_API_BASE_URL}/query_race_brain_ai",
            json={"user_input": user_query},
            timeout=60 # Increased timeout for LLM calls as 70B can be slow
        )
        response.raise_for_status()
        ai_response = response.json()
        
        # Add timestamp to AI response
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

# --- Enhanced Helper Functions ---
def get_tire_status_color(temp, wear):
    """Determine tire status color based on temperature and wear."""
    if temp > 110 or wear > 80:
        return "status-critical"
    elif temp > 90 or wear > 60:
        return "status-warning"
    else:
        return "status-good"

def create_enhanced_gauge(value, title, min_val=0, max_val=100, unit="", color_ranges=None):
    """Create an enhanced gauge chart."""
    if color_ranges is None:
        color_ranges = [(0, 50, "green"), (50, 80, "yellow"), (80, 100, "red")]
    
    # Handle NaN values for gauges
    if value is None or np.isnan(value):
        value = min_val # Display min value or handle as 'N/A' in title if preferred
        title = f"{title} (N/A)" # Indicate no data

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{title} ({unit})"},
        delta = {'reference': (max_val - min_val) * 0.5, 'relative': True}, # Make delta relative
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [r[0], r[1]], 'color': r[2]} for r in color_ranges
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9 # Threshold at 90% of max
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def handle_query_submit():
    st.session_state.ai_query_submitted = True

# --- Enhanced UI Layout ---
st.markdown('<div class="main-header">🏁 United Autosports RaceBrain AI Pro 🏁</div>', unsafe_allow_html=True)

# Enhanced Sidebar
st.sidebar.title("🏎️ Race Control Center")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/United_Autosports.svg/1200px-United_Autosports.svg.png", width=200)
st.sidebar.markdown("---")

# Advanced Settings
with st.sidebar.expander("⚙️ Dashboard Settings"):
    refresh_interval = st.slider("Refresh Interval (sec)", 1, 10, 2)
    show_debug = st.checkbox("Show Debug Info", False)
    telemetry_limit = st.slider("Telemetry History Points", 20, 200, 50)

# Driver Selection
available_drivers = ["All Drivers", "Phil Hanson", "Filipe Albuquerque", "Will Owen"] 
st.session_state.selected_driver = st.sidebar.selectbox("👤 Filter by Driver", available_drivers)

# System Status
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 System Status")
status_placeholder = st.sidebar.empty()

# --- Enhanced Tab Layout ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚦 Live Race", "📊 Performance", "🧠 AI Strategist", 
    "🔧 Technical", "📈 Analytics", "🏆 Race Control"
])

# Main update loop with enhanced features
main_placeholder = st.empty()

while True:
    live_data = get_live_data()
    telemetry_history = get_telemetry_history_ui(telemetry_limit)
    
    with main_placeholder.container(): # All content within this container will refresh
        if live_data:
            current_car = live_data.get('car', {})
            race_info = live_data.get('race_info', {})
            env_info = live_data.get('environmental', {})
            
            # === Tab 1: Enhanced Live Race Dashboard ===
            with tab1:
                st.subheader("🏁 Live Race Command Center")
                
                # Key Metrics Row
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("🏎️ Position", current_car.get('position_in_class', 'N/A'), 
                             delta=current_car.get('position_change', 0)) # Position change might not be in sim output yet
                with col2:
                    st.metric("⏱️ Lap", current_car.get('lap_number', 'N/A'))
                with col3:
                    speed = current_car.get('speed_kmh', 0)
                    st.metric("🚀 Speed", f"{speed:.1f} km/h", 
                             delta=f"{speed - 250:.1f}" if speed else None) # Delta vs 250km/h baseline
                with col4:
                    fuel = current_car.get('fuel_level_liters', 0)
                    st.metric("⛽ Fuel", f"{fuel:.1f} L", 
                             delta=f"-{75-fuel:.1f}" if fuel else None) # Delta vs 75L tank
                with col5:
                    lap_time = current_car.get('last_lap_time_sec', 0)
                    st.metric("⏰ Last Lap", f"{lap_time:.2f}s" if lap_time else "N/A")
                
                # Live Gauges
                st.markdown("---")
                gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
                
                with gauge_col1:
                    speed_gauge = create_enhanced_gauge(
                        current_car.get('speed_kmh', 0), "Speed", 0, 350, "km/h", 
                        [(0, 100, "lightgray"), (100, 250, "green"), (250, 350, "red")]
                    )
                    st.plotly_chart(speed_gauge, use_container_width=True)
                
                with gauge_col2:
                    fuel_gauge = create_enhanced_gauge(
                        current_car.get('fuel_level_liters', 0), "Fuel Level", 0, 75, "L", 
                        [(0, 15, "red"), (15, 30, "yellow"), (30, 75, "green")]
                    )
                    st.plotly_chart(fuel_gauge, use_container_width=True)
                
                with gauge_col3:
                    avg_tire_temp = np.mean([
                        current_car.get(f'tire_temp_{t}_C', 0) for t in ['FL', 'FR', 'RL', 'RR']
                    ]) if all(current_car.get(f'tire_temp_{t}_C') is not None for t in ['FL', 'FR', 'RL', 'RR']) else 0 
                    temp_gauge = create_enhanced_gauge(
                        avg_tire_temp, "Avg Tire Temp", 0, 150, "°C",
                        [(0, 80, "green"), (80, 120, "yellow"), (120, 150, "red")]
                    )
                    st.plotly_chart(temp_gauge, use_container_width=True)
                
                # Enhanced Tire Status
                st.markdown("---")
                st.subheader("🏎️ Tire Status Matrix")
                tire_cols = st.columns(4)
                tires = ["FL", "FR", "RL", "RR"]
                tire_positions = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
                
                for i, (tire, position) in enumerate(zip(tires, tire_positions)):
                    with tire_cols[i]:
                        temp = current_car.get(f'tire_temp_{tire}_C', 0)
                        pressure = current_car.get(f'tire_pressure_{tire}_bar', 0)
                        wear = current_car.get(f'tire_wear_{tire}_percent', 0)
                        
                        status_class = get_tire_status_color(temp, wear)
                        
                        st.markdown(f"""
                        <div class="tire-visual {status_class}">
                            <h4>{position}</h4>
                            <p>🌡️ {temp:.1f}°C</p>
                            <p>⚡ {pressure:.1f} bar</p>
                            <p>⚠️ {wear:.1f}% wear</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Real-time Alerts
                st.markdown("---")
                st.subheader("🚨 Live Alerts & Anomalies")
                active_anomalies = live_data.get('active_anomalies', [])
                if active_anomalies:
                    for anomaly in active_anomalies:
                        severity = anomaly.get('severity', 'UNKNOWN').upper()
                        if severity == 'CRITICAL':
                            st.error(f"🔴 CRITICAL: {anomaly.get('message', 'Unknown issue')}")
                        elif severity == 'HIGH': 
                            st.warning(f"🟠 HIGH: {anomaly.get('message', 'Unknown issue')}")
                        elif severity == 'MEDIUM':
                            st.warning(f"🟡 MEDIUM: {anomaly.get('message', 'Unknown issue')}")
                        else: 
                            st.info(f"🔵 {severity}: {anomaly.get('message', 'Unknown issue')}")
                else:
                    st.success("🟢 All systems nominal - No active alerts")
            
            # === Tab 2: Enhanced Performance Analytics ===
            with tab2:
                st.subheader("📊 Performance Analytics Dashboard")
                
                if st.session_state.performance_history:
                    df_perf = pd.DataFrame(st.session_state.performance_history)
                    
                    # Performance Overview
                    perf_col1, perf_col2 = st.columns(2)
                    
                    with perf_col1:
                        # Lap Time Trend
                        fig_lap = px.line(df_perf, x='timestamp_sec', y='lap_time', 
                                         title='🏁 Lap Time Evolution',
                                         labels={'timestamp_sec': 'Time (s)', 'lap_time': 'Lap Time (s)'},
                                         color_discrete_sequence=['#FF6B35'])
                        fig_lap.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_lap, use_container_width=True)
                        
                        # Speed vs Fuel Correlation
                        fig_scatter = px.scatter(df_perf, x='fuel', y='speed',
                                               title='⛽ Fuel Level vs Speed Analysis',
                                               labels={'fuel': 'Fuel (L)', 'speed': 'Speed (km/h)'},
                                               color='tire_temp_FL_C', 
                                               color_continuous_scale='Viridis',
                                               hover_data=['lap_time', 'fuel_consumption'])
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with perf_col2:
                        # Multi-metric Performance
                        fig_multi = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Speed Over Time', 'Fuel Consumption Over Time'), 
                            vertical_spacing=0.15
                        )
                        
                        fig_multi.add_trace(
                            go.Scatter(x=df_perf['timestamp_sec'], y=df_perf['speed'],
                                     name='Speed (km/h)', line=dict(color='#F7931E')),
                            row=1, col=1
                        )
                        
                        fig_multi.add_trace(
                            go.Scatter(x=df_perf['timestamp_sec'], y=df_perf['fuel_consumption'], 
                                     name='Fuel Consumption (L/lap)', line=dict(color='#FFD23F')),
                            row=2, col=1
                        )
                        
                        fig_multi.update_layout(height=500, showlegend=True,
                                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                                font=dict(color='white'))
                        st.plotly_chart(fig_multi, use_container_width=True)
                
                # Performance Statistics
                st.markdown("---")
                st.subheader("📈 Session Statistics")
                if telemetry_history:
                    df_history = pd.json_normalize(telemetry_history)
                    df_history.columns = [col.replace('car.', '').replace('environmental.', '') 
                                        for col in df_history.columns]
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        best_lap = df_history['last_lap_time_sec'].min() if 'last_lap_time_sec' in df_history.columns else 0
                        st.metric("🏆 Best Lap Time", f"{best_lap:.3f}s" if best_lap > 0 else "N/A")
                    with stat_col2:
                        avg_speed = df_history['speed_kmh'].mean() if 'speed_kmh' in df_history.columns else 0
                        st.metric("📊 Avg Speed", f"{avg_speed:.1f} km/h" if not np.isnan(avg_speed) else "N/A")
                    with stat_col3:
                        if 'fuel_level_liters' in df_history.columns and len(df_history) > 1:
                            fuel_used = df_history['fuel_level_liters'].iloc[0] - df_history['fuel_level_liters'].iloc[-1]
                            st.metric("⛽ Total Fuel Used", f"{fuel_used:.1f}L" if fuel_used > 0 else "N/A")
                        else:
                            st.metric("⛽ Total Fuel Used", "N/A")
                    with stat_col4:
                        total_laps = df_history['lap_number'].max() if 'lap_number' in df_history.columns else 0
                        st.metric("🔄 Total Laps", str(int(total_laps)) if not np.isnan(total_laps) else "N/A")
                else:
                    st.info("Statistics will appear here once telemetry history is available.")
            
            # === Tab 3: Enhanced AI Strategist ===
            with tab3:
                st.subheader("🧠 RaceBrain AI Strategic Command")
                
                # AI Response Display
                if st.session_state.ai_response_cache and st.session_state.ai_response_cache.get('strategy_recommendation') != "AI could not generate a recommendation.":
                    ai_res = st.session_state.ai_response_cache
                    
                    # Confidence Indicator
                    confidence = ai_res.get('confidence_score', 0)
                    conf_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                    
                    st.markdown(f"""
                    <div class="ai-response">
                        <h3>🎯 Latest Strategic Recommendation {conf_color}</h3>
                        <p><strong>Confidence Level:</strong> {confidence:.1%}</p>
                        <p><strong>Generated:</strong> {ai_res.get('timestamp', 'Unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Main AI Response (displaying full raw output as requested)
                    st.text_area(
                        "📋 Full Strategic Analysis:",
                        value=ai_res['strategy_recommendation'],
                        height=500,
                        disabled=True,
                        key="ai_full_response"
                    )
                    
                    # Priority Actions
                    if ai_res.get('priority_actions'):
                        st.subheader("⚡ Immediate Action Items")
                        for i, action in enumerate(ai_res['priority_actions'], 1):
                            st.markdown(f"**{i}.** {action}")
                    
                    # Anomaly Report
                    if ai_res.get('anomaly_report') and ai_res['anomaly_report'].get('priority_level') != 'NONE':
                        with st.expander("🔍 Detailed Technical Analysis", expanded=True):
                            priority = ai_res['anomaly_report'].get('priority_level', 'UNKNOWN')
                            st.markdown(f"**Priority Level:** {priority}")
                            st.json(ai_res['anomaly_report'])
                else:
                    st.info("No strategic recommendation generated yet. Ask RaceBrain AI a question below!")
                
                st.markdown("---")
                
                # Enhanced Query Interface
                col_query1, col_query2 = st.columns([3, 1])
                with col_query1:
                    user_question = st.text_input(
                        "💬 Ask RaceBrain AI:",
                        value=st.session_state.ai_query_input_value,
                        placeholder="e.g., 'What's our optimal pit strategy?' or 'Analyze tire degradation'",
                        key="ai_query_input",
                        on_change=handle_query_submit
                    )
                
                with col_query2:
                    ask_button = st.button("🚀 Query AI", type="primary")
                
                # Quick Action Buttons
                st.markdown("**Quick Strategy Queries:**")
                quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
                with quick_col1:
                    if st.button("🏁 Pit Strategy", key="quick_pit"):
                        user_question = "What's our optimal pit window and strategy?"
                        st.session_state.ai_query_submitted = True
                        st.session_state.ai_query_input_value = user_question 
                with quick_col2:
                    if st.button("🏎️ Tire Analysis", key="quick_tire"):
                        user_question = "Analyze current tire performance and degradation"
                        st.session_state.ai_query_submitted = True
                        st.session_state.ai_query_input_value = user_question
                with quick_col3:
                    if st.button("⛽ Fuel Management", key="quick_fuel"):
                        user_question = "Evaluate fuel consumption and stint length"
                        st.session_state.ai_query_submitted = True
                        st.session_state.ai_query_input_value = user_question
                with quick_col4:
                    if st.button("🌦️ Weather Impact", key="quick_weather"):
                        user_question = "How will weather conditions affect our strategy?"
                        st.session_state.ai_query_submitted = True
                        st.session_state.ai_query_input_value = user_question
                
                # Process AI Query
                if ask_button or st.session_state.ai_query_submitted:
                    if user_question:
                        st.session_state.ai_query_submitted = False
                        
                        with st.spinner("🧠 RaceBrain AI analyzing telemetry data..."):
                            ai_output = query_race_brain_ai(user_question)
                            if ai_output:
                                st.session_state.ai_response_cache = ai_output 
                                st.session_state.chat_history.append({
                                    "role": "user", 
                                    "content": user_question,
                                    "timestamp": datetime.now()
                                })
                                st.session_state.chat_history.append({
                                    "role": "ai", 
                                    "content": ai_output['strategy_recommendation'],
                                    "timestamp": datetime.now()
                                })
                                st.session_state.ai_query_input_value = "" 
                                st.rerun() 
                            else:
                                st.error("Failed to get response from AI.")
                    else:
                        st.warning("Please enter a query for RaceBrain AI")
                        st.session_state.ai_query_submitted = False
                
                # Enhanced Chat History
                st.markdown("---")
                st.subheader("💬 Strategy Discussion History")
                if st.session_state.chat_history:
                    for message in reversed(st.session_state.chat_history): 
                        timestamp = message.get('timestamp', datetime.now()).strftime("%H:%M:%S")
                        if message["role"] == "user":
                            st.markdown(f"**🏎️ Engineer [{timestamp}]:** {message['content']}")
                        else:
                            st.markdown(f"**🧠 RaceBrain AI [{timestamp}]:** \n```\n{message['content']}\n```") 
                else:
                    st.info("Start a conversation with RaceBrain AI to see chat history")
            
            # === Tab 4: Technical Deep Dive ===
            with tab4:
                st.subheader("🔧 Technical Systems Monitor")
                
                # System Health Matrix
                tech_col1, tech_col2 = st.columns(2)
                
                with tech_col1:
                    st.markdown("**🔋 Power Unit Status**")
                    oil_temp = current_car.get('oil_temp_C', 0)
                    water_temp = current_car.get('water_temp_C', 0)
                    st.write(f"Oil Temperature: {oil_temp:.1f}°C")
                    st.write(f"Water Temperature: {water_temp:.1f}°C")
                    st.write(f"Engine RPM: {current_car.get('engine_rpm', 'N/A')}")
                    st.write(f"Hybrid Power Output: {current_car.get('hybrid_power_output_kw', 0):.1f} kW")
                    st.write(f"Hybrid Battery: {current_car.get('hybrid_battery_percent', 0):.1f}%")
                    
                    st.markdown("**🛠️ Mechanical Systems**")
                    throttle = current_car.get('throttle_percent', 0)
                    brake = current_car.get('brake_percent', 0)
                    st.write(f"Throttle: {throttle:.1f}%")
                    st.write(f"Brake: {brake:.1f}%")
                    st.write(f"Gear: {current_car.get('gear', 'N/A')}")
                
                with tech_col2:
                    st.markdown("**📡 Telemetry Systems**")
                    st.write(f"Data Link: 🟢 Connected") 
                    st.write(f"GPS Accuracy: 🟢 High") 
                    st.write(f"Sensors: 🟢 All Operational") 
                    
                    st.markdown("**🔄 Vehicle Dynamics**")
                    st.write(f"Downforce Setting: N/A (Simulated)") 
                    st.write(f"Brake Balance: N/A (Simulated)") 
                    st.write(f"Suspension FL: {current_car.get('suspension_travel_FL_mm', 0):.1f} mm")
                    st.write(f"Suspension FR: {current_car.get('suspension_travel_FR_mm', 0):.1f} mm")
                    st.write(f"Suspension RL: {current_car.get('suspension_travel_RL_mm', 0):.1f} mm")
                    st.write(f"Suspension RR: {current_car.get('suspension_travel_RR_mm', 0):.1f} mm")
                
                # Detailed Telemetry Charts
                if st.session_state.performance_history:
                    df_tech = pd.DataFrame(st.session_state.performance_history)
                    
                    tech_chart_col1, tech_chart_col2 = st.columns(2)
                    
                    with tech_chart_col1:
                        # Ensure columns exist and have data
                        if 'oil_temp_C' in df_tech.columns and 'water_temp_C' in df_tech.columns and not df_tech[['oil_temp_C', 'water_temp_C']].isnull().all().all():
                            fig_oil_water = make_subplots(specs=[[{"secondary_y": True}]])
                            fig_oil_water.add_trace(go.Scatter(x=df_tech['timestamp_sec'], y=df_tech['oil_temp_C'], name='Oil Temp (°C)'), secondary_y=False)
                            fig_oil_water.add_trace(go.Scatter(x=df_tech['timestamp_sec'], y=df_tech['water_temp_C'], name='Water Temp (°C)'), secondary_y=True)
                            fig_oil_water.update_layout(title_text="Engine Temperatures",
                                                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                                                        hovermode="x unified")
                            fig_oil_water.update_xaxes(title_text="Time (s)")
                            fig_oil_water.update_yaxes(title_text="Oil Temp (°C)", secondary_y=False)
                            fig_oil_water.update_yaxes(title_text="Water Temp (°C)", secondary_y=True)
                            st.plotly_chart(fig_oil_water, use_container_width=True)
                        else:
                            st.info("No engine temperature data available for plotting.")
                    
                    with tech_chart_col2:
                        suspension_cols = [f'suspension_travel_{t}_mm' for t in ['FL', 'FR', 'RL', 'RR']]
                        # Ensure columns exist and have data
                        if all(col in df_tech.columns for col in suspension_cols) and not df_tech[suspension_cols].isnull().all().all():
                            fig_susp = px.line(df_tech, x='timestamp_sec', 
                                              y=suspension_cols, # Now these columns will exist
                                              title='Suspension Travel (mm)')
                            fig_susp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                                                    hovermode="x unified")
                            st.plotly_chart(fig_susp, use_container_width=True)
                        else:
                            st.info("No suspension travel data available for plotting.")

                    st.markdown("---")
                    st.subheader("Brake & Throttle Analysis")
                    if 'brake_percent' in df_tech.columns and 'throttle_percent' in df_tech.columns:
                        fig_brake_throttle = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_brake_throttle.add_trace(go.Scatter(x=df_tech['timestamp_sec'], y=df_tech['brake_percent'], name='Brake (%)'), secondary_y=False)
                        fig_brake_throttle.add_trace(go.Scatter(x=df_tech['timestamp_sec'], y=df_tech['throttle_percent'], name='Throttle (%)'), secondary_y=True)
                        fig_brake_throttle.update_layout(title_text="Brake & Throttle Application",
                                                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                                                         hovermode="x unified")
                        fig_brake_throttle.update_xaxes(title_text="Time (s)")
                        fig_brake_throttle.update_yaxes(title_text="Brake (%)", secondary_y=False)
                        fig_brake_throttle.update_yaxes(title_text="Throttle (%)", secondary_y=True)
                        st.plotly_chart(fig_brake_throttle, use_container_width=True)
                    else:
                        st.info("No brake or throttle data available for plotting.")

            # === Tab 5: Advanced Analytics ===
            with tab5:
                st.subheader("📈 Advanced Race Analytics")
                
                if st.session_state.performance_history and len(st.session_state.performance_history) > 10: 
                    df_analytics = pd.DataFrame(st.session_state.performance_history)
                    
                    # Performance Correlation Matrix
                    st.subheader("🔗 Performance Correlation Analysis")
                    numeric_cols = df_analytics.select_dtypes(include=[np.number]).columns
                    key_metrics = ['speed', 'lap_time', 'fuel', 
                                 'tire_temp_FL_C', 'tire_temp_FR_C', 'fuel_consumption',
                                 'oil_temp_C', 'water_temp_C', 'brake_percent', # Added these metrics
                                 'suspension_travel_FL_mm'] # Example suspension
                    available_metrics = [col for col in key_metrics if col in numeric_cols and not df_analytics[col].isnull().all()] # Filter out all-NaN columns
                    
                    if len(available_metrics) > 1: # Need at least 2 for correlation
                        corr_matrix = df_analytics[available_metrics].corr()
                        fig_corr = px.imshow(corr_matrix, 
                                           text_auto=True, 
                                           aspect="auto",
                                           title="📊 Performance Metrics Correlation Heatmap",
                                           color_continuous_scale='RdBu')
                        fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Not enough numeric data for correlation analysis.")
                    
                    # Stint Analysis
                    st.markdown("---")
                    st.subheader("🏁 Stint Performance Analysis")
                    
                    df_analytics['fuel_diff'] = df_analytics['fuel'].diff()
                    pit_stops_indices = df_analytics[(df_analytics['fuel_diff'] > 10) & (df_analytics['fuel_diff'].notna())].index.tolist()
                    
                    if pit_stops_indices:
                        st.write(f"**Detected Pit Stops:** {len(pit_stops_indices)}")
                        
                        stint_analysis = []
                        current_stint_start_idx = 0
                        
                        for i, pit_idx in enumerate(pit_stops_indices + [len(df_analytics)]):
                            stint_data = df_analytics.iloc[current_stint_start_idx:pit_idx]
                            
                            if len(stint_data) > 5:  
                                avg_lap_time = stint_data['lap_time'].mean()
                                fuel_at_start = stint_data['fuel'].iloc[0]
                                fuel_at_end = stint_data['fuel'].iloc[-1]
                                fuel_used = fuel_at_start - fuel_at_end
                                
                                laps_in_stint = stint_data['lap_time'].count() 
                                
                                stint_analysis.append({
                                    'Stint': i + 1,
                                    'Data Points': len(stint_data), 
                                    'Avg Lap Time': f"{avg_lap_time:.3f}s" if not np.isnan(avg_lap_time) else "N/A",
                                    'Fuel Used (L)': f"{fuel_used:.1f}" if fuel_used > 0 else "N/A",
                                    'Avg Fuel/Pt (L/point)': f"{fuel_used/len(stint_data):.2f}" if len(stint_data)>0 and fuel_used>0 else "N/A"
                                })
                            current_stint_start_idx = pit_idx
                        
                        if stint_analysis:
                            st.dataframe(pd.DataFrame(stint_analysis), use_container_width=True)
                        else:
                            st.info("Not enough data to analyze stints between detected pit stops.")
                    else:
                        st.info("No pit stops detected yet in history for stint analysis. (Need fuel level changes > 10L)")
                    
                    # Predictive Analytics
                    st.markdown("---")
                    st.subheader("🔮 Predictive Performance Models")
                    
                    if 'tire_wear_FL_percent' in df_analytics.columns and len(df_analytics['tire_wear_FL_percent'].dropna()) > 5:
                        tire_wear_data = df_analytics[['timestamp_sec', 'tire_wear_FL_percent']].dropna()
                        
                        if len(tire_wear_data) >= 2: # stats.linregress needs at least 2 points
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                tire_wear_data['timestamp_sec'], tire_wear_data['tire_wear_FL_percent']
                            )
                            
                            if slope > 0: 
                                time_to_90_percent_wear = (90 - intercept) / slope
                                current_time = df_analytics['timestamp_sec'].iloc[-1]
                                
                                if time_to_90_percent_wear > current_time:
                                    remaining_time_sec = max(0, time_to_90_percent_wear - current_time)
                                    st.info(f"🔍 **Tire Prediction:** Front-left tire predicted to reach 90% wear in ~{remaining_time_sec/60:.1f} minutes of simulated race time.")
                                else:
                                    st.warning(f"🔍 **Tire Warning:** Front-left tire may already be near or above 90% wear (Current: {tire_wear_data['tire_wear_FL_percent'].iloc[-1]:.1f}%).")
                            else:
                                st.info("🔍 **Tire Prediction:** Wear trend is stable or decreasing.")
                        else:
                            st.info("🔍 **Tire Prediction:** Not enough valid data points for tire wear prediction.")
                    else:
                        st.info("🔍 **Tire Prediction:** Not enough historical data for robust tire wear prediction.")
                    
                    # Fuel consumption prediction to empty
                    if 'fuel' in df_analytics.columns and 'fuel_consumption' in df_analytics.columns and len(df_analytics) > 10:
                        recent_data = df_analytics.tail(10) # Look at recent 10 points
                        
                        fuel_consumed_recent_period = recent_data['fuel'].iloc[0] - recent_data['fuel'].iloc[-1]
                        time_elapsed_recent_period = recent_data['timestamp_sec'].iloc[-1] - recent_data['timestamp_sec'].iloc[0]
                        
                        if time_elapsed_recent_period > 0 and fuel_consumed_recent_period > 0:
                            avg_consumption_L_per_sec = fuel_consumed_recent_period / time_elapsed_recent_period
                            current_fuel = df_analytics['fuel'].iloc[-1]
                            
                            if avg_consumption_L_per_sec > 0:
                                time_to_empty_sec = current_fuel / avg_consumption_L_per_sec
                                st.info(f"⛽ **Fuel Prediction:** At current consumption, estimated {time_to_empty_sec/60:.1f} minutes of simulated race time until empty (approx {time_to_empty_sec/210:.1f} laps).") 
                            else:
                                st.info("⛽ **Fuel Prediction:** Fuel consumption stable or zero (car stopped).")
                        else:
                            st.info("⛽ **Fuel Prediction:** Not enough recent data to calculate consumption rate.")
                
                else:
                    st.info("📊 Advanced analytics will be available once sufficient telemetry data is collected (10+ data points)")
            
            # === Tab 6: Race Control Center ===
            with tab6:
                st.subheader("🏆 Race Control & Strategy Center")
                
                # Race Status Overview
                race_col1, race_col2 = st.columns(2)
                
                with race_col1:
                    st.markdown("**🏁 Race Information**")
                    st.write(f"**Session:** {race_info.get('session_type', 'Race').title()}") 
                    st.write(f"**Time of Day:** {race_info.get('time_of_day', 'N/A')}")
                    st.write(f"**Session Hour:** {race_info.get('current_hour', 'N/A')}")
                    st.write(f"**Is Night:** {'Yes' if race_info.get('is_night') else 'No'}")
                    st.write(f"**Race Time Elapsed:** {timedelta(seconds=int(race_info.get('race_time_elapsed_sec', 0)))}")
                    st.write(f"**Race Time Remaining:** {timedelta(seconds=int(race_info.get('race_time_remaining_sec', 0)))}")
                    
                    # Safety status
                    safety_car = live_data.get('race_control', {}).get('safety_car_active', False)
                    safety_status = "🟡 Safety Car Deployed" if safety_car else "🟢 Green Flag Racing"
                    st.write(f"**Race Status:** {safety_status}")
                
                with race_col2:
                    st.markdown("**🌡️ Environmental Conditions**")
                    st.write(f"**Current Weather:** {env_info.get('current_weather', 'N/A').replace('_', ' ').title()}")
                    st.write(f"**Rain Intensity:** {env_info.get('rain_intensity', 'N/A')}/3")
                    st.write(f"**Ambient Temp:** {env_info.get('ambient_temp_C', 'N/A')}°C")
                    st.write(f"**Track Temp:** {env_info.get('track_temp_C', 'N/A')}°C")
                    st.write(f"**Track Grip:** {env_info.get('track_grip_level', 'N/A')}")
                    st.write(f"**Visibility:** {env_info.get('visibility_level', 'N/A')}")
                    st.write(f"**Current Sector:** {current_car.get('current_track_segment', 'N/A').replace('_', ' ').title()}")
                    st.write(f"**Wind Speed/Dir:** {env_info.get('wind_speed_kmh', 'N/A')} km/h from {env_info.get('wind_direction_deg', 'N/A')}°") # Added
                
                # Competitor Analysis
                st.markdown("---")
                st.subheader("🏎️ Field Position & Competitor Analysis")
                
                competitors = live_data.get('competitors', [])
                if competitors:
                    df_competitors = pd.DataFrame(competitors)
                    
                    required_cols = ['current_position', 'name', 'gap_to_leader_sec', 
                                   'last_lap_time_sec', 'fuel_level_liters', 'tire_age_laps', 'pit_status']
                    for col in required_cols:
                        if col not in df_competitors.columns:
                            df_competitors[col] = 'N/A'
                    
                    df_competitors = df_competitors.sort_values(by='current_position')
                    
                    st.dataframe(
                        df_competitors[required_cols].style.format({
                            'gap_to_leader_sec': '{:.3f}s',
                            'last_lap_time_sec': '{:.3f}s',
                            'fuel_level_liters': '{:.1f}L',
                            'tire_age_laps': '{:.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    if len(df_competitors) > 1:
                        fig_gaps = px.bar(df_competitors.head(10), 
                                        x='name', 
                                        y='gap_to_leader_sec',
                                        title='🏁 Gap to Leader Analysis (Top 10)',
                                        color='gap_to_leader_sec',
                                        color_continuous_scale='Viridis')
                        fig_gaps.update_layout(xaxis_tickangle=-45,
                                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                        st.plotly_chart(fig_gaps, use_container_width=True)
                else:
                    st.info("No competitor data available yet.")
                
                # Strategy Recommendations Panel (from simulator's built-in strategy)
                st.markdown("---")
                st.subheader("🎯 Simulator's Strategy Insights")
                sim_strategy = live_data.get('strategy', {})
                if sim_strategy:
                    st.write(f"**Current Strategy:** {sim_strategy.get('current_strategy', 'N/A').title()}")
                    st.write(f"**Fuel Target Laps:** {sim_strategy.get('fuel_target_laps', 'N/A')}")
                    st.write(f"**Tire Change Recommended:** {'Yes' if sim_strategy.get('tire_change_recommended', False) else 'No'}")
                    st.write(f"**Driver Change Due:** {'Yes' if sim_strategy.get('driver_change_due', False) else 'No'}")
                    st.write(f"**Next Pit Recommendation:** {sim_strategy.get('next_pit_recommendation', 'N/A')} laps")
                else:
                    st.info("No simulator strategy insights available.")
                
                # Performance Insights
                st.markdown("---")
                st.subheader("📊 Current Session Performance Summary")
                
                if telemetry_history and len(telemetry_history) > 5:
                    df_summary = pd.json_normalize(telemetry_history)
                    df_summary.columns = [col.replace('car.', '').replace('environmental.', '') 
                                        for col in df_summary.columns]
                    
                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    
                    with summary_col1:
                        best_lap = df_summary['last_lap_time_sec'].min() if 'last_lap_time_sec' in df_summary.columns else 0
                        st.metric("🏆 Best Lap", f"{best_lap:.3f}s" if best_lap > 0 else "N/A")
                    
                    with summary_col2:
                        avg_speed = df_summary['speed_kmh'].mean() if 'speed_kmh' in df_summary.columns else 0
                        st.metric("💨 Avg Speed", f"{avg_speed:.1f} km/h" if not np.isnan(avg_speed) else "N/A")
                    
                    with summary_col3:
                        if 'fuel_consumption_current_L_per_lap' in df_summary.columns:
                            avg_fuel_economy = df_summary['fuel_consumption_current_L_per_lap'].mean()
                            st.metric("⛽ Avg Fuel Econ", f"{avg_fuel_economy:.2f} L/lap" if not np.isnan(avg_fuel_economy) else "N/A")
                        else:
                            st.metric("⛽ Avg Fuel Econ", "N/A")
                    
                    with summary_col4:
                        tire_cols_exist = [f'tire_temp_{t}_C' for t in ['FL', 'FR', 'RL', 'RR']]
                        available_tire_cols = [col for col in tire_cols_exist if col in df_summary.columns]
                        if available_tire_cols:
                            avg_tire_temp_hist = df_summary[available_tire_cols].mean().mean()
                            st.metric("🌡️ Avg Tire Temp", f"{avg_tire_temp_hist:.1f}°C" if not np.isnan(avg_tire_temp_hist) else "N/A")
                        else:
                            st.metric("🌡️ Avg Tire Temp", "N/A")

                else:
                    st.info("Summary statistics will be displayed when more history is available.")


            status_placeholder.success(f"UI refreshed: {time.strftime('%H:%M:%S')}")
            if show_debug:
                with st.sidebar.expander("AI API Debug Info", expanded=False):
                    st.json(st.session_state.ai_response_cache)
        else:
            status_placeholder.error("Waiting for telemetry data...")
            st.info("Ensure your simulator (data_simulator.py) and telemetry API (backend_api.py) are running.")

    # Pause before next refresh and force rerun for continuous updates
    time.sleep(refresh_interval)
    st.rerun() 