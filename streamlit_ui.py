import streamlit as st
import requests
import json
import time 
import pandas as pd 
import plotly.express as px 

# --- Configuration ---
BACKEND_API_BASE_URL = "http://localhost:8000" # Your FastAPI backend for telemetry
AI_API_BASE_URL = "http://localhost:8001"    # Your new FastAPI backend for AI queries

# --- Streamlit Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="United Autosports RaceBrain AI",
    initial_sidebar_state="expanded"
)

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_response_cache" not in st.session_state:
    st.session_state.ai_response_cache = {"strategy_recommendation": "AI could not generate a recommendation.", "confidence_score": 0.0, "priority_actions": [], "anomaly_report": {}} # Initialize with fallback
if "ai_query_input_value" not in st.session_state: 
    st.session_state.ai_query_input_value = ""
if "ai_query_submitted" not in st.session_state: 
    st.session_state.ai_query_submitted = False

# --- Custom Functions to Interact with Backends ---
@st.cache_data(ttl=1) # Cache data for 1 second to reduce API calls on reruns
def get_live_data():
    try:
        response = requests.get(f"{BACKEND_API_BASE_URL}/live_data")
        response.raise_for_status()
        return response.json()
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
            json={"user_input": user_query} 
        )
        response.raise_for_status()
        ai_response = response.json() 
        # --- DEBUG PRINT ---
        st.sidebar.markdown("---")
        st.sidebar.write("DEBUG: AI API Raw Response Received:")
        st.sidebar.json(ai_response) # Display raw JSON for debugging in Streamlit sidebar
        st.sidebar.markdown("---")
        # --- END DEBUG PRINT ---
        return ai_response 
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to query RaceBrain AI: {e}")
        return {"strategy_recommendation": f"Error: AI service unavailable or returned an error: {e}", "confidence_score": 0.0, "priority_actions": [], "anomaly_report": {}}

# --- Callback for text input submission ---
def handle_query_submit():
    st.session_state.ai_query_submitted = True

# --- Streamlit UI Layout ---

st.sidebar.title("🏁 Race Control Panel")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/United_Autosports.svg/1200px-United_Autosports.svg.png", width=200) 
st.sidebar.markdown("---")
refresh_interval = st.sidebar.slider("UI Refresh Interval (seconds)", 1, 10, 2)
st.sidebar.markdown(f"Next refresh in: {refresh_interval} seconds")

# --- Main Content Area with Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Dashboard", "📈 Telemetry Trends", "💬 AI Strategist", "🚦 Race Overview"])

# --- Real-time Data Update Loop ---
main_placeholder = st.empty() 

# This outer loop ensures the Streamlit UI continuously updates
while True:
    # Fetch data (will be cached for short periods)
    live_data = get_live_data()
    telemetry_history = get_telemetry_history_ui()

    with main_placeholder.container(): # All content within this container will refresh
        if live_data:
            # === Tab 1: Live Dashboard ===
            with tab1:
                st.header("📊 Current Race Status")
                current_car = live_data.get('car', {})
                race_info = live_data.get('race_info', {})
                env_info = live_data.get('environmental', {})

                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Lap Number", current_car.get('lap_number', 'N/A'))
                    st.metric("Current Driver", current_car.get('current_driver', 'N/A'))
                    st.metric("Fuel Level", f"{current_car.get('fuel_level_liters', 'N/A')} L")
                with col_info2:
                    st.metric("Race Time", f"{race_info.get('time_of_day', 'N/A')} ({race_info.get('current_hour', 'N/A')}h)")
                    st.metric("Speed", f"{current_car.get('speed_kmh', 'N/A')} km/h")
                    st.metric("Last Lap Time", f"{current_car.get('last_lap_time_sec', 'N/A')} s")
                with col_info3:
                    st.metric("Ambient Temp", f"{env_info.get('ambient_temp_C', 'N/A')} °C")
                    st.metric("Track Temp", f"{env_info.get('track_temp_C', 'N/A')} °C")
                    st.metric("Weather", env_info.get('current_weather', 'N/A').replace("_", " ").title())
                
                st.markdown("---")
                st.subheader("Tire Status")
                tire_cols = st.columns(4)
                tires = ["FL", "FR", "RL", "RR"]
                for i, tire in enumerate(tires):
                    with tire_cols[i]:
                        st.markdown(f"**{tire}**")
                        st.write(f"Temp: {current_car.get(f'tire_temp_{tire}_C', 'N/A')}°C")
                        st.write(f"Pressure: {current_car.get(f'tire_pressure_{tire}_bar', 'N/A')} bar")
                        st.write(f"Wear: {current_car.get(f'tire_wear_{tire}_percent', 'N/A')}%")
                
                st.markdown("---")
                st.subheader("Active Anomalies from Simulator")
                active_sim_anomalies = live_data.get('active_anomalies', [])
                if active_sim_anomalies:
                    for anomaly in active_sim_anomalies:
                        st.error(f"🚨 {anomaly.get('severity', 'UNKNOWN').upper()} ANOMALY: {anomaly.get('message', 'No message')}")
                else:
                    st.info("🟢 No active anomalies reported by simulator.")

            # === Tab 2: Detailed Telemetry Trends ===
            with tab2:
                st.header("📈 Telemetry Trends Over Time")
                if telemetry_history:
                    # Normalize the 'car' data for easier plotting
                    df_history_normalized = pd.json_normalize(telemetry_history)
                    
                    # Flatten keys like 'car.fuel_level_liters'
                    df_history_normalized.columns = [col.replace('car.', '').replace('environmental.', '') for col in df_history_normalized.columns]

                    st.subheader("Fuel Level (Liters)")
                    fig_fuel = px.line(df_history_normalized, x='timestamp_simulated_sec', y='fuel_level_liters', title="Fuel Level Over Time")
                    st.plotly_chart(fig_fuel, use_container_width=True)

                    st.subheader("Tire Temperatures (°C)")
                    fig_tires = px.line(df_history_normalized, x='timestamp_simulated_sec', 
                                         y=[f'tire_temp_{t}_C' for t in tires], 
                                         title="Tire Temperatures Over Time")
                    st.plotly_chart(fig_tires, use_container_width=True)
                    
                    st.subheader("Lap Time (Seconds)")
                    fig_lap_time = px.line(df_history_normalized, x='timestamp_simulated_sec', y='last_lap_time_sec', title="Last Lap Time Over Time")
                    st.plotly_chart(fig_lap_time, use_container_width=True)
                else:
                    st.info("No historical telemetry data available yet for trends.")

            # === Tab 3: AI Strategist ===
            with tab3:
                st.header("💬 Race Strategist AI")
                
                # Display current AI response prominently
                if st.session_state.ai_response_cache:
                    ai_res = st.session_state.ai_response_cache
                    st.markdown("---")
                    st.subheader("Latest AI Strategic Recommendation:")
                    # --- CRITICAL: Display the full string as-is with st.write or st.markdown ---
                    # Using st.text_area with value= and disabled=True is good for long, pre-formatted text
                    st.text_area(
                        "Full Recommendation:", 
                        value=ai_res['strategy_recommendation'], 
                        height=500, # Increased height to accommodate full output
                        disabled=True,
                        key="full_recommendation_display" # Unique key
                    )
                    # --- END CRITICAL FIX ---

                    st.write(f"Confidence: **{ai_res['confidence_score']:.2f}/1.0**")
                    if ai_res['priority_actions']:
                        st.write("Priority Actions: " + ", ".join(ai_res['priority_actions']))
                    
                    # Display detailed anomaly report if it exists
                    if ai_res['anomaly_report'] and ai_res['anomaly_report'].get('priority_level') != 'NONE':
                        with st.expander(f"Detailed Anomaly Report ({ai_res['anomaly_report'].get('priority_level', 'UNKNOWN')} Priority)"):
                            st.json(ai_res['anomaly_report'])
                    st.markdown("---")

                # Input for new query
                user_question = st.text_input(
                    "Engineer's Query:", 
                    value=st.session_state.ai_query_input_value, 
                    key="ai_query_text_input", 
                    on_change=handle_query_submit 
                )
                
                # Button to trigger AI
                if st.button("Ask RaceBrain AI") or st.session_state.ai_query_submitted:
                    if user_question:
                        st.session_state.ai_query_input_value = "" # Clear input before rerun
                        st.session_state.ai_query_submitted = False 

                        with st.spinner("RaceBrain AI is thinking..."):
                            ai_output = query_race_brain_ai(user_question)
                            if ai_output:
                                st.session_state.ai_response_cache = ai_output 
                                st.session_state.chat_history.append({"role": "user", "content": user_question})
                                # Store the full string in chat history as well
                                st.session_state.chat_history.append({"role": "ai", "content": ai_output['strategy_recommendation']})
                                st.rerun() # Force a rerun to update UI with new state
                            else:
                                st.error("Failed to get response from AI.")
                    else:
                        st.warning("Please enter a query for the AI.")
                        st.session_state.ai_query_submitted = False 
                
                st.subheader("Chat History")
                for message in reversed(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"**Engineer:** {message['content']}")
                    else:
                        st.markdown(f"**RaceBrain AI:** {message['content']}")

            # === Tab 4: Race Overview ===
            with tab4:
                st.header("🚦 Race Overview & Competitors")
                competitors = live_data.get('competitors', [])
                st.subheader("Competitor Positions")
                df_competitors = pd.DataFrame(competitors)
                if not df_competitors.empty:
                    # Ensure columns exist before sorting/selecting
                    required_cols = ['current_position', 'name', 'gap_to_leader_sec', 'last_lap_time_sec', 'fuel_level_liters', 'tire_age_laps', 'pit_status']
                    for col in required_cols:
                        if col not in df_competitors.columns:
                            df_competitors[col] = 'N/A' # Add missing columns with default value

                    df_competitors = df_competitors[required_cols].sort_values(by='current_position')
                    st.dataframe(df_competitors, use_container_width=True)
                else:
                    st.info("No competitor data available yet.")
                
                st.subheader("Current Weather & Track")
                env_data = live_data.get('environmental', {})
                track_data = live_data.get('track_info', {})
                st.write(f"**Weather:** {env_data.get('current_weather', 'N/A').replace('_', ' ').title()}")
                st.write(f"**Ambient Temp:** {env_data.get('ambient_temp_C', 'N/A')}°C")
                st.write(f"**Track Temp:** {env_data.get('track_temp_C', 'N/A')}°C")
                st.write(f"**Track Grip:** {env_data.get('track_grip_level', 'N/A')}")
                st.write(f"**Visibility:** {env_data.get('visibility_level', 'N/A')}")
                st.write(f"**Current Sector:** {current_car.get('current_track_segment', 'N/A').replace('_', ' ').title()}")
                st.write(f"**Safety Car Active:** {'Yes' if live_data.get('race_control', {}).get('safety_car_active', False) else 'No'}")


            status_message_placeholder = st.sidebar.empty()
            status_message_placeholder.success(f"UI refreshed: {time.strftime('%H:%M:%S')}")
        else:
            status_message_placeholder = st.sidebar.empty()
            status_message_placeholder.error("Waiting for telemetry data...")
            st.info("Ensure your simulator (data_simulator.py) and telemetry API (backend_api.py) are running.")

    # Pause before next refresh and force rerun for continuous updates
    time.sleep(refresh_interval)
    st.rerun() 