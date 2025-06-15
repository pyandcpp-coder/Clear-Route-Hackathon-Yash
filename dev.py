import streamlit as st
import os
os.environ["STREAMLIT_DEBUG"] = "true"
import streamlit as st
import json
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random
from scipy import stats
import re
import asyncio # New: For running async LangGraph
import operator # For LangGraph state
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq

# --- Constants from data_simulator.py ---
TRACK_LENGTH_KM = 13.626
BASE_LAP_TIME_SEC = 210.0
FASTEST_LAP_TIME_SEC = 195.0
SLOWEST_LAP_TIME_SEC = 240.0

TRACK_SECTORS = {
    "sector_1": { "length_km": 4.2, "characteristics": ["long_straight", "chicane", "medium_corners"], "avg_speed_kmh": 280, "max_speed_kmh": 340, "elevation_change_m": 15, "tire_stress": "medium", "fuel_consumption_modifier": 1.2 },
    "sector_2": { "length_km": 6.0, "characteristics": ["mulsanne_straight", "indianapolis", "arnage"], "avg_speed_kmh": 320, "max_speed_kmh": 370, "elevation_change_m": -10, "tire_stress": "low", "fuel_consumption_modifier": 1.4 },
    "sector_3": { "length_km": 3.426, "characteristics": ["porsche_curves", "ford_chicanes", "start_finish"], "avg_speed_kmh": 180, "max_speed_kmh": 250, "elevation_change_m": 5, "tire_stress": "high", "fuel_consumption_modifier": 0.9 }
}

HYPERCAR_COMPETITORS = {
    "2": { "name": "Cadillac V-Series.R #2", "manufacturer": "Cadillac", "drivers": ["Earl Bamber", "Alex Lynn", "Richard Westbrook"], "base_lap_time": 208.5, "current_gap_sec": -12.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.9, "tire_degradation_rate": 0.6, "reliability_factor": 0.92, "pit_frequency_laps": 38, "strengths": ["straight_line_speed", "fuel_efficiency"], "weaknesses": ["tire_wear", "slow_corners"] },
    "3": { "name": "Cadillac V-Series.R #3", "manufacturer": "Cadillac", "drivers": ["Sebastien Bourdais", "Renger van der Zande", "Scott Dixon"], "base_lap_time": 209.2, "current_gap_sec": 8.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.8, "tire_degradation_rate": 0.5, "reliability_factor": 0.94, "pit_frequency_laps": 40, "strengths": ["consistency", "driver_lineup"], "weaknesses": ["qualifying_pace"] },
    "5": { "name": "Porsche 963 #5", "manufacturer": "Porsche", "drivers": ["Michael Christensen", "Kevin Estre", "Laurens Vanthoor"], "base_lap_time": 207.8, "current_gap_sec": -8.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.7, "tire_degradation_rate": 0.4, "reliability_factor": 0.96, "pit_frequency_laps": 42, "strengths": ["handling", "reliability", "tire_management"], "weaknesses": ["straight_line_speed"] },
    "6": { "name": "Porsche 963 #6", "manufacturer": "Porsche", "drivers": ["Andre Lotterer", "Kevin Estre", "Laurens Vanthoor"], "base_lap_time": 208.1, "current_gap_sec": 15.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.75, "tire_degradation_rate": 0.45, "reliability_factor": 0.95, "pit_frequency_laps": 41, "strengths": ["consistency", "night_pace"], "weaknesses": ["traffic_management"] },
    "7": { "name": "Toyota GR010 Hybrid #7", "manufacturer": "Toyota", "drivers": ["Mike Conway", "Kamui Kobayashi", "Jose Maria Lopez"], "base_lap_time": 206.9, "current_gap_sec": -18.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.6, "tire_degradation_rate": 0.5, "reliability_factor": 0.93, "pit_frequency_laps": 44, "strengths": ["hybrid_system", "fuel_efficiency", "pace"], "weaknesses": ["reliability_history"] },
    "8": { "name": "Toyota GR010 Hybrid #8", "manufacturer": "Toyota", "drivers": ["Sebastien Buemi", "Ryo Hirakawa", "Brendon Hartley"], "base_lap_time": 207.2, "current_gap_sec": -5.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.65, "tire_degradation_rate": 0.48, "reliability_factor": 0.91, "pit_frequency_laps": 43, "strengths": ["hybrid_recovery", "driver_experience"], "weaknesses": ["electrical_issues"] },
    "11": { "name": "Isotta Fraschini Tipo6 #11", "manufacturer": "Isotta Fraschini", "drivers": ["Jean-Karl Vernay", "Antonio Giovinazzi", "Robin Frijns"], "base_lap_time": 213.5, "current_gap_sec": 45.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 3.1, "tire_degradation_rate": 0.7, "reliability_factor": 0.85, "pit_frequency_laps": 35, "strengths": ["innovation"], "weaknesses": ["development", "pace", "reliability"] },
    "15": { "name": "BMW M Hybrid V8 #15", "manufacturer": "BMW", "drivers": ["Dries Vanthoor", "Raffaele Marciello", "Marco Wittmann"], "base_lap_time": 209.8, "current_gap_sec": 22.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.85, "tire_degradation_rate": 0.55, "reliability_factor": 0.89, "pit_frequency_laps": 39, "strengths": ["engine_power"], "weaknesses": ["aerodynamics", "fuel_consumption"] },
    "50": { "name": "Ferrari 499P #50", "manufacturer": "Ferrari", "drivers": ["Antonio Fuoco", "Miguel Molina", "Nicklas Nielsen"], "base_lap_time": 207.5, "current_gap_sec": -3.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.8, "tire_degradation_rate": 0.52, "reliability_factor": 0.90, "pit_frequency_laps": 40, "strengths": ["aerodynamics", "cornering"], "weaknesses": ["straight_line_speed", "reliability"] },
    "51": { "name": "Ferrari 499P #51", "manufacturer": "Ferrari", "drivers": ["James Calado", "Alessandro Pier Guidi", "Antonio Giovinazzi"], "base_lap_time": 207.8, "current_gap_sec": 12.0, "fuel_tank_liters": 68.0, "fuel_consumption_base": 2.82, "tire_degradation_rate": 0.53, "reliability_factor": 0.88, "pit_frequency_laps": 40, "strengths": ["driver_lineup", "race_pace"], "weaknesses": ["qualifying", "tire_warming"] }
}

OUR_CAR_CONFIG = {
    "name": "United Autosports Oreca 07 #22",
    "class": "LMP2",
    "drivers": ["Phil Hanson", "Filipe Albuquerque", "Will Owen"],
    "base_lap_time": 225.0,
    "fuel_tank_liters": 75.0,
    "fuel_consumption_base": 3.2,
    "tire_degradation_rate": 0.8,
    "reliability_factor": 0.94,
    "pit_frequency_laps": 35,
    "target_position": 1,
    "class_competitors": 25
}

WEATHER_PATTERNS = {
    "clear": {"probability": 0.4, "grip_factor": 1.0, "visibility": 1.0},
    "partly_cloudy": {"probability": 0.3, "grip_factor": 0.98, "visibility": 0.95},
    "overcast": {"probability": 0.15, "grip_factor": 0.95, "visibility": 0.90},
    "light_rain": {"probability": 0.08, "grip_factor": 0.75, "visibility": 0.70},
    "heavy_rain": {"probability": 0.05, "grip_factor": 0.55, "visibility": 0.50},
    "fog": {"probability": 0.02, "grip_factor": 0.85, "visibility": 0.30}
}

HOURLY_WEATHER_MODIFIERS = {
    range(6, 10): {"rain_chance": 0.15, "fog_chance": 0.3},
    range(10, 16): {"rain_chance": 0.05, "fog_chance": 0.0},
    range(16, 20): {"rain_chance": 0.12, "fog_chance": 0.0},
    range(20, 24): {"rain_chance": 0.08, "fog_chance": 0.05},
    range(0, 6): {"rain_chance": 0.1, "fog_chance": 0.2}
}

ENHANCED_ANOMALIES = {
    "1": { "type": "tire_puncture_front_left", "message": "Front left tire showing rapid pressure loss - debris suspected", "duration_sec": 180, "severity": "critical", "lap_time_impact": 1.15, "repair_time": 45 },
    "2": { "type": "hybrid_system_failure", "message": "Hybrid system malfunction - reduced power output", "duration_sec": 600, "severity": "high", "lap_time_impact": 1.08, "repair_time": 120 },
    "3": { "type": "safety_car_period", "message": "Safety car deployed - incident at Indianapolis corner", "duration_sec": 420, "severity": "medium", "lap_time_impact": 1.35, "repair_time": 0 },
    "4": { "type": "sudden_weather_change", "message": "Sudden rain shower approaching - grip levels dropping", "duration_sec": 900, "severity": "high", "lap_time_impact": 1.25, "repair_time": 0 },
    "5": { "type": "engine_overheating", "message": "Engine coolant temperature rising - airflow restriction suspected", "duration_sec": 300, "severity": "critical", "lap_time_impact": 1.10, "repair_time": 180 },
    "6": { "type": "brake_balance_issue", "message": "Brake balance shifting to rear - brake disc temperature imbalance", "duration_sec": 240, "severity": "medium", "lap_time_impact": 1.05, "repair_time": 60 },
    "7": { "type": "aerodynamic_damage", "message": "Front splitter damage detected - downforce reduced", "duration_sec": 480, "severity": "high", "lap_time_impact": 1.07, "repair_time": 90 },
    "8": { "type": "fuel_flow_restriction", "message": "Fuel flow rate below optimal - filter blockage suspected", "duration_sec": 360, "severity": "medium", "lap_time_impact": 1.04, "repair_time": 75 },
    "9": { "type": "gearbox_sensor_fault", "message": "Gearbox sensor intermittent - shift quality affected", "duration_sec": 720, "severity": "medium", "lap_time_impact": 1.03, "repair_time": 45 },
    "10": { "type": "night_visibility_issue", "message": "Headlight alignment issue - reduced night visibility", "duration_sec": 1800, "severity": "high", "lap_time_impact": 1.12, "repair_time": 120 },
    "0": { "type": "reset_all_systems", "message": "All systems reset - optimal conditions restored", "duration_sec": 1, "severity": "info", "lap_time_impact": 1.0, "repair_time": 0 }
}

# Initial values for simulator internal state (used for resets)
INITIAL_TIRE_TEMPS = {"FL": 95.0, "FR": 95.0, "RL": 90.0, "RR": 90.0}
INITIAL_TIRE_PRESSURES = {"FL": 1.9, "FR": 1.9, "RL": 1.8, "RR": 1.8}
INITIAL_OIL_TEMP = 110.0
INITIAL_WATER_TEMP = 85.0
INITIAL_HYBRID_BATTERY = 90.0
INITIAL_HYBRID_POWER = 140.0
INITIAL_VISIBILITY = 1.0
WEATHER_CHECK_INTERVAL_SEC = 300


# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="United Autosports RaceBrain AI Pro",
    initial_sidebar_state="expanded",
    page_icon="🏁"
)

# Use Streamlit Secrets for GROQ_API_KEY
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
llm_temperature = 0.7

# --- Simulator State Initialization (new in Streamlit for persistence) ---
# This dictionary will hold the entire mutable state of the simulator.
# It gets stored in st.session_state and updated across Streamlit reruns.
# Start at 18h into the race as per your previous `data_simulator.py`'s default start
_initial_simulator_state = {
    "current_lap": 1,
    "current_lap_time_sec": 0.0,
    "total_simulated_time_sec": 18 * 3600,
    "our_car_fuel": OUR_CAR_CONFIG["fuel_tank_liters"],
    "lap_start_time_simulated": 18 * 3600,
    "last_lap_start_time_simulated": 0.0,
    "our_car_last_lap_time": OUR_CAR_CONFIG["base_lap_time"],
    "current_sector": 1,
    "sector_progress": 0.0,
    "tire_compound": "medium",
    "tire_age_laps": 0,
    "tire_wear": {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0},
    "tire_temperatures": INITIAL_TIRE_TEMPS.copy(),
    "tire_pressures": INITIAL_TIRE_PRESSURES.copy(),
    "current_driver_idx": 0, # Index into OUR_CAR_CONFIG["drivers"]
    "driver_stint_time": 0.0,
    "last_pit_lap": 0,
    "pit_strategy": "normal",
    "fuel_saving_mode": False,
    "push_mode": False,
    "current_weather": "clear",
    "track_temperature": 28.0,
    "ambient_temperature": 20.0,
    "wind_speed": 12.0,
    "wind_direction": 180,
    "track_grip": 1.0,
    "visibility": 1.0,
    "last_weather_check_time": 0.0,
    "last_speed_kmh": 0.0,
    "last_throttle_percent": 0.0,
    "last_brake_percent": 0.0,
    "last_engine_rpm": 0.0,
    "oil_temp_C": INITIAL_OIL_TEMP,
    "water_temp_C": INITIAL_WATER_TEMP,
    "hybrid_battery_percent": INITIAL_HYBRID_BATTERY,
    "hybrid_power_kw": INITIAL_HYBRID_POWER,
    "safety_car_active": False,
    "yellow_flag_sectors": [],
    "track_limits_warnings": 0,
    "race_incidents": [],
    "competitor_positions": {}, # Will be populated dynamically
    "active_anomalies": {},
    "command_queue_sim": [], # Store commands for simulator input
    "last_simulated_update_time": time.time() # To control simulation speed in Streamlit
}

# Initialize competitor_positions for the initial state
for car_id in HYPERCAR_COMPETITORS:
    _initial_simulator_state["competitor_positions"][car_id] = {
        "current_lap": 1,
        "gap_to_leader": HYPERCAR_COMPETITORS[car_id]["current_gap_sec"],
        "last_pit_lap": 0,
        "tire_age": 0,
        "fuel_level": HYPERCAR_COMPETITORS[car_id]["fuel_tank_liters"],
        "current_issues": [],
        "pit_strategy": "normal",
        "current_driver_index": 0
    }

# Initialize Streamlit session state
if "sim_state" not in st.session_state:
    st.session_state.sim_state = _initial_simulator_state.copy()
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
    st.session_state.selected_driver = OUR_CAR_CONFIG["drivers"][_initial_simulator_state["current_driver_idx"]]
if "driver_stint_start" not in st.session_state:
    st.session_state.driver_stint_start = time.time()
if "driver_stint_laps_start" not in st.session_state:
    st.session_state.driver_stint_laps_start = _initial_simulator_state["current_lap"]
if 'chart_counter' not in st.session_state:
    st.session_state.chart_counter = 0
if "sim_speed_factor" not in st.session_state:
    st.session_state.sim_speed_factor = 30 # Default for slider

# --- Helper Functions for Simulator Logic (adapted from data_simulator.py) ---

def calculate_sector_time_sim(sim_state_dict: dict, sector_num: int) -> float:
    sector_key = f"sector_{sector_num}"
    sector_data = TRACK_SECTORS[sector_key]
    
    base_sector_time = sim_state_dict["our_car_last_lap_time"] * (sector_data["length_km"] / TRACK_LENGTH_KM)
    
    weather_modifier = sim_state_dict["track_grip"]
    if weather_modifier < 0.8:
        if "straight" in sector_data["characteristics"]:
            weather_modifier *= 0.95
        else:
            weather_modifier *= 0.85
    
    tire_temp_avg = sum(sim_state_dict["tire_temperatures"].values()) / 4
    tire_wear_avg = sum(sim_state_dict["tire_wear"].values()) / 4
    tire_performance = 1.0
    if tire_temp_avg < 85:
        tire_performance *= 1.05
    elif tire_temp_avg > 110:
        tire_performance *= 1.08
    tire_performance *= (1 + tire_wear_avg * 0.1)
    
    fuel_impact = 1 + (sim_state_dict["our_car_fuel"] - 20) * 0.0008
    driver_fatigue = 1 + min(sim_state_dict["driver_stint_time"] / 7200, 0.02)
    
    final_time = base_sector_time * weather_modifier * tire_performance * fuel_impact * driver_fatigue
    variation = random.uniform(-0.02, 0.02)
    final_time *= (1 + variation)
    return final_time

def update_weather_system_sim(sim_state_dict: dict):
    current_hour = int((sim_state_dict["total_simulated_time_sec"] % 86400) / 3600)
    hour_angle = (current_hour - 14) * math.pi / 12
    temp_variation = 8 * math.cos(hour_angle)
    sim_state_dict["ambient_temperature"] = 20 + temp_variation + random.uniform(-0.5, 0.5)
    sim_state_dict["track_temperature"] = sim_state_dict["ambient_temperature"] + 8 + random.uniform(-1, 1)

    sim_state_dict["wind_speed"] = random.uniform(5, 25)
    sim_state_dict["wind_direction"] = random.randint(0, 359)

    if (sim_state_dict["total_simulated_time_sec"] - sim_state_dict["last_weather_check_time"]) < WEATHER_CHECK_INTERVAL_SEC:
        return

    sim_state_dict["last_weather_check_time"] = sim_state_dict["total_simulated_time_sec"]
    weather_change_chance = 0.25

    for hour_range, modifiers in HOURLY_WEATHER_MODIFIERS.items():
        if current_hour in hour_range:
            if "rain" in sim_state_dict["current_weather"]:
                weather_change_chance *= 0.3
            if modifiers.get("rain_chance", 0) > 0 and "rain" not in sim_state_dict["current_weather"]:
                weather_change_chance *= (1 + modifiers["rain_chance"])
            if modifiers.get("fog_chance", 0) > 0 and "fog" not in sim_state_dict["current_weather"]:
                weather_change_chance *= (1 + modifiers["fog_chance"])
            break

    if random.random() < weather_change_chance:
        weather_roll = random.random()
        cumulative_prob = 0
        temp_weather_patterns = WEATHER_PATTERNS.copy()
        if sim_state_dict["current_weather"] != "clear":
            temp_weather_patterns["clear"]["probability"] *= 0.8
            temp_weather_patterns[sim_state_dict["current_weather"]]["probability"] *= 1.2
        total_adjusted_prob = sum(wp["probability"] for wp in temp_weather_patterns.values())
        for wp_type in temp_weather_patterns:
            temp_weather_patterns[wp_type]["probability"] /= total_adjusted_prob

        for weather_type, data in temp_weather_patterns.items():
            cumulative_prob += data["probability"]
            if weather_roll <= cumulative_prob:
                if weather_type != sim_state_dict["current_weather"]:
                    sim_state_dict["current_weather"] = weather_type
                    sim_state_dict["track_grip"] = data["grip_factor"] + random.uniform(-0.03, 0.03)
                    sim_state_dict["visibility"] = data["visibility"] + random.uniform(-0.02, 0.02)
                    sim_state_dict["track_grip"] = max(0.4, min(1.0, sim_state_dict["track_grip"]))
                    sim_state_dict["visibility"] = max(0.2, min(1.0, sim_state_dict["visibility"]))
                break

def simulate_competitor_behavior_sim(sim_state_dict: dict):
    for car_id, competitor in HYPERCAR_COMPETITORS.items():
        pos_data = sim_state_dict["competitor_positions"][car_id]
        base_time = competitor["base_lap_time"]

        weather_impact = (2.0 - sim_state_dict["track_grip"]) * 0.5
        tire_deg_impact = pos_data["tire_age"] * competitor["tire_degradation_rate"] * 0.1
        fuel_ratio = pos_data["fuel_level"] / competitor["fuel_tank_liters"]
        fuel_impact = (1 - fuel_ratio) * 1.5
        driver_variance = random.uniform(-0.5, 0.5)
        current_sim_hour = int((sim_state_dict["total_simulated_time_sec"] % 86400) / 3600)
        is_night = (current_sim_hour >= 20 or current_sim_hour < 6)
        if "night_pace" in competitor.get("strengths", []) and is_night:
            driver_variance -= 0.5

        final_lap_time = base_time + weather_impact + tire_deg_impact + fuel_impact + driver_variance
        if sim_state_dict["our_car_last_lap_time"] > 0 and final_lap_time > 0:
            lap_time_difference = final_lap_time - sim_state_dict["our_car_last_lap_time"]
            gap_change_per_interval = lap_time_difference * (0.5 / sim_state_dict["our_car_last_lap_time"])
            pos_data["gap_to_leader"] += gap_change_per_interval
            pos_data["gap_to_leader"] = max(-300.0, min(300.0, pos_data["gap_to_leader"]))

        if (sim_state_dict["current_lap"] - pos_data["last_pit_lap"]) >= competitor["pit_frequency_laps"]:
            if random.random() < 0.15:
                pos_data["last_pit_lap"] = sim_state_dict["current_lap"]
                pos_data["tire_age"] = 0
                pos_data["fuel_level"] = competitor["fuel_tank_liters"]
                pos_data["gap_to_leader"] += 25 + random.uniform(-5, 5)
                pos_data["in_pit"] = True
        else:
            pos_data["in_pit"] = False

        pos_data["tire_age"] += 1
        fuel_consumption = competitor["fuel_consumption_base"] * (1 + random.uniform(-0.05, 0.05))
        pos_data["fuel_level"] = max(0, pos_data["fuel_level"] - fuel_consumption)

        if random.random() < (1 - competitor["reliability_factor"]) / 5000:
            issue_types = ["engine_issue", "tire_problem", "aerodynamic_damage", "electrical_fault"]
            issue = random.choice(issue_types)
            pos_data["current_issues"].append(issue)
            pos_data["gap_to_leader"] += random.uniform(20, 50)

def activate_anomaly_sim(sim_state_dict: dict, anomaly_id: str, duration: int, message: str):
    anomaly_config = ENHANCED_ANOMALIES.get(anomaly_id)
    if not anomaly_config:
        return

    if anomaly_id == "0":
        # Reset to initial state
        sim_state_dict.clear() # Clear existing state
        sim_state_dict.update(_initial_simulator_state.copy()) # Update with a fresh copy
        return

    end_time = sim_state_dict["total_simulated_time_sec"] + duration if duration > 0 else 0
    sim_state_dict["active_anomalies"][anomaly_id] = {
        "start_time": sim_state_dict["total_simulated_time_sec"],
        "end_time": end_time,
        "message": message
    }

    # Apply effects
    if anomaly_config["type"] == "tire_puncture_front_left":
        sim_state_dict["tire_wear"]["FL"] = 0.9
        sim_state_dict["tire_pressures"]["FL"] = 0.5
        sim_state_dict["tire_temperatures"]["FL"] = 120.0
    elif anomaly_config["type"] == "hybrid_system_failure":
        sim_state_dict["last_speed_kmh"] *= 0.8
        sim_state_dict["last_engine_rpm"] *= 0.8
    elif anomaly_config["type"] == "safety_car_period":
        sim_state_dict["safety_car_active"] = True
        sim_state_dict["last_speed_kmh"] = 80.0
        sim_state_dict["last_throttle_percent"] = 20.0
        sim_state_dict["last_brake_percent"] = 10.0
        sim_state_dict["last_engine_rpm"] = 4000.0
    elif anomaly_config["type"] == "sudden_weather_change":
        sim_state_dict["current_weather"] = "heavy_rain"
        sim_state_dict["track_grip"] = 0.55
        sim_state_dict["visibility"] = 0.50
        sim_state_dict["last_speed_kmh"] *= 0.7
        sim_state_dict["last_throttle_percent"] *= 0.6
        sim_state_dict["last_brake_percent"] *= 1.5
    elif anomaly_config["type"] == "engine_overheating":
        sim_state_dict["oil_temp_C"] = 130.0
        sim_state_dict["water_temp_C"] = 105.0
        sim_state_dict["last_throttle_percent"] = 40.0
        sim_state_dict["last_engine_rpm"] = 6000.0
        sim_state_dict["last_speed_kmh"] *= 0.85
    elif anomaly_config["type"] == "brake_balance_issue":
        sim_state_dict["tire_temperatures"]["FL"] += 10
        sim_state_dict["tire_temperatures"]["FR"] += 10
        sim_state_dict["tire_temperatures"]["RL"] -= 5
        sim_state_dict["tire_temperatures"]["RR"] -= 5
        sim_state_dict["last_brake_percent"] = 90.0
        sim_state_dict["last_speed_kmh"] *= 0.95
    elif anomaly_config["type"] == "aerodynamic_damage":
        sim_state_dict["last_speed_kmh"] *= 0.90
    elif anomaly_config["type"] == "fuel_flow_restriction":
        sim_state_dict["our_car_fuel"] = max(0, sim_state_dict["our_car_fuel"] - 5.0)
        sim_state_dict["last_engine_rpm"] *= 0.85
        sim_state_dict["last_throttle_percent"] = 60.0
    elif anomaly_config["type"] == "gearbox_sensor_fault":
        sim_state_dict["last_engine_rpm"] = random.uniform(4000, 10000)
        sim_state_dict["last_speed_kmh"] = random.uniform(100, 200)
    elif anomaly_config["type"] == "night_visibility_issue":
        sim_state_dict["visibility"] = 0.30

def update_anomalies_sim(sim_state_dict: dict):
    expired_anomalies = []
    for anomaly_type_id, anomaly_data in sim_state_dict["active_anomalies"].items():
        if anomaly_data.get("end_time", 0) > 0 and sim_state_dict["total_simulated_time_sec"] >= anomaly_data["end_time"]:
            expired_anomalies.append(anomaly_type_id)

    for anomaly_type_id in expired_anomalies:
        anomaly_config = ENHANCED_ANOMALIES.get(anomaly_type_id, {})
        del sim_state_dict["active_anomalies"][anomaly_type_id]

        if anomaly_config["type"] == "tire_puncture_front_left":
            sim_state_dict["tire_wear"]["FL"] = 0.0
            sim_state_dict["tire_pressures"]["FL"] = INITIAL_TIRE_PRESSURES["FL"]
            sim_state_dict["tire_temperatures"]["FL"] = INITIAL_TIRE_TEMPS["FL"]
        elif anomaly_config["type"] == "safety_car_period":
            sim_state_dict["safety_car_active"] = False
        elif anomaly_config["type"] == "sudden_weather_change":
            sim_state_dict["current_weather"] = "clear"
            sim_state_dict["track_grip"] = 1.0
            sim_state_dict["visibility"] = 1.0
        elif anomaly_config["type"] == "engine_overheating":
            sim_state_dict["oil_temp_C"] = INITIAL_OIL_TEMP
            sim_state_dict["water_temp_C"] = INITIAL_WATER_TEMP
        elif anomaly_config["type"] == "brake_balance_issue":
            sim_state_dict["tire_temperatures"]["FL"] = INITIAL_TIRE_TEMPS["FL"]
            sim_state_dict["tire_temperatures"]["FR"] = INITIAL_TIRE_TEMPS["FR"]
            sim_state_dict["tire_temperatures"]["RL"] = INITIAL_TIRE_TEMPS["RL"]
            sim_state_dict["tire_temperatures"]["RR"] = INITIAL_TIRE_TEMPS["RR"]

        sim_state_dict["last_speed_kmh"] = random.uniform(200, 250)
        sim_state_dict["last_throttle_percent"] = random.uniform(60, 80)
        sim_state_dict["last_brake_percent"] = random.uniform(10, 30)
        sim_state_dict["last_engine_rpm"] = random.uniform(7000, 9000)


def generate_enhanced_telemetry_sim(sim_state_dict: dict):
    # Simulate real-time factor, advance simulated time based on actual time elapsed
    time_since_last_update = time.time() - sim_state_dict["last_simulated_update_time"]
    sim_state_dict["total_simulated_time_sec"] += time_since_last_update * st.session_state.get("sim_speed_factor", 30)
    sim_state_dict["last_simulated_update_time"] = time.time()

    sim_state_dict["current_lap_time_sec"] = sim_state_dict["total_simulated_time_sec"] - sim_state_dict["lap_start_time_simulated"]
    sim_state_dict["driver_stint_time"] += (time_since_last_update * st.session_state.get("sim_speed_factor", 30))

    update_weather_system_sim(sim_state_dict)

    expected_sector_time = sim_state_dict["our_car_last_lap_time"] / 3 if sim_state_dict["our_car_last_lap_time"] > 0 else (BASE_LAP_TIME_SEC / 3)
    
    sim_state_dict["sector_progress"] += (time_since_last_update * st.session_state.get("sim_speed_factor", 30))
    
    if sim_state_dict["sector_progress"] >= expected_sector_time:
        sim_state_dict["current_sector"] = (sim_state_dict["current_sector"] % 3) + 1
        sim_state_dict["sector_progress"] = 0.0
        
        if sim_state_dict["current_sector"] == 1:
            sim_state_dict["current_lap"] += 1
            sim_state_dict["lap_start_time_simulated"] = sim_state_dict["total_simulated_time_sec"]
            sim_state_dict["current_lap_time_sec"] = 0.0
            sim_state_dict["tire_age_laps"] += 1
            
            sector_times = []
            for i in range(1, 4):
                sector_time = calculate_sector_time_sim(sim_state_dict, i)
                sector_times.append(sector_time)
            
            sim_state_dict["our_car_last_lap_time"] = sum(sector_times)
            
            for anomaly_id in sim_state_dict["active_anomalies"]:
                anomaly_config = ENHANCED_ANOMALIES.get(anomaly_id)
                if anomaly_config and "lap_time_impact" in anomaly_config:
                    sim_state_dict["our_car_last_lap_time"] *= anomaly_config["lap_time_impact"]
            
            base_consumption = OUR_CAR_CONFIG["fuel_consumption_base"]
            consumption_modifier = 1.0
            if sim_state_dict["fuel_saving_mode"]: consumption_modifier = 0.85
            elif sim_state_dict["push_mode"]: consumption_modifier = 1.15
            consumption_modifier *= (2.0 - sim_state_dict["track_grip"]) * 0.1 + 0.9
            fuel_used = base_consumption * consumption_modifier
            sim_state_dict["our_car_fuel"] = max(0, sim_state_dict["our_car_fuel"] - fuel_used)
            
            for tire in sim_state_dict["tire_wear"]:
                wear_rate = OUR_CAR_CONFIG["tire_degradation_rate"] / 1000
                if sim_state_dict["track_grip"] < 0.8: wear_rate *= 0.7
                elif sim_state_dict["ambient_temperature"] > 25: wear_rate *= 1.3
                sim_state_dict["tire_wear"][tire] += wear_rate * random.uniform(0.9, 1.1)
                sim_state_dict["tire_wear"][tire] = min(sim_state_dict["tire_wear"][tire], 1.0)
    
    laps_since_pit = sim_state_dict["current_lap"] - sim_state_dict["last_pit_lap"]
    should_pit = False
    
    if sim_state_dict["our_car_fuel"] < 15 and laps_since_pit > 20:
        should_pit = True
    elif sim_state_dict["tire_age_laps"] > 45 and any(wear > 0.8 for wear in sim_state_dict["tire_wear"].values()):
        should_pit = True
    elif sim_state_dict["driver_stint_time"] > 7200:
        should_pit = True
    
    if should_pit:
        pit_duration = random.uniform(22, 28)
        sim_state_dict["total_simulated_time_sec"] += pit_duration
        
        sim_state_dict["our_car_fuel"] = OUR_CAR_CONFIG["fuel_tank_liters"]
        sim_state_dict["tire_wear"] = {k: 0.0 for k in sim_state_dict["tire_wear"]}
        sim_state_dict["tire_age_laps"] = 0
        sim_state_dict["last_pit_lap"] = sim_state_dict["current_lap"]
        
        if sim_state_dict["driver_stint_time"] > 7200 or random.random() < 0.3:
            sim_state_dict["current_driver_idx"] = (sim_state_dict["current_driver_idx"] + 1) % len(OUR_CAR_CONFIG["drivers"])
            sim_state_dict["driver_stint_time"] = 0.0

    simulate_competitor_behavior_sim(sim_state_dict)
    update_anomalies_sim(sim_state_dict)

    # Process commands from Streamlit UI
    while sim_state_dict["command_queue_sim"]:
        cmd = sim_state_dict["command_queue_sim"].pop(0)
        if cmd.startswith("anomaly_"):
            anomaly_id = cmd.split("_")[1]
            anomaly_config = ENHANCED_ANOMALIES.get(anomaly_id)
            if anomaly_config:
                activate_anomaly_sim(sim_state_dict, anomaly_id, anomaly_config["duration_sec"], anomaly_config["message"])

    current_sector_name = f"sector_{sim_state_dict['current_sector']}"
    sector_data = TRACK_SECTORS[current_sector_name]
    
    target_speed_kmh = 0.0
    target_throttle_percent = 0.0
    target_brake_percent = 0.0
    target_engine_rpm = 0.0

    if "straight" in sector_data["characteristics"]:
        target_speed_kmh = random.uniform(290, min(340, sector_data["max_speed_kmh"]))
        target_throttle_percent = random.uniform(90, 100)
        target_brake_percent = random.uniform(0, 5)
        target_engine_rpm = random.uniform(9000, 10500)
    elif "chicane" in sector_data["characteristics"]:
        target_speed_kmh = random.uniform(130, 170)
        target_throttle_percent = random.uniform(45, 65)
        target_brake_percent = random.uniform(65, 85)
        target_engine_rpm = random.uniform(6500, 7500)
    elif "corner" in str(sector_data["characteristics"]):
        target_speed_kmh = random.uniform(160, 210)
        target_throttle_percent = random.uniform(55, 75)
        target_brake_percent = random.uniform(25, 55)
        target_engine_rpm = random.uniform(7500, 8500)
    else:
        target_speed_kmh = random.uniform(210, 270)
        target_throttle_percent = random.uniform(65, 85)
        target_brake_percent = random.uniform(15, 35)
        target_engine_rpm = random.uniform(8000, 9000)
    
    smoothing_factor = 0.15
    
    sim_state_dict["last_speed_kmh"] += (target_speed_kmh - sim_state_dict["last_speed_kmh"]) * smoothing_factor + random.uniform(-2, 2)
    sim_state_dict["last_throttle_percent"] += (target_throttle_percent - sim_state_dict["last_throttle_percent"]) * smoothing_factor + random.uniform(-0.5, 0.5)
    sim_state_dict["last_brake_percent"] += (target_brake_percent - sim_state_dict["last_brake_percent"]) * smoothing_factor + random.uniform(-0.5, 0.5)
    sim_state_dict["last_engine_rpm"] += (target_engine_rpm - sim_state_dict["last_engine_rpm"]) * smoothing_factor + random.uniform(-50, 50)

    sim_state_dict["last_speed_kmh"] = max(0, min(380, sim_state_dict["last_speed_kmh"]))
    sim_state_dict["last_throttle_percent"] = max(0, min(100, sim_state_dict["last_throttle_percent"]))
    sim_state_dict["last_brake_percent"] = max(0, min(100, sim_state_dict["last_brake_percent"]))
    sim_state_dict["last_engine_rpm"] = max(1000, min(11000, sim_state_dict["last_engine_rpm"]))

    if sim_state_dict["track_grip"] < 0.8:
        sim_state_dict["last_speed_kmh"] *= 0.85
        sim_state_dict["last_throttle_percent"] *= 0.9
        sim_state_dict["last_brake_percent"] *= 1.2
    
    if sim_state_dict["visibility"] < 0.8:
        sim_state_dict["last_speed_kmh"] *= 0.92
        sim_state_dict["last_throttle_percent"] *= 0.95
    
    for anomaly_id in sim_state_dict["active_anomalies"]:
        anomaly_config = ENHANCED_ANOMALIES.get(anomaly_id)
        if anomaly_config:
            if anomaly_config["type"] == "aerodynamic_damage":
                sim_state_dict["last_speed_kmh"] *= 0.93
            elif anomaly_config["type"] == "night_visibility_issue" and sim_state_dict["visibility"] < 0.9:
                sim_state_dict["last_speed_kmh"] *= 0.88
    
    base_tire_temp_front = 95.0
    base_tire_temp_rear = 90.0

    for tire_pos in ["FL", "FR", "RL", "RR"]:
        if not any(f"tire_puncture_{tire_pos.lower()}" == ENHANCED_ANOMALIES[aid]["type"] for aid in sim_state_dict["active_anomalies"]):
            current_base_temp = base_tire_temp_front if tire_pos.startswith('F') else base_tire_temp_rear
            temp_modifier = 1.0
            if sim_state_dict["last_brake_percent"] > 50: temp_modifier += 0.1
            if sim_state_dict["last_speed_kmh"] > 300: temp_modifier += 0.05
            if "corner" in str(sector_data["characteristics"]): temp_modifier += 0.08
            if sim_state_dict["ambient_temperature"] < 15: temp_modifier *= 0.95
            elif sim_state_dict["ambient_temperature"] > 30: temp_modifier *= 1.05
            temp_modifier += sim_state_dict["tire_wear"][tire_pos] * 0.2
            sim_state_dict["tire_temperatures"][tire_pos] = current_base_temp * temp_modifier + random.uniform(-1, 1)

        if not any(f"tire_puncture_{tire_pos.lower()}" == ENHANCED_ANOMALIES[aid]["type"] for aid in sim_state_dict["active_anomalies"]):
            temp_diff = sim_state_dict["tire_temperatures"][tire_pos] - current_base_temp
            pressure_change = temp_diff * 0.01
            base_pressure = INITIAL_TIRE_PRESSURES[tire_pos]
            sim_state_dict["tire_pressures"][tire_pos] = base_pressure + pressure_change + random.uniform(-0.01, 0.01)
        
    suspension_travel = {}
    base_travel = 25.0
    if "corner" in sector_data["characteristics"]: base_travel = 45.0
    elif "straight" in sector_data["characteristics"]: base_travel = 15.0
    
    for tire_pos in ["FL", "FR", "RL", "RR"]:
        travel = base_travel + random.uniform(-3, 3)
        if any(f"tire_puncture_{tire_pos.lower()}" == ENHANCED_ANOMALIES[aid]["type"] for aid in sim_state_dict["active_anomalies"]):
            travel *= 1.5
        suspension_travel[f"suspension_travel_{tire_pos}_mm"] = travel
    
    if "engine_overheating" not in sim_state_dict["active_anomalies"]:
        sim_state_dict["oil_temp_C"] = INITIAL_OIL_TEMP + (sim_state_dict["last_engine_rpm"] - 7500) * 0.002 + random.uniform(-2, 2)
        sim_state_dict["water_temp_C"] = INITIAL_WATER_TEMP + (sim_state_dict["ambient_temperature"] - 20) * 0.5 + random.uniform(-1, 1)
    
    if "hybrid_system_failure" not in sim_state_dict["active_anomalies"]:
        sim_state_dict["hybrid_battery_percent"] = random.uniform(70, 100)
        sim_state_dict["hybrid_power_kw"] = 0
        if sim_state_dict["last_throttle_percent"] > 70 and sim_state_dict["hybrid_battery_percent"] > 20:
            sim_state_dict["hybrid_power_kw"] = random.uniform(120, 160)
            sim_state_dict["hybrid_battery_percent"] -= 0.5
        elif sim_state_dict["last_throttle_percent"] < 30:
            sim_state_dict["hybrid_battery_percent"] = min(100, sim_state_dict["hybrid_battery_percent"] + 0.3)
    else:
        sim_state_dict["hybrid_battery_percent"] = max(0, sim_state_dict["hybrid_battery_percent"] - 0.1)
        sim_state_dict["hybrid_power_kw"] = 0

    competitor_data = []
    for car_id, competitor in HYPERCAR_COMPETITORS.items():
        pos_data = sim_state_dict["competitor_positions"][car_id]
        gap_seconds = pos_data["gap_to_leader"]
        
        competitor_data.append({
            "car_number": car_id,
            "name": competitor["name"],
            "manufacturer": competitor["manufacturer"],
            "drivers": competitor["drivers"],
            "current_driver": competitor["drivers"][pos_data.get("current_driver_index", 0)],
            "gap_to_leader_sec": round(gap_seconds, 2),
            "gap_status": "ahead" if gap_seconds < 0 else ("behind" if gap_seconds > 0 else "same_lap"),
            "current_lap": pos_data["current_lap"],
            "last_lap_time_sec": round(competitor["base_lap_time"] + random.uniform(-1, 1), 2),
            "tire_age_laps": pos_data["tire_age"],
            "fuel_level_liters": round(pos_data["fuel_level"], 1),
            "pit_status": "in_pit" if pos_data.get("in_pit", False) else "on_track",
            "last_pit_lap": pos_data["last_pit_lap"],
            "current_issues": pos_data.get("current_issues", []),
            "pit_strategy": pos_data.get("pit_strategy", "normal"),
            "sector_time_1": round(competitor["base_lap_time"] * 0.35 + random.uniform(-0.5, 0.5), 2),
            "sector_time_2": round(competitor["base_lap_time"] * 0.30 + random.uniform(-0.5, 0.5), 2),
            "sector_time_3": round(competitor["base_lap_time"] * 0.35 + random.uniform(-0.5, 0.5), 2)
        })
    
    competitor_data.sort(key=lambda x: x["gap_to_leader_sec"])
    for i, comp in enumerate(competitor_data):
        comp["current_position"] = i + 1
    
    current_hour = int((sim_state_dict["total_simulated_time_sec"] % 86400) / 3600)
    race_time_remaining = max(0, 24 * 3600 - sim_state_dict["total_simulated_time_sec"])
    
    average_lap_time = sim_state_dict["our_car_last_lap_time"]
    fuel_laps_remaining = max(0, sim_state_dict["our_car_fuel"] / (OUR_CAR_CONFIG["fuel_consumption_base"] * 1.1)) if (OUR_CAR_CONFIG["fuel_consumption_base"] * 1.1) > 0 else 0
    estimated_pit_window = max(0, OUR_CAR_CONFIG["pit_frequency_laps"] - (sim_state_dict["current_lap"] - sim_state_dict["last_pit_lap"]))
    
    track_limits_risk = 0.0
    if sim_state_dict["last_speed_kmh"] > sector_data["max_speed_kmh"] * 0.95 and random.random() < 0.05:
        sim_state_dict["track_limits_warnings"] += 1
        track_limits_risk = random.uniform(0.1, 0.3)
    
    is_night = (current_hour >= 21 or current_hour < 6)

    telemetry_data = {
        "timestamp_simulated_sec": round(sim_state_dict["total_simulated_time_sec"], 2),
        "race_info": {
            "current_hour": current_hour,
            "race_time_elapsed_sec": round(sim_state_dict["total_simulated_time_sec"], 2),
            "race_time_remaining_sec": round(race_time_remaining, 2),
            "time_of_day": f"{current_hour:02d}:{int((sim_state_dict['total_simulated_time_sec'] % 3600) / 60):02d}",
            "is_night": is_night
        },
        "car": {
            "name": OUR_CAR_CONFIG["name"],
            "class": OUR_CAR_CONFIG["class"],
            "current_driver": OUR_CAR_CONFIG["drivers"][sim_state_dict["current_driver_idx"]],
            "driver_stint_time_sec": round(sim_state_dict["driver_stint_time"], 2),
            "lap_number": sim_state_dict["current_lap"],
            "current_lap_time_sec": round(sim_state_dict["current_lap_time_sec"], 2),
            "last_lap_time_sec": round(sim_state_dict["our_car_last_lap_time"], 2),
            "average_lap_time_sec": round(average_lap_time, 2),
            "current_sector": sim_state_dict["current_sector"],
            "sector_progress_percent": round((sim_state_dict["sector_progress"] / expected_sector_time) * 100, 1) if expected_sector_time > 0 else 0,
            "current_track_segment": current_sector_name,
            "speed_kmh": round(sim_state_dict["last_speed_kmh"], 2),
            "engine_rpm": round(sim_state_dict["last_engine_rpm"], 2),
            "throttle_percent": round(sim_state_dict["last_throttle_percent"], 2),
            "brake_percent": round(sim_state_dict["last_brake_percent"], 2),
            "gear": max(1, min(8, int(sim_state_dict["last_speed_kmh"] / 45))),
            "drs_active": sim_state_dict["last_speed_kmh"] > 250 and "straight" in sector_data["characteristics"],
            "fuel_level_liters": round(sim_state_dict["our_car_fuel"], 2),
            "fuel_consumption_current_L_per_lap": round(OUR_CAR_CONFIG["fuel_consumption_base"] * (1 + random.uniform(-0.02, 0.02)), 2),
            "fuel_laps_remaining": round(fuel_laps_remaining, 1),
            "fuel_saving_mode": sim_state_dict["fuel_saving_mode"],
            "push_mode": sim_state_dict["push_mode"],
            "oil_temp_C": round(sim_state_dict["oil_temp_C"], 2),
            "water_temp_C": round(sim_state_dict["water_temp_C"], 2),
            "hybrid_battery_percent": round(sim_state_dict["hybrid_battery_percent"], 1),
            "hybrid_power_output_kw": round(sim_state_dict["hybrid_power_kw"], 1),
            "tire_compound": sim_state_dict["tire_compound"],
            "tire_age_laps": sim_state_dict["tire_age_laps"],
            "tire_temp_FL_C": round(sim_state_dict["tire_temperatures"]["FL"], 2),
            "tire_temp_FR_C": round(sim_state_dict["tire_temperatures"]["FR"], 2),
            "tire_temp_RL_C": round(sim_state_dict["tire_temperatures"]["RL"], 2),
            "tire_temp_RR_C": round(sim_state_dict["tire_temperatures"]["RR"], 2),
            "tire_pressure_FL_bar": round(sim_state_dict["tire_pressures"]["FL"], 3),
            "tire_pressure_FR_bar": round(sim_state_dict["tire_pressures"]["FR"], 3),
            "tire_pressure_RL_bar": round(sim_state_dict["tire_pressures"]["RL"], 3),
            "tire_pressure_RR_bar": round(sim_state_dict["tire_pressures"]["RR"], 3),
            "tire_wear_FL_percent": round(sim_state_dict["tire_wear"]["FL"] * 100, 1),
            "tire_wear_FR_percent": round(sim_state_dict["tire_wear"]["FR"] * 100, 1),
            "tire_wear_RL_percent": round(sim_state_dict["tire_wear"]["RL"] * 100, 1),
            "tire_wear_RR_percent": round(sim_state_dict["tire_wear"]["RR"] * 100, 1),
            "suspension_travel_FL_mm": round(suspension_travel["suspension_travel_FL_mm"], 2),
            "suspension_travel_FR_mm": round(suspension_travel["suspension_travel_FR_mm"], 2),
            "suspension_travel_RL_mm": round(suspension_travel["suspension_travel_RL_mm"], 2),
            "suspension_travel_RR_mm": round(suspension_travel["suspension_travel_RR_mm"], 2),
            "last_pit_lap": sim_state_dict["last_pit_lap"],
            "laps_since_pit": sim_state_dict["current_lap"] - sim_state_dict["last_pit_lap"],
            "estimated_pit_window": estimated_pit_window,
            "pit_strategy": sim_state_dict["pit_strategy"],
            "track_limits_warnings": sim_state_dict["track_limits_warnings"],
            "track_limits_risk_percent": round(track_limits_risk * 100, 1)
        },
        "environmental": {
            "current_weather": sim_state_dict["current_weather"],
            "ambient_temp_C": round(sim_state_dict["ambient_temperature"], 2),
            "track_temp_C": round(sim_state_dict["track_temperature"], 2),
            "humidity_percent": random.randint(45, 85),
            "wind_speed_kmh": round(sim_state_dict["wind_speed"], 2),
            "wind_direction_deg": sim_state_dict["wind_direction"],
            "track_grip_level": round(sim_state_dict["track_grip"], 3),
            "visibility_level": round(sim_state_dict["visibility"], 3),
            "sunrise_time": "06:30",
            "sunset_time": "21:45"
        },
        "track_info": {
            "sector_1_length_km": TRACK_SECTORS["sector_1"]["length_km"],
            "sector_2_length_km": TRACK_SECTORS["sector_2"]["length_km"],
            "sector_3_length_km": TRACK_SECTORS["sector_3"]["length_km"],
            "total_length_km": TRACK_LENGTH_KM,
            "current_sector_characteristics": sector_data["characteristics"],
            "current_sector_max_speed": sector_data["max_speed_kmh"],
            "elevation_change_current": sector_data["elevation_change_m"]
        },
        "race_control": {
            "safety_car_active": sim_state_dict["safety_car_active"],
            "yellow_flag_sectors": sim_state_dict["yellow_flag_sectors"],
            "track_status": "green" if not sim_state_dict["safety_car_active"] and not sim_state_dict["yellow_flag_sectors"] else "caution"
        },
        "competitors": competitor_data,
        "active_anomalies": [
            {
                "type": ENHANCED_ANOMALIES[anomaly_id]["type"],
                "message": ENHANCED_ANOMALIES[anomaly_id]["message"],
                "severity": ENHANCED_ANOMALIES[anomaly_id]["severity"],
                "duration_remaining_sec": round(max(0, anomaly_data["end_time"] - sim_state_dict["total_simulated_time_sec"]), 2) if anomaly_data.get("end_time", 0) > 0 else "permanent"
            }
            for anomaly_id, anomaly_data in sim_state_dict["active_anomalies"].items()
        ],
        "strategy": {
            "current_strategy": sim_state_dict["pit_strategy"],
            "fuel_target_laps": round(fuel_laps_remaining, 1),
            "tire_change_recommended": any(wear > 0.7 for wear in sim_state_dict["tire_wear"].values()),
            "driver_change_due": sim_state_dict["driver_stint_time"] > 7000,
            "next_pit_recommendation": estimated_pit_window,
            "position_in_class": 1,
            "gap_to_class_leader": 0.0,
            "laps_to_class_leader": 0
        }
    }
    return telemetry_data

# --- Custom Tools for LangChain (adapted from ai_api.py & clearroute_agent_tools.py) ---

@tool
def get_latest_telemetry_tool() -> dict:
    """
    Fetches the latest single telemetry data point from the simulated backend.
    This tool provides a snapshot of the current car, environmental,
    and competitor conditions.
    """
    # This directly uses the simulator's logic and state
    return generate_enhanced_telemetry_sim(st.session_state.sim_state)

@tool
def get_telemetry_history_tool(limit: int = 50) -> list[dict]:
    """
    Fetches a list of recent telemetry data points from the simulated backend history.
    This tool is useful for analyzing trends over time (e.g., for anomaly detection).

    Args:
        limit (int): The maximum number of historical data points to retrieve.
                     Defaults to 50.
    Returns:
        list[dict]: A list of telemetry data points, oldest first.
    """
    # This now uses the history collected in Streamlit's session state
    return list(st.session_state.performance_history)[-limit:] # Ensure it's a list

@tool
def calculate_performance_metrics(telemetry_data: list) -> dict:
    """
    Calculates key performance metrics from telemetry history.
    """
    if not telemetry_data:
        return {"error": "No telemetry data provided", "data_points_analyzed": 0}

    try:
        lap_times = []
        fuel_rates = []
        tire_temps_all = {"FL": [], "FR": [], "RL": [], "RR": []}

        for point in telemetry_data:
            if isinstance(point, dict) and "car" in point:
                lap_time = point["car"].get("last_lap_time_sec", 0)
                if lap_time > 100:
                    lap_times.append(lap_time)

                fuel_rate = point["car"].get("fuel_consumption_current_L_per_lap", 0)
                if fuel_rate > 0:
                    fuel_rates.append(fuel_rate)

                for tire in tire_temps_all.keys():
                    temp = point["car"].get(f"tire_temp_{tire}_C", 0)
                    if temp > 0:
                        tire_temps_all[tire].append(temp)
        
        # Handle cases where lists might be empty for statistics functions
        metrics = {
            "lap_time_stats": {
                "average": statistics.mean(lap_times) if lap_times else 0,
                "median": statistics.median(lap_times) if lap_times else 0,
                "std_dev": statistics.stdev(lap_times) if len(lap_times) > 1 else 0,
                "trend": "improving" if len(lap_times) >= 3 and lap_times[-1] < lap_times[0] else "degrading",
                "count": len(lap_times)
            },
            "fuel_efficiency": {
                "average_consumption": statistics.mean(fuel_rates) if fuel_rates else 0,
                "trend": "improving" if len(fuel_rates) >= 3 and fuel_rates[-1] < fuel_rates[0] else "degrading",
                "count": len(fuel_rates)
            },
            "tire_temp_analysis": {
                tire: {
                    "average": statistics.mean(temps) if temps else 0,
                    "max": max(temps) if temps else 0,
                    "trend": "rising" if len(temps) >= 3 and temps[-1] > temps[0] else "stable",
                    "count": len(temps)
                }
                for tire, temps in tire_temps_all.items()
            },
            "data_points_analyzed": len(telemetry_data)
        }

        return metrics

    except Exception as e:
        # traceback.print_exc() # Disable for Streamlit Cloud deployment
        return {"error": f"Failed to calculate metrics: {str(e)}", "data_points_analyzed": len(telemetry_data)}

# Combine all tools into a list
tools = [get_latest_telemetry_tool, get_telemetry_history_tool, calculate_performance_metrics]
tool_executor_node = ToolNode(tools)

# --- Data Classes for Better Structure ---
@dataclass
class TelemetryThresholds:
    tire_temp_warning: float = 105.0
    tire_temp_critical: float = 115.0
    tire_pressure_min: float = 1.7
    tire_pressure_max: float = 2.1
    oil_temp_warning: float = 115.0
    oil_temp_critical: float = 125.0
    water_temp_warning: float = 90.0
    water_temp_critical: float = 100.0
    lap_time_degradation_percent: float = 3.0
    fuel_consumption_excess_percent: float = 15.0
    suspension_travel_max: float = 65.0

@dataclass
class RaceContext:
    target_lap_time: float = 214.2
    base_fuel_consumption: float = 2.8
    race_duration_hours: float = 24.0
    pit_window_laps: int = 40
    safety_fuel_margin: float = 5.0

# --- LangGraph State Definition ---
class RaceBrainState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    latest_telemetry: Dict[str, Any]
    telemetry_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    anomaly_report: Dict[str, Any]
    strategy_recommendation: str
    confidence_score: float
    priority_actions: List[str]
    processing_time: float
    query_number: int

# --- LLM and Agent Definitions ---
llm = None
llm_with_tools = None

try:
    if GROQ_API_KEY: # Check if key is available from secrets
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL_NAME, temperature=llm_temperature)
        llm_with_tools = llm.bind_tools(tools)
        # print(f"✅ Groq LLM ({LLM_MODEL_NAME}) initialized successfully") # Disable print for Streamlit Cloud
    else:
        raise Exception("GROQ_API_KEY not found in Streamlit secrets.")
except Exception as e:
    # print(f"⚠️ Could not initialize Groq LLM: {e}") # Disable print for Streamlit Cloud
    st.error(f"LLM Initialization Error: {e}. AI analysis will use rule-based fallback.")


# --- Enhanced Analysis Functions (from ai_api.py) ---
class TelemetryAnalyzer:
    def __init__(self):
        self.thresholds = TelemetryThresholds()
        self.race_context_defaults = RaceContext()

    def analyze_tire_condition(self, latest_data: dict, history: list) -> dict:
        car_data = latest_data.get("car", {})
        tire_analysis = {"status": "normal", "issues": [], "recommendations": []}

        for tire in ["FL", "FR", "RL", "RR"]:
            temp = car_data.get(f"tire_temp_{tire}_C", 0)
            pressure = car_data.get(f"tire_pressure_{tire}_bar", 0)

            if temp > self.thresholds.tire_temp_critical:
                tire_analysis["status"] = "critical"
                tire_analysis["issues"].append(f"{tire} tire critically hot: {temp}°C")
                tire_analysis["recommendations"].append(f"Immediate action required for {tire} tire (temp)")
            elif temp > self.thresholds.tire_temp_warning:
                if tire_analysis["status"] == "normal": tire_analysis["status"] = "warning"
                tire_analysis["issues"].append(f"{tire} tire running hot: {temp}°C")
                tire_analysis["recommendations"].append(f"Monitor {tire} tire closely (temp)")

            if pressure is not None and (pressure < self.thresholds.tire_pressure_min or pressure > self.thresholds.tire_pressure_max):
                if tire_analysis["status"] == "normal": tire_analysis["status"] = "warning"
                tire_analysis["issues"].append(f"{tire} tire pressure abnormal: {pressure} bar")
                tire_analysis["recommendations"].append(f"Check {tire} tire pressure")
        return tire_analysis

    def analyze_fuel_strategy(self, latest_data: dict, history: list) -> dict:
        car_data = latest_data.get("car", {})
        current_fuel = car_data.get("fuel_level_liters", 0)
        consumption_rate = car_data.get("fuel_consumption_current_L_per_lap", self.race_context_defaults.base_fuel_consumption)
        current_lap_num = car_data.get("lap_number", 1) # Renamed to avoid conflict with current_lap from sim_state

        remaining_laps = (current_fuel - self.race_context_defaults.safety_fuel_margin) / consumption_rate if consumption_rate > 0 else float('inf')
        laps_since_pit = current_lap_num - car_data.get("last_pit_lap", 0) # Use lap_number for calculations
        laps_to_pit = self.race_context_defaults.pit_window_laps - (laps_since_pit % self.race_context_defaults.pit_window_laps) # Correct calculation for laps to pit window

        fuel_analysis = {
            "current_fuel": current_fuel,
            "consumption_rate": consumption_rate,
            "remaining_laps": remaining_laps,
            "laps_to_pit_window": laps_to_pit,
            "status": "normal",
            "recommendations": []
        }

        if remaining_laps < laps_to_pit and laps_to_pit < float('inf'): # Check against infinity
            fuel_analysis["status"] = "critical"
            fuel_analysis["recommendations"].append(f"Fuel shortage risk: only {remaining_laps:.1f} laps remaining (pit window in {laps_to_pit} laps)")
        elif remaining_laps < laps_to_pit + 5 and laps_to_pit < float('inf'):
            fuel_analysis["status"] = "warning"
            fuel_analysis["recommendations"].append("Consider pit strategy adjustment due to approaching fuel limit")

        if consumption_rate > self.race_context_defaults.base_fuel_consumption * (1 + self.thresholds.fuel_consumption_excess_percent/100):
            fuel_analysis["status"] = "warning" if fuel_analysis["status"] == "normal" else fuel_analysis["status"]
            fuel_analysis["recommendations"].append(f"High fuel consumption detected: {consumption_rate:.2f} L/lap")
        return fuel_analysis

def generate_fallback_strategy(user_query: str, latest_telemetry: dict, anomaly_report: dict, performance_metrics: dict) -> str:
    car_data = latest_telemetry.get("car", {})
    strategy_parts = [
        "**🏁 FALLBACK STRATEGY (LLM UNAVAILABLE)**",
        "---",
        "**CURRENT SITUATION ASSESSMENT:**",
        f"• Lap Number: {car_data.get('lap_number', 'N/A')}",
        f"• Last Lap Time: {car_data.get('last_lap_time_sec', 0):.1f}s",
        f"• Fuel Level: {car_data.get('fuel_level_liters', 0):.1f}L",
        f"• FL Tire Temp: {car_data.get('tire_temp_FL_C', 0):.1f}°C",
        "",
        "**IMMEDIATE RECOMMENDATIONS (from Anomaly Analyzer):**",
    ]
    priority_actions = anomaly_report.get("immediate_actions", [])
    if priority_actions:
        for action in priority_actions:
            strategy_parts.append(f"• {action}")
    else:
        strategy_parts.append("• Continue current pace and monitor telemetry (no critical issues detected by analyzer)")
        strategy_parts.append("• Maintain tire temperature within optimal range")
    strategy_parts.extend([
        "",
        "**STRATEGIC NOTES (from Performance Analyzer):**",
        f"• Anomaly Priority Level: {anomaly_report.get('priority_level', 'UNKNOWN')}",
        f"• Lap Time Trend: {performance_metrics.get('lap_time_stats', {}).get('trend', 'Unknown')}",
        f"• Fuel Efficiency Trend: {performance_metrics.get('fuel_efficiency', {}).get('trend', 'Unknown')}",
        "",
        "⚠️ _This recommendation is based on predefined rules due to LLM unavailability. Full contextual reasoning is limited._"
    ])
    return "\n".join(strategy_parts)

# --- Node Functions for LangGraph (adapted from ai_api.py) ---
async def fetch_and_analyze_data_node(state: RaceBrainState):
    try:
        latest_data = get_latest_telemetry_tool()
        history_data = get_telemetry_history_tool({"limit": 50}) # Use 50 for full history
        metrics = calculate_performance_metrics({"telemetry_data": history_data})

        return {
            "latest_telemetry": latest_data,
            "telemetry_history": history_data,
            "performance_metrics": metrics,
            "messages": state["messages"]
        }
    except Exception as e:
        return {
            "latest_telemetry": {},
            "telemetry_history": [],
            "performance_metrics": {"error": str(e), "data_points_analyzed": 0,
                                    "lap_time_stats": {}, "fuel_efficiency": {}, "tire_temp_analysis": {}},
            "messages": state["messages"]
        }

async def enhanced_anomaly_detection_node(state: RaceBrainState):
    try:
        latest_telemetry = state["latest_telemetry"]
        telemetry_history = state["telemetry_history"]
        performance_metrics = state["performance_metrics"]

        analyzer = TelemetryAnalyzer()
        tire_analysis = analyzer.analyze_tire_condition(latest_telemetry, telemetry_history)
        fuel_analysis = analyzer.analyze_fuel_strategy(latest_telemetry, telemetry_history)

        analysis_context = {
            "latest_telemetry": latest_telemetry,
            "performance_metrics": performance_metrics,
            "tire_analysis": tire_analysis,
            "fuel_analysis": fuel_analysis,
            "recent_history_points": len(telemetry_history),
            "active_anomalies_from_sim": latest_telemetry.get("active_anomalies", [])
        }

        anomaly_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are the Elite Anomaly Detection Specialist for United Autosports' Le Mans team.
             Your expertise combines real-time telemetry analysis with predictive modeling to identify
             critical issues before they become race-ending problems.

             ANALYSIS CONTEXT (JSON):
             {analysis_context_json_str}

             DETECTION PRIORITIES (in order):
             1. **CRITICAL SAFETY ISSUES** - Immediate race-stopping problems
                - Tire failure risk (temp >115°C, pressure <1.7 bar, rapid degradation)
                - Engine overheating (oil >125°C, water >100°C)
                - Suspension failure (travel >65mm consistently)

             2. **PERFORMANCE DEGRADATION** - Issues affecting competitive position
                - Lap time degradation >3% from baseline (214.2s)
                - Fuel consumption >15% above expected (2.8 L/lap)
                - Tire temperature imbalance >10°C between sides

             3. **STRATEGIC OPPORTUNITIES** - Competitive advantages
                - Competitor weaknesses from their data
                - Weather change impacts
                - Optimal pit window timing

             4. **PREDICTIVE ALERTS** - Future problem prevention
                - Tire wear trends suggesting early pit need
                - Fuel consumption trends affecting strategy
                - Component temperature trends

             ANALYSIS METHODOLOGY:
             - Compare current values against baseline thresholds
             - Analyze trends from performance metrics
             - Correlate multiple data points for confirmation
             - Consider race context (lap number, weather, competitors, active simulator anomalies)
             - Assess urgency and impact on race outcome

             OUTPUT FORMAT - Respond with ONLY this JSON structure (no preambles, no postambles, no markdown fences):
             {{
                 "priority_level": "CRITICAL|HIGH|MEDIUM|LOW|NONE",
                 "primary_anomaly": {{
                     "type": "specific_issue_type",
                     "affected_component": "component_name",
                     "current_values": {{"key": "value"}},
                     "deviation_from_normal": "percentage_or_description",
                     "confidence": "HIGH|MEDIUM|LOW"
                 }},
                 "secondary_issues": [
                     {{
                         "type": "issue_type",
                         "severity": "HIGH|MEDIUM|LOW",
                         "description": "brief_description"
                     }}
                 ],
                 "trend_analysis": {{
                     "lap_time_trend": "IMPROVING|STABLE|DEGRADING",
                     "tire_condition_trend": "GOOD|DEGRADING|CRITICAL",
                     "fuel_efficiency_trend": "IMPROVING|STABLE|DEGRADING"
                 }},
                 "immediate_actions": [
                     "specific_action_1",
                     "specific_action_2"
                 ],
                 "predictive_alerts": [
                     "future_issue_warning_1",
                     "future_issue_warning_2"
                 ],
                 "reasoning": "detailed_technical_explanation"
             }}
             """),
            ("human", "Analyze the current race telemetry for anomalies and strategic insights.")
        ])

        if llm is None:
            if tire_analysis["status"] != "normal" or fuel_analysis["status"] != "normal":
                anomaly_data = {
                    "priority_level": "HIGH" if tire_analysis["status"] == "critical" or fuel_analysis["status"] == "critical" else "MEDIUM",
                    "primary_anomaly": {
                        "type": tire_analysis["issues"][0] if tire_analysis["issues"] else (fuel_analysis["issues"][0] if fuel_analysis["issues"] else "Rule-based issue"),
                        "affected_component": "Tires" if tire_analysis["issues"] else "Fuel System",
                        "confidence": "HIGH",
                        "current_values": {"tire_status": tire_analysis["status"], "fuel_status": fuel_analysis["status"]},
                        "deviation_from_normal": "Based on thresholds"
                    },
                    "secondary_issues": fuel_analysis.get("issues", []),
                    "trend_analysis": {"lap_time_trend": performance_metrics.get("lap_time_stats", {}).get("trend", "UNKNOWN"), "tire_condition_trend": "DEGRADING" if tire_analysis["status"] != "normal" else "STABLE", "fuel_efficiency_trend": performance_metrics.get("fuel_efficiency", {}).get("trend", "UNKNOWN")},
                    "immediate_actions": tire_analysis.get("recommendations", []) + fuel_analysis.get("recommendations", []),
                    "predictive_alerts": [],
                    "reasoning": "Rule-based detection due to LLM unavailability."
                }
            else:
                anomaly_data = {
                    "priority_level": "NONE",
                    "primary_anomaly": {"type": "None", "confidence": "NONE", "affected_component": "N/A", "current_values": {}, "deviation_from_normal": "N/A"},
                    "secondary_issues": [],
                    "trend_analysis": {"lap_time_trend": "UNKNOWN", "tire_condition_trend": "UNKNOWN", "fuel_efficiency_trend": "UNKNOWN"},
                    "immediate_actions": [],
                    "predictive_alerts": [],
                    "reasoning": "No significant anomalies detected by rules."
                }
            response = AIMessage(content=json.dumps(anomaly_data))
        else:
            response = await llm.ainvoke(anomaly_prompt.format_messages(
                analysis_context_json_str=json.dumps(analysis_context, indent=2)
            ))
            response_text = response.content.strip()

            json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL)
            if json_match:
                try:
                    anomaly_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    anomaly_data = {
                        "priority_level": "LOW",
                        "primary_anomaly": {"type": "parsing_error", "confidence": "LOW", "affected_component": "N/A", "current_values": {}, "deviation_from_normal": "N/A"},
                        "secondary_issues": [],
                        "trend_analysis": {"lap_time_trend": "UNKNOWN", "tire_condition_trend": "UNKNOWN", "fuel_efficiency_trend": "UNKNOWN"},
                        "immediate_actions": [],
                        "predictive_alerts": [],
                        "reasoning": f"Failed to parse LLM response after extraction: {response_text[:200]}..."
                    }
            else:
                try:
                    anomaly_data = json.loads(response_text)
                except json.JSONDecodeError:
                    anomaly_data = {
                        "priority_level": "LOW",
                        "primary_anomaly": {"type": "format_error", "confidence": "LOW", "affected_component": "N/A", "current_values": {}, "deviation_from_normal": "N/A"},
                        "secondary_issues": [],
                        "trend_analysis": {"lap_time_trend": "UNKNOWN", "tire_condition_trend": "UNKNOWN", "fuel_efficiency_trend": "UNKNOWN"},
                        "immediate_actions": [],
                        "predictive_alerts": [],
                        "reasoning": f"LLM output was not pure JSON: {response_text[:200]}..."
                    }

        return {
            "anomaly_report": anomaly_data,
            "messages": state["messages"] + [response]
        }

    except Exception as e:
        # traceback.print_exc()
        return {
            "anomaly_report": {
                "priority_level": "ERROR",
                "primary_anomaly": {"type": "node_failure", "confidence": "HIGH", "affected_component": "Anomaly Detection Node", "current_values": {}, "deviation_from_normal": ""},
                "secondary_issues": [], "trend_analysis": {"lap_time_trend": "UNKNOWN", "tire_condition_trend": "UNKNOWN", "fuel_efficiency_trend": "UNKNOWN"},
                "immediate_actions": ["Check AI service logs"], "predictive_alerts": [],
                "reasoning": f"Anomaly detection node failed: {str(e)}"
            },
            "messages": state["messages"]
        }


async def strategic_decision_node(state: RaceBrainState):
    try:
        latest_telemetry = state["latest_telemetry"]
        anomaly_report = state["anomaly_report"]
        performance_metrics = state["performance_metrics"]

        user_query_message = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), HumanMessage(content="What is the optimal race strategy?"))
        user_query = user_query_message.content

        car_data = latest_telemetry.get("car", {})
        env_data = latest_telemetry.get("environmental", {})
        competitors = latest_telemetry.get("competitors", [])

        race_context_data = {
            "session_time": f"{latest_telemetry.get('timestamp_simulated_sec', 0) / 3600:.2f} hours",
            "current_lap": car_data.get("lap_number", 0),
            "position_in_race": "1st",
            "fuel_level": car_data.get("fuel_level_liters", 0),
            "last_lap_time": car_data.get("last_lap_time_sec", 0),
            "weather_conditions": f"Rain: {env_data.get('rain_intensity', 0)}/3, Grip: {env_data.get('track_grip_level', 1.0)}",
            "tire_condition": f"FL: {car_data.get('tire_temp_FL_C', 0)}°C",
            "competitor_gaps": [f"{comp.get('name', 'N/A')}: {comp.get('gap_to_leader_sec', 'N/A')}s" for comp in competitors[:3]]
        }

        strategy_prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"""You are the Chief Race Strategist for United Autosports at Le Mans 24 Hours.
             Your decisions directly impact our chances of winning this 24-hour endurance race.

             STRATEGIC FRAMEWORK:
             1. **IMMEDIATE PRIORITIES** - Address critical issues first
             2. **COMPETITIVE POSITIONING** - Maintain/improve race position
             3. **LONG-TERM OPTIMIZATION** - Manage resources for 24-hour duration
             4. **RISK MITIGATION** - Prevent race-ending failures
             5. **OPPORTUNITY EXPLOITATION** - Capitalize on competitor mistakes

             CURRENT RACE SITUATION:
             - Session Time: {{session_time_val}}
             - Current Lap: {{current_lap_val}}
             - Position: {{position_val}}
             - Fuel Level: {{fuel_level_val}}L
             - Last Lap Time: {{last_lap_time_val}}s
             - Weather: {{weather_val}}
             - Tire FL Temp: {{tire_temp_val}}
             - Competitor Gaps: {{competitor_gaps_val}}

             ANOMALY ANALYSIS RESULTS:
             Priority: {{anomaly_priority_val}}
             Primary Issue: {{anomaly_primary_type_val}}
             Confidence: {{anomaly_confidence_val}}
             Reasoning: {{anomaly_reasoning_val}}
             Immediate Actions: {{anomaly_immediate_actions_val}}

             PERFORMANCE TRENDS:
             Lap Time Trend: {{lap_trend_val}} (Avg: {{lap_avg_val:.2f}}s)
             Fuel Efficiency Trend: {{fuel_trend_val}} (Avg: {{fuel_avg_val:.2f}} L/lap)

             Provide a comprehensive strategic recommendation covering:
             1. Current situation assessment
             2. Immediate actions needed (next 5-10 laps)
             3. Short-term strategy (next 30-60 minutes)
             4. Long-term considerations
             5. Risk mitigation measures
             6. Expected outcomes

             Be specific and actionable in your recommendations.

             **IMPORTANT: Your response MUST START with "<think> (internal thought process) </think>\\n\\n**" followed by "**STRATEGY RECOMMENDATION:**\\n\\n". Then provide the full structured recommendation. Include both the <think> block and the strategic recommendation.
             """),
            ("human", "Engineer Query: {user_query_val}\\n\\nProvide comprehensive strategic recommendations based on current race conditions.")
        ])

        if llm is None:
            raw_llm_response_content = generate_fallback_strategy(user_query, latest_telemetry, anomaly_report, performance_metrics)
            response = AIMessage(content=raw_llm_response_content)
        else:
            # FIX APPLIED HERE: Using `llm.ainvoke` instead of `llm_with_tools.ainvoke`
            llm_response = await llm.ainvoke(strategy_prompt.format_messages(
                user_query_val=user_query,
                session_time_val=race_context_data["session_time"],
                current_lap_val=race_context_data["current_lap"],
                position_val=race_context_data["position_in_race"],
                fuel_level_val=race_context_data["fuel_level"],
                last_lap_time_val=race_context_data["last_lap_time"],
                weather_val=race_context_data["weather_conditions"],
                tire_temp_val=race_context_data["tire_condition"],
                competitor_gaps_val=", ".join(race_context_data["competitor_gaps"]),

                anomaly_priority_val=anomaly_report.get("priority_level", "UNKNOWN"),
                anomaly_primary_type_val=anomaly_report.get("primary_anomaly", {}).get("type", "None detected"),
                anomaly_confidence_val=anomaly_report.get("primary_anomaly", {}).get("confidence", "UNKNOWN"),
                anomaly_reasoning_val=anomaly_report.get("reasoning", "No specific reasoning provided."),
                anomaly_immediate_actions_val=", ".join(anomaly_report.get("immediate_actions", ["None required"])),

                lap_trend_val=performance_metrics.get("lap_time_stats", {}).get("trend", "Unknown"),
                lap_avg_val=performance_metrics.get("lap_time_stats", {}).get("average", 0),
                fuel_trend_val=performance_metrics.get("fuel_efficiency", {}).get("trend", "Unknown"),
                fuel_avg_val=performance_metrics.get("fuel_efficiency", {}).get("average_consumption", 0)
            ))
            raw_llm_response_content = llm_response.content.strip()
            response = llm_response

        extracted_strategy = raw_llm_response_content

        confidence_score = 0.8
        if anomaly_report.get('priority_level') == 'CRITICAL':
            confidence_score = 0.95
        elif anomaly_report.get('priority_level') == 'HIGH':
            confidence_score = 0.9
        elif anomaly_report.get('priority_level') == 'MEDIUM':
            confidence_score = 0.75
        elif anomaly_report.get('priority_level') == 'LOW':
            confidence_score = 0.6
        else: # NONE
            confidence_score = 0.5

        if anomaly_report.get('primary_anomaly', {}).get('type') not in ["parsing_error", "format_error", "node_failure"]:
            if anomaly_report.get('primary_anomaly', {}).get('confidence') == 'HIGH':
                confidence_score = min(1.0, confidence_score + 0.05)
            elif anomaly_report.get('primary_anomaly', {}).get('confidence') == 'LOW':
                confidence_score = max(0.1, confidence_score - 0.1)

        priority_actions = anomaly_report.get('immediate_actions', [])

        return {
            "strategy_recommendation": extracted_strategy,
            "confidence_score": confidence_score,
            "priority_actions": priority_actions,
            "messages": state["messages"] + [response]
        }

    except Exception as e:
        # traceback.print_exc()
        fallback_strategy = f"Strategy generation encountered an error: {str(e)}. Please check system status and try again."
        return {
            "strategy_recommendation": fallback_strategy,
            "confidence_score": 0.1,
            "priority_actions": ["Check AI service logs", "Retry request"],
            "messages": state["messages"] + [AIMessage(content=fallback_strategy)]
        }

# --- Build the Enhanced LangGraph Workflow ---
workflow = StateGraph(RaceBrainState)
workflow.add_node("fetch_and_analyze", fetch_and_analyze_data_node)
workflow.add_node("anomaly_detection", enhanced_anomaly_detection_node)
workflow.add_node("strategic_decision", strategic_decision_node)
workflow.add_edge(START, "fetch_and_analyze")
workflow.set_entry_point("fetch_and_analyze")
workflow.add_edge("fetch_and_analyze", "anomaly_detection")
workflow.add_edge("anomaly_detection", "strategic_decision")
workflow.add_edge("strategic_decision", END)
app_graph = workflow.compile()


# --- Streamlit UI and Logic ---

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
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .analytics-card {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 100%;
    }
    
    .race-control-panel {
        background: rgba(30, 30, 46, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- API Functions (Adapted for internal calls and state management) ---
# @st.cache_data(ttl=1) # Removed caching as this function now updates sim_state
def get_live_data():
    # Calling this directly triggers a simulation step and updates sim_state
    # This design means the simulation advances *every time Streamlit reruns*.
    # The `sim_speed_factor` controls how much simulated time passes per real-time Streamlit update.
    data = get_latest_telemetry_tool()

    if data and 'car' in data:
        # Check for driver change to reset stint info
        current_driver_from_sim = data['car'].get('current_driver', 'N/A')
        if st.session_state.selected_driver != current_driver_from_sim:
            st.session_state.selected_driver = current_driver_from_sim
            st.session_state.driver_stint_start = time.time()
            st.session_state.driver_stint_laps_start = data['car'].get('lap_number', 0)
        
        # Add current data point to performance history
        performance_point = {
            'timestamp_sec': data.get('timestamp_simulated_sec', time.time()),
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

        max_history_points = st.session_state.get('telemetry_limit', 100)
        if len(st.session_state.performance_history) > max_history_points:
            st.session_state.performance_history.pop(0)

    return data

def get_telemetry_history_ui(limit=50):
    # This now gets history from Streamlit's session state
    return list(st.session_state.performance_history)[-limit:] # Ensure it's a list

async def query_race_brain_ai_langgraph(user_query: str):
    # This directly invokes the LangGraph workflow
    try:
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "latest_telemetry": get_latest_telemetry_tool(), # Use the tool directly
            "telemetry_history": get_telemetry_history_tool(st.session_state.telemetry_limit), # Use the tool directly with UI limit
            "performance_metrics": {},
            "anomaly_report": {},
            "strategy_recommendation": "",
            "confidence_score": 0.0,
            "priority_actions": [],
            "query_number": st.session_state.get('query_counter', 0) + 1 # Increment query counter
        }

        # The graph executes the nodes sequentially
        final_state = await app_graph.ainvoke(initial_state)

        ai_response = {
            "strategy_recommendation": final_state.get("strategy_recommendation", "AI could not generate a recommendation."),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "priority_actions": final_state.get("priority_actions", []),
            "anomaly_report": final_state.get("anomaly_report", {}),
            "timestamp": datetime.now().isoformat()
        }
        return ai_response
    except Exception as e:
        # traceback.print_exc() # Don't print full trace in Streamlit Cloud console by default
        st.exception(e) # Display exception in Streamlit UI
        return {
            "strategy_recommendation": f"Error: AI processing failed: {str(e)}. Check Streamlit logs for full trace.",
            "confidence_score": 0.0,
            "priority_actions": [],
            "anomaly_report": {},
            "timestamp": datetime.now().isoformat()
        }

# --- Helper Functions (Your existing UI helpers) ---
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
    if st.session_state.prev_position is None:
        st.session_state.prev_position = current_position
        return "→", "#FFC107"

    change = st.session_state.prev_position - current_position

    if change > 0:
        delta_text = f"↑{change}"
        color = "#4CAF50"
    elif change < 0:
        delta_text = f"↓{abs(change)}"
        color = "#F44336"
    else:
        delta_text = "→"
        color = "#FFC107"

    st.session_state.prev_position = current_position
    return delta_text, color

def parse_llm_response(raw_response: str) -> (str, str):
    think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
    think_process = think_match.group(1).strip() if think_match else "No internal thought process provided."

    strategy_match = re.search(r"STRATEGY RECOMMENDATION:\s*(.*)", raw_response, re.DOTALL)
    strategy_recommendation = strategy_match.group(1).strip() if strategy_match else raw_response.strip()

    return think_process, strategy_recommendation

# Mock data generation functions for sections not directly from simulator
# These are still used for "filler" data where the main simulator doesn't generate it
def generate_technical_data_mock(live_data_car):
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
        "downforce_setting": "Medium",
        "brake_balance": "58% Front",
        "suspension_travel_FL_mm": live_data_car.get('suspension_travel_FL_mm', np.random.uniform(20, 50)),
        "suspension_travel_FR_mm": live_data_car.get('suspension_travel_FR_mm', np.random.uniform(20, 50)),
        "suspension_travel_RL_mm": live_data_car.get('suspension_travel_RL_mm', np.random.uniform(20, 50)),
        "suspension_travel_RR_mm": live_data_car.get('suspension_travel_RR_mm', np.random.uniform(20, 50)),
    }

def generate_analytics_data_mock():
    return pd.DataFrame({
        'metric': ['Speed', 'Fuel Efficiency', 'Tire Wear', 'Braking', 'Acceleration', 'Cornering'],
        'current': [92 + np.random.uniform(-2,2), 85 + np.random.uniform(-2,2), 78 + np.random.uniform(-2,2), 88 + np.random.uniform(-2,2), 90 + np.random.uniform(-2,2), 86 + np.random.uniform(-2,2)],
        'target': [95, 90, 85, 90, 92, 90],
        'variance': [-3, -5, -7, -2, -2, -4]
    })

def generate_race_control_data_mock(live_data):
    return {
        "competitors": live_data.get('competitors', []),
        "strategy": live_data.get('strategy', {}),
        "race_info": live_data.get('race_info', {}),
        "environmental": live_data.get('environmental', {})
    }


# --- UI Layout ---
st.markdown('<div class="main-header">🏁 UNITED AUTOSPORTS RACEBRAIN AI PRO 🏁</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🏎️ Race Control Center")
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/United_Autosports.svg/1200px-United_Autosports.svg.png", width=200)
    st.markdown("---")
    
    # Driver Information Card
    live_data_for_sidebar = get_live_data() or {} # Call the live data generator
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
        st.session_state.sim_speed_factor = st.slider("Simulator Speed Factor (X real-time)", 1, 60, st.session_state.sim_speed_factor, help="How many simulated seconds pass per real second.")
        show_debug = st.checkbox("Show Debug Info", False)
        st.session_state.telemetry_limit = st.slider("Telemetry History Points", 20, 200, 50)

        if st.button("Reset Simulator"):
            st.session_state.sim_state = _initial_simulator_state.copy() # Use the global initial state
            st.session_state.performance_history = []
            st.session_state.chat_history = []
            st.session_state.ai_response_cache = {
                "strategy_recommendation": "AI could not generate a recommendation.",
                "confidence_score": 0.0,
                "priority_actions": [],
                "anomaly_report": {}
            }
            st.session_state.prev_position = None
            st.rerun()

        st.subheader("Manual Anomaly Trigger (Simulator)")
        anomaly_options = {k: ENHANCED_ANOMALIES[k]["type"] for k in ENHANCED_ANOMALIES if k != "0"}
        selected_anomaly_id = st.selectbox("Select Anomaly to Trigger", list(anomaly_options.keys()), format_func=lambda x: anomaly_options[x], key="manual_anomaly_select")
        if st.button(f"Trigger {anomaly_options.get(selected_anomaly_id, 'Anomaly')}"):
            st.session_state.sim_state["command_queue_sim"].append(f"anomaly_{selected_anomaly_id}")
            st.info(f"Anomaly '{anomaly_options[selected_anomaly_id]}' queued for activation.")
            st.rerun()
        if st.button("Reset Anomalies (ID 0)"):
            st.session_state.sim_state["command_queue_sim"].append("anomaly_0")
            st.info("All anomalies reset.")
            st.rerun()


    # Driver Selection (This allows a visual selection, but current_driver from sim overrides actual active driver display)
    available_drivers_list_for_selectbox = OUR_CAR_CONFIG["drivers"]

    default_index = 0
    if current_driver_from_sim in available_drivers_list_for_selectbox:
        default_index = available_drivers_list_for_selectbox.index(current_driver_from_sim)

    st.selectbox("👤 Filter by Driver", available_drivers_list_for_selectbox,
                 index=default_index, key="sidebar_driver_select")
    
    # System Status
    st.markdown("---")
    st.subheader("🔌 System Status")
    status_placeholder = st.empty()
    
    # Position Tracker
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
st.session_state.chart_counter += 1

live_data = get_live_data() # This call drives the simulation step
telemetry_history = get_telemetry_history_ui(st.session_state.telemetry_limit)

tech_data = generate_technical_data_mock(live_data.get('car', {}))
analytics_data = generate_analytics_data_mock()
race_control_data = generate_race_control_data_mock(live_data)

with st.container():
    current_car = live_data.get('car', {})
    race_info = live_data.get('race_info', {})
    env_info = live_data.get('environmental', {})
    
    # Tab 1: Live Race
    with tab1:
        st.subheader("🏁 LIVE RACE COMMAND CENTER")
        
        current_position = current_car.get('position_in_class', 1)
        position_change_text, position_change_color = get_position_change_ui(current_position)
        
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
            st.metric("🚀 SPEED", f"{speed:.1f} km/h", delta=f"{speed - 250:.1f}" if speed else None)
        with col_fuel:
            fuel = current_car.get('fuel_level_liters', 0)
            st.metric("⛽ FUEL", f"{fuel:.1f} L", delta=f"{fuel - 75:.1f}" if fuel else None, delta_color="inverse")
        with col_lap_time:
            lap_time = current_car.get('last_lap_time_sec', 0)
            avg_lap_time = current_car.get('average_lap_time_sec', 0)
            delta_lap_time = lap_time - avg_lap_time if avg_lap_time else 0
            st.metric("⏰ LAST LAP", f"{lap_time:.2f}s" if lap_time else "N/A", delta=f"{delta_lap_time:.2f}s" if lap_time else None, delta_color="inverse")
        
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
            avg_tire_temp = np.mean(tire_temps) if len(tire_temps) > 0 else 0 # Ensure non-empty list for mean
            temp_gauge = create_enhanced_gauge(
                avg_tire_temp, "AVG TIRE TEMP", 0, 150, "°C",
                [(0, 80, "#4CAF50"), (80, 120, "#FFC107"), (120, 150, "#F44336")]
            )
            st.plotly_chart(temp_gauge, use_container_width=True, key=f"live_avg_tire_temp_gauge_{st.session_state.chart_counter}")
        
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
        
        st.markdown("---")
        st.subheader("🚨 LIVE ALERTS & ANOMALIES")
        active_sim_anomalies = live_data.get('active_anomalies', [])
        if active_sim_anomalies:
            for anomaly in active_sim_anomalies:
                severity = anomaly.get('severity', 'UNKNOWN').upper()
                message = anomaly.get('message', 'Unknown issue')
                
                alert_class = ""
                if severity == 'CRITICAL': alert_class = "alert-critical"
                elif severity == 'HIGH': alert_class = "alert-high"
                elif severity == 'MEDIUM': alert_class = "alert-medium"
                
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
        
        st.markdown("---")
        st.subheader("📈 SESSION STATISTICS")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        lap_times_history = [d.get('lap_time', 0) for d in st.session_state.performance_history if d.get('lap_time', 0) > 0]
        speeds_history = [d.get('speed', 0) for d in st.session_state.performance_history if d.get('speed', 0) > 0]
        fuel_consumptions_history = [d.get('fuel_consumption', 0) for d in st.session_state.performance_history if d.get('fuel_consumption', 0) > 0]
        
        with stat_col1:
            best_lap = min(lap_times_history) if lap_times_history else 0
            st.metric("🏆 BEST LAP TIME", f"{best_lap:.3f}s" if best_lap > 0 else "N/A")
        with stat_col2:
            avg_speed = np.mean(speeds_history) if speeds_history else 0
            st.metric("📊 AVG SPEED", f"{avg_speed:.1f} km/h" if not np.isnan(avg_speed) else "N/A")
        with stat_col3:
            if st.session_state.performance_history and len(st.session_state.performance_history) > 1:
                initial_fuel = st.session_state.performance_history[0].get('fuel', 0)
                final_fuel = st.session_state.performance_history[-1].get('fuel', 0)
                fuel_used = initial_fuel - final_fuel
                st.metric("⛽ TOTAL FUEL USED", f"{fuel_used:.1f}L" if fuel_used > 0 else "N/A")
            else:
                st.metric("⛽ TOTAL FUEL USED", "N/A")
        with stat_col4:
            total_laps = st.session_state.sim_state["current_lap"]
            st.metric("🔄 TOTAL LAPS", str(int(total_laps)) if not np.isnan(total_laps) else "N/A")
    
    # Tab 3: AI Strategist
    with tab3:
        st.subheader("🧠 RACEBRAIN AI STRATEGIC COMMAND")
        
        if st.session_state.ai_response_cache and st.session_state.ai_response_cache.get('strategy_recommendation') != "AI could not generate a recommendation.":
            ai_res = st.session_state.ai_response_cache
            
            confidence = ai_res.get('confidence_score', 0)
            conf_color = "#4CAF50" if confidence > 0.8 else "#FFC107" if confidence > 0.5 else "#F44336"
            conf_text = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.5 else "LOW"
            
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
            user_question_input = st.text_input(
                "💬 ASK RACEBRAIN AI:",
                value=st.session_state.ai_query_input_value,
                placeholder="e.g., 'What's our optimal pit strategy?' or 'Analyze tire degradation'",
                key="main_ai_query_input",
                on_change=handle_query_submit
            )
        
        with col_query2:
            ask_button = st.button("🚀 QUERY AI", type="primary", use_container_width=True, key="ask_ai_button")
        
        st.markdown("**QUICK STRATEGY QUERIES:**")
        quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
        
        def handle_quick_query(query_text):
            st.session_state.ai_query_input_value = query_text
            with st.spinner("🧠 RACEBRAIN AI ANALYZING TELEMETRY DATA..."):
                # Use asyncio.run for calling the async LangGraph function in a sync context
                ai_output = asyncio.run(query_race_brain_ai_langgraph(query_text))
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
                    st.rerun()
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
        
        if ask_button or st.session_state.ai_query_submitted:
            query_to_process = user_question_input
            if query_to_process:
                st.session_state.ai_query_submitted = False
                st.session_state.ai_query_input_value = ""

                with st.spinner("🧠 RACEBRAIN AI ANALYZING TELEMETRY DATA..."):
                    ai_output = asyncio.run(query_race_brain_ai_langgraph(query_to_process))
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
                st.session_state.ai_query_submitted = False
        
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
        
        st.markdown("---")
        st.subheader("📊 TELEMETRY ANALYSIS")
        
        tech_chart_col1, tech_chart_col2 = st.columns(2)
        
        with tech_chart_col1:
            df_hist_current = pd.DataFrame(st.session_state.performance_history)

            fig = go.Figure()

            if not df_hist_current.empty and 'oil_temp_C' in df_hist_current.columns and df_hist_current['oil_temp_C'].any():
                fig.add_trace(go.Scatter(
                    x=df_hist_current['timestamp_sec'],
                    y=df_hist_current['oil_temp_C'],
                    name="Oil Temp",
                    line=dict(color='#FF6B35')
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=np.arange(st.session_state.telemetry_limit),
                    y=np.random.normal(95, 3, st.session_state.telemetry_limit),
                    name="Oil Temp",
                    line=dict(color='#FF6B35')
                ))

            if not df_hist_current.empty and 'water_temp_C' in df_hist_current.columns and df_hist_current['water_temp_C'].any():
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

            suspension_cols = ['suspension_travel_FL_mm', 'suspension_travel_FR_mm', 'suspension_travel_RL_mm', 'suspension_travel_RR_mm']
            if not df_hist_current.empty and all(col in df_hist_current.columns and df_hist_current[col].any() for col in suspension_cols):
                fig.add_trace(go.Scatter(x=df_hist_current['timestamp_sec'], y=df_hist_current['suspension_travel_FL_mm'], name="Front Left", line=dict(color='#FF6B35')))
                fig.add_trace(go.Scatter(x=df_hist_current['timestamp_sec'], y=df_hist_current['suspension_travel_FR_mm'], name="Front Right", line=dict(color='#1E88E5')))
                fig.add_trace(go.Scatter(x=df_hist_current['timestamp_sec'], y=df_hist_current['suspension_travel_RL_mm'], name="Rear Left", line=dict(color='#4CAF50')))
                fig.add_trace(go.Scatter(x=df_hist_current['timestamp_sec'], y=df_hist_current['suspension_travel_RR_mm'], name="Rear Right", line=dict(color='#FFC107')))
            else:
                fig.add_trace(go.Scatter(x=np.arange(st.session_state.telemetry_limit), y=np.random.normal(45, 5, st.session_state.telemetry_limit), name="Front Left", line=dict(color='#FF6B35')))
                fig.add_trace(go.Scatter(x=np.arange(st.session_state.telemetry_limit), y=np.random.normal(42, 4, st.session_state.telemetry_limit), name="Front Right", line=dict(color='#1E88E5')))
                fig.add_trace(go.Scatter(x=np.arange(st.session_state.telemetry_limit), y=np.random.normal(38, 3, st.session_state.telemetry_limit), name="Rear Left", line=dict(color='#4CAF50')))
                fig.add_trace(go.Scatter(x=np.arange(st.session_state.telemetry_limit), y=np.random.normal(40, 4, st.session_state.telemetry_limit), name="Rear Right", line=dict(color='#FFC107')))

            fig.update_layout(
                title="SUSPENSION TRAVEL ANALYSIS",
                xaxis_title="Time (s)",
                yaxis_title="Travel (mm)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig, use_container_width=True, key=f"tech_suspension_travel_chart_{st.session_state.chart_counter}")

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
            tire_pred_info = "N/A (no data)"
            fuel_pred_info = "N/A (no data)"
            
            if st.session_state.performance_history and len(st.session_state.performance_history) > 10:
                df_analytics = pd.DataFrame(st.session_state.performance_history)
                
                # Tire Prediction
                tire_wear_data = df_analytics[['timestamp_sec', 'tire_wear_FL_percent']].dropna()
                if len(tire_wear_data) >= 2 and tire_wear_data['tire_wear_FL_percent'].any():
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        tire_wear_data['timestamp_sec'], tire_wear_data['tire_wear_FL_percent']
                    )
                    if slope > 0:
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
                    recent_data = df_analytics.tail(10)
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
            else:
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
                if 'fuel' in df_analytics.columns:
                    df_analytics['fuel_diff'] = df_analytics['fuel'].diff()
                    pit_stops_indices = df_analytics[(df_analytics['fuel_diff'] > 10) & (df_analytics['fuel_diff'].notna())].index.tolist()
                else:
                    pit_stops_indices = []

                if pit_stops_indices:
                    st.write(f"**Detected Pit Stops:** {len(pit_stops_indices)}")
                    
                    stint_analysis = []
                    current_stint_start_idx = 0
                    
                    all_pit_indices = pit_stops_indices + [len(df_analytics) - 1]
                    all_pit_indices = sorted(list(set(all_pit_indices)))
                    
                    for i, pit_idx in enumerate(all_pit_indices):
                        if pit_idx < current_stint_start_idx:
                            continue
                        
                        stint_data = df_analytics.iloc[current_stint_start_idx:pit_idx+1]
                        
                        if len(stint_data) > 5:
                            avg_lap_time_stint = stint_data['lap_time'].mean() if 'lap_time' in stint_data.columns and stint_data['lap_time'].any() else np.nan
                            
                            if 'fuel' in stint_data.columns:
                                fuel_at_start = stint_data['fuel'].iloc[0]
                                fuel_at_end = stint_data['fuel'].iloc[-1]
                                fuel_used_in_stint = fuel_at_start - fuel_at_end
                                if fuel_used_in_stint < 0:
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
                        current_stint_start_idx = pit_idx + 1
                    
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
        
        st.markdown("---")
        st.subheader("🏎️ COMPETITOR ANALYSIS")
        
        competitors = race_control_data.get('competitors', [])
        if competitors:
            df_competitors = pd.DataFrame(competitors)
            
            def highlight_our_car(s):
                if s['name'] == OUR_CAR_CONFIG["name"]: # Use our_car_config for highlighting
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
    st.info("Simulated data not available. Check simulator logic or refresh.")

# Auto-refresh loop
time.sleep(refresh_interval)
st.rerun()
