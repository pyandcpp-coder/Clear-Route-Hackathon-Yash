import time
import random
import json
import threading
import queue 
import requests
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np



BACKEND_API_URL = "http://localhost:8000/telemetry"


TRACK_LENGTH_KM = 13.626  # Official track length
TOTAL_LAPS_24H = 385  # Approximate laps for 24h race
BASE_LAP_TIME_SEC = 210.0  # ~3:30 for LMP1/Hypercar
FASTEST_LAP_TIME_SEC = 195.0  # Qualifying pace
SLOWEST_LAP_TIME_SEC = 240.0  # Traffic/issues


TRACK_SECTORS = {
    "sector_1": {  # Start/Finish to Tertre Rouge
        "length_km": 4.2,
        "characteristics": ["long_straight", "chicane", "medium_corners"],
        "avg_speed_kmh": 280,
        "max_speed_kmh": 340,
        "elevation_change_m": 15,
        "tire_stress": "medium",
        "fuel_consumption_modifier": 1.2  # High speed = more fuel
    },
    "sector_2": {  # Tertre Rouge to Mulsanne
        "length_km": 6.0,
        "characteristics": ["mulsanne_straight", "indianapolis", "arnage"],
        "avg_speed_kmh": 320,
        "max_speed_kmh": 370,  # Real Mulsanne straight speeds
        "elevation_change_m": -10,
        "tire_stress": "low",
        "fuel_consumption_modifier": 1.4  # Highest fuel consumption
    },
    "sector_3": {  # Mulsanne to Finish
        "length_km": 3.426,
        "characteristics": ["porsche_curves", "ford_chicanes", "start_finish"],
        "avg_speed_kmh": 180,
        "max_speed_kmh": 250,
        "elevation_change_m": 5,
        "tire_stress": "high",
        "fuel_consumption_modifier": 0.9  # Technical section
    }
}

# Real manufacturer data for 2024 Le Mans Hypercar class
HYPERCAR_COMPETITORS = {
    "2": {
        "name": "Cadillac V-Series.R #2",
        "manufacturer": "Cadillac",
        "drivers": ["Earl Bamber", "Alex Lynn", "Richard Westbrook"],
        "base_lap_time": 208.5,
        "current_gap_sec": -12.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.9,
        "tire_degradation_rate": 0.6,
        "reliability_factor": 0.92,  # 92% reliability
        "pit_frequency_laps": 38,
        "strengths": ["straight_line_speed", "fuel_efficiency"],
        "weaknesses": ["tire_wear", "slow_corners"]
    },
    "3": {
        "name": "Cadillac V-Series.R #3",
        "manufacturer": "Cadillac", 
        "drivers": ["Sebastien Bourdais", "Renger van der Zande", "Scott Dixon"],
        "base_lap_time": 209.2,
        "current_gap_sec": 8.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.8,
        "tire_degradation_rate": 0.5,
        "reliability_factor": 0.94,
        "pit_frequency_laps": 40,
        "strengths": ["consistency", "driver_lineup"],
        "weaknesses": ["qualifying_pace"]
    },
    "5": {
        "name": "Porsche 963 #5",
        "manufacturer": "Porsche",
        "drivers": ["Michael Christensen", "Kevin Estre", "Laurens Vanthoor"],
        "base_lap_time": 207.8,
        "current_gap_sec": -8.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.7,
        "tire_degradation_rate": 0.4,
        "reliability_factor": 0.96,
        "pit_frequency_laps": 42,
        "strengths": ["handling", "reliability", "tire_management"],
        "weaknesses": ["straight_line_speed"]
    },
    "6": {
        "name": "Porsche 963 #6",
        "manufacturer": "Porsche",
        "drivers": ["Andre Lotterer", "Kevin Estre", "Laurens Vanthoor"],
        "base_lap_time": 208.1,
        "current_gap_sec": 15.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.75,
        "tire_degradation_rate": 0.45,
        "reliability_factor": 0.95,
        "pit_frequency_laps": 41,
        "strengths": ["consistency", "night_pace"],
        "weaknesses": ["traffic_management"]
    },
    "7": {
        "name": "Toyota GR010 Hybrid #7",
        "manufacturer": "Toyota",
        "drivers": ["Mike Conway", "Kamui Kobayashi", "Jose Maria Lopez"],
        "base_lap_time": 206.9,
        "current_gap_sec": -18.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.6,  # Hybrid advantage
        "tire_degradation_rate": 0.5,
        "reliability_factor": 0.93,
        "pit_frequency_laps": 44,  # Better fuel economy
        "strengths": ["hybrid_system", "fuel_efficiency", "pace"],
        "weaknesses": ["reliability_history"]
    },
    "8": {
        "name": "Toyota GR010 Hybrid #8",
        "manufacturer": "Toyota",
        "drivers": ["Sebastien Buemi", "Ryo Hirakawa", "Brendon Hartley"],
        "base_lap_time": 207.2,
        "current_gap_sec": -5.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.65,
        "tire_degradation_rate": 0.48,
        "reliability_factor": 0.91,
        "pit_frequency_laps": 43,
        "strengths": ["hybrid_recovery", "driver_experience"],
        "weaknesses": ["electrical_issues"]
    },
    "11": {
        "name": "Isotta Fraschini Tipo6 #11",
        "manufacturer": "Isotta Fraschini",
        "drivers": ["Jean-Karl Vernay", "Antonio Giovinazzi", "Robin Frijns"],
        "base_lap_time": 213.5,  # Slower as new manufacturer
        "current_gap_sec": 45.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 3.1,
        "tire_degradation_rate": 0.7,
        "reliability_factor": 0.85,  # New car, lower reliability
        "pit_frequency_laps": 35,
        "strengths": ["innovation"],
        "weaknesses": ["development", "pace", "reliability"]
    },
    "15": {
        "name": "BMW M Hybrid V8 #15",
        "manufacturer": "BMW",
        "drivers": ["Dries Vanthoor", "Raffaele Marciello", "Marco Wittmann"],
        "base_lap_time": 209.8,
        "current_gap_sec": 22.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.85,
        "tire_degradation_rate": 0.55,
        "reliability_factor": 0.89,
        "pit_frequency_laps": 39,
        "strengths": ["engine_power"],
        "weaknesses": ["aerodynamics", "fuel_consumption"]
    },
    "50": {
        "name": "Ferrari 499P #50",
        "manufacturer": "Ferrari",
        "drivers": ["Antonio Fuoco", "Miguel Molina", "Nicklas Nielsen"],
        "base_lap_time": 207.5,
        "current_gap_sec": -3.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.8,
        "tire_degradation_rate": 0.52,
        "reliability_factor": 0.90,
        "pit_frequency_laps": 40,
        "strengths": ["aerodynamics", "cornering"],
        "weaknesses": ["straight_line_speed", "reliability"]
    },
    "51": {
        "name": "Ferrari 499P #51",
        "manufacturer": "Ferrari",
        "drivers": ["James Calado", "Alessandro Pier Guidi", "Antonio Giovinazzi"],
        "base_lap_time": 207.8,
        "current_gap_sec": 12.0,
        "fuel_tank_liters": 68.0,
        "fuel_consumption_base": 2.82,
        "tire_degradation_rate": 0.53,
        "reliability_factor": 0.88,
        "pit_frequency_laps": 40,
        "strengths": ["driver_lineup", "race_pace"],
        "weaknesses": ["qualifying", "tire_warming"]
    }
}

# Our car (United Autosports - LMP2 spec for comparison)
OUR_CAR_CONFIG = {
    "name": "United Autosports Oreca 07 #22",
    "class": "LMP2",
    "drivers": ["Phil Hanson", "Filipe Albuquerque", "Will Owen"],
    "base_lap_time": 225.0,  # LMP2 is slower than Hypercar
    "fuel_tank_liters": 75.0,  # LMP2 has larger tank
    "fuel_consumption_base": 3.2,
    "tire_degradation_rate": 0.8,  # Higher tire wear
    "reliability_factor": 0.94,
    "pit_frequency_laps": 35,
    "target_position": 1,  # In LMP2 class
    "class_competitors": 25  # Typical LMP2 field size
}

# Enhanced weather system based on real Le Mans patterns
WEATHER_PATTERNS = {
    "clear": {"probability": 0.4, "grip_factor": 1.0, "visibility": 1.0},
    "partly_cloudy": {"probability": 0.3, "grip_factor": 0.98, "visibility": 0.95},
    "overcast": {"probability": 0.15, "grip_factor": 0.95, "visibility": 0.90},
    "light_rain": {"probability": 0.08, "grip_factor": 0.75, "visibility": 0.70},
    "heavy_rain": {"probability": 0.05, "grip_factor": 0.55, "visibility": 0.50},
    "fog": {"probability": 0.02, "grip_factor": 0.85, "visibility": 0.30}  # Early morning fog
}

# Time-based weather probability (Le Mans June weather patterns)
HOURLY_WEATHER_MODIFIERS = {
    range(6, 10): {"rain_chance": 0.15, "fog_chance": 0.3},   # Morning
    range(10, 16): {"rain_chance": 0.05, "fog_chance": 0.0},  # Midday
    range(16, 20): {"rain_chance": 0.12, "fog_chance": 0.0},  # Afternoon
    range(20, 24): {"rain_chance": 0.08, "fog_chance": 0.05}, # Evening
    range(0, 6): {"rain_chance": 0.1, "fog_chance": 0.2}      # Night
}

# Enhanced anomaly system with real Le Mans scenarios
ENHANCED_ANOMALIES = {
    "1": {
        "type": "tire_puncture_front_left",
        "message": "Front left tire showing rapid pressure loss - debris suspected",
        "duration_sec": 180,
        "severity": "critical",
        "lap_time_impact": 1.15,
        "repair_time": 45
    },
    "2": {
        "type": "hybrid_system_failure",
        "message": "Hybrid system malfunction - reduced power output",
        "duration_sec": 600,
        "severity": "high",
        "lap_time_impact": 1.08,
        "repair_time": 120
    },
    "3": {
        "type": "safety_car_period",
        "message": "Safety car deployed - incident at Indianapolis corner",
        "duration_sec": 420,
        "severity": "medium",
        "lap_time_impact": 1.35,  # Much slower under safety car
        "repair_time": 0
    },
    "4": {
        "type": "sudden_weather_change",
        "message": "Sudden rain shower approaching - grip levels dropping",
        "duration_sec": 900,
        "severity": "high",
        "lap_time_impact": 1.25,
        "repair_time": 0
    },
    "5": {
        "type": "engine_overheating",
        "message": "Engine coolant temperature rising - airflow restriction suspected",
        "duration_sec": 300,
        "severity": "critical",
        "lap_time_impact": 1.10,
        "repair_time": 180
    },
    "6": {
        "type": "brake_balance_issue",
        "message": "Brake balance shifting to rear - brake disc temperature imbalance",
        "duration_sec": 240,
        "severity": "medium",
        "lap_time_impact": 1.05,
        "repair_time": 60
    },
    "7": {
        "type": "aerodynamic_damage",
        "message": "Front splitter damage detected - downforce reduced",
        "duration_sec": 480,
        "severity": "high",
        "lap_time_impact": 1.07,
        "repair_time": 90
    },
    "8": {
        "type": "fuel_flow_restriction",
        "message": "Fuel flow rate below optimal - filter blockage suspected",
        "duration_sec": 360,
        "severity": "medium",
        "lap_time_impact": 1.04,
        "repair_time": 75
    },
    "9": {
        "type": "gearbox_sensor_fault",
        "message": "Gearbox sensor intermittent - shift quality affected",
        "duration_sec": 720,
        "severity": "medium",
        "lap_time_impact": 1.03,
        "repair_time": 45
    },
    "10": {
        "type": "night_visibility_issue", 
        "message": "Headlight alignment issue - reduced night visibility",
        "duration_sec": 1800,
        "severity": "high",
        "lap_time_impact": 1.12,  # Significant at night
        "repair_time": 120
    },
    "0": {
        "type": "reset_all_systems",
        "message": "All systems reset - optimal conditions restored",
        "duration_sec": 1,
        "severity": "info",
        "lap_time_impact": 1.0,
        "repair_time": 0
    }
}

# ===== GLOBAL STATE VARIABLES =====
current_lap = 1
current_lap_time_sec = 0.0
total_simulated_time_sec = 0.0
our_car_fuel = OUR_CAR_CONFIG["fuel_tank_liters"]
lap_start_time_simulated = 0.0
last_lap_start_time_simulated = 0.0
our_car_last_lap_time = OUR_CAR_CONFIG["base_lap_time"]
current_sector = 1
sector_progress = 0.0

# Enhanced tire model
tire_compound = "medium"  # soft, medium, hard
tire_age_laps = 0
tire_wear = {"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0}
tire_temperatures = {"FL": 95.0, "FR": 95.0, "RL": 90.0, "RR": 90.0}
tire_pressures = {"FL": 1.9, "FR": 1.9, "RL": 1.8, "RR": 1.8}

# Driver and strategy state
current_driver = 0  # Index into driver list
driver_stint_time = 0.0
last_pit_lap = 0
pit_strategy = "normal"  # normal, aggressive, conservative
fuel_saving_mode = False
push_mode = False

# Environmental state
current_weather = "clear"
track_temperature = 28.0
ambient_temperature = 20.0
wind_speed = 12.0
wind_direction = 180
track_grip = 1.0
visibility = 1.0

# Race control and incidents
safety_car_active = False
yellow_flag_sectors = []
track_limits_warnings = 0
race_incidents = []

# Enhanced competitor tracking
competitor_positions = {}
for car_id in HYPERCAR_COMPETITORS:
    competitor_positions[car_id] = {
        "current_lap": 1,
        "gap_to_leader": HYPERCAR_COMPETITORS[car_id]["current_gap_sec"],
        "last_pit_lap": 0,
        "tire_age": 0,
        "fuel_level": HYPERCAR_COMPETITORS[car_id]["fuel_tank_liters"],
        "current_issues": [],
        "pit_strategy": "normal",
        "current_driver_index": 0 # Added to track driver for competitor
    }

active_anomalies = {}
command_queue = queue.Queue()
stop_simulation_event = threading.Event()



def calculate_sector_time(sector_num: int, base_lap_time: float, conditions: dict) -> float:
    """Calculate realistic sector time based on track characteristics and conditions"""
    sector_key = f"sector_{sector_num}"
    sector_data = TRACK_SECTORS[sector_key]
    
    # Base sector time (proportional to sector length)
    base_sector_time = base_lap_time * (sector_data["length_km"] / TRACK_LENGTH_KM)
    
    # Weather impact
    weather_modifier = conditions.get("grip_factor", 1.0)
    if weather_modifier < 0.8:  # Wet conditions
        if "straight" in sector_data["characteristics"]:
            weather_modifier *= 0.95  # Less impact on straights
        else:
            weather_modifier *= 0.85  # More impact on corners
    
    # Tire condition impact (using global tire_temperatures and tire_wear)
    tire_temp_avg = sum(tire_temperatures.values()) / 4
    tire_wear_avg = sum(tire_wear.values()) / 4
    
    tire_performance = 1.0
    if tire_temp_avg < 85:  # Cold tires
        tire_performance *= 1.05
    elif tire_temp_avg > 110:  # Overheating
        tire_performance *= 1.08
    
    tire_performance *= (1 + tire_wear_avg * 0.1)  # Wear impact
    
    # Fuel load impact (heavier = slower)
    fuel_impact = 1 + (our_car_fuel - 20) * 0.0008  # ~0.03s per 10L
    
    # Driver fatigue (longer stints = slight degradation)
    driver_fatigue = 1 + min(driver_stint_time / 7200, 0.02)  # Max 2% after 2 hours
    
    # Apply all modifiers
    final_time = base_sector_time * weather_modifier * tire_performance * fuel_impact * driver_fatigue
    
    # Add random variation
    variation = random.uniform(-0.02, 0.02)  # ±2% variation
    final_time *= (1 + variation)
    
    return final_time

def update_weather_system():
    """Enhanced weather system with realistic Le Mans patterns"""
    global current_weather, track_temperature, ambient_temperature, track_grip, visibility
    global wind_speed, wind_direction # Add wind to globals if modified here
    
    # Get current hour for time-based weather patterns
    current_hour = int((total_simulated_time_sec % 86400) / 3600)
    
    # Determine weather change probability
    weather_change_chance = 0.05  # 5% per update
    
    # Apply hourly modifiers
    for hour_range, modifiers in HOURLY_WEATHER_MODIFIERS.items():
        if current_hour in hour_range:
            if "rain" in current_weather:
                weather_change_chance *= 0.3  # Rain tends to persist
            break
    
    if random.random() < weather_change_chance:
        # Select new weather based on probabilities
        weather_roll = random.random()
        cumulative_prob = 0
        
        for weather_type, data in WEATHER_PATTERNS.items():
            cumulative_prob += data["probability"]
            if weather_roll <= cumulative_prob:
                if weather_type != current_weather:
                    print(f"\n🌤️  WEATHER UPDATE: {current_weather.title()} → {weather_type.title()}")
                    current_weather = weather_type
                    track_grip = data["grip_factor"] + random.uniform(-0.05, 0.05)
                    visibility = data["visibility"]
                break
    
    # Update temperatures (realistic daily cycle)
    hour_angle = (current_hour - 14) * math.pi / 12  # Peak at 2 PM
    temp_variation = 8 * math.cos(hour_angle)  # ±8°C variation
    ambient_temperature = 20 + temp_variation + random.uniform(-1, 1)
    track_temperature = ambient_temperature + 8 + random.uniform(-2, 2)
    
    # Update wind (more dynamic)
    wind_speed = random.uniform(5, 25)
    wind_direction = random.randint(0, 359)

def simulate_competitor_behavior():
    """Enhanced competitor simulation with realistic strategies"""
    global competitor_positions, our_car_last_lap_time, current_lap # our_car_last_lap_time read here
    
    for car_id, competitor in HYPERCAR_COMPETITORS.items():
        pos_data = competitor_positions[car_id]
        
        # Calculate competitor lap time with various factors
        base_time = competitor["base_lap_time"]
        
        # Weather impact
        weather_impact = 2.0 - track_grip  # Worse weather = slower
        
        # Tire age impact
        tire_deg_impact = pos_data["tire_age"] * competitor["tire_degradation_rate"]
        
        # Fuel load impact
        fuel_ratio = pos_data["fuel_level"] / competitor["fuel_tank_liters"]
        fuel_impact = fuel_ratio * 3.0  # Heavier = slower
        
        # Driver skill variance (some drivers better in certain conditions)
        driver_variance = random.uniform(-1.5, 1.5)
        # Check if current time is within night hours (20:00 to 06:00)
        current_sim_hour = int((total_simulated_time_sec % 86400) / 3600)
        is_night = (current_sim_hour >= 20 or current_sim_hour < 6)

        if "night_pace" in competitor.get("strengths", []) and is_night:
            driver_variance -= 1.0  # Better at night
        
        # Calculate final lap time
        final_lap_time = base_time + weather_impact + tire_deg_impact + fuel_impact + driver_variance
        
        # Update position relative to leader
        if our_car_last_lap_time > 0:
            time_diff = final_lap_time - our_car_last_lap_time
            pos_data["gap_to_leader"] += time_diff * 0.1  # Gradual gap changes
        
        # Pit stop logic
        if (current_lap - pos_data["last_pit_lap"]) >= competitor["pit_frequency_laps"]:
            if random.random() < 0.15:  # 15% chance to pit this lap
                print(f"🏁 {competitor['name']} pitting on lap {current_lap}")
                pos_data["last_pit_lap"] = current_lap
                pos_data["tire_age"] = 0
                pos_data["fuel_level"] = competitor["fuel_tank_liters"]
                # Simulate pit loss
                pos_data["gap_to_leader"] += 25 + random.uniform(-5, 5)  
                pos_data["in_pit"] = True # Set pit status
        else:
            pos_data["in_pit"] = False # Ensure not in pit if condition not met

        # Update tire age and fuel
        pos_data["tire_age"] += 1
        fuel_consumption = competitor["fuel_consumption_base"] * (1 + random.uniform(-0.1, 0.1))
        pos_data["fuel_level"] = max(0, pos_data["fuel_level"] - fuel_consumption)
        
        # Random reliability issues
        if random.random() < (1 - competitor["reliability_factor"]) / 1000:
            issue_types = ["engine_issue", "tire_problem", "aerodynamic_damage", "electrical_fault"]
            issue = random.choice(issue_types)
            pos_data["current_issues"].append(issue)
            pos_data["gap_to_leader"] += random.uniform(10, 30)
            print(f"⚠️  {competitor['name']} experiencing {issue}")

def generate_enhanced_telemetry():
    """Generate comprehensive and realistic telemetry data"""
    global current_lap, current_lap_time_sec, total_simulated_time_sec, our_car_fuel
    global lap_start_time_simulated, our_car_last_lap_time, current_sector, sector_progress
    global tire_age_laps, driver_stint_time, current_driver, last_pit_lap
    global tire_wear, tire_temperatures, tire_pressures
    global fuel_saving_mode, push_mode, active_anomalies, safety_car_active, yellow_flag_sectors, track_limits_warnings, race_incidents

    

    total_simulated_time_sec += 0.5  # 0.5 second intervals
    current_lap_time_sec = total_simulated_time_sec - lap_start_time_simulated
    driver_stint_time += 0.5
    
    # Update weather and track conditions
    update_weather_system()
    
    # Sector progression
    sector_progress += 0.5
    expected_sector_time = our_car_last_lap_time / 3 if our_car_last_lap_time > 0 else (BASE_LAP_TIME_SEC / 3)
    
    if sector_progress >= expected_sector_time:
        current_sector = (current_sector % 3) + 1
        sector_progress = 0.0
        
        if current_sector == 1:  # Completed a lap (returned to S1)
            current_lap += 1
            lap_start_time_simulated = total_simulated_time_sec
            current_lap_time_sec = 0.0
            tire_age_laps += 1
            
            # Calculate new lap time
            conditions = {"grip_factor": track_grip, "visibility": visibility}
            sector_times = []
            for i in range(1, 4):
                sector_time = calculate_sector_time(i, OUR_CAR_CONFIG["base_lap_time"], conditions)
                sector_times.append(sector_time)
            
            our_car_last_lap_time = sum(sector_times)
            
            # Apply anomaly effects on OUR car's lap time
            for anomaly_id, anomaly_data in active_anomalies.items():
                anomaly_config = ENHANCED_ANOMALIES.get(anomaly_id)
                if anomaly_config and "lap_time_impact" in anomaly_config:
                    our_car_last_lap_time *= anomaly_config["lap_time_impact"]
            
            # Fuel consumption
            base_consumption = OUR_CAR_CONFIG["fuel_consumption_base"]
            
            # Adjust for conditions
            if fuel_saving_mode:
                consumption_modifier = 0.85
            elif push_mode:
                consumption_modifier = 1.15
            else:
                consumption_modifier = 1.0
            
            # Weather and sector impact
            consumption_modifier *= (2.0 - track_grip) * 0.1 + 0.9  # Worse conditions = more fuel
            
            fuel_used = base_consumption * consumption_modifier
            our_car_fuel = max(0, our_car_fuel - fuel_used)
            
            # Tire wear
            for tire in tire_wear: # tire_wear is a global variable
                wear_rate = OUR_CAR_CONFIG["tire_degradation_rate"] / 1000
                if track_grip < 0.8:  # Wet conditions
                    wear_rate *= 0.7  # Less wear in wet
                elif ambient_temperature > 25:  # Hot conditions
                    wear_rate *= 1.3  # More wear when hot
                
                tire_wear[tire] += wear_rate * random.uniform(0.8, 1.2)
                tire_wear[tire] = min(tire_wear[tire], 1.0)
    
    # Pit stop logic
    laps_since_pit = current_lap - last_pit_lap
    should_pit = False
    
    if our_car_fuel < 15 and laps_since_pit > 20:  # Low fuel (arbitrary threshold)
        should_pit = True
        print(f"\n🏁 LOW FUEL PIT STOP - Lap {current_lap}")
    elif tire_age_laps > 45 and any(wear > 0.8 for wear in tire_wear.values()):  # Worn tires
        should_pit = True
        print(f"\n🏁 TIRE CHANGE PIT STOP - Lap {current_lap}")
    elif driver_stint_time > 7200:  # 2 hour driver limit
        should_pit = True
        print(f"\n🏁 DRIVER CHANGE PIT STOP - Lap {current_lap}")
    
    if should_pit:
        # Simulate pit stop
        pit_duration = random.uniform(22, 28)  # 22-28 second pit stop
        total_simulated_time_sec += pit_duration
        
        # Reset systems
        our_car_fuel = OUR_CAR_CONFIG["fuel_tank_liters"]
        tire_wear = {k: 0.0 for k in tire_wear}
        tire_age_laps = 0
        last_pit_lap = current_lap
        
        # Driver change every 2-3 stints
        if driver_stint_time > 7200 or random.random() < 0.3:
            current_driver = (current_driver + 1) % len(OUR_CAR_CONFIG["drivers"])
            driver_stint_time = 0.0
            print(f"👤 Driver change: {OUR_CAR_CONFIG['drivers'][current_driver]} now driving")
    

    simulate_competitor_behavior()
    
    # Current sector characteristics for speed/behavior
    current_sector_name = f"sector_{current_sector}"
    sector_data = TRACK_SECTORS[current_sector_name]
    
    #  realistic speed based on sector
    speed_kmh = 0.0
    throttle_percent = 0.0
    brake_percent = 0.0
    engine_rpm = 0.0

    if "straight" in sector_data["characteristics"]:
        speed_kmh = random.uniform(280, min(340, sector_data["max_speed_kmh"]))
        throttle_percent = random.uniform(85, 100)
        brake_percent = random.uniform(0, 10)
        engine_rpm = random.uniform(8500, 10500)
    elif "chicane" in sector_data["characteristics"]:
        speed_kmh = random.uniform(120, 180)
        throttle_percent = random.uniform(40, 70)
        brake_percent = random.uniform(60, 90)
        engine_rpm = random.uniform(6000, 8000)
    elif "corner" in str(sector_data["characteristics"]): # Handle list of strings
        speed_kmh = random.uniform(150, 220)
        throttle_percent = random.uniform(50, 80)
        brake_percent = random.uniform(20, 60)
        engine_rpm = random.uniform(7000, 9000)
    else:
        # Default mixed section (if characteristics don't fit specific categories)
        speed_kmh = random.uniform(200, 280)
        throttle_percent = random.uniform(60, 90)
        brake_percent = random.uniform(10, 40)
        engine_rpm = random.uniform(7500, 9500)
    
    # Apply weather and grip effects
    if track_grip < 0.8:  # Wet conditions
        speed_kmh *= 0.85
        throttle_percent *= 0.9
        brake_percent *= 1.2  # More braking needed
    
    # Apply visibility effects (night/fog)
    if visibility < 0.8:
        speed_kmh *= 0.92
        throttle_percent *= 0.95
    
    # Apply anomaly effects (on car performance characteristics)
    for anomaly_id in active_anomalies:
        anomaly_config = ENHANCED_ANOMALIES.get(anomaly_id)
        if anomaly_config:
            if anomaly_config["type"] == "engine_overheating":
                engine_rpm *= 0.9
                throttle_percent *= 0.85
            elif anomaly_config["type"] == "aerodynamic_damage":
                speed_kmh *= 0.93
            elif anomaly_config["type"] == "brake_balance_issue":
                brake_percent *= 1.15
            elif anomaly_config["type"] == "night_visibility_issue" and visibility < 0.9:
                speed_kmh *= 0.88
    
    # Update tire temperatures based on current usage
    base_tire_temp_front = 95.0
    base_tire_temp_rear = 90.0

    for tire_pos in ["FL", "FR", "RL", "RR"]:
        current_base_temp = base_tire_temp_front if tire_pos.startswith('F') else base_tire_temp_rear
        
        temp_modifier = 1.0
        if brake_percent > 50:
            temp_modifier += 0.1  # Braking heats tires
        if speed_kmh > 300:
            temp_modifier += 0.05  # High speed heats tires
        if "corner" in str(sector_data["characteristics"]):
            temp_modifier += 0.08  # Cornering heats tires
        
        # Weather cooling effect
        if ambient_temperature < 15:
            temp_modifier *= 0.95
        elif ambient_temperature > 30:
            temp_modifier *= 1.05
        
        # Apply wear heating
        temp_modifier += tire_wear[tire_pos] * 0.2
        
        tire_temperatures[tire_pos] = current_base_temp * temp_modifier + random.uniform(-2, 2)
        
        # Update tire pressures (temperature affects pressure)
        temp_diff = tire_temperatures[tire_pos] - current_base_temp
        pressure_change = temp_diff * 0.01  # ~0.01 bar per degree
        base_pressure = 1.9 if tire_pos.startswith('F') else 1.8
        tire_pressures[tire_pos] = base_pressure + pressure_change + random.uniform(-0.02, 0.02)
    
    # Generate suspension data based on sector and speed
    suspension_travel = {}
    base_travel = 25.0
    if "corner" in str(sector_data["characteristics"]):
        base_travel = 45.0
    elif "straight" in sector_data["characteristics"]:
        base_travel = 15.0
    
    for tire_pos in ["FL", "FR", "RL", "RR"]:
        travel = base_travel + random.uniform(-5, 5)
        # Apply anomaly effects (e.g., specific tire punctures increase travel)
        if f"tire_puncture_{tire_pos.lower()}" in active_anomalies: # Check if specific tire puncture anomaly is active
            travel *= 1.3
        suspension_travel[f"suspension_travel_{tire_pos}_mm"] = travel
    
    # Engine and drivetrain data
    oil_temp_C = 110 + (engine_rpm - 7500) * 0.002 + random.uniform(-3, 3)
    water_temp_C = 85 + (ambient_temperature - 20) * 0.5 + random.uniform(-2, 2)
    
    # Apply overheating anomaly
    if "engine_overheating" in active_anomalies:
        oil_temp_C += 15
        water_temp_C += 12
    
    # Generate hybrid system data (for cars that have it)
    hybrid_battery_percent = random.uniform(70, 100)
    hybrid_power_kw = 0
    if throttle_percent > 70 and hybrid_battery_percent > 20:
        hybrid_power_kw = random.uniform(120, 160)  # Typical hybrid boost
        hybrid_battery_percent -= 0.5  # Battery drain
    elif throttle_percent < 30:  # Regeneration under braking
        hybrid_battery_percent = min(100, hybrid_battery_percent + 0.3)
    
    # Generate competitor data
    competitor_data = []
    for car_id, competitor in HYPERCAR_COMPETITORS.items():
        pos_data = competitor_positions[car_id]
        
        # Calculate current position based on gap
        gap_seconds = pos_data["gap_to_leader"]
        
        # Estimate laps based on gap (rough calculation)
        estimated_lap_diff = gap_seconds / OUR_CAR_CONFIG["base_lap_time"] if OUR_CAR_CONFIG["base_lap_time"] > 0 else 0
        
        competitor_data.append({
            "car_number": car_id,
            "name": competitor["name"],
            "manufacturer": competitor["manufacturer"],
            "drivers": competitor["drivers"],
            "current_driver": competitor["drivers"][pos_data.get("current_driver_index", 0)], # Default to 0 if not tracked
            "gap_to_leader_sec": round(gap_seconds, 2),
            "gap_status": "ahead" if gap_seconds < 0 else ("behind" if gap_seconds > 0 else "same_lap"),
            "current_lap": pos_data["current_lap"],
            "last_lap_time_sec": round(competitor["base_lap_time"] + random.uniform(-2, 2), 2),
            "tire_age_laps": pos_data["tire_age"],
            "fuel_level_liters": round(pos_data["fuel_level"], 1),
            "pit_status": "in_pit" if pos_data.get("in_pit", False) else "on_track",
            "last_pit_lap": pos_data["last_pit_lap"],
            "current_issues": pos_data.get("current_issues", []),
            "pit_strategy": pos_data.get("pit_strategy", "normal"),
            "sector_time_1": round(competitor["base_lap_time"] * 0.35 + random.uniform(-1, 1), 2),
            "sector_time_2": round(competitor["base_lap_time"] * 0.30 + random.uniform(-1, 1), 2),
            "sector_time_3": round(competitor["base_lap_time"] * 0.35 + random.uniform(-1, 1), 2)
        })
    
    # Sort competitors by gap (leaders first)
    competitor_data.sort(key=lambda x: x["gap_to_leader_sec"])
    
    # Add position numbers
    for i, comp in enumerate(competitor_data):
        comp["current_position"] = i + 1
    
    # Race control and timing data
    current_hour = int((total_simulated_time_sec % 86400) / 3600)
    # Calculate time remaining from race end (assuming 24h total race duration)
    race_time_remaining = max(0, 24 * 3600 - total_simulated_time_sec)  
    
    # Calculate various race metrics
    average_lap_time = our_car_last_lap_time # Simplified, ideally average of recent laps
    fuel_laps_remaining = max(0, our_car_fuel / (OUR_CAR_CONFIG["fuel_consumption_base"] * 1.1)) if (OUR_CAR_CONFIG["fuel_consumption_base"] * 1.1) > 0 else 0
    estimated_pit_window = max(0, OUR_CAR_CONFIG["pit_frequency_laps"] - (current_lap - last_pit_lap))
    
    # Track limits and penalties
    track_limits_risk = 0.0
    if speed_kmh > sector_data["max_speed_kmh"] * 0.95:
        track_limits_risk = random.uniform(0.1, 0.3)
    
    # Determine is_night based on Le Mans sunrise/sunset hours
    is_night = (current_hour >= 21 or current_hour < 6) # Roughly 9 PM to 6 AM

    # Generate comprehensive telemetry data
    telemetry_data = {
        "timestamp_simulated_sec": round(total_simulated_time_sec, 2),
        "race_info": {
            "current_hour": current_hour,
            "race_time_elapsed_sec": round(total_simulated_time_sec, 2),
            "race_time_remaining_sec": round(race_time_remaining, 2),
            "time_of_day": f"{current_hour:02d}:{int((total_simulated_time_sec % 3600) / 60):02d}",
            "is_night": is_night
        },
        "car": {
            "name": OUR_CAR_CONFIG["name"],
            "class": OUR_CAR_CONFIG["class"],
            "current_driver": OUR_CAR_CONFIG["drivers"][current_driver],
            "driver_stint_time_sec": round(driver_stint_time, 2),
            "lap_number": current_lap,
            "current_lap_time_sec": round(current_lap_time_sec, 2),
            "last_lap_time_sec": round(our_car_last_lap_time, 2),
            "average_lap_time_sec": round(average_lap_time, 2),
            "current_sector": current_sector,
            "sector_progress_percent": round((sector_progress / expected_sector_time) * 100, 1) if expected_sector_time > 0 else 0,
            "current_track_segment": current_sector_name,
            "speed_kmh": round(speed_kmh, 2),
            "engine_rpm": round(engine_rpm, 2),
            "throttle_percent": round(throttle_percent, 2),
            "brake_percent": round(brake_percent, 2),
            "gear": max(1, min(8, int(speed_kmh / 45))),  # Rough gear calculation
            "drs_active": speed_kmh > 250 and "straight" in sector_data["characteristics"],
            "fuel_level_liters": round(our_car_fuel, 2),
            "fuel_consumption_current_L_per_lap": round(OUR_CAR_CONFIG["fuel_consumption_base"] * (1 + random.uniform(-0.05, 0.05)), 2), # More dynamic consumption
            "fuel_laps_remaining": round(fuel_laps_remaining, 1),
            "fuel_saving_mode": fuel_saving_mode,
            "push_mode": push_mode,
            "oil_temp_C": round(oil_temp_C, 2),
            "water_temp_C": round(water_temp_C, 2),
            "hybrid_battery_percent": round(hybrid_battery_percent, 1),
            "hybrid_power_output_kw": round(hybrid_power_kw, 1),
            "tire_compound": tire_compound,
            "tire_age_laps": tire_age_laps,
            # Explicitly list tire keys to avoid double-suffix
            "tire_temp_FL_C": round(tire_temperatures["FL"], 2),
            "tire_temp_FR_C": round(tire_temperatures["FR"], 2),
            "tire_temp_RL_C": round(tire_temperatures["RL"], 2),
            "tire_temp_RR_C": round(tire_temperatures["RR"], 2),
            "tire_pressure_FL_bar": round(tire_pressures["FL"], 3),
            "tire_pressure_FR_bar": round(tire_pressures["FR"], 3),
            "tire_pressure_RL_bar": round(tire_pressures["RL"], 3),
            "tire_pressure_RR_bar": round(tire_pressures["RR"], 3),
            "tire_wear_FL_percent": round(tire_wear["FL"] * 100, 1),
            "tire_wear_FR_percent": round(tire_wear["FR"] * 100, 1),
            "tire_wear_RL_percent": round(tire_wear["RL"] * 100, 1),
            "tire_wear_RR_percent": round(tire_wear["RR"] * 100, 1),
            "suspension_travel_FL_mm": round(suspension_travel["suspension_travel_FL_mm"], 2),
            "suspension_travel_FR_mm": round(suspension_travel["suspension_travel_FR_mm"], 2),
            "suspension_travel_RL_mm": round(suspension_travel["suspension_travel_RL_mm"], 2),
            "suspension_travel_RR_mm": round(suspension_travel["suspension_travel_RR_mm"], 2),
            "last_pit_lap": last_pit_lap,
            "laps_since_pit": current_lap - last_pit_lap,
            "estimated_pit_window": estimated_pit_window,
            "pit_strategy": pit_strategy,
            "track_limits_warnings": track_limits_warnings,
            "track_limits_risk_percent": round(track_limits_risk * 100, 1)
        },
        "environmental": {
            "current_weather": current_weather,
            "ambient_temp_C": round(ambient_temperature, 2),
            "track_temp_C": round(track_temperature, 2),
            "humidity_percent": random.randint(45, 85),
            "wind_speed_kmh": round(wind_speed, 2),
            "wind_direction_deg": wind_direction,
            "track_grip_level": round(track_grip, 3),
            "visibility_level": round(visibility, 3),
            "sunrise_time": "06:30", # Static for now
            "sunset_time": "21:45" # Static for now
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
            "safety_car_active": safety_car_active,
            "yellow_flag_sectors": yellow_flag_sectors,
            "track_status": "green" if not safety_car_active and not yellow_flag_sectors else "caution"
        },
        "competitors": competitor_data,
        "active_anomalies": [ # List of active anomaly descriptions
            {
                "type": ENHANCED_ANOMALIES[anomaly_id]["type"],
                "message": ENHANCED_ANOMALIES[anomaly_id]["message"],
                "severity": ENHANCED_ANOMALIES[anomaly_id]["severity"],
                "duration_remaining_sec": round(max(0, anomaly_data["end_time"] - total_simulated_time_sec), 2) if anomaly_data.get("end_time", 0) > 0 else "permanent"
            }
            for anomaly_id, anomaly_data in active_anomalies.items()
        ],
        "strategy": {
            "current_strategy": pit_strategy,
            "fuel_target_laps": round(fuel_laps_remaining, 1),
            "tire_change_recommended": any(wear > 0.7 for wear in tire_wear.values()),
            "driver_change_due": driver_stint_time > 7000,  # Nearly 2 hours
            "next_pit_recommendation": estimated_pit_window,
            "position_in_class": 1,  # Simplified for LMP2 class (assumes we are leader in class)
            "gap_to_class_leader": 0.0,
            "laps_to_class_leader": 0
        }
    }
    
    return telemetry_data


def update_anomalies():
    """Update and expire active anomalies"""
    global active_anomalies, track_grip, current_weather, safety_car_active
    
    expired_anomalies = []
    for anomaly_type_id, anomaly_data in active_anomalies.items(): # Iterating through IDs (e.g., "1", "2")
        if anomaly_data.get("end_time", 0) > 0 and total_simulated_time_sec >= anomaly_data["end_time"]:
            expired_anomalies.append(anomaly_type_id) # Collect the ID
    
    for anomaly_type_id in expired_anomalies:
        anomaly_config = ENHANCED_ANOMALIES.get(anomaly_type_id, {})
        print(f"\n✅ ANOMALY RESOLVED: {anomaly_config.get('type', anomaly_type_id).upper()}")
        del active_anomalies[anomaly_type_id]
        
        # Reset weather-related effects
        if anomaly_config["type"] == "sudden_weather_change":
            current_weather = "clear"
            track_grip = 1.0
        elif anomaly_config["type"] == "safety_car_period":
            safety_car_active = False


def activate_anomaly(anomaly_id: str, duration: int, message: str):
    """Activate a specific anomaly with effects"""
    global active_anomalies, current_weather, track_grip, safety_car_active
    
    anomaly_config = ENHANCED_ANOMALIES.get(anomaly_id)
    if not anomaly_config:
        print(f"❌ ERROR: Anomaly ID {anomaly_id} not found in ENHANCED_ANOMALIES.")
        return

    if anomaly_id == "0": # "reset_all_systems"
        active_anomalies.clear()
        current_weather = "clear"
        track_grip = 1.0
        safety_car_active = False
        print(f"\n🔄 SYSTEMS RESET: {message}")
        return
    
    print(f"\n⚠️  ANOMALY ACTIVATED: {anomaly_config['type'].upper()}")
    print(f"📋 {message}")
    print(f"🔧 Severity: {anomaly_config['severity']}")
    
    end_time = total_simulated_time_sec + duration if duration > 0 else 0
    active_anomalies[anomaly_id] = { # Store using the ID as key
        "start_time": total_simulated_time_sec,
        "end_time": end_time,
        "message": message
    }
    
    # Apply immediate effects
    if anomaly_config["type"] == "sudden_weather_change":
        current_weather = "heavy_rain"
        track_grip = 0.6
    elif anomaly_config["type"] == "safety_car_period":
        safety_car_active = True


def input_listener():
    """Enhanced input listener with better UI"""
    print("\n" + "="*50)
    print("🏁 ENHANCED LE MANS ANOMALY CONTROL SYSTEM")
    print("="*50)
    print("Available anomaly triggers:")
    # Filter out the "reset" anomaly from the main list for display
    display_anomalies = {k:v for k,v in ENHANCED_ANOMALIES.items() if k != "0"} 
    for key, anomaly in display_anomalies.items():
        severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "info": "🔵"}.get(anomaly["severity"], "⚪")
        print(f"  {key}: {severity_emoji} {anomaly['type']}")
        print(f"     └─ {anomaly['message'][:60]}...")
    print("\n💡 Commands: Enter number (1-10) to trigger, '0' to reset, 'q' to quit")
    print("="*50 + "\n")
    
    while not stop_simulation_event.is_set():
        try:
            cmd = input("🎮 Enter command: ").strip().lower()
            if cmd in ['q', 'quit', 'exit']:
                command_queue.put('q')
                break
            elif cmd in ENHANCED_ANOMALIES: # Check against full ENHANCED_ANOMALIES
                command_queue.put(cmd)
            else:
                print("❌ Invalid command. Use numbers 0-10 or 'q' to quit.")
        except (EOFError, KeyboardInterrupt):
            command_queue.put('q')
            break


def enhanced_telemetry_generator(start_hour=14, total_race_hours=24):
    """Main enhanced telemetry generation loop"""
    global total_simulated_time_sec, lap_start_time_simulated, current_driver
    
    # Initialize simulation
    total_simulated_time_sec = start_hour * 3600
    lap_start_time_simulated = total_simulated_time_sec
    
    max_simulated_time_sec = total_race_hours * 3600
    update_interval = 0.5  # Generate data every 0.5 simulated seconds
    real_time_factor = 30  # 30x speed (30 sim seconds per real second)
    
    print(f"\n🏁 STARTING ENHANCED LE MANS 24H SIMULATION")
    print(f"⏰ Starting time: {start_hour:02d}:00 ({start_hour}h into race)")
    print(f"🚗 Car: {OUR_CAR_CONFIG['name']}")
    print(f"👤 Driver: {OUR_CAR_CONFIG['drivers'][current_driver]}")
    print(f"⚡ Speed: {real_time_factor}x real-time")
    print(f"📡 Backend: {BACKEND_API_URL}")
    print("-" * 60)
    
    last_lap_print = 0
    
    while total_simulated_time_sec < max_simulated_time_sec and not stop_simulation_event.is_set():
        # Process commands
        while not command_queue.empty():
            cmd = command_queue.get()
            if cmd in ['q', 'quit']:
                print("\n🛑 Simulation stopped by user")
                stop_simulation_event.set()
                return
            elif cmd in ENHANCED_ANOMALIES:
                anomaly = ENHANCED_ANOMALIES[cmd]
                activate_anomaly(cmd, anomaly["duration_sec"], anomaly["message"])
        
        # Update anomalies
        update_anomalies()
        
        # Generate telemetry
        try:
            telemetry = generate_enhanced_telemetry()
            
            # Send to backend
            try:
                response = requests.post(BACKEND_API_URL, json=telemetry, timeout=1.0)
                if response.status_code == 200:
                    # Print periodic updates
                    if current_lap > last_lap_print:
                        hour = int((total_simulated_time_sec % 86400) / 3600)
                        minute = int((total_simulated_time_sec % 3600) / 60)
                        print(f"📊 Lap {current_lap:3d} | {hour:02d}:{minute:02d} | "
                              f"⏱️  {our_car_last_lap_time:.2f}s | "
                              f"⛽ {our_car_fuel:.1f}L | "
                              f"🌡️  {tire_temperatures['FL']:.0f}°C | "
                              f"🌤️  {current_weather.title()}")
                        last_lap_print = current_lap
                else:
                    print(f"❌ Backend error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                print("⚠️  Backend connection lost - continuing simulation")
            except requests.exceptions.Timeout:
                print("⚠️  Backend timeout - continuing simulation")
            except Exception as e:
                print(f"❌ Unexpected error sending telemetry: {e}") # Clarified error message
        
        except Exception as e:
            print(f"💥 Telemetry generation error: {e}")
        
        # Sleep for real-time factor
        time.sleep(update_interval / real_time_factor)
    
    print(f"\n🏁 SIMULATION COMPLETE!")
    print(f"📊 Total laps completed: {current_lap}")
    print(f"⏰ Race time: {int(total_simulated_time_sec / 3600)}h {int((total_simulated_time_sec % 3600) / 60)}m")


if __name__ == "__main__":
    # Start input listener thread
    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()
    
    try:
        # Start at 18h into the race as per the initial problem statement, or 14h as you had it
        enhanced_telemetry_generator(start_hour=18)  
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted (Ctrl+C)")
    finally:
        stop_simulation_event.set()
        input_thread.join(timeout=2.0)
        print("👋 Enhanced Le Mans simulator stopped gracefully")