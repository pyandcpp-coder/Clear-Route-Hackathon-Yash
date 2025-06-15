from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import operator
import re 
import statistics
import time
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass 
from dotenv import load_dotenv
import traceback


from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START 
from langgraph.prebuilt import ToolNode 

# Load environment variables
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RaceBrain AI Service",
    description="Provides strategic recommendations and anomaly detection using LangGraph agents."
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for hackathon simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️ WARNING: GROQ_API_KEY not found in environment variables. Using dummy key!")
    GROQ_API_KEY = "dummy_key_for_testing"

LLM_MODEL_NAME = "llama-3.3-70b-versatile" # Sticking with Llama3-70B for now as it's a known good model on Groq
llm_temperature = 0.7 # Good balance for generation and consistency

# --- Custom Tools for LangChain ---
import httpx

BACKEND_API_BASE_URL = "http://localhost:8000" # Your telemetry backend

async def _get_latest_telemetry_internal() -> dict: 
    """Enhanced with better error handling and fallback data"""
    url = f"{BACKEND_API_BASE_URL}/live_data"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            response.raise_for_status() # Raises an exception for 4xx/5xx responses
            data = response.json()
            return data
    except httpx.ConnectError:
        print(f"🔌 Connection Error: Backend API not available at {url}")
        return _get_mock_telemetry_data()
    except httpx.TimeoutException:
        print(f"⏰ Timeout Error: Backend API took too long to respond")
        return _get_mock_telemetry_data()
    except httpx.RequestError as exc:
        print(f"🚫 Request Error: {exc}")
        return _get_mock_telemetry_data()
    except json.JSONDecodeError:
        print(f"📄 JSON Decode Error: Invalid response format")
        return _get_mock_telemetry_data()
    except Exception as e:
        print(f"❌ Unexpected Error in _get_latest_telemetry_internal: {e}")
        traceback.print_exc() 
        return _get_mock_telemetry_data()

async def _get_telemetry_history_internal(limit: int = 50) -> list[dict]: 
    """Enhanced with better error handling and fallback data"""
    url = f"{BACKEND_API_BASE_URL}/telemetry_history?limit={limit}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data
    except Exception as e:
        print(f"❌ Error fetching history in _get_telemetry_history_internal: {e}")
        traceback.print_exc()
        return _get_mock_telemetry_history(limit)

def _get_mock_telemetry_data() -> dict:
    """Fallback mock data when backend is unavailable or fails to provide data"""
    print("🔄 Using mock telemetry data (backend unavailable or error)")
    mock_time = time.time() 
    return {
        "timestamp_simulated_sec": mock_time,
        "race_info": {
            "current_hour": int((mock_time % 86400) / 3600),
            "time_of_day": datetime.fromtimestamp(mock_time).strftime("%H:%M"),
            "is_night": False,
            "race_time_elapsed_sec": mock_time, 
            "race_time_remaining_sec": 24 * 3600 - mock_time 
        },
        "car": {
            "lap_number": 15,
            "current_lap_time_sec": 50.0, 
            "last_lap_time_sec": 215.8 + random.uniform(-5,5), 
            "fuel_level_liters": 45.2 + random.uniform(-2,2),
            "fuel_consumption_current_L_per_lap": 2.9 + random.uniform(-0.1, 0.1),
            "tire_temp_FL_C": 98.5 + random.uniform(-2,2),
            "tire_temp_FR_C": 97.8 + random.uniform(-2,2),
            "tire_temp_RL_C": 101.2 + random.uniform(-2,2),
            "tire_temp_RR_C": 100.4 + random.uniform(-2,2),
            "tire_pressure_FL_bar": 1.85 + random.uniform(-0.02,0.02),
            "tire_pressure_FR_bar": 1.87 + random.uniform(-0.02,0.02),
            "tire_pressure_RL_bar": 1.83 + random.uniform(-0.02,0.02),
            "tire_pressure_RR_bar": 1.86 + random.uniform(-0.02,0.02),
            "oil_temp_C": 108.5 + random.uniform(-2,2),
            "water_temp_C": 87.2 + random.uniform(-2,2),
            "speed_kmh": 250 + random.uniform(-10,10), 
            "current_driver": "Mock Driver", 
            "current_track_segment": "straight",
            "tire_wear_FL_percent": 20.0 + random.uniform(-5,5), 
            "tire_wear_FR_percent": 20.0 + random.uniform(-5,5), 
            "tire_wear_RL_percent": 15.0 + random.uniform(-5,5), 
            "tire_wear_RR_percent": 15.0 + random.uniform(-5,5), 
            "suspension_travel_FL_mm": 30.0 + random.uniform(-5,5), 
            "suspension_travel_FR_mm": 30.0 + random.uniform(-5,5), 
            "suspension_travel_RL_mm": 30.0 + random.uniform(-5,5), 
            "suspension_travel_RR_mm": 30.0 + random.uniform(-5,5), 
            "track_limits_warnings": 0, 
            "track_limits_risk_percent": 0.0 
        },
        "environmental": {
            "current_weather": "clear",
            "rain_intensity": 0,
            "track_grip_level": 1.0,
            "ambient_temp_C": 24.5,
            "track_temp_C": 42.1,
            "visibility_level": 1.0,
            "humidity_percent": 60, 
            "wind_speed_kmh": 15.0, 
            "wind_direction_deg": 180 
        },
        "competitors": [
            {"car_number": "M2", "name": "Mock Car #2", "gap_to_leader_sec": 12.4 + random.uniform(-1,1), "pit_status": "on_track", "current_position": 2, 
             "last_lap_time_sec": 210.0, "fuel_level_liters": 40.0, "tire_age_laps": 10, "current_driver": "C2 Driver", "current_issues": [], "pit_strategy": "normal",
             "sector_time_1": 70, "sector_time_2": 70, "sector_time_3": 70}, 
            {"car_number": "M3", "name": "Mock Car #3", "gap_to_leader_sec": 28.7 + random.uniform(-1,1), "pit_status": "on_track", "current_position": 3,
             "last_lap_time_sec": 211.0, "fuel_level_liters": 35.0, "tire_age_laps": 12, "current_driver": "C3 Driver", "current_issues": [], "pit_strategy": "normal",
             "sector_time_1": 71, "sector_time_2": 71, "sector_time_3": 71},
            {"car_number": "M4", "name": "Mock Car #4", "gap_to_leader_sec": 45.1 + random.uniform(-1,1), "pit_status": "on_track", "current_position": 4,
             "last_lap_time_sec": 212.0, "fuel_level_liters": 30.0, "tire_age_laps": 15, "current_driver": "C4 Driver", "current_issues": [], "pit_strategy": "normal",
             "sector_time_1": 72, "sector_time_2": 72, "sector_time_3": 72}
        ],
        "active_anomalies": [],
        "strategy": { 
            "current_strategy": "normal",
            "fuel_target_laps": 20,
            "tire_change_recommended": False,
            "driver_change_due": False,
            "next_pit_recommendation": 5,
            "position_in_class": 1,
            "gap_to_class_leader": 0.0,
            "laps_to_class_leader": 0
        },
        "track_info": { 
            "sector_1_length_km": 4.2,
            "sector_2_length_km": 6.0,
            "sector_3_length_km": 3.426,
            "total_length_km": 13.626,
            "current_sector_characteristics": ["straight"],
            "current_sector_max_speed": 340,
            "elevation_change_current": 0
        },
        "race_control": { 
            "safety_car_active": False,
            "yellow_flag_sectors": [],
            "track_status": "green"
        }
    }

def _get_mock_telemetry_history(limit: int) -> list[dict]:
    """Generate mock historical data"""
    history = []
    base_timestamp = time.time() - (limit * 15) 
    for i in range(limit):
        mock_data_point = _get_mock_telemetry_data()
        mock_data_point['timestamp_simulated_sec'] = base_timestamp + (i * 15) 
        
        mock_data_point['car']['fuel_level_liters'] = max(0, 45.2 - (i * 0.5))
        mock_data_point['car']['last_lap_time_sec'] = 215.8 + (i * 0.2)
        
        history.append(mock_data_point)
    return history

@tool
async def get_latest_telemetry_tool() -> dict:
    """
    Fetches the latest single telemetry data point from the backend API.
    This tool provides a snapshot of the current car, environmental,
    and competitor conditions.
    """
    return await _get_latest_telemetry_internal()

@tool
async def get_telemetry_history_tool(limit: int = 50) -> list[dict]:
    """
    Fetches a list of recent telemetry data points from the backend history.
    This tool is useful for analyzing trends over time (e.g., for anomaly detection).

    Args:
        limit (int): The maximum number of historical data points to retrieve.
                     Defaults to 50.
    Returns:
        list[dict]: A list of telemetry data points, oldest first.
    """
    return await _get_telemetry_history_internal(limit)

@tool
async def calculate_performance_metrics(telemetry_data: list) -> dict:
    """
    Calculates key performance metrics from telemetry history.
    
    Args:
        telemetry_data: List of telemetry data points
    
    Returns:
        Dict containing calculated metrics
    """
    if not telemetry_data:
        print("⚠️ No telemetry data provided for metrics calculation")
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
        print(f"❌ Error calculating metrics in calculate_performance_metrics: {e}")
        traceback.print_exc()
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
    from langchain_groq import ChatGroq
    if GROQ_API_KEY != "dummy_key_for_testing":
        llm = ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL_NAME, temperature=llm_temperature)
        llm_with_tools = llm.bind_tools(tools)
        print(f"✅ Groq LLM ({LLM_MODEL_NAME}) initialized successfully")
    else:
        raise Exception("No valid API key provided for Groq.")
except Exception as e:
    print(f"⚠️ Could not initialize Groq LLM: {e}")
    print("🔄 AI analysis will use rule-based fallback if LLM is unavailable.")


# --- Enhanced Analysis Functions ---
class TelemetryAnalyzer:
    def __init__(self):
        self.thresholds = TelemetryThresholds()
        self.race_context_defaults = RaceContext() 
    
    def analyze_tire_condition(self, latest_data: dict, history: list) -> dict:
        car_data = latest_data.get("car", {})
        
        tire_analysis = {
            "status": "normal", 
            "issues": [],
            "recommendations": []
        }
        
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
        current_lap = car_data.get("lap_number", 1)
        
        remaining_laps = (current_fuel - self.race_context_defaults.safety_fuel_margin) / consumption_rate if consumption_rate > 0 else float('inf')
        
        laps_since_pit = current_lap % self.race_context_defaults.pit_window_laps
        laps_to_pit = self.race_context_defaults.pit_window_laps - laps_since_pit
        
        fuel_analysis = {
            "current_fuel": current_fuel,
            "consumption_rate": consumption_rate,
            "remaining_laps": remaining_laps,
            "laps_to_pit_window": laps_to_pit,
            "status": "normal",
            "recommendations": []
        }
        
        if remaining_laps < laps_to_pit:
            fuel_analysis["status"] = "critical"
            fuel_analysis["recommendations"].append(f"Fuel shortage risk: only {remaining_laps:.1f} laps remaining")
        elif remaining_laps < laps_to_pit + 5:
            fuel_analysis["status"] = "warning"
            fuel_analysis["recommendations"].append("Consider pit strategy adjustment")
        
        if consumption_rate > self.race_context_defaults.base_fuel_consumption * (1 + self.thresholds.fuel_consumption_excess_percent/100):
            fuel_analysis["status"] = "warning" if fuel_analysis["status"] == "normal" else fuel_analysis["status"]
            fuel_analysis["recommendations"].append(f"High fuel consumption detected: {consumption_rate:.2f} L/lap")
        
        return fuel_analysis

def generate_fallback_strategy(user_query: str, latest_telemetry: dict, anomaly_report: dict, performance_metrics: dict) -> str:
    """Generate strategy when LLM is not available or fails"""
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

# --- Node Functions for LangGraph ---
async def fetch_and_analyze_data_node(state: RaceBrainState):
    print("\n--- Node: Fetching and Analyzing Data ---")
    
    try:
        latest_data = await get_latest_telemetry_tool.ainvoke({}) 
        history_data = await get_telemetry_history_tool.ainvoke({"limit": 10}) 
        metrics = await calculate_performance_metrics.ainvoke({"telemetry_data": history_data})
        
        return {
            "latest_telemetry": latest_data,
            "telemetry_history": history_data,
            "performance_metrics": metrics,
            "messages": state["messages"] 
        }
    except Exception as e:
        print(f"❌ Error in fetch_and_analyze_data_node: {e}")
        traceback.print_exc()
        return {
            "latest_telemetry": _get_mock_telemetry_data(),
            "telemetry_history": _get_mock_telemetry_history(10),
            "performance_metrics": {"error": str(e), "data_points_analyzed": 0,
                                    "lap_time_stats": {}, "fuel_efficiency": {}, "tire_temp_analysis": {}}, 
            "messages": state["messages"]
        }

async def enhanced_anomaly_detection_node(state: RaceBrainState):
    print("\n--- Node: Enhanced Anomaly Detection ---")
    
    try:
        latest_telemetry = state["latest_telemetry"]
        telemetry_history = state["telemetry_history"]
        performance_metrics = state["performance_metrics"]
        
        analyzer = TelemetryAnalyzer()
        tire_analysis = analyzer.analyze_tire_condition(latest_telemetry, telemetry_history)
        fuel_analysis = analyzer.analyze_fuel_strategy(latest_telemetry, telemetry_history)
        
        # Prepare comprehensive data for LLM
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

        if llm is None: # Fallback if LLM not initialized
            print("⚠️ LLM not available for anomaly detection, generating rule-based anomaly report.")
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
            else: # No anomaly detected by rules
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
        else: # LLM is available
            response = await llm.ainvoke(anomaly_prompt.format_messages(
                analysis_context_json_str=json.dumps(analysis_context, indent=2) 
            ))
            response_text = response.content.strip()

            json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL)
            if json_match:
                try:
                    anomaly_data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    print(f"Warning: Extracted string was not valid JSON after regex. Original: {response_text[:200]}")
                    anomaly_data = {
                        "priority_level": "LOW",
                        "primary_anomaly": {"type": "parsing_error", "confidence": "LOW", "affected_component": "N/A", "current_values": {}, "deviation_from_normal": "N/A"},
                        "secondary_issues": [],
                        "trend_analysis": {"lap_time_trend": "UNKNOWN", "tire_condition_trend": "UNKNOWN", "fuel_efficiency_trend": "UNKNOWN"},
                        "immediate_actions": [],
                        "predictive_alerts": [],
                        "reasoning": f"Failed to parse LLM response after extraction: {response_text[:200]}..."
                    }
            else: # No JSON block found, try to parse the whole output
                try:
                    anomaly_data = json.loads(response_text)
                except json.JSONDecodeError:
                    print(f"Warning: AnomalyDetectionAgent did not output pure JSON. Original: {response_text[:200]}")
                    anomaly_data = {
                        "priority_level": "LOW",
                        "primary_anomaly": {"type": "format_error", "confidence": "LOW", "affected_component": "N/A", "current_values": {}, "deviation_from_normal": "N/A"},
                        "secondary_issues": [],
                        "trend_analysis": {"lap_time_trend": "UNKNOWN", "tire_condition_trend": "UNKNOWN", "fuel_efficiency_trend": "UNKNOWN"},
                        "immediate_actions": [],
                        "predictive_alerts": [],
                        "reasoning": f"LLM output was not pure JSON: {response_text[:200]}..."
                    }
        
        print(f"🔍 Anomaly Detection Results: {anomaly_data.get('priority_level', 'UNKNOWN')} priority")
        
        return {
            "anomaly_report": anomaly_data,
            "messages": state["messages"] + [response]
        }
        
    except Exception as e:
        print(f"❌ Error in enhanced_anomaly_detection_node: {e}")
        traceback.print_exc()
        return {
            "anomaly_report": {
                "priority_level": "ERROR",
                "primary_anomaly": {"type": "node_failure", "confidence": "HIGH", "affected_component": "Anomaly Detection Node", "current_values": {}, "deviation_from_normal": ""},
                "secondary_issues": [], "trend_analysis": {"lap_time_trend": "UNKNOWN", "tire_condition_trend": "UNKNOWN", "fuel_efficiency_trend": "UNKNOWN"},
                "immediate_actions": ["Check AI service logs"], "predictive_alerts": [],
                "reasoning": f"Anomaly detection node failed: {str(e)}"
            },
            "messages": state["messages"] # Preserve original messages even on error
        }

# --- GLOBAL HACK FOR DEBUGGING STRATEGY OUTPUT ---
# This variable will temporarily store the raw strategy output if LangGraph state propagation fails
_global_strategy_cache: Optional[str] = None
# --- END GLOBAL HACK ---


async def strategic_decision_node(state: RaceBrainState):
    global _global_strategy_cache
    print("\n--- Node: Strategic Decision Making ---")
    
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
        
        # Format the prompt with actual data
        if llm is None: # Fallback if LLM not available
            raw_llm_response_content = generate_fallback_strategy(user_query, latest_telemetry, anomaly_report, performance_metrics)
            response = AIMessage(content=raw_llm_response_content)
        else: # LLM is available
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
            raw_llm_response_content = llm_response.content.strip() # Get the raw content from LLM
            response = llm_response # Keep original response object for messages
            print("✅ LLM strategy generated successfully")
        
        extracted_strategy = raw_llm_response_content # This will contain the full raw LLM output, including preamble and structured part

        # --- GLOBAL HACK: Store strategy in a global variable ---
        _global_strategy_cache = extracted_strategy
        # --- END GLOBAL HACK ---


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

        print(f"StrategyDecisionAgent Raw Response: {extracted_strategy}") # Print the extracted strategy
        
        return {
            "strategy_recommendation": extracted_strategy, # This should now contain the full raw LLM output
            "confidence_score": confidence_score,
            "priority_actions": priority_actions,
            "messages": state["messages"] + [response]
        }
        
    except Exception as e:
        print(f"❌ Error in strategic_decision_node: {e}")
        traceback.print_exc()
        fallback_strategy = f"Strategy generation encountered an error: {str(e)}. Please check system status and try again."
        
        # --- GLOBAL HACK: Store fallback strategy in global variable ---
        _global_strategy_cache = fallback_strategy
        # --- END GLOBAL HACK ---

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

# Global counter for query numbering
query_counter = 0 

# --- API Endpoint to query the AI ---
@app.post("/query_race_brain_ai")
async def query_ai_endpoint(user_input_json: Dict[str, str]): 
    global query_counter 
    query_counter += 1 
    
    user_input = user_input_json.get("user_input", "") 
    if not user_input:
        raise HTTPException(status_code=400, detail="'user_input' field is required in the request body.")

    print(f"\n--- API Call ({query_counter}): Received query '{user_input}' ---")
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "latest_telemetry": {},
        "telemetry_history": [],
        "performance_metrics": {},
        "anomaly_report": {},
        "strategy_recommendation": "",
        "confidence_score": 0.0,
        "priority_actions": [],
        "query_number": query_counter 
    }

    start_processing_time = time.time() 
    try:
        # --- CRITICAL FIX: Run the graph and let it update the global cache ---
        await app_graph.ainvoke(initial_state) # Just run it, the node will update the global cache
        
        # --- CRITICAL FIX: Retrieve from the global cache ---
        global _global_strategy_cache
        final_strategy_text = _global_strategy_cache if _global_strategy_cache is not None else "AI could not generate a recommendation."
        _global_strategy_cache = None # Clear cache after use
        # --- END CRITICAL FIX ---
            
    except Exception as e:
        print(f"❌ ERROR processing AI query: {e}")
        traceback.print_exc() 
        # In case of full graph crash, provide a generic error strategy
        final_strategy_text = f"AI processing error: {str(e)}. Please check AI service logs."
        
    processing_time_val = round(time.time() - start_processing_time, 2)

    # For confidence and priority actions, we'll have to make assumptions or fetch from the graph's internal state directly
    # This is where the global cache is a hack. A cleaner way would be to correctly return the final_state.
    # But for a hackathon, let's just make sure the strategy text is there.
    # For now, let's make dummy confidence/actions for the payload.
    # If the _global_strategy_cache has content, assume high confidence, else low.
    confidence_for_payload = 0.9 if final_strategy_text != "AI could not generate a recommendation." else 0.0
    priority_actions_for_payload = ["Monitor main strategy display"] if final_strategy_text != "AI could not generate a recommendation." else []
    
    response_payload = {
        "strategy_recommendation": final_strategy_text,
        "confidence_score": confidence_for_payload,
        "priority_actions": priority_actions_for_payload,
        "anomaly_report": {}, # Placeholder, as anomaly_report isn't directly passed via global hack
        "raw_messages_trace": [] # Not passing messages via this hack either
    }
    print(f"API Endpoint Returning Payload for Query {query_counter}: {json.dumps(response_payload, indent=2)}")
    return response_payload


# --- Diagnostic endpoint ---
@app.get("/diagnostic")
async def diagnostic_check():
    """Comprehensive system diagnostic"""
    print("\n🔍 Running system diagnostic...")
    
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "groq_api_configured": GROQ_API_KEY != "dummy_key_for_testing",
        "backend_api_status": "unknown", 
        "llm_status": "available" if llm is not None else "unavailable (fallback in use)",
        "tools_status": {
            "get_latest_telemetry_tool": "callable",
            "get_telemetry_history_tool": "callable",
            "calculate_performance_metrics": "callable"
        },
        "mock_data_functions_exist": _get_mock_telemetry_data is not None and _get_mock_telemetry_history is not None,
        "langgraph_graph_compiled": app_graph is not None,
        "current_time_gmt": datetime.utcnow().isoformat() + "Z"
    }

    # Ping backend API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_API_BASE_URL}/live_data", timeout=2)
            response.raise_for_status()
            diagnostics["backend_api_status"] = "operational"
        print("✅ Backend API (telemetry) is reachable")
    except Exception as e:
        diagnostics["backend_api_status"] = f"unreachable: {str(e)}"
        print(f"❌ Backend API (telemetry) is NOT reachable: {e}")

    try:
        # Test a minimal graph run to check AI execution path
        test_state = await app_graph.ainvoke({"messages": [HumanMessage(content="test diagnostic query")], "query_number": 0, "processing_time": 0.0}) # Provide required fields
        diagnostics["ai_graph_test_run"] = "successful"
        diagnostics["ai_graph_test_output_summary"] = {
            "anomaly_priority": test_state.get('anomaly_report', {}).get('priority_level'),
            "strategy_start": test_state.get('strategy_recommendation', 'N/A')[:50] + "..."
        }
        print("✅ AI graph test run successful")
    except Exception as e:
        diagnostics["ai_graph_test_run"] = f"failed: {str(e)}"
        diagnostics["ai_graph_test_output_summary"] = "N/A"
        print(f"❌ AI graph test run FAILED: {e}")
        traceback.print_exc()

    print("✅ Diagnostic check complete")
    return diagnostics


# --- Main execution block for Uvicorn ---
if __name__ == "__main__":
    print("\n🚀 Starting RaceBrain AI FastAPI Service...")
    print("✅ Ensure your GROQ_API_KEY is correctly set in the .env file.")
    print("✅ This service will listen on http://localhost:8001 (or another specified port).")
    print("✅ Make sure your data_simulator.py (to backend_api:8000) is running first.")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)