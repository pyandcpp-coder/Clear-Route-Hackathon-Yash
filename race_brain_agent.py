import os
import json
import operator
import re 
import statistics
from typing import Literal, TypedDict, Annotated, List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

from langgraph.prebuilt import ToolNode 


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in a .env file.")

LLM_MODEL_NAME = "gemma2-9b-it" 
llm_temperature = 0.3  


from clearroute_agent_tools import get_latest_telemetry, get_telemetry_history


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
    

    lap_time_degradation_percent: float = 3.0  # 3% slower than baseline
    fuel_consumption_excess_percent: float = 15.0  # 15% above expected
    
    suspension_travel_max: float = 65.0

@dataclass
class RaceContext:
    target_lap_time: float = 214.2
    base_fuel_consumption: float = 2.8
    race_duration_hours: float = 24.0
    pit_window_laps: int = 40
    safety_fuel_margin: float = 5.0  

@tool
async def get_latest_telemetry_tool() -> dict:
    """
    Fetches the latest single telemetry data point from the backend API.
    This tool provides a snapshot of the current car, environmental,
    and competitor conditions.
    """
    return await get_latest_telemetry()

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
    return await get_telemetry_history(limit)

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
        return {"error": "No telemetry data provided"}
    
    try:

        lap_times = [
            point.get("car", {}).get("last_lap_time_sec", 0)
            for point in telemetry_data
            if point.get("car", {}).get("last_lap_time_sec", 0) > 180  
        ]
        
        # Extract fuel consumption rates
        fuel_rates = [
            point.get("car", {}).get("fuel_consumption_liters_per_lap", 0)
            for point in telemetry_data
            if point.get("car", {}).get("fuel_consumption_liters_per_lap", 0) > 0
        ]
        
        # Calculate tire temperature trends
        tire_temps = {
            "FL": [point.get("car", {}).get("tire_temp_FL_C", 0) for point in telemetry_data],
            "FR": [point.get("car", {}).get("tire_temp_FR_C", 0) for point in telemetry_data],
            "RL": [point.get("car", {}).get("tire_temp_RL_C", 0) for point in telemetry_data],
            "RR": [point.get("car", {}).get("tire_temp_RR_C", 0) for point in telemetry_data]
        }
        
        metrics = {
            "lap_time_stats": {
                "average": statistics.mean(lap_times) if lap_times else 0,
                "median": statistics.median(lap_times) if lap_times else 0,
                "std_dev": statistics.stdev(lap_times) if len(lap_times) > 1 else 0,
                "trend": "improving" if len(lap_times) >= 3 and lap_times[-1] < lap_times[0] else "degrading"
            },
            "fuel_efficiency": {
                "average_consumption": statistics.mean(fuel_rates) if fuel_rates else 0,
                "trend": "improving" if len(fuel_rates) >= 3 and fuel_rates[-1] < fuel_rates[0] else "degrading"
            },
            "tire_temp_analysis": {
                tire: {
                    "average": statistics.mean([t for t in temps if t > 0]) if temps else 0,
                    "max": max([t for t in temps if t > 0]) if temps else 0,
                    "trend": "rising" if len(temps) >= 3 and temps[-1] > temps[0] else "stable"
                }
                for tire, temps in tire_temps.items()
            },
            "data_points_analyzed": len(telemetry_data)
        }
        
        return metrics
    except Exception as e:
        return {"error": f"Failed to calculate metrics: {str(e)}"}

# Combine all tools into a list
tools = [get_latest_telemetry_tool, get_telemetry_history_tool, calculate_performance_metrics]

# Initialize ToolNode with our tools
tool_executor_node = ToolNode(tools)

# --- Enhanced LangGraph State Definition ---
class RaceBrainState(TypedDict):
    """
    Represents the enhanced state of our race analysis workflow.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    latest_telemetry: Dict[str, Any]
    telemetry_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    anomaly_report: Dict[str, Any]  # Changed to dict for better structure
    strategy_recommendation: str
    confidence_score: float  # New field for recommendation confidence
    priority_actions: List[str]  # New field for prioritized actions

# --- LLM and Agent Definitions ---
from langchain_groq import ChatGroq

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=LLM_MODEL_NAME, temperature=llm_temperature) 
llm_with_tools = llm.bind_tools(tools)

# --- Enhanced Analysis Functions ---
class TelemetryAnalyzer:
    def __init__(self):
        self.thresholds = TelemetryThresholds()
        self.race_context = RaceContext()
    
    def analyze_tire_condition(self, latest_data: dict, history: list) -> dict:
        """Analyze tire conditions with trend analysis"""
        car_data = latest_data.get("car", {})
        
        tire_analysis = {
            "status": "normal",
            "issues": [],
            "recommendations": []
        }
        
        # Check current tire temperatures and pressures
        for tire in ["FL", "FR", "RL", "RR"]:
            temp = car_data.get(f"tire_temp_{tire}_C", 0)
            pressure = car_data.get(f"tire_pressure_{tire}_bar", 0)
            
            if temp > self.thresholds.tire_temp_critical:
                tire_analysis["status"] = "critical"
                tire_analysis["issues"].append(f"{tire} tire critically hot: {temp}°C")
                tire_analysis["recommendations"].append(f"Immediate action required for {tire} tire")
            elif temp > self.thresholds.tire_temp_warning:
                tire_analysis["status"] = "warning" if tire_analysis["status"] == "normal" else tire_analysis["status"]
                tire_analysis["issues"].append(f"{tire} tire running hot: {temp}°C")
                tire_analysis["recommendations"].append(f"Monitor {tire} tire closely")
            
            if pressure < self.thresholds.tire_pressure_min or pressure > self.thresholds.tire_pressure_max:
                tire_analysis["status"] = "warning" if tire_analysis["status"] == "normal" else tire_analysis["status"]
                tire_analysis["issues"].append(f"{tire} tire pressure abnormal: {pressure} bar")
                tire_analysis["recommendations"].append(f"Check {tire} tire pressure")
        
        return tire_analysis
    
    def analyze_fuel_strategy(self, latest_data: dict, history: list) -> dict:
        """Analyze fuel consumption and strategy"""
        car_data = latest_data.get("car", {})
        current_fuel = car_data.get("fuel_level_liters", 0)
        consumption_rate = car_data.get("fuel_consumption_liters_per_lap", self.race_context.base_fuel_consumption)
        current_lap = car_data.get("lap_number", 1)
        
        # Calculate remaining laps with current fuel
        remaining_laps = (current_fuel - self.race_context.safety_fuel_margin) / consumption_rate
        
        # Estimate laps to next pit window
        laps_since_pit = current_lap % self.race_context.pit_window_laps
        laps_to_pit = self.race_context.pit_window_laps - laps_since_pit
        
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
        
        if consumption_rate > self.race_context.base_fuel_consumption * (1 + self.thresholds.fuel_consumption_excess_percent/100):
            fuel_analysis["status"] = "warning" if fuel_analysis["status"] == "normal" else fuel_analysis["status"]
            fuel_analysis["recommendations"].append(f"High fuel consumption detected: {consumption_rate:.2f} L/lap")
        
        return fuel_analysis

# --- Enhanced Node Functions for LangGraph ---

async def fetch_and_analyze_data_node(state: RaceBrainState):
    print("\n--- Node: Fetching and Analyzing Data ---")
    
    # Fetch data
    latest_data = await get_latest_telemetry_tool.ainvoke({}) 
    history_data = await get_telemetry_history_tool.ainvoke({"limit": 20})
    
    # Calculate performance metrics
    metrics = await calculate_performance_metrics.ainvoke({"telemetry_data": history_data})
    
    return {
        "latest_telemetry": latest_data,
        "telemetry_history": history_data,
        "performance_metrics": metrics,
        "messages": state["messages"] 
    }

async def enhanced_anomaly_detection_node(state: RaceBrainState):
    print("\n--- Node: Enhanced Anomaly Detection ---")
    
    latest_telemetry = state["latest_telemetry"]
    telemetry_history = state["telemetry_history"]
    performance_metrics = state["performance_metrics"]
    
    # Initialize analyzer
    analyzer = TelemetryAnalyzer()
    
    # Perform local analysis
    tire_analysis = analyzer.analyze_tire_condition(latest_telemetry, telemetry_history)
    fuel_analysis = analyzer.analyze_fuel_strategy(latest_telemetry, telemetry_history)
    
    # Prepare comprehensive data for LLM
    analysis_context = {
        "latest_telemetry": latest_telemetry,
        "performance_metrics": performance_metrics,
        "tire_analysis": tire_analysis,
        "fuel_analysis": fuel_analysis,
        "recent_history_points": len(telemetry_history)
    }

    anomaly_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are the Elite Anomaly Detection Specialist for United Autosports' Le Mans team.
         Your expertise combines real-time telemetry analysis with predictive modeling to identify
         critical issues before they become race-ending problems.

         ANALYSIS CONTEXT:
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
         - Consider race context (lap number, weather, competitors)
         - Assess urgency and impact on race outcome
         
         OUTPUT FORMAT - Respond with ONLY this JSON structure:
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

    response = await llm.ainvoke(anomaly_prompt.format_messages(
        analysis_context_json_str=json.dumps(analysis_context, indent=2) # Pass as placeholder
    ))
    
    # Parse the JSON response
    try:
        # Extract JSON from response
        response_text = response.content.strip()
        json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response_text, re.DOTALL)
        if json_match:
            anomaly_data = json.loads(json_match.group(1))
        else:
            anomaly_data = json.loads(response_text)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing anomaly detection response: {e}")
        anomaly_data = {
            "priority_level": "LOW",
            "primary_anomaly": {"type": "parsing_error", "confidence": "LOW"},
            "reasoning": f"Failed to parse LLM response: {response_text[:200]}..."
        }
    
    print(f"Anomaly Detection Results: {anomaly_data.get('priority_level', 'UNKNOWN')} priority")
    
    return {
        "anomaly_report": anomaly_data,
        "messages": state["messages"] + [response]
    }

async def strategic_decision_node(state: RaceBrainState):
    print("\n--- Node: Strategic Decision Making ---")

    latest_telemetry = state["latest_telemetry"]
    anomaly_report = state["anomaly_report"] # This is now a dict, thanks to your improvements!
    performance_metrics = state["performance_metrics"]
    
    # Extract user query
    user_query_message = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), HumanMessage(content="What is the optimal race strategy?"))
    user_query = user_query_message.content
    
    # Build comprehensive race context (this is a Python dict)
    car_data = latest_telemetry.get("car", {})
    env_data = latest_telemetry.get("environmental", {})
    competitors = latest_telemetry.get("competitors", [])
    
    # --- Renamed variable and converted to JSON string here ---
    race_context_data = { # Renamed to avoid confusion with the string version in prompt
        "session_time": f"{latest_telemetry.get('timestamp_simulated_sec', 0) / 3600:.2f} hours",
        "current_lap": car_data.get("lap_number", 0),
        "position_in_race": "1st",  # Assuming leader for now
        "fuel_level": car_data.get("fuel_level_liters", 0),
        "last_lap_time": car_data.get("last_lap_time_sec", 0),
        "weather_conditions": f"Rain: {env_data.get('rain_intensity', 0)}/3, Grip: {env_data.get('track_grip_level', 1.0)}",
        "tire_condition": f"FL: {car_data.get('tire_temp_FL_C', 0)}°C", # Fixed key if simulator updated
        "competitor_gaps": [f"{comp['name']}: {comp['gap_to_leader_sec']}s" for comp in competitors[:3]]
    }
    race_context_json_str = json.dumps(race_context_data, indent=2) # Convert dict to JSON string for prompt
    # --- END Variable Rename and JSON conversion ---

    # Prepare summaries for the prompt (these are f-strings that produce simple text summaries)
    anomaly_summary_str = f"""
    Priority: {anomaly_report.get('priority_level', 'UNKNOWN')}
    Primary Issue: {anomaly_report.get('primary_anomaly', {}).get('type', 'None detected')}
    Immediate Actions: {', '.join(anomaly_report.get('immediate_actions', ['None required']))}
    """
    
    performance_summary_str = f"""
    Lap Time Trend: {performance_metrics.get('lap_time_stats', {}).get('trend', 'Unknown')}
    Average Lap Time: {performance_metrics.get('lap_time_stats', {}).get('average', 0):.2f}s
    Fuel Efficiency: {performance_metrics.get('fuel_efficiency', {}).get('average_consumption', 0):.2f} L/lap
    """

    strategy_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         f"""You are the Chief Race Strategist for United Autosports at Le Mans.
         Your decisions directly impact our chances of winning this 24-hour endurance race.
         
         STRATEGIC FRAMEWORK:
         1. **IMMEDIATE PRIORITIES** - Address critical issues first
         2. **COMPETITIVE POSITIONING** - Maintain/improve race position  
         3. **LONG-TERM OPTIMIZATION** - Manage resources for 24-hour duration
         4. **RISK MITIGATION** - Prevent race-ending failures
         5. **OPPORTUNITY EXPLOITATION** - Capitalize on competitor mistakes
         
         CURRENT RACE CONTEXT:
         {{race_context_json_str_placeholder}} # <-- Use placeholder for the JSON string here
         
         ANOMALY ANALYSIS RESULTS:
         {{anomaly_summary_placeholder}}
         
         PERFORMANCE TRENDS:
         {{performance_summary_placeholder}}
         
         STRATEGIC DECISION PROCESS:
         1. Assess immediate threats from anomaly report
         2. Evaluate competitive position vs. competitors
         3. Analyze resource management (fuel, tires, driver)
         4. Consider weather and track condition impacts
         5. Formulate multi-horizon strategy (next 30 min, 2 hours, race end)
         
         RESPONSE FORMAT:
         **STRATEGIC ASSESSMENT:**
         Current Situation: [Brief status summary]
         Priority Level: [CRITICAL/HIGH/MEDIUM/LOW]
         Confidence: [HIGH/MEDIUM/LOW] (based on data quality and certainty)
         
         **IMMEDIATE ACTIONS (Next 5-10 laps):**
         1. [Specific immediate action with reasoning]
         2. [Secondary immediate action]
         3. [Monitoring requirement]
         
         **SHORT-TERM STRATEGY (Next 30-60 minutes):**
         - [Pit strategy decision]
         - [Driver instructions]  
         - [Setup adjustments if needed]
         
         **LONG-TERM CONSIDERATIONS (Next 2-4 hours):**
         - [Resource management plan]
         - [Competitive positioning strategy]
         - [Contingency preparations]
         
         **RISK MITIGATION MEASURES:**
         - [Primary risk and mitigation]
         - [Secondary risk and mitigation]
         
         **EXPECTED OUTCOMES:**
         If executed properly: [Positive outcomes]
         Risks if not executed: [Negative consequences]
         
         **KEY MONITORING POINTS:**
         - [Critical metric to watch]
         - [Trigger point for strategy adjustment]
         """),
        ("human", "Query: {user_query}\n\nProvide comprehensive strategic recommendations based on current race conditions.")
    ])
    
    response = await llm.ainvoke(strategy_prompt.format_messages(
        user_query=user_query,
        race_context_json_str_placeholder=race_context_json_str, # Pass the JSON string here
        anomaly_summary_placeholder=anomaly_summary_str, # Pass summary strings as placeholders
        performance_summary_placeholder=performance_summary_str # Pass summary strings as placeholders
    ))
    
    # Calculate confidence score based on data quality and anomaly certainty
    confidence_score = 0.8  # Base confidence
    if anomaly_report.get('primary_anomaly', {}).get('confidence') == 'HIGH':
        confidence_score += 0.1
    elif anomaly_report.get('primary_anomaly', {}).get('confidence') == 'LOW':
        confidence_score -= 0.2
    
    # Extract priority actions from response (still using anomaly_report for now)
    priority_actions = anomaly_report.get('immediate_actions', []) # This list will come from the LLM's response later

    print(f"StrategyDecisionAgent Raw Response: {response.content}")
    
    return {
        "strategy_recommendation": response.content,
        "confidence_score": confidence_score,
        "priority_actions": priority_actions,
        "messages": state["messages"] + [response]
    }


# --- Build the Enhanced LangGraph Workflow ---

workflow = StateGraph(RaceBrainState)

# Add nodes
workflow.add_node("fetch_and_analyze", fetch_and_analyze_data_node)
workflow.add_node("anomaly_detection", enhanced_anomaly_detection_node)
workflow.add_node("strategic_decision", strategic_decision_node)
workflow.add_node("tool_node", tool_executor_node)

# Define the entry point
workflow.set_entry_point("fetch_and_analyze")

# Define edges (transitions between nodes)
workflow.add_edge("fetch_and_analyze", "anomaly_detection")
workflow.add_edge("anomaly_detection", "strategic_decision")

# Conditional edge for tool calling
workflow.add_conditional_edges(
    "strategic_decision",
    lambda x: "tool_node" if x["messages"] and x["messages"][-1].tool_calls else END,
    {
        "tool_node": "tool_node",
        END: END
    },
)

workflow.add_edge("tool_node", "strategic_decision")

# Compile the graph
app_graph = workflow.compile()

# --- Enhanced Main Execution Loop ---
async def run_enhanced_race_brain():
    print("🏁 ENHANCED RaceBrain AI: Elite Strategy Co-Pilot 🏁")
    print("=" * 55)
    print("Advanced telemetry analysis with predictive insights")
    print("Type your queries below. Type 'quit' to exit.")
    print("=" * 55)
    
    session_queries = 0
    
    while True:
        print(f"\n[Query #{session_queries + 1}]")
        user_input = input("🔧 Engineer Query > ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n🏁 RaceBrain AI session ended. Good luck with the race!")
            break

        session_queries += 1
        start_time = datetime.now()

        # Enhanced initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "latest_telemetry": {},
            "telemetry_history": [],
            "performance_metrics": {},
            "anomaly_report": {},
            "strategy_recommendation": "",
            "confidence_score": 0.0,
            "priority_actions": []
        }

        final_state = {}
        try:
            print("\n⚡ Processing telemetry and generating strategy...")
            async for s in app_graph.astream(initial_state):
                if "__end__" not in s:
                    # Show progress indicators
                    if "fetch_and_analyze" in s:
                        print("📊 Analyzing telemetry data...")
                    elif "anomaly_detection" in s:
                        print("🔍 Detecting anomalies...")
                    elif "strategic_decision" in s:
                        print("🎯 Formulating strategy...")
                else:
                    final_state = s["__end__"]
                    
        except Exception as e:
            print(f"\n❌ ERROR during analysis: {e}")
            continue

        # Enhanced output formatting
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print(f"🏆 RACEBRAIN AI STRATEGIC RECOMMENDATION #{session_queries}")
        print(f"⏱️  Processing Time: {processing_time:.2f}s")
        print(f"📊 Confidence Score: {final_state.get('confidence_score', 0.0):.2f}/1.0")
        print(f"{'='*60}")
        
        # Display priority actions if any
        priority_actions = final_state.get('priority_actions', [])
        if priority_actions:
            print("🚨 PRIORITY ACTIONS:")
            for i, action in enumerate(priority_actions, 1):
                print(f"   {i}. {action}")
            print()
        
        # Display anomaly status
        anomaly_report = final_state.get('anomaly_report', {})
        priority_level = anomaly_report.get('priority_level', 'NONE')
        if priority_level != 'NONE':
            priority_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(priority_level, '⚪')
            print(f"{priority_emoji} ANOMALY STATUS: {priority_level}")
            if anomaly_report.get('primary_anomaly', {}).get('type'):
                print(f"   Primary Issue: {anomaly_report['primary_anomaly']['type']}")
            print()
        
        # Display main strategy recommendation
        recommendation = final_state.get("strategy_recommendation", "No recommendation generated.")
        print("📋 STRATEGY RECOMMENDATION:")
        print("-" * 40)
        print(recommendation)
        print("=" * 60)

if __name__ == "__main__":
    import asyncio
    
    print("\n🚀 SYSTEM REQUIREMENTS CHECK:")
    print("✅ Ensure 'data_simulator.py' is running (Port 8000)")
    print("✅ Ensure 'backend_api.py' is running (FastAPI server)")
    print("✅ GROQ API key is configured in .env file")
    print("\n🏁 Starting Enhanced RaceBrain AI...\n")
    
    asyncio.run(run_enhanced_race_brain())