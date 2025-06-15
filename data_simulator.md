# 🏁 Le Mans 24H Race Data Simulator Documentation

## Table of Contents
1.  [Overview](#1-overview)
2.  [Core Configuration Parameters](#2-core-configuration-parameters)
    *   [Track Data (`TRACK_SECTORS`)](#track-data-track_sectors)
    *   [Hypercar Competitors (`HYPERCAR_COMPETITORS`)](#hypercar-competitors-hypercar_competitors)
    *   [Our Car Configuration (`OUR_CAR_CONFIG`)](#our-car-configuration-our_car_config)
    *   [Weather Patterns (`WEATHER_PATTERNS`, `HOURLY_WEATHER_MODIFIERS`)](#weather-patterns-weather_patterns-hourly_weather_modifiers)
    *   [Enhanced Anomalies (`ENHANCED_ANOMALIES`)](#enhanced-anomalies-enhanced_anomalies)
3.  [Global State Variables](#3-global-state-variables)
4.  [Key Functions](#4-key-functions)
    *   [`calculate_sector_time(sector_num, base_lap_time, conditions)`](#calculatesector_timesector_num-base_lap_time-conditions)
    *   [`update_weather_system()`](#update_weather_system)
    *   [`simulate_competitor_behavior()`](#simulate_competitor_behavior)
    *   [`generate_enhanced_telemetry()`](#generate_enhanced_telemetry)
    *   [`update_anomalies()`](#update_anomalies)
    *   [`activate_anomaly(anomaly_id, duration, message)`](#activate_anomalyanomaly_id-duration-message)
    *   [`input_listener()`](#input_listener)
    *   [`enhanced_telemetry_generator(start_hour, total_race_hours)`](#enhanced_telemetry_generatorstart_hour-total_race_hours)
5.  [Simulation Flow](#5-simulation-flow)
6.  [Realism and Data Quality](#6-realism-and-data-quality)
7.  [Telemetry Output Structure](#7-telemetry-output-structure)
8.  [How to Run](#8-how-to-run)

---

## 1. Overview

The `data_simulator.py` script is a sophisticated Python-based simulator designed to generate realistic, dynamic, and comprehensive telemetry data for a 24-hour endurance race, specifically modeled after the Le Mans 24 Hours. It simulates the behavior of a single "our car" (an LMP2 prototype) and a set of Hypercar competitors, taking into account various environmental factors, car dynamics, and race incidents.

The primary purpose of this simulator is to provide a rich, continuous stream of data to a backend API (`http://localhost:8000/telemetry`), which can then be consumed by a frontend dashboard (like a Streamlit application) or an AI/ML model for real-time analysis and strategic decision-making.

**Key Features:**
*   **Time-based Simulation:** Progresses in simulated time, allowing for day-night cycles and long-duration race scenarios.
*   **Dynamic Car Physics:** Calculates lap times, fuel consumption, tire wear, and component temperatures based on current conditions.
*   **Realistic Environmental System:** Simulates changing weather, track temperature, ambient temperature, wind, and visibility, with Le Mans-specific probabilities.
*   **Competitor Simulation:** Tracks multiple AI-driven competitor cars with their own characteristics, pit strategies, and reliability factors.
*   **Enhanced Anomaly System:** Introduces unexpected race incidents (e.g., punctures, engine issues, safety cars, sudden weather changes) that critically impact car performance and parameters.
*   **Interactive Control:** Allows a user to manually trigger anomalies via the console, simulating race control interventions.
*   **Smoothed Data Output:** Implements smoothing techniques to prevent erratic, "jittery" data, ensuring a more realistic and actionable data stream for analytics.

## 2. Core Configuration Parameters

The simulator's behavior is driven by several predefined dictionaries and constants.

### Track Data (`TRACK_SECTORS`)
Defines the characteristics of the Le Mans track, divided into three sectors.
*   `TRACK_LENGTH_KM`: Total track length in kilometers.
*   `TOTAL_LAPS_24H`: Approximate laps for a 24-hour race.
*   `BASE_LAP_TIME_SEC`, `FASTEST_LAP_TIME_SEC`, `SLOWEST_LAP_TIME_SEC`: Baseline lap time references.
*   Each sector has:
    *   `length_km`: Length of the sector.
    *   `characteristics`: List of descriptive tags (e.g., "long_straight", "chicane", "medium_corners").
    *   `avg_speed_kmh`, `max_speed_kmh`: Typical and maximum speeds for the sector.
    *   `elevation_change_m`: Elevation changes within the sector.
    *   `tire_stress`: Indication of tire wear intensity.
    *   `fuel_consumption_modifier`: Multiplier for fuel consumption in that sector.

### Hypercar Competitors (`HYPERCAR_COMPETITORS`)
A dictionary mapping car numbers (as strings) to their detailed configurations. These represent the main rivals in the top Hypercar class. Each competitor has:
*   `name`, `manufacturer`, `drivers`: Basic identification.
*   `base_lap_time`: Their ideal lap time.
*   `current_gap_sec`: Initial gap to the leader (our car is assumed leader in its class).
*   `fuel_tank_liters`, `fuel_consumption_base`: Fuel characteristics.
*   `tire_degradation_rate`: Rate at which their tires wear.
*   `reliability_factor`: Probability of mechanical issues.
*   `pit_frequency_laps`: Ideal number of laps between pit stops.
*   `strengths`, `weaknesses`: Qualitative attributes affecting their performance under certain conditions (e.g., "night_pace").

### Our Car Configuration (`OUR_CAR_CONFIG`)
Defines the specifications for the simulated "our car" (United Autosports Oreca 07 LMP2).
*   `name`, `class`, `drivers`: Basic identification.
*   `base_lap_time`: Base lap time for our car (LMP2 is slower than Hypercar).
*   `fuel_tank_liters`, `fuel_consumption_base`: Fuel characteristics.
*   `tire_degradation_rate`: Tire wear rate.
*   `reliability_factor`: Reliability of our car.
*   `pit_frequency_laps`: Ideal number of laps between pit stops.
*   `target_position`, `class_competitors`: Race context for our car.

### Weather Patterns (`WEATHER_PATTERNS`, `HOURLY_WEATHER_MODIFIERS`)
Defines possible weather states and how they affect the track.
*   `WEATHER_PATTERNS`: Each pattern (`clear`, `partly_cloudy`, `light_rain`, `heavy_rain`, `fog`) has a `probability`, `grip_factor` (multiplier for track grip), and `visibility`.
*   `HOURLY_WEATHER_MODIFIERS`: Modifies the probability of rain and fog based on the simulated hour of the day, reflecting realistic Le Mans weather trends (e.g., higher fog chance in the morning).
*   `WEATHER_CHECK_INTERVAL_SEC`: A new constant (`300` seconds / 5 simulated minutes) to control how frequently the simulator evaluates a major weather pattern change, preventing rapid, unrealistic weather flickering.

### Enhanced Anomalies (`ENHANCED_ANOMALIES`)
A crucial dictionary defining various race incidents that can be triggered. Each anomaly has:
*   `type`: A descriptive name (e.g., `tire_puncture_front_left`, `engine_overheating`, `safety_car_period`).
*   `message`: A human-readable message for the event.
*   `duration_sec`: How long the anomaly lasts (0 for permanent until reset).
*   `severity`: `critical`, `high`, `medium`, `info`.
*   `lap_time_impact`: A multiplier applied to lap times during the anomaly.
*   `repair_time`: Not directly used in current simulation logic, but indicates time needed for repair.
*   Special anomaly `"0"` (`reset_all_systems`) allows immediate clearing of all active issues and reset of car parameters to normal.

## 3. Global State Variables

These variables maintain the current state of the simulation and the car.

*   `current_lap`: The current lap number for our car.
*   `current_lap_time_sec`: Time elapsed in the current lap.
*   `total_simulated_time_sec`: Total race time elapsed.
*   `our_car_fuel`: Current fuel level in our car's tank.
*   `lap_start_time_simulated`: `total_simulated_time_sec` at the start of the current lap.
*   `last_lap_start_time_simulated`: `total_simulated_time_sec` at the start of the previous lap.
*   `our_car_last_lap_time`: The time taken for the previous complete lap.
*   `current_sector`: The current track sector (1, 2, or 3).
*   `sector_progress`: Time spent in the current sector.
*   `tire_compound`: Current tire compound used.
*   `tire_age_laps`: Number of laps on current set of tires.
*   `tire_wear`: Dictionary storing wear percentage for each tire (`FL`, `FR`, `RL`, `RR`).
*   `tire_temperatures`: Dictionary storing temperature for each tire.
*   `tire_pressures`: Dictionary storing pressure for each tire.
*   `current_driver`: Index of the current driver from `OUR_CAR_CONFIG["drivers"]`.
*   `driver_stint_time`: Time elapsed in the current driver's stint.
*   `last_pit_lap`: Lap number of the last pit stop.
*   `pit_strategy`: Current overall pit strategy (e.g., "normal").
*   `fuel_saving_mode`, `push_mode`: Boolean flags for driving modes.
*   `current_weather`: Current weather condition.
*   `track_temperature`, `ambient_temperature`: Current environmental temperatures.
*   `wind_speed`, `wind_direction`: Current wind conditions.
*   `track_grip`: Overall grip level of the track surface.
*   `visibility`: Current visibility level.
*   `last_weather_check_time`: Timestamp of the last time a full weather change check was performed.
*   **Smoothed Performance Globals:**
    *   `last_speed_kmh`, `last_throttle_percent`, `last_brake_percent`, `last_engine_rpm`: These store the values from the *previous* simulation step for speed, throttle, brake, and RPM. They are crucial for implementing the smoothing logic, preventing abrupt jumps in these metrics.
*   **Initial Value Constants:** `INITIAL_TIRE_TEMPS`, `INITIAL_TIRE_PRESSURES`, `INITIAL_OIL_TEMP`, `INITIAL_WATER_TEMP`, `INITIAL_HYBRID_BATTERY`, `INITIAL_HYBRID_POWER`, `INITIAL_VISIBILITY`: These store the default "healthy" values for various car parameters, used primarily for resetting systems after an anomaly resolves.
*   `safety_car_active`: Boolean flag indicating a safety car period.
*   `yellow_flag_sectors`: List of sectors under yellow flag.
*   `track_limits_warnings`: Counter for track limits violations.
*   `race_incidents`: List of recorded race incidents.
*   `competitor_positions`: A dictionary tracking the real-time state of each Hypercar competitor (lap, gap, fuel, tires, issues, pit status).
*   `active_anomalies`: A dictionary storing currently active anomalies and their `start_time` and calculated `end_time`.
*   `command_queue`: A `queue.Queue` for inter-thread communication, allowing the `input_listener` to send commands to the main simulation loop.
*   `stop_simulation_event`: A `threading.Event` used to signal the simulation threads to stop gracefully.

## 4. Key Functions

### `calculate_sector_time(sector_num, base_lap_time, conditions)`
Calculates the time taken to complete a specific track sector.
*   **Inputs:** `sector_num` (1-3), `base_lap_time` (reference lap time), `conditions` (dictionary containing `grip_factor` and `visibility`).
*   **Logic:**
    *   Starts with a base sector time proportional to the sector's length relative to the total track length.
    *   Applies modifiers based on:
        *   `weather_modifier`: Track grip (reduces speed, especially in corners, in wet conditions).
        *   `tire_performance`: Average tire temperature (cold/overheating tires reduce performance) and average tire wear (increases time).
        *   `fuel_impact`: Current fuel load (heavier car is slower).
        *   `driver_fatigue`: Increases lap time slightly over a long stint.
    *   Adds a small random variation (±2%) for natural fluctuation.

### `update_weather_system()`
Manages dynamic weather changes and environmental conditions.
*   **Logic:**
    *   Continuously updates `ambient_temperature`, `track_temperature`, `wind_speed`, and `wind_direction` based on a daily cycle (cosine wave for temperature) and small random fluctuations.
    *   Periodically checks for major weather pattern changes (controlled by `WEATHER_CHECK_INTERVAL_SEC`). When a check occurs, it:
        *   Determines a `weather_change_chance` influenced by the current hour of day (via `HOURLY_WEATHER_MODIFIERS`).
        *   Rolls a random number against this chance.
        *   If a change is indicated, it selects a new `current_weather` type based on `WEATHER_PATTERNS` probabilities, with a slight bias towards persisting the current weather.
        *   Updates `track_grip` and `visibility` according to the new weather type.

### `simulate_competitor_behavior()`
Updates the state of each Hypercar competitor.
*   **Logic:**
    *   For each competitor, it calculates a `final_lap_time` based on their `base_lap_time`, `track_grip` (weather impact), `tire_age` (`tire_deg_impact`), `fuel_level` (`fuel_load_impact`), and `driver_variance` (including night pace strength).
    *   Updates the `gap_to_leader` (relative to our car) proportionally over the simulation interval (0.5 seconds), ensuring smooth gap changes.
    *   Implements a basic pit stop logic: if `pit_frequency_laps` is met, there's a chance they'll pit, resetting their tires and fuel, and adding a time penalty to their gap.
    *   Updates `tire_age` and `fuel_level` based on simulated laps and consumption.
    *   Introduces a small chance of a random reliability issue, which adds a time penalty to their gap.

### `generate_enhanced_telemetry()`
The core function that runs at each simulation step (0.5-second interval) to update all car parameters and compile the telemetry data.
*   **Logic Flow:**
    1.  **Time Progression:** Increments `total_simulated_time_sec` and `current_lap_time_sec`.
    2.  **Weather Update:** Calls `update_weather_system()`.
    3.  **Sector Progression:** Determines if the car has completed a sector. If so, updates `current_sector` and resets `sector_progress`.
    4.  **Lap Completion:** If `current_sector` loops back to 1, a lap is completed:
        *   Increments `current_lap` and resets `lap_start_time_simulated`.
        *   Calculates `our_car_last_lap_time` by summing sector times, applying any active anomaly impacts (`lap_time_impact`).
        *   Calculates `fuel_used` based on `base_consumption`, `fuel_saving_mode`/`push_mode`, and `track_grip`, then updates `our_car_fuel`.
        *   Updates `tire_wear` for all tires based on `tire_degradation_rate`, `track_grip`, and `ambient_temperature`.
    5.  **Pit Stop Logic:** Checks conditions for a pit stop (low fuel, worn tires, driver stint time). If a pit is needed, it simulates the pit stop duration by adding time to `total_simulated_time_sec`, then resets `our_car_fuel`, `tire_wear`, `tire_age_laps`, and potentially changes `current_driver`.
    6.  **Competitor Simulation:** Calls `simulate_competitor_behavior()` to update competitor states.
    7.  **Dynamic Car Performance (Smoothed):**
        *   Determines `target_` values (e.g., `target_speed_kmh`, `target_throttle_percent`) based on the `current_sector` characteristics (e.g., high speed on straights, lower in chicanes).
        *   Applies a **smoothing algorithm** (using `smoothing_factor`) to gradually transition the `last_` values towards these `target_` values, adding small random noise for realism. This prevents jerky data.
        *   Clamps all calculated values (`speed_kmh`, `throttle_percent`, `brake_percent`, `engine_rpm`) within realistic physical limits.
        *   Applies general `weather` and `visibility` effects to these performance metrics.
        *   *Important Note*: Direct parameter manipulation from `activate_anomaly` (see below) can override these calculations if a critical anomaly is active.
    8.  **Tire Temperatures & Pressures:** Updates these based on `speed`, `brake_percent`, `sector_characteristics`, `ambient_temperature`, and `tire_wear`. (Again, direct anomaly effects take precedence if a puncture is active).
    9.  **Suspension Data:** Generates travel values based on sector type, with increased travel for punctures.
    10. **Engine & Drivetrain Data:** Updates `oil_temp_C` and `water_temp_C` based on `engine_rpm` and `ambient_temperature`.
    11. **Hybrid System Data:** Simulates battery charge/discharge and power output based on `throttle_percent`.
    12. **Race Control Metrics:** Calculates `fuel_laps_remaining`, `estimated_pit_window`, `track_limits_warnings`, and `track_limits_risk`.
    13. **Telemetry Compilation:** Gathers all current state variables into a structured JSON-compatible dictionary. This dictionary is the output that is sent to the backend.

### `update_anomalies()`
Manages the lifecycle of active anomalies.
*   **Logic:**
    *   Iterates through `active_anomalies` to check if their `end_time` has been reached.
    *   If an anomaly has expired, it prints a resolution message and `del`etes it from `active_anomalies`.
    *   **Crucially, it also resets the specific car parameters affected by that anomaly to their normal operating ranges or initial values.** This ensures that when an anomaly resolves, the telemetry values return from their critical state to a healthy one (e.g., tire pressure returns to normal after a puncture is "repaired").

### `activate_anomaly(anomaly_id, duration, message)`
Triggers a specific race anomaly.
*   **Inputs:** `anomaly_id` (string key from `ENHANCED_ANOMALIES`), `duration` (in seconds), `message` (descriptive text).
*   **Logic:**
    *   Looks up the `anomaly_config` based on `anomaly_id`.
    *   If `anomaly_id` is "0" (`reset_all_systems`), it clears all active anomalies and resets *all* global car parameters (`tire_wear`, `tire_temperatures`, `oil_temp_C`, `speed_kmh`, etc.) to their pristine/initial state.
    *   For other anomalies, it adds the anomaly to the `active_anomalies` dictionary with its `start_time` and calculated `end_time`.
    *   **Direct Parameter Manipulation:** This is where the "critical range" values are immediately enforced. For example:
        *   **`tire_puncture_front_left`**: `tire_wear["FL"]` is set to 0.9, `tire_pressures["FL"]` to 0.5 bar, and `tire_temperatures["FL"]` to 120.0°C.
        *   **`hybrid_system_failure`**: `last_speed_kmh` and `last_engine_rpm` are immediately reduced.
        *   **`safety_car_period`**: `safety_car_active` is set to `True`, and `last_speed_kmh`, `last_throttle_percent`, `last_brake_percent`, `last_engine_rpm` are set to reflect safety car speeds.
        *   **`engine_overheating`**: `oil_temp_C` and `water_temp_C` are set to critical high values, and `last_throttle_percent` is capped.
        *   **`sudden_weather_change`**: `current_weather` becomes `heavy_rain`, `track_grip` and `visibility` drop significantly, and `last_speed_kmh` is reduced.
    *   By directly setting the values (especially the `last_` values for smoothed parameters), the telemetry immediately reflects the critical state, and the smoothing logic then trends from this critical point rather than gradually entering it.

### `input_listener()`
Runs in a separate thread to listen for user commands from the console.
*   **Functionality:**
    *   Displays a menu of available anomalies (1-10) and a reset option (0), along with a quit command (`q`).
    *   Puts valid commands into the `command_queue` for the main simulation loop to process.
    *   Catches `EOFError` and `KeyboardInterrupt` to ensure graceful shutdown.

### `enhanced_telemetry_generator(start_hour, total_race_hours)`
The main entry point for the simulation.
*   **Inputs:** `start_hour` (initial hour into the 24h race), `total_race_hours`.
*   **Logic:**
    *   Initializes `total_simulated_time_sec` and `lap_start_time_simulated` based on `start_hour`.
    *   Initializes initial temperature and hybrid states.
    *   Sets `max_simulated_time_sec` for the race duration.
    *   Sets `update_interval` (how frequently data is generated, 0.5s) and `real_time_factor` (how many simulated seconds pass per real second, default 30x).
    *   Starts a loop that continues until the race ends or the `stop_simulation_event` is set.
    *   Inside the loop:
        *   Checks `command_queue` for user input (quit or anomaly triggers).
        *   Calls `update_anomalies()` to check for and resolve expired anomalies.
        *   Calls `generate_enhanced_telemetry()` to update the car state and generate the latest telemetry dictionary.
        *   Sends the generated telemetry (as JSON) to the `BACKEND_API_URL` via an HTTP POST request.
        *   Prints periodic summary updates to the console (lap, time, last lap time, fuel, tire temp, weather).
        *   Pauses the thread using `time.sleep(update_interval / real_time_factor)` to control the simulation speed relative to real-time.

## 5. Simulation Flow

1.  **Initialization:** Global variables are set to initial values. The `input_listener` starts in a separate thread.
2.  **Main Loop (`enhanced_telemetry_generator`):**
    *   Runs continuously, advancing simulated time in 0.5-second steps.
    *   Each step involves:
        *   Processing user commands (e.g., activating an anomaly).
        *   Updating and expiring existing anomalies, *critically resetting parameters upon resolution*.
        *   Calculating the car's state:
            *   Sector progression and lap completion.
            *   Fuel consumption.
            *   Tire wear, temperature, and pressure.
            *   Engine, brake, throttle, and suspension dynamics, **smoothed over time**.
            *   Hybrid system state.
        *   Simulating competitor actions (lap times, pits, issues).
        *   Updating environmental conditions (weather, temperature, wind).
        *   **Compiling all this data into a structured JSON telemetry object.**
        *   **Sending the telemetry object to the backend API.**
    *   The loop pauses briefly to control the real-time speed of the simulation (e.g., 30x faster than real life).
3.  **Anomaly Activation:** When a user triggers an anomaly, the `activate_anomaly` function is called. This function *immediately* modifies the relevant global state variables (e.g., setting a tire's pressure to 0.5 bar, an engine temp to 130°C), forcing the telemetry values into a critical range for the anomaly's duration.
4.  **Anomaly Resolution:** Once an anomaly's duration expires, `update_anomalies` resets the affected parameters to their normal operational values, allowing the car to return to a healthy state and the telemetry values to return to normal ranges.

## 6. Realism and Data Quality

The simulator emphasizes realism and data quality through several mechanisms:

*   **Smoothed Transitions:** Instead of instant jumps, performance metrics (speed, throttle, brake, RPM) smoothly transition between states, mimicking real-world inertia. This is achieved by calculating a `target_value` and then gradually moving the `current_value` (`last_speed_kmh`, etc.) towards that target with a `smoothing_factor`.
*   **Parameter Interdependencies:** Tire temperatures are affected by braking, speed, and ambient temperature. Fuel consumption is influenced by driving mode and track grip. This creates a more cohesive simulation.
*   **Time-Based Events:** Weather patterns and driver fatigue are linked to the simulated time of day and race duration.
*   **Direct Critical Values:** When an anomaly occurs, the simulator directly assigns critical or abnormal numerical values to the affected parameters. This means any consuming application (like a Streamlit dashboard) will see the "problem" directly in the numbers (e.g., tire pressure at 0.5 bar, engine oil at 130°C), rather than just relying on a boolean flag or separate alert. This makes the data inherently more expressive of the car's state.
*   **Randomness within Bounds:** Random variations are applied to prevent a completely deterministic simulation, but these variations are typically within small, realistic bounds to avoid chaotic data.

## 7. Telemetry Output Structure

The `generate_enhanced_telemetry()` function outputs a comprehensive JSON object at each step. This is the data structure sent to your backend and consumed by your Streamlit app. Key sections include:

*   `timestamp_simulated_sec`: Current simulated time.
*   `race_info`: `current_hour`, `race_time_elapsed_sec`, `race_time_remaining_sec`, `time_of_day`, `is_night`.
*   `car`: Detailed metrics for our car:
    *   Driver and stint info.
    *   Lap data (`lap_number`, `current_lap_time_sec`, `last_lap_time_sec`, `average_lap_time_sec`).
    *   Track progress (`current_sector`, `sector_progress_percent`).
    *   Performance (`speed_kmh`, `engine_rpm`, `throttle_percent`, `brake_percent`, `gear`, `drs_active`).
    *   Fuel data (`fuel_level_liters`, `fuel_consumption_current_L_per_lap`, `fuel_laps_remaining`, `fuel_saving_mode`, `push_mode`).
    *   Temperatures (`oil_temp_C`, `water_temp_C`).
    *   Hybrid system (`hybrid_battery_percent`, `hybrid_power_output_kw`).
    *   Tires (`tire_compound`, `tire_age_laps`, `tire_temp_FL_C/FR_C/RL_C/RR_C`, `tire_pressure_FL_bar/etc`, `tire_wear_FL_percent/etc`).
    *   Suspension (`suspension_travel_FL_mm/etc`).
    *   Pit strategy (`last_pit_lap`, `laps_since_pit`, `estimated_pit_window`, `pit_strategy`).
    *   Race control specific to our car (`track_limits_warnings`, `track_limits_risk_percent`).
*   `environmental`: Current weather and track conditions: `current_weather`, `ambient_temp_C`, `track_temp_C`, `humidity_percent`, `wind_speed_kmh`, `wind_direction_deg`, `track_grip_level`, `visibility_level`, `sunrise_time`, `sunset_time`.
*   `track_info`: Static track configuration details.
*   `race_control`: Global race status: `safety_car_active`, `yellow_flag_sectors`, `track_status`.
*   `competitors`: A list of dictionaries, each detailing a competitor's current state (`car_number`, `name`, `gap_to_leader_sec`, `last_lap_time_sec`, `fuel_level_liters`, `tire_age_laps`, `pit_status`, `current_issues`, `pit_strategy`, `sector_time_X`).
*   `active_anomalies`: A list of currently active anomalies, including their `type`, `message`, `severity`, and `duration_remaining_sec`. This provides contextual information for *why* parameters might be in a critical range.
*   `strategy`: Our car's high-level strategy overview: `current_strategy`, `fuel_target_laps`, `tire_change_recommended`, `driver_change_due`, `next_pit_recommendation`, `position_in_class`, `gap_to_class_leader`, `laps_to_class_leader`.

## 8. How to Run

To run the simulator and use it with your backend/frontend:

1.  **Ensure Python is installed:** Python 3.8+ is recommended.
2.  **Install Dependencies:**
    ```bash
    pip install requests numpy
    ```
3.  **Run the Simulator:**
    Open a terminal and navigate to the directory containing `data_simulator.py`.
    ```bash
    python data_simulator.py
    ```
    The simulator will start generating data and printing updates to the console. You will also see a prompt to enter commands (anomaly triggers).
    *   Type a number `1` through `10` and press Enter to activate an anomaly.
    *   Type `0` to reset all active anomalies.
    *   Type `q` or `quit` to stop the simulator.

**Note:** This simulator is designed to work in conjunction with a separate backend API (e.g., `backend.py` from your project) that listens on `http://localhost:8000/telemetry` for the POST requests containing the telemetry data. Ensure your backend service is running before starting the simulator if you want the data to be received.