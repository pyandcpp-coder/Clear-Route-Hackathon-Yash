# 🏁 Le Mans RaceBrain AI Pro: Real-time Strategic Co-Pilot 🏎️

This project is an innovative, agentic AI solution designed to provide a measurable, engineering-driven advantage during the high-pressure 24 Hours of Le Mans. It acts as a "RaceBrain AI" co-pilot, transforming an overwhelming flood of real-time telemetry data into decisive, race-winning strategic insights.

Developed for a hackathon, this solution focuses on `Track 1: Live Race Strategy & Operations`, delivering clear, actionable recommendations in a time-critical environment.

## 🌟 Features

*   **Realistic Telemetry Simulation:** A highly detailed Python-based simulator (`data_simulator.py`) generates live telemetry data (tire temps, fuel, engine, weather, competitor positions) with dynamic, interactive anomaly injection (e.g., tire punctures, sudden rain, engine issues).
*   **Robust Data Pipeline:**
    *   `data_simulator.py` streams data to a dedicated FastAPI backend (`backend_api.py`).
    *   `backend_api.py` acts as a central telemetry hub, storing and serving live and historical data via REST endpoints.
*   **Agentic AI Core (LangGraph + Groq):**
    *   `ai_api.py` hosts the intelligent core, implemented using `LangGraph` for multi-agent orchestration.
    *   Leverages `Groq`'s lightning-fast LLM inference (`llama3-70b-8192` or `deepseek-llm/deepseek-coder-6.7b-instruct`) for real-time decision-making.
    *   **`fetch_and_analyze_data_node`:** Gathers and pre-processes live/historical telemetry, including calculating performance metrics.
    *   **`enhanced_anomaly_detection_node`:** An intelligent agent that identifies complex, multi-factor anomalies in real-time, extracting structured JSON reports even from conversational LLM output.
    *   **`strategic_decision_node`:** The "Chief Strategist" agent that synthesizes anomaly reports, live race context, and performance trends to generate comprehensive, actionable strategic recommendations. Includes fallback logic if the LLM is unavailable.
*   **Interactive Web UI (Streamlit):**
    *   `streamlit_ui.py` provides a visually rich, real-time dashboard for race engineers.
    *   Features include: Live KPI display, dynamic telemetry charts, real-time anomaly alerts, competitor overview, and a conversational chat interface to interact directly with the RaceBrain AI.
    *   Allows engineers to query the AI in natural language and receive immediate strategic advice.
*   **Modular Microservices Architecture:** The solution is divided into distinct Python services (simulator, telemetry API, AI API, UI), promoting scalability, maintainability, and clear separation of concerns.
*   **Error Handling & Fallbacks:** Robust error handling throughout the pipeline, including mock data generation and rule-based fallback strategy generation if APIs or LLMs are unreachable.


## 🚀 Getting Started

Follow these steps to set up and run the entire RaceBrain AI system.

### Prerequisites

*   Python 3.8+ (Python 3.10+ recommended)
*   `git` installed on your system

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/pyandcpp-coder/Clear-Route-Hackathon-Yash.git
    cd Clear-Route-Hackathon-Yash
    ```

2.  **Create and Activate Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Groq API Key:**
    *   Sign up for a Groq API key at [Groq Console](https://console.groq.com/).
    *   Create a file named `.env` in the root of your `Clear-Route-Hackathon-Yash` directory (same level as `ai_api.py`).
    *   Add your Groq API key to this file:
        ```
        GROQ_API_KEY=gsk_YOUR_ACTUAL_GROQ_API_KEY_HERE
        ```
    *   **Important:** Do NOT commit your `.env` file to Git. It's already included in `.gitignore`.

## 🏃 How to Run (4 Separate Terminals Required)

Each component of the system runs as a separate microservice. Open **four distinct terminal windows** in your project directory and follow these steps in order. Ensure your virtual environment (`.venv`) is activated in each terminal.

### 1. Start the Telemetry Data Simulator

This generates the live race data.
**Terminal 1:**
```bash python data_simulator.py ```

### 2. Start the Telemetry Data Backend API
This FastAPI service receives data from the simulator and serves it to other components.
Terminal 2:

```uvicorn backend_api:app --reload --port 8000```


### 3. Start the RaceBrain AI Service
This FastAPI service hosts the LangGraph AI agents and processes strategic queries.
Terminal 3:

``` uvicorn ai_api:app --reload --port 8001 ```


### 4. Start the Streamlit Web UI

``` streamlit run streamlit_ui.py ```







