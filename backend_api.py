from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import collections


app = FastAPI(
    title="Le Mans Race Data API",
    description="Provides real-time telemetry, environmental, and competitor data for United Autosports."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

latest_telemetry_data = {}


telemetry_history = collections.deque(maxlen=200) 

@app.post("/telemetry")
async def receive_telemetry(data: dict):
    """
    Receives a single telemetry data point from the simulator.
    This is where the simulator will POST its data.
    """
    global latest_telemetry_data
    global telemetry_history

    if not data:
        raise HTTPException(status_code=400, detail="Empty data received.")

    latest_telemetry_data = data
    telemetry_history.append(data)

    print(f"Received telemetry point: Simulated Time = {data.get('timestamp_simulated_sec', 'N/A')}")
    print(json.dumps(data, indent=2)) 

    return {"status": "success", "message": "Telemetry received and updated."}

@app.get("/live_data")
async def get_live_data():
    """
    Returns the latest comprehensive telemetry data point.
    This is what the UI and Agentic AI will poll for current state.
    """
    if not latest_telemetry_data:
        raise HTTPException(status_code=404, detail="No telemetry data available yet.")
    return latest_telemetry_data

@app.get("/telemetry_history")
async def get_telemetry_history(limit: int = 50):
    """
    Returns a list of recent telemetry data points from history.
    Useful for agents that need to analyze trends over time.
    """
    if not telemetry_history:
        return []
    return list(telemetry_history)[-limit:]



if __name__ == "__main__":
    # uvicorn backend_api:app --reload --port 8000
    #    --reload: Automatically reloads the server on code changes.
    #    --port 8000: Runs on port 8000.
    
    print("Starting FastAPI server...")
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=True)