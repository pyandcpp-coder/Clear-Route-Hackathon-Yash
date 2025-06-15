import httpx
import json
BACKEND_API_BASE_URL = "http://localhost:8000" 


async def get_latest_telemetry() -> dict:
    """
    Fetches the latest single telemetry data point from the backend.
    This tool provides a snapshot of the current car, environmental,
    and competitor conditions.
    """
    url = f"{BACKEND_API_BASE_URL}/live_data"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5.0)
            response.raise_for_status() 
            data = response.json()
            return data
        except httpx.RequestError as exc:
            print(f"[Tool Error] An error occurred while requesting {exc.request.url!r}: {exc}")
            return {"error": f"Failed to fetch latest telemetry: {exc}"}
        except json.JSONDecodeError:
            print(f"[Tool Error] Failed to decode JSON from response for {url}")
            return {"error": "Invalid JSON response from telemetry endpoint"}

async def get_telemetry_history(limit: int = 50) -> list[dict]:
    """
    Fetches a list of recent telemetry data points from the backend history.
    This tool is useful for analyzing trends over time (e.g., for anomaly detection).

    Args:
        limit (int): The maximum number of historical data points to retrieve.
                     Defaults to 50.
    Returns:
        list[dict]: A list of telemetry data points, oldest first.
    """
    url = f"{BACKEND_API_BASE_URL}/telemetry_history?limit={limit}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return data
        except httpx.RequestError as exc:
            print(f"[Tool Error] An error occurred while requesting {exc.request.url!r}: {exc}")
            return {"error": f"Failed to fetch telemetry history: {exc}"}
        except json.JSONDecodeError:
            print(f"[Tool Error] Failed to decode JSON from response for {url}")
            return {"error": "Invalid JSON response from telemetry history endpoint"}

if __name__ == "__main__":
    import asyncio
    async def test_tools_direct():
        print("Testing get_latest_telemetry directly from clearroute_agent_tools.py...")
        latest = await get_latest_telemetry()
        print(f"Latest data keys: {list(latest.keys()) if isinstance(latest, dict) else latest}")

        print("\nTesting get_telemetry_history directly...")
        history = await get_telemetry_history(limit=5)
        print(f"History count: {len(history)}")
        if history:
            print(f"First history item keys: {list(history[0].keys())}")

    print("Ensure backend_api.py is running before running this test.")
    asyncio.run(test_tools_direct())