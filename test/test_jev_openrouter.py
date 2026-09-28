import os
import requests

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not configured.")

url = "https://openrouter.ai/api/alpha/decisions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "typesafe/jev-1.13",
    "state": {
        "nearest_obstacle_m": 4.2,
        "left_clearance_m": 5.8,
        "right_clearance_m": 1.4,
        "forward_clearance_m": 4.2,
        "speed_mps": 2.0,
        "road_clear": True,
    },
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "Which vehicle action should be selected based only on the supplied state?",
            "criteria": {
                "forward": "Forward movement is safe and useful.",
                "left": "Turning left is the safest useful action.",
                "right": "Turning right is the safest useful action.",
                "stop": "Stopping is the safest appropriate action.",
            },
        }
    },
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=30,
)

print("HTTP status:", response.status_code)

if not response.ok:
    print("Response:")
    print(response.text)
    response.raise_for_status()

data = response.json()

print("\nJev API: PASS")
print("Model:", data.get("model"))

answer = data["answers"]["action"]

print("Action:", answer["choice"])
print("Confidence:", answer["confidence"])
print("Probabilities:", answer["probabilities"])

if "usage" in data:
    print("Usage:", data["usage"])