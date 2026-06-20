from typing import TypedDict
from langgraph.graph import StateGraph , START , END
import os 
import requests
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import time

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
API_KEY = os.getenv("TICKETMASTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

model="llama-3.3-70b-versatile"

)

#shared state
class TrafficState(TypedDict):
    query: str

    city: str
    intent: str
    time_reference: str

    traffic_data: dict
    weather_data: dict
    event_data: dict

    fused_context: dict
    prediction: dict
    explanation: str
    final_response: str

    routes: list

#all agents 
def planner_agent(state: TrafficState) -> TrafficState:
    query = state["query"]

    prompt = f"""
You are a planner agent.

Return ONLY valid JSON (no explanation, no markdown).

Format:
{{
  "city": "",
  "intent": "traffic_analysis | traffic_prediction | general_query",
  "time_reference": "now | today | tomorrow | tonight"
}}

Query: {query}
"""
    response = llm.invoke(prompt)
    content = response.content
    #print("RAW LLM OUTPUT:\n", content) 
    content = content.strip()

    if "```" in content:
        content = content.replace("```json", "").replace("```", "")

    try:
        result = json.loads(content)
    except Exception as e:
        print("JSON PARSE FAILED:", e)

        result = {
            "city": "London",
            "intent": "traffic_prediction",
            "time_reference": "tomorrow"
        }

    state["city"] = result["city"]
    state["intent"] = result["intent"]
    state["time_reference"] = result["time_reference"]

    return {
    "city": result["city"],
    "intent": result["intent"],
    "time_reference": result["time_reference"]
    }


def router_agent(state: TrafficState) -> TrafficState:

    state["routes"] = ["traffic_agent", "weather_agent", "event_agent"]

    return {
    "routes": ["traffic_agent", "weather_agent", "event_agent"]
    }


def traffic_agent(state: TrafficState) -> TrafficState:

    lat = 51.5074
    lon = -0.1278

    url = (
        f"https://api.tomtom.com/traffic/services/4/"
        f"flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}"
        f"&key={TOMTOM_API_KEY}"
    )

    response = requests.get(url)

    data = response.json()

    flow = data["flowSegmentData"]

    congestion_ratio = (
    flow["currentSpeed"] /
    flow["freeFlowSpeed"]
    )

    if congestion_ratio < 0.5:
        congestion_level = "high"
    elif congestion_ratio < 0.8:
        congestion_level = "medium"
    else:
        congestion_level = "low"

    state["traffic_data"] = {
        "current_speed": flow["currentSpeed"],
        "free_flow_speed": flow["freeFlowSpeed"],
        "congestion_level": congestion_level
    }
    return {
    "traffic_data": {
        "current_speed": flow["currentSpeed"],
        "free_flow_speed": flow["freeFlowSpeed"],
        "congestion_level": congestion_level
    }
    }


def weather_agent(state: TrafficState) -> TrafficState:

    city = state["city"]

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}"
        f"&appid={OPENWEATHER_API_KEY}"
        f"&units=metric"
    )

    response = requests.get(url)

    data = response.json()

    weather_type = data["weather"][0]["main"].lower()

    is_raining = weather_type in [
        "rain",
        "drizzle",
        "thunderstorm"
    ]

    state["weather_data"] = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "weather": weather_type,
        "is_raining": is_raining,
        "wind_speed": data["wind"]["speed"]
    }

    return {
    "weather_data": {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "weather": weather_type,
        "is_raining": is_raining,
        "wind_speed": data["wind"]["speed"]
    }
    }


def event_agent(state: TrafficState) -> TrafficState:

    city = state["city"]

    url = (
        f"https://app.ticketmaster.com/discovery/v2/events.json"
        f"?city={city}"
        f"&apikey={API_KEY}"
    )

    response = requests.get(url)

    data = response.json()

    events = data.get("_embedded", {}).get("events", [])

    event_names = []

    for event in events[:5]:
        event_names.append(event["name"])

    major_event_day = len(events) > 10
    state["event_data"] = {
        "event_count": len(events),
        "major_events": event_names,
        "major_event_day": major_event_day
    }

    return {
    "event_data": {
        "event_count": len(events),
        "major_events": event_names,
        "major_event_day": major_event_day
    }
    }


def predictor_agent(state: TrafficState) -> TrafficState:

    #fused = state.get("fused_context", {})
    traffic = state.get("traffic_data", {})
    weather = state.get("weather_data", {})
    events = state.get("event_data", {})
    

    risk_score = 0

    traffic_level = traffic.get("congestion_level", "low")

    if traffic_level == "high":
        risk_score += 3
    elif traffic_level == "medium":
        risk_score += 2
    else:
        risk_score += 1

    if weather.get("is_raining", False):
        risk_score += 2

    if events.get("major_event_day", False):
        risk_score += 3

    if risk_score >= 7:
        prediction = "Severe traffic congestion expected"
    elif risk_score >= 4:
        prediction = "Moderate traffic congestion expected"
    else:
        prediction = "Normal traffic flow expected"

    state["prediction"] = {
        "risk_score": risk_score,
        "prediction": prediction
    }

    return {
    "prediction": {
        "risk_score": risk_score,
        "prediction": prediction
    }
    }


def explainability_agent(state: TrafficState) -> TrafficState:

    traffic = state.get("traffic_data", {})
    weather = state.get("weather_data", {})
    events = state.get("event_data", {})
    prediction = state.get("prediction", {})

    prompt = f"""
You are a traffic explainability agent.

Explain the prediction in simple human language.

Use the following information:

FUSED CONTEXT:
{weather,events,traffic}

PREDICTION:
{prediction}

Rules:
- Keep it short (3–5 lines)
- Do not use technical terms like risk_score
- Explain like you are talking to a normal user
- Mention traffic, weather, and events impact
"""

    response = llm.invoke(prompt)

    state["explanation"] = response.content

    return {
    "explanation": response.content
    }



#defining langgraph nodes
graph = StateGraph(TrafficState)

graph.add_node("planner", planner_agent)
graph.add_node("router", router_agent)
graph.add_node("traffic", traffic_agent)
graph.add_node("weather", weather_agent)
graph.add_node("event", event_agent)
graph.add_node("predictor", predictor_agent)
graph.add_node("explainability", explainability_agent)

graph.add_edge(START, "planner")
graph.add_edge("planner", "router")

def route_decision(state: TrafficState):
    return state["routes"]

graph.add_conditional_edges(
    "router",
    route_decision,
    {
        "traffic_agent": "traffic",
        "weather_agent": "weather",
        "event_agent": "event"
    }
)
#connecting all to fusion
graph.add_edge("traffic", "predictor")
graph.add_edge("weather", "predictor")
graph.add_edge("event", "predictor")

graph.add_edge("predictor", "explainability")
graph.add_edge("explainability", END)

app = graph.compile()

#print("\nGRAPH STRUCTURE:\n")
print(app.get_graph().draw_ascii())

print("\n🚗 Traffic Intelligence Chatbot")
print("Type 'exit' to quit.\n")

while True:

    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        start_time = time.perf_counter()

        result = app.invoke({
            "query": user_query
        })

        end_time = time.perf_counter()

        print("\n🤖 Traffic Intelligence Report:")

        prediction = result.get("prediction", {})
        print("\nPrediction:")
        print(prediction.get("prediction"))

        print("\nExplanation:")
        print(result.get("explanation", "No explanation generated"))
        print(f"\nLatency: {end_time - start_time:.3f} seconds")

        print("\n" + "-" * 50 + "\n")

    except Exception as e:
        print(f"\nError: {e}\n")

