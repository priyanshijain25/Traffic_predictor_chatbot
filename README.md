# Traffic_predictor_chatbot

## Overview
Traffic Intelligence Multi-Agent System is a LangGraph-based AI application that predicts traffic conditions by combining traffic, weather, and event data from multiple external APIs.
The system uses multiple specialized agents to gather information, reason over the collected context, estimate traffic impact, and generate human-readable explanations.

## Architecture
<img width="1550" height="1518" alt="diagram-export-20-06-2026-04_50_25" src="https://github.com/user-attachments/assets/b0f760e7-43f0-4ce4-9308-7e5ad1215a24" />

## Chatbot Demo
<img width="1280" height="946" alt="image" src="https://github.com/user-attachments/assets/77f86760-ee5b-4a79-a04e-b153e7a9d113" />

## Features
* Multi-Agent Architecture using LangGraph
* LLM-powered Planner Agent using Groq
* Real-time Traffic Data from TomTom API
* Weather Insights from OpenWeather API
* Event Detection from Ticketmaster API
* Context Fusion across multiple data sources
* Traffic Congestion Prediction
* AI-generated Explanations
* Terminal-based Chatbot Interface

  
## Tech Stack

Frameworks
* LangGraph
* LangChain
* Python

LLM
* Groq
* Llama 3.3 70B Versatile

APIs
* TomTom Traffic API
* OpenWeather API
* Ticketmaster API

## Requirements
- Python 3.10+
- Groq API Key
- OpenWeather API Key
- TomTom API Key
- Ticketmaster API Key

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/priyanshijain25/Traffic_predictor_chatbot.git
cd Traffic_predictor_chatbot
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

#### macOS/Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
OPENWEATHER_API_KEY=your_openweather_api_key
TOMTOM_API_KEY=your_tomtom_api_key
TICKETMASTER_API_KEY=your_ticketmaster_api_key
GROQ_API_KEY=your_groq_api_key
```

### 6. Run the Application

```bash
python3 main.py
```

---

## Example Usage

```text
You: Will traffic worsen in London tomorrow?

Prediction:
Moderate traffic congestion expected

Explanation:
Traffic may worsen due to rainy weather and increased city events causing higher road usage.
```
