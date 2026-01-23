import os
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import uvicorn
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain / tools (your original setup) ---------------------------------
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch

# Tavily - A research-oriented search API for AI agents

# --- Masumi SDK (per your example) --------------------------------------------
from masumi.config import Config
from masumi.payment import Payment, Amount

# --- Local crew/agent code (if you need CrewAI) --------------------------------
# from crew_definition import ResearchCrew  
# optional: used if you use CrewAI

# Load env
load_dotenv(override=False)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("masumi_agent")

# Environment config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_IDENTIFIER = os.getenv("AGENT_IDENTIFIER")
PAYMENT_SERVICE_URL = os.getenv("PAYMENT_SERVICE_URL")
PAYMENT_API_KEY = os.getenv("PAYMENT_API_KEY")
NETWORK = os.getenv("NETWORK", "Preprod")
PAYMENT_AMOUNT = os.getenv("PAYMENT_AMOUNT", "10000000")
PAYMENT_UNIT = os.getenv("PAYMENT_UNIT", "lovelace")
SELLER_VKEY = os.getenv("SELLER_VKEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

if not AGENT_IDENTIFIER:
    logger.warning("AGENT_IDENTIFIER is not set. Set it before registering with Masumi.")

# -----------------------------------------------------------------------------
# Your original LangChain agent setup (adapted)
# -----------------------------------------------------------------------------

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
    HAVE_ZONEINFO = True
except Exception:
    HAVE_ZONEINFO = False
    import pytz

# Initialize LLM
llm = ChatOpenAI(temperature=0.4, model_name="gpt-4o-mini", max_tokens=500)

# Weather tool (kept as-is)
@tool
def get_weather_update(city_name: str) -> str:
    "Using this tool get the weather update for a given city."
    if not WEATHERSTACK_API_KEY:
        return "❌ Weatherstack API key is missing. Please set WEATHERSTACK_API_KEY in your .env file."
    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city_name}"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        print("Data is here:", data)
        if "error" in data:
            return f"⚠️ Error: {data['error'].get('info', 'Unknown error')}"
        loc = data["location"]["name"]
        temperature = data["current"]["temperature"]
        description = data["current"]["weather_descriptions"][0]
        humidity = data["current"]["humidity"]
        feels_like = data["current"]["feelslike"]
        return (f"🌍 Weather in {loc}:\n- {description}\n- Temperature: {temperature}°C (Feels like {feels_like}°C)\n- Humidity: {humidity}%")
    except Exception as e:
        return f"❌ Failed to fetch weather data: {e}"

# Current date tool
def now_in_tz(tz_name: str = "Asia/Kolkata") -> datetime:
    if HAVE_ZONEINFO:
        return datetime.now(ZoneInfo(tz_name))
    else:
        return datetime.now(pytz.timezone(tz_name))

@tool
def get_current_date(timezone_str: str = "Asia/Kolkata") -> str:
    "Using this tool get the current date and time in a specified timezone."
    try:
        now = now_in_tz(timezone_str)
        return now.strftime("%A, %B %d, %Y. The time is %I:%M %p (%Z)")
    except Exception:
        now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        return now.strftime("%A, %B %d, %Y. The time is %I:%M %p (IST)")

# Search tools
# DuckDuckGoSearch - A privacy-focused general web search engine
search = DuckDuckGoSearchRun()

tavily_search_tool = TavilySearch(max_results=5, topic="general")

tools = [tavily_search_tool, get_weather_update, get_current_date]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant to help someone who wants to know about weather details and any general information. "
        "Use the tools to provide accurate information."
    ),
)

# -----------------------------------------------------------------------------
# Masumi payment config
# -----------------------------------------------------------------------------
masumi_config = Config(payment_service_url=PAYMENT_SERVICE_URL, payment_api_key=PAYMENT_API_KEY)

# In-memory job store (for demo only)
jobs: Dict[str, Dict[str, Any]] = {}
payment_instances: Dict[str, Payment] = {}

# FastAPI app
app = FastAPI(title="LangChain + Masumi Agent API")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class AskRequest(BaseModel):
    message: str
    timezone: Optional[str] = None

class StartJobRequest(BaseModel):
    identifier_from_purchaser: str
    input_data: dict

class ProvideInputRequest(BaseModel):
    job_id: str

# -----------------------------------------------------------------------------
# Helper: run your (blocking) agent.invoke in the threadpool
# -----------------------------------------------------------------------------
async def invoke_agent(payload: dict) -> dict:
    """Run agent.invoke in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(agent.invoke, payload)

# -----------------------------------------------------------------------------
# Routes: existing endpoints from your app
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
async def ask_agent(req: AskRequest):
    try:
        payload = {"messages": [{"role": "user", "content": req.message}]}
        result = await invoke_agent(payload)
        messages = result.get("messages") or []
        if messages:
            assistant_msg = messages[-1].content
            return {"reply": assistant_msg}
        return {"reply": str(result)}
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

# -----------------------------------------------------------------------------
# Masumi MIP-003 endpoints
# -----------------------------------------------------------------------------
@app.post("/start_job")
async def start_job(data: StartJobRequest):
    """Initiates a job and creates a Masumi payment request."""
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Starting job {job_id} for purchaser {data.identifier_from_purchaser}")

        # Prepare payment amounts
        amounts = [Amount(amount=PAYMENT_AMOUNT, unit=PAYMENT_UNIT)]

        payment = Payment(
            agent_identifier=AGENT_IDENTIFIER,
            config=masumi_config,
            identifier_from_purchaser=data.identifier_from_purchaser,
            input_data=data.input_data,
            network=NETWORK,
        )

        logger.info("Creating payment request with Masumi...")
        payment_request = await payment.create_payment_request()
        blockchain_identifier = payment_request["data"]["blockchainIdentifier"]
        payment.payment_ids.add(blockchain_identifier)

        # Persist job
        jobs[job_id] = {
            "status": "awaiting_payment",
            "payment_status": "pending",
            "blockchain_identifier": blockchain_identifier,
            "input_data": data.input_data,
            "result": None,
            "identifier_from_purchaser": data.identifier_from_purchaser,
        }

        # start monitoring (callback will be called on completion)
        payment_instances[job_id] = payment

        async def payment_callback(blockchain_identifier: str, job_id_inner=job_id):
            await handle_payment_status(job_id_inner, blockchain_identifier)

        # start_status_monitoring likely runs an internal loop; it may be async
        await payment.start_status_monitoring(payment_callback)

        return {
            "status": "success",
            "job_id": job_id,
            "blockchainIdentifier": blockchain_identifier,
            "submitResultTime": payment_request["data"].get("submitResultTime"),
            "unlockTime": payment_request["data"].get("unlockTime"),
            "externalDisputeUnlockTime": payment_request["data"].get("externalDisputeUnlockTime"),
            "agentIdentifier": AGENT_IDENTIFIER,
            "sellerVKey": SELLER_VKEY,
            "identifierFromPurchaser": data.identifier_from_purchaser,
            "amounts": amounts,
            "input_hash": getattr(payment, "input_hash", None),
            "payByTime": payment_request["data"].get("payByTime"),
        }

    except KeyError as e:
        logger.exception("Missing required field in start_job")
        raise HTTPException(status_code=400, detail=f"Bad Request: missing field {e}")
    except Exception as e:
        logger.exception("start_job error")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_payment_status(job_id: str, payment_id: str) -> None:
    """Called by Masumi SDK when payment is confirmed. Execute the agent job and complete payment."""
    try:
        logger.info(f"Payment {payment_id} completed for job {job_id}")
        job = jobs.get(job_id)
        if not job:
            logger.error(f"Unknown job {job_id}")
            return

        job["status"] = "running"

        # Execute the agent with the job's input_data
        # For compatibility we pass a messages payload like /ask
        user_text = job["input_data"].get("text") or str(job["input_data"]) 
        payload = {"messages": [{"role": "user", "content": user_text}]}

        result = await invoke_agent(payload)
        messages = result.get("messages") or []
        assistant_msg = messages[-1].content if messages else str(result)

        # Mark payment completed on Masumi (send result back)
        payment_instance = payment_instances.get(job_id)
        result_string = assistant_msg
        if payment_instance:
            try:
                await payment_instance.complete_payment(payment_id, result_string)
            except Exception:
                logger.exception("Failed to call complete_payment on masumi payment instance")

        job["status"] = "completed"
        job["payment_status"] = "completed"
        job["result"] = assistant_msg

        # stop monitoring
        if job_id in payment_instances:
            try:
                payment_instances[job_id].stop_status_monitoring()
            except Exception:
                logger.exception("Error stopping payment monitoring")
            del payment_instances[job_id]

        logger.info(f"Job {job_id} completed and payment finalized")

    except Exception as e:
        logger.exception("Error in handle_payment_status")
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
        if job_id in payment_instances:
            try:
                payment_instances[job_id].stop_status_monitoring()
            except Exception:
                pass
            del payment_instances[job_id]


@app.get("/status")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # try to update payment status from instance if available
    if job_id in payment_instances:
        try:
            status = await payment_instances[job_id].check_payment_status()
            job["payment_status"] = status.get("data", {}).get("status")
        except Exception:
            logger.exception("Error checking payment status")
            job["payment_status"] = job.get("payment_status", "unknown")

    result = job.get("result")
    return {
        "job_id": job_id,
        "status": job["status"],
        "payment_status": job["payment_status"],
        "result": result,
    }


@app.get("/availability")
async def check_availability():
    return {"status": "available", "type": "masumi-agent", "message": "Server operational."}


@app.get("/input_schema")
async def input_schema():
    return {
        "input_data": [
            {
                "id": "text",
                "type": "string",
                "name": "Task Description",
                "data": {
                    "description": "Search weather update of Kolkata for today.",
                    "placeholder": "Enter your task description here",
                },
            }
        ]
    }

# -----------------------------------------------------------------------------
# Run app
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 3000))
    host = os.environ.get("API_HOST", "0.0.0.0")
    logger.info(f"Starting Masumi Agent API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
