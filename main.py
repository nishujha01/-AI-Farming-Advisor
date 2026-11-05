"""
FastAPI Application — REST endpoints for the Smart Farming AI Agent
"""
import logging
import sys
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import get_settings

# Lazy imports — loaded only when first request arrives so the server
# starts immediately even if heavy ML packages are still installing.
_agent = None
_lang_processor = None
_LANGUAGE_NAMES = None

def _get_agent():
    global _agent
    if _agent is None:
        from agent import get_farming_agent
        _agent = get_farming_agent()
    return _agent

def _get_lang():
    global _lang_processor, _LANGUAGE_NAMES
    if _lang_processor is None:
        from language_processor import get_language_processor, LANGUAGE_NAMES as LN
        _lang_processor = get_language_processor()
        _LANGUAGE_NAMES = LN
    return _lang_processor, _LANGUAGE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
settings = get_settings()


# -----------------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Farmer's question")
    language: Optional[str] = Field(None, description="Language override (ISO 639-1 code, e.g. 'hi')")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Farm latitude for weather")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Farm longitude for weather")


class QueryResponse(BaseModel):
    query: str
    detected_language: str
    language_name: str
    intents: list
    answer: str
    answer_english: Optional[str] = None
    sources_used: list


class IngestRequest(BaseModel):
    docs_dir: str = Field(default="./data/knowledge_base")


class HealthResponse(BaseModel):
    status: str
    app_name: str
    model: str
    version: str = "1.0.0"


# -----------------------------------------------------------------------
# App lifecycle
# -----------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting %s", settings.app_name)
    logger.info("IBM Granite Model: %s", settings.granite_model_id)
    logger.info("=" * 60)

    # Auto-ingest knowledge base on startup (non-fatal if ML deps missing)
    try:
        agent = _get_agent()
        chunks = agent.ingest_knowledge_base()
        logger.info("Knowledge base ingested: %d chunks", chunks)
    except Exception as exc:
        logger.warning("Knowledge base ingestion skipped (run 'pip install -r requirements.txt' to enable): %s", exc)

    yield
    logger.info("%s shutting down.", settings.app_name)


# -----------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    description=(
        "AI-powered agricultural advisor for small-scale farmers. "
        "Powered by IBM Granite (watsonx.ai) and RAG technology."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
import os
if os.path.exists("./static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the farmer-facing frontend."""
    try:
        with open("./static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Smart Farming AI Agent</h1><p>API running. Visit <a href='/docs'>/docs</a></p>")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        model=settings.granite_model_id,
    )


@app.get("/languages")
async def list_languages():
    """List supported languages."""
    _, lang_names = _get_lang()
    return {
        "supported_languages": [
            {"code": code, "name": name}
            for code, name in lang_names.items()
        ]
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Main endpoint — ask the farming agent a question.
    Supports natural language in any of the supported regional languages.
    """
    try:
        agent = _get_agent()
        result = await agent.answer(
            raw_query=request.query,
            language_override=request.language,
            latitude=request.latitude,
            longitude=request.longitude,
        )
        return QueryResponse(**result)
    except ValueError as exc:
        # Configuration error (missing API keys etc.)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(exc)}")


@app.post("/ingest")
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger knowledge base document ingestion (runs in background).
    """
    def _ingest():
        agent = _get_agent()
        chunks = agent.ingest_knowledge_base(request.docs_dir)
        logger.info("Background ingestion complete: %d chunks", chunks)

    background_tasks.add_task(_ingest)
    return {"message": "Knowledge base ingestion started in background.", "docs_dir": request.docs_dir}


@app.get("/weather")
async def get_weather(lat: float = 20.59, lon: float = 78.96, days: int = 7):
    """Get weather forecast for given coordinates."""
    try:
      from farming_tools import WeatherTool
      tool = WeatherTool()
      data = await tool.get_forecast(lat, lon, days)
      return data
    except ImportError as e:
      raise HTTPException(status_code=503, detail=f"Module not ready: {e}. Run: pip install -r requirements.txt")


@app.get("/market-prices")
async def get_market_prices(commodity: str = "tomato", state: Optional[str] = None):
    """Get current mandi prices for a commodity."""
    try:
      from farming_tools import MarketPriceTool
      tool = MarketPriceTool()
      data = await tool.get_prices(commodity, state)
      return data
    except ImportError as e:
      raise HTTPException(status_code=503, detail=f"Module not ready: {e}. Run: pip install -r requirements.txt")


@app.get("/crop-calendar")
async def get_crop_calendar(month: Optional[int] = None):
    """Get seasonal crop recommendations for current or specified month."""
    try:
      from farming_tools import CropAdvisoryTool
      tool = CropAdvisoryTool()
      return tool.get_season_recommendation(month)
    except ImportError as e:
      raise HTTPException(status_code=503, detail=f"Module not ready: {e}. Run: pip install -r requirements.txt")


@app.get("/schemes")
async def get_schemes(query: Optional[str] = None):
    """List government agricultural schemes, optionally filtered by query."""
    try:
      from farming_tools import GovernmentSchemeTool
      tool = GovernmentSchemeTool()
      if query:
          return {"schemes": tool.search_scheme(query)}
      return {"schemes": tool.get_all_schemes()}
    except ImportError as e:
      raise HTTPException(status_code=503, detail=f"Module not ready: {e}. Run: pip install -r requirements.txt")
