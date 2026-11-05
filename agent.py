"""
Core Farming Agent — Orchestrates RAG + Granite LLM + Live Tools
This is the main reasoning loop: detect intent -> gather context -> generate answer
"""
import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

_executor = ThreadPoolExecutor(max_workers=2)

from rag_pipeline import get_rag_pipeline
from granite_llm import get_granite_llm, build_prompt
from farming_tools import (
    WeatherTool,
    MarketPriceTool,
    CropAdvisoryTool,
    PestDiseaseTool,
    GovernmentSchemeTool,
)
from language_processor import get_language_processor

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Intent detection helpers
# -----------------------------------------------------------------------
INTENT_PATTERNS = {
    "weather": [
        r"\bweather\b", r"\brain\b", r"\bforecast\b", r"\bclimate\b",
        r"\btemperature\b", r"\bmonsoon\b", r"\bdrought\b",
        r"\bमौसम\b", r"\bबारिश\b", r"\bवर्षा\b",
    ],
    "market_price": [
        r"\bprice\b", r"\bmandi\b", r"\brate\b", r"\bmarket\b", r"\bsell\b",
        r"\bbhav\b", r"\bmssp\b", r"\bmssp\b",
        r"\bभाव\b", r"\bमंडी\b", r"\bदाम\b", r"\bकीमत\b",
    ],
    "crop_recommendation": [
        r"\bcrop\b", r"\bplant\b", r"\bsow\b", r"\bgrow\b", r"\bseed\b",
        r"\bseason\b", r"\bkharif\b", r"\brabi\b", r"\bzaid\b",
        r"\bफसल\b", r"\bबुवाई\b", r"\bखेती\b",
    ],
    "pest_disease": [
        r"\bpest\b", r"\bdisease\b", r"\binsect\b", r"\bfungus\b",
        r"\baphid\b", r"\bborer\b", r"\bblight\b", r"\bmildew\b",
        r"\bकीट\b", r"\bरोग\b", r"\bकीड़े\b",
    ],
    "soil": [
        r"\bsoil\b", r"\bfertilizer\b", r"\bcompost\b", r"\bnutrient\b",
        r"\bpH\b", r"\borganic\b", r"\birrigation\b", r"\bwater\b",
        r"\bमिट्टी\b", r"\bखाद\b", r"\bउर्वरक\b",
    ],
    "scheme": [
        r"\bscheme\b", r"\bsubsidy\b", r"\bgovernment\b", r"\bpm.kisan\b",
        r"\bkcc\b", r"\binsurance\b", r"\bloan\b",
        r"\bयोजना\b", r"\bसब्सिडी\b", r"\bसरकार\b",
    ],
}


def detect_intents(query: str) -> list:
    """Return a list of intents detected in the query (multiple possible)."""
    q_lower = query.lower()
    detected = []
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q_lower, re.IGNORECASE):
                if intent not in detected:
                    detected.append(intent)
                break
    return detected or ["general"]


def extract_location(query: str) -> Optional[str]:
    """Simple heuristic to extract a city/location name from the query."""
    location_keywords = ["in ", "at ", "near ", "for ", "ke liye ", "mein "]
    q_lower = query.lower()
    for kw in location_keywords:
        idx = q_lower.find(kw)
        if idx != -1:
            remainder = query[idx + len(kw):].strip()
            word = remainder.split()[0].rstrip("?.!,") if remainder else None
            if word and len(word) > 2:
                return word.capitalize()
    return None


def extract_commodity(query: str) -> str:
    """Extract commodity name from market price queries."""
    commodities = [
        "tomato", "potato", "onion", "wheat", "rice", "cotton", "sugarcane",
        "maize", "soybean", "groundnut", "mustard", "chilli", "garlic",
        "टमाटर", "आलू", "प्याज", "गेहूं", "चावल", "कपास", "मक्का",
    ]
    q_lower = query.lower()
    for c in commodities:
        if c in q_lower:
            return c
    # Return first content word as fallback
    words = [w for w in q_lower.split() if len(w) > 3]
    return words[0] if words else "tomato"


# -----------------------------------------------------------------------
# Default coordinates for common Indian cities
# -----------------------------------------------------------------------
CITY_COORDS = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "chennai": (13.0827, 80.2707),
    "kolkata": (22.5726, 88.3639),
    "bangalore": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "chandigarh": (30.7333, 76.7794),
    "patna": (25.5941, 85.1376),
    "bhopal": (23.2599, 77.4126),
    "nagpur": (21.1458, 79.0882),
    "amritsar": (31.6340, 74.8723),
}

DEFAULT_COORDS = (20.5937, 78.9629)  # Center of India


def get_coords_for_location(location: Optional[str]) -> tuple:
    if location:
        return CITY_COORDS.get(location.lower(), DEFAULT_COORDS)
    return DEFAULT_COORDS


# -----------------------------------------------------------------------
# Main Farming Agent
# -----------------------------------------------------------------------
class FarmingAgent:
    """
    The Smart Farming AI Agent.
    Pipeline: Query → Language Detection → Intent Detection →
              Tool Gathering → RAG Retrieval → Granite LLM → Response Translation
    """

    def __init__(self):
        self.rag = get_rag_pipeline()
        self.llm = get_granite_llm()
        self.lang_processor = get_language_processor()
        self.weather_tool = WeatherTool()
        self.market_tool = MarketPriceTool()
        self.crop_tool = CropAdvisoryTool()
        self.pest_tool = PestDiseaseTool()
        self.scheme_tool = GovernmentSchemeTool()

    async def answer(
        self,
        raw_query: str,
        language_override: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: takes a raw farmer query, returns structured answer.
        """
        # 1. Language detection + translation to English
        english_query, detected_lang = self.lang_processor.process_query(
            raw_query, override_lang=language_override
        )
        logger.info("Query [%s -> en]: %s", detected_lang, english_query[:100])

        # 2. Intent detection
        intents = detect_intents(english_query)
        logger.info("Detected intents: %s", intents)

        # 3. Gather live context from tools
        live_context_parts = []

        if "weather" in intents:
            location = extract_location(english_query)
            if latitude and longitude:
                coords = (latitude, longitude)
            else:
                coords = get_coords_for_location(location)
            weather_data = await self.weather_tool.get_forecast(*coords)
            live_context_parts.append(
                "LIVE WEATHER DATA:\n" + self.weather_tool.format_for_context(weather_data)
            )

        if "market_price" in intents:
            commodity = extract_commodity(english_query)
            market_data = await self.market_tool.get_prices(commodity)
            live_context_parts.append(
                "LIVE MARKET PRICES:\n" + self.market_tool.format_for_context(market_data)
            )

        if "crop_recommendation" in intents:
            season_info = self.crop_tool.get_season_recommendation()
            live_context_parts.append(
                "SEASONAL CROP CALENDAR:\n" + self.crop_tool.format_for_context(season_info)
            )

        if "pest_disease" in intents:
            pest_name = self._extract_pest_name(english_query)
            pest_info = self.pest_tool.get_pest_info(pest_name) if pest_name else None
            if pest_info:
                live_context_parts.append(
                    "PEST/DISEASE INFO:\n" + self.pest_tool.format_for_context(pest_info)
                )

        if "scheme" in intents:
            schemes = self.scheme_tool.search_scheme(english_query)
            if not schemes:
                schemes = self.scheme_tool.get_all_schemes()[:3]
            live_context_parts.append(
                "GOVERNMENT SCHEMES:\n" + self.scheme_tool.format_for_context(schemes)
            )

        # 4. RAG retrieval from knowledge base
        rag_docs = self.rag.retrieve(english_query)
        rag_context = self.rag.format_context(rag_docs)

        # 5. Combine all context
        full_context = ""
        if live_context_parts:
            full_context += "\n\n".join(live_context_parts) + "\n\n"
        full_context += "KNOWLEDGE BASE EXCERPTS:\n" + rag_context

        # 6. Build prompt and generate with Granite (run in thread — blocking call)
        prompt = build_prompt(question=english_query, context=full_context)
        loop = asyncio.get_event_loop()
        english_answer = await loop.run_in_executor(
            _executor, self.llm.generate, prompt
        )

        # 7. Translate answer back to farmer's language
        final_answer = self.lang_processor.process_response(english_answer, detected_lang)

        return {
            "query": raw_query,
            "detected_language": detected_lang,
            "language_name": self.lang_processor.get_language_name(detected_lang),
            "intents": intents,
            "answer": final_answer,
            "answer_english": english_answer if detected_lang != "en" else None,
            "sources_used": [doc.metadata.get("source", "knowledge base") for doc in rag_docs],
        }

    def _extract_pest_name(self, query: str) -> Optional[str]:
        all_pests = self.pest_tool.get_all_pests()
        q_lower = query.lower()
        for pest in all_pests:
            if pest in q_lower:
                return pest
        return None

    def ingest_knowledge_base(self, docs_dir: str = "./data/knowledge_base") -> int:
        """Trigger ingestion of all knowledge base documents."""
        return self.rag.ingest_documents(docs_dir)


# Module-level singleton
_agent: Optional[FarmingAgent] = None


def get_farming_agent() -> FarmingAgent:
    global _agent
    if _agent is None:
        _agent = FarmingAgent()
    return _agent
