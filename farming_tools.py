"""
Farming Tools — Live data fetchers for weather, market prices, and soil info.
Each tool returns structured data that is serialized and injected into the RAG context.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import httpx

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# -----------------------------------------------------------------------
# Weather Tool (Open-Meteo — free, no API key required)
# -----------------------------------------------------------------------
class WeatherTool:
    """Fetch 7-day weather forecast for given lat/lon coordinates."""

    WMO_CODES = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail",
    }

    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7,
    ) -> Dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max",
            "current_weather": "true",
            "forecast_days": days,
            "timezone": "auto",
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(settings.open_meteo_url, params=params)
                resp.raise_for_status()
                data = resp.json()
            return self._parse_forecast(data)
        except Exception as exc:
            logger.error("Weather API error: %s", exc)
            return {"error": str(exc), "available": False}

    def _parse_forecast(self, raw: Dict) -> Dict[str, Any]:
        current = raw.get("current_weather", {})
        daily = raw.get("daily", {})
        dates = daily.get("time", [])

        forecast_days = []
        for i, date in enumerate(dates):
            code = daily.get("weathercode", [0] * 7)[i]
            forecast_days.append({
                "date": date,
                "condition": self.WMO_CODES.get(code, "Unknown"),
                "max_temp_c": daily.get("temperature_2m_max", [])[i],
                "min_temp_c": daily.get("temperature_2m_min", [])[i],
                "rain_mm": daily.get("precipitation_sum", [])[i],
                "max_wind_kmh": daily.get("windspeed_10m_max", [])[i],
            })

        return {
            "available": True,
            "current": {
                "temperature_c": current.get("temperature"),
                "windspeed_kmh": current.get("windspeed"),
                "condition": self.WMO_CODES.get(current.get("weathercode", 0), "Unknown"),
            },
            "forecast": forecast_days,
            "retrieved_at": datetime.utcnow().isoformat(),
        }

    def format_for_context(self, weather: Dict) -> str:
        if not weather.get("available"):
            return "Weather data unavailable."
        lines = [
            f"Current weather: {weather['current']['condition']}, "
            f"{weather['current']['temperature_c']}°C, "
            f"Wind {weather['current']['windspeed_kmh']} km/h",
            "",
            "7-Day Forecast:",
        ]
        for d in weather.get("forecast", []):
            lines.append(
                f"  {d['date']}: {d['condition']}, "
                f"{d['min_temp_c']}–{d['max_temp_c']}°C, "
                f"Rain: {d['rain_mm']} mm"
            )
        return "\n".join(lines)


# -----------------------------------------------------------------------
# Market Price Tool (data.gov.in / Agmarknet)
# -----------------------------------------------------------------------
class MarketPriceTool:
    """Fetch current mandi (market) prices for agricultural commodities."""

    async def get_prices(
        self,
        commodity: str,
        state: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        if not settings.agmarknet_api_key or settings.agmarknet_api_key == "your_agmarknet_api_key_here":
            return self._fallback_prices(commodity)

        params: Dict[str, Any] = {
            "api-key": settings.agmarknet_api_key,
            "format": "json",
            "filters[commodity]": commodity,
            "limit": limit,
        }
        if state:
            params["filters[state]"] = state

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(settings.agmarknet_api_url, params=params)
                resp.raise_for_status()
                data = resp.json()
            return self._parse_prices(data, commodity)
        except Exception as exc:
            logger.error("Market price API error: %s", exc)
            return self._fallback_prices(commodity)

    def _parse_prices(self, raw: Dict, commodity: str) -> Dict[str, Any]:
        records = raw.get("records", [])
        prices = []
        for r in records:
            prices.append({
                "state": r.get("state", ""),
                "district": r.get("district", ""),
                "market": r.get("market", ""),
                "commodity": r.get("commodity", commodity),
                "variety": r.get("variety", ""),
                "min_price": r.get("min_price", "N/A"),
                "max_price": r.get("max_price", "N/A"),
                "modal_price": r.get("modal_price", "N/A"),
                "date": r.get("arrival_date", ""),
            })
        return {"available": True, "commodity": commodity, "prices": prices}

    def _fallback_prices(self, commodity: str) -> Dict[str, Any]:
        """Return sample data when live API is unavailable."""
        sample = {
            "tomato":    {"min": 800,  "max": 1800, "modal": 1200},
            "potato":    {"min": 600,  "max": 1200, "modal": 900},
            "onion":     {"min": 700,  "max": 2000, "modal": 1400},
            "wheat":     {"min": 2100, "max": 2350, "modal": 2200},
            "rice":      {"min": 1800, "max": 2200, "modal": 2000},
            "cotton":    {"min": 5500, "max": 6800, "modal": 6200},
            "sugarcane": {"min": 290,  "max": 320,  "modal": 305},
            "maize":     {"min": 1500, "max": 1900, "modal": 1700},
            "soybean":   {"min": 3800, "max": 4500, "modal": 4100},
        }
        key = commodity.lower()
        price = sample.get(key, {"min": "N/A", "max": "N/A", "modal": "N/A"})
        return {
            "available": True,
            "commodity": commodity,
            "note": "Sample reference prices (₹/quintal). Check local mandi for actual rates.",
            "prices": [
                {
                    "market": "Reference",
                    "commodity": commodity,
                    "min_price": price["min"],
                    "max_price": price["max"],
                    "modal_price": price["modal"],
                }
            ],
        }

    def format_for_context(self, market_data: Dict) -> str:
        if not market_data.get("available"):
            return "Market price data unavailable."
        commodity = market_data.get("commodity", "")
        note = market_data.get("note", "")
        lines = [f"Market prices for {commodity} (₹/quintal):"]
        if note:
            lines.append(f"Note: {note}")
        for p in market_data.get("prices", []):
            location = p.get("market") or p.get("district") or "Market"
            lines.append(
                f"  {location}: Min ₹{p['min_price']} | "
                f"Max ₹{p['max_price']} | Modal ₹{p['modal_price']}"
            )
        return "\n".join(lines)


# -----------------------------------------------------------------------
# Crop Advisory Tool (rule-based + knowledge base lookup)
# -----------------------------------------------------------------------
class CropAdvisoryTool:
    """Provides season- and soil-based crop recommendations."""

    CROP_CALENDAR = {
        "kharif": {
            "months": [6, 7, 8, 9],
            "sow": "June–July",
            "harvest": "October–November",
            "crops": ["Rice", "Maize", "Sorghum", "Pearl Millet", "Cotton",
                      "Soybean", "Groundnut", "Sugarcane", "Turmeric", "Ginger"],
        },
        "rabi": {
            "months": [10, 11, 12, 1, 2, 3],
            "sow": "October–November",
            "harvest": "March–April",
            "crops": ["Wheat", "Barley", "Chickpea", "Mustard", "Lentil",
                      "Peas", "Potato", "Tomato", "Onion", "Garlic"],
        },
        "zaid": {
            "months": [3, 4, 5, 6],
            "sow": "March–April",
            "harvest": "June–July",
            "crops": ["Watermelon", "Muskmelon", "Cucumber", "Pumpkin",
                      "Moong", "Urad", "Sunflower", "Bitter Gourd"],
        },
    }

    SOIL_CROPS = {
        "sandy":    ["Groundnut", "Watermelon", "Carrot", "Sweet Potato", "Cowpea"],
        "clay":     ["Rice", "Wheat", "Cotton", "Sugarcane", "Mustard"],
        "loam":     ["Maize", "Soybean", "Potato", "Tomato", "Vegetables"],
        "black":    ["Cotton", "Sorghum", "Sunflower", "Safflower", "Wheat"],
        "red":      ["Groundnut", "Pulses", "Ragi", "Cotton", "Tobacco"],
        "alluvial": ["Wheat", "Rice", "Sugarcane", "Maize", "Vegetables"],
        "laterite": ["Cashew", "Coconut", "Tea", "Coffee", "Rubber"],
    }

    def get_season_recommendation(self, month: Optional[int] = None) -> Dict[str, Any]:
        m = month or datetime.now().month
        for season, info in self.CROP_CALENDAR.items():
            if m in info["months"]:
                return {
                    "season": season.capitalize(),
                    "sowing_time": info["sow"],
                    "harvest_time": info["harvest"],
                    "recommended_crops": info["crops"],
                    "current_month": datetime(2024, m, 1).strftime("%B"),
                }
        return {"season": "Transition", "recommended_crops": [], "current_month": str(m)}

    def get_soil_recommendation(self, soil_type: str) -> Dict[str, Any]:
        key = soil_type.lower().split()[0]
        crops = self.SOIL_CROPS.get(key, [])
        return {
            "soil_type": soil_type,
            "suitable_crops": crops,
            "note": "Verify with local agricultural extension officer for your specific field.",
        }

    def format_for_context(self, season: Dict, soil: Optional[Dict] = None) -> str:
        lines = [
            f"Current Season: {season.get('season')} "
            f"(Month: {season.get('current_month')})",
            f"Sowing Window: {season.get('sowing_time', 'N/A')}",
            f"Harvest Window: {season.get('harvest_time', 'N/A')}",
            f"Recommended Crops: {', '.join(season.get('recommended_crops', []))}",
        ]
        if soil and soil.get("suitable_crops"):
            lines += [
                "",
                f"Best crops for {soil['soil_type']} soil: "
                f"{', '.join(soil['suitable_crops'])}",
            ]
        return "\n".join(lines)


# -----------------------------------------------------------------------
# Pest & Disease Tool
# -----------------------------------------------------------------------
class PestDiseaseTool:
    """Static knowledge base of common crop pests and diseases with control measures."""

    PEST_DB = {
        "aphid": {
            "affected_crops": ["Wheat", "Mustard", "Cotton", "Vegetables"],
            "symptoms": "Yellowing leaves, sticky honeydew, curled leaves",
            "organic_control": "Neem oil spray (5ml/L), soap water spray, ladybug introduction",
            "chemical_control": "Imidacloprid 17.8% SL @ 0.5ml/L or Thiamethoxam 25% WG",
            "prevention": "Avoid over-fertilizing with nitrogen; inspect undersides of leaves weekly",
        },
        "whitefly": {
            "affected_crops": ["Tomato", "Cotton", "Chilli", "Brinjal"],
            "symptoms": "Yellowing, stunted growth, sooty mold on leaves",
            "organic_control": "Yellow sticky traps, neem oil 2%, reflective mulch",
            "chemical_control": "Acetamiprid 20% SP @ 0.2g/L or Spiromesifen 240 SC",
            "prevention": "Remove crop debris, maintain field hygiene",
        },
        "stem borer": {
            "affected_crops": ["Rice", "Maize", "Sugarcane"],
            "symptoms": "Dead heart in vegetative stage, white ear in reproductive stage",
            "organic_control": "Trichogramma egg parasitoid release, pheromone traps",
            "chemical_control": "Chlorpyrifos 20 EC @ 2.5ml/L or Cartap hydrochloride 4G",
            "prevention": "Timely sowing, destroy crop stubble after harvest",
        },
        "blast": {
            "affected_crops": ["Rice", "Wheat"],
            "symptoms": "Diamond-shaped lesions with grey center, neck rot",
            "organic_control": "Seed treatment with Trichoderma, silicon-rich amendments",
            "chemical_control": "Tricyclazole 75% WP @ 0.6g/L or Isoprothiolane 40% EC",
            "prevention": "Resistant varieties, balanced nitrogen, avoid water stress",
        },
        "powdery mildew": {
            "affected_crops": ["Wheat", "Grapes", "Mango", "Cucurbits"],
            "symptoms": "White powdery patches on leaves and stems",
            "organic_control": "Baking soda spray (5g/L), sulfur dust, neem oil",
            "chemical_control": "Propiconazole 25% EC @ 1ml/L or Hexaconazole 5% SC",
            "prevention": "Proper spacing for airflow, remove infected plant parts",
        },
        "leaf curl": {
            "affected_crops": ["Tomato", "Chilli", "Cotton"],
            "symptoms": "Upward or downward curling of leaves, yellowing, stunting",
            "organic_control": "Control whitefly vector with neem oil, remove infected plants",
            "chemical_control": "Thiamethoxam 25 WG for vector control",
            "prevention": "Use virus-free certified seeds, rogue out infected plants early",
        },
    }

    def get_pest_info(self, pest_name: str) -> Optional[Dict[str, Any]]:
        key = pest_name.lower().strip()
        for pest_key, info in self.PEST_DB.items():
            if pest_key in key or key in pest_key:
                return {"pest": pest_key, **info}
        return None

    def get_all_pests(self) -> list:
        return list(self.PEST_DB.keys())

    def format_for_context(self, pest_info: Dict) -> str:
        if not pest_info:
            return "No specific pest information found."
        return (
            f"Pest/Disease: {pest_info['pest'].title()}\n"
            f"Affected Crops: {', '.join(pest_info['affected_crops'])}\n"
            f"Symptoms: {pest_info['symptoms']}\n"
            f"Organic Control: {pest_info['organic_control']}\n"
            f"Chemical Control: {pest_info['chemical_control']}\n"
            f"Prevention: {pest_info['prevention']}"
        )


# -----------------------------------------------------------------------
# Government Scheme Tool
# -----------------------------------------------------------------------
class GovernmentSchemeTool:
    """Information about Indian government agricultural schemes."""

    SCHEMES = [
        {
            "name": "PM-KISAN",
            "full_name": "Pradhan Mantri Kisan Samman Nidhi",
            "benefit": "₹6,000/year direct income support in 3 installments to farmer families",
            "eligibility": "Small & marginal farmers with cultivable land",
            "how_to_apply": "Visit local Patwari/Revenue Officer or pmkisan.gov.in",
        },
        {
            "name": "PM Fasal Bima Yojana",
            "full_name": "Pradhan Mantri Fasal Bima Yojana",
            "benefit": "Crop insurance against natural calamities, pest attacks, and weather risks",
            "eligibility": "All farmers growing notified crops in notified areas",
            "how_to_apply": "Apply through nearest bank branch or Common Service Centre",
        },
        {
            "name": "KCC",
            "full_name": "Kisan Credit Card",
            "benefit": "Short-term credit up to ₹3 lakh at subsidized 4% interest for crop expenses",
            "eligibility": "All farmers, tenant farmers, oral lessees",
            "how_to_apply": "Apply at any bank branch (SBI, PNB, Co-op banks)",
        },
        {
            "name": "Soil Health Card",
            "full_name": "Soil Health Card Scheme",
            "benefit": "Free soil testing and card with crop-wise fertilizer recommendations",
            "eligibility": "All farmers",
            "how_to_apply": "Contact local Krishi Vigyan Kendra (KVK) or agriculture department",
        },
        {
            "name": "eNAM",
            "full_name": "National Agriculture Market",
            "benefit": "Online trading platform for better price discovery, transparent bidding",
            "eligibility": "Farmers registered at APMC mandis linked to eNAM",
            "how_to_apply": "Register at enam.gov.in or through mandi office",
        },
        {
            "name": "PMKSY",
            "full_name": "Pradhan Mantri Krishi Sinchayee Yojana",
            "benefit": "Subsidy on drip/sprinkler irrigation to improve 'More Crop Per Drop'",
            "eligibility": "All farmers, priority to small/marginal",
            "how_to_apply": "Apply through state agriculture/horticulture department",
        },
    ]

    def get_all_schemes(self) -> list:
        return self.SCHEMES

    def search_scheme(self, query: str) -> list:
        q = query.lower()
        return [
            s for s in self.SCHEMES
            if q in s["name"].lower() or q in s["full_name"].lower() or q in s["benefit"].lower()
        ]

    def format_for_context(self, schemes: list) -> str:
        if not schemes:
            return "No matching government schemes found."
        parts = []
        for s in schemes:
            parts.append(
                f"Scheme: {s['full_name']} ({s['name']})\n"
                f"Benefit: {s['benefit']}\n"
                f"Eligibility: {s['eligibility']}\n"
                f"How to Apply: {s['how_to_apply']}"
            )
        return "\n\n".join(parts)
