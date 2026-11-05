# 🌾 Smart Farming AI Agent
### Powered by IBM Granite (watsonx.ai) · RAG Technology · Multi-language Support

An AI-powered agricultural advisor for **small-scale and marginal farmers** in India.  
Ask questions in **Hindi, Punjabi, Telugu, Tamil, Kannada, Marathi, Bengali, Gujarati, or English** and get instant, data-driven farming guidance.

---

## 🚀 Features

| Feature | Description |
|---|---|
| **IBM Granite LLM** | IBM's `granite-13b-chat-v2` on watsonx.ai answers crop, soil, and farming queries |
| **RAG Pipeline** | ChromaDB vector store retrieves relevant knowledge from the farming knowledge base |
| **Live Weather** | 7-day weather forecast via Open-Meteo (free, no API key needed) |
| **Mandi Prices** | Current market rates for 10+ commodities via data.gov.in / Agmarknet API |
| **Crop Calendar** | Season-wise crop recommendations (Kharif / Rabi / Zaid) |
| **Pest & Disease** | Identification and organic + chemical control measures |
| **Government Schemes** | PM-KISAN, KCC, Fasal Bima, PMKSY, eNAM, Soil Health Card |
| **Multi-language** | Input auto-detected and translated; response returned in farmer's language |
| **Web UI** | Clean browser-based chat interface for farmers |

---

## 🏗 Architecture

```
Farmer Query (any language)
        │
        ▼
Language Detection (langdetect)
        │
        ▼
Translate to English (GoogleTranslator)
        │
        ├──────────────────────────────────────────────┐
        │  Intent Detection (weather/crop/market/pest)  │
        │                                              │
        ▼                                              ▼
Live Tool Calls                              RAG Retrieval
  • Open-Meteo Weather                     (ChromaDB + sentence-transformers)
  • Agmarknet Market Prices
  • Crop Calendar                                    │
  • Pest/Disease DB                                  │
  • Gov Scheme DB                                    │
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
              Build Prompt (context + query)
                           │
                           ▼
              IBM Granite LLM (watsonx.ai)
                           │
                           ▼
              Translate Response → Farmer's Language
                           │
                           ▼
                  Final Answer to Farmer
```

---

## 📁 Project Structure

```
FARMING AGENT/
├── main.py                    # FastAPI application + REST endpoints
├── agent.py                   # Core agent orchestration loop
├── granite_llm.py             # IBM Granite / watsonx.ai integration
├── rag_pipeline.py            # RAG: ChromaDB ingestion + retrieval
├── farming_tools.py           # Weather, market, crop, pest, scheme tools
├── language_processor.py      # Multi-language detection + translation
├── config.py                  # Pydantic settings from .env
├── ingest.py                  # CLI script to ingest knowledge base
├── requirements.txt           # Python dependencies
├── env.example                # Environment variable template
├── start.bat                  # Windows startup script
├── start.sh                   # Linux/Mac startup script
├── static/
│   └── index.html             # Farmer-facing web UI
└── data/
    └── knowledge_base/
        └── farming_knowledge.txt   # Seed knowledge base document
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- IBM Cloud Lite account → [cloud.ibm.com](https://cloud.ibm.com)
- watsonx.ai project created in IBM Cloud

### Step 1 — Get IBM Cloud Credentials

1. Sign up at [cloud.ibm.com](https://cloud.ibm.com) (Lite tier is **free**)
2. Go to **watsonx.ai** → Create a project
3. Copy your **Project ID** from project settings
4. Go to **Manage → Access (IAM)** → Create an **API Key**
5. Note the **Service URL** (e.g. `https://us-south.ml.cloud.ibm.com`)

### Step 2 — Configure Environment

```bash
# Copy the template
cp env.example .env

# Edit .env and fill in your IBM credentials:
WATSONX_API_KEY=your_actual_api_key
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_actual_project_id
GRANITE_MODEL_ID=ibm/granite-13b-chat-v2
```

### Step 3 — Install & Run

**Windows:**
```bat
start.bat
```

**Linux / macOS:**
```bash
chmod +x start.sh && ./start.sh
```

**Manual:**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ingest.py                # Ingest knowledge base
uvicorn main:app --reload --port 8000
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Farmer web UI |
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Ask a farming question |
| `GET` | `/weather` | Live weather forecast |
| `GET` | `/market-prices` | Mandi commodity prices |
| `GET` | `/crop-calendar` | Season crop recommendations |
| `GET` | `/schemes` | Government schemes |
| `POST` | `/ingest` | Trigger KB re-ingestion |
| `GET` | `/languages` | Supported languages |
| `GET` | `/docs` | Interactive API docs (Swagger) |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "टमाटर के लिए सबसे अच्छा मौसम कौन सा है?"}'
```

Response:
```json
{
  "query": "टमाटर के लिए सबसे अच्छा मौसम कौन सा है?",
  "detected_language": "hi",
  "language_name": "Hindi (हिंदी)",
  "intents": ["crop_recommendation"],
  "answer": "टमाटर की खेती के लिए सबसे अच्छा मौसम रबी (अक्टूबर-दिसंबर) है...",
  "sources_used": ["farming_knowledge.txt"]
}
```

---

## 🌍 Supported Languages

| Code | Language |
|---|---|
| `en` | English |
| `hi` | Hindi (हिंदी) |
| `pa` | Punjabi (ਪੰਜਾਬੀ) |
| `te` | Telugu (తెలుగు) |
| `ta` | Tamil (தமிழ்) |
| `kn` | Kannada (ಕನ್ನಡ) |
| `mr` | Marathi (मराठी) |
| `bn` | Bengali (বাংলা) |
| `gu` | Gujarati (ગુજરાતી) |
| `or` | Odia (ଓଡ଼ିଆ) |

---

## 🛡 IBM Cloud Lite — Free Tier Usage

This project is designed to run on **IBM Cloud Lite (free)** services:

| Service | Free Tier Limit |
|---|---|
| **watsonx.ai** | Foundation model inference (token-based free tier) |
| **IBM Granite 13B Chat** | Available on Lite tier via watsonx.ai |
| **Watson Discovery** | 1 instance, 1,000 documents free (optional) |
| **Cloud Object Storage** | 25 GB free (optional, for storing KB docs) |

---

## 📌 Sample Questions to Ask

- *"What is the best crop to grow in black cotton soil this season?"*
- *"आज प्याज का मंडी भाव क्या है?"* (What is today's onion mandi rate?)
- *"ਮੇਰੀ ਕਣਕ ਤੇ ਕੀੜੇ ਲੱਗ ਗਏ ਹਨ, ਕੀ ਕਰਾਂ?"* (Insects on my wheat, what to do?)
- *"How do I apply for PM-KISAN scheme?"*
- *"What fertilizer should I use for tomatoes?"*
- *"Will it rain this week? Should I irrigate my field?"*

---

## 🤝 Contributing

1. Add more documents to `data/knowledge_base/` (PDF, TXT, CSV, DOCX)
2. Run `python ingest.py` to re-index
3. Extend `farming_tools.py` to add new live data sources
4. Add more languages in `language_processor.py`

---

*Built with IBM Granite · LangChain · ChromaDB · FastAPI · Open-Meteo*
