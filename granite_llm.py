"""
IBM Granite LLM Integration via ibm-watsonx-ai SDK
Falls back to a helpful stub when ibm-watsonx-ai is not yet installed.
"""
import logging
from typing import Optional, Dict, Any

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


FARMING_SYSTEM_PROMPT = """You are an expert AI agricultural advisor helping small-scale farmers.
You have deep knowledge of:
- Crop selection based on season, soil type, and climate
- Soil health, fertilizer use, and irrigation best practices
- Pest and disease identification and organic/chemical control measures
- Weather interpretation and its effect on farming decisions
- Current market (mandi) prices and when/where to sell produce
- Government agricultural schemes and subsidies

Guidelines:
- Give practical, actionable advice suited for small-scale and marginal farmers.
- Use simple language. Avoid jargon unless you explain it.
- If you reference prices or weather, mention that figures should be verified locally.
- Always prioritize farmer safety and sustainable practices.
- When context from the knowledge base is provided, use it to ground your answer.
- If you don't know something, say so honestly rather than guessing.

Answer the farmer's question below using the provided context (if any)."""


PROMPT_TEMPLATE = """{system_prompt}

--- RETRIEVED KNOWLEDGE BASE CONTEXT ---
{context}
--- END OF CONTEXT ---

Farmer's Question: {question}

Your Expert Answer:"""


def build_prompt(question: str, context: str = "") -> str:
    return PROMPT_TEMPLATE.format(
        system_prompt=FARMING_SYSTEM_PROMPT,
        context=context if context else "No specific context retrieved.",
        question=question,
    )


class GraniteLLM:
    """
    Wrapper around ibm-watsonx-ai ModelInference.
    Gracefully handles the case where the package is not yet installed.
    """

    def __init__(self):
        self._model = None
        self._model_id = None   # track which model is loaded
        self._sdk_available = False
        self._check_sdk()

    def _check_sdk(self):
        try:
            import ibm_watsonx_ai  # noqa: F401
            self._sdk_available = True
            logger.info("ibm-watsonx-ai SDK found.")
        except ImportError:
            self._sdk_available = False
            logger.warning("ibm-watsonx-ai not installed yet. Install with: pip install ibm-watsonx-ai")

    def _get_model(self):
        # Reload if model ID changed (e.g. .env updated)
        if self._model is not None and self._model_id == settings.granite_model_id:
            return self._model
        self._model = None

        if not self._sdk_available:
            # Try again — package may have finished installing
            self._check_sdk()
            if not self._sdk_available:
                raise ImportError(
                    "ibm-watsonx-ai is still installing. Please wait 1-2 minutes and try again."
                )

        if not settings.watsonx_api_key or settings.watsonx_api_key == "your_ibm_cloud_api_key_here":
            raise ValueError(
                "WATSONX_API_KEY is not configured. Please set it in your .env file. "
                "Get your key at https://cloud.ibm.com/ → watsonx.ai"
            )
        if not settings.watsonx_project_id or settings.watsonx_project_id == "your_watsonx_project_id_here":
            raise ValueError("WATSONX_PROJECT_ID is not configured. Please set it in your .env file.")

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        credentials = Credentials(
            url=settings.watsonx_url,
            api_key=settings.watsonx_api_key,
        )
        params: Dict[str, Any] = {
            GenParams.MAX_NEW_TOKENS: settings.llm_max_new_tokens,
            GenParams.TEMPERATURE: settings.llm_temperature,
            GenParams.TOP_P: settings.llm_top_p,
            GenParams.TOP_K: settings.llm_top_k,
            GenParams.REPETITION_PENALTY: settings.llm_repetition_penalty,
            GenParams.STOP_SEQUENCES: ["Farmer's Question:", "---"],
        }
        self._model = ModelInference(
            model_id=settings.granite_model_id,
            credentials=credentials,
            project_id=settings.watsonx_project_id,
            params=params,
        )
        self._model_id = settings.granite_model_id
        logger.info("LLM model loaded: %s", settings.granite_model_id)
        return self._model

    def generate(self, prompt: str) -> str:
        model = self._get_model()
        try:
            response = model.generate_text(prompt=prompt)
            text = response.strip() if isinstance(response, str) else str(response)
            # Truncate at natural stop markers left in output
            for stop in ["Farmer's Question:", "--- RETRIEVED"]:
                if stop in text:
                    text = text[:text.index(stop)].strip()
            return text
        except Exception as exc:
            logger.error("Granite generation error: %s", exc)
            raise

    def generate_stream(self, prompt: str):
        model = self._get_model()
        try:
            for chunk in model.generate_text_stream(prompt=prompt):
                yield chunk
        except Exception as exc:
            logger.error("Granite streaming error: %s", exc)
            raise


_granite_llm: Optional[GraniteLLM] = None


def get_granite_llm() -> GraniteLLM:
    global _granite_llm
    if _granite_llm is None:
        _granite_llm = GraniteLLM()
    return _granite_llm
