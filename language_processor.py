"""
Multi-language Support Layer
Detects input language, translates to English for processing,
then translates response back to the farmer's language.
Supports: English, Hindi, Punjabi, Telugu, Tamil, Kannada, Marathi, Bengali, Gujarati, Odia
"""
import logging
from typing import Optional, Tuple

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Language code → display name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "te": "Telugu (తెలుగు)",
    "ta": "Tamil (தமிழ்)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "mr": "Marathi (मराठी)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "or": "Odia (ଓଡ଼ିଆ)",
}

# Greetings in different languages for the welcome message
GREETINGS = {
    "en": "Hello! I'm your Smart Farming Advisor. How can I help you today?",
    "hi": "नमस्ते! मैं आपका स्मार्ट कृषि सलाहकार हूं। आज मैं आपकी कैसे मदद कर सकता हूं?",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ ਸਮਾਰਟ ਖੇਤੀਬਾੜੀ ਸਲਾਹਕਾਰ ਹਾਂ। ਅੱਜ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
    "te": "నమస్కారం! నేను మీ స్మార్ట్ వ్యవసాయ సలహాదారుడిని. ఈరోజు నేను మీకు ఎలా సహాయం చేయగలను?",
    "ta": "வணக்கம்! நான் உங்கள் ஸ்மார்ட் விவசாய ஆலோசகர். இன்று நான் உங்களுக்கு எவ்வாறு உதவலாம்?",
    "kn": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಲಹೆಗಾರ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
    "mr": "नमस्कार! मी तुमचा स्मार्ट शेती सल्लागार आहे. आज मी तुम्हाला कशी मदत करू शकतो?",
    "bn": "নমস্কার! আমি আপনার স্মার্ট কৃষি উপদেষ্টা। আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
    "gu": "નમસ્તે! હું તમારો સ્માર્ટ ખેતી સલાહકાર છું. આજે હું તમારી કેવી રીતે મદદ કરી શકું?",
    "or": "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କ ସ୍ମାର୍ଟ କୃଷି ପରାମର୍ଶଦାତା। ଆଜି ମୁଁ ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରିବି?",
}


class LanguageProcessor:
    """
    Handles language detection and bidirectional translation.
    Translates farmer queries → English (for RAG + LLM)
    then translates LLM answers → farmer's original language.
    """

    def __init__(self):
        self.supported = settings.supported_language_list
        self.default_lang = settings.default_language

    def detect_language(self, text: str) -> str:
        """Detect language of input text. Returns ISO 639-1 code."""
        if not settings.auto_detect_language:
            return self.default_lang
        try:
            from langdetect import detect, LangDetectException
            detected = detect(text)
            code_map = {"zh-cn": "zh", "zh-tw": "zh"}
            detected = code_map.get(detected, detected)
            return detected if detected in self.supported else "en"
        except ImportError:
            return self.default_lang
        except Exception:
            return self.default_lang

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text from source_lang to English."""
        if source_lang == "en":
            return text
        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source=source_lang, target="en").translate(text)
            return translated or text
        except ImportError:
            return text  # Package not installed yet — pass through as-is
        except Exception as exc:
            logger.warning("Translation to English failed: %s", exc)
            return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate English LLM response to target_lang."""
        if target_lang == "en":
            return text
        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source="en", target=target_lang).translate(text)
            return translated or text
        except ImportError:
            return text  # Package not installed yet — return English
        except Exception as exc:
            logger.warning("Translation to %s failed: %s", target_lang, exc)
            return text

    def process_query(self, raw_query: str, override_lang: Optional[str] = None) -> Tuple[str, str]:
        """
        Detect language and translate query to English.
        Returns: (english_query, detected_language_code)
        """
        lang = override_lang if override_lang else self.detect_language(raw_query)
        english_query = self.translate_to_english(raw_query, lang)
        return english_query, lang

    def process_response(self, english_response: str, target_lang: str) -> str:
        """Translate an English LLM response to the target language."""
        return self.translate_from_english(english_response, target_lang)

    def get_greeting(self, lang: str) -> str:
        """Return the welcome greeting in the given language."""
        return GREETINGS.get(lang, GREETINGS["en"])

    def get_language_name(self, lang_code: str) -> str:
        """Return the human-readable name for a language code."""
        return LANGUAGE_NAMES.get(lang_code, lang_code.upper())


# Module-level singleton
_lang_processor: Optional[LanguageProcessor] = None


def get_language_processor() -> LanguageProcessor:
    global _lang_processor
    if _lang_processor is None:
        _lang_processor = LanguageProcessor()
    return _lang_processor
