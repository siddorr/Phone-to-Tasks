import os
import logging
from openai import OpenAI
from typing import Optional, Dict
import re

logger = logging.getLogger(__name__)

class TextTranslator:
    """Handles translation of text to English using OpenAI"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def detect_language(self, text: str) -> Dict[str, str]:
        """Detect the language of the input text"""
        
        system_prompt = """Detect the language of the given text and return a JSON response with:
{
  "language": "language_name",
  "language_code": "ISO_code", 
  "confidence": 0.95,
  "is_english": true/false
}

Only respond with the JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:500]}  # Use first 500 chars for detection
                ],
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Detected language: {result.get('language', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "language": "Unknown",
                "language_code": "unknown",
                "confidence": 0.0,
                "is_english": False
            }
    
    def translate_to_english(self, text: str, source_language: str = None) -> Optional[str]:
        """Translate text to English"""
        
        if source_language:
            system_prompt = f"""You are a professional translator. Translate the following {source_language} text to English.

Requirements:
- Maintain the original meaning and context
- Use natural, fluent English
- Preserve the conversational tone
- Keep proper names unchanged
- If the text is already in English, return it as-is

Return only the translated text, no explanations or comments."""
        else:
            system_prompt = """You are a professional translator. Translate the following text to English.

Requirements:
- Detect the source language automatically
- Maintain the original meaning and context  
- Use natural, fluent English
- Preserve the conversational tone
- Keep proper names unchanged
- If the text is already in English, return it as-is

Return only the translated text, no explanations or comments."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            
            translated_text = response.choices[0].message.content.strip()
            logger.info("Translation completed successfully")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None
    
    def translate_if_needed(self, text: str) -> Dict[str, str]:
        """Detect language and translate to English if needed"""
        
        # First detect the language
        language_info = self.detect_language(text)
        
        result = {
            "original_text": text,
            "detected_language": language_info.get("language", "Unknown"),
            "language_code": language_info.get("language_code", "unknown"),
            "is_english": language_info.get("is_english", False),
            "confidence": language_info.get("confidence", 0.0),
            "translated_text": text,  # Default to original
            "translation_needed": False
        }
        
        # If not English, translate it
        if not language_info.get("is_english", False) and language_info.get("confidence", 0) > 0.7:
            logger.info(f"Translating from {language_info.get('language')} to English")
            
            translated = self.translate_to_english(
                text, 
                source_language=language_info.get("language")
            )
            
            if translated and translated != text:
                result["translated_text"] = translated
                result["translation_needed"] = True
                logger.info("Translation completed")
            else:
                logger.warning("Translation failed or text unchanged")
        
        return result
