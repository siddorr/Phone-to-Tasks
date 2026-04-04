#!/usr/bin/env python3
"""
Test translation functionality
"""

import sys
from utils import load_api_key
from translator import TextTranslator

def test_translation():
    """Test the translation module"""
    
    print("🌐 Translation Test")
    print("="*40)
    
    # Get API key
    api_key = load_api_key()
    if not api_key:
        print("❌ No API key found!")
        return
    
    # Initialize translator
    translator = TextTranslator(api_key=api_key)
    
    # Test texts in different languages
    test_texts = [
        "Hello, this is a test call in English.",
        "Привет, это тестовый звонок на русском языке.",
        "שלום, זהו שיחת טלפון לבדיקה בעברית.",
        "Hola, esta es una llamada de prueba en español."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Test {i}:")
        print(f"Original: {text}")
        
        # Test translation
        result = translator.translate_if_needed(text)
        
        print(f"Language: {result['detected_language']}")
        print(f"Translation needed: {result['translation_needed']}")
        
        if result['translation_needed']:
            print(f"Translated: {result['translated_text']}")
        else:
            print("No translation needed - already in English")
        
        print("-" * 40)

if __name__ == "__main__":
    test_translation()
