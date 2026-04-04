"""
Utility functions for Phone-to-Tasks project
"""

import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_api_key():
    """Load OpenAI API key from multiple sources"""
    
    # Try environment variable first
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        logger.info("API key loaded from environment variable")
        return api_key
    
    # Try loading from api_key.txt file
    key_file = Path(__file__).parent / "api_key.txt"
    if key_file.exists():
        try:
            with open(key_file, 'r') as f:
                content = f.read().strip()
                # Skip comment lines and empty lines
                lines = [line.strip() for line in content.split('\n') 
                        if line.strip() and not line.strip().startswith('#')]
                if lines:
                    api_key = lines[0]
                    if api_key != "your-openai-api-key-here":
                        logger.info("API key loaded from api_key.txt file")
                        return api_key
        except Exception as e:
            logger.error(f"Error reading API key file: {e}")
    
    logger.warning("No API key found")
    return None

def validate_api_key(api_key: str) -> bool:
    """Validate that the API key looks correct"""
    if not api_key:
        return False
    
    # Basic OpenAI API key format check
    if api_key.startswith('sk-') and len(api_key) > 40:
        return True
    
    return False
