import os
import logging
import json
from openai import OpenAI
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
from translator import TextTranslator

logger = logging.getLogger(__name__)

class CallAnalyzer:
    """Analyzes transcribed call text to extract actions and insights"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.translator = TextTranslator(api_key=api_key, model=model)
        
    def extract_action_items(self, transcript: str, context: Dict = None) -> List[Dict]:
        """Extract action items from call transcript"""
        
        system_prompt = """You are an expert at analyzing phone call transcripts and extracting actionable items.

Extract all action items from the conversation. For each action item, identify:
1. What needs to be done (description)
2. Who is responsible (me, other person, or specific name)
3. When it should be done (deadline/timeframe if mentioned)
4. Priority level (high, medium, low)
5. Category (work, personal, follow-up, etc.)
6. Confidence score (0.0-1.0) - how confident you are this is actually an action item

Return results as a JSON array of objects with these fields:
- description: string
- responsible: string 
- deadline: string (or null)
- priority: string
- category: string
- confidence: float
- context: string (brief context from conversation)

Only extract clear, actionable items. Don't include vague statements or casual mentions."""

        if context:
            system_prompt += f"\n\nAdditional context about the caller: {json.dumps(context)}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript}
                ],
                temperature=0.1
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Extracted {len(result)} action items")
            return result
            
        except Exception as e:
            logger.error(f"Action extraction failed: {e}")
            return []
    
    def extract_personal_insights(self, transcript: str, existing_data: Dict = None) -> Dict:
        """Extract personal information and insights from the call"""
        
        system_prompt = """Analyze this phone call transcript to extract personal information and insights.

Extract the following categories of information:

1. PEOPLE: Names, relationships, contact info, roles
2. PROJECTS: Work projects, personal goals, ongoing activities
3. PREFERENCES: Communication style, scheduling preferences, interests
4. COMMITMENTS: Promises made or received, agreements
5. EVENTS: Meetings, deadlines, important dates mentioned
6. PATTERNS: Behavioral patterns, decision-making style

Return as JSON with these sections:
{
  "people": [{"name": "", "relationship": "", "role": "", "notes": ""}],
  "projects": [{"name": "", "status": "", "deadline": "", "priority": ""}],
  "preferences": [{"type": "", "description": "", "confidence": 0.8}],
  "commitments": [{"description": "", "from_who": "", "to_who": "", "deadline": ""}],
  "events": [{"description": "", "date": "", "type": ""}],
  "patterns": [{"observation": "", "confidence": 0.7}]
}

Only include information that is clearly stated or strongly implied."""

        if existing_data:
            system_prompt += f"\n\nExisting database information: {json.dumps(existing_data)}"
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript}
                ],
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info("Personal insights extraction completed")
            return result
            
        except Exception as e:
            logger.error(f"Insights extraction failed: {e}")
            return {}
            
    def get_call_summary(self, transcript: str) -> Dict:
        """Generate a summary of the call"""
        
        system_prompt = """Create a concise summary of this phone call.

Return JSON with:
{
  "duration_estimate": "X minutes",
  "participants": ["name1", "name2"],
  "main_topics": ["topic1", "topic2"],
  "call_type": "business|personal|support|other",
  "mood_tone": "professional|casual|urgent|friendly",
  "outcome": "brief description of what was accomplished",
  "follow_up_needed": true/false
}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript}
                ],
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info("Call summary generated")
            return result
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {}
            
    def analyze_call(self, transcript: str, context: Dict = None) -> Dict:
        """Perform complete analysis of a call transcript"""
        
        logger.info("Starting comprehensive call analysis")
        
        # Step 1: Detect language and translate if needed
        translation_info = self.translator.translate_if_needed(transcript)
        
        # Use translated text for analysis if translation was needed
        analysis_text = translation_info["translated_text"]
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "original_transcript": transcript,
            "translation_info": translation_info,
            "analysis_transcript": analysis_text,  # The text used for analysis
            "summary": self.get_call_summary(analysis_text),
            "action_items": self.extract_action_items(analysis_text, context),
            "insights": self.extract_personal_insights(analysis_text, context),
            "processed": False,  # Will be set to True after user verification
            "confidence": 0.0
        }
        
        # Calculate overall confidence score
        action_confidences = [item.get("confidence", 0.5) for item in analysis["action_items"]]
        if action_confidences:
            analysis["confidence"] = sum(action_confidences) / len(action_confidences)
            
        logger.info(f"Call analysis completed with confidence: {analysis['confidence']:.2f}")
        logger.info(f"Language detected: {translation_info['detected_language']}")
        if translation_info['translation_needed']:
            logger.info("Translation to English was performed")
            
        return analysis
