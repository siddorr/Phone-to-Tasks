import os
import logging
import whisper
from openai import OpenAI
from pydub import AudioSegment
import tempfile
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class SpeechToTextConverter:
    """Handles conversion of audio files to text using OpenAI Whisper"""
    
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.local_whisper = None
        
    def load_local_whisper(self, model_size: str = "base"):
        """Load local Whisper model for offline processing"""
        try:
            self.local_whisper = whisper.load_model(model_size)
            logger.info(f"Loaded local Whisper model: {model_size}")
        except Exception as e:
            logger.error(f"Failed to load local Whisper model: {e}")
            
    def convert_audio_format(self, audio_path: str) -> str:
        """Convert audio to format suitable for Whisper"""
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to WAV, mono, 16kHz (Whisper optimal format)
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                audio.export(tmp_file.name, format="wav")
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return audio_path
            
    def transcribe_with_api(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper API"""
        try:
            # Convert audio format if needed
            processed_audio = self.convert_audio_format(audio_path)
            
            with open(processed_audio, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="text"
                )
                
            # Clean up temporary file if created
            if processed_audio != audio_path:
                os.unlink(processed_audio)
                
            logger.info("Audio transcription completed successfully")
            return transcript
            
        except Exception as e:
            logger.error(f"API transcription failed: {e}")
            return None
            
    def transcribe_with_local(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using local Whisper model"""
        if not self.local_whisper:
            logger.warning("Local Whisper model not loaded")
            return None
            
        try:
            result = self.local_whisper.transcribe(audio_path)
            logger.info("Local transcription completed successfully")
            return result["text"]
            
        except Exception as e:
            logger.error(f"Local transcription failed: {e}")
            return None
            
    def transcribe(self, audio_path: str, use_local: bool = False) -> Optional[str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            use_local: Whether to use local model (offline) or API
            
        Returns:
            Transcribed text or None if failed
        """
        if use_local:
            return self.transcribe_with_local(audio_path)
        else:
            return self.transcribe_with_api(audio_path)
