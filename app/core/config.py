import os
import logging
from pydantic import BaseSettings, Field, field_validator
from typing import Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class LogLevel(str, Enum):
    """Valid logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class Settings(BaseSettings):
    """Application settings using Pydantic for validation"""
    PROJECT_NAME: str = Field("Custom Video Clipper", description="Name of the project")
    GEMINI_API_KEY: Optional[str] = Field(None, description="API key for Gemini AI")
    LOGGING_LEVEL: LogLevel = Field(LogLevel.INFO, description="Application logging level")
    
    # Additional settings with defaults
    OUTPUT_DIR: str = Field("./clips", description="Directory for saving output clips")
    MAX_CLIP_DURATION: int = Field(60, description="Maximum duration of a generated clip in seconds")
    ALLOWED_FILE_EXTENSIONS: list[str] = Field(
        ["mp4", "mov", "avi", "wmv", "mkv"], 
        description="Allowed video file extensions"
    )
    
    @field_validator("GEMINI_API_KEY", pre=True)
    def validate_gemini_api_key(cls, v):
        if not v:
            v = os.getenv("GEMINI_API_KEY")
            if not v:
                logging.warning("GEMINI_API_KEY not set in environment variables")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()