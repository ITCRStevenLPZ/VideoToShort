import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Custom Video Clipper"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    LOGGING_LEVEL: str = os.getenv("LOGGING_LEVEL", "INFO")

settings = Settings()