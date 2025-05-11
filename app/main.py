from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import api_router
from core.logging_config import setup_logging
import logging
import os
from core.config import settings

# Set up logging
logger = setup_logging()
logger.info("Starting Custom Video Clipper application")

# Create necessary directories
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.OUTPUT_DIR, "clips"), exist_ok=True)
os.makedirs("./data/vector_index", exist_ok=True)

app = FastAPI(title="Custom Video Clipper API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes from the api_router
app.include_router(api_router)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to Custom Video Clipper API. Use /docs for API documentation.",
        "enhanced_processing": "Available at /enhanced/process",
        "version": "2.0"
    }