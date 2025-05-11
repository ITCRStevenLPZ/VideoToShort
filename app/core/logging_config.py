import logging
import os
from logging.handlers import RotatingFileHandler
from core.config import settings

def setup_logging():
    """
    Configure logging for the application.
    Sets up console and file handlers with proper formatting.
    """
    # Create logs directory if it doesn't exist
    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure the root logger
    log_level = getattr(logging, settings.LOGGING_LEVEL)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Create a file handler for logging to a file
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "app.log"),
        maxBytes=10485760,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add the file handler to the root logger
    logging.getLogger().addHandler(file_handler)
    
    # Set specific loggers to different levels as needed
    # For example, to suppress excessive log messages from libraries:
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Return the configured logger
    return logging.getLogger(__name__)