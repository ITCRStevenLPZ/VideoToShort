# Logging Guide

## Overview
This project uses a unified logging system for all components. Logs are written to both the console and a file (`logs/app.log`).

## Log Levels
- DEBUG: Detailed debug information
- INFO: General application flow
- WARNING: Unexpected behavior that doesn't affect core functionality
- ERROR: Failures that prevent normal operation
- CRITICAL: Critical failures that require immediate attention

## Configuration
You can set the logging level in your .env file:
```
LOGGING_LEVEL=DEBUG  # For maximum detail during development
LOGGING_LEVEL=INFO   # For normal operation
LOGGING_LEVEL=WARNING  # For production (only shows warnings and errors)
```

## Viewing Logs
- Console: Logs are displayed in real-time in your terminal
- Log File: Logs are stored in `logs/app.log`
- The log files rotate when they reach 10MB, keeping up to 5 backup files

## Example Usage
```python
import logging

# Get a logger for your module
logger = logging.getLogger(__name__)

# Use different levels based on importance
logger.debug("Detailed information for debugging")
logger.info("General information about application flow")
logger.warning("Something unexpected but not critical")
logger.error("An error that affects functionality")
logger.critical("A critical error that requires immediate attention")
```