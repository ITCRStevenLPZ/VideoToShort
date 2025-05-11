import json
from core.config import settings
import google.generativeai as genai
import logging
from typing import List, Optional

from models.service_models import GeminiConfig

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    def __init__(self, config: Optional[GeminiConfig] = None):
        """
        Initialize the GeminiAnalyzer with the Gemini API configuration.
        
        Args:
            config: Optional GeminiConfig containing model settings
        """
        try:
            logger.info("Initializing GeminiAnalyzer")
            
            # Use provided config or create default one
            self.config = config if config else GeminiConfig()
            
            # Configure the Gemini AI API
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.debug("Gemini API configured successfully")
            
            # Pre-initialize the model to verify configuration
            self.model = genai.GenerativeModel(
                self.config.model_name,
                generation_config={"temperature": self.config.temperature, "max_output_tokens": self.config.max_output_tokens}
            )
            logger.info(f"GeminiAnalyzer initialized successfully with model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GeminiAnalyzer: {str(e)}", exc_info=True)
            raise

    def analyze_transcript(self, transcript_text: str, topics: List[str]) -> str:
        """
        Analyze a transcript using Gemini AI to find text that matches specific topics.
        
        Args:
            transcript_text: The full transcript text to analyze
            topics: List of topics or instructions to search for in the transcript
            
        Returns:
            str: The extracted text from Gemini's response
        """
        try:
            logger.info(f"Analyzing transcript with topics: {topics}")
            logger.debug(f"Transcript length: {len(transcript_text)} characters")
            
            # Create the prompt for Gemini
            prompt = (
                "Given the following transcript, find and paste the exact text that best matches the provided topics or instructions. "
                "Do not introduce or explain the text. The topics/instructions are: "
                f"{', '.join(topics)}. Here is the transcript:\n\n{transcript_text}\n\n"
                "Provide the exact text without any introduction or explanation."
            )
            logger.debug("Created prompt for Gemini analysis")
            
            # Generate the response using the model
            logger.info("Sending request to Gemini API")
            response = self.model.generate_content(prompt)
            
            # Check if response is valid
            if not response or not hasattr(response, 'text') or not response.text:
                logger.warning("Received empty or invalid response from Gemini")
                return ""
                
            logger.info("Received response from Gemini API")
            logger.debug(f"Response length: {len(response.text)} characters")
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error analyzing transcript with Gemini: {str(e)}", exc_info=True)
            # Return empty string on error to prevent cascading failures
            return ""

    # def parse_response(self, response: str) -> List[Dict[str, float]]:
    #     """
    #     Parse the response from the Gemini AI model to extract timestamps.
        
    #     Args:
    #         response: The response string from the Gemini AI model in JSON format
            
    #     Returns:
    #         List of dictionaries with 'start' and 'end' keys
    #     """
    #     logger.debug("Parsing Gemini response for timestamps")
    #     timestamps = []

    #     # Remove markdown code identifiers
    #     response = response.replace("```json", "")
    #     response = response.replace("```", "")
        
    #     # Replace single quotes with double quotes
    #     response = response.replace("'", '"')
        
    #     try:
    #         # Parse the JSON response
    #         data = json.loads(response)
    #         logger.debug(f"Successfully parsed response as JSON: {data}")
            
    #         # Extract the timestamps
    #         for item in data.get('timestamps', []):
    #             try:
    #                 # Validate with Pydantic
    #                 timestamp = Timestamp(start=item.get('start'), end=item.get('end'))
    #                 timestamps.append(timestamp.dict())
    #                 logger.debug(f"Extracted timestamp: {timestamp.start} to {timestamp.end}")
    #             except Exception as e:
    #                 logger.warning(f"Invalid timestamp data: {e}")
        
    #     except json.JSONDecodeError as e:
    #         # Handle JSON parsing error
    #         logger.error(f"Failed to parse response as JSON: {str(e)}", exc_info=True)
        
    #     logger.info(f"Extracted {len(timestamps)} timestamps from Gemini response")
    #     return timestamps