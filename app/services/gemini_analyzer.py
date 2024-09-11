import json
from core.config import settings
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    def analyze_transcript(self, transcript_text: str, topics: list[str]):
        # Configure the Gemini AI API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        # Initialize the Generative Model
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = (
            "Given the following transcript, find and paste the exact text that best matches the provided topics or instructions. "
            "Do not introduce or explain the text. The topics/instructions are: "
            f"{', '.join(topics)}. Here is the transcript:\n\n{transcript_text}\n\n"
            "Provide the exact text without any introduction or explanation."
        )
        
        # Generate the idea using the model
        response = model.generate_content(prompt)
        
        print(response.text)
        
        return response.text
    

    # def parse_response(self, response):
    #     """
    #     Parse the response from the Gemini AI model to extract timestamps.
        
    #     Args:
    #         response (str): The response string from the Gemini AI model in JSON format.
            
    #     Returns:
    #         list[dict]: A list of dictionaries with 'start' and 'end' keys.
    #     """
    #     timestamps = []

    #     # Remove markdown code identifiers
    #     response = response.replace("```json", "")
    #     response = response.replace("```", "")
        
    #     # Replace single quotes with double quotes
    #     response = response.replace("'", '"')
        
    #     try:
    #         # Parse the JSON response
    #         data = json.loads(response)
            
    #         # Extract the timestamps
    #         for item in data.get('timestamps', []):
    #             start = item.get('start')
    #             end = item.get('end')
    #             if start is not None and end is not None:
    #                 timestamps.append({'start': start, 'end': end})
        
    #     except json.JSONDecodeError:
    #         # Handle JSON parsing error
    #         print("Failed to parse response as JSON.")
        
    #     return timestamps