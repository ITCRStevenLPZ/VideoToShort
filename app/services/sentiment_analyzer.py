import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Service for analyzing sentiment in text segments.
    Uses transformer-based sentiment analysis models to score text sentiment.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the sentiment analyzer with the specified model.
        
        Args:
            model_name: Name of the Hugging Face model to use for sentiment analysis
        """
        logger.info(f"Initializing SentimentAnalyzer with model: {model_name}")
        
        try:
            # Check if GPU is available
            device = 0 if torch.cuda.is_available() else -1
            
            # Load the model and tokenizer
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
            
            logger.info(f"Sentiment analysis model loaded successfully" + 
                       (f" (using GPU)" if device == 0 else " (using CPU)"))
            
        except Exception as e:
            logger.error(f"Error loading sentiment analysis model: {str(e)}", exc_info=True)
            raise
    
    def analyze(self, text: str) -> float:
        """
        Analyze the sentiment of a single text segment.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between 0 (negative) and 1 (positive)
        """
        try:
            # Handle empty text
            if not text or text.strip() == "":
                logger.warning("Empty text provided for sentiment analysis")
                return 0.5  # Neutral sentiment for empty text
            
            # Truncate text if it's too long (most models have a token limit)
            if len(text) > 500:
                logger.warning(f"Truncating long text for sentiment analysis (length: {len(text)})")
                text = text[:500]
            
            # Get sentiment prediction
            result = self.sentiment_pipeline(text)[0]
            
            # Extract sentiment score
            # If positive, use the score directly; if negative, use 1 - score
            score = result["score"]
            if result["label"].lower() == "positive":
                sentiment = score
            else:
                sentiment = 1.0 - score
            
            logger.debug(f"Sentiment analysis result: {sentiment:.4f} for text: '{text[:50]}...'")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
            return 0.5  # Return neutral sentiment on error
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Analyze sentiment for a batch of text segments.
        
        Args:
            texts: List of text segments to analyze
            batch_size: Size of batches for processing
            
        Returns:
            List of sentiment scores between 0 (negative) and 1 (positive)
        """
        try:
            if not texts:
                logger.warning("Empty batch provided for sentiment analysis")
                return []
            
            # Process texts in batches to avoid memory issues
            all_scores = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Truncate any texts that are too long
                processed_batch = [text[:500] if len(text) > 500 else text for text in batch]
                
                # Skip empty texts
                processed_batch = [text if text.strip() else "neutral" for text in processed_batch]
                
                # Get predictions for the batch
                results = self.sentiment_pipeline(processed_batch)
                
                # Process results
                batch_scores = []
                for result in results:
                    if result["label"].lower() == "positive":
                        sentiment = result["score"]
                    else:
                        sentiment = 1.0 - result["score"]
                    batch_scores.append(sentiment)
                
                all_scores.extend(batch_scores)
            
            logger.info(f"Processed sentiment for {len(texts)} text segments")
            return all_scores
            
        except Exception as e:
            logger.error(f"Error in batch sentiment analysis: {str(e)}", exc_info=True)
            return [0.5] * len(texts)  # Return neutral sentiment on error
            
    def get_sentiment_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """
        Calculate sentiment distribution statistics for a collection of texts.
        
        Args:
            texts: List of text segments to analyze
            
        Returns:
            Dictionary with sentiment distribution statistics:
            - average: average sentiment score
            - median: median sentiment score
            - positive_ratio: ratio of positive sentiments (>0.6)
            - negative_ratio: ratio of negative sentiments (<0.4)
            - neutral_ratio: ratio of neutral sentiments (between 0.4 and 0.6)
            - min: minimum sentiment score
            - max: maximum sentiment score
        """
        if not texts:
            logger.warning("Empty list provided for sentiment distribution")
            return {
                "average": 0.5,
                "median": 0.5,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0,
                "min": 0.5,
                "max": 0.5
            }
        
        # Analyze all texts
        scores = self.analyze_batch(texts)
        
        # Calculate statistics
        scores_array = np.array(scores)
        positive_count = np.sum(scores_array > 0.6)
        negative_count = np.sum(scores_array < 0.4)
        neutral_count = len(scores) - positive_count - negative_count
        
        distribution = {
            "average": float(np.mean(scores_array)),
            "median": float(np.median(scores_array)),
            "positive_ratio": float(positive_count / len(scores)),
            "negative_ratio": float(negative_count / len(scores)),
            "neutral_ratio": float(neutral_count / len(scores)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array))
        }
        
        logger.info(f"Calculated sentiment distribution: {distribution}")
        return distribution