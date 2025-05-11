from sentence_transformers import SentenceTransformer, util
import os
import logging
import torch
from typing import List, Dict, Any, Optional, Union

from services.gemini_analyzer import GeminiAnalyzer
from models.service_models import BERTAnalyzerConfig, SimilarityMethod
from models.clip_request import ClipAnalysisResult, Sentence

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class BERTAnalyzer:
    def __init__(self, config: Optional[BERTAnalyzerConfig] = None):
        """
        Initialize the BERT Analyzer with optional configuration.
        
        Args:
            config: Optional BERTAnalyzerConfig containing model settings
        """
        try:
            # Use provided config or create default one
            self.config = config if config else BERTAnalyzerConfig()
            
            logger.info(f"Initializing BERTAnalyzer with model: {self.config.model_name}")
            # Load a pre-trained sentence transformer model
            self.model = SentenceTransformer(self.config.model_name)
            self.gemini_analyzer = GeminiAnalyzer()
            logger.info("BERTAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BERTAnalyzer: {e}", exc_info=True)
            raise

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Generate an embedding for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            A tensor containing the text embedding
        """
        try:
            # Return the sentence embedding from sentence-transformers
            return self.model.encode(text, convert_to_tensor=True)
        except Exception as e:
            logger.error(f"Error embedding text: {e}", exc_info=True)
            raise

    def calculate_similarity(
        self, 
        transcript: List[Dict[str, Any]], 
        topics: List[str], 
        threshold: Optional[float] = None,
        video_length: Optional[int] = None
    ) -> Optional[Union[ClipAnalysisResult, Dict[str, Any]]]:
        """
        Calculate similarity between transcript and topics.
        
        Args:
            transcript: List of transcript sentence dictionaries with text and timestamps
            topics: List of topics to search for in the transcript
            threshold: Similarity threshold (0-1). Uses config value if not provided
            video_length: Optional video length in seconds to cap timestamp
            
        Returns:
            A ClipAnalysisResult object or None if no matches found
        """
        try:
            # Use provided threshold or default from config
            threshold = threshold if threshold is not None else self.config.threshold
            logger.info(f"Calculating similarity with {len(topics)} topics and threshold: {threshold}")
            
            # Determine the device (GPU if available, otherwise CPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.debug(f"Using device: {device}")
    
            # Concatenate transcript texts into a single string
            if not transcript:
                logger.warning("Empty transcript provided to calculate_similarity")
                return None
                
            transcript_text = ' '.join([entry['text'] for entry in transcript])
            logger.debug(f"Created transcript text with {len(transcript_text)} characters")
    
            # Analyze transcript to extract the exact matching text related to topics
            logger.info(f"Requesting Gemini analysis for topics: {topics}")
            gemini_output = self.gemini_analyzer.analyze_transcript(transcript_text, topics)
            gemini_output = self.format_remover(gemini_output)
    
            # Split the output text from GeminiAI into sentences
            gemini_sentences = self.split_into_sentences(gemini_output)
            logger.info(f"Extracted {len(gemini_sentences)} sentences from GeminiAI")
            logger.debug(f"Gemini sentences: {gemini_sentences}")
    
            # Embed the Gemini sentences and move to the same device
            logger.debug("Creating embeddings for Gemini sentences")
            gemini_embeddings = [self.embed_text(sentence).to(device) for sentence in gemini_sentences]
    
            matched_sentences = []
            logger.info(f"Comparing {len(transcript)} transcript sentences with Gemini sentences")
            for sentence in transcript:
                sentence_text = sentence['text']
                sentence_embedding = self.embed_text(sentence_text).to(device)
    
                # Calculate similarity with each Gemini sentence
                max_similarity = 0
                try:
                    for gemini_embedding in gemini_embeddings:
                        # Choose similarity method based on config
                        if self.config.similarity_method == SimilarityMethod.COSINE:
                            similarity = util.pytorch_cos_sim(sentence_embedding, gemini_embedding).item()
                        elif self.config.similarity_method == SimilarityMethod.DOT_PRODUCT:
                            similarity = torch.dot(sentence_embedding, gemini_embedding).item()
                        else:  # default to cosine
                            similarity = util.pytorch_cos_sim(sentence_embedding, gemini_embedding).item()
                            
                        if similarity > max_similarity:
                            max_similarity = similarity
                except Exception as e:
                    logger.error(f"Error calculating similarity embeddings: {e}", exc_info=True)
    
                # If the similarity exceeds the threshold, consider it a match
                if max_similarity >= threshold:
                    logger.debug(f"Match found with similarity {max_similarity}: {sentence_text[:50]}...")
                    matched_sentences.append({
                        'text': sentence_text,
                        'start_timestamp': sentence['start_timestamp'],
                        'end_timestamp': sentence['end_timestamp'],
                        'similarity': max_similarity
                    })
    
            if not matched_sentences:
                logger.warning("No matching sentences found in the transcription.")
                return None
    
            # Sort matched sentences by start timestamp
            matched_sentences.sort(key=lambda x: x['start_timestamp'])
            logger.info(f"Found {len(matched_sentences)} matching sentences, sorted by timestamp")
    
            # Merge timestamps of all matched sentences
            start_timestamp = matched_sentences[0]['start_timestamp']
            end_timestamp = matched_sentences[-1]['end_timestamp']
            logger.info(f"Merged time range: {start_timestamp} to {end_timestamp}")
    
            if video_length is not None and end_timestamp > video_length:
                logger.info(f"Capping end timestamp to video length: {video_length}")
                end_timestamp = video_length
    
            # Create a clip representing the combined text and timestamps
            combined_text = ' '.join([sentence['text'] for sentence in matched_sentences])
            
            # Create ClipAnalysisResult using Pydantic model
            clip = ClipAnalysisResult(
                text=combined_text,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                similarity=max(sentence['similarity'] for sentence in matched_sentences)
            )
            logger.debug(f"Created clip with length: {clip.end_timestamp - clip.start_timestamp} seconds")
    
            # Save the result
            self.save_similarity_results([clip.dict()], self.config.output_file)
            logger.info(f"Saved similarity results to {self.config.output_file}")
    
            # Return as dict to maintain backward compatibility
            return clip.dict()
    
        except Exception as e:
            logger.error(f"Error in calculate_similarity: {e}", exc_info=True)
            raise

    def format_remover(self, text: str) -> str:
        """
        Remove unnecessary formatting from the text.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text with newlines and carriage returns removed
        """
        try:
            return text.replace('\n', ' ').replace('\r', ' ').strip()
        except Exception as e:
            logger.error(f"Error in format_remover: {e}", exc_info=True)
            raise

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split the text into sentences based on simple punctuation rules.
        
        Args:
            text: The text to split into sentences
            
        Returns:
            List of sentences
        """
        try:
            return [sentence.strip() for sentence in text.split('.') if sentence.strip()]
        except Exception as e:
            logger.error(f"Error in split_into_sentences: {e}", exc_info=True)
            raise

    def is_exact_match(self, transcript_sentence: str, gemini_sentence: str) -> bool:
        """
        Check if the given transcript sentence matches the GeminiAI sentence exactly.
        
        Args:
            transcript_sentence: Sentence from transcript
            gemini_sentence: Sentence from Gemini output
            
        Returns:
            True if exact match, False otherwise
        """
        try:
            return transcript_sentence.strip().lower() == gemini_sentence.strip().lower()
        except Exception as e:
            logger.error(f"Error in is_exact_match: {e}", exc_info=True)
            raise

    def save_similarity_results(self, similarity_results: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save the similarity results to a text file in a legible format.
        
        Args:
            similarity_results: List of result dictionaries
            file_path: Path to output file
        """
        try:
            logger.debug(f"Saving similarity results to {file_path}")
            if os.path.exists(file_path):
                os.remove(file_path)

            with open(file_path, "w") as file:
                for result in similarity_results:
                    file.write(f"Text: {result['text']}\n")
                    file.write(f"Start Timestamp: {result['start_timestamp']}\n")
                    file.write(f"End Timestamp: {result['end_timestamp']}\n")
                    if 'similarity' in result:
                        file.write(f"Similarity: {result['similarity']}\n")
                    file.write("\n" + "-"*50 + "\n\n")
            logger.debug("Results file written successfully")
        except Exception as e:
            logger.error(f"Error in save_similarity_results: {e}", exc_info=True)
            raise