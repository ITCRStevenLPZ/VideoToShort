import os
import logging
import json
import numpy as np
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
import pickle
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer
import faiss

from models.service_models import VectorIndexConfig, ModelType
from services.sentiment_analyzer import SentimentAnalyzer

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class VectorIndexService:
    """
    Service for building and querying vector indices of transcript segments.
    Implements semantic search over transcript segments using FAISS vector store.
    """
    
    def __init__(self, config: Optional[VectorIndexConfig] = None):
        """
        Initialize the vector index service with configuration.
        
        Args:
            config: Optional configuration for the service
        """
        # Initialize with default config if none provided
        self.config = config if config else VectorIndexConfig()
        logger.info(f"Initializing VectorIndexService with config: {self.config.dict()}")
        
        # Create the base index directory if it doesn't exist
        os.makedirs(self.config.index_path, exist_ok=True)
        
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(self.config.model_name.value)
            logger.info(f"Loaded embedding model: {self.config.model_name.value}")
            
            # Set device if GPU is enabled
            if self.config.use_gpu and torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to("cuda")
                logger.info("Using GPU for embedding computation")
            else:
                logger.info("Using CPU for embedding computation")
                
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}", exc_info=True)
            raise
            
        # Initialize sentiment analyzer if enabled
        if self.config.precompute_sentiment:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("Initialized sentiment analyzer for segment sentiment scoring")
            except Exception as e:
                logger.error(f"Error initializing sentiment analyzer: {str(e)}", exc_info=True)
                logger.warning("Sentiment analysis will be disabled")
                self.sentiment_analyzer = None
                self.config.precompute_sentiment = False
        else:
            self.sentiment_analyzer = None
            
        # Current index properties
        self.current_index = None
        self.current_index_name = None
        self.segments = []
        self.index_metadata = {}
        
    def _get_index_path(self, index_name: str) -> Tuple[str, str, str]:
        """
        Get paths for index files.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Tuple of (index_dir, faiss_index_path, metadata_path)
        """
        # Sanitize index name to be a valid filename
        clean_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in index_name)
        
        # Generate paths
        index_dir = os.path.join(self.config.index_path, clean_name)
        faiss_index_path = os.path.join(index_dir, "index.faiss")
        metadata_path = os.path.join(index_dir, "metadata.json")
        segments_path = os.path.join(index_dir, "segments.pickle")
        
        return index_dir, faiss_index_path, metadata_path, segments_path
        
    def create_index(self, transcript_path: str, index_name: str) -> bool:
        """
        Create a new vector index from a transcript file.
        
        Args:
            transcript_path: Path to the transcript JSON file
            index_name: Name for the new index
            
        Returns:
            True if index was created successfully
        """
        try:
            logger.info(f"Creating new index '{index_name}' from transcript: {transcript_path}")
            
            # Load transcript data
            with open(transcript_path, 'r') as f:
                transcript = json.load(f)
                
            # Extract segments from transcript
            # Transcript structure depends on your transcription service
            segments = self._extract_segments_from_transcript(transcript)
            
            if not segments:
                logger.error("No segments extracted from transcript")
                return False
                
            logger.info(f"Extracted {len(segments)} segments from transcript")
            
            # Create embeddings for all segments
            texts = [segment['text'] for segment in segments]
            embeddings = self._create_embeddings(texts)
            
            # Add sentiment scores if enabled
            if self.config.precompute_sentiment and self.sentiment_analyzer:
                logger.info("Computing sentiment scores for segments")
                sentiment_scores = self.sentiment_analyzer.analyze_batch(texts)
                
                # Add sentiment scores to segments
                for i, score in enumerate(sentiment_scores):
                    segments[i]['sentiment'] = score
                    
                logger.info("Added sentiment scores to segments")
            
            # Create FAISS index
            dimension = self.config.dimension
            faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = self._normalize_embeddings(embeddings)
            
            # Add vectors to index
            faiss_index.add(normalized_embeddings)
            
            # Save the index
            index_dir, faiss_index_path, metadata_path, segments_path = self._get_index_path(index_name)
            
            # Create index directory
            os.makedirs(index_dir, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(faiss_index, faiss_index_path)
            
            # Save segments
            with open(segments_path, 'wb') as f:
                pickle.dump(segments, f)
                
            # Save metadata
            metadata = {
                'name': index_name,
                'created_at': datetime.now().isoformat(),
                'transcript_path': transcript_path,
                'model': self.config.model_name.value,
                'dimension': dimension,
                'num_segments': len(segments),
                'has_sentiment': self.config.precompute_sentiment
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Update current index
            self.current_index = faiss_index
            self.current_index_name = index_name
            self.segments = segments
            self.index_metadata = metadata
            
            logger.info(f"Successfully created index '{index_name}' with {len(segments)} segments")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}", exc_info=True)
            return False
            
    def load_index(self, index_name: str) -> bool:
        """
        Load an existing vector index.
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            True if index was loaded successfully
        """
        try:
            logger.info(f"Loading index: {index_name}")
            
            # Get paths
            index_dir, faiss_index_path, metadata_path, segments_path = self._get_index_path(index_name)
            
            # Check if index exists
            if not os.path.exists(faiss_index_path) or not os.path.exists(segments_path):
                logger.warning(f"Index '{index_name}' not found")
                return False
                
            # Load FAISS index
            faiss_index = faiss.read_index(faiss_index_path)
            
            # Load segments
            with open(segments_path, 'rb') as f:
                segments = pickle.load(f)
                
            # Load metadata if available
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                # Create basic metadata if not found
                metadata = {
                    'name': index_name,
                    'loaded_at': datetime.now().isoformat(),
                    'num_segments': len(segments)
                }
                
            # Update current index
            self.current_index = faiss_index
            self.current_index_name = index_name
            self.segments = segments
            self.index_metadata = metadata
            
            logger.info(f"Successfully loaded index '{index_name}' with {len(segments)} segments")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}", exc_info=True)
            return False
            
    def query_index(
        self, 
        query: str, 
        top_k: int = None, 
        sentiment_filter: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the current index with a text query.
        
        Args:
            query: Text query to search for
            top_k: Number of results to return (defaults to config value)
            sentiment_filter: Optional minimum sentiment score filter
            
        Returns:
            List of matching segments with metadata
        """
        try:
            if self.current_index is None:
                logger.error("No index loaded")
                return []
                
            # Use default top_k from config if not specified
            if top_k is None:
                top_k = self.config.top_k
                
            logger.info(f"Querying index '{self.current_index_name}' with: '{query}' (top_k={top_k})")
            
            # Embed the query
            query_embedding = self._create_embeddings([query])[0]
            
            # Normalize query embedding for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search the index
            D, I = self.current_index.search(query_embedding, top_k * 3)  # Get extra results for filtering
            
            # Get the matching segments
            matches = []
            for i, (idx, score) in enumerate(zip(I[0], D[0])):
                if idx < len(self.segments):
                    match = self.segments[idx].copy()
                    match['similarity'] = float(score)
                    matches.append(match)
                    
            # Filter by sentiment if requested
            if sentiment_filter is not None and matches and 'sentiment' in matches[0]:
                matches = [m for m in matches if m.get('sentiment', 0) >= sentiment_filter]
                
            # Sort by similarity and limit to top_k
            matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:top_k]
            
            # Now sort chronologically for clip generation
            matches = sorted(matches, key=lambda x: x['start_timestamp'])
            
            logger.info(f"Found {len(matches)} matching segments")
            return matches
            
        except Exception as e:
            logger.error(f"Error querying index: {str(e)}", exc_info=True)
            return []
            
    def list_available_indices(self) -> List[str]:
        """
        List all available indices.
        
        Returns:
            List of index names
        """
        try:
            indices = []
            
            # List all subdirectories in the index path
            for item in os.listdir(self.config.index_path):
                item_path = os.path.join(self.config.index_path, item)
                
                # Check if it's a directory and contains index files
                if os.path.isdir(item_path):
                    faiss_path = os.path.join(item_path, "index.faiss")
                    segments_path = os.path.join(item_path, "segments.pickle")
                    
                    if os.path.exists(faiss_path) and os.path.exists(segments_path):
                        indices.append(item)
                        
            return indices
            
        except Exception as e:
            logger.error(f"Error listing indices: {str(e)}", exc_info=True)
            return []
            
    def delete_index(self, index_name: str) -> bool:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if index was deleted successfully
        """
        try:
            logger.info(f"Deleting index: {index_name}")
            
            # Get the index directory
            index_dir, _, _, _ = self._get_index_path(index_name)
            
            # Check if it exists
            if not os.path.exists(index_dir):
                logger.warning(f"Index '{index_name}' not found")
                return False
                
            # Delete the directory
            shutil.rmtree(index_dir)
            
            # Clear current index if it's the one being deleted
            if self.current_index_name == index_name:
                self.current_index = None
                self.current_index_name = None
                self.segments = []
                self.index_metadata = {}
                
            logger.info(f"Successfully deleted index '{index_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}", exc_info=True)
            return False
            
    def _extract_segments_from_transcript(self, transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract segments from transcript data.
        
        Args:
            transcript: Transcript data
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        try:
            # Check if transcript has 'results' field (typical structure)
            if 'results' in transcript:
                # Process Google Speech-to-Text style transcript
                results = transcript['results']
                
                # Extract segments from each result
                for result in results:
                    if 'alternatives' in result and result['alternatives']:
                        alt = result['alternatives'][0]  # Take first alternative
                        
                        if 'transcript' in alt:
                            text = alt['transcript']
                            
                            # Get timestamp if available
                            start_time = 0.0
                            end_time = 0.0
                            
                            if 'words' in alt and alt['words']:
                                words = alt['words']
                                start_time = float(words[0].get('startTime', 0).replace('s', ''))
                                end_time = float(words[-1].get('endTime', 0).replace('s', ''))
                            
                            segments.append({
                                'text': text,
                                'start_timestamp': start_time,
                                'end_timestamp': end_time
                            })
            # Check if transcript has a sentences list (for improved structure)
            elif 'sentences' in transcript:
                # Process custom transcript format with sentences
                for sentence in transcript['sentences']:
                    segments.append({
                        'text': sentence['text'],
                        'start_timestamp': sentence['start_timestamp'],
                        'end_timestamp': sentence['end_timestamp']
                    })
            # Fall back to basic structure
            elif isinstance(transcript, list):
                # Assume list of sentence objects
                for item in transcript:
                    if isinstance(item, dict) and 'text' in item:
                        segments.append({
                            'text': item['text'],
                            'start_timestamp': item.get('start_timestamp', 0.0),
                            'end_timestamp': item.get('end_timestamp', 0.0)
                        })
            else:
                logger.warning("Unrecognized transcript format")
                
            return segments
            
        except Exception as e:
            logger.error(f"Error extracting segments from transcript: {str(e)}", exc_info=True)
            return []
            
    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        try:
            # Encode all texts to embeddings
            with torch.no_grad():
                embeddings = self.embedding_model.encode(
                    texts, 
                    convert_to_numpy=True, 
                    show_progress_bar=True if len(texts) > 100 else False
                )
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            raise
            
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity.
        
        Args:
            embeddings: NumPy array of embeddings
            
        Returns:
            Normalized embeddings
        """
        # Calculate norms
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms[norms == 0] = 1.0
        
        # Normalize
        normalized = embeddings / norms
        
        return normalized.astype('float32')