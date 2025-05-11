from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum
import os
from pathlib import Path

class VideoFormat(str, Enum):
    """Supported video formats"""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WMV = "wmv"
    MKV = "mkv"

class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"

class ModelType(str, Enum):
    """Available embedding models"""
    PARAPHRASE_MINI = "paraphrase-MiniLM-L6-v2"
    MPNET_BASE = "all-mpnet-base-v2"
    MULTILINGUAL_MINI = "paraphrase-multilingual-MiniLM-L12-v2"
    MINILM = "all-MiniLM-L6-v2"
    DISTILBERT = "distilbert-base-nli-mean-tokens"
    CLIP = "clip-ViT-B-32"

class SimilarityMethod(str, Enum):
    """Methods for calculating similarity"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"

class VideoProcessorConfig(BaseModel):
    """Configuration for video processing"""
    codec: str = Field("libx264", description="Video codec to use for encoding")
    output_format: VideoFormat = Field(VideoFormat.MP4, description="Output video format")
    frame_rate: Optional[int] = Field(None, description="Output frame rate (fps)")
    resolution: Optional[Tuple[int, int]] = Field(None, description="Output resolution (width, height)")
    
    @field_validator('resolution')
    def validate_resolution(cls, v):
        if v is not None:
            width, height = v
            if width % 2 != 0 or height % 2 != 0:
                raise ValueError("Width and height must be even numbers")
        return v

class VideoToAudioConfig(BaseModel):
    """Configuration for video to audio conversion"""
    output_format: AudioFormat = Field(AudioFormat.WAV, description="Output audio format")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    channels: int = Field(1, description="Number of audio channels (1=mono, 2=stereo)")
    
    @field_validator('sample_rate')
    def validate_sample_rate(cls, v):
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of {valid_rates}")
        return v

class TranscriptionConfig(BaseModel):
    """Configuration for transcription service"""
    chunk_duration: float = Field(60.0, description="Duration of each audio chunk in seconds")
    overlap_duration: float = Field(1.0, description="Overlap between chunks in seconds")
    embedding_model: ModelType = Field(ModelType.PARAPHRASE_MINI, description="Model for sentence embeddings")
    language_code: str = Field("en-US", description="Language code for speech recognition")
    
    @field_validator('chunk_duration')
    def validate_chunk_duration(cls, v):
        if v <= 0:
            raise ValueError("Chunk duration must be positive")
        return v

class BERTAnalyzerConfig(BaseModel):
    """Configuration for BERT Analyzer"""
    model_name: ModelType = Field(ModelType.PARAPHRASE_MINI, description="Model for text embeddings")
    similarity_method: SimilarityMethod = Field(SimilarityMethod.COSINE, description="Method for similarity calculation")
    threshold: float = Field(0.6, description="Threshold for similarity matching")
    output_file: str = Field("./CLIP_results.txt", description="Path to output file")
    
    @field_validator('threshold')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v

class GeminiConfig(BaseModel):
    """Configuration for Gemini Analyzer"""
    model_name: str = Field("gemini-1.5-flash", description="Gemini model to use")
    max_output_tokens: int = Field(2048, description="Maximum number of output tokens")
    temperature: float = Field(0.2, description="Sampling temperature")
    
    @field_validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

# New config models for enhanced flow implementation

class VectorIndexConfig(BaseModel):
    """Configuration for vector indexing service"""
    model_name: ModelType = Field(ModelType.PARAPHRASE_MINI, description="Model for text embeddings")
    index_name: str = Field(..., description="Name of the vector index")
    similarity_method: SimilarityMethod = Field(SimilarityMethod.COSINE, description="Method for similarity calculation")
    dimension: Optional[int] = Field(None, description="Dimension of the vector embeddings")
    store_path: str = Field("./indices", description="Path to store indices")
    
    @field_validator('store_path')
    def validate_store_path(cls, v):
        path = Path(v)
        path.mkdir(exist_ok=True, parents=True)
        return str(path)

class SentimentAnalyzerConfig(BaseModel):
    """Configuration for sentiment analysis service"""
    model_name: str = Field("distilbert-base-uncased-finetuned-sst-2-english", 
                          description="Sentiment analysis model to use")
    threshold: float = Field(0.6, description="Confidence threshold for sentiment classification")
    
    @field_validator('threshold')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v

class ClipGenerationStrategy(str, Enum):
    """Available strategies for clip generation"""
    INDIVIDUAL = "individual"  # Create one clip per segment
    CONSECUTIVE = "consecutive"  # Group consecutive segments
    SINGLE = "single"  # Create one clip with all segments
    TOPIC_BASED = "topic_based"  # Group segments by topic
    SENTIMENT_BASED = "sentiment_based"  # Group segments by sentiment

class EnhancedProcessingConfig(BaseModel):
    """Configuration for enhanced processing flow"""
    vector_index: VectorIndexConfig
    sentiment_analysis: Optional[SentimentAnalyzerConfig] = None
    similarity_threshold: float = Field(0.7, description="Threshold for similarity matching")
    max_segments: int = Field(5, description="Maximum number of segments to include")
    clip_strategy: ClipGenerationStrategy = Field(
        ClipGenerationStrategy.INDIVIDUAL, 
        description="Strategy to use for generating clips"
    )
    query_expansion: bool = Field(False, description="Whether to expand query with related terms")
    min_segment_duration: float = Field(1.0, description="Minimum duration of a segment in seconds")
    only_positive_sentiment: bool = Field(False, description="Filter for only positive sentiment segments")
    
    @field_validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v
        
    @field_validator('max_segments')
    def validate_max_segments(cls, v):
        if v <= 0:
            raise ValueError("Maximum segments must be positive")
        return v