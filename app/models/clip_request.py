from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class TopicType(str, Enum):
    """Types of topics that can be analyzed"""
    KEYWORD = "keyword"
    PHRASE = "phrase"
    THEME = "theme"

class Timestamp(BaseModel):
    """Timestamp information with start and end times"""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    
    @field_validator('start', 'end')
    def validate_timestamps(cls, v):
        if v < 0:
            raise ValueError("Timestamp cannot be negative")
        return v
    
    @model_validator
    def validate_timestamp_order(cls, values):
        start = values.get('start')
        end = values.get('end')
        if start is not None and end is not None and start > end:
            raise ValueError(f"Start time ({start}) cannot be greater than end time ({end})")
        return values

class WordTimestamp(BaseModel):
    """Timestamp information for a single word"""
    word: str = Field(..., description="The word text")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")

class Sentence(BaseModel):
    """Sentence with timestamp and optional embedding"""
    text: str = Field(..., description="Sentence text")
    start_timestamp: float = Field(..., description="Start time in seconds")
    end_timestamp: float = Field(..., description="End time in seconds")
    embedding: Optional[List[float]] = Field(None, description="Sentence embedding vector")

class ClipRequest(BaseModel):
    """Request model for video clip generation"""
    video_path: str = Field(..., description="Path to the video file")
    topics: List[str] = Field(..., description="List of topics to search for")
    threshold: Optional[float] = Field(0.6, description="Similarity threshold (0-1)")
    
    @field_validator('threshold')
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v

class ClipAnalysisResult(BaseModel):
    """Result model for clip analysis"""
    text: str = Field(..., description="Matched text content")
    start_timestamp: float = Field(..., description="Start time in seconds")
    end_timestamp: float = Field(..., description="End time in seconds")
    similarity: Optional[float] = Field(None, description="Similarity score (0-1)")

class TranscriptionRequest(BaseModel):
    """Request model for transcription"""
    video_path: str = Field(..., description="Path to the video file")
    chunk_duration: Optional[float] = Field(60.0, description="Duration of each chunk in seconds")
    overlap_duration: Optional[float] = Field(1.0, description="Overlap between chunks in seconds")

# New models for enhanced processing flow

class SentimentInfo(BaseModel):
    """Sentiment information for a segment"""
    score: float = Field(..., description="Sentiment score (-1 to 1)")
    is_positive: bool = Field(..., description="Whether sentiment is positive")
    is_negative: bool = Field(..., description="Whether sentiment is negative")
    is_neutral: bool = Field(..., description="Whether sentiment is neutral")
    
    @field_validator('score')
    def validate_score(cls, v):
        if not -1 <= v <= 1:
            raise ValueError("Sentiment score must be between -1 and 1")
        return v

class SegmentResult(BaseModel):
    """Enhanced segment result with sentiment and similarity information"""
    id: int = Field(..., description="Segment ID")
    text: str = Field(..., description="Segment text content")
    start_timestamp: float = Field(..., description="Start time in seconds")
    end_timestamp: float = Field(..., description="End time in seconds")
    similarity: float = Field(..., description="Similarity score (0-1)")
    sentiment: Optional[SentimentInfo] = Field(None, description="Sentiment information")
    
    @field_validator('similarity')
    def validate_similarity(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Similarity must be between 0 and 1")
        return v

class ClipResult(BaseModel):
    """Result model for a generated clip with enhanced information"""
    clip_id: str = Field(..., description="Unique clip identifier")
    segments: List[SegmentResult] = Field(..., description="Segments in this clip")
    start_timestamp: float = Field(..., description="Start time in seconds")
    end_timestamp: float = Field(..., description="End time in seconds")
    duration: float = Field(..., description="Duration in seconds")
    text: str = Field(..., description="Combined text content")
    clip_path: Optional[str] = Field(None, description="Path to the generated clip file")
    average_similarity: float = Field(..., description="Average similarity score")
    average_sentiment: Optional[float] = Field(None, description="Average sentiment score")

class EnhancedClipRequest(BaseModel):
    """Enhanced request model for the new flow"""
    video_path: str = Field(..., description="Path to the video file")
    query: str = Field(..., description="Search query text")
    index_name: Optional[str] = Field(None, description="Name of the vector index to use")
    top_k: Optional[int] = Field(5, description="Number of top segments to return")
    sentiment_filter: Optional[float] = Field(None, description="Minimum sentiment threshold (-1 to 1)")
    clip_strategy: Optional[str] = Field("merge_adjacent", description="Strategy for clip generation")
    min_clip_duration: Optional[float] = Field(5.0, description="Minimum clip duration in seconds")
    max_clip_duration: Optional[float] = Field(60.0, description="Maximum clip duration in seconds")
    
    @field_validator('sentiment_filter')
    def validate_sentiment_filter(cls, v):
        if v is not None and not -1 <= v <= 1:
            raise ValueError("Sentiment filter must be between -1 and 1")
        return v
        
    @field_validator('top_k')
    def validate_top_k(cls, v):
        if v is not None and v <= 0:
            raise ValueError("top_k must be positive")
        return v

class EnhancedProcessingResponse(BaseModel):
    """Response model for the enhanced processing flow"""
    clips: List[ClipResult] = Field(..., description="Generated clips")
    total_segments_found: int = Field(..., description="Total number of matching segments found")
    total_clips_generated: int = Field(..., description="Total number of clips generated")
    query: str = Field(..., description="Original query text")
    sentiment_distribution: Optional[Dict[str, float]] = Field(None, description="Overall sentiment distribution")
    processing_info: Dict[str, Any] = Field({}, description="Additional processing information")