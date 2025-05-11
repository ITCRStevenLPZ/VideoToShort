import os
import logging
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

from core.config import settings
from models.clip_request import (
    EnhancedClipRequest, 
    EnhancedProcessingResponse, 
    ClipResult, 
    SegmentResult,
    SentimentInfo
)
from models.service_models import (
    EnhancedProcessingConfig, 
    ClipGenerationStrategy,
    VectorIndexConfig
)
from services.vector_index_service import VectorIndexService
from services.sentiment_analyzer import SentimentAnalyzer
from services.video_processor import VideoProcessor
from services.video_to_audio import VideoToAudioService
from services.transcription_service import TranscriptionService

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class EnhancedProcessor:
    """
    Enhanced processing service implementing the complete flow from the flowchart.
    Integrates vector indexing, sentiment analysis, and clip generation.
    """
    
    def __init__(self, config: Optional[EnhancedProcessingConfig] = None):
        """
        Initialize the enhanced processor with configuration.
        
        Args:
            config: Optional processing configuration
        """
        # Initialize with default config if none provided
        self.config = config if config else EnhancedProcessingConfig(
            vector_index=VectorIndexConfig(index_name="default_index")
        )
        logger.info(f"Initializing EnhancedProcessor with config: {self.config}")
        
        # Initialize component services
        self.vector_service = VectorIndexService(config=self.config.vector_index)
        
        # Initialize sentiment analyzer if configured
        if self.config.sentiment_analysis:
            self.sentiment_analyzer = SentimentAnalyzer(config=self.config.sentiment_analysis)
            logger.info("Sentiment analysis enabled")
        else:
            self.sentiment_analyzer = None
            logger.info("Sentiment analysis disabled")
            
        # Initialize video processing services
        self.video_processor = VideoProcessor()
        self.video_to_audio = VideoToAudioService(output_directory=settings.OUTPUT_DIR)
        self.transcription_service = TranscriptionService()
        
        # Create output directory if it doesn't exist
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        
    def process_video(self, request: EnhancedClipRequest) -> EnhancedProcessingResponse:
        """
        Process a video according to the enhanced flow.
        
        Args:
            request: The processing request with parameters
            
        Returns:
            EnhancedProcessingResponse with results
        """
        try:
            logger.info(f"Processing video: {request.video_path}")
            video_path = request.video_path
            
            # Validate video path
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            # Get video file basename for naming indices and outputs
            video_filename = Path(video_path).stem
            
            # Step 1: Prepare transcript and index
            # First check if we have an existing index
            index_name = request.index_name or self.config.index_name_template.format(filename=video_filename)
            index_exists = self.vector_service.load_index(index_name)
            
            if not index_exists:
                logger.info(f"No existing index found for {video_filename}")
                if self.config.create_index_if_missing:
                    # This performs the full Indexing Phase from the flowchart
                    logger.info("Creating new index from video")
                    # 1. Convert video to audio
                    logger.info("Converting video to audio...")
                    audio_path = self.video_to_audio.convert_video_to_audio(video_path)
                    
                    # 2. Transcribe audio
                    logger.info("Transcribing audio...")
                    transcript = self.transcription_service.transcribe_audio(audio_path, fileName=video_filename)
                    
                    # Save transcript for reference
                    transcript_path = os.path.join(settings.OUTPUT_DIR, f"{video_filename}_transcript.json")
                    with open(transcript_path, "w") as f:
                        json.dump(transcript, f, indent=2)
                    logger.info(f"Saved transcript to {transcript_path}")
                    
                    # 3. Create vector index from transcript
                    # This handles the entire Indexing Phase:
                    # - Load Transcript JSON
                    # - Chunk into Timestamped Segments
                    # - Compute Embeddings
                    # - Build & Store Vector Index
                    # - Precompute & Store Sentiment (if enabled)
                    logger.info("Creating vector index...")
                    self.vector_service.create_index(transcript_path, index_name)
                    logger.info(f"Created index: {index_name}")
                else:
                    logger.error("No index found and create_index_if_missing is False")
                    raise ValueError("No vector index found for video and automatic creation is disabled")
            
            # Now we have a valid index loaded
            
            # Step 2: Process query (Query Phase)
            # - Receive User Topic (already done via request)
            # - Embed Query
            # - Vector Similarity Search
            # - Retrieve Top-K Segments
            # - Analyze Sentiment (if not precomputed)
            # - Sort Segments Chronologically
            logger.info(f"Querying index with: '{request.query}'")
            
            # Use provided top_k if specified, otherwise use config default
            top_k = request.top_k if request.top_k is not None else self.config.top_k
            
            # Use provided sentiment filter if specified
            sentiment_filter = request.sentiment_filter if request.sentiment_filter is not None else self.config.sentiment_filter
            
            # Search the index
            matching_segments = self.vector_service.query_index(
                query=request.query,
                top_k=top_k,
                sentiment_filter=sentiment_filter
            )
            
            total_segments_found = len(matching_segments)
            logger.info(f"Found {total_segments_found} matching segments")
            
            if total_segments_found == 0:
                # Return early if no segments found
                return EnhancedProcessingResponse(
                    clips=[],
                    total_segments_found=0,
                    total_clips_generated=0,
                    query=request.query,
                    processing_info={"message": "No matching segments found"}
                )
            
            # Step 3: Generate clips from matching segments
            # Convert segments to SegmentResult objects
            segment_results = []
            for i, segment in enumerate(matching_segments):
                # Create SentimentInfo if sentiment is available
                sentiment_info = None
                if segment.get('sentiment') is not None:
                    score = segment['sentiment']
                    sentiment_info = SentimentInfo(
                        score=score,
                        is_positive=score > 0.2,
                        is_negative=score < -0.2,
                        is_neutral=-0.2 <= score <= 0.2
                    )
                
                # Create segment result
                segment_results.append(
                    SegmentResult(
                        id=i,
                        text=segment['text'],
                        start_timestamp=segment['start_timestamp'],
                        end_timestamp=segment['end_timestamp'],
                        similarity=segment.get('similarity', 0.0),
                        sentiment=sentiment_info
                    )
                )
            
            # Generate clips based on the selected strategy
            clip_strategy = request.clip_strategy or self.config.clip_strategy.value
            
            if clip_strategy == ClipGenerationStrategy.SINGLE_SEGMENT.value:
                # Create one clip per segment
                clips = self._generate_individual_clips(segment_results, video_path)
            elif clip_strategy == ClipGenerationStrategy.MERGE_ADJACENT.value:
                # Merge adjacent segments within max_segment_gap
                clips = self._generate_merged_clips(
                    segment_results, 
                    video_path,
                    max_gap=self.config.max_segment_gap,
                    min_duration=request.min_clip_duration or self.config.min_clip_duration,
                    max_duration=request.max_clip_duration or self.config.max_clip_duration
                )
            elif clip_strategy == ClipGenerationStrategy.MERGE_ALL.value:
                # Merge all segments into one clip
                clips = self._generate_single_clip(segment_results, video_path)
            else:
                # Default to merge_adjacent
                clips = self._generate_merged_clips(
                    segment_results, 
                    video_path,
                    max_gap=self.config.max_segment_gap,
                    min_duration=request.min_clip_duration or self.config.min_clip_duration,
                    max_duration=request.max_clip_duration or self.config.max_clip_duration
                )
            
            # Calculate overall sentiment distribution if we have sentiment info
            sentiment_distribution = None
            if self.config.use_sentiment and self.sentiment_analyzer:
                texts = [segment.text for segment in segment_results]
                sentiment_distribution = self.sentiment_analyzer.get_sentiment_distribution(texts)
                
            # Create final response
            response = EnhancedProcessingResponse(
                clips=clips,
                total_segments_found=total_segments_found,
                total_clips_generated=len(clips),
                query=request.query,
                sentiment_distribution=sentiment_distribution,
                processing_info={
                    "video_filename": video_filename,
                    "index_name": index_name,
                    "clip_strategy": clip_strategy,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Processing completed: generated {len(clips)} clips from {total_segments_found} segments")
            return response
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise
    
    def _generate_individual_clips(
        self, 
        segments: List[SegmentResult], 
        video_path: str
    ) -> List[ClipResult]:
        """
        Generate one clip per segment.
        
        Args:
            segments: List of segments to create clips from
            video_path: Path to the source video
            
        Returns:
            List of generated clip results
        """
        clips = []
        
        for segment in segments:
            # Generate a unique ID for this clip
            clip_id = f"clip_{uuid.uuid4().hex[:8]}"
            
            # Create timestamp list for video processor
            timestamp = {
                'start': segment.start_timestamp,
                'end': segment.end_timestamp
            }
            
            # Create clip directory if needed
            clip_dir = os.path.join(settings.OUTPUT_DIR, "clips")
            os.makedirs(clip_dir, exist_ok=True)
            
            # Generate the clip
            logger.info(f"Creating clip for segment: {segment.id} ({timestamp})")
            clip_paths = self.video_processor.create_clips(
                video_path=video_path,
                timestamps=[timestamp],
                output_dir=clip_dir
            )
            
            # Create sentiment info if available
            avg_sentiment = segment.sentiment.score if segment.sentiment else None
            
            # Create clip result
            clip_result = ClipResult(
                clip_id=clip_id,
                segments=[segment],
                start_timestamp=segment.start_timestamp,
                end_timestamp=segment.end_timestamp,
                duration=segment.end_timestamp - segment.start_timestamp,
                text=segment.text,
                clip_path=clip_paths[0] if clip_paths else None,
                average_similarity=segment.similarity,
                average_sentiment=avg_sentiment
            )
            
            clips.append(clip_result)
            
        logger.info(f"Generated {len(clips)} individual clips")
        return clips
    
    def _generate_merged_clips(
        self, 
        segments: List[SegmentResult], 
        video_path: str,
        max_gap: float,
        min_duration: float,
        max_duration: float
    ) -> List[ClipResult]:
        """
        Generate clips by merging adjacent segments within a maximum gap.
        
        Args:
            segments: List of segments to create clips from
            video_path: Path to the source video
            max_gap: Maximum gap between segments to merge (seconds)
            min_duration: Minimum clip duration (seconds)
            max_duration: Maximum clip duration (seconds)
            
        Returns:
            List of generated clip results
        """
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_timestamp)
        
        # Group adjacent segments
        clip_groups = []
        current_group = []
        
        for segment in sorted_segments:
            if not current_group:
                # First segment in a new group
                current_group.append(segment)
            else:
                # Check if this segment is adjacent to the last segment in the current group
                last_segment = current_group[-1]
                gap = segment.start_timestamp - last_segment.end_timestamp
                
                # Check if we should add to current group or start a new one
                current_duration = last_segment.end_timestamp - current_group[0].start_timestamp
                if gap <= max_gap and current_duration + (segment.end_timestamp - segment.start_timestamp) <= max_duration:
                    # Close enough to merge and within max duration
                    current_group.append(segment)
                else:
                    # Start a new group
                    clip_groups.append(current_group)
                    current_group = [segment]
        
        # Add the last group if it exists
        if current_group:
            clip_groups.append(current_group)
            
        logger.info(f"Grouped {len(sorted_segments)} segments into {len(clip_groups)} clip groups")
        
        # Generate clips for each group
        clips = []
        
        for i, group in enumerate(clip_groups):
            # Generate a unique ID for this clip
            clip_id = f"clip_{uuid.uuid4().hex[:8]}"
            
            # Calculate start and end timestamps
            start_timestamp = group[0].start_timestamp
            end_timestamp = group[-1].end_timestamp
            duration = end_timestamp - start_timestamp
            
            # Apply minimum duration by extending clip if needed
            if duration < min_duration:
                # Add half the needed duration to each end
                extend_by = (min_duration - duration) / 2
                start_timestamp = max(0, start_timestamp - extend_by)
                end_timestamp = end_timestamp + extend_by
                duration = end_timestamp - start_timestamp
                logger.debug(f"Extended clip {i} to meet minimum duration: {duration:.2f}s")
            
            # Create timestamp list for video processor
            timestamp = {
                'start': start_timestamp,
                'end': end_timestamp
            }
            
            # Create clip directory if needed
            clip_dir = os.path.join(settings.OUTPUT_DIR, "clips")
            os.makedirs(clip_dir, exist_ok=True)
            
            # Generate the clip
            logger.info(f"Creating clip for group {i+1} ({timestamp})")
            clip_paths = self.video_processor.create_clips(
                video_path=video_path,
                timestamps=[timestamp],
                output_dir=clip_dir
            )
            
            # Calculate average similarity
            avg_similarity = sum(segment.similarity for segment in group) / len(group)
            
            # Calculate average sentiment if available
            sentiment_scores = [segment.sentiment.score for segment in group if segment.sentiment]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
            
            # Combine text
            combined_text = " ".join(segment.text for segment in group)
            
            # Create clip result
            clip_result = ClipResult(
                clip_id=clip_id,
                segments=group,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                duration=duration,
                text=combined_text,
                clip_path=clip_paths[0] if clip_paths else None,
                average_similarity=avg_similarity,
                average_sentiment=avg_sentiment
            )
            
            clips.append(clip_result)
            
        logger.info(f"Generated {len(clips)} merged clips")
        return clips
    
    def _generate_single_clip(
        self, 
        segments: List[SegmentResult], 
        video_path: str
    ) -> List[ClipResult]:
        """
        Generate a single clip containing all segments.
        
        Args:
            segments: List of segments to create clip from
            video_path: Path to the source video
            
        Returns:
            List containing a single clip result
        """
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_timestamp)
        
        # Generate a unique ID for this clip
        clip_id = f"clip_{uuid.uuid4().hex[:8]}"
        
        # Use the start time of the first segment and end time of the last segment
        start_timestamp = sorted_segments[0].start_timestamp
        end_timestamp = sorted_segments[-1].end_timestamp
        duration = end_timestamp - start_timestamp
        
        # Create timestamp list for video processor
        timestamp = {
            'start': start_timestamp,
            'end': end_timestamp
        }
        
        # Create clip directory if needed
        clip_dir = os.path.join(settings.OUTPUT_DIR, "clips")
        os.makedirs(clip_dir, exist_ok=True)
        
        # Generate the clip
        logger.info(f"Creating single clip for all segments ({timestamp})")
        clip_paths = self.video_processor.create_clips(
            video_path=video_path,
            timestamps=[timestamp],
            output_dir=clip_dir
        )
        
        # Calculate average similarity
        avg_similarity = sum(segment.similarity for segment in sorted_segments) / len(sorted_segments)
        
        # Calculate average sentiment if available
        sentiment_scores = [segment.sentiment.score for segment in sorted_segments if segment.sentiment]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else None
        
        # Combine text
        combined_text = " ".join(segment.text for segment in sorted_segments)
        
        # Create clip result
        clip_result = ClipResult(
            clip_id=clip_id,
            segments=sorted_segments,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            duration=duration,
            text=combined_text,
            clip_path=clip_paths[0] if clip_paths else None,
            average_similarity=avg_similarity,
            average_sentiment=avg_sentiment
        )
        
        logger.info(f"Generated single clip containing {len(sorted_segments)} segments")
        return [clip_result]
        
    def get_available_indices(self) -> List[str]:
        """
        Get a list of all available vector indices.
        
        Returns:
            List of index names
        """
        try:
            return self.vector_service.list_indices()
        except Exception as e:
            logger.error(f"Error getting available indices: {str(e)}", exc_info=True)
            return []