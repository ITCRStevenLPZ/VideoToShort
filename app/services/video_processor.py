
import logging
import os
from typing import List, Dict, Any, Optional

from moviepy import VideoFileClip

from models.service_models import VideoProcessorConfig, VideoFormat
from models.clip_request import Timestamp

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, config: Optional[VideoProcessorConfig] = None):
        """
        Initialize the VideoProcessor with optional configuration.
        
        Args:
            config: Optional VideoProcessorConfig containing processing settings
        """
        # Use provided config or create default one
        self.config = config if config else VideoProcessorConfig()
        logger.debug(f"VideoProcessor initialized with config: {self.config.dict()}")
        
    def create_clips(
        self, 
        video_path: str, 
        timestamps: List[Dict[str, float]], 
        output_dir: str
    ) -> List[str]:
        """
        Create video clips from the given video file based on specified timestamps.
        
        Args:
            video_path: Path to the source video file
            timestamps: List of dictionaries containing start and end times for each clip
            output_dir: Directory where clips will be saved
            
        Returns:
            List of paths to the created clip files
        """
        clips = []
        try:
            logger.info(f"Creating video clips from '{video_path}'")
            logger.debug(f"Using timestamps: {timestamps}")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Ensured output directory exists: {output_dir}")
            
            # Load the video file
            logger.info("Loading source video file")
            video = VideoFileClip(video_path)
            logger.debug(f"Video loaded successfully. Duration: {video.duration} seconds")
            
            # Process each timestamp
            for i, timestamp_dict in enumerate(timestamps):
                # Validate timestamp with Pydantic
                timestamp = Timestamp(**timestamp_dict)
                logger.info(f"Processing clip {i+1}/{len(timestamps)}: {timestamp.start} to {timestamp.end}")
                
                # Validate timestamps against video duration
                if timestamp.start < 0 or timestamp.end > video.duration:
                    logger.warning(
                        f"Timestamp out of bounds: {timestamp.start}-{timestamp.end}, "
                        f"video duration: {video.duration}"
                    )
                    # Adjust timestamps to fit within video duration
                    original_start, original_end = timestamp.start, timestamp.end
                    timestamp.start = max(0, timestamp.start)
                    timestamp.end = min(timestamp.end, video.duration)
                    logger.info(
                        f"Adjusted timestamp from {original_start}-{original_end} "
                        f"to {timestamp.start}-{timestamp.end}"
                    )
                
                # Create the subclip
                clip = video.subclip(timestamp.start, timestamp.end)
                
                # Apply optional configurations from VideoProcessorConfig
                if self.config.frame_rate is not None:
                    clip = clip.set_fps(self.config.frame_rate)
                    logger.debug(f"Set frame rate to {self.config.frame_rate} fps")
                    
                if self.config.resolution is not None:
                    width, height = self.config.resolution
                    clip = clip.resize((width, height))
                    logger.debug(f"Resized clip to {width}x{height}")
                
                # Generate output filename
                extension = self.config.output_format.value
                clip_filename = f"{timestamp.start}_{timestamp.end}.{extension}"
                clip_path = os.path.join(output_dir, clip_filename)
                logger.info(f"Writing clip to: {clip_path}")
                
                # Write the clip with progress logging
                clip.write_videofile(
                    clip_path, 
                    codec=self.config.codec,
                    logger=None  # Disable moviepy's own logger to avoid duplicate logs
                )
                
                logger.info(f"Clip {i+1} created successfully")
                clips.append(clip_path)
            
            logger.info(f"All {len(clips)} clips created successfully")
            return clips
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up video object if it exists
            if 'video' in locals():
                logger.debug("Closing video file")
                video.close()