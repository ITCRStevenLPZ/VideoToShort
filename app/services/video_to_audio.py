import subprocess
import os
import logging
from typing import Optional

from moviepy import VideoFileClip

from models.service_models import VideoToAudioConfig, AudioFormat

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class VideoToAudioService:
    def __init__(self, output_directory: str = './', config: Optional[VideoToAudioConfig] = None):
        """
        Initialize the service with an output directory and configuration.
        
        Args:
            output_directory: The directory where the audio files will be saved
            config: Optional audio conversion configuration
        """
        self.output_directory = output_directory
        self.config = config if config else VideoToAudioConfig()
        logger.debug(f"VideoToAudioService initialized with output directory: {output_directory}")
        logger.debug(f"Using audio config: {self.config.dict()}")

    def convert_video_to_audio(self, video_path: str) -> str:
        """
        Convert the input video file to an audio file.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Path to the saved audio file
        """
        # Generate the output audio filename based on the original video filename
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_audio_filename = f"{base_filename}.{self.config.output_format.value}"
        output_audio_path = os.path.join(self.output_directory, output_audio_filename)
        _output_audio_path = f"{output_audio_path[:-4]}_{self.config.sample_rate}hz.{self.config.output_format.value}"

        logger.info(f"Converting video to audio: {video_path} -> {_output_audio_path}")

        # Check if the audio file already exists
        if os.path.exists(_output_audio_path):
            logger.info(f"Audio file already exists at {_output_audio_path}, skipping conversion")
            return _output_audio_path

        # Load the video file
        logger.debug(f"Loading video file: {video_path}")
        video_clip = None
        try:
            video_clip = VideoFileClip(video_path)
            logger.debug(f"Video loaded successfully. Duration: {video_clip.duration} seconds")

            # Extract the audio from the video and save it as an audio file
            logger.info("Extracting audio from video")
            audio = video_clip.audio
            if audio is None:
                logger.warning("Video has no audio track")
                raise ValueError("The video file does not contain an audio track")
            
            # Determine codec based on format
            codec = self._get_codec_for_format(self.config.output_format)
            
            logger.info(f"Writing audio to file: {output_audio_path}")
            audio.write_audiofile(
                output_audio_path, 
                codec=codec,
                logger=None  # Disable moviepy's own logger
            )
            logger.debug(f"Audio written to {output_audio_path}")
            
            # Convert the audio to the proper format with sample rate
            logger.info(f"Converting audio to {self.config.sample_rate}Hz {self.config.channels}-channel format")
            conversion_cmd = [
                'ffmpeg', 
                '-i', output_audio_path, 
                '-ar', str(self.config.sample_rate), 
                '-ac', str(self.config.channels), 
                _output_audio_path, 
                '-y'
            ]
            logger.debug(f"Running ffmpeg command: {' '.join(conversion_cmd)}")
            
            process = subprocess.run(
                conversion_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.debug(f"ffmpeg conversion completed successfully")
            
            # Remove original audio file after successful conversion
            if os.path.exists(_output_audio_path) and os.path.exists(output_audio_path):
                logger.debug(f"Removing intermediate audio file: {output_audio_path}")
                os.remove(output_audio_path)
            
            logger.info(f"Audio conversion completed successfully: {_output_audio_path}")
            return _output_audio_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed: {str(e)}", exc_info=True)
            logger.debug(f"ffmpeg stdout: {e.stdout}")
            logger.debug(f"ffmpeg stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error converting video to audio: {str(e)}", exc_info=True)
            # If an error occurs, remove the partially created audio files
            for path in [output_audio_path, _output_audio_path]:
                if os.path.exists(path):
                    logger.debug(f"Removing partial audio file: {path}")
                    os.remove(path)
            raise
        finally:
            # Close the video clip to free resources
            if video_clip is not None:
                logger.debug("Closing video clip")
                video_clip.close()
                
    def _get_codec_for_format(self, format: AudioFormat) -> str:
        """
        Get the appropriate codec for the specified audio format
        
        Args:
            format: The audio format
            
        Returns:
            Codec string to use with moviepy
        """
        codec_map = {
            AudioFormat.WAV: 'pcm_s16le',
            AudioFormat.FLAC: 'flac',
            AudioFormat.MP3: 'libmp3lame'
        }
        return codec_map.get(format, 'pcm_s16le')