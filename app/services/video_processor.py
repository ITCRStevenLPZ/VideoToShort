from moviepy.editor import VideoFileClip
import logging
import os

logger = logging.getLogger(__name__)

class VideoProcessor:
    def create_clips(self, video_path: str, timestamps: list[dict], output_dir: str):
        clips = []
        try:
            video = VideoFileClip(video_path)
            for timestamp in timestamps:
                start, end = timestamp['start'], timestamp['end']  # Keep full precision of start and end times
                clip = video.subclip(start, end)  # Subclip with exact start and end times
                clip_path = os.path.join(output_dir, f"{start}_{end}.mp4")
                clip.write_videofile(clip_path, codec="libx264")
                clips.append(clip_path)
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        return clips