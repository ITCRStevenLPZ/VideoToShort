import subprocess
from moviepy.editor import VideoFileClip
import os

class VideoToAudioService:
    def __init__(self, output_directory='./'):
        """
        Initialize the service with an optional output directory.
        
        :param output_directory: The directory where the .wav file will be saved.
        """
        self.output_directory = output_directory

    def convert_video_to_audio(self, video_path: str):
        """
        Convert the input video file to a .wav audio file.
        
        :param video_path: Path to the input video file.
        :return: Path to the saved .wav audio file.
        """
        # Generate the output audio filename based on the original video filename
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_audio_filename = f"{base_filename}.wav"
        output_audio_path = os.path.join(self.output_directory, output_audio_filename)
        _output_audio_path = f"{output_audio_path[:-4]}_16k.wav"

        print(f"Converting video to audio: {video_path} -> {output_audio_path}")

        # Check if the audio file already exists
        if os.path.exists(_output_audio_path):
            return _output_audio_path

        # Load the video file
        video_clip = VideoFileClip(video_path)

        try:
            # Extract the audio from the video and save it as a .wav file
            audio = video_clip.audio
            audio.write_audiofile(output_audio_path, codec='pcm_s16le')  # PCM format for .wav files
        except Exception as e:
            # If an error occurs, remove the partially created audio file
            if os.path.exists(output_audio_path):
                os.remove(output_audio_path)
            raise e
        finally:
            # Close the clips to free resources
            video_clip.close()
            # Convert the audio to 16kHz mono format using ffmpeg
           

            subprocess.run([
                'ffmpeg', '-i', output_audio_path, '-ar', '16000', '-ac', '1', _output_audio_path, '-y'
            ], check=True)

        return _output_audio_path