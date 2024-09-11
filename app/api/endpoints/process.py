import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
from services.bert_analizer import BERTAnalyzer
from services.gemini_analyzer import GeminiAnalyzer
from services.transcription_service import TranscriptionService
from services.video_processor import VideoProcessor
import logging
import tempfile
import os

from services.video_to_audio import VideoToAudioService

router = APIRouter()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

@router.post("/process/")
async def process_clips(
    video: UploadFile = File(...), 
    topics: List[str] = Form(...),
    chunkDivider: int = Form(...),
    threshold: float = Form(...),
    chunkSize: int = Form(...),
    fileName: str = Form(...)
):
    bert_analyzer = BERTAnalyzer()
    processor = VideoProcessor()
    # transcriptor = TranscriptionService()
    video_to_audio = VideoToAudioService()

    try:
        # Create a temporary file to save the video
        temp_video_path = f"./{fileName}"
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(video.file.read())

        # Ensure the output directory exists
        output_dir = "clips"
        os.makedirs(output_dir, exist_ok=True)

        # Convert the video to audio
        # output_audio_path = video_to_audio.convert_video_to_audio(temp_video_path)

        # Transcribe the audio to obtain sentence-level timestamps
        # transcription = transcriptor.transcribe_audio(output_audio_path, chunk_duration=chunkDivider, fileName=fileName)
        transcription = []

        output_file = f"./{fileName}.json"

        logging.info(f"Transcription file: {output_file}")

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                sentences = json.load(f)
                transcription = sentences
        
        logging.info(f"Transcription was loaded from file: {output_file}")

        # Analyze the transcription with BERT
        transcript_analysis = bert_analyzer.calculate_similarity(transcription, topics=topics, threshold=threshold)

        # Extract the timestamps from the top 3 chunks
        timestamps = []
        for entry in transcript_analysis[:3]:
            try:
                timestamps.append({'start': entry['start_timestamp'], 'end': entry['end_timestamp']})
            except ValueError as e:
                logging.error(f"Invalid timestamp format: {entry}")
                raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {entry}")

        # Process the video clips
        clips = processor.create_clips(temp_video_path, timestamps, output_dir)
        return {"clips": clips}
    except Exception as e:
        logging.error(f"Failed to process clips: {e}")
        raise HTTPException(status_code=500, detail="Failed to process video clips")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@router.post("/transcribe/")
async def transcribe_video(
    video: UploadFile = File(...), 
    chunkDivider: int = Form(...),
    fileName: str = Form(...)
):
    transcriptor = TranscriptionService()
    video_to_audio = VideoToAudioService()

    try:
        temp_video_path = f"./{fileName}"
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(video.file.read())

        # Convert the video to audio
        output_audio_path = video_to_audio.convert_video_to_audio(temp_video_path)

        # Transcribe the audio to obtain sentence-level timestamps
        transcription = transcriptor.transcribe_audio(output_audio_path, chunk_duration=chunkDivider, fileName=fileName)

        return {"transcription": transcription}
    except Exception as e:
        logging.error(f"Failed to transcribe video: {e}")
        raise HTTPException(status_code=500, detail="Failed to transcribe video")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)