import json
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import List, Any
import logging
import tempfile
import os

from services.bert_analizer import BERTAnalyzer
from services.transcription_service import TranscriptionService
from services.video_processor import VideoProcessor
from services.video_to_audio import VideoToAudioService

from models.clip_request import (
    ClipRequest, 
    TranscriptionRequest, 
    Timestamp, 
    Sentence, 
    ClipResponse, 
    ClipAnalysisResult,
    TranscriptionResult
)
from core.config import settings

# Get a logger specific to this module
logger = logging.getLogger(__name__)

router = APIRouter(tags=["video-processing"])

@router.post("/process/", response_model=ClipResponse)
async def process_clips(
    video: UploadFile = File(...), 
    topics: List[str] = Form(...),
    chunk_divider: int = Form(...),
    threshold: float = Form(...),
    chunk_size: int = Form(...),
    file_name: str = Form(...)
):
    # Create Pydantic model instance for validation
    clip_request = ClipRequest(
        topics=topics,
        chunk_divider=chunk_divider,
        threshold=threshold,
        chunk_size=chunk_size,
        file_name=file_name
    )
    
    logger.info(f"Processing video clip with topics: {clip_request.topics}")
    bert_analyzer = BERTAnalyzer()
    processor = VideoProcessor()
    video_to_audio = VideoToAudioService()

    try:
        # Create a temporary file to save the video
        temp_video_path = f"./{clip_request.file_name}"
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(video.file.read())
        logger.debug(f"Video saved temporarily to {temp_video_path}")

        # Ensure the output directory exists
        output_dir = settings.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        # Load existing transcription if available
        transcription_sentences = []
        output_file = f"./{clip_request.file_name}.json"
        
        if os.path.exists(output_file):
            logger.info(f"Found existing transcription file: {output_file}")
            with open(output_file, "r") as f:
                sentences_data = json.load(f)
                # Validate each sentence with Pydantic
                for sentence_data in sentences_data:
                    try:
                        sentence = Sentence(**sentence_data)
                        transcription_sentences.append(sentence_data)
                    except Exception as e:
                        logger.warning(f"Invalid sentence data: {e}")
            logger.info(f"Loaded transcription with {len(transcription_sentences)} sentences")
        else:
            logger.warning(f"No transcription file found at {output_file}. An empty transcription will be used.")

        # Analyze the transcription with BERT
        logger.info(f"Starting BERT analysis with threshold: {clip_request.threshold}")
        transcript_analysis = bert_analyzer.calculate_similarity(
            transcription_sentences, 
            topics=clip_request.topics, 
            threshold=clip_request.threshold
        )
        
        if not transcript_analysis:
            logger.warning("BERT analysis returned no results")
            return ClipResponse(clips=[], message="No relevant content found for the specified topics")
        
        # Make transcript_analysis into a list if it's not already
        if not isinstance(transcript_analysis, list):
            transcript_analysis = [transcript_analysis]
            
        logger.info(f"BERT analysis found {len(transcript_analysis)} relevant segments")

        # Extract the timestamps from the analysis results (take up to 3)
        timestamps = []
        for entry in transcript_analysis[:3]:
            try:
                # Validate timestamp with Pydantic
                timestamp = Timestamp(start=entry['start_timestamp'], end=entry['end_timestamp'])
                timestamps.append({'start': timestamp.start, 'end': timestamp.end})
                logger.debug(f"Added timestamp: {timestamp.start} to {timestamp.end}")
            except (ValueError, KeyError) as e:
                logger.error(f"Invalid timestamp format in entry: {entry}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {entry}")

        # Process the video clips
        logger.info(f"Creating {len(timestamps)} video clips")
        clips = processor.create_clips(temp_video_path, timestamps, output_dir)
        logger.info(f"Successfully created {len(clips)} clips")
        
        # Create response using Pydantic model
        return ClipResponse(clips=clips)
    except Exception as e:
        logger.error(f"Failed to process clips: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process video clips: {str(e)}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_video_path):
            logger.debug(f"Removing temporary file: {temp_video_path}")
            os.remove(temp_video_path)

@router.post("/transcribe/", response_model=TranscriptionResult)
async def transcribe_video(
    video: UploadFile = File(...), 
    chunk_divider: int = Form(...),
    file_name: str = Form(...)
):
    # Create Pydantic model instance for validation
    transcription_request = TranscriptionRequest(
        chunk_divider=chunk_divider,
        file_name=file_name
    )
    
    logger.info(f"Transcribing video with filename: {transcription_request.file_name}")
    transcriptor = TranscriptionService()
    video_to_audio = VideoToAudioService()

    try:
        temp_video_path = f"./{transcription_request.file_name}"
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(video.file.read())
        logger.debug(f"Video saved temporarily to {temp_video_path}")

        # Convert the video to audio
        logger.info("Converting video to audio")
        output_audio_path = video_to_audio.convert_video_to_audio(temp_video_path)
        logger.debug(f"Audio file created at {output_audio_path}")

        # Transcribe the audio to obtain sentence-level timestamps
        logger.info(f"Starting transcription with chunk duration: {transcription_request.chunk_divider}")
        transcription = transcriptor.transcribe_audio(
            output_audio_path, 
            chunk_duration=transcription_request.chunk_divider, 
            fileName=transcription_request.file_name
        )
        logger.info(f"Transcription completed with {len(transcription)} sentences")

        # Validate sentences with Pydantic
        validated_sentences = []
        for sentence_data in transcription:
            try:
                sentence = Sentence(**sentence_data)
                validated_sentences.append(sentence_data)
            except Exception as e:
                logger.warning(f"Invalid sentence data: {e}")
        
        # Create response using Pydantic model
        return TranscriptionResult(sentences=validated_sentences)
    except Exception as e:
        logger.error(f"Failed to transcribe video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to transcribe video: {str(e)}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_video_path):
            logger.debug(f"Removing temporary file: {temp_video_path}")
            os.remove(temp_video_path)