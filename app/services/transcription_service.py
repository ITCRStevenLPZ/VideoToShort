import os
import json
import logging
from google.cloud import speech
import torchaudio
from sentence_transformers import SentenceTransformer
import spacy
import subprocess
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union

from models.service_models import TranscriptionConfig, ModelType
from models.clip_request import Sentence, WordTimestamp

# Get a logger specific to this module
logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """
        Initialize the TranscriptionService with optional configuration.
        
        Args:
            config: Optional TranscriptionConfig containing transcription settings
        """
        logger.info("Initializing TranscriptionService")
        
        # Use provided config or create default one
        self.config = config if config else TranscriptionConfig()
        logger.debug(f"TranscriptionService config: {self.config.dict()}")
        
        try:
            # Load SpaCy for sentence detection
            logger.debug("Loading SpaCy model for sentence detection")
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load a pre-trained SentenceTransformer model for embeddings
            embedding_model_name = self.config.embedding_model.value
            logger.debug(f"Loading SentenceTransformer model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            
            # Initialize Google Speech-to-Text client
            logger.debug("Initializing Google Speech-to-Text client")
            self.speech_client = speech.SpeechClient()
            
            logger.info("TranscriptionService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TranscriptionService: {str(e)}", exc_info=True)
            raise

    def transcribe_audio(
        self, 
        audio_path: str, 
        fileName: str = "output", 
        chunk_duration: Optional[float] = None,
        overlap_duration: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe the given audio input into text with word-level timestamps and embeddings, in chunks.

        Args:
            audio_path: Path to the audio file
            fileName: Base name for the output JSON file
            chunk_duration: Duration of each audio chunk in seconds (overrides config if provided)
            overlap_duration: Overlap between chunks to prevent sentence splitting (overrides config if provided)
            
        Returns:
            List of dictionaries, each containing a sentence, its start/end timestamp, and sentence embedding
        """
        # Use provided parameters or defaults from config
        chunk_duration = chunk_duration if chunk_duration is not None else self.config.chunk_duration
        overlap_duration = overlap_duration if overlap_duration is not None else self.config.overlap_duration
        
        logger.info(f"Starting audio transcription for {audio_path}")
        logger.debug(f"Parameters: fileName={fileName}, chunk_duration={chunk_duration}, overlap_duration={overlap_duration}")
        
        output_file = f"./{fileName}.json"
        logger.debug(f"Output will be saved to {output_file}")

        # Check if the output file already exists
        if os.path.exists(output_file):
            logger.info(f"Found existing transcription at {output_file}, loading instead of re-transcribing")
            try:
                with open(output_file, "r") as f:
                    sentences_data = json.load(f)
                
                # Validate with Pydantic
                validated_sentences = []
                for sentence_data in sentences_data:
                    try:
                        # Create and validate the Sentence model
                        sentence = Sentence(**sentence_data)
                        validated_sentences.append(sentence_data)
                    except Exception as e:
                        logger.warning(f"Invalid sentence in transcription: {e}")
                
                logger.info(f"Loaded {len(validated_sentences)} sentences from existing transcription")
                return validated_sentences
            except json.JSONDecodeError as e:
                logger.warning(f"Error loading existing transcription: {str(e)}. Will re-transcribe.")
                # Continue with transcription if file exists but can't be loaded

        try:
            # Convert audio to FLAC format required by Google Speech-to-Text
            logger.info("Converting audio to FLAC format")
            audio_flac_path = self.convert_to_flac(audio_path)
            logger.debug(f"Audio converted to FLAC: {audio_flac_path}")

            # Load the audio using torchaudio
            logger.debug(f"Loading audio file with torchaudio")
            audio, sample_rate = torchaudio.load(audio_flac_path)
            logger.info(f"Audio loaded: duration={audio.size(1)/sample_rate:.2f}s, sample_rate={sample_rate}Hz")

            # Calculate the chunk size and overlap in terms of samples
            chunk_size = int(chunk_duration * sample_rate)  # Chunk size in samples
            overlap_size = int(overlap_duration * sample_rate)  # Overlap size in samples
            num_chunks = (audio.size(1) + chunk_size - 1) // chunk_size  # Number of chunks
            logger.info(f"Will process audio in {num_chunks} chunks of {chunk_duration}s with {overlap_duration}s overlap")

            transcription_text = ""
            word_timestamps_data = []

            # Iterate through the audio and transcribe each chunk
            for i in tqdm(range(num_chunks), desc="Transcribing", unit="chunk"):
                logger.info(f"Processing chunk {i+1}/{num_chunks}")
                start_idx = max(i * chunk_size - overlap_size, 0)
                end_idx = min((i + 1) * chunk_size, audio.size(1))
                audio_chunk = audio[:, start_idx:end_idx]
                chunk_duration_actual = (end_idx - start_idx) / sample_rate
                logger.debug(f"Chunk {i+1} duration: {chunk_duration_actual:.2f}s")

                # Convert the chunk back to FLAC
                chunk_flac_path = f"{audio_flac_path.replace('.flac', '')}_chunk_{i}.flac"
                logger.debug(f"Saving chunk to temporary file: {chunk_flac_path}")
                torchaudio.save(chunk_flac_path, audio_chunk, sample_rate)

                # Transcribe the chunk using Google Speech-to-Text
                logger.debug(f"Transcribing chunk {i+1} with Google Speech-to-Text")
                chunk_transcription, chunk_word_timestamps = self.transcribe_with_google(
                    chunk_flac_path, 
                    i * chunk_duration - (overlap_duration if i > 0 else 0)
                )
                logger.debug(f"Chunk {i+1} transcription: {len(chunk_word_timestamps)} words, {len(chunk_transcription)} chars")

                # Accumulate the transcription and word-level timestamps
                transcription_text += chunk_transcription + " "
                word_timestamps_data.extend(chunk_word_timestamps)

                # Remove the temporary chunk file
                logger.debug(f"Removing temporary chunk file: {chunk_flac_path}")
                os.remove(chunk_flac_path)

            logger.info(f"Complete transcription: {len(word_timestamps_data)} words, {len(transcription_text)} chars")

            # Now, detect sentences and generate embeddings
            logger.info("Generating sentences with embeddings")
            
            # Validate word timestamps with Pydantic
            validated_word_timestamps = []
            for word_data in word_timestamps_data:
                try:
                    word_timestamp = WordTimestamp(**word_data)
                    validated_word_timestamps.append(word_data)
                except Exception as e:
                    logger.warning(f"Invalid word timestamp: {e}")
            
            sentences = self.generate_sentences_with_embeddings(transcription_text.strip(), validated_word_timestamps)
            logger.info(f"Generated {len(sentences)} sentences with embeddings")

            # Save the transcription to the JSON file
            logger.info(f"Saving transcription to {output_file}")
            with open(output_file, "w") as f:
                json.dump(sentences, f, indent=4)
            logger.info("Transcription saved successfully")

            return sentences
            
        except Exception as e:
            logger.error(f"Error during audio transcription: {str(e)}", exc_info=True)
            raise

    def convert_to_flac(self, audio_path: str) -> str:
        """Convert audio file to FLAC format using ffmpeg"""
        try:
            flac_path = audio_path.replace(".wav", ".flac")
            logger.debug(f"Converting {audio_path} to FLAC format: {flac_path}")
            
            cmd = ["ffmpeg", "-i", audio_path, flac_path, "-y"]
            logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            
            process = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True,
                text=True
            )
            
            logger.debug("FLAC conversion completed successfully")
            return flac_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed: {str(e)}", exc_info=True)
            logger.debug(f"ffmpeg stdout: {e.stdout}")
            logger.debug(f"ffmpeg stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error converting to FLAC: {str(e)}", exc_info=True)
            raise

    def transcribe_with_google(self, audio_flac_path: str, offset_time: float) -> tuple[str, List[Dict[str, Any]]]:
        """
        Use Google Cloud Speech-to-Text API to transcribe audio with word-level timestamps.

        Args:
            audio_flac_path: Path to the FLAC audio file
            offset_time: The start time of this chunk relative to the full audio
            
        Returns:
            Tuple of (transcription_text, word_timestamps)
        """
        try:
            logger.debug(f"Reading audio file: {audio_flac_path}")
            with open(audio_flac_path, "rb") as audio_file:
                content = audio_file.read()
            logger.debug(f"Read {len(content)} bytes from audio file")

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
                sample_rate_hertz=16000,
                language_code=self.config.language_code,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True  # Enable automatic punctuation
            )

            logger.debug("Sending request to Google Speech-to-Text API")
            response = self.speech_client.recognize(config=config, audio=audio)
            logger.debug(f"Received response with {len(response.results)} results")

            transcription_text = ""
            word_timestamps = []

            for i, result in enumerate(response.results):
                alternative = result.alternatives[0]
                transcription_text += alternative.transcript + " "
                logger.debug(f"Result {i+1}: {len(alternative.words)} words")
                
                for word_info in alternative.words:
                    word = word_info.word
                    start_time = word_info.start_time.total_seconds() + offset_time
                    end_time = word_info.end_time.total_seconds() + offset_time
                    
                    word_timestamp = WordTimestamp(
                        word=word,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    word_timestamps.append(word_timestamp.dict())

            logger.debug(f"Transcription completed: {len(word_timestamps)} words")
            return transcription_text.strip(), word_timestamps
            
        except Exception as e:
            logger.error(f"Error in Google Speech-to-Text transcription: {str(e)}", exc_info=True)
            raise

    def generate_sentences_with_embeddings(
        self, 
        transcription_text: str, 
        word_timestamps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate sentences from transcription text and assign real timestamps to them.

        Args:
            transcription_text: Full transcription text
            word_timestamps: Word-level timestamps
            
        Returns:
            List of sentences with start/end timestamps and embeddings
        """
        try:
            logger.debug("Generating sentences from transcription text")
            sentences = []
            
            # Check if we have text to process
            if not transcription_text.strip():
                logger.warning("Empty transcription text, no sentences to generate")
                return sentences
                
            # Check if we have enough word timestamps
            if not word_timestamps or len(word_timestamps) < 1:
                logger.warning("No word timestamps available, can't generate sentences with timestamps")
                return sentences
                
            # Process the text with SpaCy to detect sentences
            logger.debug("Processing text with SpaCy for sentence detection")
            doc = self.nlp(transcription_text)
            
            sentence_start_idx = 0
            for i, sentence in enumerate(doc.sents):
                sentence_text = sentence.text.strip()
                if not sentence_text:  # Skip empty sentences
                    logger.debug(f"Skipping empty sentence at position {i}")
                    continue
                    
                sentence_words = sentence_text.split()
                logger.debug(f"Processing sentence {i+1}: {len(sentence_words)} words")
                
                # Check if we have enough words left in the word_timestamps
                if sentence_start_idx + len(sentence_words) > len(word_timestamps):
                    logger.warning(
                        f"Sentence word count exceeds remaining timestamps. "
                        f"Words needed: {len(sentence_words)}, Words left: {len(word_timestamps) - sentence_start_idx}"
                    )
                    # Use the available timestamps and adjust
                    end_idx = len(word_timestamps) - 1
                else:
                    end_idx = sentence_start_idx + len(sentence_words) - 1
                
                # Get the start and end timestamps from the first and last word in the sentence
                start_time = word_timestamps[sentence_start_idx]["start_time"]
                end_time = word_timestamps[end_idx]["end_time"]
                logger.debug(f"Sentence {i+1} timestamps: {start_time:.2f}s to {end_time:.2f}s")

                # Generate sentence embedding
                logger.debug(f"Generating embedding for sentence {i+1}")
                sentence_embedding = self.embed_text(sentence_text).cpu().tolist()

                # Create sentence with Pydantic model and convert to dict
                sentence_obj = Sentence(
                    text=sentence_text,
                    start_timestamp=start_time,
                    end_timestamp=end_time,
                    embedding=sentence_embedding
                )
                
                sentences.append(sentence_obj.dict())

                sentence_start_idx += len(sentence_words)

            logger.info(f"Generated {len(sentences)} sentences with timestamps and embeddings")
            return sentences
            
        except Exception as e:
            logger.error(f"Error generating sentences with embeddings: {str(e)}", exc_info=True)
            raise

    def embed_text(self, text: str):
        """Generate sentence embedding using sentence-transformers"""
        try:
            return self.embedding_model.encode(text, convert_to_tensor=True)
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}", exc_info=True)
            raise