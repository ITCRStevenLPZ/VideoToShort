import os
import json
from google.cloud import speech
import torchaudio
from sentence_transformers import SentenceTransformer
import spacy
import subprocess
from tqdm import tqdm


class TranscriptionService:
    def __init__(self, embedding_model_name: str = "paraphrase-MiniLM-L6-v2"):
        # Load SpaCy for sentence detection
        self.nlp = spacy.load("en_core_web_sm")

        # Load a pre-trained SentenceTransformer model for embeddings
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Initialize Google Speech-to-Text client
        self.speech_client = speech.SpeechClient()

    def transcribe_audio(self, audio_path, fileName="output", chunk_duration=60.0, overlap_duration=1.0):
        """
        Transcribe the given audio input into text with word-level timestamps and embeddings, in chunks.

        :param audio_path: Path to the audio file.
        :param fileName: Base name for the output JSON file.
        :param chunk_duration: Duration of each audio chunk in seconds.
        :param overlap_duration: Overlap between chunks to prevent sentence splitting.
        :return: List of dictionaries, each containing a sentence, its start/end timestamp, and sentence embedding.
        """
        output_file = f"./{fileName}.json"

        # Check if the output file already exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                sentences = json.load(f)
                return sentences

        # Convert audio to FLAC format required by Google Speech-to-Text
        audio_flac_path = self.convert_to_flac(audio_path)

        # Load the audio using torchaudio
        audio, sample_rate = torchaudio.load(audio_flac_path)

        # Calculate the chunk size and overlap in terms of samples
        chunk_size = int(chunk_duration * sample_rate)  # Chunk size in samples
        overlap_size = int(overlap_duration * sample_rate)  # Overlap size in samples
        num_chunks = (audio.size(1) + chunk_size - 1) // chunk_size  # Number of chunks

        transcription_text = ""
        word_timestamps = []

        # Iterate through the audio and transcribe each chunk
        for i in tqdm(range(num_chunks), desc="Transcribing", unit="chunk"):
            start_idx = max(i * chunk_size - overlap_size, 0)
            end_idx = min((i + 1) * chunk_size, audio.size(1))
            audio_chunk = audio[:, start_idx:end_idx]

            # Convert the chunk back to FLAC
            chunk_flac_path = f"{audio_flac_path.replace('.flac', '')}_chunk_{i}.flac"
            torchaudio.save(chunk_flac_path, audio_chunk, sample_rate)

            # Transcribe the chunk using Google Speech-to-Text
            chunk_transcription, chunk_word_timestamps = self.transcribe_with_google(chunk_flac_path, i * chunk_duration)

            # Accumulate the transcription and word-level timestamps
            transcription_text += chunk_transcription + " "
            word_timestamps.extend(chunk_word_timestamps)

            # Remove the temporary chunk file
            os.remove(chunk_flac_path)

        # Now, detect sentences and generate embeddings
        sentences = self.generate_sentences_with_embeddings(transcription_text.strip(), word_timestamps)

        # Save the transcription to the JSON file
        with open(output_file, "w") as f:
            json.dump(sentences, f, indent=4)

        return sentences

    def convert_to_flac(self, audio_path):
        flac_path = audio_path.replace(".wav", ".flac")
        subprocess.run(["ffmpeg", "-i", audio_path, flac_path, "-y"], check=True)
        return flac_path

    def transcribe_with_google(self, audio_flac_path, offset_time):
        """
        Use Google Cloud Speech-to-Text API to transcribe audio with word-level timestamps.

        :param audio_flac_path: Path to the FLAC audio file.
        :param offset_time: The start time of this chunk relative to the full audio.
        :return: Transcription text and word-level timestamps for this chunk.
        """
        with open(audio_flac_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True  # Enable automatic punctuation
        )

        response = self.speech_client.recognize(config=config, audio=audio)

        transcription_text = ""
        word_timestamps = []

        for result in response.results:
            alternative = result.alternatives[0]
            transcription_text += alternative.transcript + " "
            for word_info in alternative.words:
                word_timestamps.append({
                    "word": word_info.word,
                    "start_time": word_info.start_time.total_seconds() + offset_time,
                    "end_time": word_info.end_time.total_seconds() + offset_time
                })

        return transcription_text.strip(), word_timestamps

    def generate_sentences_with_embeddings(self, transcription_text, word_timestamps):
        """
        Generate sentences from transcription text and assign real timestamps to them.

        :param transcription_text: Full transcription text.
        :param word_timestamps: Word-level timestamps.
        :return: List of sentences with start/end timestamps and embeddings.
        """
        sentences = []
        doc = self.nlp(transcription_text)

        sentence_start_idx = 0
        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            sentence_words = sentence_text.split()

            # Get the start and end timestamps from the first and last word in the sentence
            start_time = word_timestamps[sentence_start_idx]["start_time"]
            end_time = word_timestamps[sentence_start_idx + len(sentence_words) - 1]["end_time"]

            # Generate sentence embedding
            sentence_embedding = self.embed_text(sentence_text).cpu().tolist()

            # Append the sentence with its start and end timestamps
            sentences.append({
                "text": sentence_text,
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "embedding": sentence_embedding
            })

            sentence_start_idx += len(sentence_words)

        return sentences

    def embed_text(self, text: str):
        # Generate sentence embedding using sentence-transformers
        return self.embedding_model.encode(text, convert_to_tensor=True)