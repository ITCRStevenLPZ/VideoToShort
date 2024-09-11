import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sentence_transformers import SentenceTransformer
import spacy
import subprocess
import os
import json
from tqdm import tqdm

class TranscriptionService:
    def __init__(self, model_name: str = "openai/whisper-small", embedding_model_name: str = "paraphrase-MiniLM-L6-v2"):
        # Load the pre-trained Whisper model and processor
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.target_sample_rate = 16000  # Whisper model expects 16kHz audio
        
        # Load SpaCy for sentence detection
        self.nlp = spacy.load("en_core_web_sm")

        # Load a pre-trained SentenceTransformer model for embeddings
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def transcribe_audio(self, audio_path, language="en", fileName="output", chunk_duration=30.0):
        """
        Transcribe the given audio input into text with sentence-level timestamps and embeddings.
        
        :param audio_path: Path to the audio file.
        :param language: Language of the transcription.
        :param fileName: Base name for the output JSON file.
        :param chunk_duration: Duration of each audio chunk in seconds.
        :return: List of dictionaries, each containing a sentence, its start/end timestamp, and sentence embedding.
        """
        output_file = f"./{fileName}.json"
        
        # Check if the output file already exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                sentences = json.load(f)
                return sentences
        
        audio, sample_rate = self.audio_wav_to_torch(audio_path)

        # Resample the audio if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            audio = resampler(audio)
            sample_rate = self.target_sample_rate

        # Ensure the audio tensor is 2D (1, N)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        chunk_size = int(chunk_duration * sample_rate)  # Chunk size in samples
        num_chunks = (audio.size(1) + chunk_size - 1) // chunk_size  # Ceiling division

        complete_transcription = []
        sentence_mappings = []

        for i in tqdm(range(num_chunks), desc="Transcribing", unit="chunk"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, audio.size(1))
            audio_chunk = audio[:, start_idx:end_idx]
            chunk_start_time = i * chunk_duration
        
            # Transcribe the audio chunk
            transcription_text = self.transcribe_chunk(audio_chunk, sample_rate, language)
        
            # Store the entire transcription text (used to detect sentences later)
            complete_transcription.append(transcription_text)

            # Track the word count and timestamp for accurate sentence boundary calculation
            words_in_chunk = transcription_text.split()
            num_words = len(words_in_chunk)
            chunk_duration_in_s = (end_idx - start_idx) / sample_rate  # Duration of chunk in seconds
            sentence_mappings.append({
                'text': transcription_text,
                'start_time': chunk_start_time,
                'end_time': chunk_start_time + chunk_duration_in_s,
                'word_count': num_words
            })

        # Combine all transcriptions into a single text
        full_transcription_text = " ".join(complete_transcription)
        
        # Now, detect sentences and calculate accurate timestamps
        sentences = self.calculate_sentence_timestamps(full_transcription_text, sentence_mappings)

        # Save the transcription to the JSON file
        with open(output_file, "w") as f:
            json.dump(sentences, f, indent=4)

        return sentences

    def transcribe_chunk(self, audio_chunk, sample_rate, language="en"):
        # Preprocess the audio to obtain input features
        input_features = self.processor(audio_chunk.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_features

        # Generate attention mask
        attention_mask = torch.ones(input_features.shape, dtype=torch.long)

        # Perform transcription and return logits
        predicted_ids = self.model.generate(
            input_features, 
            attention_mask=attention_mask, 
            language=language, 
            max_length=1000,  # Increase max_length to allow for longer transcriptions
            num_beams=5  # Use beam search for better quality
        )

        # Decode the logits to get the transcription text
        transcription_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return transcription_text

    def audio_wav_to_torch(self, audio_path):
        # Load an audio file from disk and convert it to a torch tensor
        try:
            subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            raise EnvironmentError("ffmpeg is not installed or not found in the system path. Please install ffmpeg.")

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {audio_path}: {e}")

        # Ensure the audio is mono-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform, sample_rate

    def embed_text(self, text: str):
        # Generate sentence embedding using sentence-transformers
        return self.embedding_model.encode(text, convert_to_tensor=True)

    def calculate_sentence_timestamps(self, transcription_text, sentence_mappings):
        # Group transcribed text into sentences and assign timestamps dynamically
        sentences = []
        doc = self.nlp(transcription_text)

        word_offset = 0  # Offset to track word position in chunks

        for sentence in doc.sents:
            sentence_text = sentence.text.strip()
            sentence_words = sentence_text.split()

            # Calculate start and end timestamps for this sentence based on word position in chunks
            start_time, end_time = None, None
            current_word_offset = 0  # Reset for each sentence

            for mapping in sentence_mappings:
                if word_offset + len(sentence_words) <= mapping['word_count'] + current_word_offset:
                    start_time = mapping['start_time'] + (current_word_offset / mapping['word_count']) * (mapping['end_time'] - mapping['start_time'])
                    end_time = mapping['start_time'] + ((current_word_offset + len(sentence_words)) / mapping['word_count']) * (mapping['end_time'] - mapping['start_time'])
                    break
                else:
                    current_word_offset += mapping['word_count']

            # Generate sentence embedding
            sentence_embedding = self.embed_text(sentence_text).cpu().tolist()

            # Append the sentence with its start and end timestamps
            sentences.append({
                "text": sentence_text,
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "embedding": sentence_embedding
            })

            # Update the word offset
            word_offset += len(sentence_words)

        return sentences