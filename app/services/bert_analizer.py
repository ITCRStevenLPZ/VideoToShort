from sentence_transformers import SentenceTransformer, util
import os
import logging
import torch

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

from services.gemini_analyzer import GeminiAnalyzer

class BERTAnalyzer:
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        try:
            # Load a pre-trained sentence transformer model
            self.model = SentenceTransformer(model_name)
            self.gemini_analyzer = GeminiAnalyzer()
        except Exception as e:
            print(f"Error initializing BERTAnalyzer: {e}")
            raise

    def embed_text(self, text: str):
        try:
            # Return the sentence embedding from sentence-transformers
            return self.model.encode(text, convert_to_tensor=True)
        except Exception as e:
            print(f"Error embedding text: {e}")
            raise

    
    def calculate_similarity(self, transcript: list[dict], topics: list[str], threshold: float = 0.6, video_length: int = None):
        try:
            # Determine the device (GPU if available, otherwise CPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
            # Concatenate transcript texts into a single string
            transcript_text = ' '.join([entry['text'] for entry in transcript])
    
            # Analyze transcript to extract the exact matching text related to topics
            gemini_output = self.gemini_analyzer.analyze_transcript(transcript_text, topics)
            gemini_output = self.format_remover(gemini_output)
    
            # Split the output text from GeminiAI into sentences
            gemini_sentences = self.split_into_sentences(gemini_output)
    
            print(f"Extracted sentences from GeminiAI: {gemini_sentences}")
    
            # Embed the Gemini sentences and move to the same device
            gemini_embeddings = [self.embed_text(sentence).to(device) for sentence in gemini_sentences]
    
            matched_sentences = []
            for sentence in transcript:
                sentence_text = sentence['text']
                sentence_embedding = self.embed_text(sentence_text).to(device)
    
                # Calculate similarity with each Gemini sentence
                max_similarity = 0
                try:
                    for gemini_embedding in gemini_embeddings:
                        similarity = util.pytorch_cos_sim(sentence_embedding, gemini_embedding).item()
                        if similarity > max_similarity:
                            max_similarity = similarity
                except Exception as e:
                    print(f"Error calculating similarity embeddings: {e}")
    
                # If the similarity exceeds the threshold, consider it a match
                if max_similarity >= threshold:
                    matched_sentences.append({
                        'text': sentence_text,
                        'start_timestamp': sentence['start_timestamp'],
                        'end_timestamp': sentence['end_timestamp'],
                        'similarity': max_similarity
                    })
    
            if not matched_sentences:
                print("No matching sentences found in the transcription.")
                return None
    
            # Sort matched sentences by start timestamp
            matched_sentences.sort(key=lambda x: x['start_timestamp'])
    
            # Merge timestamps of all matched sentences
            start_timestamp = matched_sentences[0]['start_timestamp']
            end_timestamp = matched_sentences[-1]['end_timestamp']
    
            if video_length is not None and end_timestamp > video_length:
                end_timestamp = video_length
    
            # Create a clip representing the combined text and timestamps
            clip = {
                'text': ' '.join([sentence['text'] for sentence in matched_sentences]),
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp
            }
    
            # Save the result
            self.save_similarity_results([clip], './CLIP_results.txt')
    
            return clip
    
        except Exception as e:
            print(f"Error in calculate_similarity: {e}")
            raise
    def format_remover(self, text: str):
        try:
            """ Remove unnecessary formatting from the text. """
            return text.replace('\n', ' ').replace('\r', ' ').strip()
        except Exception as e:
            print(f"Error in format_remover: {e}")
            raise

    def split_into_sentences(self, text: str):
        try:
            """ Split the text into sentences based on simple punctuation rules. """
            return [sentence.strip() for sentence in text.split('.') if sentence.strip()]
        except Exception as e:
            print(f"Error in split_into_sentences: {e}")
            raise

    def is_exact_match(self, transcript_sentence: str, gemini_sentence: str):
        try:
            """
            Check if the given transcript sentence matches the GeminiAI sentence exactly.
            This can be enhanced with more flexible matching (e.g., semantic matching) if needed.
            """
            return transcript_sentence.strip().lower() == gemini_sentence.strip().lower()
        except Exception as e:
            print(f"Error in is_exact_match: {e}")
            raise

    def save_similarity_results(self, similarity_results, file_path):
        try:
            """
            Save the similarity results (clip information) to a text file in a legible format.
            """
            if os.path.exists(file_path):
                os.remove(file_path)

            with open(file_path, "w") as file:
                for result in similarity_results:
                    file.write(f"Text: {result['text']}\n")
                    file.write(f"Start Timestamp: {result['start_timestamp']}\n")
                    file.write(f"End Timestamp: {result['end_timestamp']}\n")
                    file.write("\n" + "-"*50 + "\n\n")
        except Exception as e:
            print(f"Error in save_similarity_results: {e}")
            raise