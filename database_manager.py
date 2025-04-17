import chromadb
import numpy as np
import torch
import librosa
import openl3
from transformers import AutoTokenizer, AutoModel
import os

class DatabaseManager:
    def __init__(self):
        # Use ChromaDB cloud with API v2
        print("\nInitializing ChromaDB client...")
        self.client = chromadb.HttpClient(
            ssl=True,
            host='api.trychroma.com',
            tenant='862c8d20-8fd1-4590-869e-e42a23a87b69',
            database='TS_songs',
            headers={
                'x-chroma-token': 'ck-79J5zWUA4sst94wTDJRUoTRd8Eu6ifPVMDWJDHZCfneF'
            }
        )
        print("ChromaDB client initialized")
        
        print("\nCreating/getting collections...")
        self.collection = self.client.get_or_create_collection(name="taylor_swift_songs")
        print("Collections created/retrieved")
        
        # Text processing models for mood matching
        print("\nLoading text processing models...")
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        print("Text models loaded")
        
        # Verify collections
        try:
            songs_count = self.collection.count()
            print(f"\nCurrent database state:")
            print(f"Songs collection: {songs_count} songs")
        except Exception as e:
            print(f"Error checking collection counts: {str(e)}")
        
    def preprocess_audio(self, audio_path, target_sr=16000, duration=60):
        """Preprocess audio file for embedding generation.
        
        Process the first 60 seconds of the audio file.
        """
        print(f"Loading audio file: {audio_path}")
        # Load only the first 60 seconds of the audio
        y, sr = librosa.load(audio_path, sr=target_sr, duration=duration)
        
        # Apply a high-pass filter to reduce background noise
        y = librosa.effects.preemphasis(y)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Trim silence from the beginning and end
        y, _ = librosa.effects.trim(y, top_db=30)
        
        print(f"Processed audio length: {len(y)/target_sr:.2f} seconds")    
        return y, sr

    
    def get_audio_embedding(self, audio_path):
        """Generate embedding for an audio file using OpenL3.
        
        Process the first 30 seconds of the song.
        """
        print(f"\n=== Processing song: {audio_path} ===")
        
        # Process first 30 seconds
        audio, sr = self.preprocess_audio(audio_path, duration=30)
        
        # Resample if needed
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000
        
        print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
        
        try:
            # Generate embedding with fixed parameter order
            embedding, _ = openl3.get_audio_embedding(
                audio, 
                sr, 
                model=None,
                input_repr="mel256",
                content_type="music",
                embedding_size=512
            )
            
            print(f"Raw embedding shape: {embedding.shape}")
            
            # Ensure embedding is a 1D array
            embedding = embedding.squeeze()
            if embedding.ndim > 1:
                embedding = embedding.mean(axis=0)
            
            # Apply min-max scaling to preserve relative differences
            embedding_min = np.min(embedding)
            embedding_max = np.max(embedding)
            if embedding_max > embedding_min:
                embedding = (embedding - embedding_min) / (embedding_max - embedding_min)
            
            print(f"Final embedding shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise
    
    def get_text_embedding(self, text):
        """Generate embedding for a text description."""
        # Tokenize and get model output
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            # Use the [CLS] token embedding as the sentence embedding
            text_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return text_embedding
    
    def add_song(self, song_id, audio_path, metadata):
        """Add a song to the database.
        
        Process the first 60 seconds of the song.
        """
        print(f"\nAdding song: {song_id}")
        
        try:
            # Process first 60 seconds
            audio, sr = self.preprocess_audio(audio_path, duration=60)
            
            # Resample if needed
            if sr != 48000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                sr = 48000
            
            print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
            
            # Generate embedding with fixed parameter order
            embedding, _ = openl3.get_audio_embedding(
                audio, 
                sr, 
                model=None,
                input_repr="mel256",
                content_type="music",
                embedding_size=512
            )
            
            print(f"Raw embedding shape: {embedding.shape}")
            
            # Ensure embedding is a 1D array
            embedding = embedding.squeeze()
            if embedding.ndim > 1:
                embedding = embedding.mean(axis=0)
            
            # Apply min-max scaling to preserve relative differences
            embedding_min = np.min(embedding)
            embedding_max = np.max(embedding)
            if embedding_max > embedding_min:
                embedding = (embedding - embedding_min) / (embedding_max - embedding_min)
            
            print(f"Final embedding shape: {embedding.shape}")
            
            # Add metadata about the segment
            metadata.update({
                "segment": "first_60_seconds"
            })
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[metadata],
                ids=[song_id]
            )
            
            print(f"Added first 60 seconds of {song_id}")
            
            # Verify the addition
            try:
                count = self.collection.count()
                print(f"Total records in database: {count}")
            except Exception as e:
                print(f"Error checking database count: {str(e)}")
                
        except Exception as e:
            print(f"Error in add_song: {str(e)}")
            raise
    
    def search_similar_songs(self, query_embedding, n_results=3):
        """Search for similar songs in the database.
        
        Improved to return more results and use a better similarity metric.
        """
        print("\n=== Searching for similar songs ===")
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Number of results requested: {n_results}")
        
        # Apply min-max scaling to query embedding
        query_min = np.min(query_embedding)
        query_max = np.max(query_embedding)
        if query_max > query_min:
            query_embedding = (query_embedding - query_min) / (query_max - query_min)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        print(f"Found {len(results['ids'][0])} matches")
        
        # Store both distance and similarity in the results
        processed_results = []
        for i, (song_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            # Convert L2 distance to cosine similarity
            # For normalized vectors, cosine similarity = 1 - (L2_distance^2)/2
            similarity = 1 - (distance * distance) / 2
            
            # Apply a scaling factor to make scores more meaningful
            # This will make the scores range from about 0.3 to 1.0
            similarity = 0.3 + 0.7 * similarity
            
            result = {
                'song_id': song_id,
                'distance': distance,
                'similarity': similarity
            }
            processed_results.append(result)
            
            print(f"  {i+1}. Song ID: {song_id}, Distance: {distance:.4f}, Similarity: {similarity:.2%}")
        
        print("===================================\n")
        return processed_results
    
    def search_by_mood(self, mood_description, n_results=5):
        """Search for songs that match a mood description."""
        print(f"\n=== Searching for songs matching mood: '{mood_description}' ===")
        
        # Generate embedding for the mood description
        mood_embedding = self.get_text_embedding(mood_description)
        
        # Normalize the embedding
        mood_embedding = mood_embedding / (np.linalg.norm(mood_embedding) + 1e-8)
        
        # Search in the collection
        results = self.collection.query(
            query_embeddings=[mood_embedding.tolist()],
            n_results=n_results
        )
        
        print(f"Found {len(results['ids'][0])} mood matches")
        
        # Improved similarity scoring system
        for i, (song_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            # Convert distance to a more intuitive similarity score
            similarity = 1 / (1 + distance/2)
            similarity = 0.1 + 0.9 * similarity
            
            print(f"  {i+1}. Song ID: {song_id}, Distance: {distance:.4f}, Similarity: {similarity:.2%}")
        
        print("===================================\n")
        return results
        
    def calculate_similarity_score(self, query_embedding, target_embedding):
        """Calculate similarity score between two embeddings.
        
        Improved to use cosine similarity which is better for high-dimensional vectors.
        """
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        target_norm = target_embedding / (np.linalg.norm(target_embedding) + 1e-8)
        
        # Compute cosine similarity
        return np.dot(query_norm, target_norm) 