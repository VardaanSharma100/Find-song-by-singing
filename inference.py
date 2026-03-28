import torch
import torch.nn.functional as F
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import json
import re
import os
from src.utils.logger import get_logger
from src.data.preprocess import process_single_audio
from src.models.siamese import SiameseNetwork
from src.utils.config import EMBEDDING_DIM
logger = get_logger(__name__)
def record_hum(duration=16, sample_rate=22050, filename="temp_hum.wav"):
  
    try:
        print(f"\nMICROPHONE ON: Start humming for {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait() 
        print(" Recording complete! Processing...")
        
        write(filename, sample_rate, audio_data)
        logger.info("Successfully recorded and saved temporary hum.")
        return filename
    except Exception as e:
        logger.error(f"Error in record_hum: {e}")
        return None


def search_database(hum_vector, database_path, url_mapping_path):
    try:
        song_db = torch.load(database_path, weights_only=False)
        
        with open(url_mapping_path, 'r', encoding='utf-8') as f:
            url_map = json.load(f)

        best_match_name = None
        best_similarity_score = -1.0

        for song_name, chunk_matrix in song_db.items():
            chunk_tensors = torch.tensor(chunk_matrix) 
            
            
            similarities = F.cosine_similarity(hum_vector, chunk_tensors)
            
            max_sim_for_song = torch.max(similarities).item()

            if max_sim_for_song > best_similarity_score:
                best_similarity_score = max_sim_for_song
                best_match_name = song_name

        youtube_link = url_map.get(best_match_name)
        if not youtube_link or youtube_link == "No Link Found":
            import urllib.parse
            search_query = urllib.parse.quote(best_match_name)
            youtube_link = f"https://www.youtube.com/results?search_query={search_query}"
        
        logger.info(f"Successfully matched: {best_match_name} with score {best_similarity_score}")
        return best_match_name, best_similarity_score, youtube_link
    except Exception as e:
        logger.error(f"Error in search_database: {e}")
        return "Error", -1.0, "Error"

def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading AI...")
        model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
        model.load_state_dict(torch.load("models/best_hum_model.pth", map_location=device, weights_only=True))
        model.to(device)
        model.eval() 
        
        temp_wav = record_hum(duration=16)
        if not temp_wav:
           logger.error("Failed to record hum. Exiting.")
           return
        
        mel, pitch = process_single_audio(temp_wav, duration=16)
        mel = mel.unsqueeze(0).to(device)
        pitch = pitch.unsqueeze(0).to(device)
        
        with torch.no_grad():
            hum_vector = model(mel, pitch)
            
        print("Searching the database...")
        match_name, score, link = search_database(
            hum_vector.cpu(), 
            database_path="data/song_index.pt", 
            url_mapping_path="data/url_mapping.json"
        )
        
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        logger.info("Inference completed successfully.")
    except Exception as e:
        logger.error(f"Error in main inference loop: {e}")
        
    print("\n=========================================")
    print(f"MATCH FOUND: {match_name}")
    print(f"CONFIDENCE SCORE: {score * 100:.1f}%")
    print(f" LISTEN HERE: {link}")
    print("=========================================\n")

if __name__ == "__main__":
    main()
