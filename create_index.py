import torch
import os
import sys
import numpy as np
import librosa


sys.path.append(os.path.join(os.path.dirname(__file__)))
from src.utils.logger import get_logger
logger = get_logger(__name__)

from src.data.preprocess import process_single_audio 
from src.models.siamese import SiameseNetwork
from src.utils.config import EMBEDDING_DIM

def build_search_index(model_path, songs_dir, output_index_path, device='cpu'):
    try:
        print(f"Loading frozen AI from {model_path}...")
        
        model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() 
        song_database = {}
        valid_extensions = ('.wav', '.mp3','.webm', '.m4a')
        
        CHUNK_LENGTH_SEC = 16 
        
        print(f"Scanning directory: {songs_dir}")
        
        with torch.no_grad():
            for filename in os.listdir(songs_dir):
                if filename.lower().endswith(valid_extensions):
                    song_path = os.path.join(songs_dir, filename)
                    song_name = os.path.splitext(filename)[0]
                    
                    print(f"Chopping and Indexing: {song_name}...")
                    
                    duration = librosa.get_duration(path=song_path)
                    chunk_embeddings = []
                    
                    for start_time in np.arange(0, duration - CHUNK_LENGTH_SEC, CHUNK_LENGTH_SEC):
                        
                        mel_tensor, pitch_tensor = process_single_audio(
                            song_path, 
                            offset=start_time, 
                            duration=CHUNK_LENGTH_SEC
                        )
                        
                        mel_tensor = mel_tensor.unsqueeze(0).to(device)
                        pitch_tensor = pitch_tensor.unsqueeze(0).to(device)

                        embedding = model(mel_tensor, pitch_tensor)
                        chunk_embeddings.append(embedding.squeeze(0).cpu().numpy())
                    
                    if len(chunk_embeddings) > 0:
                        song_database[song_name] = np.array(chunk_embeddings)

        print(f"\nSuccessfully indexed {len(song_database)} songs (thousands of chunks!).")
        
        torch.save(song_database, output_index_path)
        
        file_size_kb = os.path.getsize(output_index_path) / 1024
        print(f"Search Index physically saved to {output_index_path} (Size: ~{file_size_kb:.2f} KB)")
        logger.info(f"Built search index successfully at {output_index_path}")
    except Exception as e:
        logger.error(f"Error building search index: {e}")

if __name__ == "__main__":
    BEST_MODEL = "models/best_hum_model.pth"
    REFERENCE_SONGS = "data/reference_songs" 
    OUTPUT_INDEX = "data/song_index.pt"
    
    os.makedirs("data", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_search_index(BEST_MODEL, REFERENCE_SONGS, OUTPUT_INDEX, device)
