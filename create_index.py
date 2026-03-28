import torch
import os
import sys
import numpy as np
import librosa
from tqdm import tqdm
import warnings
import concurrent.futures

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append(os.path.join(os.path.dirname(__file__)))
from src.utils.logger import get_logger
logger = get_logger(__name__)

from src.data.preprocess import process_single_audio 
from src.models.siamese import SiameseNetwork
from src.utils.config import EMBEDDING_DIM

def extract_song_features(song_path, chunk_length_sec):
    try:
        duration = librosa.get_duration(path=song_path)
    except Exception as e:
        return None, f"Failed to get duration: {e}"
        
    starts = np.arange(0, duration - chunk_length_sec, chunk_length_sec)
    features = []
    for start_time in starts:
        try:
            mel_tensor, pitch_tensor = process_single_audio(
                song_path, 
                offset=start_time, 
                duration=chunk_length_sec
            )
            features.append((mel_tensor, pitch_tensor))
        except Exception as e:
            continue
    return features, None

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
        num_workers = max(1, os.cpu_count() - 1)
        print(f"Extracting features using {num_workers} parallel workers...")
        
        with torch.no_grad():
            valid_files = [f for f in os.listdir(songs_dir) if f.lower().endswith(valid_extensions)]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_song = {
                    executor.submit(extract_song_features, os.path.join(songs_dir, filename), CHUNK_LENGTH_SEC): os.path.splitext(filename)[0]
                    for filename in valid_files
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_song), total=len(valid_files), desc="Indexing Songs"):
                    song_name = future_to_song[future]
                    try:
                        features, error = future.result()
                        if error:
                            logger.warning(f"Error for {song_name}: {error}")
                            continue
                            
                        if not features:
                            continue
                            
                        mel_batch = torch.stack([f[0] for f in features]).to(device)
                        pitch_batch = torch.stack([f[1] for f in features]).to(device)
                        
                        
                        embeddings = model(mel_batch, pitch_batch)
                        song_database[song_name] = embeddings.cpu().numpy()
                    except Exception as e:
                        logger.warning(f"Failed to process {song_name}: {e}")

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
