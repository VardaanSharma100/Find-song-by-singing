import os
import sys
import pandas as pd
from datasets import load_dataset
from pydub import AudioSegment
import yt_dlp
import ast
import time  


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import get_logger
logger = get_logger(__name__)

try:
    os.makedirs("data/raw/songs", exist_ok=True)
    os.makedirs("data/raw/temp_downloads", exist_ok=True)

    metadata_url = "https://raw.githubusercontent.com/amanteur/CHAD/main/metadata/dataset.csv"
    print("Loading the Metadata Map...")
    df = pd.read_csv(metadata_url)

    print("Syncing with your hummings...")
    hf_dataset = load_dataset("amanteur/CHAD_hummings")['train']

    def download_and_cut(youtube_id, start_time, end_time, output_path):
        try:
            temp_file = f"temp_downloads/{youtube_id}" 
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_file,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
                'quiet': True,
                'no_warnings': True,
                'cookiefile': 'cookies.txt'
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if not os.path.exists(temp_file + ".mp3"):
                    ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
            
            audio = AudioSegment.from_file(temp_file + ".mp3")
            snippet = audio[start_time * 1000 : end_time * 1000]
            snippet.export(output_path, format="wav")
        except Exception as e:
            logger.error(f"Error in download_and_cut for {youtube_id}: {e}")

    print("Starting aligned extraction...")
    logger.info("Starting aligned extraction...")
    for i, item in enumerate(hf_dataset):
        output_file = f"data/raw/songs/song_{i}.wav"
        if os.path.exists(output_file):
            print(f"[{i}] Already exists. Skipping...")
            continue
        parts = item['__key__'].split('/')
        g_id, f_id = parts[0], parts[1]
        
        match = df[(df['group_id'] == g_id) & (df['fragment_id'] == int(f_id)) & (df['audio_type'] == 'original')]
        
        if match.empty:
            match = df[(df['group_id'] == g_id) & (df['fragment_id'] == int(f_id)) & (df['audio_type'] == 'cover')]

        if not match.empty:
            yt_id = match.iloc[0]['youtube_id']
            
            interval_str = match.iloc[0]['interval']
            interval = ast.literal_eval(interval_str)
            start, end = interval[0], interval[1]
            
            try:
                print(f"[{i}] Downloading snippet for {yt_id} (Time: {start}s - {end}s)...")
                download_and_cut(yt_id, start, end, output_file)
                
                time.sleep(1.5) 
                
            except Exception as e:
                print(f"Error downloading {yt_id}: {e}")
                logger.error(f"Error downloading {yt_id}: {e}")
        else:
            print(f"Skipping {i}: No matching studio/cover found in the CSV map.")

    print("\nAll downloads complete. Check 'data/raw/songs/'.")
    logger.info("All downloads complete in get_aligned_songs.")
except Exception as e:
    logger.error(f"Critical error in get_aligned_songs: {e}")
