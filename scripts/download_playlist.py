import yt_dlp
import os
import json
import logging
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import get_logger
logger = get_logger(__name__)

def download_playlist_and_map_urls(playlist_url, output_folder="data/reference_songs", cookie_path="cookies.txt"):
    try:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs("data", exist_ok=True) 
        
        print(f"Starting authenticated download to: {output_folder}\n")
        logger.info(f"Starting playlist download: {playlist_url}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'extract_audio': True,
            'audio_format': 'mp3',
            'audio_quality': '192K',
            'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'), 
            'ignoreerrors': True, 
            'quiet': False,
            'cookiefile': 'cookies.txt',
            'sleep_interval_requests': 2,   
            'sleep_interval': 5,              
            'max_sleep_interval': 15,       
            'nooverwrites': True,
            'continue_dl': True,
            'download_archive': os.path.join("data", "downloaded_archive.txt"), 
        }

        url_mapping = {}
        mapping_path = "data/url_mapping.json"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=True)
            
            print("\nExtracting URLs for your search engine...")
            if 'entries' in info:
                for entry in info['entries']:
                    if entry is not None:
                        title = entry.get('title')
                        url = entry.get('original_url') or entry.get('webpage_url')
                        url_mapping[title] = url
            
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(url_mapping, f, indent=4)
            
        print(f"URL database successfully saved to: {mapping_path}")
        print("\nPlaylist download complete! Your AI is ready to index.")
        logger.info("Playlist download and mapping complete.")
    except Exception as e:
        logger.error(f"Error in download_playlist_and_map_urls: {e}")

if __name__ == "__main__":
    PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLGe4ujo789uRkyM-DqDBq08aKky5S2kcn"
    download_playlist_and_map_urls(PLAYLIST_URL)
