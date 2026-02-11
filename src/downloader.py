import os
import gdown
import requests
import concurrent.futures
from .config import DOWNLOAD_DIR, MAX_WORKERS

def download_file(url, output_filename):
    """
    Downloads a file from a URL (Google Drive or direct) to the local download directory.
    Returns the local path if successful, else None.
    """
    if not url:
        return None
    
    output_path = os.path.join(DOWNLOAD_DIR, output_filename)
    
    # Check if already exists (Simple caching)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
        
    try:
        # User gdown for Drive links
        if "drive.google.com" in url:
            # gdown handles the extraction of ID and download
            # We need to pass the raw URL to gdown, it's smart enough
            # But the 'url' passed here might be the 'cleaned' one with uc?id=...
            # gdown wants the 'link' usually.
            # Let's try downloading.
            gdown.download(url, output_path, quiet=True, fuzzy=True)
        else:
            # standard download
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        if os.path.exists(output_path):
            return output_path
        return None
        
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def batch_download(team_data):
    """
    Downloads files for multiple teams in parallel.
    team_data: List of dicts with 'Team Name' and 'Download Link'
    Returns a dict mapping Team Name to Local File Path.
    """
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_team = {}
        for team in team_data:
            name = team.get("Team Name")
            link = team.get("Download Link")
            # Create a safe filename
            safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()
            # We don't know extension yet, but gdown might detect it.
            # We'll default to a temporary name and rename later or just detect type.
            # Actually gdown will try to keep the name. 
            # Let's explicitly give a name WITHOUT extension and let detection happen later, 
            # OR just assume it downloads what it is.
            # Better strategy: Let gdown save it, then rename? 
            # No, we need trackability.
            # We'll save as keys.
            # Let's try to guess content disposition or just download as 'TeamName_App' and check magic numbers.
            # For hackathon simplicity, we'll try to append a guessed extension or check after.
            # Let's just use the Team Name as prefix.
            fname = f"{safe_name}_presentation" 
            
            future = executor.submit(download_file, link, fname)
            future_to_team[future] = name
            
        for future in concurrent.futures.as_completed(future_to_team):
            team_name = future_to_team[future]
            try:
                path = future.result()
                if path:
                    # Detect extension if missing
                    # If gdown saved it, it might have done it purely as the name we gave '..._presentation'
                    # We should check the file header or try to open with pptx.
                    
                    # Basic rename based on magic bytes or just return the path
                    results[team_name] = path
            except Exception as e:
                print(f"Error downloading for {team_name}: {e}")
                
    return results
