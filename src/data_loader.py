import pandas as pd
import streamlit as st
import re

REQUIRED_COLUMNS = ["Team Name", "PPT Link"]

def clean_drive_link(link):
    """
    Extracts the file ID from a Google Drive link to convert it to a direct download link (if public)
    or just cleans it for gdown.
    """
    if not isinstance(link, str):
        return None
    
    # Extract file ID using regex
    # Common patterns:
    # https://drive.google.com/file/d/FILE_ID/view
    # https://docs.google.com/presentation/d/FILE_ID/edit
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?id={file_id}"
    return link

def load_data(file_path_or_url):
    """
    Loads data from a local CSV/Excel or Google Sheet public CSV URL.
    Returns a pandas DataFrame with normalized columns.
    """
    try:
        if file_path_or_url.endswith('.csv'):
            df = pd.read_csv(file_path_or_url)
        elif file_path_or_url.endswith('.xlsx'):
            df = pd.read_excel(file_path_or_url)
        else:
            return None, "Unsupported file format. Please use CSV or Excel."
        
        # Validation
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return None, f"Missing required columns: {missing_cols}"
            
        # Clean Links
        df['Download Link'] = df['PPT Link'].apply(clean_drive_link)
        
        return df, None
    except Exception as e:
        return None, str(e)
