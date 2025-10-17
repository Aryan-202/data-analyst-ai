import os
import pandas as pd
from typing import Optional, Dict, Any
import magic

def get_file_extension(file_path: str) -> str:
    """Get file extension from file path"""
    return os.path.splitext(file_path)[1].lower().replace('.', '')

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)

def detect_file_type(file_path: str) -> str:
    """Detect file type using python-magic"""
    try:
        file_type = magic.from_file(file_path, mime=True)
        return file_type
    except:
        return "unknown"

def safe_read_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Safely read CSV file with error handling"""
    try:
        return pd.read_csv(file_path, **kwargs)
    except UnicodeDecodeError:
        # Try different encodings
        for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
            try:
                return pd.read_csv(file_path, encoding=encoding, **kwargs)
            except UnicodeDecodeError:
                continue
        raise