import os
import json
from typing import List, Dict
from datetime import datetime

ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILES = 3
FILES_METADATA_PATH = 'uploads/files_metadata.json'

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_upload_folder(upload_folder):
    """Create upload folder if it doesn't exist."""
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        
def get_files_metadata() -> List[Dict]:
    """Get metadata of all uploaded files."""
    if os.path.exists(FILES_METADATA_PATH):
        with open(FILES_METADATA_PATH, 'r') as f:
            return json.load(f)
    return []

def save_files_metadata(metadata: List[Dict]):
    """Save metadata of uploaded files."""
    with open(FILES_METADATA_PATH, 'w') as f:
        json.dump(metadata, f)

def add_file_metadata(filename: str, filepath: str, active: bool = False):
    """Add metadata for a new file."""
    metadata = get_files_metadata()
    metadata.append({
        'filename': filename,
        'filepath': filepath,
        'uploaded_at': datetime.now().isoformat(),
        'active': active
    })
    save_files_metadata(metadata)

def remove_file(filename: str):
    """Remove a file and its metadata."""
    metadata = get_files_metadata()
    metadata = [m for m in metadata if m['filename'] != filename]
    save_files_metadata(metadata)

def can_upload_more_files() -> bool:
    """Check if more files can be uploaded."""
    return len(get_files_metadata()) < MAX_FILES