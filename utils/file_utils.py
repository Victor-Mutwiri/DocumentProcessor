import os
import json
from typing import List, Dict
from datetime import datetime

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}
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
        
def get_files_metadata(user_id: int) -> List[Dict]:
    """Get metadata of all uploaded files for a specific user."""
    if os.path.exists(FILES_METADATA_PATH):
        with open(FILES_METADATA_PATH, 'r') as f:
            all_metadata = json.load(f)
            return [file for file in all_metadata if file['user_id'] == user_id]
    return []

def save_files_metadata(metadata: List[Dict], user_id: int):
    """Save metadata of uploaded files."""
    if os.path.exists(FILES_METADATA_PATH):
        with open(FILES_METADATA_PATH, 'r') as f:
            all_metadata = json.load(f)
    else:
        all_metadata = []
    
    # Deactivate all files for the user
    for file in all_metadata:
        if file['user_id'] == user_id:
            file['active'] = False
    # Update or add new metadata
    for file in metadata:
        for i, existing_file in enumerate(all_metadata):
            if existing_file['filename'] == file['filename'] and existing_file['user_id'] == file['user_id']:
                all_metadata[i] = file
                break
        else:
            all_metadata.append(file)
    
    with open(FILES_METADATA_PATH, 'w') as f:
        json.dump(all_metadata, f)

def add_file_metadata(filename: str, filepath: str, user_id: int, active: bool = False):
    """Add metadata for a new file."""
    metadata = get_files_metadata(user_id)
    metadata.append({
        'filename': filename,
        'filepath': filepath,
        'uploaded_at': datetime.now().isoformat(),
        'active': active,
        'user_id': user_id
    })
    save_files_metadata(metadata)
    
    
def remove_file(filename: str, user_id: int):
    """Remove a file and its metadata."""
    if os.path.exists(FILES_METADATA_PATH):
        with open(FILES_METADATA_PATH, 'r') as f:
            all_metadata = json.load(f)
        
        # Filter out the file to be removed
        all_metadata = [m for m in all_metadata if not (m['filename'] == filename and m['user_id'] == user_id)]
        
        # Save the updated metadata
        with open(FILES_METADATA_PATH, 'w') as f:
            json.dump(all_metadata, f)

""" def remove_file(filename: str, user_id: int):
    Remove a file and its metadata.
    metadata = get_files_metadata(user_id)
    metadata = [m for m in metadata if m['filename'] != filename]
    save_files_metadata(metadata) """

def can_upload_more_files(user_id: int) -> bool:
    """Check if more files can be uploaded."""
    return len(get_files_metadata(user_id)) < MAX_FILES