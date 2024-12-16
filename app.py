from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from services.document_processor import DocumentProcessor
from utils.file_utils import (
    allowed_file, create_upload_folder, get_files_metadata, 
    add_file_metadata, remove_file, can_upload_more_files
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize document processor
document_processor = DocumentProcessor()

# Create upload folder if it doesn't exist
create_upload_folder(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    files = get_files_metadata()
    return render_template('index.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) > 3:
        return jsonify({'error': 'Maximum 3 files allowed'}), 400

    if not can_upload_more_files():
        return jsonify({'error': 'Maximum file limit reached. Delete some files first.'}), 400

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
            add_file_metadata(filename, filepath)
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    # Process documents
    try:
        document_processor.process_documents(uploaded_files)
        return jsonify({'message': 'Files uploaded and processed successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    try:
        response = document_processor.chat_with_documents(data['message'])
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        remove_file(filename)
        # Clear document processor state if this was the last active file
        metadata = get_files_metadata()
        if not any(file.get('active', False) for file in metadata):
            document_processor.clear_document_state()
        return jsonify({'message': 'File deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/toggle-active/<filename>', methods=['POST'])
def toggle_active(filename):
    try:
        metadata = get_files_metadata()
        active_files = []
        for file in metadata:
            if file['filename'] == filename:
                file['active'] = not file['active']
                if file['active']:
                    active_files.append(file['filepath'])
            else:
                file['active'] = False
        save_files_metadata(metadata)
        
        # Process only active files
        if active_files:
            document_processor.process_documents(active_files)
        else:
            document_processor.clear_document_state()
        return jsonify({'message': 'File toggled successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 