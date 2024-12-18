from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from services.document_processor import DocumentProcessor
from utils.file_utils import (
    allowed_file, create_upload_folder, get_files_metadata, 
    add_file_metadata, remove_file, can_upload_more_files,
    save_files_metadata
)
from datetime import datetime

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
    
    # If there are files but none are active, set the first one as active
    if files and not any(file.get('active') for file in files):
        files[0]['active'] = True
        save_files_metadata(files)
        
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
    metadata = get_files_metadata()
    
    # Deactivate all existing files
    for file in metadata:
        file['active'] = False
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
            metadata.append({
                'filename': filename,
                'filepath': filepath,
                'uploaded_at': datetime.now().isoformat(),
                'active': True
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    save_files_metadata(metadata)

    # Process documents
    try:
        processing_success = document_processor.process_documents(uploaded_files)
        if not processing_success:
            return jsonify({'error': 'Failed to process documents'}), 500
        return jsonify({'message': 'Files uploaded and processed successfully'})
    except Exception as e:
        print(f"Error processing documents: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # Debug logging
        print("Chat request received")
        
        # Get active files based on selection
        metadata = get_files_metadata()
        if data.get('document'):  # If specific document is selected
            active_files = [file['filepath'] for file in metadata 
                          if file['filename'] == data['document']]
            print(f"Selected document: {data['document']}")
        else:  # Use all active documents
            active_files = [file['filepath'] for file in metadata 
                          if file.get('active')]
            
        print(f"Active files: {active_files}")
        
        if not active_files:
            return jsonify({'error': 'No active documents selected'}), 400

        # Process documents if needed
        if not document_processor.document_chunks:
            print("Processing selected documents")
            success = document_processor.process_documents(active_files)
            if not success:
                return jsonify({'error': 'Failed to process documents'}), 500

        response = document_processor.chat_with_documents(data['message'])
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
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

@app.route('/document-summary/<filename>', methods=['GET'])
def get_document_summary(filename):
    try:
        print(f"Summary requested for: {filename}")  # Debug log
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")  # Debug log
            return jsonify({'error': 'File not found'}), 404

        # Get summary from document processor
        summary = document_processor.get_document_summary(filepath)
        
        if not summary:
            print("No summary generated")  # Debug log
            return jsonify({'error': 'Failed to generate summary'}), 500
            
        if 'summary' not in summary:
            print("Summary missing from response")  # Debug log
            return jsonify({'error': 'Invalid summary format'}), 500
        
        # Extract risks from summary
        risks = [line for line in summary['summary'].split('\n') 
                if 'risk' in line.lower() or 'flag' in line.lower()]
        
        response_data = {
            'summary': summary['summary'],
            'risks': '<br>'.join(risks) if risks else 'No significant risks identified.'
        }
        
        print("Successfully generated summary response")  # Debug log
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()  # Print full stack trace
        return jsonify({'error': str(e)}), 500

@app.route('/set-active-document', methods=['POST'])
def set_active_document():
    try:
        data = request.json
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400

        metadata = get_files_metadata()
        
        # Set all documents as inactive first
        for file in metadata:
            file['active'] = False
            
        # Set the selected document as active
        for file in metadata:
            if file['filename'] == data['filename']:
                file['active'] = True
                break
        
        save_files_metadata(metadata)
        
        # Process the active document
        active_filepath = next((file['filepath'] for file in metadata if file['active']), None)
        if active_filepath:
            document_processor.clear_document_state()
            document_processor.process_documents([active_filepath])
            
        return jsonify({'message': 'Active document updated successfully'})
    except Exception as e:
        print(f"Error setting active document: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 