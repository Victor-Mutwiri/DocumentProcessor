from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv
import logging
from services.document_processor import EnhancedDocumentProcessor
from utils.file_utils import (
    allowed_file, create_upload_folder, get_files_metadata, get_contract_files_metadata,
    add_file_metadata, add_contract_file_metadata, remove_file, can_upload_more_files,
    save_files_metadata, remove_contract_file, can_upload_more_contract_files, save_contract_files_metadata
)
from utils.auth_utilis import authenticate_user, register_user
from datetime import datetime
from models import db, User, Document
from multiprocessing import freeze_support



freeze_support()

# Global variable for document processor
document_processor = None

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    logger.debug("Starting application creation")
    load_dotenv()
    
    app = Flask(__name__)
    logger.debug("Flask app instance created")
    CORS(app, supports_credentials=True)
    
    # Configuration
    app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['CONTRACT_FOLDER'] = 'contracts'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    logger.debug("Basic configuration completed")

    db.init_app(app)

    # Initialize document processor
    global document_processor
    with app.app_context():
        try:
            document_processor = EnhancedDocumentProcessor()
        except Exception as e:
            print(f"Warning: Failed to initialize document processor: {e}")
            document_processor = None

    # Create upload folder if it doesn't exist
    create_upload_folder(app.config['UPLOAD_FOLDER'])
    create_upload_folder(app.config['CONTRACT_FOLDER'])

    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        if origin in ["https://sheria.vimtec.co.ke", "http://localhost:5173", "https://doc-processor-theta.vercel.app"]:
            response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Session-Id'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    # User management
    class UserAdminView(ModelView):
        column_list = ['id', 'name', 'created_at', 'last_login', 'is_active', 'total_documents']
        column_searchable_list = ['name']
        column_filters = ['is_active', 'created_at']

    # Document management
    class DocumentAdminView(ModelView):
        column_list = ['filename', 'user', 'uploaded_at', 'file_size', 'is_active', 'processing_status']
        column_searchable_list = ['filename', 'user.name']
        column_filters = ['is_active', 'uploaded_at', 'processing_status']

    # Register admin views
    admin = Admin(app, name='Document Processor Admin', template_mode='bootstrap4')
    admin.add_view(UserAdminView(User, db.session))
    admin.add_view(DocumentAdminView(Document, db.session))
    
    register_routes(app)
    
    return app

def register_routes(app):

    @app.route('/')
    def index():
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        files = get_files_metadata(session['user_id'])
        return jsonify({'files': files})

    @app.route('/login', methods=['POST'])
    def login():
        data = request.json
        name = data.get('name')
        password = data.get('password')
        user = authenticate_user(name, password)
        if user:
            session['user_id'] = user['id']
            session['session_id'] = str(user['id'])  # Set session ID to user ID
            return jsonify({'message': 'Login successful', 'session_id': session['session_id']})
        return jsonify({'error': 'Invalid credentials'}), 401

    @app.route('/register', methods=['POST'])
    def register():
        data = request.json
        name = data.get('name')
        password = data.get('password')
        user = register_user(name, password)
        if user:
            session['user_id'] = user['id']
            session['session_id'] = str(user['id'])  # Set session ID to user ID
            return jsonify({'message': 'Registration successful', 'session_id': session['session_id']})
        return jsonify({'error': 'User already exists'}), 400

    @app.route('/logout')
    def logout():
        session.pop('user_id', None)
        session.pop('session_id', None)
        return jsonify({'message': 'Logout successful'})

    @app.route('/api/upload', methods=['POST'])
    def upload_files():
        session_id = request.headers.get('Session-Id')
        if session_id:
            session['user_id'] = session_id
            session['session_id'] = session_id
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if len(files) > 3:
            return jsonify({'error': 'Maximum 3 files allowed'}), 400

        if not can_upload_more_files(session['user_id']):
            return jsonify({'error': 'Maximum file limit reached. Delete some files first.'}), 400

        uploaded_files = []
        metadata = get_files_metadata(session['user_id'])
        
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
                    'active': True,
                    'user_id': session['user_id']
                })
            else:
                return jsonify({'error': 'Invalid file type'}), 400

        save_files_metadata(metadata, session['user_id'])

        # Process documents
        try:
            processing_success = document_processor.process_documents(uploaded_files)
            if not processing_success:
                return jsonify({'error': 'Failed to process documents'}), 500
            return jsonify({'message': 'Files uploaded and processed successfully'})
        except Exception as e:
            print(f"Error processing documents: {str(e)}")  # Debug log
            return jsonify({'error': str(e)}), 500


    @app.route('/api/files', methods=['GET'])
    def get_uploaded_files():
        session_id = request.headers.get('Session-Id')
        if not session_id:
            return jsonify({'error': 'Unauthorized'}), 401

        # Directly set user_id to session_id for testing
        session['user_id'] = session_id

        try:
            metadata = get_files_metadata(session['user_id'])
            return jsonify({'files': metadata})
        except Exception as e:
            print(f"Error fetching files metadata: {str(e)}")
            return jsonify({'error': str(e)}), 500


    def get_user_id_from_session(session_id):
        # Implement this function to retrieve the user_id from the session_id
        return session.get('user_id') if session.get('session_id') == session_id else None


    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        session_id = request.headers.get('Session-Id')
        if not session_id:
            return jsonify({'error': 'Unauthorized'}), 401

        # Set user_id based on session_id
        session['user_id'] = session_id
        user_id = session['user_id']

        try:
            # Debug logging
            print(f"Chat request received from session_id: {session_id}")

            # Fetch metadata for the user's files
            metadata = get_files_metadata(user_id)
            active_files = []

            if 'document' in data and data['document']:
                # If a specific document is selected
                active_files = [file['filepath'] for file in metadata if file['filename'] == data['document']]
                print(f"Selected document: {data['document']}")
            else:
                # Use all active documents if no specific document is selected
                active_files = [file['filepath'] for file in metadata if file.get('active')]

            print(f"Active files: {active_files}")

            if not active_files:
                return jsonify({'error': 'No active documents selected'}), 400

            # Process documents if not already processed
            if not document_processor.document_chunks:
                print("Processing selected documents")
                success = document_processor.process_documents(active_files)
                if not success:
                    return jsonify({'error': 'Failed to process documents'}), 500

            # Generate a response from the chat processor
            response = document_processor.chat_with_documents(data['message'], user_id)
            return jsonify({'response': response})

        except Exception as e:
            print(f"Error in chat endpoint: {str(e)}")
            return jsonify({'error': str(e)}), 500

            

    @app.route('/api/delete/<filename>', methods=['POST'])
    def delete_file(filename):
        session_id = request.headers.get('Session-Id')
        if not session_id:
            return jsonify({'error': 'Unauthorized'}), 401

        # Set session variables like other endpoints
        session['user_id'] = session_id
        session['session_id'] = session_id

        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"File {filename} removed from server.")

            # Remove file from metadata using session['user_id']
            remove_file(filename, session['user_id'])

            # Clear document processor state if last active file
            metadata = get_files_metadata(session['user_id'])
            if not any(file.get('active', False) for file in metadata):
                document_processor.clear_document_state()

            return jsonify({'message': 'File deleted successfully'})

        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return jsonify({'error': str(e)}), 500

        
    @app.route('/api/delete-contract/<filename>', methods=['POST'])
    def delete_contract(filename):
        session_id = request.headers.get('Session-Id')
        if not session_id:
            return jsonify({'error': 'Unauthorized'}), 401

        # Set session variables like other endpoints
        session['user_id'] = session_id
        session['session_id'] = session_id

        try:
            filepath = os.path.join(app.config['CONTRACT_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"File {filename} removed from server.")

            # Remove file from metadata using session['user_id']
            remove_contract_file(filename, session['user_id'])

            # Clear document processor state if last active file
            metadata = get_contract_files_metadata(session['user_id'])
            if not any(file.get('active', False) for file in metadata):
                document_processor.clear_document_state()

            return jsonify({'message': 'File deleted successfully'})

        except Exception as e:
            print(f"Error deleting file: {str(e)}")
            return jsonify({'error': str(e)}), 500


    @app.route('/api/toggle-active/<filename>', methods=['POST'])
    def toggle_active(filename):
        session_id = request.headers.get('Session-Id')
        if not session_id:
            return jsonify({'error': 'Unauthorized'}), 401

        session['user_id'] = session_id
        session['session_id'] = session_id

        try:
            metadata = get_files_metadata(session['user_id'])
            active_files = []
            for file in metadata:
                if file['filename'] == filename:
                    file['active'] = not file['active']
                    if file['active']:
                        active_files.append(file['filepath'])
                else:
                    file['active'] = False
            save_files_metadata(metadata, session['user_id'])
            
            if active_files:
                document_processor.process_documents(active_files)
            else:
                document_processor.clear_document_state()
            return jsonify({'message': 'File toggled successfully'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    @app.route('/api/document-summary/<filename>', methods=['GET'])
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

    @app.route('/api/set-active-document', methods=['POST'])
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
        


    # New endpoint to fetch the processing status
    @app.route('/api/processing-status', methods=['GET'])
    def get_processing_status():
        try:
            status = document_processor.get_processing_status()
            return jsonify({'status': status})
        except Exception as e:
            print(f"Error fetching processing status: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/review-contract', methods=['POST'])
    def review_contract():
        session_id = request.headers.get('Session-Id')
        print(f"Session ID received: {session_id}")
        if not session_id:
            return jsonify({'error': 'Unauthorized'}), 401

        # Set session variables
        session['user_id'] = session_id
        session['session_id'] = session_id

        # Retrieve the contract file metadata for the user
        contract_metadata = get_contract_files_metadata(session['user_id'])
        if not contract_metadata or len(contract_metadata) == 0:
            return jsonify({'error': 'No contract file found for the user'}), 404

        # There should be only one contract file per user
        contract_file = contract_metadata[0]
        filepath = contract_file['filepath']
        print(f"Checking file path: {filepath}")
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404

        try:
            # Use your existing document processor to review the contract
            review = document_processor.review_contract([filepath])
            return jsonify({'review': review})
        except Exception as e:
            print(f"Error reviewing contract: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
        
    @app.route('/api/contract-upload', methods=['POST'])
    def contract_uploads():
        session_id = request.headers.get('Session-Id')
        if session_id:
            session['user_id'] = session_id
            session['session_id'] = session_id
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if len(files) > 1:
            return jsonify({'error': 'Maximum 1 file allowed'}), 400

        if not can_upload_more_contract_files(session['user_id']):
            return jsonify({'error': 'Maximum file limit reached. Delete the existing file first.'}), 400

        uploaded_files = []
        contractmetadata = get_contract_files_metadata(session['user_id'])  # Changed from get_files_metadata
        
        # Deactivate all existing files
        for file in contractmetadata:
            file['active'] = False
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['CONTRACT_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filepath)
                contractmetadata.append({
                    'filename': filename,
                    'filepath': filepath,
                    'uploaded_at': datetime.now().isoformat(),
                    'active': True,
                    'user_id': session['user_id']
                })
            else:
                return jsonify({'error': 'Invalid file type'}), 400

        save_contract_files_metadata(contractmetadata, session['user_id'])  # This is correct

        # Process documents
        try:
            # Create a new instance of document processor for contracts if needed
            processing_success = document_processor.process_documents(uploaded_files)
            if not processing_success:
                return jsonify({'error': 'Failed to process documents'}), 500
            return jsonify({'message': 'Files uploaded and processed successfully'})
        except Exception as e:
            print(f"Error processing documents: {str(e)}")  # Debug log
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/contract_files', methods=['GET'])
    def get_uploaded_contracts():
        session_id = request.headers.get('Session-Id')
        if not session_id:
            return jsonify({'error': 'Unauthorized'}), 401

        # Directly set user_id to session_id for testing
        session['user_id'] = session_id

        try:
            metadata = get_contract_files_metadata(session['user_id'])
            return jsonify({'files': metadata})
        except Exception as e:
            print(f"Error fetching files metadata: {str(e)}")
            return jsonify({'error': str(e)}), 500

def main():
    logger.info("Starting main application")
    from multiprocessing import freeze_support
    freeze_support()
    
    app = create_app()
    if app is None:
        logger.error("Failed to create Flask application")
        return
    
    logger.info("Application created successfully, starting server")
    
    """ if os.getenv('FLASK_ENV') == 'production':
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, use_reloader=False) """
        
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)