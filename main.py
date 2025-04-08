from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS, cross_origin
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
from utils.auth_utilis import authenticate_user, register_user, load_users, save_users, register_admin, authenticate_admin
from datetime import datetime
from models import db, User, Document
from multiprocessing import freeze_support
from flask import redirect
from flask_session import Session
import time
import psutil
import threading




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
    CORS(app, supports_credentials=True, origins=["https://doc-processor-theta.vercel.app", "http://localhost:5173", "https://sheria.vimtec.co.ke"])
    
    
    #Session configuration
    #app.config['SESSION_TYPE'] = 'filesystem'
    #app.config['SESSION_COOKIE_SECURE'] = False  # Set to False if not using HTTPS in development
    #app.config['SESSION_COOKIE_HTTPONLY'] = True
    #app.config['SESSION_COOKIE_SAMESITE'] = 'None' # Use 'None' if you need cross-site requests to work
    
    #sess= Session()
    #sess.init_app(app)
    
    # other Configuration
    app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['CONTRACT_FOLDER'] = 'contracts'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    #Additional configurations
    
    
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
        form_columns = ['name', 'is_active']  # Allow toggling active status in the admin panel

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
    
    # Start the system metrics logger in a background thread
    threading.Thread(target=log_memory_usage, daemon=True).start()
    
    return app

def log_system_metrics():
    """Log system metrics periodically."""
    while True:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        logger.info(f"Memory: {memory.percent}%, CPU: {cpu}%, Disk: {disk.percent}%")
        time.sleep(60)  # Log every 60 seconds

def log_memory_usage():
    """Log memory usage of the Flask backend process periodically."""
    process = psutil.Process(os.getpid())  # Get the current process
    while True:
        memory_info = process.memory_info()  # Get memory usage details
        log_message = (
            f"Flask Backend Memory Usage: RSS={memory_info.rss / (1024 ** 2):.2f} MB, "
            f"VMS={memory_info.vms / (1024 ** 2):.2f} MB"
        )
        # Check if 'shared' attribute exists
        if hasattr(memory_info, 'shared'):
            log_message += f", Shared={memory_info.shared / (1024 ** 2):.2f} MB"
        
        logger.info(log_message)
        time.sleep(10)  # Log every 10 seconds

def register_routes(app):
    
    @app.route('/api/health-check')
    def health_check():
        return jsonify({"status": "running"})
    
    @app.route('/api/memory-usage', methods=['GET'])
    def memory_usage():
        """Get memory usage of the Flask backend process."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_data = {
            "rss": memory_info.rss / (1024 ** 2),  # Resident Set Size in MB
            "vms": memory_info.vms / (1024 ** 2),  # Virtual Memory Size in MB
        }
        # Check if 'shared' attribute exists
        if hasattr(memory_info, 'shared'):
            memory_data["shared"] = memory_info.shared / (1024 ** 2)  # Shared memory in MB

        return jsonify(memory_data), 200
    
    
    #Admin endpoints
    @app.route('/admin/register', methods=['POST'])
    def admin_register():
        """Register a new admin."""
        if request.method == 'OPTIONS':
            return jsonify({'message': 'CORS preflight successful'}), 200
    
        try:
            data = request.json
            name = data.get('name')
            password = data.get('password')

            if not name or not password:
                return jsonify({'error': 'Name and password are required'}), 400

            admin = register_admin(name, password)
            if admin:
                return jsonify({'message': 'Admin registered successfully', 'admin': admin}), 201
            else:
                return jsonify({'error': 'Admin already exists'}), 400
        except Exception as e:
            logger.error(f"Error registering admin: {e}")
            return jsonify({'error': 'Failed to register admin'}), 500


    @app.route('/admin/login', methods=['POST'])
    def admin_login():
        """Login an admin."""
        try:
            data = request.json
            name = data.get('name')
            password = data.get('password')

            if not name or not password:
                return jsonify({'error': 'Name and password are required'}), 400

            admin = authenticate_admin(name, password)
            if admin:
                session['admin_id'] = admin['id']
                return jsonify({'message': 'Login successful', 'admin_id': admin['id']}), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
        except Exception as e:
            logger.error(f"Error logging in admin: {e}")
            return jsonify({'error': 'Failed to login admin'}), 500
    
    @app.route('/admin/logout', methods=['POST'])
    def admin_logout():
        """Logout an admin by removing their session."""
        try:
            session.pop('admin_id', None)  # Remove the admin session
            return jsonify({'message': 'Admin logged out successfully'}), 200
        except Exception as e:
            logger.error(f"Error logging out admin: {e}")
            return jsonify({'error': 'Failed to logout admin'}), 500
    
    #New endpoints
    
    @app.route('/api/users', methods=['GET'])
    def get_users():
        """Fetch all users with their details, uploaded files, and contracts."""
        try:
            users = load_users()
            user_data = []
            for user in users:
                files = get_files_metadata(user['id'])
                contracts = get_contract_files_metadata(user['id'])
                user_data.append({
                    "id": user.id,
                    "name": user.name,
                    "is_active": user.is_active,
                    "files": [{"filename": file['filename'], "uploaded_at": file['uploaded_at']} for file in files],
                    "contracts": [{"filename": contract['filename'], "uploaded_at": contract['uploaded_at']} for contract in contracts]
                })
            return jsonify(user_data), 200
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            return jsonify({"error": "Failed to fetch users"}), 500


    @app.route('/api/users/<int:user_id>', methods=['DELETE'])
    def delete_user(user_id):
        """Delete a user and all associated data."""
        try:
            users = load_users()
            user = next((u for u in users if u['id'] == user_id), None)
            if not user:
                return jsonify({"error": "User not found"}), 404

            # Delete user's files
            files = get_files_metadata(user_id)
            for file in files:
                remove_file(file['filename'], user_id)

            # Delete user's contracts
            contracts = get_contract_files_metadata(user_id)
            for contract in contracts:
                remove_contract_file(contract['filename'], user_id)

            # Remove user from the list and save
            users = [u for u in users if u['id'] != user_id]
            save_users(users)

            return jsonify({"message": "User deleted successfully"}), 200
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return jsonify({"error": "Failed to delete user"}), 500


    @app.route('/api/users/<int:user_id>/toggle-status', methods=['POST'])
    def toggle_user_status(user_id):
        """Activate or deactivate a user."""
        try:
            user = User.query.get(user_id)
            if not user:
                return jsonify({"error": "User not found"}), 404

            # Toggle the user's active status
            user.is_active = not user.is_active
            db.session.commit()

            status = "activated" if user.is_active else "deactivated"
            return jsonify({"message": f"User {status} successfully"}), 200
        except Exception as e:
            logger.error(f"Error toggling user status: {e}")
            return jsonify({"error": "Failed to toggle user status"}), 500


#Old endpoints

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