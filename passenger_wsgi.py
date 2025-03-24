import sys
import os

# Set the project directory (Update this path to your actual Flask app directory)
project_home = u'/home/vimtecco/domains/vimtec.co.ke/public_html/vimtec.co.ke/sheria/aide'

# Add the project directory to sys.path
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set the environment variable for the Flask app
os.environ['FLASK_APP'] = 'app'  # Modify if app.py is inside a folder

# Import the application
from app import app as application
