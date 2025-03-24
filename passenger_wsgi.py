import sys
import os

# Set the project directory
project_home = u'/home/vimtecco/domains/vimtec.co.ke/public_html/vimtec.co.ke/sheria/aide'

# Add the project directory to sys.path
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import the application
from app import application