import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

USERS_FILE_PATH = 'uploads/users.json'
ADMIN_FILE_PATH = 'uploads/admin.json'

def load_admins():
    """Load admin data from admin.json."""
    if os.path.exists(ADMIN_FILE_PATH):
        with open(ADMIN_FILE_PATH, 'r') as f:
            return json.load(f)
    return []

def save_admins(admins):
    """Save admin data to admin.json."""
    with open(ADMIN_FILE_PATH, 'w') as f:
        json.dump(admins, f)

def authenticate_admin(name, password):
    """Authenticate an admin by name and password."""
    admins = load_admins()
    for admin in admins:
        if admin['name'] == name and check_password_hash(admin['password'], password):
            return admin
    return None

def register_admin(name, password):
    """Register a new admin."""
    admins = load_admins()
    if any(admin['name'] == name for admin in admins):
        return None  # Admin already exists
    admin_id = max(admin['id'] for admin in admins) + 1 if admins else 1
    hashed_password = generate_password_hash(password)
    new_admin = {'id': admin_id, 'name': name, 'password': hashed_password}
    admins.append(new_admin)
    save_admins(admins)
    return new_admin

def load_users():
    if os.path.exists(USERS_FILE_PATH):
        with open(USERS_FILE_PATH, 'r') as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE_PATH, 'w') as f:
        json.dump(users, f)

def authenticate_user(name, password):
    users = load_users()
    for user in users:
        if user['name'] == name and check_password_hash(user['password'], password):
            return user
    return None

def register_user(name, password):
    users = load_users()
    if any(user['name'] == name for user in users):
        return None  # User already exists
    user_id = max(user['id'] for user in users) + 1 if users else 1
    hashed_password = generate_password_hash(password)
    new_user = {'id': user_id, 'name': name, 'password': hashed_password}
    users.append(new_user)
    save_users(users)
    return new_user