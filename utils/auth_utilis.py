import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

USERS_FILE_PATH = 'uploads/users.json'

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