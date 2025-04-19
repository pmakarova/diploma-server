#!/usr/bin/env python
# -*- coding: utf-8 -*-
import hashlib
import logging
from datetime import datetime, timedelta
import secrets

from database.db_manager import get_db
from config import Config

logger = logging.getLogger(__name__)

def hash_password(password):
    """Хэш-пароль с использованием SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(name, email, password):
    """Создание нового пользователя."""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Проверка существования пользователя с таким email
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return {"success": False, "message": "User with this email already exists"}
        
        # Хэш-пароль
        password_hash = hash_password(password)
        
        # Создание пользователя
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash)
        )
        user_id = cursor.lastrowid
        db.commit()
        
        return {"success": True, "user_id": user_id}
    except Exception as e:
        logger.exception(f"Error creating user: {str(e)}")
        return {"success": False, "message": f"Error creating user: {str(e)}"}

def authenticate_user(email, password):
    """Авторизация пользователя."""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Найти пользователя по email
        cursor.execute("SELECT id, name, password_hash FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            return {"success": False, "message": "User not found"}
        
        # Проверка пароля
        password_hash = hash_password(password)
        if password_hash != user['password_hash']:
            return {"success": False, "message": "Invalid password"}
        
        # Создание токена
        token = secrets.token_hex(32)
        expires_at = datetime.now() + timedelta(days=Config.TOKEN_EXPIRY)
        
        # Удаление старого токена
        cursor.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user['id'],))
        
        # Сохранение нового токена
        cursor.execute(
            "INSERT INTO auth_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user['id'], token, expires_at)
        )
        db.commit()
        
        return {
            "success": True, 
            "user_id": user['id'], 
            "name": user['name'], 
            "token": token, 
            "expires_at": expires_at.isoformat()
        }
    except Exception as e:
        logger.exception(f"Ошибка аутентификации пользователя: {str(e)}")
        return {"success": False, "message": f"Authentication error: {str(e)}"}

def get_user_by_id(user_id):
    """Получение информации о пользователе по ID."""
    try:
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute("SELECT id, name, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return None
        
        return dict(user)
    except Exception as e:
        logger.exception(f"Ошибка получения пользователя: {str(e)}")
        return None