#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from datetime import datetime

from database.db_manager import get_db

logger = logging.getLogger(__name__)

def validate_token(token):
    """Проверка токена аутентификации."""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Найти токен в базе данных
        cursor.execute("""
            SELECT auth_tokens.id, auth_tokens.user_id, auth_tokens.expires_at, users.name 
            FROM auth_tokens 
            JOIN users ON auth_tokens.user_id = users.id 
            WHERE auth_tokens.token = ?
        """, (token,))
        
        token_data = cursor.fetchone()
        
        if not token_data:
            return {"valid": False, "message": "Token not found"}
        
        # Проверка срока действия токена
        expires_at = datetime.fromisoformat(token_data['expires_at'])
        if expires_at < datetime.now():
            return {"valid": False, "message": "Token expired"}
        
        return {
            "valid": True, 
            "user_id": token_data['user_id'],
            "name": token_data['name']
        }
    except Exception as e:
        logger.exception(f"Ошибка проверки токена: {str(e)}")
        return {"valid": False, "message": f"Error validating token: {str(e)}"}

def logout_user(token):
    """Удаление токена аутентификации."""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Удалить токен из базы данных
        cursor.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
        db.commit()
        
        return {"success": True}
    except Exception as e:
        logger.exception(f"Ошибка выхода пользователя из системы: {str(e)}")
        return {"success": False, "message": f"Error during logout: {str(e)}"}