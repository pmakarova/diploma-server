#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import g # для хранения общих данных во время запроса
import logging
import sqlite3

from config import Config

logger = logging.getLogger(__name__)

def get_db():
    """Подключение к базе данных."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(Config.DATABASE_PATH)
        db.row_factory = sqlite3.Row  # обращение к колонкам по имени
    return db

def close_connection(exception):
    """Закрытие соединения с базой данных по окончании запроса."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    """Инициализация базы данных при первом запуске."""
    db = get_db()
    cursor = db.cursor() # для выполнения SQL-запросов
    
    # Создание таблицы пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Создание таблицы пользователей для хранения токенов аутентификации
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS auth_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        token TEXT UNIQUE NOT NULL,
        expires_at TIMESTAMP NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    db.commit()
    logger.info("База данных инициализирована")