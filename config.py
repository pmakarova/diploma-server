#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

class Config:
    '''Сохраняет все пути и параметры приложения.'''
    '''Перед запуском проверьте доступность модели и доступ к порту!'''
    
    # Пути и основные настройки
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model_lstm_60.h5')  # Путь к модели распознавания жестов
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'users.db')  # Путь к базе данных
    PORT = int(os.environ.get('PORT', 5000))  # Порт для сервера
    
    # Настройки распознавания
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.75))  # Порог уверенности для распознавания жестов
    AUTO_RECOGNITION_ENABLED = os.environ.get('AUTO_RECOGNITION_ENABLED', 'True').lower() == 'true'  # Переключатель автоматического распознавания
    
    # Настройки аутентификации
    TOKEN_EXPIRY = int(os.environ.get('TOKEN_EXPIRY', 30))  # Срок действия токена в днях
    
    # Путь к карте меток — если указан, переопределяет расположение по умолчанию рядом с моделью
    LABEL_MAP_PATH = os.environ.get('LABEL_MAP_PATH', '')
    
    # Параметры обработки последовательности
    MIN_RECOGNITION_INTERVAL = float(os.environ.get('MIN_RECOGNITION_INTERVAL', 0.2))  # Минимальный интервал между распознаваниями
    SEQUENCE_BUFFER_SIZE = int(os.environ.get('SEQUENCE_BUFFER_SIZE', 10))  # Размер буфера последовательности
    
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'