#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

class Config:
    '''Хранит все пути и параметры приложения.'''
    '''Перед запуском проверить наличие модели и доступного порта!!'''
    
    
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model_lstm_drop.h5') # путь к модели определения жестов
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'users.db') # путь к базе данных
    PORT = int(os.environ.get('PORT', 5000)) # порт, на котором будет работать сервер
    
    # Настройки для распознавания жестов
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.6)) # порог уверенности модели
    AUTO_RECOGNITION_ENABLED = os.environ.get('AUTO_RECOGNITION_ENABLED', 'True').lower() == 'true' # автоматическое распознавание жестов
    
    # Настройки аутентификации
    TOKEN_EXPIRY = int(os.environ.get('TOKEN_EXPIRY', 30)) # время жизни токена
    
    # Поддерживаемые жесты
    SUPPORTED_GESTURES = [
        "бинокль",
        "здравствуйте",
        "привет",
        "читать"
    ]
    SUPPORTED_GESTURES = sorted(SUPPORTED_GESTURES)

    # Очереди
    FEATURE_QUEUE_SIZE = int(os.environ.get('FEATURE_QUEUE_SIZE', 50)) # для сбора признаков
    RESULT_QUEUE_SIZE = int(os.environ.get('RESULT_QUEUE_SIZE', 20)) # для результатов распознавания
    
    # Параметры для обработки последовательностей
    MIN_RECOGNITION_INTERVAL = float(os.environ.get('MIN_RECOGNITION_INTERVAL', 0.1)) # минимальный интервал между распознаванием жестов
    SEQUENCE_BUFFER_SIZE = int(os.environ.get('SEQUENCE_BUFFER_SIZE', 10)) # размер буфера для последовательностей