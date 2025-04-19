#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import sys
import codecs
import io

def setup_logger():
    """
    Настройка логирования для приложения.
    """
    # Исправление кодировки для вывода в консоль (безопасная проверка)
    if sys.platform == 'win32':
        # Проверяем, что stdout еще не обернут
        if hasattr(sys.stdout, 'buffer'):
            try:
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
            except AttributeError:
                # Если уже обернут, не делаем ничего
                pass
    
    # Создание логгера
    logger = logging.getLogger()
    
    # Установка уровня логирования
    logger.setLevel(logging.INFO)
    
    # Формат логирования
    log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Удаляем существующие обработчики, чтобы избежать дублирования
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Создание консольного обработчика
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Создание обработчика для записи в файл
    # Проверка наличия переменной окружения LOG_DIR
    log_dir = os.environ.get('LOG_DIR', 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'sign_language_server.log')
    
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log-file создан в {log_file}")
    except (PermissionError, IOError) as e:
        logger.warning(f"Не удалось создать Log-file: {str(e)}. Работа с консолью продолжается.")
    
    return logger