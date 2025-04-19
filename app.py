#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
import threading
import sys
import codecs

# Установка кодировки для вывода в консоль ПЕРЕД импортом логгера
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

from config import Config
from database.db_manager import init_db
from recognition.model_loader import AUTO_RECOGNITION_ENABLED, load_model
from recognition.feature_collector import process_feature_sequences
from api.auth_routes import auth_bp
from api.gesture_routes import gesture_bp
from utils.logger import setup_logger

# Инициализация Flask приложения
app = Flask(__name__)

# Загрузка конфигурации из config.py
app.config.from_object(Config)

# Запуск логирования
logger = setup_logger()

# Регистрация API-маршрутов
app.register_blueprint(auth_bp, url_prefix='/api')
app.register_blueprint(gesture_bp)

# Корневой маршрут (health check)
@app.route('/', methods=['GET'])
def index():
    from recognition.model_loader import model
    from recognition.feature_collector import feature_data_queue, result_queue
    
    logger.info(f"Request to / from {request.remote_addr}")
    
    # Проверка статуса модели
    model_status = "Not loaded"
    if model is not None:
        model_status = "Loaded and ready"
    elif AUTO_RECOGNITION_ENABLED:
        model_status = "Should be loaded (error?)"

    return jsonify({
        "status": "running",
        "message": "Sign Language Recognition Server",
        "auto_recognition_enabled": AUTO_RECOGNITION_ENABLED,
        "model_status": model_status,
        "feature_queue_size": feature_data_queue.qsize(),
        "result_queue_size": result_queue.qsize()
    })

if __name__ == '__main__':
    # Инициализация базы данных
    with app.app_context():
        init_db()
    
    # Загрузка модели и инициализация потоков
    logger.info("Инициализация сервера...")
    if AUTO_RECOGNITION_ENABLED:
        logger.info("Загрузка модели распознавания жестов...")
        if not load_model():
             logger.error("Не удалось загрузить модель при запуске. Сервер НЕ будет запущен.")
             exit(1)
        logger.info("Модель успешно загружена.")

    # Запуск фонового потока для обработки последовательностей признаков
    # При daemon=True поток будет завершен при завершении основного потока
    processing_thread = threading.Thread(
        target=process_feature_sequences, 
        daemon=True, 
        name="FeatureProcessorThread"
    )
    processing_thread.start()

    logger.info(f"Запуск Flask-сервера на порту {Config.PORT}...")
    logger.info(f"Автоматическое распознавание при запуске: {'ВКЛЮЧЕНО' if Config.AUTO_RECOGNITION_ENABLED else 'ВЫКЛЮЧЕНО'}")
    
    # Запуск Flask-сервера
    app.run(
        host='0.0.0.0', 
        port=Config.PORT, 
        debug=False, 
        threaded=True, 
        use_reloader=False
    )
    
    logger.info("Сервер остановлен.")