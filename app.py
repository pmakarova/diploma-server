#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
import sys
import codecs

if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

from config import Config
from database.db_manager import init_db
from recognition.model_loader import AUTO_RECOGNITION_ENABLED, load_model, ACTION_LABELS
from api.auth_routes import auth_bp
from api.gesture_routes import gesture_bp
from utils.logger import setup_logger

app = Flask(__name__)
app.config.from_object(Config)

logger = setup_logger()

# Регистрация маршрутов
app.register_blueprint(auth_bp, url_prefix='/api')
app.register_blueprint(gesture_bp)

@app.route('/', methods=['GET'])
def index():
    from recognition.model_loader import model
    
    model_status = "Loaded and ready" if model else "Not loaded"
    
    return jsonify({
        "status": "running",
        "message": "Sign Language Recognition Server",
        "auto_recognition_enabled": AUTO_RECOGNITION_ENABLED,
        "model_status": model_status,
        "num_classes": len(ACTION_LABELS)
    })

if __name__ == '__main__':
    # Инициализация БД
    with app.app_context():
        init_db()
    
    # Загрузка модели
    if AUTO_RECOGNITION_ENABLED:
        logger.info("Загрузка модели...")
        if not load_model():
            logger.error("Не удалось загрузить модель!")
            exit(1)
        logger.info("Модель загружена")

    logger.info(f"Запуск сервера на порту {Config.PORT}")
    
    app.run(
        host='0.0.0.0', 
        port=Config.PORT, 
        debug=False, 
        threaded=True, 
        use_reloader=False
    )