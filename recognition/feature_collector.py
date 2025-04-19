#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time
import queue
import numpy as np

from config import Config
from recognition.gesture_processor import recognize_gesture
from recognition.model_loader import AUTO_RECOGNITION_ENABLED

logger = logging.getLogger(__name__)

# Queues
feature_data_queue = queue.Queue(maxsize=Config.FEATURE_QUEUE_SIZE)  # очередь для данных признаков (кадров)
result_queue = queue.Queue(maxsize=Config.RESULT_QUEUE_SIZE)  # очередь для результатов распознавания

def process_feature_sequences():
    """Обработка полученных признаков."""
    feature_buffer = []  # Буфер для последних 10 наборов признаков (массивы numpy)
    last_recognition_time = time.time()
    min_recognition_interval = Config.MIN_RECOGNITION_INTERVAL  # минимальный интервал между распознаваниями 

    logger.info("Начат поток обработки последовательности признаков.")

    while True:
        try:
            # Получение данных с небольшим тайм-аутом
            try:
                feature_data = feature_data_queue.get(timeout=0.5) 
                item_type = feature_data.get('type')
                timestamp = feature_data.get('timestamp')
            except queue.Empty:
                if time.time() - last_recognition_time > 5.0 and feature_buffer:
                   logger.info("Нет новых кадров в течение длительного времени.")
                   feature_buffer.clear()
                continue

            if item_type == 'features_frame' and AUTO_RECOGNITION_ENABLED:
                current_features = feature_data.get('features')  # numpy array (126,)

                # Проверка типа и формы
                if not isinstance(current_features, np.ndarray) or current_features.shape != (126,):
                    logger.warning(f"Объект неправильного типа/формы: {type(current_features)}, shape: {getattr(current_features, 'shape', 'N/A')}.")
                    feature_data_queue.task_done()
                    continue

                # Добавление признаков в буфер
                feature_buffer.append(current_features)

                if len(feature_buffer) > Config.SEQUENCE_BUFFER_SIZE:
                    feature_buffer.pop(0)  # Удалить самый старый набор

                current_time = time.time()
                if len(feature_buffer) == Config.SEQUENCE_BUFFER_SIZE and (current_time - last_recognition_time) >= min_recognition_interval:
                    logger.info(f"Буфер заполнен ({len(feature_buffer)} кадрами). Распознавание...")

                    buffer_copy = list(feature_buffer)

                    result = recognize_gesture(buffer_copy) 
                    last_recognition_time = time.time() 

                    if result and result.get("gesture"): 
                        result['recognition_timestamp'] = str(int(last_recognition_time * 1000))
                        result['last_frame_timestamp'] = timestamp

                        try:
                            result_queue.put_nowait(result)
                        except queue.Full:
                             logger.warning("Очередь результатов заполнена, результат из потока пропущен!")
                    else:
                         logger.info("Жест не распознан.")

            elif not AUTO_RECOGNITION_ENABLED and item_type == 'features_frame':
                 if feature_buffer:
                     feature_buffer.clear()

            else:
                # Got something unexpected from queue
                logger.warning(f"Неожиданный тип данных '{item_type}'.")

            feature_data_queue.task_done()

        except Exception as e:
            logger.exception(f"Критическая ошибка в потоке обработки признаков: {str(e)}")
            time.sleep(0.5)