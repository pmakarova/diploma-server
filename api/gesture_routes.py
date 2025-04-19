import logging
import queue
import time
import numpy as np
from flask import Blueprint, request, jsonify

from config import Config
from recognition.model_loader import get_model, load_model, AUTO_RECOGNITION_ENABLED
from recognition.feature_collector import feature_data_queue, result_queue
from recognition.gesture_processor import check_sequence_variation, recognize_gesture

logger = logging.getLogger(__name__)

# Blueprint-объект для группировки маршрутов жестов
gesture_bp = Blueprint('gesture', __name__)

# Маршрут для получения признаков жестов
@gesture_bp.route('/features', methods=['POST'])
def receive_features():
    #logger.info(f"POST запрос к /features от {request.remote_addr}")

    if not request.is_json:
        logger.error("Запрос не JSON")
        return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 415

    try:
        features_data = request.get_json()

        if not features_data or 'features' not in features_data:
            logger.error("Отсутствует ключ 'features' в JSON")
            return jsonify({"status": "error", "message": "Отсутствует ключ 'features' в JSON"}), 400

        features = features_data['features']
        # Извлечение времени клиента 
        client_timestamp = features_data.get('timestamp', None)
        server_received_timestamp = str(int(time.time() * 1000))
        #logger.debug(f"Характеристики получены, timestamp: {client_timestamp}, server ts: {server_received_timestamp}")

        # Проверка формата данных
        if not isinstance(features, (list, tuple)):
             msg = f"Неверный формат признаков. Ожидаемый список/кортеж, получен: {type(features)}"
             logger.error(msg)
             return jsonify({"status": "error", "message": msg}), 400

        num_features_received = len(features)
        #logger.debug(f"Получено {num_features_received} значений признаков.")

        # Обработка одного кадра (126 функций)
        if num_features_received == 126:
            #logger.info("Получен один набор признаков (126 значений). Добавление в очередь.")

            # Добавлять в очередь только если включено автоматическое распознавание
            if AUTO_RECOGNITION_ENABLED:
                try:
                    feature_set = np.array([float(x) for x in features], dtype=np.float32)
                except (ValueError, TypeError) as e:
                    logger.error(f"Ошибка преобразования объектов в float: {e}. Data: {str(features)[:100]}...")
                    return jsonify({"status": "error", "message": "Non-numeric data in features"}), 400

                non_zero_count = np.count_nonzero(feature_set)
                if non_zero_count < 10:  # Если почти все нули (10 — произвольный порог)
                     logger.warning(f"Получен почти пустой набор функций ({non_zero_count} ненулевых элементов из 126). Пропуск.")
                     return jsonify({"status": "success", "message": "Features received (mostly empty, skipped)", "timestamp": server_received_timestamp}), 200

                data_to_queue = {
                    'features': feature_set,  # numpy array (126,)
                    'timestamp': client_timestamp or server_received_timestamp,
                    'type': 'features_frame'
                }

                try:
                    feature_data_queue.put_nowait(data_to_queue)
                    #logger.debug(f"Набор функций добавлен в очередь. Размер очереди: {feature_data_queue.qsize()}")
                except queue.Full:
                    logger.warning("Очередь функций полна, данные пропущены!")
                    # Очистить старые данные
                    try:
                        feature_data_queue.get_nowait() # Удалить самый старый элемент
                        feature_data_queue.task_done()
                        feature_data_queue.put_nowait(data_to_queue)
                    except queue.Empty:
                        pass
                    except queue.Full:
                        logger.error("Очередь заполнена после очистки! Сервер перегружен.")
                        return jsonify({"status": "error", "message": "Server overloaded (queue full)"}), 503

            else:
                 logger.info("Автоматическое распознавание отключено, функции игнорируются.")
                 return jsonify({"status": "success", "message": "Features received (auto-recognition disabled)", "timestamp": server_received_timestamp}), 200

        # Обработка последовательности (1260 features)
        elif num_features_received == 1260:
            logger.info("Получена последовательность из 10 наборов признаков (1260 значений).")

            if AUTO_RECOGNITION_ENABLED:
                current_model = get_model()
                if current_model is None:
                    logger.warning("Модель не загружена, попытка загрузить...")
                    success = load_model()
                    if not success:
                        logger.error("Не удалось загрузить модель по запросу")
                        return jsonify({
                            "status": "error", 
                            "message": "Server is still initializing or model file is missing. Please try again in a moment.", 
                            "timestamp": server_received_timestamp
                        }), 200 
                    current_model = get_model()
                
                if current_model is None:
                    logger.error("Модель по-прежнему недоступна после попытки загрузки.")
                    return jsonify({
                        "status": "error", 
                        "message": "Recognition service temporarily unavailable", 
                        "timestamp": server_received_timestamp
                    }), 200 
                
                try:
                    all_features_np = np.array([float(x) for x in features], dtype=np.float32)
    
                    if all_features_np.size != 1260:
                        logger.error(f"Ожидалось 1260 функций, получено {all_features_np.size} после преобразования.")
                        return jsonify({"status": "error", "message": "Invalid number of features after conversion"}), 400

                    feature_sequence_np = all_features_np.reshape((10, 126))

                    # Проверка на наличие одинаковых кадров в последовательности
                    has_variation = check_sequence_variation(feature_sequence_np)
                    if not has_variation:
                        logger.warning(f"Последовательность содержит идентичные кадры. Это может снизить качество распознавания.")

                    feature_sequence_list = [frame for frame in feature_sequence_np]

                    # Check data quality
                    non_zero_count = np.count_nonzero(feature_sequence_np)
                    if non_zero_count < 50:  # Arbitrary threshold for sequence
                        logger.warning(f"Последовательность содержит очень мало данных ({non_zero_count} ненулевых из 1260). Пропуск.")
                        return jsonify({"status": "success", "message": "Sequence received (mostly empty, skipped)", "timestamp": server_received_timestamp}), 200

                    logger.info(f"Начинаем распознавание полученной последовательности из 10 кадров...")
                    result = recognize_gesture(feature_sequence_list)  # Pass list of numpy arrays

                    # Если жест распознан с достаточной уверенностью, добавить в очередь результатов
                    if result and result.get("gesture"): 
                        logger.info(f"Распознан жест {result['gesture']} с уверенностью {result['confidence']:.2f}")
                        result['client_timestamp'] = client_timestamp
                        result['server_received_timestamp'] = server_received_timestamp
                        result['recognition_timestamp'] = str(int(time.time() * 1000))
                        try:
                            result_queue.put_nowait(result)
                            #logger.debug(f"Результат '{result['gesture']}' добавлен в очередь. Размер: {result_queue.qsize()}")
                        except queue.Full:
                             logger.warning("Очередь результатов полна, результат пропущен!")
                    else:
                        logger.info("Жест из последовательности не распознан (низкая уверенность или ошибка).")

                except (ValueError, TypeError) as e:
                    logger.error(f"Ошибка преобразования последовательности признаков в число с плавающей точкой: {e}. Data: {str(features)[:100]}...")
                    return jsonify({"status": "error", "message": "Non-numeric data in sequence features"}), 400
                except Exception as e:
                    logger.exception(f"Ошибка обработки последовательности функций: {str(e)}")
                    return jsonify({"status": "error", "message": f"Sequence processing error: {str(e)}"}), 500
            else:
                logger.info("Автоматическое распознавание отключено, последовательность функций игнорируется.")
                return jsonify({"status": "success", "message": "Sequence received (auto-recognition disabled)", "timestamp": server_received_timestamp}), 200

        # Неверное количество признаков
        else:
            msg = f"Неверное количество признаков. Ожидалось 126 или 1260, получено: {num_features_received}"
            logger.error(msg)
            return jsonify({"status": "error", "message": msg}), 400

        return jsonify({"status": "success", "message": "Features received", "timestamp": server_received_timestamp})

    except Exception as e:
        logger.exception(f"Критическая ошибка обработки/запроса признаков: {str(e)}")
        return jsonify({"error": "Internal server error processing features"}), 500

# Маршрут для получения результатов распознавания жестов
@gesture_bp.route('/translation', methods=['GET'])
def get_translation():
    #logger.debug(f"Запрос к /translation от {request.remote_addr}")
    try:
        try:
            result = result_queue.get_nowait() 
            result_queue.task_done()
            logger.info(f"Жест='{result.get('gesture', 'N/A')}', Уверенность={result.get('confidence', 0.0):.2f}, ID={result.get('class_id', -1)}")

            result['server_timestamp_ms'] = int(time.time() * 1000)
            return jsonify(result)
        
        # Нет новых результатов - нормальное состояние, если нет жестов
        except queue.Empty: 
            #logger.debug("В очереди нет новых результатов.")
            return jsonify({
                "gesture": "",
                "confidence": 0.0,
                "class_id": -1,
                "server_timestamp_ms": int(time.time() * 1000)
            })
    except Exception as e:
        logger.exception(f"Ошибка отправки перевода: {str(e)}")
        return jsonify({"error": "Internal server error getting translation"}), 500

# Маршрут для получения информации о поддерживаемых жестах
@gesture_bp.route('/toggle_auto_recognition', methods=['POST'])
def toggle_auto_recognition():
    global AUTO_RECOGNITION_ENABLED
    if request.is_json:
        data = request.get_json()
        if 'enabled' in data and isinstance(data['enabled'], bool):
            new_state = data['enabled']
            if new_state != AUTO_RECOGNITION_ENABLED:
                AUTO_RECOGNITION_ENABLED = new_state
                logger.info(f"Автораспознавание изменено на: {'ВКЛЮЧЕНО' if AUTO_RECOGNITION_ENABLED else 'ВЫКЛЮЧЕНО'}")

                # Загрузка модели, если еще не загружена
                if AUTO_RECOGNITION_ENABLED and get_model() is None:
                    logger.info("Попытка загрузить модель после включения автоматического распознавания...")
                    if not load_model():
                        logger.error("Не удалось загрузить модель после включения режима!")
                        AUTO_RECOGNITION_ENABLED = False
                        return jsonify({"status": "error", "message": "Failed to load model", "auto_recognition": AUTO_RECOGNITION_ENABLED}), 500
                elif not AUTO_RECOGNITION_ENABLED:
                    logger.info("Автораспознавание отключено. Модель не будет использоваться, пока не будет включено автораспознавание.")
                    # Очистка очереди
                    while not feature_data_queue.empty():
                        try: feature_data_queue.get_nowait()
                        except queue.Empty: break
                        feature_data_queue.task_done()
                    while not result_queue.empty():
                        try: result_queue.get_nowait()
                        except queue.Empty: break
                        result_queue.task_done()
                    logger.info("Очереди признаков и результатов очищены.")

            return jsonify({"status": "success", "auto_recognition": AUTO_RECOGNITION_ENABLED})
        else:
            return jsonify({"status": "error", "message": "Invalid request format: expected {'enabled': true/false}"}), 400
    else:
        return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 415