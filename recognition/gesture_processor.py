#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import time
import numpy as np

from config import Config
from recognition.model_loader import get_model, model, ACTION_LABELS, AUTO_RECOGNITION_ENABLED

logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    0: 'безопасный', 1: 'бумага', 2: 'вероятно', 3: 'весна', 4: 'вода',
    5: 'воздух', 6: 'восемьдесят', 7: 'восьмой', 8: 'высокий', 9: 'газета',
    10: 'девяносто', 11: 'довольно', 12: 'дорога', 13: 'друг', 14: 'ё',
    15: 'ждать', 16: 'картина', 17: 'кроме', 18: 'кусок', 19: 'мертвый',
    20: 'мне', 21: 'мочь', 22: 'назад', 23: 'наконец', 24: 'народ',
    25: 'нас', 26: 'нежный', 27: 'несколько', 28: 'обычно', 29: 'одиннадцать',
    30: 'особенно', 31: 'переводить', 32: 'письмо', 33: 'под', 34: 'последний',
    35: 'постепенно', 36: 'правый', 37: 'рука', 38: 'с', 39: 'сердце',
    40: 'следующий', 41: 'слушать', 42: 'способ', 43: 'сторона', 44: 'сын',
    45: 'твердый', 46: 'твой', 47: 'тебе', 48: 'только', 49: 'тот',
    50: 'тысяча', 51: 'уже', 52: 'улица', 53: 'хороший', 54: 'цена',
    55: 'шестнадцать', 56: 'шестой', 57: 'шестьдесят', 58: 'шум', 59: 'я'
}

def check_sequence_variation(feature_sequence_np):
    """Анализ кадров на различность."""
    
    differences = []
    for i in range(1, len(feature_sequence_np)):
        current_frame = feature_sequence_np[i]
        prev_frame = feature_sequence_np[i-1]
        
        # Абсолютная разница
        diff = np.mean(np.abs(current_frame - prev_frame))
        differences.append(diff)
    
    # Средняя разница между соседними кадрами
    avg_difference = np.mean(differences) if differences else 0
        
    # Если средняя разница между кадрами меньше порога
    if avg_difference < 0.0005: 
        # logger.warning("Последовательность содержит копии одного и того же кадра.")
        # Все равно возвращается True, чтобы продолжить обработку (данные могут быть полезны для распознавания)
        return True
    return True

def recognize_gesture(features_sequence):
    """Распознавание жестов из последовательности."""
    model = get_model()

    if not AUTO_RECOGNITION_ENABLED:
        # logger.warning("Попытка распознавания с отключенным AUTO_RECOGNITION_ENABLED.")
        return {"gesture": "", "confidence": 0.0, "class_id": -1}

    if model is None:
        logger.error("Модель не загружена, распознавание невозможно.")
        return {"gesture": "Error: Model not loaded", "confidence": 0.0, "class_id": -1}

    try:
        # Проверка формата входных данных
        if not isinstance(features_sequence, (list, tuple)) or not all(isinstance(item, np.ndarray) for item in features_sequence):
             logger.error(f"recognize_gesture ожидает список/кортеж массивов numpy, получено: {type(features_sequence)}")
             return {"gesture": "Error: Invalid input data type", "confidence": 0.0, "class_id": -1}

        if len(features_sequence) != 20:
            logger.error(f"recognize_gesture ожидает 10 кадров, получено: {len(features_sequence)}")
            return {"gesture": "Error: Invalid sequence length", "confidence": 0.0, "class_id": -1}

        for i, frame in enumerate(features_sequence):
            if frame.shape != (126,):
                logger.error(f"Кадр {i} имеет форму {frame.shape}, а ожидалось (126,)")
                return {"gesture": "Error: Invalid frame shape", "confidence": 0.0, "class_id": -1}

        # Преобразовать список массивов numpy в один массив numpy с формой [20, 126]
        input_data_np = np.array(features_sequence, dtype=np.float32) 

        # Проверка качества данных
        non_zero_count = np.count_nonzero(input_data_np)
        total_elements = input_data_np.size  # 20 * 126
        non_zero_percentage = (non_zero_count / total_elements) * 100 if total_elements > 0 else 0

        if non_zero_percentage < 3.0:
            logger.warning(f"Низкое качество данных: {non_zero_percentage:.2f}% ненулевые элементы ({non_zero_count}/{total_elements}). Распознавание может быть неточным.")
            return {"gesture": "", "confidence": 0.0, "class_id": -1}

        # [20, 126] -> [1, 20, 126]
        input_data = np.expand_dims(input_data_np, axis=0)

        # Прогноз
        start_time = time.time()
        predictions = model.predict(input_data, verbose=0)  
        prediction_time = time.time() - start_time
        #logger.debug(f"Время прогнозирования модели: {prediction_time:.4f} сек.")

        # Поиск класса с высоким доверием
        scaled_predictions = predictions[0]
        predicted_class_index = np.argmax(scaled_predictions)
        confidence = float(scaled_predictions[predicted_class_index])

        confidence_threshold = Config.CONFIDENCE_THRESHOLD 

        # Фильтр по доверию с пониженным порогом
        if confidence >= confidence_threshold:
            gesture_name = CLASS_MAPPING.get(predicted_class_index, f"Unknown_ID_{predicted_class_index}")
            class_id_to_return = int(predicted_class_index)

            # Топ-5 прогнозов
            try: 
                top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                top_5_values = predictions[0][top_5_indices]
                
                top_5_labels = [CLASS_MAPPING.get(idx, f"Unknown_{idx}") for idx in top_5_indices]
                top_5_info = [f"{label}: {value:.3f}" for label, value in zip(top_5_labels, top_5_values)]
                logger.info(f"Жест распознан: {gesture_name} (id: {class_id_to_return}) | Уверенность: {confidence:.3f} | Top-5: {', '.join(top_5_info)}")
            except Exception as e_top5:
                logger.error(f"Ошибка получения топ-5 прогнозов: {e_top5}")
                logger.info(f"Gesture recognized: {gesture_name} (id: {class_id_to_return}) | Уверенность: {confidence:.3f}")

            return {
                "gesture": gesture_name,
                "confidence": confidence,
                "class_id": class_id_to_return
            }
        else:
            # Низкая уверенность - жест не должен быть распознан
            gesture_name = ""
            confidence_to_return = 0.0
            class_id_to_return = -1
        
            top_class_name = CLASS_MAPPING.get(predicted_class_index, f"Unknown_ID_{predicted_class_index}")
            logger.info(f"Низкая уверенность ({confidence:.3f} < {confidence_threshold:.3f}). Наиболее вероятный класс: {top_class_name} (id: {predicted_class_index}), but result not returned.")
            
            return {
                "gesture": gesture_name,
                "confidence": confidence_to_return,
                "class_id": class_id_to_return
            }

    except Exception as e:
        logger.exception(f"Ошибка распознавания жеста: {str(e)}")
        return {
            "gesture": "Recognition error",
            "confidence": 0.0,
            "class_id": -1
        }