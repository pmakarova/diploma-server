#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import Config

logger = logging.getLogger(__name__)

# Инициализация модели
model = None

AUTO_RECOGNITION_ENABLED = Config.AUTO_RECOGNITION_ENABLED

# Словарь для хранения меток действий
def create_sorted_labels(words):
    """
    Создание словаря {id: слово}, отсортированного в алфавитном порядке с автоматически присвоенными индексами
    """
    sorted_words = sorted(words)
    return {i: word for i, word in enumerate(sorted_words)}

# Создание словаря ACTION_LABELS
ACTION_LABELS = create_sorted_labels(Config.SUPPORTED_GESTURES)

# Создание обратного отображения для поиска по слову
ACTION_LABELS_REVERSE = {word: id for id, word in ACTION_LABELS.items()}

def load_model():
    """Загрузка модели для распознавания жестов."""
    global model, AUTO_RECOGNITION_ENABLED
    
    if not AUTO_RECOGNITION_ENABLED:
        logger.info("Автоматическое распознавание отключено. Модель не загружена.")
        return True  # Считать успешным, так как модель не нужна

    if model is None:  # Только один раз
        try:
            logger.info(f"Загрузка модели из {Config.MODEL_PATH}")

            # Настройка логирования TensorFlow
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR

            # Проверка существования файла
            if not os.path.exists(Config.MODEL_PATH):
                logger.error(f"Файл модели не найден по пути: {Config.MODEL_PATH}")
                return False

            # Загрузка модели
            model = keras.models.load_model(Config.MODEL_PATH)
            logger.info("Модель загружена")

            # «Прогрев» модели и проверка выходной структуры
            try:
                # Expected input shape: (batch_size, sequence_length, num_features)
                # In our case: (1, 10, 126)
                test_input = np.zeros((1, 10, 126), dtype=np.float32)
                test_output = model.predict(test_input, verbose=0)

                logger.info(f"Модель успешно загружена и проверена. Форма выходных данных: {test_output.shape}")

                # Проверка количества классов
                num_classes_model = test_output.shape[1]
                num_classes_labels = len(ACTION_LABELS)
                logger.info(f"Модель имеет {num_classes_model} выходных классов.")

                if num_classes_labels != num_classes_model:
                    logger.warning(f"ВНИМАНИЕ: Количество меток ({num_classes_labels}) не соответствует количеству выходных классов модели ({num_classes_model})!")
                else:
                    logger.info("Количество меток соответствует выходным данным модели.")

                logger.info("Модель успешно загружена и проверена.")
                return True

            except Exception as e:
                logger.exception(f"Ошибка проверки модели: {str(e)}")
                model = None  # Сброс модели если проверка не удалась
                return False

        except Exception as e:
            logger.exception(f"Критическая ошибка при загрузке модели: {str(e)}")
            model = None  # Сброс модели
            return False

    return True  # Модель уже загружена

def get_model():
    """Получить загруженный экземпляр модели."""
    global model
    return model