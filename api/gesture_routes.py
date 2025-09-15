import logging
import time
import numpy as np
from flask import Blueprint, request, jsonify

from recognition.model_loader import get_model, load_model, AUTO_RECOGNITION_ENABLED
from recognition.gesture_processor import recognize_gesture

logger = logging.getLogger(__name__)

gesture_bp = Blueprint('gesture', __name__)

@gesture_bp.route('/features', methods=['POST'])
def process_gesture():
    """Получает данные и сразу возвращает результат - все в одном запросе"""
    
    if not request.is_json:
        return jsonify({"gesture": "", "confidence": 0.0, "class_id": -1, "error": "Not JSON"}), 400

    if not AUTO_RECOGNITION_ENABLED:
        return jsonify({"gesture": "", "confidence": 0.0, "class_id": -1, "message": "Recognition disabled"})

    try:
        data = request.get_json()
        features = data.get('features', [])
        
        # Только готовые последовательности
        if len(features) != 2520:
            return jsonify({"gesture": "", "confidence": 0.0, "class_id": -1, "error": "Invalid features count"})

        model = get_model()
        if model is None:
            if not load_model():
                return jsonify({"gesture": "", "confidence": 0.0, "class_id": -1, "error": "Model not available"})
            model = get_model()

        # Обработка данных
        features_np = np.array(features, dtype=np.float32)
        num_frames = len(features) // 126
        sequence = features_np.reshape((num_frames, 126))
        
        # Проверка качества
        if np.count_nonzero(sequence) < 50:
            return jsonify({"gesture": "", "confidence": 0.0, "class_id": -1, "message": "Low quality data"})

        # Распознание жеста
        result = recognize_gesture([frame for frame in sequence])
        
        if result and result.get("gesture"):
            logger.info(f"Распознан: {result['gesture']} ({result['confidence']:.3f})")
            return jsonify({
                "gesture": result["gesture"],
                "confidence": result["confidence"], 
                "class_id": result["class_id"],
                "server_timestamp_ms": int(time.time() * 1000)
            })
        else:
            return jsonify({"gesture": "", "confidence": 0.0, "class_id": -1, "message": "No gesture detected"})
            
    except Exception as e:
        logger.exception(f"Ошибка обработки жеста: {e}")
        return jsonify({"gesture": "", "confidence": 0.0, "class_id": -1, "error": str(e)})