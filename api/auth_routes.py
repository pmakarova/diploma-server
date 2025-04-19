import logging
import re
from flask import Blueprint, request, jsonify

from models.user import create_user, authenticate_user, get_user_by_id
from models.auth_token import validate_token, logout_user

logger = logging.getLogger(__name__)

# Blueprint-объект для группировки маршрутов аутентификации
auth_bp = Blueprint('auth', __name__)

# Маршрут регистрации
@auth_bp.route('/register', methods=['POST'])
def register():
    #logger.info(f"POST запрос к /api/register от {request.remote_addr}") 
    if not request.is_json:
        return jsonify({"success": False, "message": "Content-Type must be application/json"}), 415 # Unsupported Media Type
    
    try:
        data = request.get_json()

        # Обязательные поля
        required_fields = ["name", "email", "password"]

        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"success": False, "message": f"Field {field} is required"}), 400 # Bad Request
        
        # Проверка email
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, data["email"]):
            return jsonify({"success": False, "message": "Invalid email format"}), 400
        
        # Проверка пароля
        if len(data["password"]) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400
        
        # Создание пользователя
        result = create_user(data["name"], data["email"], data["password"])
        
        if result["success"]:
            # Автоматический вход и возврат токена
            auth_result = authenticate_user(data["email"], data["password"])
            return jsonify(auth_result)
        else:
            return jsonify(result), 400
        
    except Exception as e:
        logger.exception(f"Ошибка при регистрации: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500 # Internal Server Error

# Маршрут входа
@auth_bp.route('/login', methods=['POST'])
def login():
    #logger.info(f"POST запрос к /api/login от {request.remote_addr}")
    if not request.is_json:
        return jsonify({"success": False, "message": "Content-Type must be application/json"}), 415
    
    try:
        data = request.get_json()
        
        # Проверка обязательных полей
        if "email" not in data or not data["email"] or "password" not in data or not data["password"]:
            return jsonify({"success": False, "message": "Email and password are required"}), 400
        
        # Аутентификация
        result = authenticate_user(data["email"], data["password"])
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 401 # Unauthorized
        
    except Exception as e:
        logger.exception(f"Ошибка при входе: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

# Маршрут выхода
@auth_bp.route('/logout', methods=['POST'])
def logout():
    #logger.info(f"POST запрос к /api/logout от {request.remote_addr}")
    
    try:
        # Извлечение токена из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"success": False, "message": "Missing authentication token"}), 401
        
        token = auth_header.split('Bearer ')[1]
        
        # Проверка токена
        token_check = validate_token(token)
        if not token_check["valid"]:
            return jsonify({"success": False, "message": token_check["message"]}), 401
        
        # Выход пользователя
        result = logout_user(token)
        return jsonify(result)
        
    except Exception as e:
        logger.exception(f"Ошибка при выходе: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

# Маршрут получения информации о пользователе
@auth_bp.route('/user', methods=['GET'])
def get_user():
    #logger.info(f"GET запрос к /api/user от {request.remote_addr}")
    
    try:
        # Извлечение токена из заголовка
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"success": False, "message": "Missing authentication token"}), 401
        
        token = auth_header.split('Bearer ')[1]
        
        # Проверка токена
        token_check = validate_token(token)
        if not token_check["valid"]:
            return jsonify({"success": False, "message": token_check["message"]}), 401
        
        # Получение информации о пользователе
        user = get_user_by_id(token_check["user_id"])
        if not user:
            return jsonify({"success": False, "message": "User not found"}), 404 # Not Found
        
        return jsonify({"success": True, "user": user})
        
    except Exception as e:
        logger.exception(f"Ошибка при получении пользователя: {str(e)}")
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500