from app.models.user import User, UserRole
from fastapi import HTTPException
from app.models.chatbot_model import ChatbotConfig

def validate_user_role(role: str) -> str:
    """Validate user role"""
    if role not in [r.value for r in UserRole]:
        raise ValueError(f"Invalid role: {role}")
    return role

async def toggle_user_status(user: User) -> User:
    """Toggle user active status"""
    user.is_active = not user.is_active
    if not user.is_verified:
        user.is_verified = True
    return user

async def toggle_chatbot_status(chatbot: ChatbotConfig) -> User:
    """Toggle user active status"""
    chatbot.memory_status = not chatbot.memory_status
    return chatbot

async def toggle_true_user_status(user: User) -> User:
    """Toggle user active status"""
    user.is_active = True
    return user

async def toggle_false_user_status(user: User) -> User:
    """Toggle user active status"""
    user.is_active = False
    return user

async def toggle_user_paid_status(user: User) -> User:
    """Toggle user paid status"""
    # user.is_paid = not user.is_paid
    return user