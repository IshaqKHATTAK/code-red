from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.models.chatbot_model import ChatbotConfig
from fastapi import HTTPException
from app.models.user import UserRole

async def get_bot_supervisors(db: AsyncSession, chatbot_id: int):
    """Retrieve all supervisors assigned to a chatbot."""
    query = select(User).filter(User.supervisor_chatbot_ids.contains([chatbot_id]))
    result = await db.execute(query)
    supervisors = result.scalars().all()
    
    return supervisors

async def get_bot_users(db: AsyncSession, chatbot_id: int):
    """Retrieve all supervisors assigned to a chatbot."""
    query = select(User).filter(User.chatbot_ids.contains([chatbot_id]))
    result = await db.execute(query)
    supervisors = result.scalars().all()
    
    return supervisors


async def get_user_organization_admin(db: AsyncSession, organization_id:int):
    query = (
        select(User)
        .filter(
            User.organization_id == organization_id,  # Match organization_id
            User.role == UserRole.ADMIN  # Must have ADMIN role
        )
    )
    result = await db.execute(query)
    return result.scalar_one_or_none()
async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Get user by email from database."""
    query = select(User).filter(User.email == email)
    result = await db.execute(query)
    return result.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, id: int) -> User | None:
    """Get user by email from database."""
    query = select(User).filter(User.id == id)
    result = await db.execute(query)
    return result.scalar_one_or_none()

from app.models.organization import Organization

async def get_organization(db: AsyncSession, org_id: int):
    query = select(Organization).filter(Organization.id == org_id)
    result = await db.execute(query)
    return result.scalar_one_or_none()

async def get_user_by_id(db: AsyncSession, user_id: int) -> User:
    """Get user by ID from database."""
    query = select(User).filter(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

async def get_all_organization_users(db: AsyncSession, org_id: int):
    """Get user by ID from database."""
    query = select(User).filter(User.organization_id == org_id)
    result = await db.execute(query)
    users = result.scalars().all() 
    # if not users:
    #     raise HTTPException(status_code=404, detail="User not found")
    return users

async def get_all_organization_chatbots(db: AsyncSession, org_id: int):
    """Get user by ID from database."""
    query = select(ChatbotConfig).filter(ChatbotConfig.organization_id == org_id)
    result = await db.execute(query)
    chatbots = result.scalars().all() 
    # if not chatbots:
    #     raise HTTPException(status_code=404, detail="User not found")
    return chatbots

async def get_chatbot_by_id(db: AsyncSession, bot_id: int) -> User:
    """Get user by ID from database."""
    query = select(ChatbotConfig).filter(ChatbotConfig.id == bot_id)
    result = await db.execute(query)
    chatbot = result.scalar_one_or_none()
    if not chatbot:
        raise HTTPException(status_code=404, detail="User not found")
    return chatbot

async def get_organization_external_chatbot(db: AsyncSession, org_id: int) -> User:
    """Get user by ID from database."""
    query = select(ChatbotConfig).filter(
        ChatbotConfig.organization_id == org_id,
        ChatbotConfig.chatbot_type == "External"
    )
    result = await db.execute(query)
    chatbot = result.scalar_one_or_none()
    
    return chatbot