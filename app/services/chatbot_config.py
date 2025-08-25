from app.models.chatbot_model import ChatbotConfig
from fastapi import HTTPException,status
from datetime import datetime, timezone
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError

async def _get_chatbot_config(data,session):
    result = await session.execute(select(ChatbotConfig).filter(ChatbotConfig.id == data.id)) 
    return result.scalars().first()

async def create_chatbot_config(data, session):
    try:    
        chatbot_config = ChatbotConfig(
            # llm_model_name=data.llm_model_name,
            llm_temperature=data.llm_temperature,
            llm_prompt=data.llm_prompt,
            llm_role=data.llm_role,
            llm_streaming=data.llm_streaming,  
        )
        session.add(chatbot_config)
        await session.commit()
        return chatbot_config
    except SQLAlchemyError as e:
        await session.rollback()  
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error occurred: {str(e)}"
        )
    except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}"
            )

async def get_chatbot_config(data,session):
    chatbot_config = await _get_chatbot_config(data, session)
    if chatbot_config is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No chatbot config found in database")
    
    return chatbot_config

async def update_chatbot_config(data, session):
    chatbot_setup = await _get_chatbot_config(data, session)
    if chatbot_setup is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No chatbot config found in database")
    
    update_dict = {k: v for k, v in data.dict().items() if v is not None}
    if len(update_dict) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid data found in request")

    for key, value in update_dict.items():
        setattr(chatbot_setup, key, value)
    
    chatbot_setup.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return chatbot_setup

