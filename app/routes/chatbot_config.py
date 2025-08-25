from fastapi import APIRouter, Depends, status 
from app.schemas.request.chatbot_config import UpdateChatbotConfigRequest,InputData,CreateData
from app.schemas.response.chatbot_config import ChatbotConfigResponse
from app.services import chatbot_config
from app.common import database_config
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse


chatbot_config_routes = APIRouter(
    prefix="/api/v1/admin",
    tags=["Admin"],
    responses={404: {"description": "Not found"}},
    # dependencies=[Depends(oauth2_scheme), Depends(validate_access_token), Depends(is_admin)]
)

# @chatbot_config_routes.post("/create", status_code=status.HTTP_200_OK)
# async def update_admin_config(data:CreateData, session: Session = Depends(database_config.get_async_db)): 
#     config_data =  await chatbot_config.create_chatbot_config(data, session)
#     return JSONResponse({"message": "Chatbot config has been created successfully"})

# @chatbot_config_routes.post("/config", status_code=status.HTTP_200_OK, response_model=ChatbotConfigResponse)
# async def update_admin_config(data:InputData, session: Session = Depends(database_config.get_async_db)): #, 
#     config_data =  await chatbot_config.get_chatbot_config(data, session)
#     return ChatbotConfigResponse(id = config_data.id, llm_model_name=config_data.llm_model_name, llm_temperature=config_data.llm_temperature, llm_prompt=config_data.llm_prompt, llm_role=config_data.llm_role, llm_streaming=config_data.llm_streaming)

# @chatbot_config_routes.put("/update", status_code=status.HTTP_200_OK)
# async def update_admin_config(data: UpdateChatbotConfigRequest, session: Session = Depends(database_config.get_async_db)): #, session: Session = Depends(get_session)
#     await chatbot_config.update_chatbot_config(data, session)
#     return JSONResponse({"message": "Chatbot config has been updated successfully"})

