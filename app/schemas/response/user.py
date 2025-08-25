from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from app.models.user import UserRole

class ChatbotInfo(BaseModel):
    chatbot_id: int
    chatbot_name: str


class ExistingUser(BaseModel):
    name: str
    email: str

class BulkUploadResponse(BaseModel):
    created_users: List[dict]
    existing_users: List[dict]
    total_created: int
    total_existing: int


class MeUserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    total_messages: int
    organization_id: Optional[int] = None
    organization_name: Optional[str] = None
    chatbots: Optional[List[int]] = []
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    is_verified: bool = False
    verified_at: Optional[datetime] = None
    avatar_url: Optional[str] = None
    trail_expirey: Optional[datetime] = None
    can_create_external_bot: Optional[int] = 0
    allowed_dept_chatbots: Optional[int] = 0 
    user_job: Optional[str] = None
    supervisors: Optional[List[int]] = []
    image_generation: bool = False
    # is_supervisor:  bool = False

    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    total_messages: int
    organization_id: Optional[int] = None
    organization_name: Optional[str] = None
    chatbots: Optional[List[ChatbotInfo]] = []
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    is_verified: bool = False
    verified_at: Optional[datetime] = None
    avatar_url: Optional[str] = None
    trail_expirey: Optional[datetime] = None
    can_create_external_bot: Optional[int] = 0
    allowed_dept_chatbots: Optional[int] = 0 
    user_job: Optional[str] = None
    is_image_generation_allwed: Optional[bool] = None
    chat_model_name: Optional[str] = None

    # is_supervisor:  bool = False

    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"

class TokenData(BaseModel):
    user_id: int
    email: str
    role: UserRole


class PasswordResetResponse(BaseModel):
    message: str
    status: str
    
# class CreatedUser(BaseModel):
#     id: int
#     name: str
#     email: str
#     user: UserResponse
