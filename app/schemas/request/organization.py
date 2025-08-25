from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class OrganizationBase(BaseModel):
    name: str

class OrganizationCreate(OrganizationBase):
    logo: Optional[str] = None
    website_link: Optional[str] = None
    sso_domain: str
    allowed_dept_chatbots: int 
    can_create_external_bot: bool 
    is_image_generation_allow: Optional[bool] = None
    chat_model_name: Optional[str] = None

   

class OrganizationUpdate(OrganizationBase):
    pass

class OrganizationResponse(OrganizationBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Organization User schemas
class OrganizationUserAdd(BaseModel):
    user_id: int
    role: str = "user"

class OrganizationUserUpdate(BaseModel):
    name: Optional[str] = None
    password: Optional[str] = None
    confirm_password: Optional[str] = None
    chatbot_ids: Optional[List[int]] = None
    user_job: Optional[str] = None

class SuperAdminUpdateUser(OrganizationUserUpdate):
    organization_id: int
    user_id: int

class OrganizationUserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }