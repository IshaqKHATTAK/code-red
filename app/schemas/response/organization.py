from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class UserData(BaseModel):
    name: str
    email: str

class SuperAdminOrganizations(BaseModel):
    admin_id: int
    organization_id: int
    admin_name: str
    organization_name: str
    is_active: bool
    email: Optional[str] = None
    image_generation: bool = False
    # is_paid: bool 
    # current_plan: Optional[str] = None


class SuperAdminOrganizationsAll(BaseModel):
    admin_id: int
    organization_id: int
    admin_name: str
    organization_name: str
    is_active: bool
    email: Optional[str] = None
    allowed_dept_chatbots: Optional[int] = None
    image_generation: bool = False

class AllSuperAdminOrganizationsWithout(BaseModel):
    organizations: List[SuperAdminOrganizationsAll] = []
    total_organizations: int = 0

class AllSuperAdminOrganizations(BaseModel):
    organizations: List[SuperAdminOrganizations] = []
    total_organizations: int = 0

class ChatbotInfo(BaseModel):
    chatbot_id: int
    chatbot_name: str

class OrganizationResponse(BaseModel):
    id: int
    name: str
    email: str
    total_messages : Optional[int] = None
    is_active:bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    role: Optional[str] = None
    chatbots: Optional[List[ChatbotInfo]] = []
    user_job: Optional[str] = None
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

class OrganizationUsersResponse(BaseModel):
    organization_name: str
    organization_id: int
    users_data: List[OrganizationResponse]
    total_users: int

    class Config:
        from_attributes = True

class UpdateOrganization(BaseModel):
    website_url: Optional[str] = None
    logo : Optional[str] = None
    name: Optional[str] = None
    sso_domain: Optional[str] = None
    allowed_dept_chatbots: Optional[int] = None
    can_create_external_bot:  Optional[bool] = None
    chat_model_name: Optional[str] = None
    is_image_generation_allwed: Optional[bool] = None


class UpdateUserProfile(BaseModel):
    name : Optional[str] = None
    avatar_url : Optional[str] = None
    password: Optional[str] = None
    confirm_password: Optional[str] = None

class AdminiUpdateUserProfile(BaseModel):
    name : Optional[str] = None
    avatar_url : Optional[str] = None
    website_link : Optional[str] = None