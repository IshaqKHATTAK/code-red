from pydantic import BaseModel, EmailStr
from typing import Optional, List
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    SUPER_ADMIN = "super_admin"

class AdminSignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str
    organization_name: str
    logo: Optional[str] = None
    website_link: Optional[str] = None
    sso_domain: str
    description: Optional[str] = None
    allowed_department_bots : int
    allowed_external_bot: bool
    chat_model_name: Optional[str]=None
    is_image_generation_allwed: Optional[bool] = None
    

class LoginUser(BaseModel):
    email: EmailStr
    password: str

class UserCreate(BaseModel):
    """For admin to create organization users"""
    name: str
    email: EmailStr
    password: Optional[str] = None
    confirm_password: Optional[str] = None
    active: bool = False
    chatbot_ids: Optional[List[int]] = None 
    role: str = "USER"
    user_job: Optional[str] = None

class SuperAdminCreateUser(UserCreate):
    organization_id: int

class UserUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

class UserOnboardRequest(BaseModel):
    email: EmailStr
    name: str
    organization_name: Optional[str] = None

class VerifyUser(BaseModel):
    email: EmailStr
    token: str

class PasswordResetRequestOnCreate(BaseModel):
    useremail: str
    new_password: str

class PasswordResetRequest(BaseModel):
    token: str
    useremail: str
    new_password: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str
