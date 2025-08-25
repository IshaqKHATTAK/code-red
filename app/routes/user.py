from fastapi import APIRouter, Depends, HTTPException, status, Response, Request, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.common.database_config import get_async_db
from app.models.user import User
from app.schemas.request.user import UserCreate, UserUpdate, PasswordChange,VerifyUser,LoginUser, UserOnboardRequest, AdminSignupRequest, PasswordResetRequestOnCreate, PasswordResetRequest
from app.schemas.response.user import MeUserResponse, UserResponse, LoginResponse, PasswordResetResponse
from app.models.user import UserRole
from app.schemas.response.chatbot_config import LlmModelEnum
from app.services.user import (
    create_user,
    login_user,
    onboard_user,
    get_user_list,
    get_user_by_id_with_auth,
    update_user_details,
    toggle_user_active_status_service,
    toggle_user_paid_status_service,
    create_user,
    async_logout_user
)
from fastapi.responses import JSONResponse
from typing import List
from app.services.auth import get_current_user
from fastapi.security import OAuth2PasswordRequestForm
from app.common.security import SecurityManager
from app.services import email
from app.utils.db_helpers import get_user_by_email, get_user_organization_admin,get_organization
from datetime import timedelta


router = APIRouter(
    prefix="/api/v1/users",
    tags=["users"],
    dependencies=[Depends(get_current_user)]
)

public_router = APIRouter(
    prefix="/api/v1/users", #
    tags=["users"]
)
from app.common.env_config import get_envs_setting

envs = get_envs_setting()

@public_router.get("/refresh", response_model=LoginResponse)
async def refresh_token(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_async_db)
):
    """Generate new access token using refresh token from cookie"""
    # Get refresh token from cookie
    print(f'starts')
    security_manager = SecurityManager()
    refresh_token = request.cookies.get(security_manager.refresh_cookie_name)
    print(f'refresh token == {refresh_token}')
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing"
        )
    
    # Use security manager to refresh token
    tokens = await security_manager.refresh_access_token(refresh_token, db)
    print(f'token = {tokens}')
    # Set new refresh token cookie
    security_manager.set_session_cookies(response, refresh_token)
    print(f'fnsihed')
    return tokens

@public_router.post("/create-organization", response_model=UserResponse)
async def signup(
    signup_data: AdminSignupRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Sign up new admin with their organization and send verification email"""
    # Create user with is_verified=False
    # print(f'can = {signup_data.allowed_external_bot}')
    if signup_data.password != signup_data.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Password and confirm password does not match."
        )
    if not signup_data.allowed_external_bot and signup_data.allowed_department_bots < 1:
        if signup_data.chat_model_name or signup_data.is_image_generation_allwed:
            raise HTTPException(
            status_code=400,
            detail="You need to allow at least one chatbot to set image generation and chat model of organization."
        )
        
    if signup_data.chat_model_name and signup_data.chat_model_name not in LlmModelEnum.__members__.values():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model name '{signup_data.chat_model_name}'. Allowed values: {list(LlmModelEnum.__members__.values())}"
            )
    user = await create_user(db, signup_data, current_user)
    from app.utils.database_helper import get_orgnization
    org = await get_orgnization(user.organization_id, db)
    print(f'chat model name = {org.chat_model_name}')
    response = UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role,  # Assuming role is an Enum
        total_messages=user.total_messages,
        organization_id=org.id if org else None,
        organization_name=org.name if org else None,
        chatbots=user.chatbot_ids or [],
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        is_verified=user.is_verified,
        verified_at=user.verified_at,
        avatar_url=user.avatar_url,
        trail_expirey= None,
        can_create_external_bot=org.can_create_external_bot if org else 0,
        allowed_dept_chatbots=org.allowed_dept_chatbots if org else 0,
        chat_model_name=org.chat_model_name,
        is_image_generation_allwed = org.is_image_generation_allow
        #is_supervisor=user.is_supervisor if hasattr(user, "is_supervisor") else False,
    )
    return response
    # Generate verification token
    # verification_token = SecurityManager.create_verification_token(user)
    #Send verification email in background
    # background_tasks.add_task(
    #     email.send_verification_email,
    #     user.email,
    #     user.name,
    #     verification_token
    # )
    
    # return UserResponse.model_validate(user)

@router.get("/me", response_model=MeUserResponse)
async def get_current_user_details(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get current user details"""
    org_data = await get_organization(db, current_user.organization_id)
    image_generation = False
    # if current_user.role in [UserRole.USER]:
    #     #get admin of organization
    #     admin_user = await get_user_organization_admin(db, current_user.organization_id)
    #     image_generation = admin_user.is_image_generation_allow
    # else:
    #     image_generation = current_user.is_image_generation_allow
    #     # org_plan = admin_user.
    if current_user.role in [UserRole.USER, UserRole.ADMIN]:
        # organization = await get_organization(db=db, org_id=current_user.organization_id)
        image_generation = org_data.is_image_generation_allow if org_data.is_image_generation_allow else False
    return MeUserResponse(
        id = current_user.id,
        name = current_user.name,
        email = current_user.email,
        role = current_user.role,
        total_messages = current_user.total_messages,
        organization_id = current_user.organization_id if current_user.role != UserRole.SUPER_ADMIN else None,
        organization_name = org_data.name if current_user.role != UserRole.SUPER_ADMIN and org_data else "",
        chatbots  = current_user.chatbot_ids if current_user.chatbot_ids else None,
        is_active = current_user.is_active,
        created_at = current_user.created_at,
        updated_at = current_user.updated_at,
        is_verified  =current_user.is_verified,
        verified_at = current_user.verified_at,
        avatar_url  = current_user.avatar_url,
        image_generation = image_generation,
        # is_supervisor = True if current_user.supervisor_chatbot_ids and len(current_user.supervisor_chatbot_ids) > 0 else False,
        allowed_dept_chatbots = None if current_user.role == UserRole.USER else org_data.allowed_dept_chatbots if org_data else 0,
        can_create_external_bot = None if current_user.role == UserRole.USER else org_data.can_create_external_bot if org_data else True,
        trail_expirey =  current_user.created_at + timedelta(days=envs.FREE_TRAIL_DAYS),
        supervisors = current_user.supervisor_chatbot_ids if current_user.supervisor_chatbot_ids else None
    )

@public_router.post("/login", response_model=LoginResponse)
async def login(
    response: Response,
    request: Request,
    login_data:  OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_async_db)
):
    """Login user if email is verified"""
    # Check if user is verified before allowing login
    user = await get_user_by_email(db, login_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Account not found"
        )
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email before logging in"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivted."
        )
    if user.role == UserRole.USER:
        admin = await get_user_organization_admin(db=db, organization_id=user.organization_id)
        if not admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your organization has been deactivted. Please refer to your organization admin."
            )   
    return await login_user(login_data, response, request, db)

@public_router.post("/onboard", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def onboard(user_data: UserOnboardRequest, db: AsyncSession = Depends(get_async_db)):
    return await onboard_user(db, user_data)

@router.patch("/{user_id}/toggle-active")
async def toggle_user_active_status(
    user_id: int,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    user = await toggle_user_active_status_service(user_id, current_user, db)
    return UserResponse.model_validate(user)

@router.patch("/{user_id}/toggle-paid")
async def toggle_user_paid_status(
    user_id: int,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    user = await toggle_user_paid_status_service(user_id, current_user, db)
    return UserResponse.model_validate(user)

# @router.get("/{user_id}", response_model=UserResponse)
# async def get_user(
#     user_id: int,
#     db: AsyncSession = Depends(get_async_db),
#     current_user: User = Depends(get_current_user)
# ) -> UserResponse:
#     user = await get_user_by_id_with_auth(user_id, current_user, db)
#     return UserResponse.model_validate(user)

# @router.get("/", response_model=List[UserResponse])
# async def get_users(
#     skip: int = 0,
#     limit: int = 10,
#     db: AsyncSession = Depends(get_async_db),
#     current_user: User = Depends(get_current_user)
# ) -> List[UserResponse]:
#     users = await get_user_list(skip, limit, current_user, db)
#     return [UserResponse.model_validate(user) for user in users]

# @router.patch("/{user_id}", response_model=UserResponse)
# async def update_user(
#     user_id: int,
#     user_data: UserUpdate,
#     db: AsyncSession = Depends(get_async_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Update user details"""
#     return await update_user_details(user_id, user_data, current_user, db)


@public_router.get("/verify/{useremail}/{token}")
async def verify_email(
    token: str,
    useremail: str,
    db: AsyncSession = Depends(get_async_db)
):
    if not useremail and token:
        raise HTTPException(status_code=400, detail="Enter email and token both.")
    return await email.verify_email(token, useremail, db)

@public_router.post("/reset-password/", response_model=PasswordResetResponse)
async def reset_password_endpoint(
    request: PasswordResetRequest, 
    db: AsyncSession = Depends(get_async_db)
):
    return await email.reset_password(request.token, request.useremail, request.new_password, db)


@public_router.post("/reset-password/{token}", response_model=PasswordResetResponse)
async def reset_password_endpoint(
    token: str,
    request: PasswordResetRequestOnCreate, 
    db: AsyncSession = Depends(get_async_db)
):
    return await email.reset_password(token, request.useremail, request.new_password, db)


@router.post("/change-password", status_code = status.HTTP_200_OK)
async def get_current_user_details(
    request: PasswordChange, 
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user)
):
    """Reset the current user password."""
    if request.new_password != request.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New passwords do not match"
        )
    return await email.change_user_password(current_user, request.current_password, request.new_password, db)


@router.post('/logout', status_code = status.HTTP_200_OK)
async def logout_user(response: Response,session: AsyncSession = Depends(get_async_db), current_user: User = Depends(get_current_user)):
    try:
        response_data = await async_logout_user(session, current_user)
        response.delete_cookie(key="refresh_token", domain=envs.COOKIE_DOMAIN)
        response.delete_cookie(key="access_token", domain=envs.COOKIE_DOMAIN)
    
        return JSONResponse({"message": response})
    except HTTPException as e:
        raise e
    

    