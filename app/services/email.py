from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from fastapi import BackgroundTasks
from typing import List
from pydantic import EmailStr
from app.common.env_config import get_envs_setting
from sqlalchemy.ext.asyncio import AsyncSession
from app.common.database_config import get_async_db
from app.services import payment
from app.utils.db_helpers import get_user_by_email
from app.common.security import SecurityManager
# from app.services.user import verify_user_email
from fastapi import Depends, HTTPException, status
from app.models.user import User
settings = get_envs_setting()

# Email configuration using environment variables
conf = ConnectionConfig(
    MAIL_USERNAME = settings.MAIL_USERNAME,
    MAIL_PASSWORD = settings.MAIL_PASSWORD,
    MAIL_FROM = settings.MAIL_FROM,
    MAIL_PORT = settings.MAIL_PORT,
    MAIL_SERVER = settings.MAIL_SERVER,
    MAIL_STARTTLS = settings.MAIL_STARTTLS,
    MAIL_SSL_TLS = settings.MAIL_SSL_TLS,
    USE_CREDENTIALS = settings.MAIL_USE_CREDENTIALS,
    MAIL_DEBUG=settings.MAIL_DEBUG,
    MAIL_FROM_NAME=settings.MAIL_FROM_NAME
)
    
    
async def send_verification_email(email: EmailStr, name: str, token: str):
    """Send verification email to user"""
    try:
        # Create verification link
        verification_url = f"{settings.FRONTEND_HOST}users/{email}/verify/{token}"
        # Email template
        html = f"""
            <div style="font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                <h2 style="color: #2c3e50;">Welcome to DINBOT, {name}!</h2>
                <p>We are thrilled to have you on board. To get started, please confirm your email address by clicking the button below.</p>
                
                <div style="text-align: center; margin: 20px 0;">
                    <a href="{verification_url}" 
                    style="background-color: #007bff; color: #fff; text-decoration: none; padding: 12px 20px; border-radius: 5px; font-size: 16px; display: inline-block;">
                        Verify Your Email
                    </a>
                </div>

                <p>Verifying your email helps us secure your account and provide you with the best experience.</p>
                <p><strong>Note:</strong> This verification link is valid for 24 hours. If you did not sign up for an account, please ignore this email.</p>
                
                <p>Welcome aboard, <br><strong>The DINBOT Team</strong></p>
            </div>
        """
        
        # Create message
        message = MessageSchema(
            subject="Verify Your Email",
            recipients=[email],
            body=html,
            subtype="html"
        )
        
        # Send email
        fm = FastMail(conf)
        await fm.send_message(message)
        print(f"Verification email sent to {email}")
        
    except Exception as e:
        print(f"Error sending email: {e}")
        raise 

async def send_set_password_email(email: EmailStr, name: str, token: str):
    """Send email to newly created user to set their account password"""
    try:
        # Create set password link
        set_password_url = f"{settings.FRONTEND_HOST}set-password/{token}"
                            

        # Email template
        html = f"""
            <div style="font-family: Arial, sans-serif; color: #333; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                <h2 style="color: #2c3e50;">Welcome to DINBOT, {name}!</h2>
                <p>An account has been created for you by our admin team. To activate your account and start using DINBOT, please set your password by clicking the button below.</p>

                <div style="text-align: center; margin: 20px 0;">
                    <a href="{set_password_url}" 
                    style="background-color: #17a2b8; color: #fff; text-decoration: none; padding: 12px 20px; border-radius: 5px; font-size: 16px; display: inline-block;">
                        Set Your Password
                    </a>
                </div>

                <p>This link is valid for limited time. Once your password is set, you can log in using your email and new password.</p>
                <p>If you were not expecting this email, please ignore it.</p>

                <p>Thanks and welcome, <br><strong>The DINBOT Team</strong></p>
            </div>
        """

        # Create message
        message = MessageSchema(
            subject="Set Up Your DINBOT Account",
            recipients=[email],
            body=html,
            subtype="html"
        )

        # Send email
        fm = FastMail(conf)
        await fm.send_message(message)
        print(f"Set password email sent to {email}")

    except Exception as e:
        print(f"Error sending set password email: {e}")
        raise


async def verify_email(
    token,
    useremail,
    db: AsyncSession = Depends(get_async_db)
):
    """Verify user email with token"""
    try:
        # Get user
        user = await get_user_by_email(db, useremail)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        # Check if already verified
        if user.is_verified:
            return {
                "message": "Email already verified",
                "status": "info"
            }
        # created_token = user.get_context_string(settings.VERIFICATION_TOKEN_SECRET)
        created_token = SecurityManager.create_verification_token(user)
        # Decode and validate token

        try:
            token_valid = SecurityManager.verify_token(created_token, token)
        except Exception as verify_exec:
            token_valid = False

        if not token_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This link either expired or no more valid")
        # Update user verification status
        from app.services.user import verify_user_email
        user = await verify_user_email(db, useremail)
        
        stripeId = await payment.create_customer(user)
        if not stripeId:
            raise HTTPException(status_code=400, detail="Couldn't create stripe id for customer")
        user.stripeId = stripeId
        
        db.add(user)  # Ensure the user object is added to the session
        await db.commit()
        await db.refresh(user)
        return {
            "message": "Email verified successfully",
            "status": "success"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    

async def reset_password(token: str, useremail: str, new_password: str, db: AsyncSession):
    """
    Reset the password for a user after validating the provided token.
    """
    try:
        # Retrieve the user by email
        user = await get_user_by_email(db, useremail)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        
        # Generate the token based on the current user data (adjust as needed)
        created_token = SecurityManager.create_verification_token(user)
        
        try:
            token_valid = SecurityManager.verify_token(created_token, token)
        except Exception:
            token_valid = False

        if not token_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
        
        # Hash the new password and update the user's password field
        hashed_password = SecurityManager().get_password_hash(new_password)
        user.hashed_password = hashed_password
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        return {"message": "Password reset successfully", "status": "success"}
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Password reset failed: {str(e)}")


async def change_user_password(user: User, current_password: str, new_password: str, db: AsyncSession):
    """
    Change the password for a user after verifying the current password.
    """
    # Verify the current password
    if not SecurityManager().verify_password(current_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Hash the new password
    hashed_password = SecurityManager().get_password_hash(new_password)
    
    # Update user password
    user.hashed_password = hashed_password
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return {"message": "Password changed successfully", "status": "success"}

# async def send_password_reset_invitation(
#     email: EmailStr, 
#     name: str, 
#     token: str,
#     organization_name: str
# ):
#     """
#     Send password reset invitation email to newly created users
    
#     Args:
#         email: User's email address
#         name: User's name
#         token: Password reset token
#         organization_name: Name of the organization
#     """
#     try:
#         # Create password reset link
#         reset_url = f"http://{settings.FRONTEND_HOST}/reset-password?token={token}"
        
#         # Email template
#         html = f"""
#             <h3>Welcome to {organization_name}, {name}!</h3>
#             <p>Your account has been created. To access the platform, please set your password by clicking the link below:</p>
#             <p>
#                 <a href="{reset_url}">
#                     Set Your Password
#                 </a>
#             </p>
#             <p>This link will expire in 24 hours.</p>
#             <p>If you have any issues, please reach out to support.</p>
#             <p>Thank you,<br>The {organization_name} Team</p>
#         """
        
#         # Create message
#         message = MessageSchema(
#             subject=f"Welcome to {organization_name} - Set Your Password",
#             recipients=[email],
#             body=html,
#             subtype="html"
#         )
        
#         # Send email
#         fm = FastMail(conf)
#         await fm.send_message(message)
#         print(f"Password reset invitation email sent to {email}")
        
#     except Exception as e:
#         print(f"Error sending invitation email: {e}")
#         raise


# async def send_bulk_password_reset_invitations(
#     users: List[dict], 
#     organization_name: str,
#     background_tasks: BackgroundTasks
# ):
#     """
#     Send password reset invitations to multiple users
    
#     Args:
#         users: List of user dictionaries with email, name, and id
#         organization_name: Name of the organization
#         background_tasks: FastAPI background tasks
#     """
#     for user in users:
#         token = SecurityManager.create_verification_token(user["user"])
        
#         # Add email sending task to background
#         background_tasks.add_task(
#             send_password_reset_invitation,
#             email=user["email"],
#             name=user["name"],
#             token=token,
#             organization_name=organization_name
#         )
    
#     return len(users)

