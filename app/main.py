from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from app.routes import landing,user, organization, chatbot_config, user_chat, user_document, payment, stripe
from app.routes import notifications
from app.common.database_config import get_db
from app.services.super_admin import create_super_admin
from sqlalchemy.orm import Session
from app.common.env_config import get_envs_setting
from fastapi import FastAPI, HTTPException
import smtplib

envs = get_envs_setting()

# origins = [
#     "*",  # Adjust the port if your React app runs on a different port
#     envs.FRONTEND_HOST,  # The default port for FastAPI, if you want to allow it
#     # Add any other origins you want to allow
# ]

def create_application():
    application = FastAPI()
    
    # Add debug route
    @application.get("/api/v1/debug")
    async def debug():
        return {"routes": [{"path": route.path, "name": route.name} for route in application.routes]}
    
    application.include_router(user.router)
    application.include_router(user.public_router)
    application.include_router(
        organization.router, 
        prefix="/api/v1/organizations",  # This was causing /organizations/organizations/
        tags=["organizations"]
    )
    application.include_router(chatbot_config.chatbot_config_routes)
    application.include_router(user_chat.chats_routes)
    application.include_router(user_chat.feedback_router)
    application.include_router(user_document.document_upload_routes)
    # application.include_router(payment.payments_router_protected)
    # application.include_router(stripe.stripe_router) 
    application.include_router(notifications.announcement_router) 
    application.include_router(landing.landing_router) 
    application.include_router(landing.faqs_router) 
    
    return application

app = create_application()
app.add_middleware(
    CORSMiddleware,
    allow_origins=envs.BACKEND_CORS_ORIGINS,  # Allows specified origins to make requests
    allow_credentials=True,  # Allows cookies to be included in cross-origin HTTP requests
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    print(f'envs == {envs.ALGORITHM}, {envs.DATABASE_URI_ASYNC}')
    print(f'users == {envs.POSTGRES_USER}, {envs.POSTGRES_HOST}, {envs.POSTGRES_PORT}, ')
    print(f'en = {envs}')
    return {"status": "ok"}


# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/create-super-admin/")
async def create_super_admin_endpoint(db: Session = Depends(get_db)):
    response = create_super_admin(db)
    print(response)
    return response

smtp_server = envs.MAIL_SERVER
port = envs.MAIL_PORT 
username = envs.MAIL_USERNAME
password = envs.MAIL_PASSWORD

@app.get("/test-email/")
def test_smtp_connection():
    """
    Test the SMTP connection and return the result.
    """
    try:
        # Connect to the SMTP server
        print(f'smtp started execution')
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()  # Start TLS encryption
            server.login(username, password)  # Authenticate
            print("SMTP connection successful!")
            return {"status": "success", "message": "SMTP connection successful!"}
    except Exception as e:
        # Log and return the error
        print(f"SMTP connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"SMTP connection failed: {e}")
    

@app.get("/sent-test/")
def test_smtp_connection():
    """
    Test the SMTP connection and send a test email.
    """
    recipient_email = "mi5315562@gmail.com"
    test_subject = "Test Email from FastAPI"
    test_body = "This is a test email to confirm the SMTP connection."

    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(username, password)
            message = f"Subject: {test_subject}\n\n{test_body}"
            server.sendmail(envs.MAIL_FROM, recipient_email, message)
            print("SMTP connection successful and test email sent!")
            return {"status": "success", "message": "SMTP connection successful and test email sent!"}
    except Exception as e:
        print(f"SMTP connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"SMTP connection failed: {e}")
