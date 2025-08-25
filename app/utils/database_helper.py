

from app.models.organization import  Organization
from app.models.chatbot_model import ChatbotDocument, UrlSweep, ChatbotConfig
from sqlalchemy.future import select
from sqlalchemy import delete
from sqlalchemy.orm import joinedload
# async def get_file_from_user_db(Organization_id, filename, session):
#     result = await session.execute(select(ChatbotConfig).filter(ChatbotConfig.id == data.id)) 
#     return result.scalars().first()

async def get_document(document_name, chatbot_id, session):
    docs = await session.execute(
            select(ChatbotDocument).filter(
                (ChatbotDocument.document_name == document_name) & 
                (ChatbotDocument.chatbot_id == chatbot_id)
            )
        )
    return docs.scalars().first()

async def delete_document(document_name: str, organization_id, session):
    docs = await session.execute(
        select(ChatbotDocument).filter(
        (ChatbotDocument.document_name == document_name) &
        (ChatbotDocument.organization_id == organization_id)
        )
    )
    document = docs.scalars().first()
    if document:
        await session.delete(document)
        await session.commit()  
        return True
    
    return False

async def update_document_status(document_name: str, new_status: str, organization_id, session):
    docs = await session.execute(
        select(ChatbotDocument).filter((ChatbotDocument.document_name == document_name) & (ChatbotDocument.organization_id == organization_id))
    )
    document = docs.scalars().first()
    
    if document:
        document.status = new_status
        await session.commit()  
        return True
    
    return False

async def insert_document_entry(
        chatbot_id: int,
        document_name: str,
        content_type: str,
        status: str,
        session
    ):
    new_document = ChatbotDocument(
        chatbot_id=chatbot_id,
        document_name=document_name,
        content_type=content_type,
        status=status
    )
    
    session.add(new_document)
    await session.commit()  
    await session.refresh(new_document)  

    return new_document

async def delete_document_entry(document_id: int, db):
    query = delete(ChatbotDocument).where(ChatbotDocument.id == document_id)
    await db.execute(query)
    await db.commit()

async def delete_webscrap_entry(website_id: int, db):
    query = delete(ChatbotDocument).where(ChatbotDocument.id == website_id)
    await db.execute(query)
    await db.commit()

async def insert_webscrap_entry(
        chatbot_id: int,
        url: str,
        sweap_domain: str,
        content_type: str,
        status: str,
        session
    ):

    if sweap_domain:
        new_document = ChatbotDocument(
            chatbot_id=chatbot_id,
            document_name=url,
            content_type=content_type,
            url_sweep_option = UrlSweep.Domain.value,
            status=status
        )
    else:
        new_document = ChatbotDocument(
            chatbot_id=chatbot_id,
            document_name=url,
            content_type=content_type,
            url_sweep_option = UrlSweep.website_page.value,
            status=status
        )
    session.add(new_document)
    await session.commit()  
    await session.refresh(new_document)  

    return new_document

async def get_webscrap_entery(id, chatbot_id, session):
    docs = await session.execute(
            select(ChatbotDocument).filter(
                (ChatbotDocument.id == id) & 
                (ChatbotDocument.chatbot_id == chatbot_id)
            )
        )
    return docs.scalars().first()

async def get_orgnization(organization_id: int, session):
    query = (
        select(Organization)
        .filter(Organization.id == organization_id)
    )
    org_result = await session.execute(query)
    organization = org_result.scalars().first()
    return organization
    

async def get_organization_and_documents(organization_id: int, session=None):
    
    query = (
        select(Organization)
        .options(joinedload(Organization.documents))
        .filter(Organization.id == organization_id)
    )
    org_result = await session.execute(query)
    organization = org_result.scalars().first()
    if not organization:
        return {"error": "Organization not found"}
    
    return organization



async def increment_org_message_count(org_id, session):
    query = (
        select(Organization)
        .filter(Organization.id == org_id)
    )
    org_result = await session.execute(query)
    organization = org_result.scalars().first()
    if not organization:
        return {"error": "Organization not found"}
    organization.total_messages_count += 2
    await session.commit()
    return organization.total_messages_count


async def increment_chatbot_message_count(chatbot_id, session):
    """
    Increment the chatbot's total_chatbot_messages count by one asynchronously.
    """
    result = await session.execute(
        select(ChatbotConfig).filter(ChatbotConfig.id == chatbot_id)
    )
    chatbot = result.scalars().first()

    if chatbot:
        chatbot.total_chatbot_messages_count += 2
        await session.commit()
        return chatbot.total_chatbot_messages_count
    
    return None


async def increment_admin_chatbot_message_count(chatbot_id, session):
    """
    Increment the chatbot's total_chatbot_messages count by one asynchronously.
    """
    result = await session.execute(
        select(ChatbotConfig).filter(ChatbotConfig.id == chatbot_id)
    )
    chatbot = result.scalars().first()

    if chatbot:
        chatbot.admin_per_days_messages_count += 2
        await session.commit()
        return chatbot.admin_per_days_messages_count
    
    return None


async def increment_chatbot_per_day_message_count(chatbot_id, session):
    """
    Increment the chatbot's total_chatbot_messages count by one asynchronously.
    """
    result = await session.execute(
        select(ChatbotConfig).filter(ChatbotConfig.id == chatbot_id)
    )
    chatbot = result.scalars().first()

    if chatbot:
        chatbot.per_day_messages = chatbot.per_day_messages or 0
        chatbot.per_day_messages += 2
        await session.commit()
        return chatbot.per_day_messages
    
    return None

from datetime import date, timedelta, datetime, timezone
from sqlalchemy.orm.attributes import flag_modified
async def increment_public_chatbot_per_day_message_count(chatbot, session):
    today = datetime.utcnow().date().isoformat() # YYYY-MM-DD format
            
    if chatbot.public_last_7_days_messages is None:
        chatbot.public_last_7_days_messages = {}

    if isinstance(chatbot.public_last_7_days_messages, str):
        import json
        chatbot.public_last_7_days_messages = json.loads(chatbot.public_last_7_days_messages)

    print(f"Before update: {chatbot.public_last_7_days_messages}")

    # Increment today's count
    chatbot.public_last_7_days_messages[today] = chatbot.public_last_7_days_messages.get(today, 0) + 2

    # Keep only the last 7 days
    seven_days_ago = (date.today() - timedelta(days=6)).isoformat()
    chatbot.public_last_7_days_messages = {k: v for k, v in chatbot.public_last_7_days_messages.items() if k >= seven_days_ago}

    # Mark JSON as modified
    flag_modified(chatbot, "public_last_7_days_messages")

    print(f"Before commit: {chatbot.public_last_7_days_messages}")

    # 🔥 Commit and refresh
    session.add(chatbot) 
    await session.commit()

    print(f"After commit: {chatbot.public_last_7_days_messages}")

    return chatbot.public_last_7_days_messages


async def get_last_seven_days_count(chatbot_id, session):
    """
    Retrieve the last 7 days' message counts for a chatbot.
    """
    result = await session.execute(
        select(ChatbotConfig.public_last_7_days_messages)
        .where(ChatbotConfig.id == chatbot_id)
    )
    row = result.scalar()
    
    return row if row else {}






from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.common.database_config import get_async_db
scheduler = AsyncIOScheduler()

async def reset_chatbot_message_counts():
    """
    Resets the admin message count and total chatbot messages count for all chatbots daily.
    """
    print(f"scheduler triggtered")
    async for session in get_async_db():    # Use async session for DB operations
        try:
            result = await session.execute(select(ChatbotConfig))
            chatbots = result.scalars().all()

            for chatbot in chatbots:
                chatbot.admin_per_days_messages_count = 0
                chatbot.total_chatbot_messages_count = 0  
            
            await session.commit()
            print(f"Chatbot message counts reset at {datetime.now()}.")
        except Exception as e:
            print(f"Error resetting chatbot messages: {e}")
        finally:
            await session.close()  
# Schedule the reset function to run daily at 00:00 UTC
# scheduler.add_job(reset_chatbot_message_counts, "cron", hour=0, minute=0)

#monthly scheduler
scheduler.add_job(reset_chatbot_message_counts, "cron", day=1, hour=0, minute=0)
async def start_scheduler():
    scheduler.start()
    print("APScheduler started...")