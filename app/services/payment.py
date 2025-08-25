from app.common.env_config import get_envs_setting
import stripe
from stripe import Subscription
import logging
from fastapi import HTTPException
from app.models.user import Plan, User, UserRole
from app.models.chatbot_model import ChatbotConfig
from pydantic import BaseModel
from sqlalchemy.future import select

logger = logging.getLogger(__name__)
settings = get_envs_setting()
# This is your test secret API key.
# stripe.api_key = settings.STRIPE_SECRET_KEY

class SubscriptionStatus(BaseModel):
    plan: Plan
    status: bool

async def create_customer(user: User):
    try:
        # Create a new customer in Stripe
        stripe_customer = await stripe.Customer.create_async(
            email=user.email,
            name=user.name,
            description="Customer for {}".format(user.email),
        )
        # Save the stripe customer ID in your database for future use
        # For example: update_user_with_stripe_id(user.email, stripe_customer.id)
        
        return stripe_customer.id
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

async def delete_customer(stripe_id: str):
    try:
        stripe_customer = await stripe.Customer.delete_async(sid=stripe_id)
        return stripe_customer.id
    except stripe.error.StripeError as e:
        raise Exception(f"Stripe API Error: {e}")

async def checkSubscriptionStatus(subscriptionId): 
    print(f'subscription id inside == {subscriptionId}')
    subscription = await Subscription.retrieve_async(id=subscriptionId, expand= ['items.data.price.product'])
    if(subscription):
        status = subscription["status"]
        items = subscription["items"]["data"][0]
        plan = items["price"]["product"]["name"]  
        plan_enum = Plan.from_string(plan)
        print(f'plan enum == {plan_enum}')
        if(status == "active"):
            return SubscriptionStatus(plan=plan_enum, status=True)
        else:
            return SubscriptionStatus(plan=Plan.free, status=False)
    else:
        raise HTTPException(status_code=400, detail="Subscription was not found")

async def create_checkout_session(userId, customer_id, price_id):
    try:
        # Attempt to create a checkout session
        session = await stripe.checkout.Session.create_async(
            customer=customer_id,
            payment_method_types=['card'],  # Add or remove as per your requirement
            line_items=[{
                'price': price_id,  # Price ID passed from the frontend
                'quantity': 1,
            }],
            client_reference_id= userId,
            mode='subscription',
            success_url=f"{settings.FRONTEND_HOST}payment/success",
            cancel_url=f"{settings.FRONTEND_HOST}payment/failure",
        )
        # Check if the session is successfully created
        if not session:
            raise HTTPException(status_code=404, detail="Failed to create checkout session")
        return session
    except stripe.error.StripeError as e:
        # Handle Stripe API errors specifically
        print(f"Stripe API error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle other generic exceptions if needed
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred")

async def update_user_payment(customerId, subscriptionId, session):
    result = await session.execute(select(User).where(User.stripeId == customerId))
    user = result.scalars().first()
    # user = session.query(User).filter(User.stripeId == customerId).first()
    
    if not user:
        raise HTTPException(status_code=400, detail="The user does not exist!")
    
    subscription = await checkSubscriptionStatus(subscriptionId)
    print(f'plan itself == {subscription}')
    if subscription.status: 
        user.is_paid = True
        user.current_plan = subscription.plan
    else:
        print("⚠️ No active plan found. Marking user as Free Tier.")  
        user.current_plan = Plan.free

    session.add(user)
    await session.commit() 
    await session.refresh(user)
    return user

from sqlalchemy.ext.asyncio import AsyncSession

async def adjust_tier_degration_change(
        db: AsyncSession,
        organization_id: int,
        allowed_chatbot_count: int,
        current_user: User
):
    from app.services.organization import remove_chatbot_from_organization_service
    
    chatbots = await _get_all_org_chatbots(organization_id, db)
    excess_count = len(chatbots) - allowed_chatbot_count
    if excess_count <= 0:
        return

    chatbots_to_delete = sorted(chatbots, key=lambda x: x.id)[:excess_count]

    # Delete the excess chatbots
    for chatbot in chatbots_to_delete:
        await remove_chatbot_from_organization_service(db, organization_id, chatbot.id, current_user)
    
    return

# async def adjust_tier_degration_change(
#         db: AsyncSession,
#         current_user: User,
# ):
#     from app.services.organization import remove_user_from_organization_service, remove_chatbot_from_organization_service
    
#     chatbots = await _get_all_org_chatbots(current_user.organization_id, db)
#     chatbots_to_delete = sorted(chatbots, key=lambda x: x.id)[:-3]
#     for chatbot in chatbots_to_delete:
#         await remove_chatbot_from_organization_service(db, current_user.organization_id, chatbot.id, current_user)


    # if current_user.current_plan == Plan.free:
    #     chatbots = await _get_all_org_chatbots(current_user.organization_id, db)
    #     users = await _get_all_org_users(current_user.organization_id, db)
    #     if not current_user.is_paid:
    #         for chatbot in chatbots:
    #             await remove_chatbot_from_organization_service(db, current_user.organization_id, chatbot.id, current_user)
    #         for user in users:
    #             if user.role == UserRole.USER:
    #                 await remove_user_from_organization_service(db, current_user.organization_id, user.id, current_user)
    #     if current_user.is_paid:
    #         chatbots_to_delete = sorted(chatbots, key=lambda x: x.id)[:-3]
    #         for chatbot in chatbots_to_delete:
    #             await remove_chatbot_from_organization_service(db, current_user.organization_id, chatbot.id, current_user)
    #         users_sorted = sorted(users, key=lambda x: x.created_at, reverse=True)
    #         users_to_delete = users_sorted[1:]
    #         for user in users_to_delete:
    #             if user.role == UserRole.USER:
    #                 await remove_user_from_organization_service(db, current_user.organization_id, user.id, current_user)
            
    # elif current_user.current_plan == Plan.tier_1:
    #     chatbots = await _get_all_org_chatbots(current_user.organization_id, db)
    #     for chatbot in chatbots:
    #         await remove_chatbot_from_organization_service(db, current_user.organization_id, chatbot.id, current_user)
    #     # users = _get_all_org_users(current_user.organization_id, db)
    #     # for user in users:
    #     #     await remove_user_from_organization_service(db, current_user.organization_id, user.id, current_user)

    # elif current_user.current_plan == Plan.tier_2:
    #     chatbots = await _get_all_org_chatbots(current_user.organization_id, db)
        # if len(chatbots) > settings.TEIR3_CHATBOTS:
        #     chatbots_to_delete = sorted(chatbots, key=lambda x: x.id)[:-3]
        #     for chatbot in chatbots_to_delete:
        #         await remove_chatbot_from_organization_service(db, current_user.organization_id, chatbot.id, current_user)
    
    # return


##################################################Helper###############################
async def _get_all_org_chatbots(organization_id, db):
    chatbots = await db.execute(
        select(ChatbotConfig)
        .filter(ChatbotConfig.organization_id == organization_id)
        .filter(ChatbotConfig.chatbot_type == "Department")
    )
    return chatbots.scalars().all()

async def _get_all_org_users(organization_id, db):
    users = await db.execute(select(User).filter(User.organization_id == organization_id))
    return users.scalars().all()