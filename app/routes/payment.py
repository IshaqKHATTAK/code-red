from stripe import StripeError, stripe
import logging
from fastapi import HTTPException, APIRouter, Depends
from app.common.env_config import get_envs_setting
from app.services.auth import get_current_user
from app.models.user import User
from app.common.database_config import get_async_db
from sqlalchemy.orm import Session
from app.services import payment
from app.models.user import Plan
from app.services.payment import adjust_tier_degration_change

logger = logging.getLogger(__name__)
envs = get_envs_setting()

# This is your test secret API key.
# stripe.api_key = envs.STRIPE_SECRET_KEY

payments_router_protected = APIRouter(
    prefix="/api/v1/payment",
    tags=["Payment"],
    responses={404: {"description": "Not found, something wrong with auth"}},
    # dependencies=[Depends(oauth2_scheme), Depends(validate_access_token),]
)


# @payments_router_protected.post("/create-portal-session")
# async def customer_portal(user: User = Depends(get_current_user)):
#     try:
#         return_url = envs.FRONTEND_HOST

#         portalSession = await stripe.billing_portal.Session.create_async(
#             customer=user.stripeId,
#             return_url=return_url,
#         )
#         return portalSession.url
#     except Exception as e:
#         logger.exception(e)
#         raise HTTPException(status_code=500, detail="Server error")

 
@payments_router_protected.get("/plans")
async def get_prices(session: Session = Depends(get_async_db), current_user: User = Depends(get_current_user)):
    try:
        prices = await stripe.Price.list_async(active=True, expand=['data.product'])
        filtered_plans = [
            {
                "name": price.product.name,
                "price_id": price.id,
                "active": price.active,
                "currency": price.currency,
                "description": price.product.get("description", "") if price.product else "",
                "metadata": price.product.get("metadata", {}) if price.product else {},
                "price":price.unit_amount//100,
            }
            for price in reversed(prices.data)
        ]
        user_plans = [
            {
                "current_plan": current_user.current_plan.value,  
            }
        ]
        return {"plans": filtered_plans,"user_plans": user_plans}
    except StripeError as e:
        logger.exception(e)
        raise HTTPException(status_code=400, detail=str(e))
    
@payments_router_protected.get("/update-plans")
async def create_payment_session(priceId: str, user: User = Depends(get_current_user)):
    print('Stripe ID', user.stripeId)
    try:
        if not user.stripeId:
            raise HTTPException(status_code=400, detail="Stripe ID not found")
        
        # ✅ Check if the user already has an active subscription
        subscriptions = await stripe.Subscription.list_async(customer=user.stripeId, status="active")
        print(f'subscriptions == {subscriptions}')
        
        if subscriptions.data:
            subscription = subscriptions.data[0]  # Assuming one active subscription per user

            # ✅ Create Stripe Checkout Session to show available plans directly
            session = await stripe.checkout.Session.create_async(
                customer=user.stripeId,
                mode="subscription",
                line_items=[
                    {"price": "price_1QpTIlHjOc8rdeEV6TXpVyyn", "quantity": 1},  # Replace with your real Stripe price IDs
                    {"price": "price_1QpTGoHjOc8rdeEVasycCZpd", "quantity": 1},
                    {"price": "price_1QpTDjHjOc8rdeEVrqJ4VE9z", "quantity": 1},
                ],
                subscription_data={
                    "trial_from_plan": False,  # Keeps the existing subscription structure
                    "metadata": {"subscription_id": subscription.id},  # Store subscription ID
                },
                success_url=f"{envs.FRONTEND_HOST}/payment/success",
                cancel_url=f"{envs.FRONTEND_HOST}/payment/failure",
            )

            return {"checkout_url": session.url}  # ✅ Redirect user directly to plan selection
        return
    except HTTPException as http_exc:
        # This will handle our custom raised HTTPExceptions
        logger.exception(http_exc)
        raise http_exc    
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Server error")
        


@payments_router_protected.get("/create-payment-session")
async def create_payment_session(priceId: str, user: User = Depends(get_current_user)):
    print('Stripe ID', user.stripeId)
    try:
        if not user.stripeId:
            raise HTTPException(status_code=400, detail="Stripe ID not found")
        
        subscriptions = await stripe.Subscription.list_async(customer=user.stripeId, status="active")
        print(f'subscriptions == {subscriptions}')
        if subscriptions.data:
            session = await stripe.billing_portal.Session.create_async(
                customer=user.stripeId,
                return_url=f"{envs.FRONTEND_HOST}/"  # User will return here after managing their subscription
            )
            return {"checkout_url": session.url}
 
        client_secret = await payment.create_checkout_session(user.id, user.stripeId, priceId)
        
        return {"checkout_url": client_secret.url} 
    except HTTPException as http_exc:
        # This will handle our custom raised HTTPExceptions
        logger.exception(http_exc)
        raise http_exc    
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Server error")

@payments_router_protected.post("/cancel-subscription")
async def cancel_subscription(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_async_db),
    # cancel_immediately: bool = False
):
    try:
        if not user.stripeId:
            raise HTTPException(status_code=400, detail="User has no Stripe ID.")

        # 🔍 Find the active subscription
        subscriptions = await stripe.Subscription.list_async(customer=user.stripeId, status="active")
        if not subscriptions.data:
            raise HTTPException(status_code=400, detail="No active subscription found.")
        subscription = subscriptions.data[0]  # Assume one active subscription per user
        # if cancel_immediately:
        #     # ❌ Cancel the subscription immediately
        #     canceled_subscription = await stripe.Subscription.cancel_async(subscription.id)
        # else:
        canceled_subscription = await stripe.Subscription.modify_async(
                subscription.id,
                cancel_at_period_end=True
            )
        user.is_paid = False
        user.current_plan = Plan.free
        await session.commit()
        session.refresh(user)
        await adjust_tier_degration_change(session, user)
        return {
            "message": "Subscription cancellation requested.",
            "subscription_id": canceled_subscription.id,
            "cancellation_status": "at period end",
            # "cancellation_status": "immediate" if cancel_immediately else "at period end",
            "cancellation_date": canceled_subscription.current_period_end  # Timestamp of when it cancels
        }
    except HTTPException as http_exc:
        logger.exception(http_exc)
        raise http_exc    
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Server error")
