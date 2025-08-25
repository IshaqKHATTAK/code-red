from sqlalchemy import Text,Column, Integer, String, TIMESTAMP, Boolean
from app.common.database_config import Base
from datetime import datetime, UTC
from sqlalchemy.orm import relationship

class Organization(Base):
    __tablename__ = "organizations"

    #Primary Key
    id = Column(Integer, primary_key=True, index=True)

    #Basic Details
    description =Column(Text, nullable=True, default=None)
    sso_domain = Column(String)
    name = Column(String, index=True)
    logo = Column(String(1000), nullable=True)
    website_link = Column(String(1000), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, default=lambda: datetime.now(UTC))
    is_image_generation_allow = Column(Boolean, default=False, nullable=True)
    chat_model_name = Column(String, default="gpt-4o", nullable=True)
    # current_plan = Column(String(50), nullable=True)
    allowed_dept_chatbots = Column(Integer, nullable=True,  default=0)
    can_create_external_bot = Column(Boolean, default=False)
    total_messages_count = Column(Integer, nullable=True,  default=0)
    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    chatbots = relationship("ChatbotConfig", back_populates="organization", cascade="all, delete-orphan")
    def __repr__(self):
        return f"<Organization {self.name}>"
