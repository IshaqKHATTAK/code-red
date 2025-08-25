from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class WebscrapUrl(BaseModel):
    website_link: str
    website_url_id: int
    sweep_domain: bool
    sweep_url: bool
    status: str

class DocumentInfo(BaseModel):
    document_name: str
    document_id: int
    document_status: Optional[str] = None

class QATemplateData(BaseModel):
    question: str
    answer: str
    status: Optional[str] = None
    title: Optional[str] = None
    id:int

class UpdateQATemplateData(BaseModel):
    question: str
    answer: str
    title: Optional[str] = None
    id: Optional[int] = None

class ChatbotMemory(BaseModel):
    text: str
    memory_id: Optional[int] = None

class ChatbotSuggestion(BaseModel):
    suggestion_text: str

class ChatbotSuggestionUpdate(BaseModel):
    suggestion_text: str
    suggestion_id: Optional[int] = None

class ChatbotMemoryResponse(BaseModel):
    text: str
    creator: str
    memory_id: int
    status: str

class GetChatbotMemoryResponse(BaseModel):
    chatbot_data: list[ChatbotMemoryResponse]
    bot_memory_status: bool

class ChatbotSuggestionResponse(BaseModel):
    suggestion_id: int
    suggestion_text: str

class GetGuardrails(BaseModel):
    guardrail: str
    id: int

class ChatbotFileUpdateResponse(BaseModel):
    website_url: Optional[List[WebscrapUrl]] = []
    bot_documents: Optional[List[DocumentInfo]] = []


class Supervisors(BaseModel):
    name: str
    supervisor_id: int

class PromptData(BaseModel):
    title: Optional[str] = ''
    text: Optional[str] = ''

class ChatbotConfigResponse(BaseModel):
    id: int
    # llm_model_name: str
    llm_temperature: float
    llm_prompt: Optional[List[PromptData]] = []
    llm_role: str
    status: Optional[str] = None
    llm_streaming: bool = True
    chatbot_type: str
    chatbot_name: str
    website_url: Optional[List[WebscrapUrl]] = []
    bot_documents: Optional[List[DocumentInfo]] = []
    qa_templates: Optional[List[QATemplateData]] = []
    guardrails: Optional[List[GetGuardrails]] = []
    supervisors: Optional[List[Supervisors]] = []
    avatar: Optional[str] = None
    class Config:
        from_attributes = True

class WebsiteUrl(BaseModel):
    website_link: str
    sweep_domain: bool
    sweep_url: bool
    website_url_id: Optional[int] = None
    
class WebsiteRemoved(BaseModel):
    website_link: str
    website_url_id: int

class DocumentRemoved(BaseModel):
    document_name: str
    document_id: int



class ChatbotPrompt(BaseModel):
    trianing_text: Optional[List[PromptData]] = None

class DetailsRequest(BaseModel):
    chatbot_name: str
    chatbot_role: str
    avatar: Optional[str] = None
from enum import Enum

class LlmModelEnum(str, Enum):
    gpt_40 = "gpt-4o"
    gpt_40_mini = "gpt-4o-mini"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"

class ImageGenerationLlmModelEnum(str, Enum):
    dall_e_2 = "dall-e-2"
    dall_e_3 = "dall-e-3"

class BotLlmRequest(BaseModel):
    model_name: Optional[LlmModelEnum] = None

class ImageGenerationBotLlmRequest(BaseModel):
    model_name: Optional[ImageGenerationLlmModelEnum] = None

class AppModels(BaseModel):
    chat_model_name: Optional[BotLlmRequest] = None
    image_model_name: Optional[ImageGenerationBotLlmRequest] = None

class ChatBotCreation(BaseModel):
    id: int
    chatbot_type: str
    chatbot_name: str
    avatar: str
    # llm_model_name: str
    
class AllChatbotResponse(BaseModel):
    org_avatar: Optional[str] = None
    chatbots: List[ChatBotCreation]



class Guardrails(BaseModel):
    guardrails: str