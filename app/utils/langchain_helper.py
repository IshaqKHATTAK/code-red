from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from sqlalchemy.exc import SQLAlchemyError
import backoff
import logging
from sqlalchemy import func
from fastapi import HTTPException,status
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import ToolMessage
from app.models.chatbot_model import Threads, Messages, ChatbotDocument
from sqlalchemy import select, asc, desc
from sqlalchemy.orm import Session
from langchain.tools import Tool
from sqlalchemy.future import select
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Dict
from sqlalchemy import select
import openai
from typing import Optional
from app.models.user import User
from app.models.organization import Organization
from pinecone import Pinecone as PineconeClient
from app.common.env_config import get_envs_setting
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from typing import Annotated
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser



envs = get_envs_setting()
from langchain.globals import set_verbose
set_verbose(True)
logger = logging.getLogger(__name__)

def _simple_prompt_assistant(llm, system_message: str):
  prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message,
            ),
            MessagesPlaceholder(variable_name="user_input"),
        ]
    )
  return prompt | llm

def load_llm(api_key, name, temperature = 0.2):
    '''Load model and return'''
    return ChatOpenAI(model = name, api_key = api_key, temperature = temperature)

def load_llm_in_json_mode(api_key, name, temperature = 0.2):
    '''Load model and return'''
    return ChatOpenAI(model = name, api_key = api_key, temperature = temperature, model_kwargs={ "response_format": { "type": "json_object" } })

async def format_user_question(user_input, images_urls):
    question = {
                "role": "human",
                "content": [
                    {
                        "type": "text",
                        "text": user_input
                    }
                ]
            }
    if images_urls:
        for image_url in images_urls:
            image_url_formatted = {
                "type": "image_url",
                "image_url": {
                    "url": image_url, 
                }
            }
            question["content"].append(image_url_formatted)

    return question
# SETUP KNOWLEDGE BASE CHAIN

class ModerationResponseFormat(BaseModel):
    response: str = Field(description="Response either ALLOWED or NOT_ALLOWED")
        
async def _create_moderation_chain(llm,  guardrails, user_input):
    """
    Creates a moderation chain using guardrails from database
    
    Args:
        llm: The language model to use
        chatbot_id: ID of the chatbot to get guardrails for
        db_session: Database session
    """
 # Default if no guardrails found
    try:
        MODERATION_PROMPT = f"""
        You are a content moderator. You must respond with ONLY 'NOT_ALLOWED' if the input mentions or asks about any of these forbidden topics:

        FORBIDDEN TOPICS:
        {guardrails}
        
        For all other topics, respond with 'ALLOWED'.
        Remember: Your response must be ONLY 'ALLOWED' or 'NOT_ALLOWED' - no other text.

        ###Output format:
        {{{{"response":"ALLOWED"}}}}
        {{{{"response":"NOT_ALLOWED"}}}}
        YOU MUST FOLLOW THE EXACT JSON FORMAT
        """
        
        moderation_prompt = ChatPromptTemplate.from_messages([
            ("system", MODERATION_PROMPT),
            ("human", "{input}")
        ])
        moderation_chain = moderation_prompt | llm.with_structured_output(ModerationResponseFormat, strict = True)
        moderation_result = await moderation_chain.ainvoke({"input": user_input})
        print(f'meration results == {moderation_result.response}')
        if moderation_result.response.strip() == "ALLOWED":
            return True,''
        else:
            return False, "I apologize, but I'm not permitted to discuss this topic. Please feel free to ask me something else that aligns with our usage policies."
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request."
        )    

    # return moderation_prompt | llm | StrOutputParser()

def format_docs(docs):
    """Format documents into a single string with clear separation"""
    formatted_texts = []
    for match in docs:
        text = match['metadata'].get('text') or match['metadata'].get('summary', '')
        if text:
            formatted_texts.append(text)
    return "\n\n".join(formatted_texts)

async def get_relevant_context(query, embeddings, index, chatbot_id, org_id):
    # Get query embedding and search
    formatted_texts = []
    image_links = []

    # query_embedding = embeddings.embed_query(query)
    raw_results = index.query(
        vector=query,
        namespace=f'{org_id}-kb',
        top_k=4,
        include_metadata=True,
        filter={
            "content_source": "doc",
            "chatbot_id":chatbot_id
        }
    )
    # Filter results where metadata["chatbot_id"] matches the given chatbot_id
    filtered_results = [
        doc for doc in raw_results["matches"]
    ]
    for match in filtered_results:
        metadata = match.get('metadata', {})
        if metadata.get("content_type") == "image":
            print(f'image detected and added {metadata["file_path"]}')
            image_links.append(metadata["file_path"])
        else:
            print(f'No image detected == {metadata["chunk_num"]}')
            text = metadata.get('text') or metadata.get('summary', '')
            if text:
                formatted_texts.append(text)
    print(f"data in doc retrival == {formatted_texts}")
    return {
        "text": "\n\n".join(formatted_texts) if formatted_texts else None,
        "images": image_links if image_links else "No revelent image"
    }
    
async def get_relevant_scrapped_context(query, embeddings, index, chatbot_id, org_id):
    # Get query embedding and search
    # query_embedding = embeddings.embed_query(query)
    raw_results = index.query(
        vector=query,
        namespace=f'{org_id}-kb',
        top_k=4,
        include_metadata=True,
        filter={
            "content_source": "url",
            "chatbot_id":chatbot_id
        }
    )
    filtered_results = [
        doc for doc in raw_results["matches"]
    ]
    print(f"data in scrapped retrival == {filtered_results}")
    return filtered_results

async def get_relevant_prompt_context(query, embeddings, index, chatbot_id, org_id):
    # Get query embedding and search
    # query_embedding = embeddings.embed_query(query)
    raw_results = index.query(
        vector=query,
        namespace=f'{org_id}-kb',
        top_k=4,
        include_metadata=True,
        filter={
            "content_source": "prompt",
            "chatbot_id":chatbot_id
        }
    )
    filtered_results = [
        doc for doc in raw_results["matches"]
    ]
    print(f"data in scrapped prompt == {filtered_results}")
    return filtered_results

async def get_relevant_qa_context(query, embeddings, index, chatbot_id, org_id):
    # Get query embedding and search
    # query_embedding = embeddings.embed_query(query)
    raw_results = index.query(
        vector=query,
        namespace=f'{org_id}-kb',
        top_k=4,
        include_metadata=True,
        filter={
            "content_source": "qa_pair",
            "chatbot_id":chatbot_id
        }
    )
    
    filtered_results = [
        doc for doc in raw_results["matches"]
    ]
    return filtered_results

async def get_relevant_memory_context(query, embeddings, index, chatbot_id, org_id, memory_status):
    # Get query embedding and search
    # query_embedding = embeddings.embed_query(query)
    if not memory_status:
        return []
    raw_results = index.query(
        vector=query,
        namespace=f'{org_id}-kb',
        top_k=4,
        include_metadata=True,
        filter={
            "content_source": "memory",
            "chatbot_id":chatbot_id
        }
    )
    filtered_results = [
        doc for doc in raw_results["matches"]
    ]
    print(f"data in memory retrival == {filtered_results}")
    return filtered_results

#Create the complete chain with moderation

# async def moderated_chain(moderation_chain, user_input) -> str:
#     try:
#         # Check moderation
#         moderation_result = await moderation_chain.ainvoke({"input": user_input})
        
#         if moderation_result.strip() != "ALLOWED":
#             return "I apologize, but I'm not permitted to discuss this topic. Please feel free to ask me something else that aligns with our usage policies."
        
#         # Process through main chain with config
#         return await main_chain.ainvoke(
#             {"input": input, "chat_history": []},
#             config=config
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing request: {str(e)}"
#         )



def _create_custom_parser(schema_model):
    return JsonOutputParser(pydantic_object=schema_model) 



# async def construct_kb_chain(LLM_ROLE, LLM_PROMPT, llm, guardrails, chatbot_id: int, user_id: int, db_session):
async def construct_kb_chain(LLM_ROLE, memory_status, llm, chatbot_id: int, user_id: int, org_id: int, db_session, enable_image_generation=False, user_input = None):
    """
    Construct knowledge base chain with organization-specific namespace
    """
    try:
        # Initialize components
        print(f'embdings model == {envs.EMBEDDINGS_MODEL}')
        embeddings = OpenAIEmbeddings(api_key=envs.OPENAI_API_KEY, model=envs.EMBEDDINGS_MODEL)
        pc = PineconeClient(api_key=envs.PINECONE_API_KEY)
        index = pc.Index(envs.PINECONE_KNOWLEDGE_BASE_INDEX)
        
        # Setup chain components with namespace
        config = RunnableConfig(
            tags=["kb-retrieval"],
            metadata={"namespace": chatbot_id},
            recursion_limit=25,
            max_concurrency=5
        )
        image_instruction = (
            "- **If the user asks for an image, you MUST call the `generate_image` tool instead of responding with text.**\n"
            "- If unsure, assume the user wants an image if they mention words like 'create', 'draw', or 'generate'.\n"
            "YOU MUST CALL **generate_image** TOOL WHEN IMAGE NEED TO BE GENERATED.\n"
        ) if enable_image_generation else ""

        # Define structured output format using Pydantic
        class StructuredFormat(BaseModel):
            answer: str = Field(description="Response to user question/input")
            image_url: Optional[str] = Field(description="Generated image URL if it exists, otherwise null")
        query_embedding = embeddings.embed_query(user_input)
        # parser = PydanticOutputParser(pydantic_object=StructuredFormat)
        building_stories_parser = _create_custom_parser(schema_model=StructuredFormat)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a {LLM_ROLE} assistant. Your primary objective is to generate precise, contextually relevant, and well-structured responses. 
            Prioritize the provided data sources, stick to the exact provided content, prioritize the following references, ensuring that key details are incorporated before relying on general knowledge. 

            **Key Instructions:**
            - Respond in the same language as the user's input. Default to English if uncertain.
                {image_instruction}
                - You MUST describe only the provided images.   
                - DO NOT generate a random image description.    
                
            
            **Contextual References:**
            - **Primary Training Data:** {{prompt_context}}
            - **File Extracted Data:** {{context}}
            - **Web Scraped Insights:** {{scrapped_context}}
            - **Example Response Format Reference :** {{qa_context}}
                    - This provides response format and style guidelines.  
                    - Use it as a guiding framework for structuring and formatting responses.  
            
            Given the above references, generate a response that is factually accurate, contextually aligned, and well-structured and formatted.
            Clearly indicate when information is drawn from specific contexts to enhance transparency.
            
            \n**Output Format must follow the below json schema:**  
            {{{{'answer':'Response to user question/input'
                'image_url':'Generated image URL if it exists, otherwise null'
            }}}}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """{input}""")
            ])

        async def async_format_docs(x):
            context_data = await get_relevant_context(query = query_embedding, embeddings=embeddings,index=index, chatbot_id=chatbot_id, org_id=org_id)
            return context_data

        async def async_format_scrapped(x):
            context_data = await get_relevant_scrapped_context(query = query_embedding, embeddings=embeddings,index=index, chatbot_id=chatbot_id, org_id=org_id)
            return format_docs(context_data)
        
        async def async_format_prompt(x):
            context_data = await get_relevant_prompt_context(query = query_embedding, embeddings=embeddings,index=index, chatbot_id=chatbot_id, org_id=org_id)
            return format_docs(context_data)
        
        async def async_format_qa(x):
            context_data = await get_relevant_qa_context(query = query_embedding, embeddings=embeddings,index=index, chatbot_id=chatbot_id, org_id=org_id)
            return format_docs(context_data)

        async def async_format_memory(x):
            context_data = await get_relevant_memory_context(query = query_embedding, embeddings=embeddings,index=index, chatbot_id=chatbot_id, org_id=org_id, memory_status = memory_status)
            return format_docs(context_data)
        
        tools = []
        if enable_image_generation:
            tools.append(generate_image)

        setup_and_retrieval = RunnableParallel({
            "Memory":RunnableLambda(async_format_memory),
            "prompt_context": RunnableLambda(async_format_prompt),
            "context": RunnableLambda(async_format_docs),  
            "scrapped_context": RunnableLambda(async_format_scrapped),
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
            "qa_context": RunnableLambda(async_format_qa),
        })
        
        if enable_image_generation:
            # Define function calling chain
            main_chain = setup_and_retrieval | prompt | llm.bind_tools(tools, strict = True)
            
        else:
            # Default text-based response chain
            main_chain = setup_and_retrieval | prompt | llm.with_structured_output(StructuredFormat, strict = True)
            
        # return moderated_chain
        return main_chain

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error constructing chain: {str(e)}"
        )


# This function should run when user gets created as per user there is only chat.
async def _create_thread_entery(thread_id,user_id ,db_session):
    try:    
      thread_entery = Threads(
            thread_id = str(thread_id),
            user_id = user_id,
            title = 'nothing',
            )
      db_session.add(thread_entery)
      await db_session.commit()
      return thread_entery
    except SQLAlchemyError as e:
      await db_session.rollback()  
      raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error occurred: {str(e)}"
        )
    except Exception as e:
      raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=f"An unexpected error occurred: {str(e)}"
      )


@backoff.on_exception(backoff.expo, Exception, max_time=90, jitter=backoff.random_jitter, logger=logger)
async def _add_message_database(thread_id, message_uuid,role, message, db_session, is_image = None, images_urls = None, organization_id = 0):
    try:   
        if is_image:
            message_data = Messages(
                thread_id = thread_id,
                organization_admin_id = organization_id,
                role = role,
                message_uuid = message_uuid,
                message_content = message,
                is_image = True,
                images_urls = images_urls)
            db_session.add(message_data)
        else:
            message_data = Messages(
                thread_id = thread_id,
                organization_admin_id = 1,
                message_uuid = message_uuid,
                role = role,
                message_content = message)
            db_session.add(message_data)

        db_session.add(message_data)
        await db_session.commit()
        return 
    except SQLAlchemyError as e:
        await db_session.rollback()  
        raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error occurred: {str(e)}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

async def _create_langchain_history(thread_id, db_session):
    langchain_chat_history = []
    result = await db_session.execute(select(Messages).filter(Messages.thread_id == str(thread_id)).order_by(asc(Messages.created_timestamp)) )
    for message in result.scalars().all():
        if message.role == "User":
            langchain_chat_history.append(HumanMessage(content=message.content))
        else:
            langchain_chat_history.append(AIMessage(content=message.content))
        print(message.message_content)
    
    return langchain_chat_history


import json
async def _load_last_10_messages(thread_id, db_session):
    langchain_chat_history = []
    result = await db_session.execute(select(Messages).filter(Messages.thread_id == thread_id).order_by(desc(Messages.created_timestamp)).limit(10) )
    last_10_messages = result.scalars().all()[::-1]
    for message in last_10_messages:
        if message.role == "User":
            print(f'human emssage == {message.message_content}')
            langchain_chat_history.append(HumanMessage(content=json.loads(message.message_content)))
            # if message.is_image:
            #     langchain_chat_history.append(HumanMessage(content=[ 
            #             {"type": "text", "text": message.message_content},
            #             {"type": "image_url", "image_url": {"url": message.images_urls}}
            #         ]))
            # else:
            #     langchain_chat_history.append(HumanMessage(content=message.message_content))
            
        elif message.role == "Assistant":
            langchain_chat_history.append(AIMessage(content=message.message_content))
        
            # if message.is_image:
            #     langchain_chat_history.append(AIMessage(content=[
            #         {"type": "text", "text": message.message_content},
            #         {"type": "image_url", "image_url": {"url": message.images_urls}}
            #     ]))
            # else:
        elif message.role == "Tool":
            if not message.is_image:
                print(f'tool message called')
                load_data = json.loads(message.message_content)
                additional_kwargs = load_data.get('additional_kwargs', {})  # Default to empty dict if missing
                response_metadata = load_data.get('response_metadata', {})  # Default to empty dict if missing
                tool_calls = load_data.get('tool_calls', [])  # Default to empty list if missing
                message_id = load_data.get('id', '') 
                
                langchain_chat_history.append(AIMessage(
                    content = '',
                    additional_kwargs = additional_kwargs, 
                    response_metadata = response_metadata, 
                    id = message_id, 
                    tool_calls = tool_calls))
            else:
                print(f'tool response called with ruls == {message.images_urls[0]}')
                langchain_chat_history.append(ToolMessage(content=message.message_content,  tool_call_id=message.images_urls[0]))
                
        
    return langchain_chat_history
import math
from app.schemas.response.user_chat import Chats, GetMessagesResponse, GetMessagesResponseInternal
async def _load_message_history(thread_id, db_session, skip: int, limit: int, internal = False):
    langchain_chat_history = []
    # Get total message count for pagination
    total_messages_query = await db_session.execute(
        select(func.count()).filter(Messages.thread_id == thread_id)
    )
    total_messages = total_messages_query.scalar()
    if skip > total_messages:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Messages not exist for this limit.')
    
    print(f'skip == {skip} and limit ={limit}')
    if skip == 0 and limit == 0:
        result = await db_session.execute(
            select(Messages)
            .filter(Messages.thread_id == thread_id)
            .order_by(desc(Messages.created_timestamp))  # Fetch latest messages first
            .offset(0)
            .limit(20)
        )
    else:
        result = await db_session.execute(
            select(Messages)
            .filter(Messages.thread_id == thread_id)
            .order_by(asc(Messages.created_timestamp))
            .offset(skip) 
            .limit(limit)
        )
    load_messages = result.scalars().all()
    # if skip == 0 and limit == 0:
    #     load_messages.reverse()
    chat_messages = []
    image_generation = False
    for message in load_messages:
        print(f'user role == {message.role.value}')
        if message.role == 'User':
            user_input = json.loads(message.message_content)
            print(f'user get')
            chat_entry = Chats(
            role='human',
            message= user_input[0]["text"],
            images_urls=message.images_urls if message.images_urls else None,
            message_id = message.message_uuid
            )
            chat_messages.append(chat_entry)
        elif message.role == 'Assistant':
            print(f'asssitant messsage')
            chat_entry = Chats(
                role='ai',
                message= message.message_content,
                images_urls=message.images_urls if message.images_urls else None,
                message_id = message.message_uuid
            )
            chat_messages.append(chat_entry)
        else:
            # data = json.loads(message.message_content)
            print(f'tool data')
    if load_messages: 
        last_message = load_messages[-1] 
        if last_message.role == 'Tool' or (last_message.role == "User" and last_message.is_image):  # Assuming 'Tool' is the role name
            image_generation = True
            special_message = Chats(
                role='ai',
                message="",
                images_urls=None,
                message_id=last_message.message_uuid
            )
            chat_messages.append(special_message)
    if internal:
        return GetMessagesResponseInternal(
             id = thread_id,
            image_generation = image_generation,
            chat_messages=chat_messages,
            offset = skip if skip else -1
        )
    return GetMessagesResponse(id=thread_id, chat_messages=chat_messages, image_generation = image_generation, total_messages=total_messages)


def _builtin_parser_assistant(llm: BaseChatModel, system_message: str):
    """
    A LangChain-based function that extracts user intent and generates an image description if needed.
    
    Parameters:
    - llm: The language model instance.
    - format: Output format (structured JSON).
    - system_message: System instruction for the assistant.
    
    Returns:
    - A chain that determines if an image is needed and generates an appropriate prompt.
    """ 
    system_message += '''\nOutput in JSON having the following schema:
    {{
    'intent':'yes/no'
    'image_description':'detailed description of image that user wants'
    }}'''
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"), 
            MessagesPlaceholder(variable_name="user_input"),  
        ]
    )
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_message),
    #         MessagesPlaceholder(variable_name="user_input"),  # Last messages in conversation
    #     ]
    # )

    # structured_format = {
    #     "intent": JsonOutputKey(description="Does the user want an image? (Yes/No)"),
    #     "image_description": JsonOutputKey(description="A detailed image generation prompt if intent is 'Yes', otherwise null")
    # }
    
    class structured_format(BaseModel):
        intent: str = Field(description="Does the user want an image? (Yes/No)")
        image_description: str = Field(description="A detailed image generation prompt if intent is 'Yes', otherwise null")
        
    return prompt | llm.with_structured_output(structured_format, method="json_mode")


async def intent_classifier(llm, chat_history, user_input):
    system_message = '''You are an intelligent assistant that detects if the user wants an image and generates a relevant image based on provided input/description.
    You should intelligently decide if the user wants an image based on the user input and the last two messages from the chat history.
    
    Chat History: {chat_history}'''
    
    formatted_sys_message = system_message.format(chat_history=chat_history)
    print(f'formated prompt == {formatted_sys_message}')
    # Create the assistant pipeline
    assistant_chain = _builtin_parser_assistant(llm, system_message=formatted_sys_message)

    # Ensure chat_history is properly formatted
    chat_history_messages = [{"role": "user", "content": msg} for msg in chat_history[-2:]]  # Last two messages

    results = await assistant_chain.ainvoke({
        "chat_history": chat_history_messages,  # Corrected: passing history correctly
        "user_input": [{"role": "user", "content": user_input}]
    })

    return results 
    # user_messages = [
    #     {"role": "user", "content": f"{last_message}"},
    #     {"role": "user", "content": f"{second_last_message}"}
    # ]
    # Run the assistant pipeline
    # result = assistant_chain.invoke({"user_input": user_messages})
    
    # print(f'result == {result}')
    # return result

import openai
import requests

@tool
def generate_image(image_prompt: Annotated[str, "description of an image user wants to generate"]):
    """Use this tool to generate an image."""
    try:
        print('generate imge calld')
        response =  openai.OpenAI().images.generate(
            model="dall-e-3", 
            prompt=image_prompt,
            n=1,
            size='1024x1024'
        )
        image_url = response.data[0].url
        return image_url
        # Create directory for storing images
        image_dir = "static/generated_images"
        import os
        import aiohttp
        import aiofiles
        
        os.makedirs(image_dir, exist_ok=True)

        filename = f"test.png"
        file_path = os.path.join(image_dir, filename)
        # Download image
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(img_response.content)
            
            print("Image generation completed!")
            # return file_path  # Return local file path
            return image_url
        else:
            print("Failed to download image.")
            return ""
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(image_url) as resp:
        #         if resp.status == 200:
        #             async with aiofiles.open(file_path, "wb") as f:
        #                 await f.write(await resp.read())
        #             print('image generation done')
        #             return file_path  # Return local path
        #         else:
        #             return ''
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return None
    



