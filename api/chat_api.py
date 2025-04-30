# Add this to your process_pdf.py file or create a new file called chat_api.py
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import requests
import json
from dotenv import load_dotenv
import random

# Create API router
router = APIRouter()

# Models for chat request and response
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    pdf_content: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model: str

# OpenAI API configuration
# Load API key
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Hugging Face API configuration
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "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")

# API endpoints for different models
MODEL_ENDPOINTS = {
    "gpt-4.5": "https://api.openai.com/v1/chat/completions",
    "openai-o3": "https://api.openai.com/v1/chat/completions",
    "openai-o3-mini": "https://api.openai.com/v1/chat/completions",
    "llama-4": "https://api-inference.huggingface.co/models/meta-llama/Llama-3-70b-chat-hf",
    "mistral-8x7b": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek": "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-33b-instruct"
}

# OpenAI model mapping
OPENAI_MODELS = {
    "gpt-4.5": "gpt-4",  # Use gpt-4 as a stand-in for gpt-4.5
    "openai-o3": "gpt-3.5-turbo",  # Use gpt-3.5-turbo as a stand-in for o3
    "openai-o3-mini": "gpt-3.5-turbo"  # Use gpt-3.5-turbo as a stand-in for o3-mini
}

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat API endpoint that routes to the appropriate model provider.
    """
    model_id = request.model
    messages = request.messages
    pdf_content = request.pdf_content
    
    try:
        # Route to the appropriate model provider
        if model_id in ["gpt-4.5", "openai-o3", "openai-o3-mini"]:
            response = await generate_openai_response(model_id, messages, pdf_content)
        else:
            response = await generate_huggingface_response(model_id, messages, pdf_content)
        
        return {"response": response, "model": model_id}
    
    except Exception as e:
        print(f"Error generating chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

async def generate_openai_response(model_id: str, messages: List[Message], pdf_content: Optional[str] = None):
    """
    Generate a response using OpenAI API.
    """
    openai_model = OPENAI_MODELS.get(model_id, "gpt-4")
    
    # Construct the system message with PDF content if available
    system_content = "You are a helpful AI assistant."
    
    if pdf_content:
        system_content += " I'll provide you with content from a PDF document. Use this content to answer the user's questions accurately. If the answer isn't in the PDF content, tell the user that the information is not in the document."
        system_content += f"\n\nPDF CONTENT:\n{pdf_content[:10000]}"  # Limit to first 10K chars
    
    # Convert messages to OpenAI format
    openai_messages = [{"role": "system", "content": system_content}]
    
    for msg in messages:
        if msg.role in ["user", "assistant"]:
            openai_messages.append({"role": msg.role, "content": msg.content})
    
    # Call OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": openai_model,
        "messages": openai_messages,
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    response = requests.post(
        MODEL_ENDPOINTS[model_id],
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    result = response.json()
    assistant_response = result["choices"][0]["message"]["content"]
    
    return assistant_response

async def generate_huggingface_response(model_id: str, messages: List[Message], pdf_content: Optional[str] = None):
    """
    Generate a response using Hugging Face Inference API.
    """
    # Format system and context prompt
    system_prompt = "You are a helpful AI assistant."
    
    if pdf_content:
        system_prompt += " I'll provide you with content from a PDF document. Use this content to answer the user's questions accurately. If the answer isn't in the PDF content, tell the user that the information is not in the document."
    
    # Create conversation history
    conversation = []
    
    for msg in messages:
        if msg.role == "user":
            conversation.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            conversation.append({"role": "assistant", "content": msg.content})
    
    # Format prompt based on the model
    prompt = ""
    
    if model_id == "mistral-8x7b":
        # Mixtral format
        prompt = "<s>[INST] "
        if pdf_content:
            prompt += f"{system_prompt}\n\nPDF CONTENT:\n{pdf_content[:10000]}\n\n"
        
        for i, msg in enumerate(conversation):
            if i == len(conversation) - 1:  # Last message
                prompt += f"{msg['content']} [/INST]"
            elif msg["role"] == "user":
                prompt += f"{msg['content']} [/INST]"
            else:  # assistant
                prompt += f" {msg['content']} </s><s>[INST] "
    
    elif model_id == "llama-4":
        # Llama format
        prompt = "<|system|>\n" + system_prompt + "\n"
        
        if pdf_content:
            prompt += f"\nPDF CONTENT:\n{pdf_content[:10000]}\n"
        
        for msg in conversation:
            if msg["role"] == "user":
                prompt += f"\n<|user|>\n{msg['content']}"
            else:
                prompt += f"\n<|assistant|>\n{msg['content']}"
        
        prompt += "\n<|assistant|>\n"
    
    else:  # deepseek and others
        # Generic format
        prompt = f"### System:\n{system_prompt}\n\n"
        
        if pdf_content:
            prompt += f"### PDF Content:\n{pdf_content[:10000]}\n\n"
        
        for msg in conversation:
            if msg["role"] == "user":
                prompt += f"### Human: {msg['content']}\n\n"
            else:
                prompt += f"### Assistant: {msg['content']}\n\n"
        
        prompt += "### Assistant:"
    
    # Call Hugging Face API
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 800,
            "return_full_text": False
        }
    }
    
    response = requests.post(
        MODEL_ENDPOINTS[model_id],
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
    
    result = response.json()
    
    # Extract the generated text
    if isinstance(result, list) and len(result) > 0:
        if "generated_text" in result[0]:
            return result[0]["generated_text"]
    
    # Fallback for other response formats
    return str(result)

# Handle case where model API is unavailable (e.g., rate limits)
@router.post("/api/chat/fallback", response_model=ChatResponse)
async def chat_fallback(request: ChatRequest):
    """
    Fallback chat endpoint that provides mock responses when model APIs are unavailable.
    """
    model_id = request.model
    messages = request.messages
    
    # Get the last user message
    user_message = ""
    for msg in reversed(messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    # Generate a mock response
    mock_responses = [
        f"I've analyzed your question about '{user_message}'. Based on the PDF content, I found that...",
        f"The PDF document contains information related to your question. According to the document...",
        f"I couldn't find specific information about '{user_message}' in the PDF. However, the document does mention...",
        f"That's an interesting question. The PDF content suggests that...",
        "I don't see information about this specific topic in the PDF. Would you like me to search for something else?"
    ]
    
    response = random.choice(mock_responses)
    
    return {"response": response, "model": model_id}