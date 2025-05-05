# chat_api.py - Backend API for chat with PDF using RAG pipeline
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import time
import httpx
import requests
import re
import numpy as np
import faiss
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get(<your openai api key>)
HUGGINGFACE_API_KEY = os.environ.get(<your huggingface api key)

# Create API router
router = APIRouter()

# Model for chat request
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    pdf_content: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    model: str

# Function to track API usage for analytics
async def track_api_usage(
    model: str,
    feature: str,
    input_tokens: int,
    output_tokens: int,
    response_time: float,
    document_size: Optional[float] = None
):
    """
    Track API usage for analytics.
    """
    try:
        # Round token counts to integers to prevent API errors
        input_tokens = round(input_tokens)
        output_tokens = round(output_tokens)
        
        # Create the payload
        usage_data = {
            "model": model,
            "feature": feature,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_time": response_time,
            "document_size": document_size
        }
        
        # Make a request to the analytics API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/track-usage", 
                json=usage_data
            )
            
        if response.status_code != 200:
            print(f"Error tracking API usage: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error tracking API usage: {str(e)}")

# Function to estimate token count
def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a text string.
    This is a simplified estimation - accurate counts come from the API.
    """
    if not text:
        return 0
    
    # A rough estimate: 1 token ≈ 4 characters or 0.75 words for English text
    word_count = len(text.split())
    char_count = len(text)
    
    # Use both estimates and take the average
    word_based_estimate = word_count / 0.75
    char_based_estimate = char_count / 4
    
    return round((word_based_estimate + char_based_estimate) / 2)

# API endpoints for different models
MODEL_ENDPOINTS = {
    "gpt-4.5": "https://api.openai.com/v1/chat/completions",
    "openai-o3": "https://api.openai.com/v1/chat/completions",
    "openai-o4-mini": "https://api.openai.com/v1/chat/completions",
    "llama-4": "https://api-inference.huggingface.co/models/meta-llama/Llama-3-70b-chat-hf",
    "mistral-8x7b": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek": "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-1.3b-instruct",
}

# OpenAI model mapping
OPENAI_MODELS = {
    "gpt-4.5": "gpt-4",  # Use gpt-4 as a stand-in for gpt-4.5
    "openai-o3": "gpt-3.5-turbo",  # Use gpt-3.5-turbo as a stand-in for o3
    "openai-o4-mini": "gpt-3.5-turbo"  # Use gpt-3.5-turbo as a stand-in for o4-mini
}

# Maximum number of PDFs to keep in cache
MAX_CACHE_SIZE = 5

# ================ Text Processing Functions ================

def clean_plain_text(markdown_str: str) -> str:
    """
    Cleans markdown string to plain text for embeddings.
    """
    if not markdown_str:
        return ""
        
    # Remove markdown images
    text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_str)
    
    # Replace HTML tags like <br> with newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    
    # Remove markdown special characters
    text = re.sub(r'[#>*_\-\|`]', '', text)
    
    # Remove markdown links while preserving the text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    # Remove LaTeX math expressions
    text = re.sub(r'\$.*?\$', '', text)

    # Normalize whitespace and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()

    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Split text into chunks with specified size and overlap.
    """
    if not text:
        return []
        
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Add the current chunk to the list of chunks
            chunks.append(' '.join(current_chunk))
            
            # Keep some sentences for overlap
            overlap_size = 0
            overlap_sentences = []
            
            for s in reversed(current_chunk):
                if overlap_size + len(s) <= chunk_overlap:
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                else:
                    break
            
            # Start a new chunk with overlap
            current_chunk = overlap_sentences
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# ================ Vector Database Functions ================

# Global cache for vector databases
vector_db_cache = {}

def generate_pdf_id(pdf_content: str) -> str:
    """
    Generate a unique ID for the PDF content using hash and timestamp.
    """
    # Use first 1000 chars to create a hash (for performance)
    content_sample = pdf_content[:1000] if pdf_content else "empty_pdf"
    content_hash = hashlib.md5(content_sample.encode()).hexdigest()[:8]
    timestamp = int(time.time())
    return f"pdf_{content_hash}_{timestamp}"

def manage_cache_size():
    """
    Ensure the cache doesn't grow too large by removing oldest entries.
    """
    global vector_db_cache
    
    while len(vector_db_cache) > MAX_CACHE_SIZE:
        # Get and remove the oldest key (first item in dict)
        oldest_key = next(iter(vector_db_cache))
        print(f"Removing {oldest_key} from vector database cache")
        del vector_db_cache[oldest_key]

def get_embeddings(texts: List[str], model: str = "openai") -> List[List[float]]:
    """
    Get embeddings for the given texts using OpenAI's embedding model.
    """
    if model == "openai":
        # Use OpenAI embeddings API
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": batch,
                "model": "text-embedding-ada-002"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
            result = response.json()
            batch_embeddings = [item["embedding"] for item in result["data"]]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    else:
        # Use Hugging Face embeddings API (fallback)
        embeddings = []
        
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        for text in texts:
            payload = {
                "inputs": text,
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(
                "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
            
            embedding = response.json()
            embeddings.append(embedding)
        
        return embeddings

def create_vector_database(chunks: List[str], metadata: List[Dict], pdf_id: str) -> Dict:
    """
    Create a FAISS vector database for the chunks and return the database and related data.
    """
    # Get embeddings for the chunks
    embeddings = get_embeddings(chunks)
    
    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    return {
        "index": index,
        "chunks": chunks,
        "metadata": metadata,
        "dimension": dimension,
        "created_at": time.time()
    }

def get_or_create_vector_db(pdf_content: str, pdf_id: str = None) -> Dict:
    """
    Get an existing vector database or create a new one for the given PDF content.
    Using a unique ID ensures each PDF has its own vector database.
    """
    global vector_db_cache
    
    # Generate a unique ID for this PDF if not provided
    if pdf_id is None:
        pdf_id = generate_pdf_id(pdf_content)
    
    # Log cache status
    print(f"Vector DB Cache status: {len(vector_db_cache)} entries")
    
    # Clean cache if too large
    manage_cache_size()
    
    # Process the PDF content (always process, don't use cache)
    clean_text = clean_plain_text(pdf_content)
    
    # Chunk the text
    chunks = chunk_text(clean_text)
    
    # Create metadata for each chunk
    metadata = [{"chunk_id": i, "source": pdf_id} for i in range(len(chunks))]
    
    # Create the vector database
    print(f"Creating new vector database for {pdf_id}")
    vector_db = create_vector_database(chunks, metadata, pdf_id)
    
    # Cache the database
    vector_db_cache[pdf_id] = vector_db
    
    return vector_db

def search_vector_database(vector_db: Dict, query: str, top_k: int = 3) -> List[Dict]:
    """
    Search the vector database for chunks similar to the query.
    """
    # Get query embedding
    query_embedding = get_embeddings([query])[0]
    query_embedding_np = np.array([query_embedding]).astype('float32')
    
    # Search the index
    index = vector_db["index"]
    distances, indices = index.search(query_embedding_np, top_k)
    
    # Get the matching chunks and metadata
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(vector_db["chunks"]):
            results.append({
                "text": vector_db["chunks"][idx],
                "score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity score
                "metadata": vector_db["metadata"][idx]
            })
    
    return results

# ================ Chat API Functions ================

def get_chat_history_as_text(messages: List[ChatMessage], max_messages: int = 5) -> str:
    """
    Converts chat history to text format for context.
    """
    # Get the last N messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    # Format as text
    history_text = ""
    for msg in recent_messages:
        role_prefix = "User" if msg.role == "user" else "Assistant"
        history_text += f"{role_prefix}: {msg.content}\n\n"
    
    return history_text.strip()

def extract_last_user_query(messages: List[ChatMessage]) -> str:
    """
    Extract the last user query from the messages.
    """
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    
    return ""

# Change this section in your chat_api.py
# Inside the chat function where it processes PDF content

@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat API endpoint that routes to the appropriate model provider.
    """
    try:
        # Start timing
        start_time = time.time()
        
        # Extract data from request
        model_id = request.model
        messages = request.messages
        pdf_content = request.pdf_content
        
        # Calculate document size if available
        document_size = len(pdf_content) / 1024 if pdf_content else None
        
        # Implement RAG if PDF content is provided
        enhanced_context = ""
        
        if pdf_content and len(messages) > 0:
            try:
                # Extract the last user query
                user_query = extract_last_user_query(messages)
                
                if user_query:
                    # Generate a unique ID for this PDF to ensure each upload has its own vector DB
                    pdf_id = generate_pdf_id(pdf_content)
                    
                    # Always create a new vector database for new PDF content
                    vector_db = get_or_create_vector_db(pdf_content, pdf_id)
                    
                    # Log which PDF is being used
                    print(f"Using vector database: {pdf_id}")
                    
                    # Search for relevant chunks
                    search_results = search_vector_database(vector_db, user_query, top_k=3)
                    
                    # Extract text from search results
                    if search_results:
                        enhanced_context = "RELEVANT PDF CONTENT:\n"
                        for i, result in enumerate(search_results):
                            enhanced_context += f"[Excerpt {i+1}]: {result['text']}\n\n"
                    else:
                        enhanced_context = "No directly relevant content found in the PDF for this specific query."
            except Exception as e:
                print(f"Error in RAG processing: {str(e)}")
                # Continue with regular processing if RAG fails
                enhanced_context = ""
                
        
        # Generate the response
        if model_id in OPENAI_MODELS:  # Check if it's an OpenAI model
            result = await generate_openai_response(model_id, messages, pdf_content, enhanced_context)
            
            # For OpenAI, we may have the full response with token counts
            if isinstance(result, dict) and "usage" in result:
                input_tokens = result["usage"]["prompt_tokens"]
                output_tokens = result["usage"]["completion_tokens"]
                content = result["choices"][0]["message"]["content"]
            else:
                content = result
                # Generate input text for token estimation
                input_text = ""
                for msg in messages:
                    input_text += f"{msg.role}: {msg.content}\n"
                if pdf_content:
                    input_text += f"PDF Content: {pdf_content}\n"
                
                input_tokens = estimate_token_count(input_text)
                output_tokens = estimate_token_count(content)
        else:
            # Handle potential errors with Hugging Face API
            try:
                content = await generate_huggingface_response(model_id, messages, pdf_content, enhanced_context)
            except Exception as e:
                print(f"Error with Hugging Face model {model_id}: {str(e)}")
                # Fall back to OpenAI if Hugging Face fails
                fallback_model = "gpt-3.5-turbo"
                print(f"Falling back to OpenAI model: {fallback_model}")
                result = await generate_openai_response("openai-o3", messages, pdf_content, enhanced_context)
                if isinstance(result, dict) and "choices" in result:
                    content = result["choices"][0]["message"]["content"]
                else:
                    content = str(result)
            
            # Generate input text for token estimation
            input_text = ""
            for msg in messages:
                input_text += f"{msg.role}: {msg.content}\n"
            if pdf_content:
                input_text += f"PDF Content: {pdf_content}\n"
            
            input_tokens = estimate_token_count(input_text)
            output_tokens = estimate_token_count(content)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Track API usage for analytics
        await track_api_usage(
            model=model_id,
            feature="chat",
            input_tokens=round(input_tokens),  # Ensure integer
            output_tokens=round(output_tokens),  # Ensure integer
            response_time=response_time,
            document_size=document_size
        )
        
        return {"response": content, "model": model_id}
    
    except Exception as e:
        print(f"Error generating chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

async def generate_openai_response(
    model_id: str, 
    messages: List[ChatMessage], 
    pdf_content: Optional[str] = None,
    enhanced_context: Optional[str] = None
):
    """
    Generate a response using OpenAI API with RAG enhancement.
    """
    openai_model = OPENAI_MODELS.get(model_id, "gpt-4")
    
    # Construct the system message with PDF content if available
    system_content = "You are a helpful AI assistant specialized in analyzing PDF documents."
    system_content += " When answering questions about the document:"
    system_content += "\n- Reference specific sections from the document"
    system_content += "\n- If information isn't found in the document, clearly state it's not in the available content"
    system_content += "\n- Use the exact terminology from the document when possible"
    system_content += "\n- Focus primarily on the RELEVANT PDF CONTENT sections provided"
    
    # Add the enhanced context if available (from RAG)
    if enhanced_context:
        system_content += f"\n\n{enhanced_context}"
    # Add the general PDF content as fallback/reference
    elif pdf_content:
        system_content += f"\n\nPDF CONTENT OVERVIEW:\n{pdf_content[:4000]}..."  # Limit to first 4K chars + ellipsis
    
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
    
    try:
        response = requests.post(
            MODEL_ENDPOINTS[model_id],
            headers=headers,
            json=payload,
            timeout=30  # Add timeout to prevent hanging requests
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result
    except requests.exceptions.Timeout:
        raise Exception("OpenAI API request timed out after 30 seconds")
    except Exception as e:
        raise e

async def generate_huggingface_response(
    model_id: str, 
    messages: List[ChatMessage], 
    pdf_content: Optional[str] = None,
    enhanced_context: Optional[str] = None
):
    """
    Generate a response using Hugging Face Inference API with RAG enhancement.
    """
    # Format system and context prompt
    system_prompt = "You are a helpful AI assistant specialized in analyzing PDF documents."
    system_prompt += " When answering questions about the document:"
    system_prompt += "\n- Reference specific sections from the document"
    system_prompt += "\n- If information isn't found in the document, clearly state it's not in the available content"
    system_prompt += "\n- Use the exact terminology from the document when possible"
    system_prompt += "\n- Focus primarily on the RELEVANT PDF CONTENT sections provided"
    
    # Add the enhanced context if available (from RAG)
    if enhanced_context:
        system_prompt += f"\n\n{enhanced_context}"
    # Add the general PDF content as fallback/reference
    elif pdf_content:
        # More limited context for non-OpenAI models
        system_prompt += f"\n\nPDF CONTENT OVERVIEW:\n{pdf_content[:3000]}..."  # Limit to first 3K chars + ellipsis
    
    # Create conversation history
    conversation = []
    
    for msg in messages:
        if msg.role == "user":
            conversation.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            conversation.append({"role": "assistant", "content": msg.content})
    
    # Format prompt based on the model
    prompt = ""
    
    if model_id == "mistral-8x7b" or model_id == "mistral-7b":
        # Mixtral/Mistral format
        prompt = "<s>[INST] "
        # Add system prompt
        prompt += f"{system_prompt}\n\n"
        
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
        
        for msg in conversation:
            if msg["role"] == "user":
                prompt += f"\n<|user|>\n{msg['content']}"
            else:
                prompt += f"\n<|assistant|>\n{msg['content']}"
        
        prompt += "\n<|assistant|>\n"
    
    elif model_id == "tinyllama":
        # TinyLlama format (similar to Llama)
        prompt = "<|system|>\n" + system_prompt + "\n"
        
        for msg in conversation:
            if msg["role"] == "user":
                prompt += f"\n<|user|>\n{msg['content']}"
            else:
                prompt += f"\n<|assistant|>\n{msg['content']}"
        
        prompt += "\n<|assistant|>\n"
    
    elif model_id == "gemma-2b":
        # Gemma format
        prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
        
        for msg in conversation:
            if msg["role"] == "user":
                prompt += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
            else:
                prompt += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
        
        prompt += "<start_of_turn>model\n"
    
    else:  # deepseek and others
        # Generic format
        prompt = f"### System:\n{system_prompt}\n\n"
        
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
    
    try:
        response = requests.post(
            MODEL_ENDPOINTS[model_id],
            headers=headers,
            json=payload,
            timeout=30  # Add timeout to prevent hanging requests
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
    except requests.exceptions.Timeout:
        raise Exception("Hugging Face API request timed out after 30 seconds")
    except Exception as e:
        raise e

# Handle case where model API is unavailable (e.g., rate limits)
@router.post("/api/chat/fallback", response_model=ChatResponse)
async def chat_fallback(request: ChatRequest):
    """
    Fallback chat endpoint that provides mock responses when model APIs are unavailable.
    """
    model_id = request.model
    
    # Get the last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    # Generate a mock response based on the user's query
    mock_responses = [
        f"I've analyzed your question about '{user_message}'. Based on the PDF content, I found that...",
        f"The PDF document contains information related to your question. According to the document...",
        f"I couldn't find specific information about '{user_message}' in the PDF. However, the document does mention...",
        f"That's an interesting question. The PDF content suggests that...",
        "I don't see information about this specific topic in the PDF. Would you like me to search for something else?"
    ]
    
    import random
    response = random.choice(mock_responses)
    
    return {"response": response, "model": model_id}