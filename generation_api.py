# generation_api.py - Backend API for content generation with RAG pipeline
from fastapi import APIRouter, HTTPException, Body, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import requests
import json
import pandas as pd
import io
import tempfile
import re
from dotenv import load_dotenv
import faiss
import numpy as np
from datetime import datetime
import html
import time
import httpx

# Create API router
router = APIRouter()

# Models for generation request and response
class SingleGenerationRequest(BaseModel):
    model: str
    activity: str
    definition: str
    pdf_content: Optional[str] = None
    mode: str = "single"
    prompt: Optional[str] = None  # Field for custom prompt

class GenerationResponse(BaseModel):
    content: str
    model: str

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "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")

# API endpoints for different models
MODEL_ENDPOINTS = {
    "gpt-4.5": "https://api.openai.com/v1/chat/completions",
    "openai-o3": "https://api.openai.com/v1/chat/completions",
    "openai-o4-mini": "https://api.openai.com/v1/chat/completions",
    "llama-4": "https://api-inference.huggingface.co/models/meta-llama/Llama-3-70b-chat-hf",
    "mistral-8x7b": "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek": "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-7b-instruct-v1.5"
}

# OpenAI model mapping
OPENAI_MODELS = {
    "gpt-4.5": "gpt-4",  # Use gpt-4 as a stand-in for gpt-4.5
    "openai-o3": "gpt-3.5-turbo", 
    "openai-o4-mini": "gpt-3.5-turbo"
}

# Default prompt for RAG pipeline
DEFAULT_PROMPT = """
### Instructions:
You are an expert in World Bank Global Education and education policy analysis. Your task is to determine if the activity name and definition provided in the query align with relevant content in the given context.

### Task:
- Extract up to 3 sentences from the provided context that semantically align with the given activity name and definition.
- Start each sentence with a '*' character.
- If no relevant content exists, respond with: "NO RELEVANT CONTEXT FOUND".
- Do not generate new sentences, rephrase, summarize, or add external information.
- Do not infer meaning beyond what is explicitly stated in the context.
- Not every definition may have meaningful content; in such cases, return "NO RELEVANT CONTEXT FOUND".

### Query:
Activity Name and Definition: {query}

### Context:
{context_text}

### Response Format:
- If relevant sentences are found:
  * Sentence 1 from context
  * Sentence 2 from context
  * Sentence 3 from context (if applicable)
- If no relevant content is found:
  NO RELEVANT CONTEXT FOUND

### Strict Guidelines:
- Only extract sentences exactly as they appear in the provided context.
- Do not provide reasons, explanations, or additional commentary.
- Do not summarize, reword, or infer additional meaning beyond the explicit text.
- Ensure strict semantic alignment between the definition and the extracted sentences.
"""

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

# Global cache for vector databases
vector_db_cache = {}

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
        "dimension": dimension
    }

def get_or_create_vector_db(pdf_content: str, pdf_id: str = "current_pdf") -> Dict:
    """
    Get an existing vector database or create a new one for the given PDF content.
    """
    global vector_db_cache
    
    # Return cached database if available
    if pdf_id in vector_db_cache:
        return vector_db_cache[pdf_id]
    
    # Process the PDF content
    clean_text = clean_plain_text(pdf_content)
    
    # Chunk the text
    chunks = chunk_text(clean_text)
    
    # Create metadata for each chunk
    metadata = [{"chunk_id": i, "source": pdf_id} for i in range(len(chunks))]
    
    # Create the vector database
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

# ================ API Endpoints ================

@router.post("/api/generate", response_model=GenerationResponse)
async def generate_content(request: SingleGenerationRequest):
    """
    Generate content for a single activity using RAG pipeline.
    """
    model_id = request.model
    activity = request.activity
    definition = request.definition
    pdf_content = request.pdf_content
    custom_prompt = request.prompt
    
    try:
        # Check if PDF content is provided
        if not pdf_content:
            return {"content": "No PDF content provided. Please upload a PDF file.", "model": model_id}
        
        # Create query from activity and definition
        query = f"Activity: {activity}\nDefinition: {definition}"
        
        # Create or get vector database
        vector_db = get_or_create_vector_db(pdf_content)
        
        # Search for relevant chunks
        search_results = search_vector_database(vector_db, query)
        
        # Extract text from search results
        context_text = "\n\n".join([result["text"] for result in search_results])
        
        # Record start time for analytics
        start_time = time.time()
        
        # Generate response using the appropriate model
        if model_id in ["gpt-4.5", "openai-o3", "openai-o4-mini"]:
            content = await generate_openai_response(model_id, query, context_text, custom_prompt)
        else:
            content = await generate_huggingface_response(model_id, query, context_text, custom_prompt)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Track API usage for analytics
        # Estimate tokens
        input_tokens = estimate_token_count(query + context_text)
        output_tokens = estimate_token_count(content)
        document_size = len(pdf_content) / 1024  # Size in KB
        
        await track_api_usage(
            model=model_id,
            feature="generation",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_time=response_time,
            document_size=document_size
        )
        
        # Escape HTML special characters in activity and definition for safety
        escaped_activity = html.escape(activity)
        escaped_definition = html.escape(definition)
        
        # Process LLM content: replace newlines with <br> while preserving any HTML that might be present
        formatted_content = content.replace('\n', '<br>')
        
        # Build proper HTML Table
        html_table = f"""
        <h2 style="text-align: center;">Analysis Result</h2>
        <table border="1" style="margin-left:auto;margin-right:auto;border-collapse:collapse;width:80%;">
            <thead style="background-color:#f2f2f2;">
                <tr>
                    <th style="padding:8px;">Activity Name</th>
                    <th style="padding:8px;">Definition</th>
                    <th style="padding:8px;">Matched Content</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding:8px;text-align:left;">{escaped_activity}</td>
                    <td style="padding:8px;text-align:left;">{escaped_definition}</td>
                    <td style="padding:8px;">{formatted_content}</td>
                </tr>
            </tbody>
        </table>
        """
        
        # Return result
        return {"content": html_table, "model": model_id}
    
    except Exception as e:
        print(f"Error generating content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

# In the bulk generation function, rename the html variable to avoid conflicts
@router.post("/api/generate-bulk")
async def generate_bulk_content(
    file: UploadFile = File(...),
    model: str = Form(...),
    query_limit: int = Form(10),
    pdf_content: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None)
):
    """
    Generate content for multiple activities in bulk from an Excel file using RAG pipeline.
    """
    try:
        # Check if PDF content is provided
        if not pdf_content:
            return JSONResponse(
                status_code=400,
                content={"detail": "No PDF content provided. Please upload a PDF file."}
            )
        
        # Read the Excel file
        content = await file.read()
        
        # Determine the file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:  # xlsx or xls
            df = pd.read_excel(io.BytesIO(content))
        
        # Validate the dataframe
        if 'Activity Name' not in df.columns or 'Definition' not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"detail": "Excel file must contain 'Activity Name' and 'Definition' columns"}
            )
        
        # Limit the number of rows to process
        if query_limit > 0:
            df = df.head(query_limit)
        
        # Create or get vector database
        vector_db = get_or_create_vector_db(pdf_content)
        
        # Track overall processing metrics
        bulk_start_time = time.time()
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Process each row
        results = []
        
        for index, row in df.iterrows():
            activity = row['Activity Name']
            definition = row['Definition']
            
            try:
                # Create query from activity and definition
                query = f"Activity Name: {activity}\nDefinition: {definition}"
                
                # Search for relevant chunks
                search_results = search_vector_database(vector_db, query)
                
                # Extract text from search results
                context_text = "\n\n".join([result["text"] for result in search_results])
                
                # Record start time for this item
                item_start_time = time.time()
                
                # Generate response using the appropriate model
                if model in ["gpt-4.5", "openai-o3", "openai-o4-mini"]:
                    content = await generate_openai_response(model, query, context_text, prompt)
                else:
                    content = await generate_huggingface_response(model, query, context_text, prompt)
                
                # Calculate response time for this item
                item_response_time = time.time() - item_start_time
                
                # Estimate tokens for this item
                item_input_tokens = estimate_token_count(query + context_text)
                item_output_tokens = estimate_token_count(content)
                
                # Add to total tokens
                total_input_tokens += item_input_tokens
                total_output_tokens += item_output_tokens
                
                results.append({
                    "Activity Name": activity,
                    "Definition": definition,
                    "Matched Content": content,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "Activity Name": activity,
                    "Definition": definition,
                    "Matched Content": f"Error: {str(e)}",
                    "status": "error"
                })
        
        # Calculate overall response time
        bulk_response_time = time.time() - bulk_start_time
        
        # Track API usage for analytics
        document_size = len(pdf_content) / 1024  # Size in KB
        
        await track_api_usage(
            model=model,
            feature="generation",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            response_time=bulk_response_time,
            document_size=document_size
        )
        
        # Create a results DataFrame for easy handling
        result_df = pd.DataFrame(results)
        
        # Build HTML Table for better display - RENAMED the variable to html_table
        html_table = """
            <div style="text-align:center;">
                <h2 style="margin-bottom:20px;">Bulk Activity Matching Results</h2>
                <table border="1" style="margin:auto; border-collapse:collapse; width:90%;">
                    <thead style="background-color:#f2f2f2;">
                        <tr>
                            <th style="padding:10px;">Activity Name</th>
                            <th style="padding:10px;">Definition</th>
                            <th style="padding:10px;">Matched Content</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        for _, row in result_df.iterrows():
            # Safely escape HTML entities
            activity_value = str(row['Activity Name'])
            definition_value = str(row['Definition'])
            content_value = str(row['Matched Content']).replace("\n", "<br>")  # HTML line break
            
            html_table += f"""
                        <tr>
                            <td style="padding:10px;">{html.escape(activity_value)}</td>
                            <td style="padding:10px;">{html.escape(definition_value)}</td>
                            <td style="padding:10px;">{content_value}</td>
                        </tr>
                    """

        html_table += """
                    </tbody>
                </table>
            </div>
        """
        
        return {"content": html_table, "model": model, "results": results}
    
    except Exception as e:
        print(f"Error processing bulk generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing bulk generation: {str(e)}")

# ================ Model Interaction Functions ================

async def generate_openai_response(
    model_id: str, 
    query: str, 
    context_text: str,
    custom_prompt: Optional[str] = None
):
    """
    Generate a response using OpenAI API.
    """
    openai_model = OPENAI_MODELS.get(model_id, "gpt-4")
    
    # Use custom prompt if provided, otherwise use default
    prompt_template = custom_prompt if custom_prompt else DEFAULT_PROMPT
    
    # Replace placeholders in prompt
    system_content = prompt_template.replace("{query}", query).replace("{context_text}", context_text)
    
    # Estimate input tokens
    input_tokens = estimate_token_count(system_content)
    
    # Record start time for response time tracking
    start_time = time.time()
    
    # Call OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": system_content}
        ],
        "temperature": 0.6,
        "max_tokens": 500
    }
    
    response = requests.post(
        MODEL_ENDPOINTS[model_id],
        headers=headers,
        json=payload
    )
    
    # Calculate response time
    response_time = time.time() - start_time
    
    if response.status_code != 200:
        raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    result = response.json()
    generated_content = result["choices"][0]["message"]["content"]
    
    # Get the actual token usage from the response if available
    if "usage" in result:
        input_tokens = result["usage"]["prompt_tokens"]
        output_tokens = result["usage"]["completion_tokens"]
    else:
        # Estimate output tokens if not provided
        output_tokens = estimate_token_count(generated_content)
    
    # Track API usage for analytics
    await track_api_usage(
        model=openai_model,
        feature="generation",  # This could be "chat" or "extraction" depending on caller
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_time=response_time
    )
    
    return generated_content

async def generate_huggingface_response(
    model_id: str, 
    query: str, 
    context_text: str,
    custom_prompt: Optional[str] = None
):
    """
    Generate a response using Hugging Face Inference API.
    """
    # Use custom prompt if provided, otherwise use default
    prompt_template = custom_prompt if custom_prompt else DEFAULT_PROMPT
    
    # Replace placeholders in prompt
    prompt_content = prompt_template.replace("{query}", query).replace("{context_text}", context_text)
    
    # Estimate input tokens - rough estimation for non-OpenAI models
    input_tokens = estimate_token_count(prompt_content)
    
    # Record start time for response time tracking
    start_time = time.time()
    
    # Call Hugging Face API
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt_content,
        "parameters": {
            "temperature": 0.6,
            "max_new_tokens": 500,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    response = requests.post(
        MODEL_ENDPOINTS[model_id],
        headers=headers,
        json=payload
    )
    
    # Calculate response time
    response_time = time.time() - start_time
    
    if response.status_code != 200:
        raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
    
    result = response.json()
    
    # Extract the generated text
    if isinstance(result, list) and len(result) > 0:
        if "generated_text" in result[0]:
            generated_content = result[0]["generated_text"].strip()
        else:
            generated_content = str(result)
    else:
        generated_content = str(result)
    
    # Estimate output tokens
    output_tokens = estimate_token_count(generated_content)
    
    # Track API usage for analytics
    await track_api_usage(
        model=model_id,
        feature="generation",  # This could be "chat" or "extraction" depending on caller
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_time=response_time
    )
    
    return generated_content