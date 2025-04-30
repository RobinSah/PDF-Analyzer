# Add these imports at the top of your generation_api.py file
import time
from datetime import datetime
import os
import json
from typing import Dict, Any, Optional

# Add this function after your existing imports and before your API endpoints
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
        # Using direct request to avoid circular imports
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/api/track-usage", 
                json=usage_data
            )
            
        if response.status_code != 200:
            print(f"Error tracking API usage: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error tracking API usage: {str(e)}")

# Now, modify the generate_openai_response function to track usage
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
    input_tokens = len(system_content.split()) * 1.33  # Rough estimation
    
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
    
    # Get the actual token usage from the response
    input_tokens = result.get("usage", {}).get("prompt_tokens", int(input_tokens))
    output_tokens = result.get("usage", {}).get("completion_tokens", len(generated_content.split()) * 1.33)
    
    # Track API usage for analytics
    document_size = len(context_text) / 1024  # Size in KB
    
    # Determine feature based on calling function or route
    # This is a simplified approach - adjust based on your app structure
    import inspect
    frame = inspect.currentframe()
    try:
        # Look at the call stack to determine the feature
        if frame and frame.f_back:
            caller = frame.f_back.f_code.co_name
            if "chat" in caller:
                feature = "chat"
            elif "extract" in caller or "process_pdf" in caller:
                feature = "extraction"
            else:
                feature = "generation"
        else:
            feature = "generation"  # Default
    finally:
        del frame  # Avoid reference cycles
    
    await track_api_usage(
        model=openai_model,
        feature=feature,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_time=response_time,
        document_size=document_size
    )
    
    return generated_content

# Similarly, modify the generate_huggingface_response function
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
    input_tokens = len(prompt_content.split()) * 1.33
    
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
    output_tokens = len(generated_content.split()) * 1.33
    
    # Determine feature based on calling function or route
    import inspect
    frame = inspect.currentframe()
    try:
        # Look at the call stack to determine the feature
        if frame and frame.f_back:
            caller = frame.f_back.f_code.co_name
            if "chat" in caller:
                feature = "chat"
            elif "extract" in caller or "process_pdf" in caller:
                feature = "extraction"
            else:
                feature = "generation"
        else:
            feature = "generation"  # Default
    finally:
        del frame  # Avoid reference cycles
    
    # Track API usage for analytics
    document_size = len(context_text) / 1024  # Size in KB
    await track_api_usage(
        model=model_id,
        feature=feature,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        response_time=response_time,
        document_size=document_size
    )
    
    return generated_content