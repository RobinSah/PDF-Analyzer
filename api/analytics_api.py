from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Create router
router = APIRouter()

# Pydantic models for API requests and responses
class UsageData(BaseModel):
    model: str
    feature: str
    input_tokens: int
    output_tokens: int
    response_time: float
    document_size: Optional[float] = None

class AnalyticsSummary(BaseModel):
    totalTokens: int
    totalCost: float
    avgResponseTime: float
    documentsProcessed: int

class ModelComparisonData(BaseModel):
    models: List[str]
    tokens: List[int]
    costs: List[float]
    avgResponseTime: List[float]

class TimeAnalysisData(BaseModel):
    dates: List[str]
    processingTimes: List[float]
    documentSizes: List[float]

class CostAnalysisData(BaseModel):
    dates: List[str]
    models: Dict[str, List[float]]

class TokenUsageData(BaseModel):
    dates: List[str]
    extraction: List[int]
    generation: List[int]
    chat: List[int]

class AnalyticsResponse(BaseModel):
    tokenUsage: TokenUsageData
    costAnalysis: CostAnalysisData
    timeAnalysis: TimeAnalysisData
    modelComparison: ModelComparisonData
    summary: AnalyticsSummary

# Storage location
ANALYTICS_FILE = "analytics_data.json"

# Initialize analytics data structure
analytics_data = {
    "token_usage": {},
    "cost_data": {},
    "response_times": {},
    "model_usage": {},
    "documents_processed": {}
}

# List of models that should not be displayed in model comparisons
EXCLUDED_MODELS = ["pdf-search", "semantic-search", "gpt-3.5-turbo"]

# API pricing constants (per 1K tokens)
MODEL_PRICING = {
    "gpt-4": {
        "input": 0.03,
        "output": 0.06
    },
    "gpt-3.5-turbo": {
        "input": 0.0015,
        "output": 0.002
    },
    "mixtral-8x7b": {
        "input": 0.0006,
        "output": 0.0012
    },
    "llama-3": {
        "input": 0.0004,
        "output": 0.0008
    },
    "mistral-ocr-latest": {
        "input": 0.001,  # Example pricing for OCR
        "output": 0.001
    },
    "semantic-search": {
        "input": 0.0002,  # Example pricing for search
        "output": 0.0002
    },
    "pdf-search": {
        "input": 0.0002,  # Example pricing for search
        "output": 0.0002
    },
    "default": {
        "input": 0.001,
        "output": 0.002
    }
}

# Track API usage
@router.post("/api/track-usage")
async def track_usage(request_data: UsageData):
    """
    Track API usage including tokens, costs, and response times.
    """
    try:
        # Extract data from request
        model_name = request_data.model
        feature = request_data.feature
        input_tokens = request_data.input_tokens
        output_tokens = request_data.output_tokens
        response_time = request_data.response_time
        document_size = request_data.document_size
        
        # Get current date for grouping
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Initialize date entry if it doesn't exist
        if current_date not in analytics_data["token_usage"]:
            analytics_data["token_usage"][current_date] = {
                "extraction": 0,
                "generation": 0,
                "chat": 0
            }
            
        if current_date not in analytics_data["cost_data"]:
            analytics_data["cost_data"][current_date] = {}
            
        if current_date not in analytics_data["response_times"]:
            analytics_data["response_times"][current_date] = {
                "times": [],
                "document_sizes": []
            }
            
        if current_date not in analytics_data["documents_processed"]:
            analytics_data["documents_processed"][current_date] = 0
        
        # Update token usage
        if feature in analytics_data["token_usage"][current_date]:
            analytics_data["token_usage"][current_date][feature] += input_tokens + output_tokens
            
        # Calculate cost based on pricing
        model_pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        # Update cost data
        if model_name not in analytics_data["cost_data"][current_date]:
            analytics_data["cost_data"][current_date][model_name] = 0
        analytics_data["cost_data"][current_date][model_name] += total_cost
        
        # Update response times
        analytics_data["response_times"][current_date]["times"].append(response_time)
        
        # Update document sizes and count if document_size is provided
        if document_size:
            analytics_data["response_times"][current_date]["document_sizes"].append(document_size)
            
            # Update documents processed for any feature with document_size
            analytics_data["documents_processed"][current_date] += 1
        
        # Update model usage
        if model_name not in analytics_data["model_usage"]:
            analytics_data["model_usage"][model_name] = {
                "tokens": 0,
                "cost": 0,
                "response_times": []
            }
        analytics_data["model_usage"][model_name]["tokens"] += input_tokens + output_tokens
        analytics_data["model_usage"][model_name]["cost"] += total_cost
        analytics_data["model_usage"][model_name]["response_times"].append(response_time)
        
        # Save analytics to file
        save_analytics_data()
            
        return {"status": "success"}
        
    except Exception as e:
        print(f"Error tracking usage: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error tracking usage: {str(e)}")

# Get analytics data
@router.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    time_range: str = Query("week", description="Time range for analytics data"),
):
    """
    Get analytics data for the specified time range.
    """
    try:
        # Determine the start date based on time range
        today = datetime.now()
        if time_range == "day":
            start_date = today
        elif time_range == "week":
            start_date = today - timedelta(days=7)
        elif time_range == "month":
            start_date = today - timedelta(days=30)
        elif time_range == "year":
            start_date = today - timedelta(days=365)
        else:
            start_date = today - timedelta(days=7)
            
        start_date_str = start_date.strftime("%Y-%m-%d")
        
        # Filter data by date range
        filtered_token_usage = {
            date: data for date, data in analytics_data["token_usage"].items() 
            if date >= start_date_str
        }
        
        filtered_cost_data = {
            date: data for date, data in analytics_data["cost_data"].items() 
            if date >= start_date_str
        }
        
        filtered_response_times = {
            date: data for date, data in analytics_data["response_times"].items() 
            if date >= start_date_str
        }
        
        filtered_documents_processed = {
            date: count for date, count in analytics_data["documents_processed"].items()
            if date >= start_date_str
        }
        
        # Prepare the response data
        response = {
            "tokenUsage": prepare_token_usage_data(filtered_token_usage),
            "costAnalysis": prepare_cost_analysis_data(filtered_cost_data),
            "timeAnalysis": prepare_time_analysis_data(filtered_response_times),
            "modelComparison": prepare_model_comparison_data(),
            "summary": prepare_summary_data(
                filtered_token_usage, 
                filtered_cost_data, 
                filtered_response_times, 
                filtered_documents_processed
            )
        }
        
        return response
        
    except Exception as e:
        print(f"Error getting analytics data: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting analytics data: {str(e)}")

# Helper function to save analytics data to a file
def save_analytics_data():
    """
    Save analytics data to a JSON file.
    """
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(analytics_data, f, indent=2)
    except Exception as e:
        print(f"Error saving analytics data: {e}")
        traceback.print_exc()

# Helper functions to prepare data for the frontend
def prepare_token_usage_data(token_usage_data):
    """
    Transform token usage data for frontend visualization.
    """
    # Get dates in chronological order
    dates = sorted(token_usage_data.keys())
    
    # Prepare data structure for charts
    result = {
        "dates": [],
        "extraction": [],
        "generation": [],
        "chat": []
    }
    
    # Populate data for each date
    for date in dates:
        # Format date for display (e.g., "2023-04-28" to "Apr 28")
        display_date = datetime.strptime(date, "%Y-%m-%d").strftime("%b %d")
        result["dates"].append(display_date)
        
        # Add data for each feature
        result["extraction"].append(token_usage_data[date].get("extraction", 0))
        result["generation"].append(token_usage_data[date].get("generation", 0))
        result["chat"].append(token_usage_data[date].get("chat", 0))
    
    return result

def prepare_cost_analysis_data(cost_data):
    """
    Transform cost data for frontend visualization.
    """
    # Get dates in chronological order
    dates = sorted(cost_data.keys())
    
    # Get all models used, excluding filtered models
    all_models = set()
    for date_data in cost_data.values():
        for model in date_data.keys():
            if model not in EXCLUDED_MODELS:
                all_models.add(model)
    
    # Prepare data structure for charts
    result = {
        "dates": [],
        "models": {model: [] for model in all_models}
    }
    
    # Populate data for each date
    for date in dates:
        # Format date for display
        display_date = datetime.strptime(date, "%Y-%m-%d").strftime("%b %d")
        result["dates"].append(display_date)
        
        # Add cost for each model
        for model in all_models:
            result["models"][model].append(cost_data[date].get(model, 0))
    
    return result

def prepare_time_analysis_data(response_times_data):
    """
    Transform response time data for frontend visualization.
    """
    # Get dates in chronological order
    dates = sorted(response_times_data.keys())
    
    # Prepare data structure for charts
    result = {
        "dates": [],
        "processingTimes": [],
        "documentSizes": []
    }
    
    # Populate data for each date
    for date in dates:
        # Format date for display
        display_date = datetime.strptime(date, "%Y-%m-%d").strftime("%b %d")
        result["dates"].append(display_date)
        
        # Calculate average response time for the day
        times = response_times_data[date]["times"]
        avg_time = sum(times) / len(times) if times else 0
        result["processingTimes"].append(avg_time)
        
        # Calculate average document size for the day
        sizes = response_times_data[date]["document_sizes"]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        result["documentSizes"].append(avg_size)
    
    return result

def prepare_model_comparison_data():
    """
    Prepare model comparison data for frontend visualization.
    """
    # Get all models, excluding non-LLM "models" like search features and gpt-3.5-turbo
    models = [model for model in analytics_data["model_usage"].keys() 
              if model not in EXCLUDED_MODELS]
    
    # If no models have been used yet, provide some default data
    if not models:
        return {
            "models": ["gpt-4", "mixtral-8x7b", "llama-3"],
            "tokens": [0, 0, 0],
            "costs": [0, 0, 0],
            "avgResponseTime": [0, 0, 0]
        }
    
    # Prepare result structure
    result = {
        "models": models,
        "tokens": [],
        "costs": [],
        "avgResponseTime": []
    }
    
    # Populate data for each model
    for model in models:
        model_data = analytics_data["model_usage"].get(model, {})
        
        # Token usage
        result["tokens"].append(model_data.get("tokens", 0))
        
        # Cost
        result["costs"].append(model_data.get("cost", 0))
        
        # Average response time
        times = model_data.get("response_times", [])
        avg_time = sum(times) / len(times) if times else 0
        result["avgResponseTime"].append(avg_time)
    
    return result

def prepare_summary_data(token_usage, cost_data, response_times, documents_processed):
    """
    Prepare summary metrics for the dashboard overview.
    """
    # Calculate total tokens
    total_tokens = 0
    for date_data in token_usage.values():
        for feature, tokens in date_data.items():
            total_tokens += tokens
    
    # Calculate total cost
    total_cost = 0
    for date_data in cost_data.values():
        for model, cost in date_data.items():
            total_cost += cost
    
    # Calculate average response time
    all_times = []
    for date_data in response_times.values():
        all_times.extend(date_data["times"])
    
    avg_response_time = sum(all_times) / len(all_times) if all_times else 0
    
    # Count total documents processed
    total_documents = sum(documents_processed.values())
    
    return {
        "totalTokens": total_tokens,
        "totalCost": total_cost,
        "avgResponseTime": avg_response_time,
        "documentsProcessed": total_documents
    }

# Initialize analytics data from file if exists
def load_analytics_data():
    """
    Load analytics data from the JSON file if it exists.
    """
    global analytics_data
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, "r") as f:
                loaded_data = json.load(f)
                
                # Ensure all required keys exist
                for key in ["token_usage", "cost_data", "response_times", "model_usage", "documents_processed"]:
                    if key not in loaded_data:
                        loaded_data[key] = {}
                
                analytics_data = loaded_data
                print(f"Loaded analytics data from {ANALYTICS_FILE}")
        else:
            # Initialize with empty structure and save
            save_analytics_data()
            print(f"Initialized new analytics data file: {ANALYTICS_FILE}")
    except Exception as e:
        print(f"Error loading analytics data: {e}")
        traceback.print_exc()
        # Continue with empty data

# Load data on startup
load_analytics_data()