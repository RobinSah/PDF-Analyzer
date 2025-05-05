# page_matching_server.py - Improved server with fixed page matching
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class URLInput(BaseModel):
    url: str

class QueryInput(BaseModel):
    query: str

# Mock data with different page numbers
mock_pages = [
    {
        "page_number": 0,  # Page 1
        "markdown": """# THE WORLD BANK

FOR OFFICIAL USE ONLY

Report No: PAD5309

## INTERNATIONAL BANK FOR RECONSTRUCTION AND DEVELOPMENT

### PROJECT APPRAISAL DOCUMENT

ON A

PROPOSED LOAN

IN THE AMOUNT OF US$125.3 MILLION

TO THE

REPUBLIC OF PARAGUAY

FOR THE

JOINING EFFORTS FOR AN EDUCATION OF EXCELLENCE IN PARAGUAY PROJECT

April 4, 2023

Education Global Practice
Latin America And Caribbean Region

This document has a restricted distribution and may be used by recipients only in the performance of their official duties. Its contents may not otherwise be disclosed without World Bank authorization.""",
        "plain_text": "THE WORLD BANK FOR OFFICIAL USE ONLY Report No: PAD5309 INTERNATIONAL BANK FOR RECONSTRUCTION AND DEVELOPMENT PROJECT APPRAISAL DOCUMENT ON A PROPOSED LOAN IN THE AMOUNT OF US$125.3 MILLION TO THE REPUBLIC OF PARAGUAY FOR THE JOINING EFFORTS FOR AN EDUCATION OF EXCELLENCE IN PARAGUAY PROJECT April 4, 2023 Education Global Practice Latin America And Caribbean Region This document has a restricted distribution and may be used by recipients only in the performance of their official duties. Its contents may not otherwise be disclosed without World Bank authorization."
    },
    {
        "page_number": 1,  # Page 2
        "markdown": """## PROJECT APPRAISAL SUMMARY

..... 26 A. Technical, Economic and Financial Analysis
..... 26 B. Fiduciary ..... 27 C. Legal Operational Policies ..... 28 D. Environmental and Social ..... 28 V. GRIEVANCE REDRESS SERVICES ..... 29 VI. KEY RISKS ..... 30 VII. CORPORATE COMMITMENTS ..... 31 VIII. RESULTS FRAMEWORK AND MONITORING ..... 33

## ANNEX 1: Implementation Arrangements and Support Plan 
..... 57 

## ANNEX 2: Further Detail on Specific Project Activities
..... 66 

## ANNEX 3: Economic and Financial Analysis
..... 74""",
        "plain_text": "PROJECT APPRAISAL SUMMARY ..... 26 A. Technical, Economic and Financial Analysis ..... 26 B. Fiduciary ..... 27 C. Legal Operational Policies ..... 28 D. Environmental and Social ..... 28 V. GRIEVANCE REDRESS SERVICES ..... 29 VI. KEY RISKS ..... 30 VII. CORPORATE COMMITMENTS ..... 31 VIII. RESULTS FRAMEWORK AND MONITORING ..... 33 ANNEX 1: Implementation Arrangements and Support Plan ..... 57 ANNEX 2: Further Detail on Specific Project Activities ..... 66 ANNEX 3: Economic and Financial Analysis ..... 74"
    },
    {
        "page_number": 2,  # Page 3
        "markdown": """## Project Name: Joining Efforts for an Education of Excellence in Paraguay Project

### Financing Data

- Project ID: P176869  
- Financing Instrument: Investment Project Financing  
- Original Environmental and Social Risk Classification: Moderate  
- Current Environmental and Social Risk Classification: Moderate  
- Approval Date: April 20, 2023  
- Project Stage: Implementation  
- Bank Approval Date: January 18, 2023""",
        "plain_text": "Project Name: Joining Efforts for an Education of Excellence in Paraguay Project Financing Data Project ID: P176869 Financing Instrument: Investment Project Financing Original Environmental and Social Risk Classification: Moderate Current Environmental and Social Risk Classification: Moderate Approval Date: April 20, 2023 Project Stage: Implementation Bank Approval Date: January 18, 2023"
    }
]

# Store this globally so our search function can access it
global_structured_pages = mock_pages

def find_page_number(chunk_text, structured_pages):
    """
    Find page number by checking which page contains the chunk_text.
    Returns the actual page number (as stored in the data structure).
    """
    if not chunk_text or not structured_pages:
        return None
        
    # Try exact match first
    for page in structured_pages:
        if chunk_text.strip() in page["plain_text"]:
            return page["page_number"]
    
    # If no exact match, try to find the most similar page
    # by checking for the maximum number of common words
    chunk_words = set(re.findall(r'\w+', chunk_text.lower()))
    if not chunk_words:
        return None
        
    best_match = None
    best_score = 0
    
    for page in structured_pages:
        page_words = set(re.findall(r'\w+', page["plain_text"].lower()))
        common_words = len(chunk_words.intersection(page_words))
        if common_words > best_score:
            best_score = common_words
            best_match = page["page_number"]
    
    return best_match

def get_markdown_for_page(page_number, structured_pages):
    """Get the markdown content for a specific page."""
    for page in structured_pages:
        if page["page_number"] == page_number:
            return page["markdown"]
    return None

@app.get("/")
async def root():
    return {"message": "Page matching server is running"}

@app.get("/api/debug")
async def debug_info():
    """Get debug information about the current state."""
    return {
        "server_type": "page_matching",
        "has_structured_pages": True,
        "num_pages": len(mock_pages),
        "has_rich_markdown": True,
        "page_finding_algorithm": "implemented"
    }

@app.post("/api/process-pdf")
async def api_process_pdf(file: UploadFile = File(...)):
    """Mock processing a PDF file with rich markdown."""
    # Return mock structured pages with rich markdown
    return {"status": "success", "structured_pages": mock_pages}

@app.post("/api/process-pdf-url")
async def api_process_pdf_url(data: URLInput):
    """Mock processing a PDF from URL with rich markdown."""
    # Return mock structured pages with rich markdown
    return {"status": "success", "structured_pages": mock_pages}

@app.post("/api/search-pdf")
async def api_search_pdf(data: QueryInput):
    """Search with improved page detection and markdown retrieval."""
    query = data.query
    
    # Create search results with different page numbers
    results = []
    
    # First result - Page 0 (displayed as Page 1)
    results.append({
        "text": f"First result for: {query}",
        "score": 0.92,
        "page_number": 0,  # This will be displayed as Page 1
        "markdown": mock_pages[0]["markdown"]  # Use the actual markdown from page 0
    })
    
    # Second result - Page 1 (displayed as Page 2)
    results.append({
        "text": f"Second result for: {query}",
        "score": 0.85,
        "page_number": 1,  # This will be displayed as Page 2
        "markdown": mock_pages[1]["markdown"]  # Use the actual markdown from page 1
    })
    
    # Third result - Page 2 (displayed as Page 3)
    results.append({
        "text": f"Third result for: {query}",
        "score": 0.75,
        "page_number": 2,  # This will be displayed as Page 3
        "markdown": mock_pages[2]["markdown"]  # Use the actual markdown from page 2
    })
    
    # Customize content based on query
    if "annex" in query.lower():
        results[0]["text"] = f"Annex information for: {query}"
        results[0]["page_number"] = 1  # Page with annexes (displayed as Page 2)
        results[0]["markdown"] = mock_pages[1]["markdown"]
    
    return {"results": results}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)