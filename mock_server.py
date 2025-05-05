# improved_mock_server.py - Server with rich markdown support
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

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

# Rich markdown mock data
mock_pages = [
    {
        "page_number": 0,
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
        "page_number": 1,
        "markdown": """## PROJECT APPRAISAL SUMMARY

..... 26 A. Technical, Economic and Financial Analysis
..... 26 B. Fiduciary ..... 27 C. Legal Operational Policies ..... 28 D. Environmental and Social ..... 28 V. GRIEVANCE REDRESS SERVICES ..... 29 VI. KEY RISKS ..... 30 VII. CORPORATE COMMITMENTS ..... 31 VIII. RESULTS FRAMEWORK AND MONITORING ..... 33

## ANNEX 1: Implementation Arrangements and Support Plan 
..... 57 

## ANNEX 2: Further Detail on Specific Project Activities
..... 66 

## ANNEX 3: Economic and Financial Analysis
..... 74 

## ANNEX 4: Key Design Features of Infrastructure Interventions under the Project
..... 78 

## ANNEX 5: Gender Gap Analysis
..... 81 

## ANNEX 6: References
..... 90 

## DATASHEET BASIC INFORMATION
Country(ies): Paraguay""",
        "plain_text": "PROJECT APPRAISAL SUMMARY ..... 26 A. Technical, Economic and Financial Analysis ..... 26 B. Fiduciary ..... 27 C. Legal Operational Policies ..... 28 D. Environmental and Social ..... 28 V. GRIEVANCE REDRESS SERVICES ..... 29 VI. KEY RISKS ..... 30 VII. CORPORATE COMMITMENTS ..... 31 VIII. RESULTS FRAMEWORK AND MONITORING ..... 33 ANNEX 1: Implementation Arrangements and Support Plan ..... 57 ANNEX 2: Further Detail on Specific Project Activities ..... 66 ANNEX 3: Economic and Financial Analysis ..... 74 ANNEX 4: Key Design Features of Infrastructure Interventions under the Project ..... 78 ANNEX 5: Gender Gap Analysis ..... 81 ANNEX 6: References ..... 90 DATASHEET BASIC INFORMATION Country(ies): Paraguay"
    },
    {
        "page_number": 2,
        "markdown": """## Project Name: Joining Efforts for an Education of Excellence in Paraguay Project

### Financing Data

- Project ID: P176869  
- Financing Instrument: Investment Project Financing  
- Original Environmental and Social Risk Classification: Moderate  
- Current Environmental and Social Risk Classification: Moderate  
- Approval Date: April 20, 2023  
- Project Stage: Implementation  
- Bank Approval Date: January 18, 2023  
- Financing source Description Effectiveness IBRD/IDA: This loan requires that an environmental specialist, a social specialist, an indigenous people's specialist and an inclusive education specialist have been hired as part of the UEPP, with qualifications, experience and terms of reference satisfactory to the Bank, as set forth in the Operational Manual.""",
        "plain_text": "Project Name: Joining Efforts for an Education of Excellence in Paraguay Project Financing Data Project ID: P176869 Financing Instrument: Investment Project Financing Original Environmental and Social Risk Classification: Moderate Current Environmental and Social Risk Classification: Moderate Approval Date: April 20, 2023 Project Stage: Implementation Bank Approval Date: January 18, 2023 Financing source Description Effectiveness IBRD/IDA: This loan requires that an environmental specialist, a social specialist, an indigenous people's specialist and an inclusive education specialist have been hired as part of the UEPP, with qualifications, experience and terms of reference satisfactory to the Bank, as set forth in the Operational Manual."
    },
    {
        "page_number": 3,
        "markdown": """## Digital Competence Framework

[https://unevoc.unesco.org/home/Digital+Competence+Framework](https://unevoc.unesco.org/home/Digital+Competence+Framework)

To be built from the MEC's social network INNOVA, available on MEC's dedicated portal (Paraguay Aprende). Paraguay Aprende is a free and unrestricted access portal with multimedia and interactive educational content for teachers and students from public schools. Awards will take the shape of inkind prizes akin to those for teachers and school directors and also include MEC's institutional support with the National Directorate of Intellectual Property (Dirección Nacional de la Propiedad Intelectual) to patent products/services.

connectivity, but are expected to receive digital devices under the Project; and (iii) a group 3 made of 1,327 UCEIs that do not have digital devices or connectivity to date, and are expected to receive both under the Project. The assessment of learning outcomes will use three sets of data: (i) SNEPE 2018, which is the last national standardized exam for Paraguay and was undertaken in November 2018, prior to the deployment of any Internet connectivity or distribution of digital devices to educational actors; (ii) SNEPE 2024 (estimated), which will be undertaken after the implementation of Component 1, and as such will capture the early impacts of computer-assisted learning programs and of digital educational strategies.""",
        "plain_text": "Digital Competence Framework https://unevoc.unesco.org/home/Digital+Competence+Framework To be built from the MEC's social network INNOVA, available on MEC's dedicated portal (Paraguay Aprende). Paraguay Aprende is a free and unrestricted access portal with multimedia and interactive educational content for teachers and students from public schools. Awards will take the shape of inkind prizes akin to those for teachers and school directors and also include MEC's institutional support with the National Directorate of Intellectual Property (Dirección Nacional de la Propiedad Intelectual) to patent products/services. connectivity, but are expected to receive digital devices under the Project; and (iii) a group 3 made of 1,327 UCEIs that do not have digital devices or connectivity to date, and are expected to receive both under the Project. The assessment of learning outcomes will use three sets of data: (i) SNEPE 2018, which is the last national standardized exam for Paraguay and was undertaken in November 2018, prior to the deployment of any Internet connectivity or distribution of digital devices to educational actors; (ii) SNEPE 2024 (estimated), which will be undertaken after the implementation of Component 1, and as such will capture the early impacts of computer-assisted learning programs and of digital educational strategies."
    }
]

@app.get("/")
async def root():
    return {"message": "Improved mock server is running"}

@app.get("/api/debug")
async def debug_info():
    """Get debug information about the current state."""
    return {
        "server_type": "improved_mock",
        "has_structured_pages": True,
        "num_pages": len(mock_pages),
        "has_rich_markdown": True,
        "openai_api_key_set": os.environ.get("OPENAI_API_KEY") is not None,
    }

@app.post("/api/process-pdf")
async def api_process_pdf(file: UploadFile = File(...)):
    """Mock processing a PDF file with rich markdown."""
    print(f"Received file: {file.filename}")
    # Return mock structured pages with rich markdown
    return {"status": "success", "structured_pages": mock_pages}

@app.post("/api/process-pdf-url")
async def api_process_pdf_url(data: URLInput):
    """Mock processing a PDF from URL with rich markdown."""
    print(f"Received URL: {data.url}")
    # Return mock structured pages with rich markdown
    return {"status": "success", "structured_pages": mock_pages}

@app.post("/api/search-pdf")
async def api_search_pdf(data: QueryInput):
    """Mock searching within a processed PDF with rich markdown results."""
    print(f"Received search query: {data.query}")
    query = data.query
    
    # Customize mock results based on query content
    if "annex" in query.lower():
        mock_results = [
            {
                "text": f"ANNEX 2: Further Detail on Specific Project Activities related to {query}",
                "score": 0.92,
                "page_number": 1,
                "markdown": f"## ANNEX 2: Further Detail on Specific Project Activities\n\nInformation related to **{query}** can be found in this section. The annex provides detailed specifications for project implementation."
            },
            {
                "text": f"ANNEX 1: Implementation Arrangements and Support Plan mentions {query}",
                "score": 0.85,
                "page_number": 1,
                "markdown": f"## ANNEX 1: Implementation Arrangements and Support Plan\n\nThis section includes references to **{query}** and outlines the implementation approach."
            },
            {
                "text": f"ANNEX 3: Economic and Financial Analysis relevant to {query}",
                "score": 0.78,
                "page_number": 1,
                "markdown": f"## ANNEX 3: Economic and Financial Analysis\n\nThe economic analysis considers factors related to **{query}** and provides financial projections."
            }
        ]
    elif "digital" in query.lower() or "competence" in query.lower():
        mock_results = [
            {
                "text": f"Digital Competence Framework and related information about {query}",
                "score": 0.94,
                "page_number": 3,
                "markdown": f"## Digital Competence Framework\n\n[https://unevoc.unesco.org/home/Digital+Competence+Framework](https://unevoc.unesco.org/home/Digital+Competence+Framework)\n\nTo be built from the MEC's social network INNOVA, available on MEC's dedicated portal (Paraguay Aprende). This relates to **{query}** and provides a framework for digital skills."
            },
            {
                "text": f"Digital devices distribution under the Project related to {query}",
                "score": 0.87,
                "page_number": 3,
                "markdown": f"Connectivity and **{query}** are addressed in the project through digital devices distribution to educational actors. A group 3 made of 1,327 UCEIs that do not have digital devices or connectivity to date will receive support."
            }
        ]
    else:
        # Default mock results
        mock_results = [
            {
                "text": f"Found relevant information about {query} in the project overview",
                "score": 0.89,
                "page_number": 0,
                "markdown": f"# THE WORLD BANK\n\nFOR OFFICIAL USE ONLY\n\nThis section contains information about **{query}** in the context of the Paraguay education project."
            },
            {
                "text": f"Project Appraisal Summary mentions {query}",
                "score": 0.82,
                "page_number": 1,
                "markdown": f"## PROJECT APPRAISAL SUMMARY\n\nThe project appraisal includes considerations related to **{query}** and outlines relevant components."
            },
            {
                "text": f"Financing Data related to {query}",
                "score": 0.77,
                "page_number": 2,
                "markdown": f"### Financing Data\n\n- Project ID: P176869\n- Relevant to **{query}**: Investment Project Financing"
            }
        ]
    
    return {"results": mock_results}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)