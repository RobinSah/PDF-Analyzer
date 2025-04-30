from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import process_pdf
import chat_api
import generation_api
import analytics_api  # Import the new analytics API module

# Create the FastAPI app
app = FastAPI(title="World Bank PDF Analyzer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers from different modules
app.include_router(process_pdf.router)
app.include_router(chat_api.router)
app.include_router(generation_api.router)
app.include_router(analytics_api.router)  # Add the analytics router

@app.get("/")
async def root():
    return {
        "message": "Welcome to the World Bank PDF Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "PDF Processing": ["/api/process-pdf", "/api/process-pdf-url", "/api/search-pdf"],
            "Chat": ["/api/chat"],
            "Generation": ["/api/generate", "/api/generate-bulk"],
            "Analytics": ["/api/track-usage", "/api/analytics"]
        }
    }

# Run the app using Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)