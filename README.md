# World Bank PDF Analyzer

An AI-powered application for analyzing, extracting insights, and interacting with World Bank documents.

## Overview

World Bank PDF Analyzer leverages the latest AI technologies to help users extract valuable information from World Bank documents, generate insights, and enhance project planning. The application combines document OCR, semantic search, and interactive chat capabilities to provide a comprehensive document analysis solution.

## Key Features

- **Document Preview**: Access Mistral OCR for high-quality text extraction from PDF documents
- **Chat with PDF**: Ask questions about your documents and receive contextually relevant answers
- **Generate Results**: Create comprehensive reports and insights based on document content
- **Analytics Dashboard**: Track usage metrics, costs, and performance across different models
- **Multi-Model Support**: Choose from various AI models including OpenAI GPT and open-source alternatives
- **Advanced RAG Pipeline**: Retrieval-Augmented Generation for accurate document-based responses

## Project Architecture

### Frontend
- React with TypeScript
- Tailwind CSS for responsive UI
- Theme support (Light, Dark, and Futuristic modes)

### Backend
- FastAPI (Python) for efficient API endpoints
- Vector database (FAISS) for semantic document search
- Multiple LLM integrations via API connectors
- Analytics tracking and performance monitoring

## Project Structure

```
/
├── api/                      # FastAPI backend
│   ├── __init__.py
│   ├── .env                  # Environment variables
│   ├── analytics_api.py      # Analytics endpoints
│   ├── analytics_data.json   # Usage data storage
│   ├── chat_api.py           # Chat functionality
│   ├── generation_api.py     # Content generation
│   ├── main.py               # Application entry point
│   └── process_pdf.py        # PDF processing pipeline
│
├── src/                      # Frontend React application
│   ├── components/           # UI components
│   ├── services/             # API service connectors
│   ├── App.tsx               # Main application
│   └── main.tsx              # Entry point
│
├── dist/                     # Built frontend assets
└── node_modules/             # Frontend dependencies
```

## Setup Instructions

### Prerequisites

- Node.js (v16+)
- Python (v3.8+)
- API keys for AI models:
  - OpenAI API key
  - Hugging Face API key
  - Mistral API key (optional)

### Backend Setup

1. Navigate to the API directory:
   ```
   cd api
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   ```

5. Start the backend server:
   ```
   uvicorn main:app --reload --port 8000
   ```

### Frontend Setup

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm run dev
   ```

The application will be available at http://localhost:5173

## API Endpoints

### Document Processing
- `POST /api/process-pdf`: Process a PDF document
- `GET /api/search-pdf`: Search within the document

### Chat
- `POST /api/chat`: Chat with PDF context

### Generation
- `POST /api/generate`: Generate content based on document context

### Analytics
- `POST /api/track-usage`: Track API usage metrics
- `GET /api/analytics`: Get analytics data

## Usage Guide

### Analyzing Documents
1. Upload your World Bank PDF document
2. View the extracted text and structure
3. Use the search function to find specific information

### Chatting with Documents
1. Upload a PDF or select from previous uploads
2. Ask questions about the document content
3. Select your preferred AI model
4. Receive contextually relevant answers

### Generating Content
1. Choose content generation options
2. Select model and parameters
3. Generate reports, summaries, or insights

### Analytics Dashboard
1. Track token usage, costs, and performance metrics
2. Compare different models and features
3. Filter by time period

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.