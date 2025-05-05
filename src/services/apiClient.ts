// apiClient.ts - Centralized API client for frontend applications
import axios from 'axios';
import analyticsService from './analyticsService';

// Define API base URL
const API_BASE_URL = 'http://localhost:8000';

// Interfaces for API responses
interface StructuredPage {
  page_number: number;
  markdown: string;
  plain_text: string;
}

interface ProcessPdfResponse {
  status: string;
  structured_pages: StructuredPage[];
}

interface SearchResult {
  text: string;
  score: number;
  page_number: number;
}

interface SearchResponse {
  results: SearchResult[];
}

interface ChatMessage {
  role: string;
  content: string;
}

interface ChatResponse {
  response: string;
  model: string;
}

interface GenerateResponse {
  content: string;
  model: string;
}

// Create axios instance with base URL
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Add response interceptor to track API response times
apiClient.interceptors.request.use(function (config) {
  // Attach timing measurement to the request for later use
  config.metadata = { startTime: new Date().getTime() };
  return config;
}, function (error) {
  return Promise.reject(error);
});

apiClient.interceptors.response.use(function (response) {
  // Calculate response time
  const startTime = response.config.metadata.startTime;
  const endTime = new Date().getTime();
  const responseTimeMs = endTime - startTime;
  const responseTimeSec = responseTimeMs / 1000; // Convert to seconds for consistency with backend
  
  // Track API usage if a model was used
  const url = response.config.url;
  
  // Skip tracking for analytics endpoints to avoid recursion
  if (!url || url.includes('/api/analytics') || url.includes('/api/track-usage')) {
    return response;
  }
  
  // Determine which API was called
  if (url.includes('/api/process-pdf') || url.includes('/api/process-pdf-url')) {
    // Track PDF processing
    const fileSize = response.config.data ? response.config.data.size || 0 : 0;
    const sizeKB = fileSize / 1024;
    
    // Estimated tokens based on page count from response
    const pageCount = response.data.structured_pages ? response.data.structured_pages.length : 0;
    const inputTokensEstimate = sizeKB * 50; // Rough estimate: 50 tokens per KB
    const outputTokensEstimate = pageCount * 500; // Rough estimate: 500 tokens per page
    
    analyticsService.trackApiUsage({
      model: 'mistral-ocr-latest',
      feature: 'extraction',
      input_tokens: Math.round(inputTokensEstimate),
      output_tokens: Math.round(outputTokensEstimate),
      response_time: responseTimeSec,
      document_size: sizeKB
    }).catch(err => console.error('Error tracking PDF processing:', err));
  }
  else if (url.includes('/api/generate')) {
    // Track content generation
    const requestData = response.config.data ? JSON.parse(response.config.data) : {};
    const model = requestData.model || 'unknown';
    const activity = requestData.activity || '';
    const definition = requestData.definition || '';
    const query = `${activity}\n${definition}`;
    
    // Estimate tokens
    const inputTokensEstimate = analyticsService.estimateTokenCount(query);
    const outputTokensEstimate = analyticsService.estimateTokenCount(
      response.data.content ? response.data.content : ''
    );
    
    analyticsService.trackApiUsage({
      model: model,
      feature: 'generation',
      input_tokens: inputTokensEstimate,
      output_tokens: outputTokensEstimate,
      response_time: responseTimeSec
    }).catch(err => console.error('Error tracking content generation:', err));
  }
  else if (url.includes('/api/chat')) {
    // Track chat conversations
    const requestData = response.config.data ? JSON.parse(response.config.data) : {};
    const model = requestData.model || 'unknown';
    const messages = requestData.messages || [];
    const context = requestData.context || '';
    
    // Calculate input tokens from messages
    let input = context ? context + '\n\n' : '';
    messages.forEach(msg => {
      input += `${msg.role}: ${msg.content}\n`;
    });
    
    const inputTokensEstimate = analyticsService.estimateTokenCount(input);
    const outputTokensEstimate = analyticsService.estimateTokenCount(
      response.data.response ? response.data.response : ''
    );
    
    analyticsService.trackApiUsage({
      model: model,
      feature: 'chat',
      input_tokens: inputTokensEstimate,
      output_tokens: outputTokensEstimate,
      response_time: responseTimeSec,
      document_size: context ? (context.length / 1024) : undefined
    }).catch(err => console.error('Error tracking chat:', err));
  }
  
  return response;
}, function (error) {
  return Promise.reject(error);
});

// Error handling wrapper
const apiWrapper = async <T,>(
  apiCall: () => Promise<T>,
  errorMessage: string = 'An error occurred'
): Promise<{ data: T | null; error: string | null }> => {
  try {
    const data = await apiCall();
    return { data, error: null };
  } catch (error) {
    console.error(error);
    const message = error instanceof Error ? error.message : errorMessage;
    return { data: null, error: message };
  }
};

// PDF Processing
export const processPdf = async (file: File): Promise<ProcessPdfResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await apiClient.post<ProcessPdfResponse>(
      '/api/process-pdf',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error processing PDF:', error);
    throw error;
  }
};

export const processPdfUrl = async (url: string): Promise<ProcessPdfResponse> => {
  try {
    const response = await apiClient.post<ProcessPdfResponse>(
      '/api/process-pdf-url',
      { url }
    );
    return response.data;
  } catch (error) {
    console.error('Error processing PDF URL:', error);
    throw error;
  }
};

// Content Generation
export const generateContent = async (params: {
  model: string;
  activity?: string;
  definition?: string;
  prompt?: string;
}): Promise<GenerateResponse> => {
  try {
    const response = await apiClient.post<GenerateResponse>('/api/generate', params);
    return response.data;
  } catch (error) {
    console.error('Error generating content:', error);
    throw error;
  }
};

export const generateBulkContent = async (
  file: File,
  pdfFile?: File,
  queryLimit?: number,
  model?: string,
  prompt?: string
): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);
  
  if (queryLimit) {
    formData.append('query_limit', queryLimit.toString());
  }
  
  if (pdfFile) {
    // First process the PDF to get the text content
    const pdfContent = await processPdf(pdfFile);
    
    // Add the PDF content to the form data
    if (pdfContent.structured_pages) {
      const plainText = pdfContent.structured_pages
        .map(page => page.plain_text)
        .join('\n\n');
      
      formData.append('pdf_content', plainText);
    }
  }
  
  if (model) {
    formData.append('model', model);
  }
  
  if (prompt) {
    formData.append('prompt', prompt);
  }
  
  try {
    const response = await apiClient.post(
      '/api/generate-bulk',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error processing bulk generation:', error);
    throw error;
  }
};

// Chat with PDF
export const chatWithPdf = async (
  model: string,
  messages: ChatMessage[],
  context?: string
): Promise<ChatResponse> => {
  try {
    const response = await apiClient.post<ChatResponse>('/api/chat', {
      model,
      messages,
      context
    });
    return response.data;
  } catch (error) {
    console.error('Error in chat:', error);
    throw error;
  }
};

// PDF Search
export const searchPdf = async (query: string): Promise<SearchResponse> => {
  try {
    const response = await apiClient.post<SearchResponse>('/api/search-pdf', { query });
    return response.data;
  } catch (error) {
    console.error('Error searching PDF:', error);
    throw error;
  }
};

// Analytics
export const fetchAnalytics = async (timeRange: string = 'week'): Promise<any> => {
  try {
    const response = await apiClient.get(`/api/analytics?time_range=${timeRange}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching analytics:', error);
    throw error;
  }
};

// Wrapped versions of the API calls with better error handling
export const safeProcessPdf = (file: File) => 
  apiWrapper(() => processPdf(file), 'Error processing PDF file');

export const safeProcessPdfUrl = (url: string) => 
  apiWrapper(() => processPdfUrl(url), 'Error processing PDF from URL');

export const safeSearchPdf = (query: string) => 
  apiWrapper(() => searchPdf(query), 'Error searching PDF');

export const safeChatWithPdf = (model: string, messages: ChatMessage[], context?: string) => 
  apiWrapper(() => chatWithPdf(model, messages, context), 'Error in chat conversation');

export const safeGenerateContent = (params: any) => 
  apiWrapper(() => generateContent(params), 'Error generating content');

export default {
  // Direct API calls
  processPdf,
  processPdfUrl,
  generateContent,
  generateBulkContent,
  chatWithPdf,
  searchPdf,
  fetchAnalytics,
  
  // Safe wrapped versions
  safeProcessPdf,
  safeProcessPdfUrl,
  safeSearchPdf,
  safeChatWithPdf,
  safeGenerateContent
};