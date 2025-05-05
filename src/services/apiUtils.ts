import axios from 'axios';
import analyticsService from './analyticsService';

const API_BASE_URL = 'http://localhost:8000';

/**
 * Enhanced axios API call that tracks analytics data
 */
export const trackableApiCall = async ({
  endpoint,
  method = 'POST',
  data = {},
  model,
  feature,
  input,
  expectedOutputLength = 500
}: {
  endpoint: string;
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  data?: any;
  model: string;
  feature: 'extraction' | 'generation' | 'chat';
  input: string;
  expectedOutputLength?: number;
}) => {
  const startTime = performance.now();
  
  try {
    // Make the API call
    const response = await axios({
      method,
      url: `${API_BASE_URL}${endpoint}`,
      data,
    });
    
    // Calculate response time
    const endTime = performance.now();
    const responseTime = (endTime - startTime) / 1000; // Convert to seconds
    
    // Estimate token counts
    const inputTokens = analyticsService.estimateTokenCount(input);
    
    // For the output, either use the response content or estimate based on expectedOutputLength
    const outputContent = typeof response.data.content === 'string' 
      ? response.data.content 
      : JSON.stringify(response.data);
    
    const outputTokens = analyticsService.estimateTokenCount(outputContent) || expectedOutputLength;
    
    // Calculate document size in KB
    const documentSize = input.length / 1024;
    
    // Track the usage asynchronously (don't await this)
    analyticsService.trackApiUsage({
      model,
      feature,
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      response_time: responseTime,
      document_size: documentSize
    });
    
    return response.data;
  } catch (error) {
    console.error(`Error in API call to ${endpoint}:`, error);
    throw error;
  }
};

/**
 * Upload a file with progress tracking and analytics
 */
export const uploadFileWithTracking = async ({
  file,
  endpoint,
  model,
  feature,
  onProgress,
  additionalFormData = {}
}: {
  file: File;
  endpoint: string;
  model: string;
  feature: 'extraction' | 'generation' | 'chat';
  onProgress?: (progress: number) => void;
  additionalFormData?: Record<string, any>;
}) => {
  const startTime = performance.now();
  
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    // Add any additional form data
    Object.entries(additionalFormData).forEach(([key, value]) => {
      formData.append(key, String(value));
    });
    
    // Make the API call with progress tracking
    const response = await axios.post(`${API_BASE_URL}${endpoint}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      }
    });
    
    // Calculate response time
    const endTime = performance.now();
    const responseTime = (endTime - startTime) / 1000; // Convert to seconds
    
    // Estimate token counts based on file size
    const inputTokens = Math.round(file.size / 4); // Very rough estimate
    const outputTokens = response.data ? analyticsService.estimateTokenCount(JSON.stringify(response.data)) : 1000;
    
    // Track the usage asynchronously
    analyticsService.trackApiUsage({
      model,
      feature,
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      response_time: responseTime,
      document_size: file.size / 1024 // Convert to KB
    });
    
    return response.data;
  } catch (error) {
    console.error(`Error uploading file to ${endpoint}:`, error);
    throw error;
  }
};

export default {
  trackableApiCall,
  uploadFileWithTracking
};