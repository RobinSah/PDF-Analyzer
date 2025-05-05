import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Interface for tracking API usage
interface UsageData {
  model: string;
  feature: 'extraction' | 'generation' | 'chat';
  input_tokens: number;
  output_tokens: number;
  response_time: number;
  document_size?: number;
}

// Interface for analytics response
export interface AnalyticsResponse {
  tokenUsage: {
    dates: string[];
    extraction: number[];
    generation: number[];
    chat: number[];
  };
  costAnalysis: {
    dates: string[];
    models: {
      [key: string]: number[];
    };
  };
  timeAnalysis: {
    dates: string[];
    processingTimes: number[];
    documentSizes: number[];
  };
  modelComparison: {
    models: string[];
    tokens: number[];
    costs: number[];
    avgResponseTime: number[];
  };
  summary: {
    totalTokens: number;
    totalCost: number;
    avgResponseTime: number;
    documentsProcessed: number;
  };
}

/**
 * Track API usage statistics
 */
export const trackApiUsage = async (data: UsageData): Promise<boolean> => {
  try {
    await axios.post(`${API_BASE_URL}/api/track-usage`, data);
    return true;
  } catch (error) {
    console.error('Error tracking API usage:', error);
    return false;
  }
};

/**
 * Estimate token count for a text string
 * This is a simplified estimation - accurate counts come from the API
 */
export const estimateTokenCount = (text: string): number => {
  if (!text) return 0;
  
  // A rough estimate: 1 token ≈ 4 characters or 0.75 words for English text
  const wordCount = text.trim().split(/\s+/).length;
  const charCount = text.length;
  
  // Use both estimates and take the average
  const wordBasedEstimate = wordCount / 0.75;
  const charBasedEstimate = charCount / 4;
  
  return Math.round((wordBasedEstimate + charBasedEstimate) / 2);
};

/**
 * Fetch analytics data 
 */
export const fetchAnalytics = async (
  timeRange: 'day' | 'week' | 'month' | 'year' = 'week'
): Promise<AnalyticsResponse> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/api/analytics?time_range=${timeRange}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching analytics data:', error);
    throw error;
  }
};

/**
 * Calculate API cost for a given model and token count
 */
export const calculateApiCost = (
  model: string, 
  inputTokens: number, 
  outputTokens: number
): number => {
  // Pricing per 1K tokens (approximate values)
  const pricing: Record<string, { input: number; output: number }> = {
    'gpt-4': { input: 0.03, output: 0.06 },
    'gpt-3.5-turbo': { input: 0.0015, output: 0.002 },
    'mixtral-8x7b': { input: 0.0006, output: 0.0012 },
    'llama-3': { input: 0.0004, output: 0.0008 },
    'default': { input: 0.001, output: 0.002 } // Default fallback pricing
  };

  // Get the pricing for the specified model, or use default
  const modelPricing = pricing[model] || pricing.default;
  
  // Calculate cost
  const inputCost = (inputTokens / 1000) * modelPricing.input;
  const outputCost = (outputTokens / 1000) * modelPricing.output;
  
  return inputCost + outputCost;
};

export default {
  trackApiUsage,
  estimateTokenCount,
  fetchAnalytics,
  calculateApiCost
};