import React, { useState, useEffect } from 'react';
import { ChevronDown, FilePlus, FileSpreadsheet, FileText, Download, ThumbsUp, ThumbsDown } from 'lucide-react';
import Button from '../ui/Button';
import Card from '../ui/Card';
import SingleEntryForm from './SingleEntryForm';
import BulkGenerationForm from './BulkGenerationForm';
import GeneratedContent from './GeneratedContent';
import GenerationModelSelector from './GenerationModelSelector';
import { useTheme } from '../ThemeContext';
import axios from 'axios';

type GenerationMode = 'single' | 'bulk';

const API_BASE_URL = 'http://localhost:8000';
// Predefined prompts for the dropdown
const PREDEFINED_PROMPTS = [
  {
    id: 'world-bank-expert',
    name: 'World Bank Expert Mode',
    content: `### Instructions:
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
- Ensure strict semantic alignment between the definition and the extracted sentences.`
  },
  {
    id: 'edu-mode',
    name: 'World Bank Education',
    content: "World-Bank-Edu expert mode → Retrieve ≤ 3 exact sentences that semantically match the 'Activity Name & Definition'; prefix each with '-'. If no match, output exactly NO RELEVANT CONTEXT FOUND. No paraphrase, no inference."
  },
  {
    id: 'verbatim-lines',
    name: 'Verbatim Extraction',
    content: "Task: locate up to three verbatim lines in the supplied context that align with this activity description → {query}. Return each line as - sentence. If nothing aligns, return NO RELEVANT CONTEXT FOUND (all-caps). Do not alter wording or add commentary."
  },
  {
    id: 'policy-analyst',
    name: 'Policy Analyst',
    content: "Instruction (Edu-policy analyst): Scan {context_text}. Copy at most 3 sentences whose meaning overlaps the following activity definition. Bullet them with '-'. If zero overlap, output NO RELEVANT CONTEXT FOUND. Absolutely no re-phrasing or summarizing."
  }
];

const GenerationSection: React.FC = () => {
  // Get theme context
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();
  
  const [mode, setMode] = useState<GenerationMode>('single');
  const [generatedContent, setGeneratedContent] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-4.5');
  const [selectedPrompt, setSelectedPrompt] = useState(PREDEFINED_PROMPTS[0]);
  const [customPrompt, setCustomPrompt] = useState('');
  const [isPromptDropdownOpen, setIsPromptDropdownOpen] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);
  const [previewPrompt, setPreviewPrompt] = useState<string | null>(null);
  const [downloadData, setDownloadData] = useState<any>(null);

  const handleSingleGeneration = async (activity: string, definition: string, file: File | null, modelId: string) => {
    if (!file) {
      setGeneratedContent('Error: Please upload a PDF file for context.');
      return;
    }

    // Clear the preview prompt when generating content
    setPreviewPrompt(null);
    
    setIsGenerating(true);
    setSelectedModel(modelId); // Update the selected model state
    setProcessingStatus('Processing PDF...');

    try {
      let pdfContent = null;
      
      // Upload PDF if provided
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        
        setProcessingStatus('Extracting text from PDF...');
        const response = await axios.post(`${API_BASE_URL}/api/process-pdf`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        
        if (response.data.structured_pages) {
          // Create plain text representation
          pdfContent = response.data.structured_pages
            .map((page: any) => `${page.plain_text}`)
            .join('\n\n');
        }
      }

      if (!pdfContent) {
        setIsGenerating(false);
        setProcessingStatus(null);
        setGeneratedContent('Error: Failed to extract text from PDF. Please try again with a different file.');
        return;
      }
      
      // Get the prompt to use (either selected predefined or custom)
      const promptToUse = customPrompt.trim() ? customPrompt : selectedPrompt.content;
      
      // Generate content with RAG pipeline
      setProcessingStatus('Generating matches with RAG pipeline...');
      const generationResponse = await axios.post(`${API_BASE_URL}/api/generate`, {
        model: modelId,
        activity: activity,
        definition: definition,
        pdf_content: pdfContent,
        mode: 'single',
        prompt: promptToUse
      });
      
      setGeneratedContent(generationResponse.data.content);
      // Store data for potential CSV download
      setDownloadData(generationResponse.data);
    } catch (error) {
      console.error('Error generating content:', error);
      setGeneratedContent('Error: Failed to generate content. Please try again.');
    } finally {
      setIsGenerating(false);
      setProcessingStatus(null);
    }
  };

  const handleBulkGeneration = async (file: File, pdfFile: File | null, queryLimit: number, modelId: string) => {
    if (!pdfFile) {
      setGeneratedContent('Error: Please upload a PDF file for context.');
      return;
    }

    // Clear the preview prompt when generating content
    setPreviewPrompt(null);
    
    setIsGenerating(true);
    setSelectedModel(modelId); // Update the selected model state
    setProcessingStatus('Processing files...');
    
    try {
      let pdfContent = null;
      
      // Upload PDF if provided
      if (pdfFile) {
        const formData = new FormData();
        formData.append('file', pdfFile);
        
        setProcessingStatus('Extracting text from PDF...');
        const response = await axios.post(`${API_BASE_URL}/api/process-pdf`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        
        if (response.data.structured_pages) {
          // Create plain text representation
          pdfContent = response.data.structured_pages
            .map((page: any) => `${page.plain_text}`)
            .join('\n\n');
        }
      }

      if (!pdfContent) {
        setIsGenerating(false);
        setProcessingStatus(null);
        setGeneratedContent('Error: Failed to extract text from PDF. Please try again with a different file.');
        return;
      }
      
      // Get the prompt to use (either selected predefined or custom)
      const promptToUse = customPrompt.trim() ? customPrompt : selectedPrompt.content;
      
      // Upload Excel file for bulk processing
      setProcessingStatus(`Processing ${queryLimit === 0 ? 'all' : queryLimit} entries through RAG pipeline...`);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('query_limit', queryLimit.toString());
      
      if (pdfContent) {
        formData.append('pdf_content', pdfContent);
      }
      
      formData.append('model', modelId);
      formData.append('prompt', promptToUse);
      
      const response = await axios.post(`${API_BASE_URL}/api/generate-bulk`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setGeneratedContent(response.data.content || `Processed ${file.name} successfully. Generated content for ${queryLimit === 0 ? 'all' : queryLimit} entries.`);
      // Store data for CSV download
      setDownloadData(response.data);
    } catch (error) {
      console.error('Error processing bulk generation:', error);
      setGeneratedContent(`Error: Failed to process ${file.name}. Please check your file and try again.`);
    } finally {
      setIsGenerating(false);
      setProcessingStatus(null);
    }
  };

  const handleModeChange = (newMode: GenerationMode) => {
    setMode(newMode);
    setGeneratedContent(null);
    setPreviewPrompt(null);
  };

  const handleSelectPrompt = (prompt: typeof PREDEFINED_PROMPTS[0]) => {
    setSelectedPrompt(prompt);
    setIsPromptDropdownOpen(false);
  };

  const handlePreviewPrompt = () => {
    // Show the prompt in the preview area (sidebar), not in the main content area
    const promptToUse = customPrompt.trim() ? customPrompt : selectedPrompt.content;
    setPreviewPrompt(promptToUse);
  };

  const handleDownload = () => {
    if (!generatedContent) return;
    
    // Create a content type and format for download
    let contentType = 'text/html';
    let fileExtension = '.html';
    let content = generatedContent;
    
    // For CSV downloads in bulk mode
    if (mode === 'bulk' && downloadData && downloadData.results) {
      // Convert results to CSV
      const csvContent = convertResultsToCSV(downloadData.results);
      content = csvContent;
      contentType = 'text/csv';
      fileExtension = '.csv';
    }
    
    // Create a Blob with the content
    const blob = new Blob([content], { type: contentType });
    
    // Create a URL for the Blob
    const url = URL.createObjectURL(blob);
    
    // Create a temporary anchor element
    const a = document.createElement('a');
    a.href = url;
    
    // Set the filename based on the mode
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    a.download = mode === 'single' 
      ? `activity-match-${timestamp}${fileExtension}` 
      : `bulk-matches-${timestamp}${fileExtension}`;
    
    // Append the anchor to the body
    document.body.appendChild(a);
    
    // Trigger a click on the anchor
    a.click();
    
    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // Helper function to convert results to CSV
  const convertResultsToCSV = (results: any[]) => {
    if (!results || results.length === 0) return '';
    
    // Get headers from the first result
    const headers = Object.keys(results[0]).filter(key => key !== 'status');
    
    // Create CSV header row
    let csv = headers.join(',') + '\n';
    
    // Add each row of data
    results.forEach(result => {
      const row = headers.map(header => {
        // Escape commas and quotes in the content
        let value = result[header] || '';
        if (typeof value === 'string') {
          value = value.replace(/"/g, '""');
          // If the value contains commas, quotes or newlines, wrap it in quotes
          if (value.includes(',') || value.includes('"') || value.includes('\n')) {
            value = `"${value}"`;
          }
        }
        return value;
      }).join(',');
      
      csv += row + '\n';
    });
    
    return csv;
  };

  // Fix for ModelSelector to prevent auto-generation
  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
    // Don't trigger generation here
  };

  return (
    <div className={`w-full h-full flex ${theme.background}`}>
      {/* Main content area */}
      <div className={`flex-1 p-6 overflow-y-auto ${theme.background}`}>
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h1 className={`text-3xl font-bold mb-2 ${theme.title}`}>
              Content Extraction
            </h1>
            <p className={theme.secondaryText}>
              Extract context-matched content based on your input and uploaded PDFs
            </p>
          </div>

          <div className="mb-6">
            <div className={`flex space-x-2 border-b ${theme.border}`}>
              <button
                className={`px-4 py-2 font-medium text-sm border-b-2 ${
                  mode === 'single'
                    ? `border-amber-500 ${currentTheme === 'futuristic' ? 'text-blue-300' : currentTheme === 'dark' ? 'text-amber-300' : 'text-amber-600'}`
                    : `border-transparent ${theme.secondaryText} hover:${theme.text} hover:border-gray-300`
                }`}
                onClick={() => handleModeChange('single')}
              >
                Single Entry
              </button>
              <button
                className={`px-4 py-2 font-medium text-sm border-b-2 ${
                  mode === 'bulk'
                    ? `border-amber-500 ${currentTheme === 'futuristic' ? 'text-blue-300' : currentTheme === 'dark' ? 'text-amber-300' : 'text-amber-600'}`
                    : `border-transparent ${theme.secondaryText} hover:${theme.text} hover:border-gray-300`
                }`}
                onClick={() => handleModeChange('bulk')}
              >
                Bulk Generation
              </button>
            </div>
          </div>

          {processingStatus && (
            <div className={`mb-4 p-3 ${
              currentTheme === 'futuristic' 
                ? 'bg-blue-900/30 border border-blue-500/30' 
                : currentTheme === 'dark'
                  ? 'bg-blue-900/30 border border-blue-700/50'
                  : 'bg-blue-50 border border-blue-200'
            } rounded-md`}>
              <p className={`${
                currentTheme === 'futuristic' 
                  ? 'text-blue-300' 
                  : currentTheme === 'dark'
                    ? 'text-blue-300'
                    : 'text-blue-700'
              } flex items-center`}>
                <svg className="animate-spin h-4 w-4 mr-2 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                {processingStatus}
              </p>
            </div>
          )}

          {/* Modified layout structure to place the form above and content below */}
          {mode === 'single' ? (
            // Single mode layout - keeps the side-by-side layout
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <div className={`${theme.card} p-6 rounded-lg ${
                  currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'
                }`}>
                  <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                    Generate Single Entry
                  </h2>
                  <SingleEntryForm onGenerate={handleSingleGeneration} />
                </div>
              </div>

              <div>
                {generatedContent ? (
                  <div>
                    <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                      Extracted Content
                    </h2>
                    <GeneratedContent 
                      content={generatedContent} 
                      mode={mode} 
                      onDownload={handleDownload}
                      theme={theme}
                    />
                  </div>
                ) : (
                  <div>
                    <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                      Extracted Content
                    </h2>
                    <div className={`h-full flex items-center justify-center ${theme.cardHighlight} border ${theme.border} border-dashed rounded-lg p-8`}>
                      <div className="text-center">
                        <div className={`mx-auto h-16 w-16 rounded-full ${
                          currentTheme === 'futuristic' 
                            ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20' 
                            : currentTheme === 'dark'
                              ? 'bg-amber-800/20'
                              : 'bg-amber-100'
                        } flex items-center justify-center mb-4`}>
                          <FilePlus className={`h-8 w-8 ${
                            currentTheme === 'futuristic' 
                              ? 'text-blue-300' 
                              : currentTheme === 'dark'
                                ? 'text-amber-300'
                                : 'text-amber-600'
                          }`} />
                        </div>
                        <h3 className={`text-lg font-medium mb-1 ${theme.text}`}>No Content Generated Yet</h3>
                        <p className={`text-sm ${theme.secondaryText} max-w-xs mx-auto`}>
                          Fill in the activity details and generate to see content here
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            // Bulk mode layout - Form above, generated content below
            <div className="flex flex-col gap-8">
              <div>
                <div className={`${theme.card} p-6 rounded-lg ${
                  currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'
                }`}>
                  <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                    Bulk Generation
                  </h2>
                  <BulkGenerationForm onGenerate={handleBulkGeneration} theme={theme} />
                </div>
              </div>

              {generatedContent ? (
                <div className="w-full">
                  <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                    Extracted Content
                  </h2>
                  <GeneratedContent 
                    content={generatedContent} 
                    mode={mode} 
                    onDownload={handleDownload}
                    theme={theme}
                  />
                </div>
              ) : (
                <div>
                  <h2 className={`text-lg font-semibold mb-4 ${theme.title}`}>
                    Extracted Content
                  </h2>
                  <div className={`h-full flex items-center justify-center ${theme.cardHighlight} border ${theme.border} border-dashed rounded-lg p-8`}>
                    <div className="text-center">
                      <div className={`mx-auto h-16 w-16 rounded-full ${
                        currentTheme === 'futuristic' 
                          ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20' 
                          : currentTheme === 'dark'
                            ? 'bg-amber-800/20'
                            : 'bg-amber-100'
                      } flex items-center justify-center mb-4`}>
                        <FileSpreadsheet className={`h-8 w-8 ${
                          currentTheme === 'futuristic' 
                            ? 'text-blue-300' 
                            : currentTheme === 'dark'
                              ? 'text-amber-300'
                              : 'text-amber-600'
                        }`} />
                      </div>
                      <h3 className={`text-lg font-medium mb-1 ${theme.text}`}>No Content Extracted Yet</h3>
                      <p className={`text-sm ${theme.secondaryText} max-w-xs mx-auto`}>
                        Upload your Excel file to generate bulk content
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right sidebar for prompts - with improved spacing */}
      <div className={`hidden lg:block w-80 border-l ${theme.border} ${theme.card} p-4 overflow-y-auto`}>
        {/* Added top padding for better spacing */}
        <div className="pt-12">
          <h3 className={`text-lg font-semibold mb-4 ${theme.title}`}>Generation Prompts</h3>
          
          {/* Predefined Prompts Dropdown */}
          <div className="mb-5">
            <h4 className={`text-sm font-medium ${theme.text} mb-2`}>Select a Predefined Prompt</h4>
            <div className="relative">
              <button
                onClick={() => setIsPromptDropdownOpen(!isPromptDropdownOpen)}
                className={`flex items-center justify-between w-full px-3 py-2 border ${theme.border} rounded-md ${
                  currentTheme === 'futuristic'
                    ? 'focus:ring-blue-500 focus:border-blue-500'
                    : currentTheme === 'dark'
                      ? 'focus:ring-amber-500 focus:border-amber-500'
                      : 'focus:ring-amber-500 focus:border-amber-500'
                } ${theme.cardHighlight} text-sm ${theme.text}`}
              >
                <span>{selectedPrompt.name}</span>
                <ChevronDown className={`h-4 w-4 ${theme.secondaryText}`} />
              </button>
              
              {isPromptDropdownOpen && (
                <div className={`absolute z-10 mt-1 w-full ${theme.card} shadow-lg rounded-md ring-1 ring-black ring-opacity-5`}>
                  <div className="py-1">
                    {PREDEFINED_PROMPTS.map((prompt) => (
                      <button
                        key={prompt.id}
                        onClick={() => handleSelectPrompt(prompt)}
                        className={`block w-full text-left px-4 py-2 text-sm ${
                          selectedPrompt.id === prompt.id
                            ? currentTheme === 'futuristic' 
                              ? 'bg-blue-900/30 text-blue-300' 
                              : currentTheme === 'dark'
                                ? 'bg-amber-900/50 text-amber-300'
                                : 'bg-amber-50 text-amber-600'
                            : `${theme.text} hover:${theme.cardHighlight}`
                        }`}
                      >
                        {prompt.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Custom Prompt Input */}
          <div className="mb-5">
            <h4 className={`text-sm font-medium ${theme.text} mb-2`}>Or Use a Custom Prompt</h4>
            <textarea
              placeholder="Enter a custom prompt (will override selected prompt above)"
              className={`w-full px-3 py-2 border ${theme.border} rounded-md ${
                currentTheme === 'futuristic'
                  ? 'focus:ring-blue-500 focus:border-blue-500'
                  : 'focus:ring-amber-500 focus:border-amber-500'
              } ${theme.cardHighlight} text-sm ${theme.text} placeholder-${
                currentTheme === 'futuristic'
                  ? 'blue-400/50'
                  : currentTheme === 'dark'
                    ? 'gray-500'
                    : 'gray-400'
              }`}
              rows={6}
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
            />
            <p className={`mt-1 text-xs ${theme.secondaryText}`}>
              Use {'{query}'} for activity name & definition and {'{context_text}'} for PDF content
            </p>
          </div>
          
          {/* RAG Pipeline Information */}
          <div className={`mb-5 p-3 ${
            currentTheme === 'futuristic' 
              ? 'bg-blue-900/30 border border-blue-500/30' 
              : currentTheme === 'dark'
                ? 'bg-amber-900/30 border border-amber-700/30'
                : 'bg-amber-50 border border-amber-200'
          } rounded-lg`}>
            <h4 className={`text-sm font-medium ${
              currentTheme === 'futuristic' 
                ? 'text-blue-300' 
                : currentTheme === 'dark'
                  ? 'text-amber-300'
                  : 'text-amber-800'
            } mb-2`}>RAG Pipeline Process</h4>
            <ol className={`text-xs ${
              currentTheme === 'futuristic' 
                ? 'text-blue-300/80' 
                : currentTheme === 'dark'
                  ? 'text-amber-300/80'
                  : 'text-amber-700'
            } list-decimal pl-4 space-y-1`}>
              <li>Extract text from uploaded PDF</li>
              <li>Create embeddings and chunked content</li>
              <li>Find most relevant chunks for each entry</li>
              <li>Apply prompt to determine sentence matches</li>
              <li>Return formatted results in table format</li>
            </ol>
          </div>
          
          {/* Preview Prompt Button */}
          <button 
            onClick={handlePreviewPrompt}
            className={`w-full mb-5 px-4 py-2 border ${theme.border} rounded-md ${
              currentTheme === 'futuristic' 
                ? 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20' 
                : currentTheme === 'dark'
                  ? 'bg-amber-700 hover:bg-amber-800 text-white'
                  : 'bg-white hover:bg-gray-50 text-gray-700'
            }`}
          >
            Preview Selected Prompt
          </button>
          
          {/* Preview Prompt Content Area */}
          {previewPrompt && (
            <div className={`p-3 border ${theme.border} rounded-lg ${theme.cardHighlight} mt-2`}>
              <h4 className={`text-xs font-medium ${theme.text} mb-2`}>Prompt Preview:</h4>
              <div className={`text-xs ${theme.secondaryText} whitespace-pre-wrap max-h-[300px] overflow-y-auto`}>
                {previewPrompt}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GenerationSection;