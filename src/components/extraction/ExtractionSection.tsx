import React, { useState, useEffect } from 'react';
import { FileUp, Link, Loader2, Search, Upload, Trash2 } from 'lucide-react';
import Button from '../ui/Button';
import Card from '../ui/Card';
import FileUploadModal from './FileUploadModal';
import { useTheme } from '../ThemeContext';
import axios from 'axios';

// Define the API base URL
const API_BASE_URL = 'http://localhost:8000';

// Define interface for search results
interface SearchResult {
  text: string;
  score: number;
  page_number: number;
  markdown?: string;
}

interface StructuredPage {
  page_number: number;
  markdown: string;
  plain_text: string;
}

const ExtractionSection: React.FC = () => {
  // Get theme from context
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();
  
  // Use session storage key to detect when page is refreshed or reopened
  const [sessionKey, setSessionKey] = useState<string>(() => {
    return sessionStorage.getItem('pdfExtractionSessionKey') || '';
  });
  
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [extractedText, setExtractedText] = useState<string | null>(null);
  const [markdownContent, setMarkdownContent] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [structuredPages, setStructuredPages] = useState<StructuredPage[]>([]);
  
  // Set to false to hide debug info in production
  const showDebugInfo = false;

  // Initialize session tracking
  useEffect(() => {
    // Check if we have a session key (returning user in same session)
    const currentSessionKey = sessionStorage.getItem('pdfExtractionSessionKey');
    
    // If page is refreshed or new session, clear stored data
    if (!currentSessionKey) {
      // Generate a new session key
      const newSessionKey = Date.now().toString();
      sessionStorage.setItem('pdfExtractionSessionKey', newSessionKey);
      setSessionKey(newSessionKey);
      
      // Clear localStorage on first load of a new session
      localStorage.removeItem('pdfMarkdownContent');
      localStorage.removeItem('pdfExtractedText');
      localStorage.removeItem('pdfStructuredPages');
      
      // Reset states
      setExtractedText(null);
      setMarkdownContent(null);
      setStructuredPages([]);
      setSearchResults([]);
    } else {
      // Retrieve stored PDF data when component mounts if we have a valid session
      const storedMarkdown = localStorage.getItem('pdfMarkdownContent');
      const storedText = localStorage.getItem('pdfExtractedText');
      const storedPages = localStorage.getItem('pdfStructuredPages');
      
      if (storedMarkdown && storedText) {
        setMarkdownContent(storedMarkdown);
        setExtractedText(storedText);
        if (storedPages) {
          try {
            setStructuredPages(JSON.parse(storedPages));
          } catch (e) {
            console.error("Failed to parse stored pages:", e);
          }
        }
      }
    }
  }, []);

  // When PDF content is available, store it in localStorage
  useEffect(() => {
    if (sessionKey && markdownContent) {
      localStorage.setItem('pdfMarkdownContent', markdownContent);
      localStorage.setItem('pdfExtractedText', extractedText || '');
      localStorage.setItem('pdfStructuredPages', JSON.stringify(structuredPages));
    }
  }, [markdownContent, extractedText, structuredPages, sessionKey]);

  const handleUpload = () => {
    setShowUploadModal(true);
  };

  const handleFileUpload = async (file: File | null, url: string | null) => {
    setShowUploadModal(false);
    if (!file && !url) return;
    
    try {
      setError(null);
      setIsProcessing(true);
      setExtractedText(null);
      setMarkdownContent(null);
      setSearchResults([]);
      setStructuredPages([]);
      console.log(`Processing ${file ? 'file' : 'URL'}: ${file?.name || url}`);
      
      let response;
      
      if (file) {
        // Handle file upload
        const formData = new FormData();
        formData.append('file', file);
        
        console.log('Sending file to API...');
        response = await axios.post(`${API_BASE_URL}/api/process-pdf`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        console.log('File upload response:', response.data);
      } else if (url) {
        // Handle URL processing
        console.log('Sending URL to API...');
        response = await axios.post(`${API_BASE_URL}/api/process-pdf-url`, { url });
        console.log('URL processing response:', response.data);
      }
      
      if (response && response.data.structured_pages && response.data.structured_pages.length > 0) {
        // Sort pages by page number
        const pages = response.data.structured_pages as StructuredPage[];
        pages.sort((a, b) => a.page_number - b.page_number);
        
        // Store structured pages for search result processing
        setStructuredPages(pages);
        
        // Combine all pages' content
        const combinedPlainText = pages.map(page => 
          `${page.plain_text}\n\n---\nPage ${page.page_number + 1}\n\n`
        ).join('\n');
        
        // Combine all pages' markdown
        const combinedMarkdown = pages.map(page => 
          `${page.markdown}\n\n---\n**Page ${page.page_number + 1}**\n\n`
        ).join('\n');
        
        // Set both plain text and markdown content
        setExtractedText(combinedPlainText);
        setMarkdownContent(combinedMarkdown);
        
        console.log(`Set combined content from ${pages.length} pages`);
      } else {
        console.log('No pages found in response:', response?.data);
        setError('No content was extracted from the PDF');
      }
    } catch (err: any) {
      console.error('Error processing PDF:', err);
      if (err.response) {
        console.error('Error response:', err.response.data);
        setError(`Server error: ${err.response.data.detail || err.response.statusText}`);
      } else if (err.request) {
        console.error('No response received:', err.request);
        setError('No response from server. Check if the server is running.');
      } else {
        setError(`Error: ${err.message || 'Unknown error'}`);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim() || !extractedText || structuredPages.length === 0) return;
    
    try {
      setIsSearching(true);
      console.log('Searching for:', searchQuery);
      
      const response = await axios.post(`${API_BASE_URL}/api/search-pdf`, { query: searchQuery });
      console.log('Search response:', response.data);
      
      if (response.data.results) {
        setSearchResults(response.data.results);
      } else {
        setSearchResults([]);
      }
    } catch (err: any) {
      console.error('Error searching PDF:', err);
      if (err.response) {
        console.error('Error response:', err.response.data);
      }
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  // Function to clear the current extraction
  const clearExtraction = () => {
    // Clear states
    setExtractedText(null);
    setMarkdownContent(null);
    setStructuredPages([]);
    setSearchResults([]);
    setSearchQuery('');
    setError(null);
    
    // Clear localStorage
    localStorage.removeItem('pdfMarkdownContent');
    localStorage.removeItem('pdfExtractedText');
    localStorage.removeItem('pdfStructuredPages');
  };

  // Function to render markdown content, preserving tables
  const renderMarkdown = () => {
    if (!markdownContent) return { __html: extractedText || '' };
    
    // Basic markdown rendering
    let html = markdownContent;
    
    // Process tables before other elements
    // Match markdown tables (rows starting with |)
    const tableRegex = /(\|[^\n]*\|\n)((?:\|[^\n]*\|\n)+)/g;
    html = html.replace(tableRegex, (match) => {
      // Convert the matched table to HTML
      const rows = match.split('\n').filter(row => row.trim().startsWith('|'));
      
      // Create HTML table with themed styling
      let tableHtml = `<table style="border-collapse: collapse; width: 100%; margin: 10px 0; ${
        currentTheme === 'dark' ? 'border-color: #4B5563;' : 
        currentTheme === 'futuristic' ? 'border-color: rgba(59, 130, 246, 0.3);' : 
        'border-color: #E5E7EB;'
      }">`;
      
      // Process header row
      if (rows.length > 0) {
        const headerCells = rows[0].split('|').filter(cell => cell.trim() !== '');
        tableHtml += '<thead><tr>';
        headerCells.forEach(cell => {
          tableHtml += `<th style="border: 1px solid ${
            currentTheme === 'dark' ? '#4B5563' : 
            currentTheme === 'futuristic' ? 'rgba(59, 130, 246, 0.3)' : 
            '#ddd'
          }; padding: 8px; text-align: left; ${
            currentTheme === 'dark' ? 'background-color: #374151; color: #F9FAFB;' : 
            currentTheme === 'futuristic' ? 'background-color: rgba(59, 130, 246, 0.1); color: #F9FAFB;' : 
            'background-color: #f9fafb; color: #111827;'
          }">${cell.trim()}</th>`;
        });
        tableHtml += '</tr></thead>';
      }
      
      // Process data rows
      if (rows.length > 1) {
        tableHtml += '<tbody>';
        for (let i = 1; i < rows.length; i++) {
          const dataCells = rows[i].split('|').filter(cell => cell.trim() !== '');
          tableHtml += '<tr>';
          dataCells.forEach(cell => {
            tableHtml += `<td style="border: 1px solid ${
              currentTheme === 'dark' ? '#4B5563' : 
              currentTheme === 'futuristic' ? 'rgba(59, 130, 246, 0.3)' : 
              '#ddd'
            }; padding: 8px; ${
              currentTheme === 'dark' ? 'color: #E5E7EB;' : 
              currentTheme === 'futuristic' ? 'color: #E5E7EB;' : 
              'color: #374151;'
            }">${cell.trim()}</td>`;
          });
          tableHtml += '</tr>';
        }
        tableHtml += '</tbody>';
      }
      
      tableHtml += '</table>';
      return tableHtml;
    });
    
    // Theme-aware text colors
    const headingColor = currentTheme === 'dark' ? '#F9FAFB' : 
                         currentTheme === 'futuristic' ? '#F9FAFB' : 
                         '#111827';
    
    const linkColor = currentTheme === 'dark' ? '#93C5FD' : 
                      currentTheme === 'futuristic' ? '#93C5FD' : 
                      '#4F46E5';
    
    const codeBackground = currentTheme === 'dark' ? '#374151' : 
                           currentTheme === 'futuristic' ? 'rgba(59, 130, 246, 0.1)' : 
                           '#F3F4F6';
    
    const codeColor = currentTheme === 'dark' ? '#E5E7EB' : 
                      currentTheme === 'futuristic' ? '#E5E7EB' : 
                      '#1F2937';
    
    const hrColor = currentTheme === 'dark' ? '#4B5563' : 
                    currentTheme === 'futuristic' ? 'rgba(59, 130, 246, 0.3)' : 
                    '#E5E7EB';
    
    const pageNumberColor = currentTheme === 'dark' ? '#93C5FD' : 
                            currentTheme === 'futuristic' ? 'rgba(147, 197, 253, 0.8)' : 
                            '#4F46E5';
    
    // Convert headings
    html = html.replace(/# (.*?)(\n|$)/g, `<h1 style="font-size: 1.5rem; font-weight: bold; margin-top: 1rem; margin-bottom: 0.5rem; color: ${headingColor};">$1</h1>$2`);
    html = html.replace(/## (.*?)(\n|$)/g, `<h2 style="font-size: 1.25rem; font-weight: bold; margin-top: 1rem; margin-bottom: 0.5rem; color: ${headingColor};">$1</h2>$2`);
    html = html.replace(/### (.*?)(\n|$)/g, `<h3 style="font-size: 1.1rem; font-weight: bold; margin-top: 1rem; margin-bottom: 0.5rem; color: ${headingColor};">$1</h3>$2`);
    
    // Convert bold and italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert links
    html = html.replace(/\[(.*?)\]\((.*?)\)/g, `<a href="$2" style="color: ${linkColor}; text-decoration: underline;" target="_blank">$1</a>`);
    
    // Convert code blocks
    html = html.replace(/```(.*?)```/gs, `<pre style="background-color: ${codeBackground}; padding: 10px; border-radius: 5px; overflow-x: auto;"><code style="color: ${codeColor};">$1</code></pre>`);
    
    // Convert inline code
    html = html.replace(/`(.*?)`/g, `<code style="background-color: ${codeBackground}; padding: 2px 4px; border-radius: 3px; color: ${codeColor};">$1</code>`);
    
    // Convert unordered lists
    const listRegex = /(?:^|\n)((?:- .*(?:\n|$))+)/g;
    html = html.replace(listRegex, (match) => {
      const items = match.split('\n').filter(item => item.trim().startsWith('- '));
      let listHtml = '<ul style="list-style-type: disc; padding-left: 20px; margin: 10px 0;">';
      items.forEach(item => {
        const content = item.replace(/^- /, '');
        listHtml += `<li style="margin-bottom: 5px;">${content}</li>`;
      });
      listHtml += '</ul>';
      return listHtml;
    });
    
    // Convert page separators
    html = html.replace(/---\n\*\*Page (.*?)\*\*/g, `<hr style="margin: 1rem 0; border: 0; border-top: 1px solid ${hrColor};"/><div style="font-weight: bold; color: ${pageNumberColor}; margin: 0.5rem 0;">Page $1</div>`);
    
    // Convert newlines to <br>
    html = html.replace(/\n/g, '<br/>');

    return { __html: html };
  };

  // Function to render markdown for search results
  const renderSearchResultMarkdown = (markdown: string) => {
    if (!markdown) return { __html: "" };
    
    // Use the same rendering function but with limited content
    let html = markdown;
    
    // Theme-aware colors
    const codeBackground = currentTheme === 'dark' ? '#374151' : 
                          currentTheme === 'futuristic' ? 'rgba(59, 130, 246, 0.1)' : 
                          '#F3F4F6';
    
    const linkColor = currentTheme === 'dark' ? '#93C5FD' : 
                      currentTheme === 'futuristic' ? '#93C5FD' : 
                      '#4F46E5';
    
    // Process tables with themed styles
    const tableRegex = /(\|[^\n]*\|\n)((?:\|[^\n]*\|\n)+)/g;
    html = html.replace(tableRegex, (match) => {
      const rows = match.split('\n').filter(row => row.trim().startsWith('|'));
      
      let tableHtml = `<table style="border-collapse: collapse; width: 100%; margin: 10px 0; ${
        currentTheme === 'dark' ? 'border-color: #4B5563;' : 
        currentTheme === 'futuristic' ? 'border-color: rgba(59, 130, 246, 0.3);' : 
        'border-color: #E5E7EB;'
      }">`;
      
      if (rows.length > 0) {
        const headerCells = rows[0].split('|').filter(cell => cell.trim() !== '');
        tableHtml += '<thead><tr>';
        headerCells.forEach(cell => {
          tableHtml += `<th style="border: 1px solid ${
            currentTheme === 'dark' ? '#4B5563' : 
            currentTheme === 'futuristic' ? 'rgba(59, 130, 246, 0.3)' : 
            '#ddd'
          }; padding: 8px; text-align: left; ${
            currentTheme === 'dark' ? 'background-color: #374151; color: #F9FAFB;' : 
            currentTheme === 'futuristic' ? 'background-color: rgba(59, 130, 246, 0.1); color: #F9FAFB;' : 
            'background-color: #f9fafb; color: #111827;'
          }">${cell.trim()}</th>`;
        });
        tableHtml += '</tr></thead>';
      }
      
      if (rows.length > 1) {
        tableHtml += '<tbody>';
        // Limit to at most 3 data rows for search results
        const maxRows = Math.min(rows.length, 4);
        for (let i = 1; i < maxRows; i++) {
          const dataCells = rows[i].split('|').filter(cell => cell.trim() !== '');
          tableHtml += '<tr>';
          dataCells.forEach(cell => {
            tableHtml += `<td style="border: 1px solid ${
              currentTheme === 'dark' ? '#4B5563' : 
              currentTheme === 'futuristic' ? 'rgba(59, 130, 246, 0.3)' : 
              '#ddd'
            }; padding: 8px; ${
              currentTheme === 'dark' ? 'color: #E5E7EB;' : 
              currentTheme === 'futuristic' ? 'color: #E5E7EB;' : 
              'color: #374151;'
            }">${cell.trim()}</td>`;
          });
          tableHtml += '</tr>';
        }
        tableHtml += '</tbody>';
      }
      
      tableHtml += '</table>';
      return tableHtml;
    });
    
    // Convert headings with theme colors
    const headingColor = currentTheme === 'dark' ? '#F9FAFB' : 
                         currentTheme === 'futuristic' ? '#F9FAFB' : 
                         '#111827';
    
    html = html.replace(/# (.*?)(\n|$)/g, `<h1 style="font-size: 1.25rem; font-weight: bold; color: ${headingColor};">$1</h1>$2`);
    html = html.replace(/## (.*?)(\n|$)/g, `<h2 style="font-size: 1.125rem; font-weight: bold; color: ${headingColor};">$1</h2>$2`);
    html = html.replace(/### (.*?)(\n|$)/g, `<h3 style="font-size: 1rem; font-weight: bold; color: ${headingColor};">$1</h3>$2`);
    
    // Convert bold and italic
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Convert links
    html = html.replace(/\[(.*?)\]\((.*?)\)/g, `<a href="$2" style="color: ${linkColor}; text-decoration: underline;" target="_blank">$1</a>`);
    
    // Convert newlines to <br>
    html = html.replace(/\n/g, '<br/>');
    
    return { __html: html };
  };

  return (
    // Modified container to eliminate gaps
    <div className={`w-full h-full ${theme.background}`}>
      <div className="px-6 py-6">
        <h1 className={`text-3xl font-bold mb-2 ${theme.title}`}>PDF Extraction</h1>
        <p className={theme.secondaryText}>Upload a PDF to extract and analyze its content</p>
      </div>

      {!extractedText && !isProcessing ? (
        <div className={`mx-6 flex flex-col items-center justify-center ${theme.cardHighlight} border-2 border-dashed ${theme.border} rounded-lg p-12 text-center`}>
          <div className={`w-16 h-16 mb-4 rounded-full ${theme.iconBg} flex items-center justify-center`}>
            <FileUp className={`w-8 h-8 ${theme.iconColor}`} />
          </div>
          <h3 className={`text-xl font-semibold mb-2 ${theme.text}`}>Upload your PDF</h3>
          <p className={`${theme.secondaryText} mb-4 max-w-md`}>
            Upload a PDF file or paste a link to extract its contents
          </p>
          <Button 
            onClick={handleUpload}
            icon={<Upload className="w-4 h-4" />}
            size="lg"
            className={currentTheme === 'futuristic' ? theme.button + ' text-white shadow-lg shadow-blue-500/20' : ''}
          >
            Select PDF
          </Button>
        </div>
      ) : (
        <div className="flex flex-col space-y-6 px-6">
          {/* Extracted Content - Full Width */}
          <div className={`${theme.card} p-6 rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'}`}>
            <div className="flex items-center justify-between mb-4">
              <h3 className={`text-lg font-semibold ${theme.text}`}>Extracted Content</h3>
              <div className="flex gap-2">
                {/* Fixed button styling for dark/futuristic modes */}
                <Button
                  variant={currentTheme === 'light' ? 'outline' : 'filled'}
                  size="sm"
                  onClick={clearExtraction}
                  className={currentTheme === 'light' ? 
                    `border ${theme.border} ${theme.text}` : 
                    currentTheme === 'futuristic' ? 
                      'bg-blue-900/30 border border-blue-500/30 text-blue-300 hover:bg-blue-800/40' : 
                      'bg-gray-700 text-gray-200 hover:bg-gray-600'
                  }
                  icon={<Trash2 className="w-4 h-4" />}
                >
                  Clear PDF
                </Button>
                <Button
                  variant={currentTheme === 'light' ? 'outline' : 'filled'}
                  size="sm"
                  onClick={handleUpload}
                  className={currentTheme === 'light' ? 
                    `border ${theme.border} ${theme.text}` : 
                    currentTheme === 'futuristic' ? 
                      'bg-blue-900/30 border border-blue-500/30 text-blue-300 hover:bg-blue-800/40' : 
                      'bg-gray-700 text-gray-200 hover:bg-gray-600'
                  }
                >
                  Change PDF
                </Button>
              </div>
            </div>
            
            {isProcessing ? (
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 className={`w-12 h-12 ${theme.iconColor} animate-spin mb-4`} />
                <p className={`${theme.text} font-medium`}>Processing PDF with Mistral OCR...</p>
              </div>
            ) : error ? (
              <div className={`${currentTheme === 'futuristic' ? 'bg-red-900/30 text-red-300' : currentTheme === 'dark' ? 'bg-red-900/50 text-red-300' : 'bg-red-50 text-red-700'} p-6 rounded-lg`}>
                <p className="font-medium mb-2">Error</p>
                <p>{error}</p>
              </div>
            ) : (
              <div className={`${theme.cardHighlight} rounded-lg p-4 max-h-[50vh] overflow-y-auto ${theme.text}`}>
                <div dangerouslySetInnerHTML={renderMarkdown()} />
              </div>
            )}
          </div>
          
          {/* Section Extraction - Full Width */}
          <div className={`${theme.card} p-6 rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'}`}>
            <h3 className={`text-lg font-semibold mb-4 ${theme.text}`}>Section Extraction</h3>
            <div className="flex mb-4">
              <input
                type="text"
                placeholder="Search for specific sections..."
                className={`flex-1 px-4 py-2 rounded-l-lg border ${theme.border} ${theme.cardHighlight} ${theme.text} focus:ring-2 ${
                  currentTheme === 'futuristic' ? 'focus:ring-blue-500 placeholder-blue-300/50' : 
                  currentTheme === 'dark' ? 'focus:ring-gray-500 placeholder-gray-500' : 
                  'focus:ring-indigo-500 placeholder-gray-400'
                } focus:border-transparent focus:outline-none`}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isProcessing || !extractedText}
              />
              <Button
                className={`rounded-l-none ${
                  currentTheme === 'futuristic' ? 
                    'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg shadow-blue-500/20' : 
                    currentTheme === 'dark' ? 
                      'bg-indigo-600 hover:bg-indigo-700 text-white' : 
                      'bg-indigo-600 hover:bg-indigo-700 text-white'
                }`}
                onClick={handleSearch}
                disabled={isProcessing || !extractedText || !searchQuery.trim() || isSearching}
                icon={isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </Button>
            </div>

            {searchResults.length > 0 ? (
              <div className="space-y-4">
                <h4 className={`font-medium text-sm ${theme.secondaryText}`}>Top Matches:</h4>
                {searchResults.map((result, index) => {
                  // Ensure page number is displayed correctly
                  const displayPageNumber = typeof result.page_number === 'number' ? 
                    result.page_number + 1 : 
                    result.page_number;
                  
                  return (
                    <div 
                      key={index} 
                      className={`${theme.card} border ${theme.border} rounded-lg p-4 ${currentTheme === 'futuristic' ? 'shadow-md shadow-blue-900/10' : 'shadow-sm'
                    }`}
                  >
                    <div className="flex justify-between items-center mb-3">
                      <span className={`text-sm font-semibold ${theme.secondaryText}`}>Page {displayPageNumber}</span>
                      <span className={`text-xs px-2 py-1 ${
                        currentTheme === 'futuristic' 
                          ? 'bg-blue-900/30 text-blue-300' 
                          : currentTheme === 'dark'
                            ? 'bg-indigo-900 text-indigo-300' 
                            : 'bg-indigo-100 text-indigo-800'
                          } rounded-full`}>
                          {(result.score * 100).toFixed(2)}% match
                        </span>
                      </div>
                      
                      {result.markdown ? (
                        <div 
                          className={`text-sm ${theme.text}`} 
                          dangerouslySetInnerHTML={renderSearchResultMarkdown(result.markdown)} 
                        />
                      ) : (
                        <p className={`text-sm ${theme.text}`}>{result.text}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : extractedText && searchQuery.trim() && !isSearching ? (
              <div className={`text-center py-8 ${theme.secondaryText}`}>
                <p>No matching sections found</p>
              </div>
            ) : null}
          </div>
        </div>
      )}

      {showUploadModal && (
        <FileUploadModal 
          onClose={() => setShowUploadModal(false)} 
          onUpload={handleFileUpload}
        />
      )}
    </div>
  );
};

export default ExtractionSection;