import React, { useState, useEffect } from 'react';
import { ChevronDown, FileUp, Send, Loader2, Trash2 } from 'lucide-react';
import Button from '../ui/Button';
import ChatMessage from './ChatMessage';
import ModelSelector from './ModelSelector';
import FileUploadModal from '../extraction/FileUploadModal';
import { useTheme } from '../ThemeContext';
import axios from 'axios';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Conversation {
  id: string;
  title: string;
  active: boolean;
  messages: Message[];
  pdfName?: string;
}

const API_BASE_URL = 'http://localhost:8000';

const ChatSection: React.FC = () => {
  // Get theme from context
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();

  const initialMessage: Message = {
    id: '1',
    role: 'assistant',
    content: 'Hello! I\'m ready to chat about your PDF. Upload a document or ask me a question.',
    timestamp: new Date(),
  };

  // Use session storage key to detect when page is refreshed or reopened
  const [sessionKey, setSessionKey] = useState<string>(() => {
    return sessionStorage.getItem('pdfChatSessionKey') || '';
  });

  const [conversations, setConversations] = useState<Conversation[]>(() => {
    // Check if we have a session key (returning user in same session)
    const currentSessionKey = sessionStorage.getItem('pdfChatSessionKey');
    
    // If page is refreshed or new session, clear localStorage
    if (!currentSessionKey) {
      // Generate a new session key
      const newSessionKey = Date.now().toString();
      sessionStorage.setItem('pdfChatSessionKey', newSessionKey);
      setSessionKey(newSessionKey);
      
      // Clear localStorage on first load of a new session
      localStorage.removeItem('pdfChatConversations');
      localStorage.removeItem('pdfMarkdownContent');
      localStorage.removeItem('pdfExtractedText');
      localStorage.removeItem('pdfStructuredPages');
      
      // Return default conversations with only "New Chat"
      return [
        { 
          id: '1', 
          title: 'New Chat', 
          active: true,
          messages: [initialMessage]
        }
      ];
    }
    
    // If we have a session key, try to load conversations from localStorage
    const savedConversations = localStorage.getItem('pdfChatConversations');
    if (savedConversations) {
      try {
        const parsed = JSON.parse(savedConversations);
        return parsed.map((conv: any) => ({
          ...conv,
          messages: conv.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
      } catch (e) {
        console.error('Failed to parse saved conversations:', e);
      }
    }
    
    // Default new chat conversation
    return [
      { 
        id: '1', 
        title: 'New Chat', 
        active: true,
        messages: [initialMessage]
      }
    ];
  });

  const [activeConversation, setActiveConversation] = useState<Conversation | null>(null);
  const [input, setInput] = useState('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [pdfContent, setPdfContent] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('gpt-4.5');
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Load the extracted PDF content from localStorage only if we have a valid session
  useEffect(() => {
    if (sessionKey) {
      const storedText = localStorage.getItem('pdfExtractedText');
      if (storedText) {
        setPdfContent(storedText);
      }
    }
  }, [sessionKey]);

  // Set the active conversation
  useEffect(() => {
    const active = conversations.find(c => c.active);
    if (active) {
      setActiveConversation(active);
    }
  }, [conversations]);

  // Save conversations to localStorage when they change
  useEffect(() => {
    if (sessionKey) {
      localStorage.setItem('pdfChatConversations', JSON.stringify(conversations));
    }
  }, [conversations, sessionKey]);

  const updateConversation = (updatedConversation: Conversation) => {
    setConversations(prev => 
      prev.map(conv => 
        conv.id === updatedConversation.id ? updatedConversation : conv
      )
    );
  };

  const handleSendMessage = async () => {
    if (!input.trim() || !activeConversation) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    // Add user message to conversation
    const updatedMessages = [...activeConversation.messages, userMessage];
    const updatedConversation = {
      ...activeConversation,
      messages: updatedMessages
    };
    updateConversation(updatedConversation);
    setActiveConversation(updatedConversation);
    setInput('');

    // Start generating response
    setIsGenerating(true);

    try {
      // Send request to the backend
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        model: selectedModel,
        messages: updatedMessages.map(msg => ({
          role: msg.role,
          content: msg.content
        })),
        pdf_content: pdfContent
      });

      // Process the response
      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: response.data.response || "I couldn't generate a response. Please try again.",
        timestamp: new Date(),
      };

      // Update with assistant response
      const finalMessages = [...updatedMessages, assistantMessage];
      const finalConversation = {
        ...updatedConversation,
        messages: finalMessages
      };
      updateConversation(finalConversation);
      setActiveConversation(finalConversation);
    } catch (error) {
      console.error('Error generating response:', error);
      
      // Add error message
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: "Sorry, I encountered an error while generating a response. Please try again.",
        timestamp: new Date(),
      };
      
      const finalMessages = [...updatedMessages, errorMessage];
      const finalConversation = {
        ...updatedConversation,
        messages: finalMessages
      };
      updateConversation(finalConversation);
      setActiveConversation(finalConversation);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileUpload = async (file: File | null, url: string | null) => {
    setShowUploadModal(false);
    
    if (file) {
      setUploadedFile(file);
      
      try {
        // Upload file to backend for processing
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await axios.post(`${API_BASE_URL}/api/process-pdf`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });
        
        if (response.data.structured_pages) {
          // Create plain text representation
          const combinedText = response.data.structured_pages
            .map((page: any) => `${page.plain_text}\n\n---\nPage ${page.page_number + 1}\n\n`)
            .join('\n');
          
          setPdfContent(combinedText);
          
          // Store PDF content in localStorage
          localStorage.setItem('pdfExtractedText', combinedText);
          
          // Create new conversation
          const newConversation: Conversation = {
            id: Date.now().toString(),
            title: file.name,
            active: true,
            messages: [
              {
                id: Date.now().toString(),
                role: 'assistant',
                content: `I've loaded your PDF "${file.name}". What would you like to know about it?`,
                timestamp: new Date(),
              }
            ],
            pdfName: file.name
          };
          
          // Deactivate all existing conversations
          const updatedConversations = conversations.map(c => ({...c, active: false}));
          
          // Add new conversation
          setConversations([newConversation, ...updatedConversations]);
          setActiveConversation(newConversation);
        }
      } catch (error) {
        console.error('Error processing PDF:', error);
        const errorMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: "I encountered an error while processing your PDF. Please try again.",
          timestamp: new Date(),
        };
        
        if (activeConversation) {
          const updatedMessages = [...activeConversation.messages, errorMessage];
          const updatedConversation = {
            ...activeConversation,
            messages: updatedMessages
          };
          updateConversation(updatedConversation);
          setActiveConversation(updatedConversation);
        }
      }
    } else if (url) {
      try {
        const response = await axios.post(`${API_BASE_URL}/api/process-pdf-url`, { url });
        
        if (response.data.structured_pages) {
          // Create plain text representation
          const combinedText = response.data.structured_pages
            .map((page: any) => `${page.plain_text}\n\n---\nPage ${page.page_number + 1}\n\n`)
            .join('\n');
          
          setPdfContent(combinedText);
          
          // Store PDF content in localStorage
          localStorage.setItem('pdfExtractedText', combinedText);
          
          // URL filename
          const urlFilename = url.split('/').pop() || 'PDF from URL';
          
          // Create new conversation
          const newConversation: Conversation = {
            id: Date.now().toString(),
            title: `PDF from ${new URL(url).hostname}`,
            active: true,
            messages: [
              {
                id: Date.now().toString(),
                role: 'assistant',
                content: `I've loaded the PDF from ${url}. What would you like to know about it?`,
                timestamp: new Date(),
              }
            ],
            pdfName: urlFilename
          };
          
          // Deactivate all existing conversations
          const updatedConversations = conversations.map(c => ({...c, active: false}));
          
          // Add new conversation
          setConversations([newConversation, ...updatedConversations]);
          setActiveConversation(newConversation);
        }
      } catch (error) {
        console.error('Error processing PDF from URL:', error);
        const errorMessage: Message = {
          id: Date.now().toString(),
          role: 'assistant',
          content: "I encountered an error while processing the PDF from URL. Please try again.",
          timestamp: new Date(),
        };
        
        if (activeConversation) {
          const updatedMessages = [...activeConversation.messages, errorMessage];
          const updatedConversation = {
            ...activeConversation,
            messages: updatedMessages
          };
          updateConversation(updatedConversation);
          setActiveConversation(updatedConversation);
        }
      }
    }
  };

  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
  };

  const switchConversation = (convId: string) => {
    const updatedConversations = conversations.map(conv => ({
      ...conv,
      active: conv.id === convId
    }));
    
    setConversations(updatedConversations);
    const newActive = updatedConversations.find(c => c.id === convId) || null;
    setActiveConversation(newActive);
  };

  const createNewChat = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      active: true,
      messages: [initialMessage]
    };
    
    // Deactivate all existing conversations
    const updatedConversations = conversations.map(c => ({...c, active: false}));
    
    // Add new conversation
    setConversations([newConversation, ...updatedConversations]);
    setActiveConversation(newConversation);
    
    // Clear the input
    setInput('');
  };

  // Add delete conversation functionality
  const deleteConversation = (convId: string) => {
    // If trying to delete the active conversation
    const isActive = conversations.find(c => c.id === convId)?.active;
    
    // Remove the conversation
    const updatedConversations = conversations.filter(c => c.id !== convId);
    
    // If there are no conversations left, create a new one
    if (updatedConversations.length === 0) {
      const newConversation: Conversation = {
        id: Date.now().toString(),
        title: 'New Chat',
        active: true,
        messages: [initialMessage]
      };
      setConversations([newConversation]);
      setActiveConversation(newConversation);
    } 
    // If we deleted the active conversation, make the first conversation active
    else if (isActive) {
      updatedConversations[0].active = true;
      setConversations(updatedConversations);
      setActiveConversation(updatedConversations[0]);
    } 
    // Otherwise just update the conversations list
    else {
      setConversations(updatedConversations);
    }
  };

  // Add a clear all conversations function
  const clearAllConversations = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      active: true,
      messages: [initialMessage]
    };
    
    setConversations([newConversation]);
    setActiveConversation(newConversation);
    setInput('');
    
    // Clear PDF content
    setPdfContent(null);
    setUploadedFile(null);
    
    // Clear localStorage
    localStorage.removeItem('pdfChatConversations');
    localStorage.removeItem('pdfMarkdownContent');
    localStorage.removeItem('pdfExtractedText');
    localStorage.removeItem('pdfStructuredPages');
  };

  return (
    <div className={`w-full h-full flex ${theme.background}`}>
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header with model selector */}
        <div className={`p-4 ${theme.card} border-b ${theme.border}`}>
          <div className="flex justify-between items-center">
            <h1 className={`text-xl font-bold ${theme.title}`}>Chat with PDF</h1>
            <ModelSelector onModelChange={handleModelChange} />
          </div>
          {uploadedFile && (
            <div className={`mt-2 flex items-center text-sm ${theme.secondaryText}`}>
              <FileUp className="w-4 h-4 mr-1" />
              <span>Using: {uploadedFile.name}</span>
            </div>
          )}
          {!uploadedFile && pdfContent && activeConversation?.pdfName && (
            <div className={`mt-2 flex items-center text-sm ${theme.secondaryText}`}>
              <FileUp className="w-4 h-4 mr-1" />
              <span>Using: {activeConversation.pdfName}</span>
            </div>
          )}
        </div>

        {/* Messages */}
        <div className={`flex-1 overflow-y-auto p-4 ${theme.cardHighlight}`}>
          <div className="max-w-3xl mx-auto space-y-4">
            {activeConversation?.messages.map((message) => (
              <ChatMessage key={message.id} message={message} theme={theme} />
            ))}
            {isGenerating && (
              <div className="flex justify-start">
                <div className="flex max-w-[85%] flex-row">
                  <div className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${currentTheme === 'futuristic' ? 'bg-blue-500' : currentTheme === 'dark' ? 'bg-indigo-600' : 'bg-emerald-500'} mr-3`}>
                    <Loader2 className="h-4 w-4 text-white animate-spin" />
                  </div>
                  <div className={`rounded-lg p-3 ${theme.card} border ${theme.border} ${theme.text}`}>
                    <p className={theme.secondaryText}>Generating response...</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Input area */}
        <div className={`border-t ${theme.border} p-4 ${theme.card}`}>
          <div className="max-w-3xl mx-auto">
            <div className={`flex items-end border ${theme.border} rounded-lg focus-within:ring-2 ${
              currentTheme === 'futuristic' ? 'focus-within:ring-blue-500 focus-within:border-blue-500' : 
              currentTheme === 'dark' ? 'focus-within:ring-indigo-500 focus-within:border-indigo-500' : 
              'focus-within:ring-indigo-500 focus-within:border-indigo-500'
            } ${theme.card}`}>
              <textarea
                className={`flex-1 p-3 block w-full rounded-l-lg focus:outline-none min-h-[60px] max-h-[200px] resize-y ${theme.card} ${theme.text}`}
                placeholder="Ask a question about your PDF..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={isGenerating}
              />
              <div className="flex items-center px-3">
                <button
                  className={`${theme.secondaryText} hover:${
                    currentTheme === 'futuristic' ? 'text-blue-400' : 
                    currentTheme === 'dark' ? 'text-indigo-400' : 
                    'text-indigo-600'
                  } transition-colors p-1 mr-1`}
                  onClick={() => setShowUploadModal(true)}
                  title="Upload PDF"
                  disabled={isGenerating}
                >
                  <FileUp className="w-5 h-5" />
                </button>
                <Button
                  className={`rounded-l-none ${
                    currentTheme === 'futuristic' ? 
                      'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg shadow-blue-500/20' : 
                      currentTheme === 'dark' ? 
                        'bg-indigo-600 hover:bg-indigo-700 text-white' : 
                        'bg-indigo-600 hover:bg-indigo-700 text-white'
                  }`}
                  onClick={handleSendMessage}
                  disabled={!input.trim() || isGenerating}
                  icon={isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                >
                  {isGenerating ? 'Generating...' : 'Send'}
                </Button>
              </div>
            </div>
            <p className={`text-xs ${theme.secondaryText} mt-2`}>
              {pdfContent 
                ? 'PDF loaded. Ask anything about it!' 
                : 'Upload a PDF first or ask a general question.'}
            </p>
          </div>
        </div>
      </div>

      {/* Conversation History - Adjusted position */}
      <div className={`hidden md:flex md:flex-col w-80 border-l ${theme.border} ${theme.card}`}>
        <div className={`p-4 border-b ${theme.border} mt-8`}>
          <h2 className={`text-lg font-semibold ${theme.title}`}>Chat History</h2>
        </div>
        <div className="p-2 flex-1 overflow-y-auto">
          <div className="flex space-x-2 mb-3">
            <Button 
              className={`flex-1 ${
                currentTheme === 'futuristic' ? 
                  'bg-blue-600 hover:bg-blue-700 text-white' : 
                  currentTheme === 'dark' ? 
                    'bg-indigo-600 hover:bg-indigo-700 text-white' : 
                    ''
              }`}
              onClick={createNewChat}
            >
              New Chat
            </Button>
            <Button 
              variant={currentTheme === 'light' ? 'outline' : 'filled'}
              className={currentTheme === 'light' ? 
                `border ${theme.border} ${theme.text} px-2` : 
                currentTheme === 'futuristic' ? 
                  'bg-blue-900/30 border border-blue-500/30 text-blue-300 hover:bg-blue-800/40 px-2' : 
                  'bg-gray-700 text-gray-200 hover:bg-gray-600 px-2'
              }
              title="Clear all conversations"
              onClick={clearAllConversations}
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
          
          <ul className="space-y-1">
            {conversations.map((conversation) => (
              <li key={conversation.id} className="flex items-center">
                <button
                  className={`flex-1 text-left px-3 py-2 rounded-md hover:${theme.cardHighlight} transition-colors ${
                    conversation.active ? 
                      currentTheme === 'futuristic' ? 'bg-blue-900/30 text-blue-300' : 
                      currentTheme === 'dark' ? 'bg-indigo-900/50 text-indigo-300' : 
                      'bg-indigo-50 text-indigo-700'
                    : theme.text
                  }`}
                  onClick={() => switchConversation(conversation.id)}
                >
                  {conversation.title}
                </button>
                <button
                  className={`p-1 ${theme.secondaryText} hover:text-red-500 transition-colors`}
                  onClick={() => deleteConversation(conversation.id)}
                  title="Delete conversation"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {showUploadModal && (
        <FileUploadModal 
          onClose={() => setShowUploadModal(false)} 
          onUpload={handleFileUpload}
        />
      )}
    </div>
  );
};

export default ChatSection;