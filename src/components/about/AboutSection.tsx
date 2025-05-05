import React from 'react';
import { Globe, Database, GitMerge, BookOpen, Zap, FileText, MessageSquare, BarChart2 } from 'lucide-react';
import { useTheme } from '../ThemeContext';

const AboutSection: React.FC = () => {
  // Get theme from context instead of managing state internally
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();

  return (
    <div className={`min-h-screen ${theme.background} ${theme.text} transition-colors duration-300`}>
      <div className="container mx-auto p-6 max-w-6xl">
        <div className="mb-8">
          <h1 className={`text-4xl font-bold mb-2 ${theme.title}`}>About World Bank PDF Analyzer</h1>
          <p className={theme.secondaryText}>Transforming document analysis with AI-powered insights</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Project Overview */}
          <div className={`${theme.card} rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'} p-6 transition-all duration-300`}>
            <h2 className={`text-2xl font-semibold mb-2 ${currentTheme === 'futuristic' ? theme.highlight : ''}`}>Project Overview</h2>
            <h3 className={`${theme.tertiaryText} text-sm mb-5`}>Revolutionizing document analysis for policy teams</h3>
            
            <div className="space-y-4 text-justify">
              <p className={theme.secondaryText}>
                World Bank PDF Analyzer is a specialized RAG (Retrieval-Augmented Generation) platform designed to help policy teams instantly locate and analyze relevant sections within World Bank Project Appraisal Documents (PADs).
              </p>
              
              <div className={`p-4 ${theme.cardHighlight} rounded-lg my-6`}>
                <h4 className={`font-medium mb-3 ${currentTheme === 'futuristic' ? theme.highlight : 'text-indigo-600'}`}>Why It Matters</h4>
                <p className={theme.secondaryText}>
                  Manually reviewing hundreds of PADs against extensive activity lists traditionally requires expert judgment and days of effort. Our platform reduces this to seconds, enabling teams to focus on analysis rather than document searching.
                </p>
              </div>
              
              <h4 className={`font-medium mb-2 ${currentTheme === 'futuristic' ? theme.highlight : 'text-indigo-600'}`}>Advanced Technical Pipeline</h4>
              <ul className={`list-disc pl-5 space-y-2 ${theme.secondaryText}`}>
                <li>
                  <span className="font-medium">Document Processing:</span> Extracts clean, structured text from PDFs using Mistral OCR technology
                </li>
                <li>
                  <span className="font-medium">Semantic Search:</span> Embeds document passages and stores them in a FAISS vector database for lightning-fast retrieval
                </li>
                <li>
                  <span className="font-medium">Intelligent Analysis:</span> Leverages high-reasoning LLMs (DeepSeek-Instruct, OpenAI) to analyze context and identify precise matching sections
                </li>
                <li>
                  <span className="font-medium">Evidence-Based Results:</span> Provides analysts with direct links to source document sections, eliminating guesswork
                </li>
              </ul>
              
              <div className={`p-4 ${theme.cardHighlight} rounded-lg my-4`}>
                <h4 className={`font-medium mb-2 ${currentTheme === 'futuristic' ? theme.highlight : 'text-indigo-600'}`}>Architecture Benefits</h4>
                <p className={theme.secondaryText}>
                  Built on a modern FastAPI + React/TypeScript stack, our platform delivers production-grade performance while maintaining intuitive one-click user experience. The system scales effortlessly to handle large document collections while providing precise, context-aware results.
                </p>
              </div>
            </div>
          </div>

          {/* Contact and Version Info */}
          <div className="space-y-6">
            <div className={`${theme.card} rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'} p-6 transition-all duration-300`}>
              <h2 className={`text-2xl font-semibold mb-2 ${currentTheme === 'futuristic' ? theme.highlight : ''}`}>Contact Information</h2>
              <h3 className={`${theme.tertiaryText} text-sm mb-5`}>Get in touch with our team</h3>
              
              <div className="space-y-5">
                <div className="flex items-start">
                  <GitMerge className={`w-5 h-5 ${theme.iconColor} mr-3 mt-0.5`} />
                  <p className={theme.secondaryText}>github.com/worldbank-pdf-analyzer</p>
                </div>
                
                <div className="flex items-start">
                  <BookOpen className={`w-5 h-5 ${theme.iconColor} mr-3 mt-0.5`} />
                  <p className={theme.secondaryText}>pdf-analyzer@worldbank.org</p>
                </div>
                
                <p className={`${theme.tertiaryText} text-sm mt-4`}>
                  World Bank PDF Analyzer is developed as an internal tool to assist with document analysis and is not an official World Bank product. For official inquiries, please contact the World Bank directly.
                </p>
              </div>
            </div>
            
            <div className={`${theme.card} rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'} p-6 transition-all duration-300`}>
              <h2 className={`text-2xl font-semibold mb-2 ${currentTheme === 'futuristic' ? theme.highlight : ''}`}>Version Information</h2>
              <h3 className={`${theme.tertiaryText} text-sm mb-5`}>Current release details</h3>
              
              <div className="space-y-4">
                <div className={`flex justify-between border-b ${theme.border} pb-3`}>
                  <span className={theme.secondaryText}>Version:</span>
                  <span className="font-medium">1.0.0</span>
                </div>
                
                <div className={`flex justify-between border-b ${theme.border} pb-3`}>
                  <span className={theme.secondaryText}>Last Updated:</span>
                  <span className="font-medium">April 30, 2025</span>
                </div>
                
                <div className={`flex justify-between border-b ${theme.border} pb-3`}>
                  <span className={theme.secondaryText}>Default Model:</span>
                  <span className="font-medium">gpt-4o-mini</span>
                </div>
                
                <p className={`${theme.tertiaryText} text-sm mt-4`}>
                  This software is continuously being improved based on user feedback and advancements in AI technology.
                  If you encounter any issues or have suggestions, please contact the development team.
                </p>
              </div>
            </div>
          </div>

          {/* Technical Stack */}
          <div className={`${theme.card} rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow'} p-6 md:col-span-2 transition-all duration-300`}>
            <h2 className={`text-2xl font-semibold mb-2 ${currentTheme === 'futuristic' ? theme.highlight : ''}`}>Technical Stack</h2>
            <h3 className={`${theme.tertiaryText} text-sm mb-6`}>Technologies powering our platform</h3>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mt-6">
              {/* AI Integration */}
              <div className={`p-4 ${theme.cardHighlight} rounded-lg flex flex-col items-center text-center transition-all hover:translate-y-[-5px] duration-300`}>
                <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
                  <Globe className={`${theme.iconColor} w-7 h-7`} />
                </div>
                <h4 className="font-medium mb-2">OpenAI Integration</h4>
                <p className={`text-sm ${theme.secondaryText}`}>Advanced GPT models for nuanced document analysis and natural language understanding</p>
              </div>
              
              {/* Hugging Face */}
              <div className={`p-4 ${theme.cardHighlight} rounded-lg flex flex-col items-center text-center transition-all hover:translate-y-[-5px] duration-300`}>
                <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
                  <Database className={`${theme.iconColor} w-7 h-7`} />
                </div>
                <h4 className="font-medium mb-2">Open Source Models</h4>
                <p className={`text-sm ${theme.secondaryText}`}>Cost-effective content generation and extraction via Hugging Face API with models like Mistral and DeepSeek</p>
              </div>
              
              {/* FAISS Vector DB */}
              <div className={`p-4 ${theme.cardHighlight} rounded-lg flex flex-col items-center text-center transition-all hover:translate-y-[-5px] duration-300`}>
                <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
                  <Zap className={`${theme.iconColor} w-7 h-7`} />
                </div>
                <h4 className="font-medium mb-2">FAISS Vector Search</h4>
                <p className={`text-sm ${theme.secondaryText}`}>High-performance similarity search for document embeddings with millisecond response times</p>
              </div>
              
              {/* OCR Technology */}
              <div className={`p-4 ${theme.cardHighlight} rounded-lg flex flex-col items-center text-center transition-all hover:translate-y-[-5px] duration-300`}>
                <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
                  <FileText className={`${theme.iconColor} w-7 h-7`} />
                </div>
                <h4 className="font-medium mb-2">Mistral OCR</h4>
                <p className={`text-sm ${theme.secondaryText}`}>State-of-the-art document processing with industry-leading accuracy for complex World Bank documents</p>
              </div>
            </div>
            
            {/* Key Features Row */}
            <div className={`mt-10 pt-8 border-t ${theme.border}`}>
              <h3 className={`text-xl font-semibold mb-6 ${currentTheme === 'futuristic' ? theme.highlight : ''}`}>Core Capabilities</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* RAG Pipeline */}
                <div className="flex items-start">
                  <div className={`p-2 ${theme.iconBg} rounded-md flex items-center justify-center mr-4`}>
                    <MessageSquare className={`${theme.iconColor} w-5 h-5`} />
                  </div>
                  <div>
                    <h4 className="font-medium mb-1">Advanced RAG Pipeline</h4>
                    <p className={`text-sm ${theme.secondaryText}`}>Context-aware retrieval system that understands document semantics, not just keywords</p>
                  </div>
                </div>
                
                {/* Analytics */}
                <div className="flex items-start">
                  <div className={`p-2 ${theme.iconBg} rounded-md flex items-center justify-center mr-4`}>
                    <BarChart2 className={`${theme.iconColor} w-5 h-5`} />
                  </div>
                  <div>
                    <h4 className="font-medium mb-1">Usage Analytics</h4>
                    <p className={`text-sm ${theme.secondaryText}`}>Comprehensive metrics tracking for optimizing performance and costs across different models</p>
                  </div>
                </div>
                
                {/* Multi-Model */}
                <div className="flex items-start">
                  <div className={`p-2 ${theme.iconBg} rounded-md flex items-center justify-center mr-4`}>
                    <Database className={`${theme.iconColor} w-5 h-5`} />
                  </div>
                  <div>
                    <h4 className="font-medium mb-1">Multi-Model Support</h4>
                    <p className={`text-sm ${theme.secondaryText}`}>Flexibility to choose from various AI providers based on specific task requirements</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutSection;