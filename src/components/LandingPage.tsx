import React from 'react';
import { 
  FileText, 
  BarChart2, 
  Globe, 
  ChevronRight, 
  MessageSquare, 
  LineChart, 
  Zap 
} from 'lucide-react';
import { useTheme } from './ThemeContext';
import ThemeSwitcher from './ThemeSwitcher';

interface LandingPageProps {
  onGetStarted: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted }) => {
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();

  return (
    <div className={`min-h-screen ${theme.background} ${theme.text} transition-colors duration-300`}>
      {/* Theme Switcher */}
      <div className="absolute top-4 right-4 z-10">
        <ThemeSwitcher />
      </div>

      {/* Hero Section */}
      <div className="flex flex-col items-center justify-center pt-20 pb-10 px-4 text-center">
        <div className="flex items-center mb-6">
          <img 
            src="https://images.seeklogo.com/logo-png/52/1/world-bank-logo-png_seeklogo-521136.png" 
            alt="World Bank Logo" 
            className={`h-16 w-auto ${currentTheme === 'futuristic' ? 'filter drop-shadow-lg' : ''}`}
          />
        </div>
        <h1 className={`text-4xl md:text-5xl font-bold mb-4 ${currentTheme === 'futuristic' ? 'text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-purple-300' : ''}`}>
          World Bank PAD Analyzer
        </h1>
        <p className={`text-xl ${theme.secondaryText} max-w-3xl mb-8`}>
          AI-Powered Document Analysis for extracting and generating results with advanced reasoning
        </p>
        <p className={`${theme.tertiaryText} max-w-2xl mb-8`}>
          Leverage the power of AI models to analyze World Bank documents. Extract insights, generate reports,
          and enhance project planning with AI assistance.
        </p>
        <button 
          onClick={onGetStarted}
          className={`${theme.button} text-white font-medium py-3 px-8 rounded-md transition-all flex items-center ${currentTheme === 'futuristic' ? 'shadow-lg shadow-blue-500/20' : ''}`}
        >
          Get Started
          <ChevronRight className="ml-2 w-5 h-5" />
        </button>
      </div>

      {/* Features Section */}
      <div className="max-w-6xl mx-auto px-4 py-16">
        <h2 className={`text-3xl font-bold text-center mb-12 ${currentTheme === 'futuristic' ? 'text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-purple-300' : ''}`}>
          Key Features
        </h2>
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {/* Feature 1 */}
          <div className={`${theme.card} p-6 rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow-sm'} transition-all duration-300 hover:translate-y-[-5px]`}>
            <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
              <FileText className={`${theme.iconColor} w-7 h-7`} />
            </div>
            <h3 className="text-xl font-semibold mb-2">Document Preview</h3>
            <p className={theme.secondaryText}>
              Access Mistral OCR for the best markdown output and extraction. Upload your own files for analysis.
            </p>
          </div>

          {/* Feature 2 */}
          <div className={`${theme.card} p-6 rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow-sm'} transition-all duration-300 hover:translate-y-[-5px]`}>
            <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
              <BarChart2 className={`${theme.iconColor} w-7 h-7`} />
            </div>
            <h3 className="text-xl font-semibold mb-2">Generate Results</h3>
            <p className={theme.secondaryText}>
              Get comprehensive results based on your input, provided prompt or your own prompt using different Open Source LLMs.
            </p>
          </div>

          {/* Feature 3 */}
          <div className={`${theme.card} p-6 rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow-sm'} transition-all duration-300 hover:translate-y-[-5px]`}>
            <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
              <Globe className={`${theme.iconColor} w-7 h-7`} />
            </div>
            <h3 className="text-xl font-semibold mb-2">AI Integration</h3>
            <p className={theme.secondaryText}>
              Powered by cutting-edge AI models to provide accurate analysis and natural language responses.
            </p>
          </div>
        </div>

        {/* Additional Features */}
        <div className="grid md:grid-cols-3 gap-8">
          {/* Feature 4 */}
          <div className={`${theme.card} p-6 rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow-sm'} transition-all duration-300 hover:translate-y-[-5px]`}>
            <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
              <MessageSquare className={`${theme.iconColor} w-7 h-7`} />
            </div>
            <h3 className="text-xl font-semibold mb-2">Chat with PDF</h3>
            <p className={theme.secondaryText}>
              Ask questions about your documents and receive contextually relevant answers using our advanced RAG pipeline.
            </p>
          </div>

          {/* Feature 5 */}
          <div className={`${theme.card} p-6 rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow-sm'} transition-all duration-300 hover:translate-y-[-5px]`}>
            <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
              <LineChart className={`${theme.iconColor} w-7 h-7`} />
            </div>
            <h3 className="text-xl font-semibold mb-2">Analytics Dashboard</h3>
            <p className={theme.secondaryText}>
              Track usage metrics, costs, and performance across different models to optimize your document processing.
            </p>
          </div>

          {/* Feature 6 */}
          <div className={`${theme.card} p-6 rounded-lg ${currentTheme === 'futuristic' ? 'shadow-xl shadow-blue-900/10' : 'shadow-sm'} transition-all duration-300 hover:translate-y-[-5px]`}>
            <div className={`p-3 ${theme.iconBg} rounded-full w-14 h-14 flex items-center justify-center mb-4`}>
              <Zap className={`${theme.iconColor} w-7 h-7`} />
            </div>
            <h3 className="text-xl font-semibold mb-2">Multi-Model Support</h3>
            <p className={theme.secondaryText}>
              Choose from multiple AI providers including OpenAI, Hugging Face, and other open-source models.
            </p>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className={`py-16 px-4 ${currentTheme === 'futuristic' ? 'bg-gradient-to-r from-blue-900/50 to-purple-900/50' : theme.cardHighlight}`}>
        <div className="max-w-4xl mx-auto text-center">
          <h2 className={`text-3xl font-bold mb-6 ${currentTheme === 'futuristic' ? 'text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-purple-300' : ''}`}>
            Ready to analyze your World Bank documents?
          </h2>
          <p className={`${theme.secondaryText} mb-8 max-w-2xl mx-auto`}>
            Start extracting valuable insights and generating high-quality content from your PDF documents today.
          </p>
          <button 
            onClick={onGetStarted}
            className={`${theme.button} text-white font-medium py-3 px-8 rounded-md transition-all ${currentTheme === 'futuristic' ? 'shadow-lg shadow-blue-500/20' : ''}`}
          >
            Get Started Now
          </button>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;