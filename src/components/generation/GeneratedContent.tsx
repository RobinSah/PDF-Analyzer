import React from 'react';
import { Download, ThumbsDown, ThumbsUp } from 'lucide-react';
import Card from '../ui/Card';
import Button from '../ui/Button';
import { useTheme } from '../ThemeContext';

interface GeneratedContentProps {
  content: string;
  mode: 'single' | 'bulk';
  onDownload: () => void;
  theme?: any; // Optional theme prop for direct theme passing
}

const GeneratedContent: React.FC<GeneratedContentProps> = ({ content, mode, onDownload, theme: propTheme }) => {
  // Use provided theme prop or get from context
  const themeContext = useTheme();
  const { currentTheme } = themeContext;
  const theme = propTheme || themeContext.getThemeClasses();
  
  // Check if content appears to be HTML (contains HTML tags)
  const isHtmlContent = content.includes('<table') || content.includes('<div') || content.includes('<h');
  
  // Format content if it's not HTML
  const formatTextContent = (content: string) => {
    const parts = content.split('\n\n');
    
    if (parts.length > 1) {
      return (
        <>
          <h3 className={`text-xl font-semibold ${theme.title} mb-4`}>{parts[0].replace('# ', '')}</h3>
          <div className="space-y-4">
            {parts.slice(1).map((part, index) => (
              <p key={index} className={`${theme.text} leading-relaxed`}>{part}</p>
            ))}
          </div>
        </>
      );
    }
    
    return <p className={`${theme.text} whitespace-pre-wrap`}>{content}</p>;
  };

  return (
    <Card className={theme.card}>
      <div className="flex justify-between items-center mb-4">
        <h3 className={`text-lg font-semibold ${theme.title}`}>Extracted Content</h3>
        <Button
          variant={currentTheme === 'light' ? 'outline' : 'filled'}
          size="sm"
          onClick={onDownload}
          icon={<Download className="w-4 h-4" />}
          className={currentTheme === 'light' ? 
            `border ${theme.border} ${theme.text}` : 
            currentTheme === 'futuristic' ? 
              'bg-blue-900/30 border border-blue-500/30 text-blue-300 hover:bg-blue-800/40' : 
              'bg-gray-700 text-gray-200 hover:bg-gray-600'
          }
        >
          {mode === 'single' ? 'Download HTML' : 'Download CSV'}
        </Button>
      </div>
      
      {/* For bulk content, use full-width container with proper scrolling */}
      <div 
        className={`${theme.cardHighlight} rounded-lg p-4 ${
          mode === 'bulk' ? 'max-h-[600px]' : 'max-h-[500px]'
        } overflow-y-auto mb-4`}
      >
        {isHtmlContent ? (
          // Render HTML content with appropriate styling for tables
          <div 
            dangerouslySetInnerHTML={{ __html: content }} 
            className={mode === 'bulk' ? 'bulk-table-container' : ''}
          />
        ) : (
          // Render text content
          formatTextContent(content)
        )}
      </div>
      
      <div className={`flex justify-between items-center pt-3 border-t ${theme.border}`}>
        <div className={`text-sm ${theme.secondaryText}`}>
          Extracted at {new Date().toLocaleTimeString()}
        </div>
        <div className="flex space-x-2">
          <button
            className={`p-1.5 ${theme.secondaryText} hover:${
              currentTheme === 'futuristic' 
                ? 'text-blue-400' 
                : currentTheme === 'dark' 
                  ? 'text-gray-300' 
                  : 'text-gray-700'
            } border ${theme.border} rounded-full hover:${
              currentTheme === 'futuristic' 
                ? 'bg-blue-900/30' 
                : currentTheme === 'dark' 
                  ? 'bg-gray-700' 
                  : 'bg-gray-50'
            }`}
            title="This looks good"
          >
            <ThumbsUp className="w-4 h-4" />
          </button>
          <button
            className={`p-1.5 ${theme.secondaryText} hover:${
              currentTheme === 'futuristic' 
                ? 'text-blue-400' 
                : currentTheme === 'dark' 
                  ? 'text-gray-300' 
                  : 'text-gray-700'
            } border ${theme.border} rounded-full hover:${
              currentTheme === 'futuristic' 
                ? 'bg-blue-900/30' 
                : currentTheme === 'dark' 
                  ? 'bg-gray-700' 
                  : 'bg-gray-50'
            }`}
            title="This needs improvement"
          >
            <ThumbsDown className="w-4 h-4" />
          </button>
        </div>
      </div>
    </Card>
  );
};

export default GeneratedContent;