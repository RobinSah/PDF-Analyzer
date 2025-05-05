import React from 'react';
import { Bot, Copy, ThumbsUp, User } from 'lucide-react';
import { useTheme } from '../ThemeContext';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        <div 
          className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${
            isUser 
              ? currentTheme === 'futuristic' 
                ? 'bg-blue-600 ml-3' 
                : currentTheme === 'dark' 
                  ? 'bg-indigo-600 ml-3' 
                  : 'bg-indigo-500 ml-3'
              : currentTheme === 'futuristic' 
                ? 'bg-blue-500 mr-3' 
                : currentTheme === 'dark' 
                  ? 'bg-emerald-600 mr-3' 
                  : 'bg-emerald-500 mr-3'
          }`}
        >
          {isUser ? (
            <User className="h-4 w-4 text-white" />
          ) : (
            <Bot className="h-4 w-4 text-white" />
          )}
        </div>
        
        <div 
          className={`rounded-lg p-3 ${
            isUser 
              ? currentTheme === 'futuristic' 
                ? 'bg-gradient-to-r from-blue-600 to-purple-700 text-white' 
                : currentTheme === 'dark' 
                  ? 'bg-gradient-to-r from-indigo-700 to-purple-800 text-white' 
                  : 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white'
              : `${theme.card} border ${theme.border} ${theme.text}`
          }`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
          
          {!isUser && (
            <div className={`mt-2 pt-2 border-t ${theme.border} flex justify-end space-x-2`}>
              <button 
                className={`${theme.secondaryText} hover:${
                  currentTheme === 'futuristic' 
                    ? 'text-blue-400' 
                    : currentTheme === 'dark' 
                      ? 'text-gray-300' 
                      : 'text-gray-700'
                } p-1 rounded-full hover:${
                  currentTheme === 'futuristic' 
                    ? 'bg-blue-900/30' 
                    : currentTheme === 'dark' 
                      ? 'bg-gray-700' 
                      : 'bg-gray-100'
                } transition-colors`}
                title="Copy to clipboard"
              >
                <Copy className="h-4 w-4" />
              </button>
              <button 
                className={`${theme.secondaryText} hover:${
                  currentTheme === 'futuristic' 
                    ? 'text-blue-400' 
                    : currentTheme === 'dark' 
                      ? 'text-gray-300' 
                      : 'text-gray-700'
                } p-1 rounded-full hover:${
                  currentTheme === 'futuristic' 
                    ? 'bg-blue-900/30' 
                    : currentTheme === 'dark' 
                      ? 'bg-gray-700' 
                      : 'bg-gray-100'
                } transition-colors`}
                title="Like this response"
              >
                <ThumbsUp className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;