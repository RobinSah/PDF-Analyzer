import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown } from 'lucide-react';
import { useTheme } from '../ThemeContext';

const models = [
  { id: 'gpt-4.5', name: 'GPT-4.5', description: 'Good for writing and exploring ideas' },
  { id: 'openai-o3', name: 'o3', description: 'Uses advanced reasoning' },
  { id: 'openai-o4-mini', name: 'o4-mini', description: 'Fast at advanced reasoning' },
  { id: 'llama-4', name: 'Llama 4', description: 'Latest Meta open-source model' },
  { id: 'mistral-8x7b', name: 'Mistral 8x7b', description: 'Powerful Mistral model' },
  { id: 'deepseek', name: 'DeepSeek', description: 'Specialized for deep reasoning' },
];

interface ModelSelectorProps {
  onModelChange?: (modelId: string) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ onModelChange }) => {
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();
  const [isOpen, setIsOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState(models[0]);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load previously selected model from localStorage
  useEffect(() => {
    const savedModelId = localStorage.getItem('selectedModelId');
    if (savedModelId) {
      const model = models.find(m => m.id === savedModelId);
      if (model) {
        setSelectedModel(model);
      }
    }
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const toggleDropdown = () => setIsOpen(!isOpen);

  const selectModel = (model: typeof models[0]) => {
    setSelectedModel(model);
    setIsOpen(false);
    localStorage.setItem('selectedModelId', model.id);
    
    if (onModelChange) {
      onModelChange(model.id);
    }
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={toggleDropdown}
        className={`flex items-center space-x-1 rounded-md px-3 py-1.5 text-sm font-medium ${
          currentTheme === 'futuristic' 
            ? 'bg-blue-900/30 text-blue-200 border border-blue-800/70 hover:bg-blue-800/40' 
            : currentTheme === 'dark'
              ? 'bg-gray-800 text-gray-200 border border-gray-700 hover:bg-gray-700'
              : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
        } focus:outline-none focus:ring-2 ${
          currentTheme === 'futuristic' 
            ? 'focus:ring-blue-500 focus:border-blue-500' 
            : currentTheme === 'dark'
              ? 'focus:ring-indigo-500 focus:border-indigo-500' 
              : 'focus:ring-indigo-500 focus:border-indigo-500'
        }`}
      >
        <span>{selectedModel.name}</span>
        <ChevronDown className={`h-4 w-4 ${
          currentTheme === 'futuristic' 
            ? 'text-blue-300' 
            : currentTheme === 'dark'
              ? 'text-gray-400' 
              : 'text-gray-500'
        }`} />
      </button>

      {isOpen && (
        <div className={`absolute right-0 mt-2 w-64 rounded-md shadow-lg z-10 ${
          currentTheme === 'futuristic' 
            ? 'bg-gray-900 border border-blue-800/70' 
            : currentTheme === 'dark'
              ? 'bg-gray-800 border border-gray-700' 
              : 'bg-white border border-gray-300'
        }`}>
          <div className="py-1" role="menu" aria-orientation="vertical">
            {models.map((model) => (
              <button
                key={model.id}
                onClick={() => selectModel(model)}
                className={`w-full text-left px-4 py-2 text-sm ${
                  selectedModel.id === model.id
                    ? currentTheme === 'futuristic'
                      ? 'bg-blue-900/50 text-blue-300'
                      : currentTheme === 'dark'
                        ? 'bg-indigo-900/50 text-indigo-300'
                        : 'bg-indigo-50 text-indigo-600'
                    : currentTheme === 'futuristic'
                      ? 'text-gray-300 hover:bg-blue-900/30'
                      : currentTheme === 'dark'
                        ? 'text-gray-300 hover:bg-gray-700'
                        : 'text-gray-700 hover:bg-gray-50'
                }`}
                role="menuitem"
              >
                <div className="font-medium">{model.name}</div>
                <div className={`text-xs ${
                  currentTheme === 'futuristic'
                    ? 'text-blue-400/70'
                    : currentTheme === 'dark'
                      ? 'text-gray-400'
                      : 'text-gray-500'
                }`}>{model.description}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelSelector;