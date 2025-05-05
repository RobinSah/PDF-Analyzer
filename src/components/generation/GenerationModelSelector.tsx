import React, { useState, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';

const models = [
    { id: 'gpt-4.5', name: 'GPT-4.5', description: 'Good for writing and exploring ideas' },
    { id: 'openai-o3', name: 'o3', description: 'Uses advanced reasoning' },
    { id: 'openai-o4-mini', name: 'o4-mini', description: 'Fast at advanced reasoning' },
    { id: 'llama-4', name: 'Llama 4', description: 'Latest Meta open-source model' },
    { id: 'mistral-8x7b', name: 'Mistral 8x7b', description: 'Powerful Mistral model' },
    { id: 'deepseek', name: 'DeepSeek', description: 'Specialized for deep reasoning' },
  ];  

interface GenerationModelSelectorProps {
  onModelChange?: (modelId: string) => void;
  defaultModel?: string;
}

const GenerationModelSelector: React.FC<GenerationModelSelectorProps> = ({ onModelChange, defaultModel }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState(
    defaultModel 
      ? models.find(m => m.id === defaultModel) || models[0]
      : models[0]
  );

  // Load previously selected model from localStorage
  useEffect(() => {
    const savedModelId = localStorage.getItem('generationSelectedModelId');
    if (savedModelId) {
      const model = models.find(m => m.id === savedModelId);
      if (model) {
        setSelectedModel(model);
        // Important: Don't call onModelChange here, as it would trigger content generation on initial load
      }
    }
  }, []);

  const toggleDropdown = () => setIsOpen(!isOpen);

  const selectModel = (model: typeof models[0]) => {
    setSelectedModel(model);
    setIsOpen(false);
    localStorage.setItem('generationSelectedModelId', model.id);
    
    if (onModelChange) {
      onModelChange(model.id);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={toggleDropdown}
        className="flex items-center space-x-1 bg-white border border-gray-300 rounded-md px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
      >
        <span>{selectedModel.name}</span>
        <ChevronDown className="h-4 w-4 text-gray-500" />
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-10">
          <div className="py-1" role="menu" aria-orientation="vertical">
            {models.map((model) => (
              <button
                key={model.id}
                onClick={() => selectModel(model)}
                className={`w-full text-left px-4 py-2 text-sm ${
                  selectedModel.id === model.id
                    ? 'bg-amber-50 text-amber-600'
                    : 'text-gray-700 hover:bg-gray-50'
                }`}
                role="menuitem"
              >
                <div className="font-medium">{model.name}</div>
                <div className="text-xs text-gray-500">{model.description}</div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GenerationModelSelector;