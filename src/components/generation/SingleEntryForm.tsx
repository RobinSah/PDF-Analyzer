import React, { useState, useRef } from 'react';
import { Upload } from 'lucide-react';
import Button from '../ui/Button';
import GenerationModelSelector from './GenerationModelSelector';

interface SingleEntryFormProps {
  onGenerate: (activity: string, definition: string, file: File | null, modelId: string) => void;
}

const SingleEntryForm: React.FC<SingleEntryFormProps> = ({ onGenerate }) => {
  const [activity, setActivity] = useState('');
  const [definition, setDefinition] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState('gpt-4.5'); // Set default model
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleModelChange = (modelId: string) => {
    // Just update the state, don't trigger generation
    setSelectedModel(modelId);
  };

  const handleGenerate = () => {
    if (!activity.trim()) {
      alert('Please enter an activity name');
      return;
    }
    
    if (!definition.trim()) {
      alert('Please enter a definition');
      return;
    }
    
    if (!file) {
      alert('Please upload a PDF file');
      return;
    }
    
    onGenerate(activity, definition, file, selectedModel);
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Generate Single Entry</h3>
        <GenerationModelSelector 
          onModelChange={handleModelChange} 
          defaultModel={selectedModel}
        />
      </div>
      
      <div className="space-y-4">
        <div>
          <label htmlFor="activity" className="block text-sm font-medium text-gray-700 mb-1">
            Activity Name
          </label>
          <input
            id="activity"
            type="text"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-amber-500 focus:border-amber-500"
            placeholder="Enter the name of the activity"
            value={activity}
            onChange={(e) => setActivity(e.target.value)}
          />
        </div>
        
        <div>
          <label htmlFor="definition" className="block text-sm font-medium text-gray-700 mb-1">
            Definition
          </label>
          <textarea
            id="definition"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-amber-500 focus:border-amber-500"
            rows={4}
            placeholder="Enter a detailed definition of the activity"
            value={definition}
            onChange={(e) => setDefinition(e.target.value)}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Supporting PDF
          </label>
          <div className="flex items-center">
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".pdf"
              onChange={handleFileChange}
            />
            <Button
              variant="outline"
              size="sm"
              className="w-full"
              onClick={triggerFileInput}
              icon={<Upload className="w-4 h-4" />}
            >
              {file ? file.name : 'Choose PDF file'}
            </Button>
          </div>
          {file && (
            <p className="mt-1 text-xs text-gray-500">
              Selected file: {file.name} ({Math.round(file.size / 1024)} KB)
            </p>
          )}
        </div>
        
        <Button
          size="lg"
          fullWidth
          onClick={handleGenerate}
          className="mt-4"
        >
          Generate Content
        </Button>
      </div>
    </div>
  );
};

export default SingleEntryForm;