import React, { useState } from 'react';
import { FileUp, Link as LinkIcon, X } from 'lucide-react';
import Button from '../ui/Button';

interface FileUploadModalProps {
  onClose: () => void;
  onUpload: (file: File | null, url: string | null) => void;
}

const FileUploadModal: React.FC<FileUploadModalProps> = ({ onClose, onUpload }) => {
  const [activeTab, setActiveTab] = useState<'file' | 'link'>('file');
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isValidatingUrl, setIsValidatingUrl] = useState(false);
  const [urlError, setUrlError] = useState<string | null>(null);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
      } else {
        alert('Please upload a PDF file');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type === 'application/pdf') {
        setFile(selectedFile);
      } else {
        alert('Please upload a PDF file');
      }
    }
  };
  
  const validateUrl = (inputUrl: string): boolean => {
    // Basic URL validation
    try {
      const url = new URL(inputUrl);
      return url.protocol === 'http:' || url.protocol === 'https:';
    } catch {
      return false;
    }
  };

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputUrl = e.target.value;
    setUrl(inputUrl);
    
    // Clear previous error
    if (urlError) setUrlError(null);
  };

  const handleSubmit = () => {
    if (activeTab === 'file') {
      if (file) {
        onUpload(file, null);
      }
    } else {
      // Validate URL before submitting
      if (!url.trim()) {
        setUrlError('Please enter a URL');
        return;
      }
      
      if (!validateUrl(url)) {
        setUrlError('Please enter a valid URL');
        return;
      }
      
      // Check if URL ends with .pdf (basic check)
      if (!url.toLowerCase().endsWith('.pdf')) {
        // We'll allow non-pdf URLs, but show a warning
        if (!confirm('The URL does not appear to be a PDF file. Continue anyway?')) {
          return;
        }
      }
      
      onUpload(null, url.trim());
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md">
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Upload PDF</h3>
          <button 
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="p-4">
          {/* Tabs */}
          <div className="flex border-b border-gray-200 mb-4">
            <button
              className={`px-4 py-2 text-sm font-medium border-b-2 ${
                activeTab === 'file'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('file')}
            >
              Upload File
            </button>
            <button
              className={`px-4 py-2 text-sm font-medium border-b-2 ${
                activeTab === 'link'
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
              onClick={() => setActiveTab('link')}
            >
              Paste Link
            </button>
          </div>
          
          {/* File Upload Tab */}
          {activeTab === 'file' && (
            <div 
              className={`border-2 ${
                isDragging ? 'border-indigo-500 bg-indigo-50' : file ? 'border-green-500 bg-green-50' : 'border-dashed border-gray-300 bg-gray-50'
              } rounded-lg p-6 transition-colors duration-200 mb-4`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="text-center">
                <FileUp className="mx-auto h-12 w-12 text-gray-400" />
                <div className="mt-4">
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer rounded-md font-medium text-indigo-600 hover:text-indigo-500 focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500"
                  >
                    <span>{file ? 'Change file' : 'Select a PDF file'}</span>
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      className="sr-only"
                      accept=".pdf"
                      onChange={handleFileChange}
                    />
                  </label>
                  <p className="text-xs text-gray-500 mt-1">
                    or drag and drop
                  </p>
                </div>
                {file && (
                  <div className="mt-4 text-sm text-gray-800">
                    Selected file: <span className="font-medium">{file.name}</span>
                    {' '}<span className="text-gray-500">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Link Tab */}
          {activeTab === 'link' && (
            <div className="mb-4">
              <label htmlFor="pdf-url" className="block text-sm font-medium text-gray-700 mb-1">
                PDF URL
              </label>
              <div className="flex rounded-md shadow-sm">
                <div className="px-3 py-2 rounded-l-md border border-r-0 border-gray-300 bg-gray-50 text-gray-500">
                  <LinkIcon className="h-5 w-5" />
                </div>
                <input
                  type="url"
                  id="pdf-url"
                  className={`flex-1 min-w-0 block w-full px-3 py-2 rounded-none rounded-r-md border ${
                    urlError ? 'border-red-500 focus:ring-red-500 focus:border-red-500' : 'border-gray-300 focus:ring-indigo-500 focus:border-indigo-500'
                  }`}
                  placeholder="https://example.com/document.pdf"
                  value={url}
                  onChange={handleUrlChange}
                />
              </div>
              {urlError ? (
                <p className="mt-1 text-xs text-red-600">{urlError}</p>
              ) : (
                <p className="mt-1 text-xs text-gray-500">
                  Enter a direct link to a PDF document
                </p>
              )}
            </div>
          )}
        </div>
        
        <div className="px-4 py-3 bg-gray-50 flex justify-end space-x-2 rounded-b-lg">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={(activeTab === 'file' && !file) || (activeTab === 'link' && !url) || isValidatingUrl}
          >
            {isValidatingUrl ? 'Validating...' : 'Upload'}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default FileUploadModal;