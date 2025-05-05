import React from 'react';
import { Sun, Moon, Monitor } from 'lucide-react';
import { useTheme, ThemeType } from './ThemeContext';

const ThemeSwitcher: React.FC = () => {
  const { currentTheme, setCurrentTheme } = useTheme();

  return (
    <div className="flex space-x-2">
      <button 
        onClick={() => setCurrentTheme('light')}
        className={`p-2 rounded-full transition-colors ${
          currentTheme === 'light' 
            ? 'bg-indigo-100 text-indigo-600' 
            : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
        }`}
        aria-label="Light theme"
        title="Light"
      >
        <Sun className="w-5 h-5" />
      </button>
      <button 
        onClick={() => setCurrentTheme('dark')}
        className={`p-2 rounded-full transition-colors ${
          currentTheme === 'dark' 
            ? 'bg-indigo-100 text-indigo-600' 
            : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
        }`}
        aria-label="Dark theme"
        title="Dark"
      >
        <Moon className="w-5 h-5" />
      </button>
      <button 
        onClick={() => setCurrentTheme('futuristic')}
        className={`p-2 rounded-full transition-colors ${
          currentTheme === 'futuristic' 
            ? 'bg-indigo-100 text-indigo-600' 
            : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
        }`}
        aria-label="Futuristic theme"
        title="Futuristic"
      >
        <Monitor className="w-5 h-5" />
      </button>
    </div>
  );
};

export default ThemeSwitcher;