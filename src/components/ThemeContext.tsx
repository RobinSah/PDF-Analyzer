import React, { createContext, useContext, useState, ReactNode } from 'react';

// Define theme types
export type ThemeType = 'light' | 'dark' | 'futuristic';

// Create a type for our theme context
type ThemeContextType = {
  currentTheme: ThemeType;
  setCurrentTheme: (theme: ThemeType) => void;
  getThemeClasses: () => {
    background: string;
    text: string;
    secondaryText: string;
    tertiaryText: string;
    card: string;
    cardHighlight: string;
    iconBg: string;
    iconColor: string;
    button: string;
    highlight: string;
    title: string;
    border: string;
  };
};

// Create the context with a default value
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Create a provider component
export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentTheme, setCurrentTheme] = useState<ThemeType>('light');

  // Function to get theme-specific classes
  const getThemeClasses = () => {
    switch (currentTheme) {
      case 'dark':
        return {
          background: 'bg-gray-900',
          text: 'text-white',
          secondaryText: 'text-gray-300',
          tertiaryText: 'text-gray-400',
          card: 'bg-gray-800',
          cardHighlight: 'bg-gray-700',
          iconBg: 'bg-indigo-900',
          iconColor: 'text-indigo-400',
          button: 'bg-indigo-600 hover:bg-indigo-700',
          highlight: 'text-indigo-400',
          title: 'text-blue-400',
          border: 'border-gray-700'
        };
      case 'futuristic':
        return {
          background: 'bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900',
          text: 'text-white',
          secondaryText: 'text-blue-100',
          tertiaryText: 'text-blue-200',
          card: 'bg-gray-900/60 backdrop-blur-md border border-blue-500/20',
          cardHighlight: 'bg-blue-900/30',
          iconBg: 'bg-gradient-to-r from-blue-500 to-purple-500',
          iconColor: 'text-white',
          button: 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700',
          highlight: 'text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400',
          title: 'text-transparent bg-clip-text bg-gradient-to-r from-blue-300 to-purple-300',
          border: 'border-blue-500/20'
        };
      default: // light
        return {
          background: 'bg-gray-50',
          text: 'text-gray-900',
          secondaryText: 'text-gray-600',
          tertiaryText: 'text-gray-500',
          card: 'bg-white',
          cardHighlight: 'bg-gray-50',
          iconBg: 'bg-indigo-100',
          iconColor: 'text-indigo-600',
          button: 'bg-indigo-600 hover:bg-indigo-700',
          highlight: 'text-indigo-600',
          title: 'text-blue-900',
          border: 'border-gray-100'
        };
    }
  };

  // Store theme in localStorage to persist across sessions
  React.useEffect(() => {
    const savedTheme = localStorage.getItem('theme') as ThemeType | null;
    if (savedTheme) {
      setCurrentTheme(savedTheme);
    }
  }, []);

  React.useEffect(() => {
    localStorage.setItem('theme', currentTheme);
  }, [currentTheme]);

  return (
    <ThemeContext.Provider value={{ currentTheme, setCurrentTheme, getThemeClasses }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Create a custom hook for using the theme context
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};