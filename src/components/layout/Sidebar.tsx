import React from 'react';
import { FileText, MessageSquare, Zap, Info, BarChart2 } from 'lucide-react';
import { Section } from '../../App';
import { useTheme } from '../ThemeContext';

interface SidebarProps {
  activeSection: Section;
  setActiveSection: (section: Section) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeSection, setActiveSection }) => {
  const { currentTheme, getThemeClasses } = useTheme();
  const theme = getThemeClasses();

  // Define theme-specific properties
  const sidebarBg = currentTheme === 'futuristic' 
    ? 'bg-gray-900 border-blue-900/40' 
    : currentTheme === 'dark' 
      ? 'bg-gray-900 border-gray-700' 
      : 'bg-white border-gray-200';

  const logoTextGradient = currentTheme === 'futuristic' 
    ? 'from-blue-300 to-blue-500' 
    : currentTheme === 'dark' 
      ? 'from-blue-300 to-indigo-400' 
      : 'from-[#002244] to-[#0A4A8F]';

  const userBadgeGradient = currentTheme === 'futuristic' 
    ? 'from-blue-500 to-purple-700' 
    : currentTheme === 'dark' 
      ? 'from-indigo-500 to-purple-700' 
      : 'from-purple-500 to-indigo-600';

  const userTextColor = currentTheme === 'futuristic' || currentTheme === 'dark'
    ? 'text-gray-200' 
    : 'text-gray-800';

  const userSubtextColor = currentTheme === 'futuristic' 
    ? 'text-blue-300/60' 
    : currentTheme === 'dark' 
      ? 'text-gray-400' 
      : 'text-gray-500';

  const navHoverBg = currentTheme === 'futuristic' 
    ? 'hover:bg-blue-900/30' 
    : currentTheme === 'dark' 
      ? 'hover:bg-gray-800' 
      : 'hover:bg-gray-100';

  // Modified navItems with theme-specific gradient colors
  const getNavItemColor = (id: Section) => {
    if (currentTheme === 'futuristic') {
      return 'from-blue-600 to-purple-700';
    } else if (currentTheme === 'dark') {
      switch (id) {
        case 'about':
          return 'from-blue-700 to-blue-900';
        case 'extraction':
        case 'generation':
        case 'chat':
          return 'from-indigo-700 to-purple-900';
        case 'analytics':
          return 'from-emerald-700 to-green-900';
        default:
          return 'from-indigo-700 to-purple-900';
      }
    } else {
      // Light theme - original colors
      switch (id) {
        case 'about':
          return 'from-blue-500 to-blue-700';
        case 'extraction':
        case 'generation':
        case 'chat':
          return 'from-indigo-500 to-purple-600';
        case 'analytics':
          return 'from-green-500 to-emerald-600';
        default:
          return 'from-indigo-500 to-purple-600';
      }
    }
  };

  const navItems = [
    { 
      id: 'about' as Section, 
      name: 'About', 
      icon: <Info className="w-5 h-5" />
    },
    { 
      id: 'extraction' as Section, 
      name: 'Document Preview', 
      icon: <FileText className="w-5 h-5" />
    },
    { 
      id: 'generation' as Section, 
      name: 'Extraction', 
      icon: <Zap className="w-5 h-5" />
    },
    { 
      id: 'chat' as Section, 
      name: 'Chat with PDF', 
      icon: <MessageSquare className="w-5 h-5" />
    },
    { 
      id: 'analytics' as Section, 
      name: 'Analytics', 
      icon: <BarChart2 className="w-5 h-5" />
    },
  ];

  return (
    <aside className={`w-20 md:w-64 ${sidebarBg} border-r flex flex-col transition-all duration-300 ease-in-out`}>
      {/* Logo */}
      <div className={`h-16 flex items-center justify-center md:justify-start px-4 border-b ${
        currentTheme === 'futuristic' ? 'border-blue-900/40' : 
        currentTheme === 'dark' ? 'border-gray-700' : 
        'border-gray-200'
      }`}>
        <div className="flex items-center">
          <img 
            src="https://images.seeklogo.com/logo-png/52/1/world-bank-logo-png_seeklogo-521136.png" 
            alt="World Bank Logo" 
            className="h-8 w-auto mr-2"
            onError={(e) => {
              // Fallback if the image fails to load
              e.currentTarget.style.display = 'none';
            }} 
          />
          <h2 className={`text-xs font-bold hidden md:block bg-gradient-to-r ${logoTextGradient} text-transparent bg-clip-text`}>
            WORLD BANK PAD Analyzer
          </h2>
        </div>
        <div className={`md:hidden flex items-center justify-center w-10 h-10 rounded-full bg-gradient-to-r ${
          currentTheme === 'futuristic' ? 'from-blue-600 to-blue-800' : 
          currentTheme === 'dark' ? 'from-indigo-700 to-blue-900' : 
          'from-[#002244] to-[#0A4A8F]'
        }`}>
          <FileText className="w-5 h-5 text-white" />
        </div>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 py-6">
        <ul className="space-y-2 px-2">
          {navItems.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => setActiveSection(item.id)}
                className={`w-full text-left flex items-center p-3 rounded-lg transition-all duration-200 ${
                  activeSection === item.id
                    ? `bg-gradient-to-r ${getNavItemColor(item.id)} text-white`
                    : `${navHoverBg} ${theme.text}`
                }`}
              >
                <span className="flex-shrink-0">{item.icon}</span>
                <span className="ml-3 font-medium hidden md:block">{item.name}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
      
      {/* User Profile */}
      <div className={`p-4 border-t ${
        currentTheme === 'futuristic' ? 'border-blue-900/40' : 
        currentTheme === 'dark' ? 'border-gray-700' : 
        'border-gray-200'
      } hidden md:block`}>
        <div className="flex items-center">
          <div className={`w-10 h-10 rounded-full bg-gradient-to-r ${userBadgeGradient} flex items-center justify-center`}>
            <span className="text-white font-medium">WB</span>
          </div>
          <div className="ml-3">
            <p className={`text-sm font-medium ${userTextColor}`}>User Account</p>
            <p className={`text-xs ${userSubtextColor}`}>Free Plan</p>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;