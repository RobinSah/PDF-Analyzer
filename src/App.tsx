import React, { useState, useEffect } from 'react';
import { FileText, MessageSquare, Zap } from 'lucide-react';
import Sidebar from './components/layout/Sidebar';
import ExtractionSection from './components/extraction/ExtractionSection';
import ChatSection from './components/chat/ChatSection';
import GenerationSection from './components/generation/GenerationSection';
import AboutSection from './components/about/AboutSection';
import AnalyticsSection from './components/analytics/AnalyticsSection';
import LandingPage from './components/LandingPage';
import { ThemeProvider } from './components/ThemeContext';
import ThemeSwitcher from './components/ThemeSwitcher';

// Define application sections
export type Section = 'landing' | 'about' | 'extraction' | 'generation' | 'chat' | 'analytics';

function App() {
  // Use a separate state to track if the check has completed
  const [checkComplete, setCheckComplete] = useState(false);
  const [activeSection, setActiveSection] = useState<Section>('landing');

  // Use useEffect to handle localStorage check
  useEffect(() => {
    // Check localStorage for hasSeenLanding flag
    const hasSeenLanding = localStorage.getItem('hasSeenLanding') === 'true';
    
    // Set active section based on localStorage
    setActiveSection(hasSeenLanding ? 'landing' : 'about');
    setCheckComplete(true);
  }, []);

  // Handle "Get Started" button click
  const handleGetStarted = () => {
    localStorage.setItem('hasSeenLanding', 'true');
    setActiveSection('about');
  };

  // Don't render anything until check is complete to avoid flashing
  if (!checkComplete) {
    return null; // or a loading spinner
  }

  return (
    <ThemeProvider>
      <div className="flex h-screen">
        {/* Only show sidebar when not on landing page */}
        {activeSection !== 'landing' && (
          <Sidebar activeSection={activeSection} setActiveSection={setActiveSection} />
        )}
        
        {/* Theme Switcher at top-right (only when not on landing) */}
        {activeSection !== 'landing' && (
          <div className="absolute top-4 right-4 z-10">
            <ThemeSwitcher />
          </div>
        )}
        
        {/* Main Content */}
        <main className={`${activeSection !== 'landing' ? 'flex-1' : 'w-full'} overflow-y-auto`}>
          {activeSection === 'landing' && <LandingPage onGetStarted={handleGetStarted} />}
          {activeSection === 'about' && <AboutSection />}
          {activeSection === 'extraction' && <ExtractionSection />}
          {activeSection === 'generation' && <GenerationSection />}
          {activeSection === 'chat' && <ChatSection />}
          {activeSection === 'analytics' && <AnalyticsSection />}
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;