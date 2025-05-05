import React from 'react';
import { TrendingUp, TrendingDown, HelpCircle } from 'lucide-react';

interface AnalyticsSummaryCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
  change?: {
    value: number;
    trend: 'up' | 'down';
  };
  tooltip?: string;
}

const AnalyticsSummaryCard: React.FC<AnalyticsSummaryCardProps> = ({
  title,
  value,
  icon,
  change,
  tooltip,
}) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm relative">
      {tooltip && (
        <div className="group absolute top-3 right-3">
          <HelpCircle className="w-4 h-4 text-gray-400" />
          <div className="absolute right-0 invisible opacity-0 transition-opacity duration-300 bg-gray-800 text-white text-xs rounded p-2 w-64 group-hover:visible group-hover:opacity-100 z-10">
            {tooltip}
          </div>
        </div>
      )}
      
      <div className="flex items-center mb-4">
        <div className="p-2 rounded-full bg-indigo-50">{icon}</div>
      </div>
      
      <h3 className="text-sm font-medium text-gray-500 mb-1">{title}</h3>
      <div className="flex items-end space-x-2">
        <p className="text-2xl font-bold text-gray-900">{value}</p>
        
        {change && (
          <div className="flex items-center mb-1">
            {change.trend === 'up' ? (
              <TrendingUp className="w-4 h-4 text-emerald-500 mr-1" />
            ) : (
              <TrendingDown className="w-4 h-4 text-rose-500 mr-1" />
            )}
            <span 
              className={`text-xs font-medium ${
                change.trend === 'up' ? 'text-emerald-600' : 'text-rose-600'
              }`}
            >
              {change.value}%
            </span>
          </div>
        )}
      </div>
      
      <p className="text-xs text-gray-500 mt-1">Compared to previous period</p>
    </div>
  );
};

export default AnalyticsSummaryCard;