import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ModelComparisonData {
  models: string[];
  tokens: number[];
  costs: number[];
  avgResponseTime: number[];
}

interface ModelComparisonChartProps {
  data: ModelComparisonData;
}

const ModelComparisonChart: React.FC<ModelComparisonChartProps> = ({ data }) => {
  const [metric, setMetric] = useState<'tokens' | 'costs' | 'time'>('tokens');

  // Transform data for recharts
  const chartData = data.models.map((model, index) => ({
    model,
    'Tokens Used': data.tokens[index],
    'Total Cost ($)': data.costs[index],
    'Avg Response Time (s)': data.avgResponseTime[index],
  }));

  // Get the appropriate data key based on the selected metric
  const getDataKey = () => {
    switch (metric) {
      case 'tokens':
        return 'Tokens Used';
      case 'costs':
        return 'Total Cost ($)';
      case 'time':
        return 'Avg Response Time (s)';
      default:
        return 'Tokens Used';
    }
  };

  // Get the appropriate color based on the selected metric
  const getBarColor = () => {
    switch (metric) {
      case 'tokens':
        return '#6366f1'; // indigo
      case 'costs':
        return '#10b981'; // emerald
      case 'time':
        return '#f97316'; // orange
      default:
        return '#6366f1';
    }
  };

  // Format the tooltip value based on the metric
  const formatTooltipValue = (value: number, name: string) => {
    if (name === 'Tokens Used') {
      return [`${value.toLocaleString()} tokens`, 'Tokens Used'];
    }
    if (name === 'Total Cost ($)') {
      return [`$${value.toFixed(2)}`, 'Total Cost'];
    }
    return [`${value}s`, 'Avg Response Time'];
  };

  return (
    <div className="h-full">
      <div className="flex space-x-4 mb-4">
        <button
          onClick={() => setMetric('tokens')}
          className={`px-3 py-1 rounded-md text-sm ${
            metric === 'tokens'
              ? 'bg-indigo-100 text-indigo-700 font-medium'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Token Usage
        </button>
        <button
          onClick={() => setMetric('costs')}
          className={`px-3 py-1 rounded-md text-sm ${
            metric === 'costs'
              ? 'bg-emerald-100 text-emerald-700 font-medium'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Cost
        </button>
        <button
          onClick={() => setMetric('time')}
          className={`px-3 py-1 rounded-md text-sm ${
            metric === 'time'
              ? 'bg-orange-100 text-orange-700 font-medium'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          Response Time
        </button>
      </div>

      <div className="h-[calc(100%-40px)]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{
              top: 20,
              right: 30,
              left: 100,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis type="category" dataKey="model" />
            <Tooltip formatter={formatTooltipValue} />
            <Legend />
            <Bar dataKey={getDataKey()} fill={getBarColor()} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ModelComparisonChart;