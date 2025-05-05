import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface CostAnalysisData {
  dates: string[];
  models: {
    [key: string]: number[];
  };
}

interface CostAnalysisChartProps {
  data: CostAnalysisData;
}

const CostAnalysisChart: React.FC<CostAnalysisChartProps> = ({ data }) => {
  // Transform data for recharts
  const chartData = data.dates.map((date, index) => {
    const entry: any = { date };
    // Add cost for each model
    Object.keys(data.models).forEach(model => {
      entry[model] = data.models[model][index];
    });
    return entry;
  });

  // Define colors for different models
  const modelColors: {[key: string]: string} = {
    'gpt-4': '#059669', // green
    'gpt-3.5-turbo': '#2563eb', // blue
    'mixtral-8x7b': '#7c3aed', // purple
    'llama-3': '#d97706' // amber
  };

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart
        data={chartData}
        margin={{
          top: 20,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip 
          formatter={(value) => [`$${parseFloat(value as string).toFixed(2)}`, undefined]}
          labelFormatter={(label) => `Date: ${label}`}
        />
        <Legend />
        {Object.keys(data.models).map((model) => (
          <Area
            key={model}
            type="monotone"
            dataKey={model}
            name={model}
            stackId="1"
            fill={modelColors[model] || '#8884d8'}
            stroke={modelColors[model] || '#8884d8'}
            fillOpacity={0.5}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
};

export default CostAnalysisChart;