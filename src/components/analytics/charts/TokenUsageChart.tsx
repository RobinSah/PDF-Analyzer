import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TokenUsageData {
  dates: string[];
  extraction: number[];
  generation: number[];
  chat: number[];
}

interface TokenUsageChartProps {
  data: TokenUsageData;
}

const TokenUsageChart: React.FC<TokenUsageChartProps> = ({ data }) => {
  // Transform data for recharts
  const chartData = data.dates.map((date, index) => ({
    date,
    Extraction: data.extraction[index],
    Generation: data.generation[index],
    Chat: data.chat[index],
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
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
          formatter={(value) => [`${value.toLocaleString()} tokens`, undefined]}
          labelFormatter={(label) => `Date: ${label}`}
        />
        <Legend />
        <Bar dataKey="Extraction" fill="#6366f1" name="Extraction" />
        <Bar dataKey="Generation" fill="#a855f7" name="Generation" />
        <Bar dataKey="Chat" fill="#3b82f6" name="Chat" />
      </BarChart>
    </ResponsiveContainer>
  );
};

export default TokenUsageChart;