import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

interface TimeAnalysisData {
  dates: string[];
  processingTimes: number[];
  documentSizes: number[];
}

interface TimeAnalysisChartProps {
  data: TimeAnalysisData;
}

const TimeAnalysisChart: React.FC<TimeAnalysisChartProps> = ({ data }) => {
  // Transform data for recharts
  const chartData = data.dates.map((date, index) => ({
    date,
    'Processing Time (s)': data.processingTimes[index],
    'Document Size (KB)': data.documentSizes[index],
  }));

  // Calculate average response time to show as reference line
  const avgResponseTime = data.processingTimes.reduce((sum, time) => sum + time, 0) / data.processingTimes.length;

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
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
        <YAxis yAxisId="left" />
        <YAxis yAxisId="right" orientation="right" />
        <Tooltip 
          formatter={(value, name) => {
            if (name === 'Processing Time (s)') {
              return [`${value}s`, 'Processing Time'];
            }
            return [`${value} KB`, 'Document Size'];
          }}
          labelFormatter={(label) => `Date: ${label}`}
        />
        <Legend />
        <ReferenceLine 
          y={avgResponseTime} 
          yAxisId="left" 
          label="Avg Time" 
          stroke="#ff7300" 
          strokeDasharray="3 3" 
        />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="Processing Time (s)"
          stroke="#f97316"
          activeDot={{ r: 8 }}
          strokeWidth={2}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="Document Size (KB)"
          stroke="#3b82f6"
          strokeDasharray="5 5"
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default TimeAnalysisChart;