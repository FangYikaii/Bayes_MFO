import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface MetricsHistoryItem {
  iteration: number;
  uniformity: number;
  coverage: number;
  adhesion: number;
  hypervolume: number;
}

interface Props {
  metricsHistory: MetricsHistoryItem[];
}

const MetricsChart: React.FC<Props> = ({ metricsHistory }) => {
  return (
    <div className="h-80 w-full bg-slate-800 rounded-lg p-4 border border-slate-700">
      <h3 className="text-slate-300 text-sm font-semibold mb-2">Optimization Trace (Multi-objective)</h3>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={metricsHistory}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis 
            dataKey="iteration" 
            stroke="#94a3b8" 
            fontSize={12} 
            label={{ value: 'Iteration', position: 'insideBottomRight', offset: -5, fill: '#94a3b8' }} 
          />
          <YAxis 
            stroke="#94a3b8" 
            fontSize={12} 
            domain={[0, 1.1]} 
            tickFormatter={(value) => value.toFixed(2)} 
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f1f5f9' }}
            itemStyle={{ color: '#f1f5f9' }}
            formatter={(value) => [value.toFixed(3), '']}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="uniformity" 
            stroke="#8884d8" 
            strokeWidth={2} 
            dot={{ r: 3, fill: '#8884d8' }} 
            name="Uniformity"
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="coverage" 
            stroke="#82ca9d" 
            strokeWidth={2} 
            dot={{ r: 3, fill: '#82ca9d' }} 
            name="Coverage"
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="adhesion" 
            stroke="#ffc658" 
            strokeWidth={2} 
            dot={{ r: 3, fill: '#ffc658' }} 
            name="Adhesion"
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="hypervolume" 
            stroke="#ff8042" 
            strokeWidth={2} 
            dot={{ r: 3, fill: '#ff8042' }} 
            name="Hypervolume"
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MetricsChart;