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
    <div className="relative" style={{ paddingLeft: '40px' }}>
      <div className="bg-slate-800 rounded-lg p-4 border border-slate-700 flex flex-col" style={{ width: '600px', height: '375px' }}>
        <h3 className="text-slate-300 text-sm font-semibold mb-2 flex-shrink-0">Optimization Trace (Multi-objective)</h3>
        <div className="flex-1 min-h-0">
          <ResponsiveContainer width="100%" height="100%">
        <LineChart 
          data={metricsHistory}
          margin={{ top: 5, right: 5, bottom: 0, left: -5 }}
        >
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke="#475569" 
            opacity={0.3}
            vertical={false}
          />
          <XAxis 
            dataKey="iteration" 
            type="number"
            stroke="#94a3b8" 
            fontSize={11}
            tick={{ fill: '#cbd5e1' }}
            tickLine={{ stroke: '#64748b' }}
            axisLine={{ stroke: '#64748b' }}
            domain={[1, 'dataMax']}
            allowDecimals={false}
            label={{ 
              value: 'Iteration', 
              position: 'insideBottom', 
              offset: 2, 
              fill: '#94a3b8',
              fontSize: 12,
              fontWeight: 500
            }}
          />
          <YAxis 
            stroke="#94a3b8" 
            fontSize={11}
            tick={{ fill: '#cbd5e1' }}
            tickLine={{ stroke: '#64748b', strokeWidth: 0 }}
            axisLine={{ stroke: '#64748b' }}
            domain={[0, 1.1]} 
            tickFormatter={(value) => value.toFixed(2)} 
            width={15}
            tickMargin={5}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1e293b', 
              border: '1px solid #475569',
              borderRadius: '6px',
              color: '#f1f5f9',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)'
            }}
            itemStyle={{ color: '#f1f5f9', padding: '2px 0' }}
            labelStyle={{ color: '#cbd5e1', fontWeight: 600, marginBottom: '4px' }}
            formatter={(value: any, name: string) => [
              typeof value === 'number' ? value.toFixed(4) : value, 
              name
            ]}
            separator=": "
          />
          <Legend 
            wrapperStyle={{ paddingTop: '10px' }}
            iconType="line"
            iconSize={12}
            formatter={(value) => <span style={{ color: '#cbd5e1', fontSize: '11px' }}>{value}</span>}
          />
          <Line 
            type="monotone" 
            dataKey="uniformity" 
            stroke="#a78bfa" 
            strokeWidth={2.5} 
            dot={false}
            activeDot={{ r: 5, fill: '#a78bfa', stroke: '#7c3aed', strokeWidth: 2 }}
            name="Uniformity"
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="coverage" 
            stroke="#34d399" 
            strokeWidth={2.5} 
            dot={false}
            activeDot={{ r: 5, fill: '#34d399', stroke: '#10b981', strokeWidth: 2 }}
            name="Coverage"
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="adhesion" 
            stroke="#fbbf24" 
            strokeWidth={2.5} 
            dot={false}
            activeDot={{ r: 5, fill: '#fbbf24', stroke: '#f59e0b', strokeWidth: 2 }}
            name="Adhesion"
            isAnimationActive={false}
          />
          <Line 
            type="monotone" 
            dataKey="hypervolume" 
            stroke="#fb7185" 
            strokeWidth={2.5} 
            dot={false}
            activeDot={{ r: 5, fill: '#fb7185', stroke: '#f43f5e', strokeWidth: 2 }}
            name="Hypervolume"
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default MetricsChart;