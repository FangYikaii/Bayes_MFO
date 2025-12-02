import React, { useState, useEffect, useCallback, useRef } from 'react';
import OptimizationCanvas from './components/OptimizationCanvas';
import MetricsChart from './components/MetricsChart';
import { Phase, Point, AlgorithmParams, IterationResult } from './types';
import { calculateAdhesion } from './utils/simulationMath';
import { Play, RotateCcw, Beaker, CheckCircle2, Settings, RefreshCw } from 'lucide-react';

export default function App() {
  const [phase, setPhase] = useState<Phase>(Phase.IDLE);
  const [points, setPoints] = useState<Point[]>([]);
  const [iteration, setIteration] = useState(0);
  const [bestAdhesion, setBestAdhesion] = useState(0);
  const [bestUniformity, setBestUniformity] = useState(0);
  const [bestCoverage, setBestCoverage] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState<string>('Ready');
  const [useRealApi, setUseRealApi] = useState(false);
  const [hypervolumeHistory, setHypervolumeHistory] = useState<number[]>([]);
  const [showParams, setShowParams] = useState(false);
  const [metricsHistory, setMetricsHistory] = useState<Array<{iteration: number, uniformity: number, coverage: number, adhesion: number, hypervolume: number}>>([]);
  const timerRef = useRef<number | null>(null);
  const iterationRef = useRef<number>(0);
  const totalIterationsRef = useRef<number>(5);

  // Algorithm Parameters
  const [algorithmParams, setAlgorithmParams] = useState<AlgorithmParams>({
    nIter: 5,
    nInit: 5,
    batchSize: 3,
    seed: 42
  });

  // API Configuration
  const API_URL = 'http://localhost:8000/api';

  // Simulation Configuration
  const OXIDE_SAMPLES = 15;
  const ORGANIC_SAMPLES = 15;
  const HYBRID_SAMPLES = 30;
  const SPEED_MS = 300; // Time between iterations

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setPhase(Phase.IDLE);
    setPoints([]);
    setIteration(0);
    setBestAdhesion(0);
    setBestUniformity(0);
    setBestCoverage(0);
    setHypervolumeHistory([]);
    setMetricsHistory([]);
    setApiStatus('Ready');
  };

  // API Health Check
  const checkApiHealth = async () => {
    try {
      const response = await fetch(`${API_URL.replace('/api', '')}/health`);
      if (response.ok) {
        setApiStatus('API Connected');
        return true;
      } else {
        setApiStatus('API Unavailable');
        return false;
      }
    } catch (error) {
      setApiStatus('API Error');
      return false;
    }
  };

  // Get Optimization Status from API
  const getOptimizationStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/status`);
      if (response.ok) {
        const data = await response.json();
        
        // Update hypervolume history
        if (data.hypervolume_history && data.hypervolume_history.length > 0) {
          setHypervolumeHistory(data.hypervolume_history);
        }
        
        // Update iteration count
        setIteration(data.current_iteration);
        
        // Update phase based on backend response
        if (data.phase === 1) {
          setPhase(Phase.OXIDE_ONLY);
        } else if (data.phase === 2) {
          setPhase(Phase.HYBRID_SEARCH);
        }
        
        return data;
      } else {
        console.error('Failed to get optimization status');
        return null;
      }
    } catch (error) {
      console.error('Error getting optimization status:', error);
      return null;
    }
  };

  // Initialize data when component mounts
  useEffect(() => {
    const initData = async () => {
      if (useRealApi) {
        const isHealthy = await checkApiHealth();
        if (isHealthy) {
          await getOptimizationStatus();
        }
      }
    };
    
    initData();
  }, [useRealApi]);

  // Initialize Optimizer via API
  const initOptimizer = async () => {
    try {
      setIsLoading(true);
      setApiStatus('Initializing Optimizer...');
      const response = await fetch(`${API_URL}/init`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ seed: algorithmParams.seed })
      });
      
      if (response.ok) {
        const data = await response.json();
        setApiStatus('Optimizer Initialized');
        return data;
      } else {
        const error = await response.json();
        setApiStatus(`Init Failed: ${error.detail}`);
        throw new Error(error.detail);
      }
    } catch (error) {
      console.error('Failed to initialize optimizer:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  // Run single iteration via API
  const runSingleIteration = async () => {
    try {
      setApiStatus('Running Iteration...');
      const response = await fetch(`${API_URL}/optimize/step`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          simulation_flag: true,
          total_iterations: algorithmParams.nIter
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // Get actual iteration number from backend
        const actualIteration = data.iteration;
        
        setApiStatus(`Running Iteration ${actualIteration}...`);
        
        // Update hypervolume history
        if (data.hypervolume_history && data.hypervolume_history.length > 0) {
          setHypervolumeHistory(data.hypervolume_history);
        }
        
        // Get the latest iteration result
        const latestIteration = data.iteration_result;
        
        // Convert API results to frontend points format
        const newPoints = latestIteration.candidates.map((candidate: any, idx: number) => ({
          id: actualIteration * 100 + idx,
          x: candidate[0] / 21, // Normalize to 0-1
          y: candidate[1] / 10,  // Normalize to 0-1
          value: candidate[11] || 0, // Combined value
          phase: Phase.HYBRID_SEARCH, // Map API phase to frontend phase
          iteration: actualIteration
        }));
        
        // Update points
        setPoints(prev => [...prev, ...newPoints]);
        
        // Update metrics
        // Get all objectives from the latest Y values
        const latestYValues = data.iteration_result.Y;
        if (latestYValues && latestYValues.length > 0) {
          // Get the latest Y values (last batch)
          const lastBatchSize = data.iteration_result.candidates.length;
          const recentYValues = latestYValues.slice(-lastBatchSize);
          
          // Update best values for all three objectives
          let batchUniformity = 0;
          let batchCoverage = 0;
          let batchAdhesion = 0;
          
          recentYValues.forEach((y: number[]) => {
            const uniformity = y[0];
            const coverage = y[1];
            const adhesion = y[2];
            
            batchUniformity = Math.max(batchUniformity, uniformity);
            batchCoverage = Math.max(batchCoverage, coverage);
            batchAdhesion = Math.max(batchAdhesion, adhesion);
            
            setBestUniformity(prev => Math.max(prev, uniformity));
            setBestCoverage(prev => Math.max(prev, coverage));
            setBestAdhesion(prev => Math.max(prev, adhesion));
          });
          
          // Update metrics history
          setMetricsHistory(prev => [...prev, {
            iteration: actualIteration,
            uniformity: batchUniformity,
            coverage: batchCoverage,
            adhesion: batchAdhesion,
            hypervolume: data.current_hypervolume
          }]);
        }
        
        // Update iteration count with actual iteration from backend
        setIteration(actualIteration);
        
        // Update phase based on backend response
        // Map backend phase (1, 2) to frontend Phase enum
        if (data.phase === 1) {
          // Phase 1: simple systems (Oxide Only or Organic Only)
          // For visualization purposes, we'll alternate between Oxide Only and Organic Only
          const currentPhase = actualIteration % 2 === 1 ? Phase.OXIDE_ONLY : Phase.ORGANIC_ONLY;
          setPhase(currentPhase);
        } else if (data.phase === 2) {
          // Phase 2: complex systems (Hybrid Search)
          setPhase(Phase.HYBRID_SEARCH);
        }
        
        return data;
      } else {
        const error = await response.json();
        setApiStatus(`Iteration Failed: ${error.detail}`);
        throw new Error(error.detail);
      }
    } catch (error) {
      console.error('Failed to run iteration:', error);
      throw error;
    }
  };

  // Run Optimization via API with real-time updates
  const runOptimization = async () => {
    try {
      setIsLoading(true);
      setApiStatus('Running Optimization...');
      
      // Reset all states before starting new optimization
      setPoints([]);
      setIteration(0);
      setBestAdhesion(0);
      setBestUniformity(0);
      setBestCoverage(0);
      setHypervolumeHistory([]);
      setMetricsHistory([]);
      setPhase(Phase.OXIDE_ONLY);
      
      // Initialize optimizer
      await initOptimizer();
      
      // Reset iteration counter
      iterationRef.current = 0;
      totalIterationsRef.current = algorithmParams.nIter;
      
      // Run exactly nIter iterations, regardless of backend return value
      for (let i = 1; i <= algorithmParams.nIter; i++) {
        await runSingleIteration(i);
        
        // Small delay to show real-time updates
        await new Promise(resolve => setTimeout(resolve, 500));
      }
      
      setApiStatus('Optimization Complete');
      setPhase(Phase.COMPLETE);
      
    } catch (error) {
      console.error('Failed to run optimization:', error);
      setApiStatus('Optimization Failed');
      setPhase(Phase.IDLE);
    } finally {
      setIsLoading(false);
    }
  };

  // Start Simulation - Either Real API or Mock
  const startSimulation = async () => {
    if (phase !== Phase.IDLE && phase !== Phase.COMPLETE) return; // Prevent double start
    
    reset();
    
    if (useRealApi) {
      // Use Real API
      try {
        setIsLoading(true);
        setApiStatus('Connecting to API...');
        
        // Check API Health
        const isHealthy = await checkApiHealth();
        if (!isHealthy) {
          setApiStatus('API Unavailable, falling back to simulation');
          setUseRealApi(false);
          setPhase(Phase.OXIDE_ONLY);
          return;
        }
        
        // Initialize Optimizer
        await initOptimizer();
        
        // Run Optimization with algorithm parameters
        await runOptimization();
        
      } catch (error) {
        console.error('API Simulation failed:', error);
        setApiStatus('API Error, falling back to simulation');
        setUseRealApi(false);
        setPhase(Phase.OXIDE_ONLY);
      } finally {
        setIsLoading(false);
      }
    } else {
      // Use Mock Simulation
      setPhase(Phase.OXIDE_ONLY);
    }
  };

  // The "Bayesian" Solver Logic (Simulated)
  const performStep = useCallback(() => {
    setPoints(prevPoints => {
      const currentCount = prevPoints.length;
      let nextPhase = phase;
      let newPoint: Point;

      // Logic to determine Next Phase transition
      if (phase === Phase.OXIDE_ONLY && currentCount >= OXIDE_SAMPLES) {
        nextPhase = Phase.ORGANIC_ONLY;
      } else if (phase === Phase.ORGANIC_ONLY && currentCount >= OXIDE_SAMPLES + ORGANIC_SAMPLES) {
        nextPhase = Phase.HYBRID_SEARCH;
      } else if (phase === Phase.HYBRID_SEARCH && currentCount >= OXIDE_SAMPLES + ORGANIC_SAMPLES + HYBRID_SAMPLES) {
        nextPhase = Phase.COMPLETE;
      }

      if (nextPhase !== phase && nextPhase !== Phase.COMPLETE) {
        setPhase(nextPhase);
        return prevPoints; // Wait one tick for phase change visual
      }
      
      if (nextPhase === Phase.COMPLETE) {
        if (timerRef.current) clearInterval(timerRef.current);
        setPhase(Phase.COMPLETE);
        return prevPoints;
      }

      // ------------------------------------
      // "Acquisition Function" Logic Mock
      // ------------------------------------
      
      let x = 0, y = 0;

      if (nextPhase === Phase.OXIDE_ONLY) {
        // Search along X axis (y near 0)
        // Simulate "Exploration" vs "Exploitation"
        // 80% chance to explore near known best, 20% random
        const bestOxide = prevPoints.filter(p => p.phase === Phase.OXIDE_ONLY).sort((a,b) => b.value - a.value)[0];
        
        if (bestOxide && Math.random() > 0.3) {
           // Exploit: search near best found so far with some noise
           x = Math.min(1, Math.max(0, bestOxide.x + (Math.random() - 0.5) * 0.2));
        } else {
           // Explore: Random
           x = Math.random();
        }
        y = 0.02; // Keep it effectively 0 (pure formula)
      } 
      
      else if (nextPhase === Phase.ORGANIC_ONLY) {
        // Search along Y axis (x near 0)
        const bestOrganic = prevPoints.filter(p => p.phase === Phase.ORGANIC_ONLY).sort((a,b) => b.value - a.value)[0];
        
        if (bestOrganic && Math.random() > 0.3) {
            y = Math.min(1, Math.max(0, bestOrganic.y + (Math.random() - 0.5) * 0.2));
        } else {
            y = Math.random();
        }
        x = 0.02; // Keep it effectively 0
      } 
      
      else if (nextPhase === Phase.HYBRID_SEARCH) {
        // Complex Global Search
        // We use "Prior Knowledge" from Phase 1 and 2.
        // We know where X was good, and where Y was good.
        // Bayesian logic would construct a 2D surface from the 1D marginals + some interaction assumption.
        
        const bestOxide = prevPoints.filter(p => p.phase === Phase.OXIDE_ONLY).sort((a,b) => b.value - a.value)[0];
        const bestOrganic = prevPoints.filter(p => p.phase === Phase.ORGANIC_ONLY).sort((a,b) => b.value - a.value)[0];

        // Strategy:
        // 1. Try mixing the best single ingredients (The "Knowledge Transfer").
        // 2. Explore the vicinity of that mix.
        // 3. Global exploration.
        
        const r = Math.random();
        
        if (r < 0.4 && bestOxide && bestOrganic) {
            // High probability: Search near the intersection of best single components
            // This visualizes the "Simple to Complex" advantage
            x = Math.min(1, Math.max(0, bestOxide.x + (Math.random() - 0.5) * 0.3));
            y = Math.min(1, Math.max(0, bestOrganic.y + (Math.random() - 0.5) * 0.3));
        } else if (r < 0.7) {
            // Local search around GLOBAL best found so far (could be a hybrid point)
            const globalBest = prevPoints.sort((a,b) => b.value - a.value)[0];
             x = Math.min(1, Math.max(0, globalBest.x + (Math.random() - 0.5) * 0.2));
             y = Math.min(1, Math.max(0, globalBest.y + (Math.random() - 0.5) * 0.2));
        } else {
            // Pure exploration
            x = Math.random();
            y = Math.random();
        }
      }

      // Simulate multi-objective values
      const adhesion = calculateAdhesion(x, y);
      const uniformity = 0.8 + Math.random() * 0.2; // 0.8-1.0
      const coverage = 0.75 + Math.random() * 0.25; // 0.75-1.0
      
      // Combine into a single value for visualization
      const combinedValue = (adhesion * 0.4) + (uniformity * 0.3) + (coverage * 0.3);
      
      newPoint = {
        id: currentCount,
        x,
        y,
        value: combinedValue,
        phase: nextPhase
      };

      setIteration(prev => prev + 1);
      setBestAdhesion(prev => Math.max(prev, adhesion));
      setBestUniformity(prev => Math.max(prev, uniformity));
      setBestCoverage(prev => Math.max(prev, coverage));
      
      return [...prevPoints, newPoint];
    });
  }, [phase]);

  // Timer Effect
  useEffect(() => {
    if (phase !== Phase.IDLE && phase !== Phase.COMPLETE) {
      timerRef.current = window.setInterval(performStep, SPEED_MS);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [phase, performStep]);


  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-6">
      <header className="max-w-5xl mx-auto mb-8 border-b border-slate-800 pb-6">
        <div className="flex justify-between items-center flex-wrap gap-4">
            <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-sky-400 to-purple-400 bg-clip-text text-transparent flex items-center gap-3">
                <Beaker className="text-sky-400" />
                Bayesian Adhesion Optimization
                </h1>
                <p className="text-slate-400 mt-2 max-w-2xl">
                Visualizing "Transfer Learning" in formulation: Optimizing Oxide and Organic layers individually (marginals) before searching the complex Hybrid space.
                </p>
            </div>
            
            <div className="flex gap-4 items-center">
                {/* Parameter Settings Toggle */}
                <button 
                onClick={() => setShowParams(!showParams)}
                className={`p-3 rounded-lg transition-all border ${isLoading ? 'bg-slate-800 text-slate-500 cursor-not-allowed border-slate-800' : 'bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border-slate-700'}`}
                title="Parameter Settings"
                >
                <Settings size={20} />
                </button>
                
                {/* API Toggle */}
                <div className="flex items-center gap-2 bg-slate-900 px-3 py-2 rounded-lg border border-slate-800">
                    <input
                        type="checkbox"
                        id="apiToggle"
                        checked={useRealApi}
                        onChange={(e) => setUseRealApi(e.target.checked)}
                        className="w-4 h-4 text-sky-600 rounded focus:ring-sky-500"
                    />
                    <label htmlFor="apiToggle" className="text-sm font-medium text-slate-300">
                        Use Real API
                    </label>
                </div>
                
                {/* API Status */}
                <div className="text-sm font-mono px-3 py-2 bg-slate-900 rounded-lg border border-slate-800">
                    <span className={`${apiStatus === 'API Connected' ? 'text-green-400' : apiStatus.includes('Error') ? 'text-red-400' : 'text-yellow-400'}`}>
                        {apiStatus}
                    </span>
                </div>
                
                {phase === Phase.IDLE || phase === Phase.COMPLETE ? (
                    <button 
                    onClick={startSimulation}
                    disabled={isLoading}
                    className={`flex items-center gap-2 px-6 py-3 rounded-lg font-bold shadow-lg transition-all ${isLoading ? 'bg-slate-700 cursor-not-allowed' : 'bg-sky-600 hover:bg-sky-500 text-white shadow-sky-900/50'}`}
                    >
                    <Play size={20} fill="currentColor" />
                    {isLoading ? 'Loading...' : 'Start Search'}
                    </button>
                ) : (
                    <button 
                    onClick={() => {
                         if (timerRef.current) clearInterval(timerRef.current);
                         setPhase(Phase.IDLE);
                    }}
                    className="flex items-center gap-2 px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white rounded-lg font-bold transition-all"
                    >
                    Pause
                    </button>
                )}
                
                <button 
                onClick={reset}
                disabled={isLoading}
                className={`p-3 rounded-lg transition-all border ${isLoading ? 'bg-slate-800 text-slate-500 cursor-not-allowed border-slate-800' : 'bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border-slate-700'}`}
                title="Reset"
                >
                <RotateCcw size={20} />
                </button>
            </div>
        </div>
        
        {/* Algorithm Parameter Settings */}
        {showParams && (
            <div className="mt-6 bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
                <h2 className="text-sm uppercase tracking-wider text-sky-400 font-bold mb-4">Algorithm Parameters</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-slate-400 mb-2">Number of Iterations</label>
                        <input
                            type="number"
                            min="1"
                            max="50"
                            value={algorithmParams.nIter}
                            onChange={(e) => setAlgorithmParams(prev => ({ ...prev, nIter: parseInt(e.target.value) }))}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                        />
                    </div>
                    
                    <div>
                        <label className="block text-sm font-medium text-slate-400 mb-2">Initial Samples</label>
                        <input
                            type="number"
                            min="3"
                            max="20"
                            value={algorithmParams.nInit}
                            onChange={(e) => setAlgorithmParams(prev => ({ ...prev, nInit: parseInt(e.target.value) }))}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                        />
                    </div>
                    
                    <div>
                        <label className="block text-sm font-medium text-slate-400 mb-2">Batch Size</label>
                        <input
                            type="number"
                            min="1"
                            max="10"
                            value={algorithmParams.batchSize}
                            onChange={(e) => setAlgorithmParams(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                        />
                    </div>
                    
                    <div>
                        <label className="block text-sm font-medium text-slate-400 mb-2">Random Seed</label>
                        <input
                            type="number"
                            value={algorithmParams.seed}
                            onChange={(e) => setAlgorithmParams(prev => ({ ...prev, seed: parseInt(e.target.value) }))}
                            className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                        />
                    </div>
                </div>
                
                <div className="mt-4 text-xs text-slate-500">
                    <p>Adjust these parameters to control the optimization process. Click "Start Search" to apply changes.</p>
                </div>
            </div>
        )}
      </header>

      <main className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Stats & Explanation */}
        <div className="space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
                <h2 className="text-sm uppercase tracking-wider text-slate-500 font-bold mb-4">Current Status</h2>
                
                <div className="space-y-4">
                    <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                        <span className="text-slate-400">Phase</span>
                        <span className={`font-mono font-bold ${
                            phase === Phase.OXIDE_ONLY ? 'text-sky-400' :
                            phase === Phase.ORGANIC_ONLY ? 'text-emerald-400' :
                            phase === Phase.HYBRID_SEARCH ? 'text-purple-400' : 
                            'text-slate-500'
                        }`}>
                            {phase === Phase.IDLE ? 'Ready' : phase.replace('_', ' ')}
                        </span>
                    </div>

                    <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                        <span className="text-slate-400">Iteration</span>
                        <span className="font-mono text-xl">{iteration}</span>
                    </div>

                    {/* Multi-objective Metrics */}
                    <div className="space-y-3 pt-2">
                        <h3 className="text-xs uppercase tracking-wider text-slate-500 font-bold">Multi-objective Results</h3>
                        
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-800 rounded-lg p-3">
                                <div className="text-xs text-slate-500">Max Adhesion</div>
                                <div className="font-mono text-xl font-bold text-yellow-400">
                                    {bestAdhesion.toFixed(1)} N/m
                                </div>
                            </div>
                            
                            <div className="bg-slate-800 rounded-lg p-3">
                                <div className="text-xs text-slate-500">Max Uniformity</div>
                                <div className="font-mono text-xl font-bold text-purple-400">
                                    {bestUniformity.toFixed(2)}
                                </div>
                            </div>
                            
                            <div className="bg-slate-800 rounded-lg p-3">
                                <div className="text-xs text-slate-500">Max Coverage</div>
                                <div className="font-mono text-xl font-bold text-green-400">
                                    {bestCoverage.toFixed(2)}
                                </div>
                            </div>
                            
                            <div className="bg-slate-800 rounded-lg p-3">
                                <div className="text-xs text-slate-500">Hypervolume</div>
                                <div className="font-mono text-xl font-bold text-blue-400">
                                    {hypervolumeHistory.length > 0 ? hypervolumeHistory[hypervolumeHistory.length - 1].toFixed(4) : '0.0000'}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-sky-500 via-emerald-500 to-purple-500 opacity-50"></div>
                <h3 className="font-bold text-slate-200 mb-2">Algorithm Strategy</h3>
                <ul className="space-y-3 text-sm text-slate-400">
                    <li className={`flex gap-3 items-start ${phase === Phase.OXIDE_ONLY ? 'text-sky-300' : ''}`}>
                        <div className={`mt-0.5 w-4 h-4 rounded-full border flex items-center justify-center shrink-0 ${phase === Phase.OXIDE_ONLY ? 'border-sky-500 bg-sky-500/20' : 'border-slate-600'}`}>
                            {phase === Phase.OXIDE_ONLY && <div className="w-2 h-2 bg-sky-500 rounded-full animate-pulse" />}
                        </div>
                        <span>
                            <strong className="block text-slate-300">1. Marginal Search (Oxide)</strong>
                            Finding optimal parameters for the pure oxide formula.
                        </span>
                    </li>
                    <li className={`flex gap-3 items-start ${phase === Phase.ORGANIC_ONLY ? 'text-emerald-300' : ''}`}>
                         <div className={`mt-0.5 w-4 h-4 rounded-full border flex items-center justify-center shrink-0 ${phase === Phase.ORGANIC_ONLY ? 'border-emerald-500 bg-emerald-500/20' : 'border-slate-600'}`}>
                            {phase === Phase.ORGANIC_ONLY && <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />}
                        </div>
                        <span>
                            <strong className="block text-slate-300">2. Marginal Search (Organic)</strong>
                            Finding optimal parameters for the pure organic formula.
                        </span>
                    </li>
                    <li className={`flex gap-3 items-start ${phase === Phase.HYBRID_SEARCH ? 'text-purple-300' : ''}`}>
                         <div className={`mt-0.5 w-4 h-4 rounded-full border flex items-center justify-center shrink-0 ${phase === Phase.HYBRID_SEARCH ? 'border-purple-500 bg-purple-500/20' : 'border-slate-600'}`}>
                            {phase === Phase.HYBRID_SEARCH && <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />}
                        </div>
                        <span>
                            <strong className="block text-slate-300">3. Hybrid Global Search</strong>
                            Using priors from steps 1 & 2 to find the synergistic sweet spot in the mixed system.
                        </span>
                    </li>
                </ul>
                
                {phase === Phase.COMPLETE && (
                    <div className="mt-4 p-3 bg-green-900/30 border border-green-800 rounded flex items-center gap-3 text-green-300">
                        <CheckCircle2 size={20} />
                        Optimization Complete!
                    </div>
                )}
            </div>
        </div>

        {/* Center & Right: Canvas and Charts */}
        <div className="lg:col-span-2 space-y-6">
            {/* The Visualizer */}
            <OptimizationCanvas 
                width={600} 
                height={400} 
                phase={phase}
                points={points}
                onAddPoint={() => {}}
            />

            {/* The Trace Chart */}
            <MetricsChart metricsHistory={metricsHistory} />
        </div>

      </main>
    </div>
  );
}