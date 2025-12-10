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
  const [totalIterations, setTotalIterations] = useState(0);  // 所有阶段的总迭代次数
  const [bestAdhesion, setBestAdhesion] = useState(0);
  const [bestUniformity, setBestUniformity] = useState(0);
  const [bestCoverage, setBestCoverage] = useState(0);
  const [bestHypervolume, setBestHypervolume] = useState(0); // 本次运行的最大超体积
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
    seed: 42,
    phase1OxideMaxIterations: 5,
    phase1OrganicMaxIterations: 5,
    phase1ImprovementThreshold: 0.05  // 保留以兼容，但后端已不使用
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
    setTotalIterations(0);
    setBestAdhesion(0);
    setBestUniformity(0);
    setBestCoverage(0);
    setBestHypervolume(0); // 重置本次运行的最大超体积
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
          // Update best hypervolume (本次运行的最大值)
          // Hypervolume 不需要归一化，直接使用后端返回的值
          const latestHv = data.hypervolume_history[data.hypervolume_history.length - 1];
          setBestHypervolume(prev => Math.max(prev, latestHv));
        }
        
        // Update iteration count
        const currentIter = data.current_iteration || data.iteration || 0;
        const currentTotalIterations = data.total_iterations || 0;  // 获取总迭代次数
        setIteration(currentIter);
        setTotalIterations(currentTotalIterations);
        
        // Update metrics from status if available
        if (data.multi_objective_results) {
          const results = data.multi_objective_results;
          if (results.max_adhesion !== undefined) {
            const normalizedAdhesion = results.max_adhesion > 1 ? results.max_adhesion / 100 : results.max_adhesion;
            setBestAdhesion(prev => Math.max(prev, normalizedAdhesion));
          }
          if (results.max_uniformity !== undefined) {
            setBestUniformity(prev => Math.max(prev, results.max_uniformity));
          }
          if (results.max_coverage !== undefined) {
            setBestCoverage(prev => Math.max(prev, results.max_coverage));
          }
          
          // Update metrics history from status if we have the data
          // 使用总迭代次数作为图表的 x 轴，这样所有阶段的数据会连续显示
          const chartIteration = totalIterations;
          if (chartIteration > 0 && data.hypervolume_history && data.hypervolume_history.length > 0) {
            const hypervolume = results.hypervolume || data.hypervolume_history[data.hypervolume_history.length - 1] || 0;
            // Hypervolume 不需要归一化，直接使用后端返回的值
            const normalizedAdhesion = (results.max_adhesion || 0) > 1 ? (results.max_adhesion || 0) / 100 : (results.max_adhesion || 0);
            
            setMetricsHistory(prev => {
              const existingIndex = prev.findIndex(item => item.iteration === chartIteration);
              if (existingIndex < 0) {
                return [...prev, {
                  iteration: chartIteration,  // 使用总迭代次数
                  uniformity: results.max_uniformity || 0,
                  coverage: results.max_coverage || 0,
                  adhesion: normalizedAdhesion,
                  hypervolume: hypervolume  // 直接使用，不归一化
                }];
              }
              return prev;
            });
          }
        }
        
        // Update phase based on backend response
        // 使用后端返回的phase和phase_1_subphase来准确显示阶段
        // 注意：data.phase 可能是数字或字符串
        const phaseValue = typeof data.phase === 'string' ? data.phase : data.phase_number || data.phase;
        
        if (phaseValue === 1 || phaseValue === 'OXIDE ONLY' || phaseValue === 'ORGANIC OPTIMIZATION') {
          // Phase 1: 根据phase_1_subphase准确显示当前子阶段
          if (data.phase_1_subphase === 'oxide') {
            setPhase(Phase.OXIDE_ONLY);
          } else if (data.phase_1_subphase === 'organic') {
            setPhase(Phase.ORGANIC_ONLY);
          } else {
            // 如果没有subphase信息，根据 phase 字符串判断
            if (phaseValue === 'OXIDE ONLY') {
              setPhase(Phase.OXIDE_ONLY);
            } else if (phaseValue === 'ORGANIC OPTIMIZATION') {
              setPhase(Phase.ORGANIC_ONLY);
            } else {
              // 默认从oxide开始
              setPhase(Phase.OXIDE_ONLY);
            }
          }
        } else if (phaseValue === 2 || phaseValue === 'HYBRID GLOBAL SEARCH') {
          // Phase 2: complex systems (Hybrid Search)
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

  // Handle API toggle - Only check health, never auto-start optimization
  useEffect(() => {
    const handleApiToggle = async () => {
      // Stop any running timers first
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      
      if (useRealApi) {
        // Only check API health when enabling Real API
        // User must click "Start Search" to begin optimization
        await checkApiHealth();
      } else {
        // When disabling Real API, reset to safe state
        // This prevents auto-start when unchecking "Use Real API"
        setApiStatus('Ready');
        // Reset phase to IDLE to prevent timer from auto-starting mock simulation
        setPhase(Phase.IDLE);
      }
    };
    
    handleApiToggle();
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
        body: JSON.stringify({ 
          seed: algorithmParams.seed,
          phase_1_oxide_max_iterations: algorithmParams.phase1OxideMaxIterations,
          phase_1_organic_max_iterations: algorithmParams.phase1OrganicMaxIterations
          // phase_1_improvement_threshold 已移除，后端不再使用
        })
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
          simulation_flag: !useRealApi, // 如果使用真实API，则simulation_flag=false；否则为true（模拟）
          total_iterations: algorithmParams.nIter
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // Get actual iteration number from backend
        const actualIteration = data.iteration;
        const currentTotalIterations = data.total_iterations || 0;  // 获取总迭代次数
        
        // 更新总迭代次数状态
        setTotalIterations(currentTotalIterations);
        
        setApiStatus(`Running Iteration ${actualIteration} (Total: ${currentTotalIterations}/${algorithmParams.nIter})...`);
        
        // Update hypervolume history
        if (data.hypervolume_history && data.hypervolume_history.length > 0) {
          setHypervolumeHistory(data.hypervolume_history);
          // Update best hypervolume (本次运行的最大值)
          // Hypervolume 不需要归一化，直接使用后端返回的值
          const latestHv = data.hypervolume_history[data.hypervolume_history.length - 1];
          setBestHypervolume(prev => Math.max(prev, latestHv));
        }
        
        // Get the latest iteration result
        const latestIteration = data.iteration_result;
        
        // Convert API results to frontend points format
        // 根据当前阶段确定点的坐标映射
        let currentPhaseEnum = Phase.HYBRID_SEARCH;
        if (data.phase === 1) {
          if (data.phase_1_subphase === 'oxide') {
            currentPhaseEnum = Phase.OXIDE_ONLY;
          } else if (data.phase_1_subphase === 'organic') {
            currentPhaseEnum = Phase.ORGANIC_ONLY;
          }
        }
        
        // 如果有阶段切换，使用新阶段
        if (data.should_switch_phase && data.new_phase !== undefined) {
          if (data.new_phase === 1) {
            if (data.new_phase_1_subphase === 'oxide') {
              currentPhaseEnum = Phase.OXIDE_ONLY;
            } else if (data.new_phase_1_subphase === 'organic') {
              currentPhaseEnum = Phase.ORGANIC_ONLY;
            }
          } else if (data.new_phase === 2) {
            currentPhaseEnum = Phase.HYBRID_SEARCH;
          }
        }
        
        // 根据阶段映射参数到可视化坐标
        // 注意：candidates 是当前阶段的有效参数，需要根据阶段确定如何映射
        const newPoints = latestIteration.candidates.map((candidate: any, idx: number) => {
          // 对于 Phase 1 Oxide: 使用 metal 参数作为 x, y
          // 对于 Phase 1 Organic: 使用 organic 参数作为 x, y
          // 对于 Phase 2: 使用所有参数，但可视化时可能需要选择两个主要参数
          let x = 0, y = 0;
          
          if (currentPhaseEnum === Phase.OXIDE_ONLY) {
            // Phase 1 Oxide: candidate 包含 metal 参数和 condition
            // 假设前两个参数是 metal_a_type 和 metal_a_concentration
            x = candidate[0] ? (candidate[0] - 1) / 19 : 0; // metal_a_type: 1-20 -> 0-1
            y = candidate[1] ? (candidate[1] - 10) / 40 : 0; // metal_a_concentration: 10-50 -> 0-1
          } else if (currentPhaseEnum === Phase.ORGANIC_ONLY) {
            // Phase 1 Organic: candidate 包含 organic 参数和 condition
            // 假设前两个参数是 organic_formula 和 organic_concentration
            x = candidate[0] ? (candidate[0] - 1) / 29 : 0; // organic_formula: 1-30 -> 0-1
            y = candidate[1] ? (candidate[1] - 0.1) / 4.9 : 0; // organic_concentration: 0.1-5 -> 0-1
          } else {
            // Phase 2: 使用混合参数，选择代表性的两个参数
            // 使用 organic_formula 和 metal_a_type 作为 x, y
            x = candidate[0] ? (candidate[0] - 1) / 29 : 0; // 假设第一个是 organic_formula
            y = candidate[6] ? (candidate[6] - 1) / 19 : 0; // 假设第7个是 metal_a_type (索引6)
          }
          
          // 获取对应的 Y 值（从所有 Y 值中取最后 candidatesCount 个）
          const allYValues = latestIteration?.Y || [];
          const candidatesCount = latestIteration?.candidates?.length || 0;
          let yValue = 0;
          if (allYValues && allYValues.length > 0 && candidatesCount > 0) {
            // 获取最新一批的 Y 值
            const recentYValues = allYValues.slice(-candidatesCount);
            if (recentYValues[idx] && Array.isArray(recentYValues[idx]) && recentYValues[idx].length >= 3) {
              // 使用三个目标的平均值作为 value
              const uniformity = recentYValues[idx][0];
              const coverage = recentYValues[idx][1];
              const adhesion = recentYValues[idx][2];
              const normalizedAdhesion = adhesion > 1 ? adhesion / 100 : adhesion;
              yValue = (uniformity + coverage + normalizedAdhesion) / 3;
            }
          }
          
          return {
            id: actualIteration * 100 + idx,
            x: Math.max(0, Math.min(1, x)), // 确保在 0-1 范围内
            y: Math.max(0, Math.min(1, y)),
            value: yValue,
            phase: currentPhaseEnum,
            iteration: actualIteration
          };
        });
        
        // Update points
        setPoints(prev => [...prev, ...newPoints]);
        
        // Update metrics
        // Get all objectives from the latest Y values
        // latestIteration.Y 包含所有历史样本的目标值，我们需要获取最新一批的
        const allYValues = latestIteration?.Y || [];
        const candidatesCount = latestIteration?.candidates?.length || 0;
        let batchUniformity = 0;
        let batchCoverage = 0;
        let batchAdhesion = 0;
        
        if (allYValues && allYValues.length > 0 && candidatesCount > 0) {
          // Get the latest Y values (last batch of candidates)
          // Y 数组包含所有历史样本，最后 candidatesCount 个是最新一批的
          const recentYValues = allYValues.slice(-candidatesCount);
          
          // Update best values for all three objectives
          recentYValues.forEach((y: number[]) => {
            if (Array.isArray(y) && y.length >= 3) {
              const uniformity = y[0];
              const coverage = y[1];
              const adhesion = y[2];
              
              batchUniformity = Math.max(batchUniformity, uniformity);
              batchCoverage = Math.max(batchCoverage, coverage);
              batchAdhesion = Math.max(batchAdhesion, adhesion);
              
              setBestUniformity(prev => Math.max(prev, uniformity));
              setBestCoverage(prev => Math.max(prev, coverage));
              // Adhesion 可能需要归一化（如果后端返回的是 0-100 范围）
              const normalizedAdhesion = adhesion > 1 ? adhesion / 100 : adhesion;
              setBestAdhesion(prev => Math.max(prev, normalizedAdhesion));
            }
          });
        }
        
        // Always update metrics history, even if no new Y values
        // Use current best values or values from data if available
        const currentHypervolume = data.current_hypervolume || 0;
        const uniformityValue = batchUniformity > 0 ? batchUniformity : (data.iteration_result?.best_objectives?.[0] || 0);
        const coverageValue = batchCoverage > 0 ? batchCoverage : (data.iteration_result?.best_objectives?.[1] || 0);
        const adhesionValue = batchAdhesion > 0 ? batchAdhesion : (data.iteration_result?.best_objectives?.[2] || 0);
        
        // Ensure values are in 0-1 range (normalize if needed)
        const normalizedAdhesion = adhesionValue > 1 ? adhesionValue / 100 : adhesionValue;
        // Hypervolume 不需要归一化，直接使用后端返回的值
        const hypervolumeValue = currentHypervolume;
        
        // Update best hypervolume (本次运行的最大值)
        setBestHypervolume(prev => Math.max(prev, hypervolumeValue));
        
        // 使用总迭代次数作为图表的 x 轴，这样所有阶段的数据会连续显示
        const chartIteration = currentTotalIterations;
        
        setMetricsHistory(prev => {
          // Check if this iteration already exists to avoid duplicates
          const existingIndex = prev.findIndex(item => item.iteration === chartIteration);
          if (existingIndex >= 0) {
            // Update existing entry
            const updated = [...prev];
            updated[existingIndex] = {
              iteration: chartIteration,  // 使用总迭代次数
              uniformity: uniformityValue,
              coverage: coverageValue,
              adhesion: normalizedAdhesion,
              hypervolume: hypervolumeValue  // 直接使用，不归一化
            };
            return updated;
          } else {
            // Add new entry
            return [...prev, {
              iteration: chartIteration,  // 使用总迭代次数
              uniformity: uniformityValue,
              coverage: coverageValue,
              adhesion: normalizedAdhesion,
              hypervolume: hypervolumeValue  // 直接使用，不归一化
            }];
          }
        });
        
        // Update iteration count with actual iteration from backend
        setIteration(actualIteration);
        
        // Update phase based on backend response
        // 优先处理阶段切换：如果有 new_phase，说明已经切换了阶段
        if (data.should_switch_phase && data.new_phase !== undefined) {
          // 阶段已经切换，使用新阶段信息
          if (data.new_phase === 1) {
            if (data.new_phase_1_subphase === 'oxide') {
              setPhase(Phase.OXIDE_ONLY);
            } else if (data.new_phase_1_subphase === 'organic') {
              setPhase(Phase.ORGANIC_ONLY);
            } else {
              setPhase(Phase.OXIDE_ONLY);
            }
          } else if (data.new_phase === 2) {
            setPhase(Phase.HYBRID_SEARCH);
          }
        } else {
          // 没有阶段切换，使用当前阶段信息
          if (data.phase === 1) {
            // Phase 1: simple systems (Oxide Only or Organic Only)
            // 根据phase_1_subphase准确显示当前子阶段
            if (data.phase_1_subphase === 'oxide') {
              setPhase(Phase.OXIDE_ONLY);
            } else if (data.phase_1_subphase === 'organic') {
              setPhase(Phase.ORGANIC_ONLY);
            } else {
              // 如果没有subphase信息，默认从oxide开始
              setPhase(Phase.OXIDE_ONLY);
            }
          } else if (data.phase === 2) {
            // Phase 2: complex systems (Hybrid Search)
            setPhase(Phase.HYBRID_SEARCH);
          }
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
      
      // Clear any existing timer (shouldn't be running for API, but just in case)
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      
      // Reset all states before starting new optimization
      setPoints([]);
      setIteration(0);
      setTotalIterations(0);
      setBestAdhesion(0);
      setBestUniformity(0);
      setBestCoverage(0);
      setBestHypervolume(0); // 重置本次运行的最大超体积
      setHypervolumeHistory([]);
      setMetricsHistory([]);
      // 初始化优化阶段: 或许可以用来初始化刚开始探索的阶段，默认是氧化物阶段
      setPhase(Phase.OXIDE_ONLY);
      
      // Initialize optimizer
      await initOptimizer();
      
      // Reset iteration counter
      iterationRef.current = 0;
      totalIterationsRef.current = algorithmParams.nIter;
      
      // Run iterations with proper phase handling
      // 注意：后端会根据阶段切换逻辑自动管理迭代，前端根据总迭代次数判断是否停止
      let consecutiveErrors = 0;
      const maxErrors = 3;
      const targetTotalIterations = algorithmParams.nIter;  // 目标总迭代次数
      
      // 使用 while 循环，根据后端返回的总迭代次数判断是否停止
      while (true) {
        try {
          const result = await runSingleIteration();
          
          // 检查是否达到目标总迭代次数
          const currentTotalIterations = result?.total_iterations || 0;
          if (currentTotalIterations >= targetTotalIterations) {
            console.log(`Reached target total iterations: ${currentTotalIterations}/${targetTotalIterations}`);
            break;
          }
          
          // 如果后端返回阶段切换，需要重新获取状态以确保同步
          if (result && result.should_switch_phase && result.new_phase !== undefined) {
            console.log(`Phase switched: ${result.current_phase} -> ${result.new_phase}`);
            
            // 如果是切换到 Phase 2，显示初始样本信息
            if (result.phase_2_initial_samples) {
              console.log(`Phase 2 initial samples: ${result.phase_2_initial_samples.count} samples`);
              console.log(`Phase 2 initial samples X:`, result.phase_2_initial_samples.X);
              console.log(`Phase 2 initial samples Y:`, result.phase_2_initial_samples.Y);
            }
            
            // 重新获取状态以确保所有信息同步
            await getOptimizationStatus();
          }
          
          // Also fetch status to ensure all metrics are updated
          await getOptimizationStatus();
          
          // Small delay to show real-time updates
          await new Promise(resolve => setTimeout(resolve, 500));
          
          consecutiveErrors = 0; // Reset error counter on success
        } catch (error) {
          console.error(`Iteration failed:`, error);
          consecutiveErrors++;
          if (consecutiveErrors >= maxErrors) {
            throw new Error(`Too many consecutive errors (${maxErrors})`);
          }
          // Continue to next iteration even if one fails
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
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
        
        // Run Optimization with algorithm parameters
        // Note: initOptimizer() is called inside runOptimization() to avoid duplicate initialization
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

      const currentIteration = prevPoints.length + 1;
      setIteration(currentIteration);
      setBestAdhesion(prev => Math.max(prev, adhesion));
      setBestUniformity(prev => Math.max(prev, uniformity));
      setBestCoverage(prev => Math.max(prev, coverage));
      
      // Normalize adhesion to 0-1 range (adhesion is 0-100)
      const normalizedAdhesion = adhesion / 100;
      
      // Calculate hypervolume (simplified: use combined value as proxy, normalized)
      const hypervolume = combinedValue / 100;
      
      // Update metrics history
      setMetricsHistory(prev => [...prev, {
        iteration: currentIteration,
        uniformity: uniformity,
        coverage: coverage,
        adhesion: normalizedAdhesion,
        hypervolume: hypervolume
      }]);
      
      return [...prevPoints, newPoint];
    });
  }, [phase]);

  // Timer Effect - Only for Mock Simulation, not for Real API
  useEffect(() => {
    // Don't start timer if using Real API - iterations are handled by API calls
    if (useRealApi) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }
    
    // Only start timer for mock simulation
    if (phase !== Phase.IDLE && phase !== Phase.COMPLETE) {
      timerRef.current = window.setInterval(performStep, SPEED_MS);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [phase, performStep, useRealApi]);


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
                
                {/* Phase 1 Advanced Parameters */}
                <div className="mt-6 pt-6 border-t border-slate-800">
                    <h3 className="text-sm uppercase tracking-wider text-sky-400 font-bold mb-4">Phase 1 Advanced Parameters</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-slate-400 mb-2">
                                Oxide Max Iterations
                                <span className="ml-2 text-xs text-slate-500">(Phase 1氧化物阶段)</span>
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="20"
                                step="1"
                                value={algorithmParams.phase1OxideMaxIterations}
                                onChange={(e) => setAlgorithmParams(prev => ({ ...prev, phase1OxideMaxIterations: parseInt(e.target.value) }))}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                            />
                            <p className="text-xs text-slate-500 mt-1">氧化物阶段最大迭代次数</p>
                        </div>
                        
                        <div>
                            <label className="block text-sm font-medium text-slate-400 mb-2">
                                Organic Max Iterations
                                <span className="ml-2 text-xs text-slate-500">(Phase 1有机物阶段)</span>
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="20"
                                step="1"
                                value={algorithmParams.phase1OrganicMaxIterations}
                                onChange={(e) => setAlgorithmParams(prev => ({ ...prev, phase1OrganicMaxIterations: parseInt(e.target.value) }))}
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-sky-500"
                            />
                            <p className="text-xs text-slate-500 mt-1">有机物阶段最大迭代次数</p>
                        </div>
                        
                    </div>
                    <div className="mt-4 p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                        <p className="text-xs text-slate-400">
                            <strong className="text-slate-300">说明：</strong>Phase 1分为两个子阶段（氧化物和有机物）。
                            当达到各自的最大迭代次数时，将自动切换到下一阶段。Phase 2（混合阶段）将使用前两个阶段的最优参数组合作为初始样本。
                        </p>
                        <p className="text-xs text-slate-400 mt-2">
                            <strong className="text-slate-300">总迭代次数：</strong>Number of Iterations 控制所有阶段加起来的总迭代次数。
                            各阶段会按顺序运行，直到达到总迭代次数。
                        </p>
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
                            phase === Phase.COMPLETE ? 'text-green-400' :
                            'text-slate-500'
                        }`}>
                            {phase === Phase.IDLE ? 'Ready' :
                             phase === Phase.OXIDE_ONLY ? 'Oxide' :
                             phase === Phase.ORGANIC_ONLY ? 'Organic' :
                             phase === Phase.HYBRID_SEARCH ? 'Mixed' :
                             phase === Phase.COMPLETE ? 'Complete' :
                             phase.replace('_', ' ')}
                        </span>
                    </div>

                    <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                        <span className="text-slate-400">Iteration</span>
                        <span className="font-mono text-xl">{iteration}</span>
                    </div>

                    <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                        <span className="text-slate-400">Total Iterations</span>
                        <span className="font-mono text-lg">
                            {totalIterations}/{algorithmParams.nIter}
                        </span>
                    </div>

                    {/* Multi-objective Metrics */}
                    <div className="space-y-3 pt-2">
                        <h3 className="text-xs uppercase tracking-wider text-slate-500 font-bold">Multi-objective Results</h3>
                        
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-slate-800 rounded-lg p-3">
                                <div className="text-xs text-slate-500">Max Adhesion</div>
                                <div className="font-mono text-lg font-bold text-yellow-400">
                                    {bestAdhesion.toFixed(4)}
                                </div>
                            </div>
                            
                            <div className="bg-slate-800 rounded-lg p-3">
                                <div className="text-xs text-slate-500">Max Uniformity</div>
                                <div className="font-mono text-xl font-bold text-purple-400">
                                    {bestUniformity.toFixed(4)}
                                </div>
                            </div>
                            
                            <div className="bg-slate-800 rounded-lg p-3">
                                <div className="text-xs text-slate-500">Max Coverage</div>
                                <div className="font-mono text-xl font-bold text-green-400">
                                    {bestCoverage.toFixed(4)}
                                </div>
                            </div>
                            
                            <div className="bg-slate-800 rounded-lg p-3">
                                <div className="text-xs text-slate-500">Hypervolume</div>
                                <div className="font-mono text-xl font-bold text-blue-400">
                                    {bestHypervolume.toFixed(4)}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden flex flex-col" style={{ height: '375px' }}>
                <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-sky-500 via-emerald-500 to-purple-500 opacity-50"></div>
                <h3 className="font-bold text-slate-200 mb-4 flex-shrink-0 -mt-2">Algorithm Strategy</h3>
                <ul className="space-y-5 text-sm text-slate-400">
                    <li className={`flex gap-4 items-start ${phase === Phase.OXIDE_ONLY ? 'text-sky-300' : ''}`}>
                        <div className={`mt-0.5 w-4 h-4 rounded-full border flex items-center justify-center shrink-0 ${phase === Phase.OXIDE_ONLY ? 'border-sky-500 bg-sky-500/20' : 'border-slate-600'}`}>
                            {phase === Phase.OXIDE_ONLY && <div className="w-2 h-2 bg-sky-500 rounded-full animate-pulse" />}
                        </div>
                        <span className="flex-1">
                            <strong className="block text-slate-300 mb-1">1. Marginal Search (Oxide)</strong>
                            Finding optimal parameters for the pure oxide formula.
                        </span>
                    </li>
                    <li className={`flex gap-4 items-start ${phase === Phase.ORGANIC_ONLY ? 'text-emerald-300' : ''}`}>
                         <div className={`mt-0.5 w-4 h-4 rounded-full border flex items-center justify-center shrink-0 ${phase === Phase.ORGANIC_ONLY ? 'border-emerald-500 bg-emerald-500/20' : 'border-slate-600'}`}>
                            {phase === Phase.ORGANIC_ONLY && <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />}
                        </div>
                        <span className="flex-1">
                            <strong className="block text-slate-300 mb-1">2. Marginal Search (Organic)</strong>
                            Finding optimal parameters for the pure organic formula.
                        </span>
                    </li>
                    <li className={`flex gap-4 items-start ${phase === Phase.HYBRID_SEARCH ? 'text-purple-300' : ''}`}>
                         <div className={`mt-0.5 w-4 h-4 rounded-full border flex items-center justify-center shrink-0 ${phase === Phase.HYBRID_SEARCH ? 'border-purple-500 bg-purple-500/20' : 'border-slate-600'}`}>
                            {phase === Phase.HYBRID_SEARCH && <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />}
                        </div>
                        <span className="flex-1">
                            <strong className="block text-slate-300 mb-1">3. Hybrid Global Search</strong>
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
            <div className="flex justify-center">
                <OptimizationCanvas 
                    width={600} 
                    height={375} 
                    phase={phase}
                    points={points}
                    onAddPoint={() => {}}
                />
            </div>

            {/* The Trace Chart */}
            <div className="flex justify-center">
                <MetricsChart metricsHistory={metricsHistory} />
            </div>
        </div>

      </main>
    </div>
  );
}