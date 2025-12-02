export enum Phase {
  IDLE = 'IDLE',
  INITIALIZING = 'INITIALIZING',
  OXIDE_ONLY = 'OXIDE_ONLY',
  ORGANIC_ONLY = 'ORGANIC_ONLY',
  HYBRID_SEARCH = 'HYBRID_SEARCH',
  COMPLETE = 'COMPLETE'
}

export interface Point {
  x: number; // 0-1, representing Oxide Parameter
  y: number; // 0-1, representing Organic Parameter
  value: number; // Combined objective value
  id: number;
  phase: Phase;
  iteration?: number; // Iteration number
}

export interface SimulationConfig {
  sampleCountOxide: number;
  sampleCountOrganic: number;
  sampleCountHybrid: number;
  speed: number;
}

export interface AlgorithmParams {
  nIter: number; // Number of iterations
  nInit: number; // Number of initial samples
  batchSize: number; // Batch size per iteration
  seed: number; // Random seed
}

export interface IterationResult {
  iteration: number;
  samples: number[][];
  objectives: number[][];
  phase: Phase;
  hypervolume: number;
}