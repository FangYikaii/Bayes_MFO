/**
 * Simulates a complex landscape of adhesion force.
 * X-axis: Oxide Parameter
 * Y-axis: Organic Parameter
 * 
 * Logic:
 * - There are local maxima along the axes (single formulas).
 * - There is a global maximum in the middle (hybrid), representing the synergy.
 */
export const calculateAdhesion = (x: number, y: number): number => {
  // Base Noise
  const noise = Math.sin(x * 50) * Math.cos(y * 50) * 2;

  // 1. Oxide Contribution (Along X axis, dampened as Y increases)
  // Optimal Oxide parameter around 0.3 and 0.8
  const oxideScore = (Math.exp(-Math.pow(x - 0.3, 2) / 0.05) * 60 + Math.exp(-Math.pow(x - 0.8, 2) / 0.05) * 50) * (1 - y * 0.8);

  // 2. Organic Contribution (Along Y axis, dampened as X increases)
  // Optimal Organic parameter around 0.6
  const organicScore = (Math.exp(-Math.pow(y - 0.6, 2) / 0.1) * 70) * (1 - x * 0.8);

  // 3. Hybrid Synergy (Interaction term - The Hidden Treasure)
  // High peak in the center-right area
  const hybridSynergy = Math.exp(-Math.pow(x - 0.65, 2) / 0.08) * Math.exp(-Math.pow(y - 0.75, 2) / 0.08) * 110;

  let total = oxideScore + organicScore + hybridSynergy + noise;

  // Clamp and Normalize loosely to 0-100 range for display
  return Math.max(0, Math.min(100, total));
};

export const getColorForValue = (value: number): string => {
  // Heatmap gradient: Blue(Low) -> Green -> Yellow -> Red(High)
  const normalized = value / 100;
  const hue = (1 - normalized) * 240; // 240 is Blue, 0 is Red
  return `hsl(${hue}, 100%, 50%)`;
};