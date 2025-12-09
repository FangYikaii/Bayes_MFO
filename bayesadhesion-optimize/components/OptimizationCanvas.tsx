import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Phase, Point } from '../types';
import { calculateAdhesion, getColorForValue } from '../utils/simulationMath';

interface Props {
  phase: Phase;
  points: Point[];
  onAddPoint: (p: Point) => void;
  width: number;
  height: number;
}

const OptimizationCanvas: React.FC<Props> = ({ phase, points, onAddPoint, width, height }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  
  // Create an offscreen canvas for the true ground truth heatmap to optimize rendering
  useEffect(() => {
    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const ctx = offscreen.getContext('2d');
    if (!ctx) return;

    const imgData = ctx.createImageData(width, height);
    for (let py = 0; py < height; py++) {
      for (let px = 0; px < width; px++) {
        // Normalize coordinates to 0-1
        const nx = px / width;
        const ny = 1 - (py / height); // Cartesian Y is up

        const val = calculateAdhesion(nx, ny);
        const hue = (1 - val / 110) * 240; // Slightly adjust max for color scaling

        // HSL to RGB conversion is complex in raw loop, simplifying to a rough pseudo-color map
        // Or using a pre-calculated palette would be faster, but this is run once.
        // Let's use a simpler RGB lerp for performance here or just HSL style drawing pixel by pixel is too slow without LUT.
        // Actually, let's just draw small rects or use fillStyle which is easier than pixel manipulation for this demo scale.
      }
    }
    
    // Efficient drawing: Draw 4x4 blocks instead of pixels
    const blockSize = 4;
    for (let py = 0; py < height; py += blockSize) {
      for (let px = 0; px < width; px += blockSize) {
        const nx = px / width;
        const ny = 1 - (py / height); 
        const val = calculateAdhesion(nx, ny);
        ctx.fillStyle = getColorForValue(val);
        ctx.fillRect(px, py, blockSize, blockSize);
      }
    }
    
    heatmapCanvasRef.current = offscreen;
  }, [width, height]);

  // Main Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // 1. Draw Background (Dark Slate)
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, width, height);

    // 2. Draw "Fog of War" / Uncertainty Logic
    // We mask the Heatmap. The heatmap is revealed based on proximity to sampled points (Gaussian Process Variance reduction)
    if (heatmapCanvasRef.current) {
      ctx.save();
      
      // Create a path that covers the whole screen
      // We will "clip" to regions we "know"
      // Actually, easier visual trick: Draw the full heatmap, then draw a semi-transparent dark layer on top, 
      // then "erase" the dark layer near points.
      
      // Step A: Draw Full Heatmap (dimmed)
      ctx.globalAlpha = 0.2;
      ctx.drawImage(heatmapCanvasRef.current, 0, 0);
      ctx.globalAlpha = 1.0;

      // Step B: Draw "Uncertainty Mask" (The dark overlay)
      // We want the areas FAR from points to be dark (Uncertainty). 
      // Areas NEAR points should be clear (High Confidence).
      
      // We'll use composite operations.
      // Create a temporary layer for the mask
      const maskCanvas = document.createElement('canvas');
      maskCanvas.width = width;
      maskCanvas.height = height;
      const maskCtx = maskCanvas.getContext('2d');
      if (maskCtx) {
        // Fill with "Fog" (Dark Gray)
        maskCtx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        maskCtx.fillRect(0, 0, width, height);

        // "Cut out" holes around known points
        maskCtx.globalCompositeOperation = 'destination-out';
        points.forEach(p => {
            const screenX = p.x * width;
            const screenY = (1 - p.y) * height;
            
            // Influence radius
            const gradient = maskCtx.createRadialGradient(screenX, screenY, 0, screenX, screenY, 80);
            gradient.addColorStop(0, 'rgba(0,0,0,1)'); // Full cut
            gradient.addColorStop(1, 'rgba(0,0,0,0)'); // No cut

            maskCtx.fillStyle = gradient;
            maskCtx.beginPath();
            maskCtx.arc(screenX, screenY, 80, 0, Math.PI * 2);
            maskCtx.fill();
        });
      }
      
      ctx.drawImage(maskCanvas, 0, 0);
      
      // Step C: Draw the Heatmap again, but only where we have "cut holes" (high confidence areas)
      // This makes the "known" areas bright and colorful.
      ctx.save();
      // Use the inverse of the mask logic visually by redrawing heatmap with a specific composite?
      // A simpler way: Just draw the mask we made on top of the faint heatmap. 
      // But we want the "revealed" area to be BRIGHT.
      
      // So: 
      // 1. Draw Bright Heatmap.
      // 2. Draw Dark Mask on top (with holes).
      // Let's redo order.
      
      // 1. Draw BRIGHT heatmap everywhere.
      ctx.globalCompositeOperation = 'source-over';
      ctx.drawImage(heatmapCanvasRef.current, 0, 0);
      
      // 2. Draw the Mask (Dark with holes) on top.
      ctx.drawImage(maskCanvas, 0, 0);
      
      ctx.restore();
    }

    // 3. Draw Grid & Axes
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath();
    // Vertical lines
    for(let i=1; i<10; i++) {
        ctx.moveTo(i * (width/10), 0);
        ctx.lineTo(i * (width/10), height);
    }
    // Horizontal lines
    for(let i=1; i<10; i++) {
        ctx.moveTo(0, i * (height/10));
        ctx.lineTo(width, i * (height/10));
    }
    ctx.stroke();
    
    // Axes Highlights
    ctx.lineWidth = 3;
    // X Axis (Oxide)
    ctx.strokeStyle = phase === Phase.OXIDE_ONLY ? '#38bdf8' : '#475569';
    ctx.beginPath();
    ctx.moveTo(0, height - 2);
    ctx.lineTo(width, height - 2);
    ctx.stroke();

    // Y Axis (Organic)
    ctx.strokeStyle = phase === Phase.ORGANIC_ONLY ? '#34d399' : '#475569';
    ctx.beginPath();
    ctx.moveTo(2, 0);
    ctx.lineTo(2, height);
    ctx.stroke();

    // 4. Draw Points
    points.forEach((p, index) => {
        const isLatest = index === points.length - 1;
        const screenX = p.x * width;
        const screenY = (1 - p.y) * height;

        ctx.beginPath();
        ctx.arc(screenX, screenY, isLatest ? 8 : 5, 0, Math.PI * 2);
        
        // Color point based on value or simply white/gold for contrast
        ctx.fillStyle = isLatest ? '#ffffff' : '#94a3b8';
        if (p.value > 90) ctx.fillStyle = '#fbbf24'; // Gold for high value
        
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();

        if (isLatest) {
            // Ping animation ring
            ctx.beginPath();
            ctx.arc(screenX, screenY, 15, 0, Math.PI * 2);
            ctx.strokeStyle = 'white';
            ctx.stroke();
        }
    });

  }, [phase, points, width, height]);

  return (
    <div className="relative" style={{ paddingLeft: '40px' }}>
      <div className="relative rounded-xl overflow-hidden shadow-2xl border border-slate-700 bg-slate-900">
        <canvas 
          ref={canvasRef} 
          width={width} 
          height={height} 
          className="block cursor-crosshair relative z-0"
        />
        
        {/* Labels - X Axis (Horizontal) */}
        <div className={`absolute bottom-2 right-4 font-mono text-sm font-bold transition-all whitespace-nowrap z-10 ${
          phase === Phase.OXIDE_ONLY ? 'text-sky-400' :
          phase === Phase.ORGANIC_ONLY ? 'text-emerald-400' :
          phase === Phase.HYBRID_SEARCH ? 'text-purple-400' :
          'text-sky-400'
        }`}>
          {phase === Phase.OXIDE_ONLY ? 'oxide parameter-A' :
           phase === Phase.ORGANIC_ONLY ? 'organic parameter-A' :
           phase === Phase.HYBRID_SEARCH ? 'hybrid parameter-A' :
           'oxide parameter-A'} &rarr;
        </div>
        
        {/* Labels - Y Axis (Vertical) - Aligned with canvas center */}
        <div className={`absolute font-mono text-sm font-bold origin-center whitespace-nowrap transition-all pointer-events-none z-10 ${
          phase === Phase.OXIDE_ONLY ? 'text-sky-400' :
          phase === Phase.ORGANIC_ONLY ? 'text-emerald-400' :
          phase === Phase.HYBRID_SEARCH ? 'text-purple-400' :
          'text-sky-400'
        }`} style={{ 
          top: '25%', 
          left: '-68px',
          transform: 'translateY(-50%) rotate(-90deg)'
        }}>
          {phase === Phase.OXIDE_ONLY ? 'oxide parameter-B' :
           phase === Phase.ORGANIC_ONLY ? 'organic parameter-B' :
           phase === Phase.HYBRID_SEARCH ? 'hybrid parameter-B' :
           'oxide parameter-B'} &rarr;
        </div>
        
        {/* Phase Indicator Overlay */}
        <div className="absolute top-2 right-2 flex flex-col items-end pointer-events-none">
          {phase === Phase.OXIDE_ONLY && (
            <div className="px-3 py-1 rounded text-xs font-bold transition-all bg-sky-500 text-white">
              Single formulation stage: oxide
            </div>
          )}
          {phase === Phase.ORGANIC_ONLY && (
            <div className="px-3 py-1 rounded text-xs font-bold transition-all bg-emerald-500 text-white">
              Single formula stage: organic
            </div>
          )}
          {phase === Phase.HYBRID_SEARCH && (
            <div className="px-3 py-1 rounded text-xs font-bold transition-all bg-purple-500 text-white">
              Multi formulation stage: Mixed
            </div>
          )}
          {phase === Phase.COMPLETE && (
            <div className="px-3 py-1 rounded text-xs font-bold transition-all bg-green-500 text-white">
              Optimization Complete
            </div>
          )}
          {phase === Phase.IDLE && (
            <div className="px-3 py-1 rounded text-xs font-bold transition-all bg-slate-800 text-slate-400">
              Ready to Start
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default OptimizationCanvas;