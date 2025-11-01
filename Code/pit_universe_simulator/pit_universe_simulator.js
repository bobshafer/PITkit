import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Zap, ZapOff, Info, Star } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, Legend } from 'recharts';

const N = 64;
const KERNEL_SIZE = 11;
const KERNEL_SIGMA = 3.0;
const STEPS_PER_FRAME = 5;

const create2DArray = (rows, cols, fill) => {
  return Array.from({ length: rows }, () => Array(cols).fill(fill));
};

const createGaussianKernel1D = (size, sigma) => {
  let kernel = Array(size).fill(0);
  let sum = 0;
  const center = Math.floor(size / 2);
  for (let i = 0; i < size; i++) {
    const x = i - center;
    const val = Math.exp(-(x * x) / (2 * sigma * sigma));
    kernel[i] = val;
    sum += val;
  }
  for (let i = 0; i < size; i++) {
    kernel[i] /= sum;
  }
  return kernel;
};

const KERNEL_1D = createGaussianKernel1D(KERNEL_SIZE, KERNEL_SIGMA);
const KERNEL_CENTER = Math.floor(KERNEL_SIZE / 2);

const separableConvolve = (field) => {
  const tempField = create2DArray(N, N, 0);
  const outField = create2DArray(N, N, 0);
  
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < KERNEL_SIZE; k++) {
        const jj = (j + k - KERNEL_CENTER + N) % N;
        sum += field[i][jj] * KERNEL_1D[k];
      }
      tempField[i][j] = sum;
    }
  }
  
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < KERNEL_SIZE; k++) {
        const ii = (i + k - KERNEL_CENTER + N) % N;
        sum += tempField[ii][j] * KERNEL_1D[k];
      }
      outField[i][j] = sum;
    }
  }
  return outField;
};

const usePitSimulator = (params, hasGalaxy) => {
  const [tau, setTau] = useState(0);
  const tauRef = useRef(0);
  const [isRunning, setIsRunning] = useState(false);
  
  const phiRef = useRef(create2DArray(N, N, 0));
  const kRef = useRef(create2DArray(N, N, 0));
  const galaxyCenterRef = useRef({ x: N/2, y: N/2, radius: 10 });
  
  const [history, setHistory] = useState([]);
  const [isLive, setIsLive] = useState(true);

  const calculateStats = (phi, k, fPhi) => {
    let sumDissonance = 0;
    let galaxyDissonance = 0;
    let bgDissonance = 0;
    let galaxyCount = 0;
    let bgCount = 0;
    
    const bins = {};
    let phiSum = 0;
    let kSum = 0;
    let phiSumSq = 0;
    let kSumSq = 0;
    let phiKSum = 0;
    
    const gx = galaxyCenterRef.current.x;
    const gy = galaxyCenterRef.current.y;
    const gr = galaxyCenterRef.current.radius;
    
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const p = phi[i][j];
        const k_val = k[i][j];
        const dissonance = Math.abs(k_val - fPhi[i][j]);
        sumDissonance += dissonance;
        
        const dx = i - gx;
        const dy = j - gy;
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        if (hasGalaxy && dist < gr) {
          galaxyDissonance += dissonance;
          galaxyCount++;
        } else {
          bgDissonance += dissonance;
          bgCount++;
        }
        
        const bin = Math.round(p * 100);
        bins[bin] = (bins[bin] || 0) + 1;
        
        phiSum += p;
        kSum += k_val;
        phiSumSq += p * p;
        kSumSq += k_val * k_val;
        phiKSum += p * k_val;
      }
    }
    
    const nTotal = N * N;
    const coherence = -sumDissonance / nTotal;
    
    let entropy = 0;
    for (const key in bins) {
      const prob = bins[key] / nTotal;
      if (prob > 0) {
        entropy -= prob * Math.log2(prob);
      }
    }
    
    const num = nTotal * phiKSum - phiSum * kSum;
    const denPhi = Math.sqrt(nTotal * phiSumSq - phiSum * phiSum);
    const denK = Math.sqrt(nTotal * kSumSq - kSum * kSum);
    let infoFlow = 0;
    if (denPhi > 0 && denK > 0) {
      infoFlow = num / (denPhi * denK);
    }
    
    const galaxySigma = galaxyCount > 0 ? galaxyDissonance / galaxyCount : 0;
    const bgSigma = bgCount > 0 ? bgDissonance / bgCount : 0;
    
    if (isNaN(coherence) || isNaN(entropy) || isNaN(infoFlow)) {
      return { coherence: 0, entropy: 0, infoFlow: 0, galaxySigma: 0, bgSigma: 0, error: true };
    }
    
    return { coherence, entropy, infoFlow, galaxySigma, bgSigma, error: false };
  };

  const step = useCallback(() => {
    const { mu, nu, alpha, beta } = params;
    const phi = phiRef.current;
    const k = kRef.current;
    
    const fPhi = separableConvolve(phi);
    
    const newPhi = create2DArray(N, N, 0);
    const newK = create2DArray(N, N, 0);
    
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const dissonance = k[i][j] - fPhi[i][j];
        const noise = (Math.random() - 0.5) * 2 * nu;
        
        newPhi[i][j] = (1 - alpha) * phi[i][j] + alpha * k[i][j] + (isLive ? noise : 0);
        
        const k_logistic = k[i][j] * (1.0 - k[i][j]);
        newK[i][j] = k[i][j] - beta * dissonance + (isLive ? (mu * k_logistic) : 0);
        
        newPhi[i][j] = Math.max(-10, Math.min(10, newPhi[i][j]));
        newK[i][j] = Math.max(-10, Math.min(10, newK[i][j]));
      }
    }
    
    phiRef.current = newPhi;
    kRef.current = newK;
    
    const stats = calculateStats(newPhi, newK, fPhi);
    return stats;
  }, [params, isLive, hasGalaxy]);

  const reset = useCallback(() => {
    setIsRunning(false);
    setTau(0);
    tauRef.current = 0;
    phiRef.current = create2DArray(N, N, 0);
    kRef.current = create2DArray(N, N, 0);
    
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        phiRef.current[i][j] = (Math.random() - 0.5) * 0.1;
        kRef.current[i][j] = (Math.random() - 0.5) * 0.1;
      }
    }
    
    if (hasGalaxy) {
      const gx = N / 2;
      const gy = N / 2;
      const gr = 10;
      galaxyCenterRef.current = { x: gx, y: gy, radius: gr };
      
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const dx = i - gx;
          const dy = j - gy;
          const dist = Math.sqrt(dx*dx + dy*dy);
          if (dist < gr) {
            const amplitude = 2.0 * Math.exp(-(dist*dist) / (2 * (gr/2) * (gr/2)));
            phiRef.current[i][j] += amplitude;
          }
        }
      }
    }
    
    setHistory([]);
  }, [hasGalaxy]);

  useEffect(() => {
    let animationFrameId;
    if (isRunning) {
      const runFrame = () => {
        let stats;
        for (let i = 0; i < STEPS_PER_FRAME; i++) {
          tauRef.current += 1;
          stats = step();
          if (stats.error) {
            console.error("Simulation unstable. Resetting.");
            reset();
            return;
          }
        }
        
        const currentTau = tauRef.current;
        setTau(currentTau);
        setHistory(h => [
          ...h.slice(-200),
          { tau: currentTau, ...stats }
        ]);
        animationFrameId = requestAnimationFrame(runFrame);
      };
      animationFrameId = requestAnimationFrame(runFrame);
    }
    return () => cancelAnimationFrame(animationFrameId);
  }, [isRunning, step, reset]);

  useEffect(() => {
    reset();
  }, [reset]);

  return { isRunning, setIsRunning, isLive, setIsLive, history, phiRef, kRef, reset, tau };
};

const FieldCanvas = React.memo(({ fieldRef, label, hasGalaxy, galaxyCenter }) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef();

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const field = fieldRef.current;
    if (!field) return;

    const width = canvas.width;
    const height = canvas.height;
    const fieldSize = field.length;
    const pixelSize = width / fieldSize;
    
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < fieldSize; i++) {
      for (let j = 0; j < fieldSize; j++) {
        if (field[i][j] < min) min = field[i][j];
        if (field[i][j] > max) max = field[i][j];
      }
    }
    const range = Math.max(0.1, max - min);

    ctx.clearRect(0, 0, width, height);
    for (let i = 0; i < fieldSize; i++) {
      for (let j = 0; j < fieldSize; j++) {
        const val = (field[i][j] - min) / range;
        const intensity = 50 + (val * 40);
        ctx.fillStyle = `hsl(${240 * (1-val)}, 100%, ${intensity}%)`;
        ctx.fillRect(j * pixelSize, i * pixelSize, pixelSize + 1, pixelSize + 1);
      }
    }
    
    if (hasGalaxy && galaxyCenter) {
      ctx.strokeStyle = 'rgba(255, 255, 0, 0.6)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(
        galaxyCenter.y * pixelSize,
        galaxyCenter.x * pixelSize,
        galaxyCenter.radius * pixelSize,
        0,
        2 * Math.PI
      );
      ctx.stroke();
    }

    animationFrameRef.current = requestAnimationFrame(draw);
  };

  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animationFrameRef.current);
  }, [fieldRef, hasGalaxy]);

  return (
    <div className="mb-4">
      <h3 className="text-sm font-semibold mb-2 text-gray-300">{label}</h3>
      <canvas ref={canvasRef} width={256} height={256} className="bg-gray-700 rounded-md border border-gray-600"></canvas>
    </div>
  );
});

const StatChart = ({ data, dataKeys, names, colors, title }) => (
  <div className="mb-4">
    <h3 className="text-sm font-semibold mb-2 text-gray-300">{title}</h3>
    <ResponsiveContainer width="100%" height={100}>
      <LineChart data={data} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
        <XAxis dataKey="tau" stroke="#666" fontSize={10} />
        <YAxis stroke="#666" fontSize={10} domain={['auto', 'auto']} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
          labelStyle={{ color: '#9ca3af' }}
        />
        <Legend wrapperStyle={{ fontSize: '10px' }} />
        {dataKeys.map((key, idx) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={colors[idx]}
            dot={false}
            strokeWidth={2}
            name={names[idx]}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  </div>
);

const PITFieldSimulator = () => {
  const [params, setParams] = useState({
    mu: 0.005,
    nu: 0.010,
    alpha: 0.09,
    beta: 0.04,
  });
  
  const [hasGalaxy, setHasGalaxy] = useState(false);
  
  const { isRunning, setIsRunning, isLive, setIsLive, history, phiRef, kRef, reset, tau } = usePitSimulator(params, hasGalaxy);

  const ParamSlider = ({ name, label, min, max, step }) => (
    <div>
      <label className="text-xs text-gray-400 block mb-1">
        {label} = {params[name].toFixed(4)}
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={params[name]}
        onChange={(e) => setParams(prev => ({
          ...prev,
          [name]: parseFloat(e.target.value)
        }))}
        className="w-full"
      />
    </div>
  );

  return (
    <div className="w-full h-full bg-gray-900 text-gray-100 p-4 sm:p-6 overflow-auto">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-2xl sm:text-3xl font-bold mb-2">PIT Universe Simulator v2.0</h1>
        <p className="text-gray-400 mb-4 text-sm sm:text-base">With Galaxy Seeding - Testing σ_int Evolution</p>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          
          <div className="lg:col-span-1 bg-gray-800 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-4">
              <button
                onClick={() => setIsRunning(!isRunning)}
                className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  isRunning ? 'bg-blue-600 hover:bg-blue-700' : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {isRunning ? <Pause size={18} /> : <Play size={18} />}
                {isRunning ? 'Pause' : 'Start'}
              </button>
              <button
                onClick={reset}
                className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                <RotateCcw size={18} />
                Reset
              </button>
            </div>
            
            <div className="flex flex-col gap-2 mb-4">
              <button
                onClick={() => setIsLive(!isLive)}
                className={`flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-colors text-xs ${
                  isLive ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-gray-600 hover:bg-gray-500'
                }`}
              >
                {isLive ? <Zap size={16} /> : <ZapOff size={16} />}
                {isLive ? 'μ-ν ON (Live)' : 'μ-ν OFF (Frozen)'}
              </button>
              
              <button
                onClick={() => { setHasGalaxy(!hasGalaxy); reset(); }}
                className={`flex items-center justify-center gap-2 px-4 py-2 rounded-lg transition-colors text-xs ${
                  hasGalaxy ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-600 hover:bg-gray-500'
                }`}
              >
                <Star size={16} />
                {hasGalaxy ? 'Galaxy Mode ON' : 'Galaxy Mode OFF'}
              </button>
            </div>
            
            <div className="mb-4 text-sm text-gray-400">
              τ = {tau}
            </div>
            
            <h3 className="text-lg font-semibold mb-3">Parameters</h3>
            <div className="space-y-4">
              <ParamSlider name="mu" label="μ (memory)" min={0} max={0.05} step={0.0001} />
              <ParamSlider name="nu" label="ν (novelty)" min={0} max={0.05} step={0.0001} />
              <ParamSlider name="alpha" label="α (Φ → K)" min={0.01} max={0.2} step={0.001} />
              <ParamSlider name="beta" label="β (K → Φ)" min={0.001} max={0.1} step={0.001} />
            </div>
            
            <div className="mt-6 p-3 bg-gray-900 rounded border border-gray-700">
              <div className="flex items-start gap-2">
                <Info size={16} className="mt-0.5 text-blue-400 flex-shrink-0" />
                <div className="text-xs text-gray-400">
                  <p className="font-semibold mb-1">Galaxy Mode:</p>
                  <p>Seeds Φ with a Gaussian "baryonic" bump. Watch how K develops a halo and σ_galaxy (yellow) evolves vs background (cyan).</p>
                </div>
              </div>
            </div>
          </div>
          
          <div className="lg:col-span-1 bg-gray-800 rounded-lg p-4">
            <FieldCanvas 
              fieldRef={phiRef} 
              label="State Field Φ (Reality)" 
              hasGalaxy={hasGalaxy}
              galaxyCenter={{ x: N/2, y: N/2, radius: 10 }}
            />
            <FieldCanvas 
              fieldRef={kRef} 
              label="Kernel Field K (Memory)"
              hasGalaxy={hasGalaxy}
              galaxyCenter={{ x: N/2, y: N/2, radius: 10 }}
            />
          </div>
          
          <div className="lg:col-span-1 bg-gray-800 rounded-lg p-4">
            <StatChart 
              data={history} 
              dataKeys={['coherence']} 
              names={['Coherence']} 
              colors={['#3b82f6']} 
              title="Coherence (Stability)"
            />
            <StatChart 
              data={history} 
              dataKeys={['infoFlow']} 
              names={['Info Flow']} 
              colors={['#10b981']} 
              title="Info Flow (Coupling)"
            />
            {hasGalaxy && (
              <StatChart 
                data={history} 
                dataKeys={['galaxySigma', 'bgSigma']} 
                names={['Galaxy σ_int', 'Background σ_int']} 
                colors={['#eab308', '#06b6d4']} 
                title="Dissonance (σ_int Analog)"
              />
            )}
            <StatChart 
              data={history} 
              dataKeys={['entropy']} 
              names={['Entropy']} 
              colors={['#f59e0b']} 
              title="Entropy (Complexity)"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default PITFieldSimulator;
