"use client";

interface FearGreedGaugeProps {
  value: number;
  label: string;
}

export function FearGreedGauge({ value, label }: FearGreedGaugeProps) {
  // Calculate rotation: 0 = -90deg (left), 100 = 90deg (right)
  const rotation = (value / 100) * 180 - 90;

  // Color based on value
  const getColor = (v: number) => {
    if (v <= 25) return "#ff2e2e"; // Red-hot - Extreme Fear
    if (v <= 45) return "#ff6b00"; // Orange-accent - Fear
    if (v <= 55) return "#ffeb3b"; // Yellow-electric - Neutral
    if (v <= 75) return "#ff6b00"; // Orange-accent - Greed
    return "#ff2e2e"; // Red-hot - Extreme Greed
  };

  const color = getColor(value);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-48 h-24 overflow-hidden">
        {/* Background arc */}
        <svg viewBox="0 0 200 100" className="w-full h-full">
          {/* Gradient arc background */}
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ff2e2e" />
              <stop offset="25%" stopColor="#ff6b00" />
              <stop offset="50%" stopColor="#ffeb3b" />
              <stop offset="75%" stopColor="#ff6b00" />
              <stop offset="100%" stopColor="#ff2e2e" />
            </linearGradient>
          </defs>

          {/* Background arc */}
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="url(#gaugeGradient)"
            strokeWidth="12"
            strokeLinecap="round"
          />

          {/* Tick marks */}
          {[0, 25, 50, 75, 100].map((tick) => {
            const angle = (tick / 100) * 180 - 90;
            const radian = (angle * Math.PI) / 180;
            const x1 = 100 + 70 * Math.cos(radian);
            const y1 = 100 + 70 * Math.sin(radian);
            const x2 = 100 + 60 * Math.cos(radian);
            const y2 = 100 + 60 * Math.sin(radian);
            return (
              <line
                key={tick}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke="rgba(255,255,255,0.4)"
                strokeWidth="2"
              />
            );
          })}
        </svg>

        {/* Needle */}
        <div
          className="absolute bottom-0 left-1/2 origin-bottom transition-transform duration-700 ease-out"
          style={{
            transform: `translateX(-50%) rotate(${rotation}deg)`,
            width: "4px",
            height: "70px",
            background: `linear-gradient(to top, ${color}, transparent)`,
            borderRadius: "2px",
          }}
        />

        {/* Center dot */}
        <div
          className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-4"
          style={{ backgroundColor: color }}
        />
      </div>

      {/* Value display */}
      <div className="mt-4 text-center">
        <div className="text-4xl font-bold" style={{ color }}>
          {value}
        </div>
        <div className="text-lg font-medium mt-1" style={{ color }}>
          {label}
        </div>
      </div>

      {/* Labels */}
      <div className="flex justify-between w-full mt-2 text-xs text-white-dim">
        <span>Extreme Fear</span>
        <span>Neutral</span>
        <span>Extreme Greed</span>
      </div>
    </div>
  );
}
