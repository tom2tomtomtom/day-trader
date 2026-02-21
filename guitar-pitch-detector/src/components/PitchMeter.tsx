import type { NoteInfo } from '../lib/noteMapping';
import type { PitchResult } from '../lib/pitchDetector';

interface PitchMeterProps {
  note: NoteInfo | null;
  pitch: PitchResult | null;
}

export function PitchMeter({ note, pitch }: PitchMeterProps) {
  const cents = note?.cents ?? 0;
  const clampedCents = Math.max(-50, Math.min(50, cents));
  // Map -50..+50 to 0..100 for the gauge position
  const gaugePosition = ((clampedCents + 50) / 100) * 100;

  const getInTuneColor = (c: number): string => {
    const absCents = Math.abs(c);
    if (absCents <= 5) return '#22c55e';  // green — in tune
    if (absCents <= 15) return '#eab308'; // yellow — close
    return '#ef4444';                     // red — out of tune
  };

  return (
    <div style={{ textAlign: 'center', padding: '2rem 0' }}>
      {/* Note display */}
      <div style={{ marginBottom: '1.5rem' }}>
        {note ? (
          <>
            <div style={{
              fontSize: '6rem',
              fontWeight: 700,
              color: getInTuneColor(cents),
              lineHeight: 1,
              fontFamily: 'system-ui, sans-serif',
              transition: 'color 0.15s ease',
            }}>
              {note.name}
              <span style={{ fontSize: '2.5rem', opacity: 0.6 }}>{note.octave}</span>
            </div>
            <div style={{
              fontSize: '1.1rem',
              color: '#94a3b8',
              marginTop: '0.5rem',
            }}>
              {pitch ? `${pitch.frequency.toFixed(1)} Hz` : ''}
              {' · '}
              {cents > 0 ? '+' : ''}{cents} cents
            </div>
          </>
        ) : (
          <div style={{
            fontSize: '2rem',
            color: '#475569',
            fontFamily: 'system-ui, sans-serif',
          }}>
            Play a note...
          </div>
        )}
      </div>

      {/* Cent deviation gauge */}
      <div style={{
        position: 'relative',
        height: '40px',
        margin: '0 auto',
        maxWidth: '400px',
      }}>
        {/* Track background */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: 0,
          right: 0,
          height: '6px',
          transform: 'translateY(-50%)',
          background: 'linear-gradient(to right, #ef4444, #eab308, #22c55e, #eab308, #ef4444)',
          borderRadius: '3px',
          opacity: 0.3,
        }} />

        {/* Center marker */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          width: '2px',
          height: '24px',
          background: '#22c55e',
          transform: 'translate(-50%, -50%)',
        }} />

        {/* Indicator needle */}
        {note && (
          <div style={{
            position: 'absolute',
            top: '50%',
            left: `${gaugePosition}%`,
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: getInTuneColor(cents),
            transform: 'translate(-50%, -50%)',
            transition: 'left 0.08s ease-out',
            boxShadow: `0 0 8px ${getInTuneColor(cents)}80`,
          }} />
        )}

        {/* Labels */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          position: 'absolute',
          bottom: '-20px',
          left: 0,
          right: 0,
          fontSize: '0.75rem',
          color: '#64748b',
        }}>
          <span>♭ flat</span>
          <span>♯ sharp</span>
        </div>
      </div>

      {/* Confidence indicator */}
      {pitch && (
        <div style={{
          marginTop: '2.5rem',
          fontSize: '0.8rem',
          color: '#64748b',
        }}>
          Confidence: {(pitch.confidence * 100).toFixed(0)}%
        </div>
      )}
    </div>
  );
}
