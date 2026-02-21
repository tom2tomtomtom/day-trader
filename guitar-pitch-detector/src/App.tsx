import { useState } from 'react';
import { usePitchDetector } from './hooks/usePitchDetector';
import { PitchMeter } from './components/PitchMeter';
import { TargetNoteMode } from './components/TargetNoteMode';
import { ScalePractice } from './components/ScalePractice';

type Mode = 'tuner' | 'target' | 'scale';

const MODES: { key: Mode; label: string; description: string }[] = [
  { key: 'tuner', label: 'Free Play', description: 'See what note you\'re playing' },
  { key: 'target', label: 'Target Note', description: 'Hit the target note' },
  { key: 'scale', label: 'Scale Practice', description: 'Play through a scale' },
];

export default function App() {
  const [mode, setMode] = useState<Mode>('tuner');
  const { isListening, currentPitch, currentNote, error, startListening, stopListening } = usePitchDetector();

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f172a',
      color: '#e2e8f0',
      fontFamily: 'system-ui, -apple-system, sans-serif',
    }}>
      <div style={{
        maxWidth: '520px',
        margin: '0 auto',
        padding: '2rem 1.5rem',
      }}>
        {/* Header */}
        <header style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h1 style={{
            fontSize: '1.5rem',
            fontWeight: 700,
            margin: 0,
            letterSpacing: '-0.02em',
          }}>
            🎸 Guitar Pitch Detector
          </h1>
          <p style={{ color: '#64748b', fontSize: '0.85rem', margin: '0.25rem 0 0' }}>
            Real-time pitch detection using the YIN algorithm
          </p>
        </header>

        {/* Mic button */}
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <button
            onClick={isListening ? stopListening : startListening}
            style={{
              padding: '0.75rem 2rem',
              borderRadius: '12px',
              border: 'none',
              background: isListening
                ? 'linear-gradient(135deg, #ef4444, #dc2626)'
                : 'linear-gradient(135deg, #6366f1, #4f46e5)',
              color: '#fff',
              fontSize: '1rem',
              fontWeight: 600,
              cursor: 'pointer',
              fontFamily: 'system-ui, sans-serif',
              boxShadow: isListening
                ? '0 4px 20px rgba(239, 68, 68, 0.3)'
                : '0 4px 20px rgba(99, 102, 241, 0.3)',
              transition: 'all 0.2s ease',
            }}
          >
            {isListening ? '⏹ Stop Listening' : '🎤 Start Listening'}
          </button>

          {error && (
            <div style={{
              marginTop: '1rem',
              padding: '0.75rem',
              borderRadius: '8px',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.2)',
              color: '#fca5a5',
              fontSize: '0.85rem',
            }}>
              {error}
            </div>
          )}
        </div>

        {/* Mode tabs */}
        <div style={{
          display: 'flex',
          gap: '4px',
          background: 'rgba(100, 116, 139, 0.08)',
          borderRadius: '12px',
          padding: '4px',
          marginBottom: '1.5rem',
        }}>
          {MODES.map((m) => (
            <button
              key={m.key}
              onClick={() => setMode(m.key)}
              title={m.description}
              style={{
                flex: 1,
                padding: '0.6rem 0.5rem',
                borderRadius: '8px',
                border: 'none',
                background: mode === m.key ? 'rgba(99, 102, 241, 0.2)' : 'transparent',
                color: mode === m.key ? '#a5b4fc' : '#64748b',
                fontSize: '0.8rem',
                fontWeight: mode === m.key ? 600 : 400,
                cursor: 'pointer',
                fontFamily: 'system-ui, sans-serif',
                transition: 'all 0.15s ease',
              }}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Active mode content */}
        {mode === 'tuner' && (
          <PitchMeter note={currentNote} pitch={currentPitch} />
        )}
        {mode === 'target' && (
          <TargetNoteMode
            currentNote={currentNote}
            currentPitch={currentPitch}
            isListening={isListening}
          />
        )}
        {mode === 'scale' && (
          <ScalePractice
            currentNote={currentNote}
            currentPitch={currentPitch}
            isListening={isListening}
          />
        )}

        {/* Footer info */}
        <footer style={{
          marginTop: '3rem',
          textAlign: 'center',
          fontSize: '0.7rem',
          color: '#475569',
          lineHeight: 1.6,
        }}>
          <p style={{ margin: 0 }}>
            Uses the YIN pitch detection algorithm for real-time frequency analysis.
          </p>
          <p style={{ margin: '0.25rem 0 0' }}>
            Works best with single notes (monophonic). For best results, play close to the mic.
          </p>
        </footer>
      </div>
    </div>
  );
}
