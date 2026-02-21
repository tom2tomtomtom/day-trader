import { useState, useEffect, useCallback, useRef } from 'react';
import { type NoteInfo, isNoteMatch, GUITAR_STANDARD_TUNING } from '../lib/noteMapping';
import type { PitchResult } from '../lib/pitchDetector';

interface TargetNoteModeProps {
  currentNote: NoteInfo | null;
  currentPitch: PitchResult | null;
  isListening: boolean;
}

// Notes commonly played on guitar (open strings + first few frets)
const TARGET_NOTES: NoteInfo[] = GUITAR_STANDARD_TUNING;

interface ScoreEntry {
  target: string;
  hit: boolean;
  cents: number;
}

export function TargetNoteMode({ currentNote, currentPitch, isListening }: TargetNoteModeProps) {
  const [targetIndex, setTargetIndex] = useState(0);
  const [tolerance, setTolerance] = useState(50);
  const [score, setScore] = useState<ScoreEntry[]>([]);
  const [isMatched, setIsMatched] = useState(false);
  const [streakCount, setStreakCount] = useState(0);
  const holdTimerRef = useRef<number | null>(null);
  const matchedRef = useRef(false);

  const targetNote = TARGET_NOTES[targetIndex];

  const advanceTarget = useCallback(() => {
    setTargetIndex((prev) => (prev + 1) % TARGET_NOTES.length);
    setIsMatched(false);
    matchedRef.current = false;
  }, []);

  // Check if current note matches target
  useEffect(() => {
    if (!currentPitch || !currentNote || matchedRef.current) {
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
        holdTimerRef.current = null;
      }
      return;
    }

    const { match, cents } = isNoteMatch(currentPitch.frequency, targetNote, tolerance);

    if (match && !holdTimerRef.current) {
      // Require holding the note for 500ms to count as a hit
      holdTimerRef.current = window.setTimeout(() => {
        matchedRef.current = true;
        setIsMatched(true);
        setScore((prev) => [...prev, { target: targetNote.fullName, hit: true, cents }]);
        setStreakCount((prev) => prev + 1);

        // Auto-advance after a short delay
        setTimeout(advanceTarget, 800);
      }, 500);
    } else if (!match && holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
  }, [currentPitch, currentNote, targetNote, tolerance, advanceTarget]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (holdTimerRef.current) clearTimeout(holdTimerRef.current);
    };
  }, []);

  const skipNote = () => {
    setScore((prev) => [...prev, { target: targetNote.fullName, hit: false, cents: 0 }]);
    setStreakCount(0);
    advanceTarget();
  };

  const resetScore = () => {
    setScore([]);
    setStreakCount(0);
    setTargetIndex(0);
  };

  const hits = score.filter((s) => s.hit).length;
  const total = score.length;
  const accuracy = total > 0 ? Math.round((hits / total) * 100) : 0;

  const matchResult = currentPitch
    ? isNoteMatch(currentPitch.frequency, targetNote, tolerance)
    : null;

  return (
    <div style={{ padding: '1.5rem 0' }}>
      {/* Target note display */}
      <div style={{
        textAlign: 'center',
        marginBottom: '2rem',
      }}>
        <div style={{ fontSize: '0.9rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
          Play this note:
        </div>
        <div style={{
          fontSize: '5rem',
          fontWeight: 700,
          color: isMatched ? '#22c55e' : '#e2e8f0',
          lineHeight: 1,
          transition: 'color 0.2s ease, transform 0.2s ease',
          transform: isMatched ? 'scale(1.1)' : 'scale(1)',
        }}>
          {targetNote.name}
          <span style={{ fontSize: '2rem', opacity: 0.5 }}>{targetNote.octave}</span>
        </div>
        <div style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '0.25rem' }}>
          {targetNote.frequency.toFixed(1)} Hz
        </div>
      </div>

      {/* Match feedback */}
      {isListening && (
        <div style={{
          textAlign: 'center',
          padding: '1rem',
          borderRadius: '12px',
          background: isMatched
            ? 'rgba(34, 197, 94, 0.1)'
            : matchResult?.match
              ? 'rgba(234, 179, 8, 0.1)'
              : 'rgba(100, 116, 139, 0.05)',
          border: `1px solid ${
            isMatched
              ? 'rgba(34, 197, 94, 0.3)'
              : matchResult?.match
                ? 'rgba(234, 179, 8, 0.3)'
                : 'rgba(100, 116, 139, 0.1)'
          }`,
          marginBottom: '1.5rem',
          transition: 'all 0.15s ease',
        }}>
          {isMatched ? (
            <span style={{ color: '#22c55e', fontSize: '1.2rem', fontWeight: 600 }}>
              ✓ Got it!
            </span>
          ) : matchResult?.match ? (
            <span style={{ color: '#eab308' }}>
              Almost — hold it steady...
              ({matchResult.cents > 0 ? '+' : ''}{matchResult.cents} cents)
            </span>
          ) : currentNote ? (
            <span style={{ color: '#94a3b8' }}>
              You're playing {currentNote.fullName}
              {matchResult ? ` (${matchResult.cents > 0 ? '+' : ''}{matchResult.cents} cents off)` : ''}
            </span>
          ) : (
            <span style={{ color: '#475569' }}>Listening...</span>
          )}
        </div>
      )}

      {/* Controls */}
      <div style={{
        display: 'flex',
        gap: '0.75rem',
        justifyContent: 'center',
        marginBottom: '1.5rem',
        flexWrap: 'wrap',
      }}>
        <button onClick={skipNote} style={buttonStyle('#475569')}>
          Skip
        </button>
        <button onClick={resetScore} style={buttonStyle('#64748b')}>
          Reset
        </button>
      </div>

      {/* Tolerance slider */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        justifyContent: 'center',
        marginBottom: '1.5rem',
      }}>
        <label style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
          Tolerance:
        </label>
        <input
          type="range"
          min="10"
          max="100"
          value={tolerance}
          onChange={(e) => setTolerance(Number(e.target.value))}
          style={{ width: '120px' }}
        />
        <span style={{ fontSize: '0.8rem', color: '#94a3b8', minWidth: '60px' }}>
          ±{tolerance} cents
        </span>
      </div>

      {/* Score display */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '2rem',
        padding: '1rem',
        background: 'rgba(100, 116, 139, 0.05)',
        borderRadius: '12px',
      }}>
        <ScoreStat label="Accuracy" value={`${accuracy}%`} />
        <ScoreStat label="Hits" value={`${hits}/${total}`} />
        <ScoreStat label="Streak" value={`${streakCount}`} highlight={streakCount >= 3} />
      </div>
    </div>
  );
}

function ScoreStat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontSize: '1.5rem',
        fontWeight: 700,
        color: highlight ? '#22c55e' : '#e2e8f0',
      }}>
        {value}
      </div>
      <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{label}</div>
    </div>
  );
}

function buttonStyle(color: string): React.CSSProperties {
  return {
    padding: '0.5rem 1.25rem',
    borderRadius: '8px',
    border: `1px solid ${color}40`,
    background: `${color}15`,
    color: '#e2e8f0',
    cursor: 'pointer',
    fontSize: '0.85rem',
    fontFamily: 'system-ui, sans-serif',
  };
}
