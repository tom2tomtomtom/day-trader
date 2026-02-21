import { useState, useEffect, useRef, useCallback } from 'react';
import { type NoteInfo, type NoteName, isNoteMatch, getScaleNotes, getAllNoteNames, SCALES } from '../lib/noteMapping';
import type { PitchResult } from '../lib/pitchDetector';

interface ScalePracticeProps {
  currentNote: NoteInfo | null;
  currentPitch: PitchResult | null;
  isListening: boolean;
}

export function ScalePractice({ currentNote, currentPitch, isListening }: ScalePracticeProps) {
  const [rootNote, setRootNote] = useState<NoteName>('E');
  const [scaleKey, setScaleKey] = useState('pentatonicMinor');
  const [tolerance, setTolerance] = useState(50);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [completedNotes, setCompletedNotes] = useState<boolean[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const holdTimerRef = useRef<number | null>(null);
  const matchedRef = useRef(false);

  const scaleNotes = getScaleNotes(rootNote, scaleKey);
  const targetNote = scaleNotes[currentIndex];

  const resetPractice = useCallback(() => {
    setCurrentIndex(0);
    setCompletedNotes([]);
    setIsComplete(false);
    matchedRef.current = false;
  }, []);

  // Reset when scale changes
  useEffect(() => {
    resetPractice();
  }, [rootNote, scaleKey, resetPractice]);

  const advanceNote = useCallback(() => {
    const nextIndex = currentIndex + 1;
    setCompletedNotes((prev) => [...prev, true]);

    if (nextIndex >= scaleNotes.length) {
      setIsComplete(true);
    } else {
      setCurrentIndex(nextIndex);
      matchedRef.current = false;
    }
  }, [currentIndex, scaleNotes.length]);

  // Check if current note matches the scale target
  useEffect(() => {
    if (!currentPitch || !currentNote || !targetNote || matchedRef.current || isComplete) {
      if (holdTimerRef.current) {
        clearTimeout(holdTimerRef.current);
        holdTimerRef.current = null;
      }
      return;
    }

    const { match } = isNoteMatch(currentPitch.frequency, targetNote, tolerance);

    if (match && !holdTimerRef.current) {
      holdTimerRef.current = window.setTimeout(() => {
        matchedRef.current = true;
        setTimeout(advanceNote, 400);
      }, 400);
    } else if (!match && holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
  }, [currentPitch, currentNote, targetNote, tolerance, advanceNote, isComplete]);

  useEffect(() => {
    return () => {
      if (holdTimerRef.current) clearTimeout(holdTimerRef.current);
    };
  }, []);

  const matchResult = currentPitch && targetNote
    ? isNoteMatch(currentPitch.frequency, targetNote, tolerance)
    : null;

  return (
    <div style={{ padding: '1.5rem 0' }}>
      {/* Scale selector */}
      <div style={{
        display: 'flex',
        gap: '0.75rem',
        justifyContent: 'center',
        marginBottom: '2rem',
        flexWrap: 'wrap',
      }}>
        <div>
          <label style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '0.25rem' }}>
            Root
          </label>
          <select
            value={rootNote}
            onChange={(e) => setRootNote(e.target.value as NoteName)}
            style={selectStyle}
          >
            {getAllNoteNames().map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
        <div>
          <label style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '0.25rem' }}>
            Scale
          </label>
          <select
            value={scaleKey}
            onChange={(e) => setScaleKey(e.target.value)}
            style={selectStyle}
          >
            {Object.entries(SCALES).map(([key, scale]) => (
              <option key={key} value={key}>{scale.name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Scale note display */}
      <div style={{
        display: 'flex',
        gap: '4px',
        justifyContent: 'center',
        flexWrap: 'wrap',
        marginBottom: '2rem',
        padding: '0 1rem',
      }}>
        {scaleNotes.map((note, i) => {
          const isCurrent = i === currentIndex && !isComplete;
          const isDone = i < completedNotes.length;
          const isActive = isCurrent && matchResult?.match;

          return (
            <div
              key={`${note.fullName}-${i}`}
              style={{
                width: '44px',
                height: '44px',
                borderRadius: '8px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.8rem',
                fontWeight: isCurrent ? 700 : 500,
                color: isDone ? '#22c55e' : isCurrent ? '#fff' : '#64748b',
                background: isDone
                  ? 'rgba(34, 197, 94, 0.15)'
                  : isActive
                    ? 'rgba(234, 179, 8, 0.2)'
                    : isCurrent
                      ? 'rgba(99, 102, 241, 0.2)'
                      : 'rgba(100, 116, 139, 0.05)',
                border: `1px solid ${
                  isDone
                    ? 'rgba(34, 197, 94, 0.3)'
                    : isCurrent
                      ? 'rgba(99, 102, 241, 0.4)'
                      : 'rgba(100, 116, 139, 0.1)'
                }`,
                transition: 'all 0.15s ease',
                transform: isCurrent ? 'scale(1.1)' : 'scale(1)',
              }}
            >
              {note.name}
              <span style={{ fontSize: '0.55rem', opacity: 0.6 }}>{note.octave}</span>
            </div>
          );
        })}
      </div>

      {/* Current target or completion */}
      <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
        {isComplete ? (
          <div>
            <div style={{ fontSize: '2rem', color: '#22c55e', fontWeight: 700, marginBottom: '0.5rem' }}>
              Scale Complete!
            </div>
            <div style={{ color: '#94a3b8', marginBottom: '1rem' }}>
              {rootNote} {SCALES[scaleKey].name} — all {scaleNotes.length} notes
            </div>
            <button onClick={resetPractice} style={actionButtonStyle}>
              Play Again
            </button>
          </div>
        ) : targetNote && (
          <div>
            <div style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
              Next note ({currentIndex + 1}/{scaleNotes.length}):
            </div>
            <div style={{
              fontSize: '3.5rem',
              fontWeight: 700,
              color: matchResult?.match ? '#eab308' : '#e2e8f0',
              lineHeight: 1,
            }}>
              {targetNote.name}
              <span style={{ fontSize: '1.5rem', opacity: 0.5 }}>{targetNote.octave}</span>
            </div>
            <div style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.25rem' }}>
              {targetNote.frequency.toFixed(1)} Hz
            </div>
          </div>
        )}
      </div>

      {/* Feedback */}
      {isListening && !isComplete && (
        <div style={{
          textAlign: 'center',
          padding: '0.75rem',
          borderRadius: '8px',
          background: matchResult?.match
            ? 'rgba(234, 179, 8, 0.1)'
            : 'rgba(100, 116, 139, 0.05)',
          marginBottom: '1rem',
          fontSize: '0.85rem',
          color: matchResult?.match ? '#eab308' : '#94a3b8',
        }}>
          {matchResult?.match
            ? 'Hold steady...'
            : currentNote
              ? `Playing: ${currentNote.fullName}`
              : 'Listening...'}
        </div>
      )}

      {/* Tolerance */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        justifyContent: 'center',
      }}>
        <label style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Tolerance:</label>
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
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  padding: '0.5rem 0.75rem',
  borderRadius: '8px',
  border: '1px solid rgba(100, 116, 139, 0.2)',
  background: '#1e293b',
  color: '#e2e8f0',
  fontSize: '0.85rem',
  fontFamily: 'system-ui, sans-serif',
  cursor: 'pointer',
};

const actionButtonStyle: React.CSSProperties = {
  padding: '0.6rem 1.5rem',
  borderRadius: '8px',
  border: '1px solid rgba(34, 197, 94, 0.3)',
  background: 'rgba(34, 197, 94, 0.15)',
  color: '#22c55e',
  cursor: 'pointer',
  fontSize: '0.9rem',
  fontWeight: 600,
  fontFamily: 'system-ui, sans-serif',
};
