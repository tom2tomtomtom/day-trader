import { useState, useRef, useCallback } from 'react';
import { createAudioPipeline, type PitchResult, type AudioPipeline } from '../lib/pitchDetector';
import { frequencyToNote, type NoteInfo } from '../lib/noteMapping';

export interface PitchState {
  isListening: boolean;
  currentPitch: PitchResult | null;
  currentNote: NoteInfo | null;
  error: string | null;
}

export function usePitchDetector() {
  const [state, setState] = useState<PitchState>({
    isListening: false,
    currentPitch: null,
    currentNote: null,
    error: null,
  });

  const pipelineRef = useRef<AudioPipeline | null>(null);

  // Smoothing: keep a short history to reduce jitter
  const historyRef = useRef<number[]>([]);
  const HISTORY_SIZE = 5;

  const smoothFrequency = useCallback((freq: number): number => {
    const history = historyRef.current;
    history.push(freq);
    if (history.length > HISTORY_SIZE) {
      history.shift();
    }

    // Median filter — more robust than mean against outliers
    const sorted = [...history].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }, []);

  const startListening = useCallback(async () => {
    if (pipelineRef.current?.isRunning()) return;

    try {
      const pipeline = createAudioPipeline((result: PitchResult | null) => {
        if (result) {
          const smoothed = smoothFrequency(result.frequency);
          const note = frequencyToNote(smoothed);
          setState({
            isListening: true,
            currentPitch: { ...result, frequency: smoothed },
            currentNote: note,
            error: null,
          });
        } else {
          setState((prev) => ({
            ...prev,
            currentPitch: null,
            currentNote: null,
          }));
        }
      });

      pipelineRef.current = pipeline;
      await pipeline.start();
      setState((prev) => ({ ...prev, isListening: true, error: null }));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to access microphone';
      setState((prev) => ({ ...prev, error: message, isListening: false }));
    }
  }, [smoothFrequency]);

  const stopListening = useCallback(() => {
    if (pipelineRef.current) {
      pipelineRef.current.stop();
      pipelineRef.current = null;
    }
    historyRef.current = [];
    setState({
      isListening: false,
      currentPitch: null,
      currentNote: null,
      error: null,
    });
  }, []);

  return {
    ...state,
    startListening,
    stopListening,
  };
}
