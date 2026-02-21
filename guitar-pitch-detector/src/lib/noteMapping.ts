/**
 * Maps frequencies to musical notes.
 *
 * Uses A4 = 440 Hz as the reference pitch (standard tuning).
 * Every note is 1 semitone apart, and there are 12 semitones per octave.
 * The formula: semitones = 12 * log2(frequency / 440)
 * Cent deviation: 100 cents = 1 semitone
 */

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] as const;

export type NoteName = typeof NOTE_NAMES[number];

export interface NoteInfo {
  name: NoteName;
  octave: number;
  frequency: number;        // exact frequency of this note
  cents: number;             // deviation from perfect pitch (-50 to +50)
  midiNumber: number;        // MIDI note number
  fullName: string;          // e.g., "A4", "C#3"
}

/** Standard guitar string frequencies in standard tuning (low to high) */
export const GUITAR_STANDARD_TUNING: NoteInfo[] = [
  noteFromMidi(40), // E2
  noteFromMidi(45), // A2
  noteFromMidi(50), // D3
  noteFromMidi(55), // G3
  noteFromMidi(59), // B3
  noteFromMidi(64), // E4
];

/** Common scales with intervals from root */
export const SCALES: Record<string, { name: string; intervals: number[] }> = {
  major:            { name: 'Major',             intervals: [0, 2, 4, 5, 7, 9, 11] },
  minor:            { name: 'Natural Minor',     intervals: [0, 2, 3, 5, 7, 8, 10] },
  pentatonicMajor:  { name: 'Pentatonic Major',  intervals: [0, 2, 4, 7, 9] },
  pentatonicMinor:  { name: 'Pentatonic Minor',  intervals: [0, 3, 5, 7, 10] },
  blues:            { name: 'Blues',              intervals: [0, 3, 5, 6, 7, 10] },
  chromatic:        { name: 'Chromatic',          intervals: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] },
};

/**
 * Convert a frequency to the nearest note info.
 */
export function frequencyToNote(frequency: number): NoteInfo {
  // Number of semitones from A4 (440 Hz)
  const semitonesFromA4 = 12 * Math.log2(frequency / 440);
  const roundedSemitones = Math.round(semitonesFromA4);
  const cents = Math.round((semitonesFromA4 - roundedSemitones) * 100);

  // MIDI note number (A4 = 69)
  const midiNumber = 69 + roundedSemitones;

  // Note name and octave
  const noteIndex = ((midiNumber % 12) + 12) % 12;
  const octave = Math.floor(midiNumber / 12) - 1;
  const name = NOTE_NAMES[noteIndex];

  // Exact frequency of this note
  const exactFrequency = 440 * Math.pow(2, roundedSemitones / 12);

  return {
    name,
    octave,
    frequency: exactFrequency,
    cents,
    midiNumber,
    fullName: `${name}${octave}`,
  };
}

/**
 * Create a NoteInfo from a MIDI note number (with 0 cents deviation).
 */
function noteFromMidi(midi: number): NoteInfo {
  const semitonesFromA4 = midi - 69;
  const frequency = 440 * Math.pow(2, semitonesFromA4 / 12);
  const noteIndex = ((midi % 12) + 12) % 12;
  const octave = Math.floor(midi / 12) - 1;
  const name = NOTE_NAMES[noteIndex];

  return {
    name,
    octave,
    frequency,
    cents: 0,
    midiNumber: midi,
    fullName: `${name}${octave}`,
  };
}

/**
 * Get all notes in a scale starting from a root note.
 * Returns notes across 2 octaves (ascending and descending).
 */
export function getScaleNotes(rootNote: NoteName, scaleKey: string): NoteInfo[] {
  const scale = SCALES[scaleKey];
  if (!scale) return [];

  const rootIndex = NOTE_NAMES.indexOf(rootNote);
  if (rootIndex === -1) return [];

  // Build scale across 2 octaves starting from octave 2 (guitar-friendly range)
  const notes: NoteInfo[] = [];
  const baseOctave = 2;

  for (let octaveOffset = 0; octaveOffset < 2; octaveOffset++) {
    for (const interval of scale.intervals) {
      const midiBase = (baseOctave + 1 + octaveOffset) * 12 + rootIndex;
      const midi = midiBase + interval;
      notes.push(noteFromMidi(midi));
    }
  }

  // Add the root of the third octave to complete
  const finalMidi = (baseOctave + 3) * 12 + rootIndex;
  notes.push(noteFromMidi(finalMidi));

  return notes;
}

/**
 * Check if a detected frequency matches a target note within a tolerance.
 */
export function isNoteMatch(
  detectedFrequency: number,
  targetNote: NoteInfo,
  centsTolerance: number = 50
): { match: boolean; cents: number } {
  const semitonesFromTarget = 12 * Math.log2(detectedFrequency / targetNote.frequency);
  const cents = Math.round(semitonesFromTarget * 100);

  return {
    match: Math.abs(cents) <= centsTolerance,
    cents,
  };
}

/**
 * Get all 12 note names.
 */
export function getAllNoteNames(): readonly NoteName[] {
  return NOTE_NAMES;
}
