/**
 * YIN pitch detection algorithm implementation.
 *
 * YIN is a fundamental frequency estimator designed for speech and music.
 * It's based on autocorrelation but adds several refinements for accuracy.
 * Typical accuracy is within +/- 2 cents, which is imperceptible to most ears.
 *
 * Reference: de Cheveigné, A., & Kawahara, H. (2002).
 * "YIN, a fundamental frequency estimator for speech and music."
 */

const DEFAULT_THRESHOLD = 0.15;
const DEFAULT_PROBABILITY_THRESHOLD = 0.7;

export interface PitchResult {
  frequency: number;    // Hz
  confidence: number;   // 0-1, how confident we are in the detection
  timestamp: number;    // performance.now() timestamp
}

/**
 * Step 1 & 2 of YIN: Compute the difference function and cumulative mean
 * normalized difference function.
 */
function cumulativeMeanNormalizedDifference(buffer: Float32Array, halfBufferSize: number): Float32Array {
  const yinBuffer = new Float32Array(halfBufferSize);
  yinBuffer[0] = 1;

  let runningSum = 0;

  for (let tau = 1; tau < halfBufferSize; tau++) {
    // Difference function
    let sum = 0;
    for (let i = 0; i < halfBufferSize; i++) {
      const delta = buffer[i] - buffer[i + tau];
      sum += delta * delta;
    }
    yinBuffer[tau] = sum;

    // Cumulative mean normalized
    runningSum += yinBuffer[tau];
    yinBuffer[tau] *= tau / runningSum;
  }

  return yinBuffer;
}

/**
 * Step 3: Absolute threshold — find the first dip below the threshold.
 */
function absoluteThreshold(yinBuffer: Float32Array, threshold: number): { tau: number; confidence: number } {
  let tau = 2;
  const length = yinBuffer.length;

  // Find the first value below the threshold where we're at a local minimum
  while (tau < length) {
    if (yinBuffer[tau] < threshold) {
      // Make sure we're at a local minimum
      while (tau + 1 < length && yinBuffer[tau + 1] < yinBuffer[tau]) {
        tau++;
      }
      return {
        tau,
        confidence: 1 - yinBuffer[tau],
      };
    }
    tau++;
  }

  // No pitch found
  return { tau: -1, confidence: 0 };
}

/**
 * Step 4: Parabolic interpolation for sub-sample accuracy.
 */
function parabolicInterpolation(yinBuffer: Float32Array, tau: number): number {
  if (tau < 1 || tau >= yinBuffer.length - 1) return tau;

  const s0 = yinBuffer[tau - 1];
  const s1 = yinBuffer[tau];
  const s2 = yinBuffer[tau + 1];

  const adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0));

  if (Math.abs(adjustment) > 1) return tau;

  return tau + adjustment;
}

/**
 * Detect pitch from an audio buffer using the YIN algorithm.
 */
export function detectPitch(
  buffer: Float32Array,
  sampleRate: number,
  threshold: number = DEFAULT_THRESHOLD,
  probabilityThreshold: number = DEFAULT_PROBABILITY_THRESHOLD
): PitchResult | null {
  const halfBufferSize = Math.floor(buffer.length / 2);

  // Step 1 & 2: CMND function
  const yinBuffer = cumulativeMeanNormalizedDifference(buffer, halfBufferSize);

  // Step 3: Absolute threshold
  const { tau, confidence } = absoluteThreshold(yinBuffer, threshold);

  if (tau === -1 || confidence < probabilityThreshold) {
    return null;
  }

  // Step 4: Parabolic interpolation
  const betterTau = parabolicInterpolation(yinBuffer, tau);

  const frequency = sampleRate / betterTau;

  // Sanity check: guitar fundamental range is roughly 70-1400 Hz
  // (drop D low E = ~73 Hz, 24th fret high E = ~1319 Hz)
  if (frequency < 60 || frequency > 1500) {
    return null;
  }

  return {
    frequency,
    confidence,
    timestamp: performance.now(),
  };
}

/**
 * Create an audio processing pipeline for real-time pitch detection.
 */
export interface AudioPipeline {
  start: () => Promise<void>;
  stop: () => void;
  isRunning: () => boolean;
}

export function createAudioPipeline(
  onPitch: (result: PitchResult | null) => void,
  bufferSize: number = 4096
): AudioPipeline {
  let audioContext: AudioContext | null = null;
  let analyserNode: AnalyserNode | null = null;
  let mediaStream: MediaStream | null = null;
  let animationFrameId: number | null = null;
  let running = false;

  const processAudio = () => {
    if (!analyserNode || !running) return;

    const buffer = new Float32Array(analyserNode.fftSize);
    analyserNode.getFloatTimeDomainData(buffer);

    // Check if there's actually audio (not just silence)
    let maxAmplitude = 0;
    for (let i = 0; i < buffer.length; i++) {
      const abs = Math.abs(buffer[i]);
      if (abs > maxAmplitude) maxAmplitude = abs;
    }

    // Only try pitch detection if there's meaningful audio
    if (maxAmplitude > 0.01) {
      const result = detectPitch(buffer, audioContext!.sampleRate);
      onPitch(result);
    } else {
      onPitch(null);
    }

    animationFrameId = requestAnimationFrame(processAudio);
  };

  return {
    async start() {
      if (running) return;

      audioContext = new AudioContext();
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      const source = audioContext.createMediaStreamSource(mediaStream);
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = bufferSize;
      source.connect(analyserNode);

      running = true;
      processAudio();
    },

    stop() {
      running = false;
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
      if (mediaStream) {
        mediaStream.getTracks().forEach((t) => t.stop());
        mediaStream = null;
      }
      if (audioContext) {
        audioContext.close();
        audioContext = null;
      }
      analyserNode = null;
    },

    isRunning() {
      return running;
    },
  };
}
