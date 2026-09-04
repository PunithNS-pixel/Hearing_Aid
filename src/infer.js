import * as ort from "onnxruntime-web/wasm";
import {
  N_FREQ_BINS,
  HOP_LENGTH,
  applyMask,
  fitToLength,
  istft,
  logMagnitude,
  stft,
} from "./dsp.js";

const GRU_HIDDEN = 128; // must match CausalCRN's gru_hidden in model.py

let sessionPromise = null;
export function getSession(modelUrl = "/model.onnx") {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
    });
  }
  return sessionPromise;
}

/**
 * Run the causal model frame-by-frame over a full waveform (the slow,
 * ~seconds part). Returns the raw spectrogram frames and raw (un-reshaped)
 * masks, so suppression_strength can be applied and reapplied afterward
 * near-instantly without re-running the network -- see applyStrengthAndReconstruct.
 *
 * @param {Float32Array} waveform mono, 16kHz
 * @param {ort.InferenceSession} session
 * @param {(progress: number) => void} [onProgress] optional 0-1 progress callback
 */
export async function computeRawMasks(waveform, session, onProgress) {
  const specFrames = stft(waveform);

  let hidden = new Float32Array(GRU_HIDDEN); // (num_layers=1, batch=1, hidden) flattened
  const rawMasks = [];
  const frameLatencies = [];

  for (let f = 0; f < specFrames.length; f++) {
    const feat = logMagnitude(specFrames[f]);
    const frameTensor = new ort.Tensor("float32", feat, [1, N_FREQ_BINS, 1]);
    const hiddenTensor = new ort.Tensor("float32", hidden, [1, 1, GRU_HIDDEN]);

    const t0 = performance.now();
    const results = await session.run({ frame: frameTensor, hidden_in: hiddenTensor });
    frameLatencies.push(performance.now() - t0);

    rawMasks.push(Float32Array.from(results.mask.data));
    hidden = Float32Array.from(results.hidden_out.data);

    if (onProgress) onProgress((f + 1) / specFrames.length);
  }

  const avgLatencyMs = frameLatencies.reduce((a, b) => a + b, 0) / frameLatencies.length;
  return { specFrames, rawMasks, avgLatencyMs, hopBudgetMs: (HOP_LENGTH / 16000) * 1000 };
}

/**
 * The fast part: reshape masks by suppressionStrength (mask ** strength,
 * see src/infer.py for the rationale) and reconstruct audio. Safe to call
 * repeatedly (e.g. on every slider drag) without touching the network.
 */
export function applyStrengthAndReconstruct(specFrames, rawMasks, targetLength, suppressionStrength = 1.0) {
  const outFrames = specFrames.map((frame, f) => {
    const rawMask = rawMasks[f];
    let mask = rawMask;
    if (suppressionStrength !== 1.0) {
      mask = new Float32Array(N_FREQ_BINS);
      for (let k = 0; k < N_FREQ_BINS; k++) {
        const clamped = Math.min(1.0, Math.max(1e-6, rawMask[k]));
        mask[k] = Math.pow(clamped, suppressionStrength);
      }
    }
    return applyMask(frame, mask);
  });

  const reconNatural = istft(outFrames);
  return fitToLength(reconNatural, targetLength);
}

/** Convenience one-shot wrapper combining both steps (used by tests/scripts). */
export async function denoiseStreaming(waveform, session, suppressionStrength = 1.0, onProgress) {
  const { specFrames, rawMasks, avgLatencyMs, hopBudgetMs } = await computeRawMasks(waveform, session, onProgress);
  const output = applyStrengthAndReconstruct(specFrames, rawMasks, waveform.length, suppressionStrength);
  return { output, avgLatencyMs, hopBudgetMs };
}
