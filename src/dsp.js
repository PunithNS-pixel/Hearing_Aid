// STFT/iSTFT matching src/features.py exactly:
//   sample_rate=16000, win_length=320, hop_length=160, n_fft=320 (== win_length),
//   Hamming window, center=False (causal -- no future-sample padding).
//
// n_fft isn't a power of 2, so this uses a direct O(n^2) DFT/IDFT rather than
// an FFT. At n=320 that's ~51K mult-adds per frame -- trivial for a browser,
// even many times faster than the 10ms real-time budget per frame requires.

export const SAMPLE_RATE = 16000;
export const WIN_LENGTH = 320;
export const HOP_LENGTH = 160;
export const N_FFT = 320;
export const N_FREQ_BINS = N_FFT / 2 + 1; // 161

function hammingWindow(n) {
  const w = new Float32Array(n);
  // periodic=True convention (denominator N, not N-1) -- matches
  // torch.hamming_window's default, which is what features.py uses.
  for (let i = 0; i < n; i++) {
    w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / n);
  }
  return w;
}

export const WINDOW = hammingWindow(WIN_LENGTH);

// Real-input one-sided DFT: returns {re, im} each of length N_FREQ_BINS.
function rdft(frame) {
  const re = new Float32Array(N_FREQ_BINS);
  const im = new Float32Array(N_FREQ_BINS);
  for (let k = 0; k < N_FREQ_BINS; k++) {
    let sumRe = 0;
    let sumIm = 0;
    const w = (-2 * Math.PI * k) / N_FFT;
    for (let n = 0; n < N_FFT; n++) {
      const angle = w * n;
      const val = frame[n];
      sumRe += val * Math.cos(angle);
      sumIm += val * Math.sin(angle);
    }
    re[k] = sumRe;
    im[k] = sumIm;
  }
  return { re, im };
}

// Inverse of rdft: one-sided complex spectrum (N_FREQ_BINS) -> N_FFT real samples,
// reconstructed via Hermitian symmetry (matches torch.istft's internal irfft).
function irdft(re, im) {
  const out = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) {
    let sum = re[0]; // DC, k=0 term (im[0] assumed ~0 for a valid one-sided spectrum)
    for (let k = 1; k < N_FREQ_BINS - 1; k++) {
      const angle = (2 * Math.PI * k * n) / N_FFT;
      // each interior bin represents itself + its mirror (Hermitian symmetry),
      // so its real contribution doubles: 2*(re*cos - im*sin)... using the
      // standard real-IDFT-from-one-sided-spectrum formula.
      sum += 2 * (re[k] * Math.cos(angle) - im[k] * Math.sin(angle));
    }
    // Nyquist bin (k = N_FFT/2, N_FFT even): no mirror, counted once
    const nyq = N_FREQ_BINS - 1;
    sum += re[nyq] * Math.cos((2 * Math.PI * nyq * n) / N_FFT);
    out[n] = sum / N_FFT;
  }
  return out;
}

/**
 * Compute causal STFT of a Float32Array waveform.
 * Returns array of frames, each {re: Float32Array(161), im: Float32Array(161)}.
 * Matches torch.stft(..., center=False, onesided=True default).
 */
export function stft(waveform) {
  const numFrames = Math.floor((waveform.length - WIN_LENGTH) / HOP_LENGTH) + 1;
  const frames = [];
  for (let f = 0; f < numFrames; f++) {
    const start = f * HOP_LENGTH;
    const windowed = new Float32Array(N_FFT);
    for (let i = 0; i < WIN_LENGTH; i++) {
      windowed[i] = waveform[start + i] * WINDOW[i];
    }
    frames.push(rdft(windowed));
  }
  return frames;
}

/**
 * Reconstruct waveform from a sequence of {re, im} spectrogram frames via
 * overlap-add synthesis. Matches torch.istft(..., center=False) exactly:
 * each frame's inverse-transform is windowed again before OLA, and the
 * accumulated buffer is divided by the accumulated window^2 (NOLA
 * normalization).
 */
export function istft(frames) {
  const numFrames = frames.length;
  const outLen = (numFrames - 1) * HOP_LENGTH + WIN_LENGTH;
  const out = new Float32Array(outLen);
  const norm = new Float32Array(outLen);

  for (let f = 0; f < numFrames; f++) {
    const timeFrame = irdft(frames[f].re, frames[f].im);
    const start = f * HOP_LENGTH;
    for (let i = 0; i < WIN_LENGTH; i++) {
      out[start + i] += timeFrame[i] * WINDOW[i];
      norm[start + i] += WINDOW[i] * WINDOW[i];
    }
  }

  for (let i = 0; i < outLen; i++) {
    if (norm[i] > 1e-8) out[i] /= norm[i];
  }
  return out;
}

/** log1p(magnitude) feature the model was trained on. */
export function logMagnitude(frame) {
  const out = new Float32Array(N_FREQ_BINS);
  for (let k = 0; k < N_FREQ_BINS; k++) {
    const mag = Math.sqrt(frame.re[k] * frame.re[k] + frame.im[k] * frame.im[k]);
    out[k] = Math.log1p(mag);
  }
  return out;
}

/** Apply a real-valued mask (0-1) to a complex spectrogram frame, keep phase. */
export function applyMask(frame, mask) {
  const re = new Float32Array(N_FREQ_BINS);
  const im = new Float32Array(N_FREQ_BINS);
  for (let k = 0; k < N_FREQ_BINS; k++) {
    re[k] = frame.re[k] * mask[k];
    im[k] = frame.im[k] * mask[k];
  }
  return { re, im };
}

/** Left-pad by HOP_LENGTH zeros before framing -- mirrors pad_for_causal_stft
 * in features.py. NOTE: this is NOT used by the main denoise pipeline (see
 * infer.js) -- src/infer.py's denoise_streaming (the actual production path
 * this app mirrors) does not apply this padding, it's only used internally
 * by features.py's own round-trip self-test. Kept here only for parity /
 * documentation purposes. */
export function padForCausalStft(waveform) {
  const out = new Float32Array(waveform.length + HOP_LENGTH);
  out.set(waveform, HOP_LENGTH);
  return out;
}

export function unpadAfterCausalIstft(waveform) {
  return waveform.slice(HOP_LENGTH);
}

/** Match torch.istft(..., length=N)'s post-hoc pad/trim: reconstruct at
 * natural length, then pad with zeros or truncate to the target length. */
export function fitToLength(waveform, targetLength) {
  if (waveform.length === targetLength) return waveform;
  const out = new Float32Array(targetLength);
  const n = Math.min(waveform.length, targetLength);
  out.set(waveform.subarray(0, n));
  return out;
}
