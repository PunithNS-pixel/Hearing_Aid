# Speech Denoiser — Web Demo

Runs the trained causal speech-denoising model entirely client-side in the
browser via ONNX Runtime Web (WASM) — no backend, nothing uploaded to a
server, deployable as a static site.

## 1. Add your trained model

This repo does not include a trained model (only architecture code was
shared). Export one from your training project:

```bash
# from the hearing-aid-denoiser project
python -m src.export_onnx --checkpoint checkpoints/best.pt \
    --onnx_path model.onnx --suppression_strength 1.0
```

Use `suppression_strength 1.0` (neutral) for the export — the web app has
a live slider that reshapes the mask in the browser (`mask ** strength`),
so you don't need to bake a fixed aggressiveness into the exported graph.
This also means visitors can explore the exact suppression/quality
trade-off documented in the training project's README, live.

Copy the exported file in:
```bash
cp model.onnx public/model.onnx
```

**Verify the file is self-contained** (a single file, no `model.onnx.data`
sitting next to it) before deploying — `src/export_onnx.py` was fixed to
always produce one file, but if you're using an older export, re-run it.

## 2. Run locally

```bash
npm install
npm run dev
```

Open the printed localhost URL, drop in a noisy wav/mp3, and it should
process in-browser.

## 3. Deploy to Vercel

This is a static Vite build — no server code, no special Vercel
configuration needed.

**Option A — Vercel CLI (fastest):**
```bash
npm install -g vercel
vercel deploy --prod
```
Follow the prompts (link/create a project). Vercel auto-detects Vite and
runs `npm run build`, serving `dist/`.

**Option B — GitHub + Vercel dashboard:**
1. Push this repo to GitHub.
2. On vercel.com, "Add New Project" → import the repo.
3. Framework preset: Vite (auto-detected). No env vars needed.
4. Deploy.

## Known constraints / honest caveats

- **Model file size:** the ONNX model is ~5-6MB, and the WASM runtime
  itself is ~14MB (down from ~28MB after switching to the
  `onnxruntime-web/wasm` CPU-only subpath — see `src/infer.js`). Total
  first-load is meaningfully sized for a web page; fine for a portfolio
  demo, worth knowing if you care about first-load speed. Both are
  browser-cached after first visit.
- **Numerical precision:** the JS STFT/ISTFT uses a direct O(n²) DFT
  (n_fft=320 isn't a power of 2, so no simple radix-2 FFT applies) rather
  than PyTorch's FFT implementation. Verified against the Python reference
  to ~40dB reconstruction SNR — inaudible in practice, but not
  bit-identical, and worth knowing if you're citing exact figures.
- **Browser audio decoding:** relies on the Web Audio API
  (`decodeAudioData`) for file decoding and resampling to 16kHz mono,
  which is broadly supported but not something this project's automated
  tests can verify without a real browser (all pipeline testing here was
  done in Node with synthetic/reference audio, not a live browser). Test
  with a real upload before considering this fully verified end-to-end.
- **Mobile Safari** has historically had WASM threading quirks; if you
  hit issues there specifically, that's the first place to look.

## Architecture

```
noisy audio file
  → decode + resample to 16kHz mono (Web Audio API)
  → STFT (src/dsp.js, matches src/features.py exactly)
  → per-frame causal model inference (ONNX Runtime Web, src/infer.js)
  → mask ** suppression_strength (live, no re-inference needed)
  → apply mask + iSTFT reconstruction
  → encode as WAV for playback (src/audio.js)
```

Every one of these steps was numerically verified against the trained
Python pipeline before this was wired into the UI — see the development
history for the specific bugs found and fixed (window periodicity
mismatch, incorrect causal padding, external ONNX data file).
