import { useEffect, useRef, useState } from "react";
import { decodeToMono16k, encodeWav } from "./audio.js";
import { applyStrengthAndReconstruct, computeRawMasks, getSession } from "./infer.js";

const MODEL_PARAMS = "1.44M";
const HOP_BUDGET_MS = 10; // 10ms hop at 16kHz, 160-sample hop

export default function App() {
  const [modelStatus, setModelStatus] = useState("loading"); // loading | ready | error
  const [fileName, setFileName] = useState(null);
  const [stage, setStage] = useState("idle"); // idle | decoding | processing | done | error
  const [progress, setProgress] = useState(0);
  const [strength, setStrength] = useState(1.0);
  const [latency, setLatency] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const noisyWaveformRef = useRef(null);
  const pipelineRef = useRef(null); // { specFrames, rawMasks }
  const [noisyUrl, setNoisyUrl] = useState(null);
  const [cleanUrl, setCleanUrl] = useState(null);

  useEffect(() => {
    getSession()
      .then(() => setModelStatus("ready"))
      .catch((e) => {
        console.error(e);
        setModelStatus("error");
      });
  }, []);

  async function handleFile(file) {
    if (!file) return;
    setFileName(file.name);
    setStage("decoding");
    setErrorMsg(null);
    setCleanUrl(null);
    setProgress(0);

    try {
      const waveform = await decodeToMono16k(file);
      noisyWaveformRef.current = waveform;
      setNoisyUrl(URL.createObjectURL(encodeWav(waveform)));

      setStage("processing");
      const session = await getSession();
      const { specFrames, rawMasks, avgLatencyMs } = await computeRawMasks(waveform, session, setProgress);
      pipelineRef.current = { specFrames, rawMasks };
      setLatency(avgLatencyMs);

      const clean = applyStrengthAndReconstruct(specFrames, rawMasks, waveform.length, strength);
      setCleanUrl(URL.createObjectURL(encodeWav(clean)));
      setStage("done");
    } catch (e) {
      console.error(e);
      setErrorMsg(e.message || String(e));
      setStage("error");
    }
  }

  function handleStrengthChange(newStrength) {
    setStrength(newStrength);
    if (!pipelineRef.current || !noisyWaveformRef.current) return;
    const { specFrames, rawMasks } = pipelineRef.current;
    const clean = applyStrengthAndReconstruct(
      specFrames,
      rawMasks,
      noisyWaveformRef.current.length,
      newStrength
    );
    setCleanUrl(URL.createObjectURL(encodeWav(clean)));
  }

  return (
    <div className="page">
      <header className="hero">
        <div className="hero-label">
          {modelStatus === "loading" && "loading model"}
          {modelStatus === "ready" && "model ready — running entirely in your browser"}
          {modelStatus === "error" && "model failed to load"}
        </div>
        <h1>A hearing aid&apos;s noise suppression, running in your browser tab.</h1>
        <p className="hero-sub">
          A {MODEL_PARAMS}-parameter causal neural network, exported to run via WebAssembly.
          Nothing you upload leaves this page — the model runs entirely client-side.
        </p>
      </header>

      <main className="demo">
        <DropZone onFile={handleFile} disabled={modelStatus !== "ready"} fileName={fileName} stage={stage} />

        {stage === "processing" && (
          <div className="progress-row">
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <span className="mono">{Math.round(progress * 100)}%</span>
          </div>
        )}

        {errorMsg && <p className="error">{errorMsg}</p>}

        {stage === "done" && (
          <>
            <div className="player-row">
              <div className="player">
                <div className="player-label noise">noisy input</div>
                <audio controls src={noisyUrl} />
              </div>
              <div className="player">
                <div className="player-label signal">denoised output</div>
                <audio controls src={cleanUrl} />
              </div>
            </div>

            <div className="strength-block">
              <div className="strength-header">
                <label htmlFor="strength">suppression strength</label>
                <span className="mono">{strength.toFixed(1)}×</span>
              </div>
              <input
                id="strength"
                type="range"
                min="1"
                max="10"
                step="0.5"
                value={strength}
                onChange={(e) => handleStrengthChange(parseFloat(e.target.value))}
              />
              <p className="strength-note">
                The model outputs a continuous suppression mask; this reshapes it (mask
                <sup>strength</sup>) without re-running the network. Higher values remove more
                background noise, at the cost of some speech detail — there is a real
                trade-off here, not a free lunch. Try both ends.
              </p>
            </div>

            {latency !== null && (
              <div className="stats-row">
                <Stat label="avg. frame latency" value={`${latency.toFixed(3)} ms`} />
                <Stat label="real-time budget" value={`${HOP_BUDGET_MS.toFixed(1)} ms`} />
                <Stat
                  label="headroom"
                  value={`${(HOP_BUDGET_MS / Math.max(latency, 0.001)).toFixed(0)}×`}
                />
              </div>
            )}
          </>
        )}
      </main>

      <footer className="footnote">
        <p>
          Causal spectral-masking model — every layer sees only past audio, no lookahead, so
          this is the same computation path a streaming embedded device would run. Trained on
          VoiceBank-DEMAND. Inference here uses ONNX Runtime Web (WASM), the same exported
          graph benchmarked for on-device latency.
        </p>
      </footer>
    </div>
  );
}

function DropZone({ onFile, disabled, fileName, stage }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      className={`dropzone ${dragOver ? "drag-over" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (!disabled) onFile(e.dataTransfer.files[0]);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        hidden
        onChange={(e) => onFile(e.target.files[0])}
      />
      {fileName ? (
        <>
          <div className="dropzone-file">{fileName}</div>
          <div className="dropzone-hint">
            {stage === "decoding" && "decoding audio…"}
            {stage === "processing" && "processing…"}
            {(stage === "done" || stage === "error") && "click or drop to try another file"}
          </div>
        </>
      ) : (
        <>
          <div className="dropzone-file">
            {disabled ? "loading model…" : "drop a noisy speech recording here"}
          </div>
          <div className="dropzone-hint">or click to browse — wav, mp3, m4a</div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="stat-value mono">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
