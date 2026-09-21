# Causal Speech Denoiser for Hearing Aids

A real-time causal speech-denoising system designed for hearing-aid-style audio processing.

The project uses a lightweight causal neural network to suppress background noise while preserving speech, and exports the trained model to **ONNX** for fully client-side inference in the browser using **ONNX Runtime Web**.

> **No backend. No audio uploads. No server-side inference.**
>
> The uploaded audio remains inside the user's browser.

---

## Demo



https://github.com/user-attachments/assets/492f6c4f-ce19-43ed-893a-04e36da4e1ea



### Live Demo

**[Try the Web Demo](https://hearing-aid-denoiser.vercel.app)**

Upload a noisy `.wav`, `.mp3`, or `.m4a` file and process it directly in your browser.


**Project Architecture**

<img width="6183" height="5175" alt="diagram-2" src="https://github.com/user-attachments/assets/763f32b7-64bc-4c2d-91e3-3194d862ce90" />


The browser performs:

```text
Audio File
    ↓
Decode + Resample
    ↓
16 kHz Mono Audio
    ↓
STFT
    ↓
Causal Neural Network
    ↓
Spectral Mask
    ↓
Mask Strength Adjustment
    ↓
iSTFT
    ↓
WAV Reconstruction
    ↓
Denoised Audio
