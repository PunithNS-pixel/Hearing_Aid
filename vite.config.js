import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    target: "es2020",
  },
  // onnxruntime-web loads its .wasm files from this path at runtime;
  // vite-plugin copies public/ as-is, and ORT's own npm postinstall
  // step (see package.json) copies the wasm binaries into public/ort/.
});
