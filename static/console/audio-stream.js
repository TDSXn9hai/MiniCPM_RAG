const INPUT_RATE = 16000;
const OUTPUT_RATE = 24000;
const PLAY_LEAD_SECONDS = 0.1;

export function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const ratio = fromRate / toRate;
  const length = Math.max(1, Math.round(samples.length / ratio));
  const output = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const sourceIndex = index * ratio;
    const lower = Math.floor(sourceIndex);
    const fraction = sourceIndex - lower;
    const next = Math.min(lower + 1, samples.length - 1);
    output[index] = samples[lower] * (1 - fraction)
      + samples[next] * fraction;
  }
  return output;
}

export function float32ToBase64(samples) {
  const bytes = new Uint8Array(
    samples.buffer,
    samples.byteOffset,
    samples.byteLength,
  );
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    binary += String.fromCharCode.apply(
      null,
      bytes.subarray(offset, offset + 8192),
    );
  }
  return btoa(binary);
}

export function base64ToFloat32(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Float32Array(bytes.buffer);
}

export class StreamingAudioIO {
  constructor({ onChunk, onLevel, onPlayback }) {
    this.onChunk = onChunk;
    this.onLevel = onLevel;
    this.onPlayback = onPlayback;
    this.inputContext = null;
    this.outputContext = null;
    this.micStream = null;
    this.source = null;
    this.worklet = null;
    this.silentGain = null;
    this.analyser = null;
    this.animation = null;
    this.active = false;
    this.pendingStop = null;
    this.generation = 0;
    this.nextPlayTime = 0;
    this.answerPcm = [];
    this.sources = new Set();
  }

  async prepare() {
    if (this.inputContext) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("当前浏览器不支持麦克风采集，请使用 localhost 或 HTTPS");
    }
    const generation = ++this.generation;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
      video: false,
    });
    if (generation !== this.generation) {
      for (const track of stream.getTracks()) track.stop();
      throw new DOMException("capture aborted", "AbortError");
    }
    this.micStream = stream;
    this.inputContext = new AudioContext({ sampleRate: INPUT_RATE });
    if (this.inputContext.state === "suspended") {
      await this.inputContext.resume();
    }
    await this.inputContext.audioWorklet.addModule(
      "/static/console/capture-worklet.js",
    );
    if (generation !== this.generation) {
      for (const track of this.micStream?.getTracks() || []) track.stop();
      await this.inputContext.close();
      this.inputContext = null;
      this.micStream = null;
      throw new DOMException("capture aborted", "AbortError");
    }
    this.source = this.inputContext.createMediaStreamSource(this.micStream);
    this.worklet = new AudioWorkletNode(
      this.inputContext,
      "console-capture",
      {
        processorOptions: {
          chunkSize: Math.round(this.inputContext.sampleRate),
        },
      },
    );
    this.silentGain = this.inputContext.createGain();
    this.silentGain.gain.value = 0;
    this.analyser = this.inputContext.createAnalyser();
    this.analyser.fftSize = 512;
    this.source.connect(this.worklet);
    this.source.connect(this.analyser);
    this.worklet.connect(this.silentGain);
    this.silentGain.connect(this.inputContext.destination);
    this.worklet.port.onmessage = ({ data }) => {
      if (data?.type === "chunk") {
        const samples = resample(
          data.audio,
          this.inputContext.sampleRate,
          INPUT_RATE,
        );
        const finalSamples = data.final
          ? samples.slice(0, INPUT_RATE)
          : samples.slice(0, INPUT_RATE);
        this.onChunk(
          finalSamples,
          Boolean(data.final),
          float32ToBase64(finalSamples),
        );
      } else if (data?.type === "stopped" && this.pendingStop) {
        const resolve = this.pendingStop;
        this.pendingStop = null;
        resolve();
      }
    };
  }

  startCapture() {
    if (!this.worklet || !this.inputContext) {
      throw new Error("麦克风尚未准备完成");
    }
    if (this.active) return;
    this.worklet.port.postMessage({ command: "start" });
    this.active = true;
    this._drawLevel();
  }

  async start() {
    await this.prepare();
    this.startCapture();
  }

  async stop() {
    if (!this.active || !this.worklet) {
      await this.abort();
      return;
    }
    this.active = false;
    const stopped = new Promise((resolve) => {
      this.pendingStop = resolve;
    });
    this.worklet.port.postMessage({ command: "stop" });
    await stopped;
    await this._teardownInput();
  }

  async abort() {
    this.generation += 1;
    this.active = false;
    if (this.pendingStop) {
      this.pendingStop();
      this.pendingStop = null;
    }
    await this._teardownInput();
  }

  async _teardownInput() {
    cancelAnimationFrame(this.animation);
    this.animation = null;
    for (const track of this.micStream?.getTracks() || []) track.stop();
    this.micStream = null;
    try { this.worklet?.disconnect(); } catch (_) {}
    try { this.source?.disconnect(); } catch (_) {}
    try { this.silentGain?.disconnect(); } catch (_) {}
    try { this.analyser?.disconnect(); } catch (_) {}
    this.worklet = null;
    this.source = null;
    this.silentGain = null;
    this.analyser = null;
    const context = this.inputContext;
    this.inputContext = null;
    await context?.close();
  }

  _drawLevel() {
    if (!this.active || !this.analyser) return;
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let energy = 0;
    for (const value of data) energy += value * value;
    this.onLevel(Math.min(1, Math.sqrt(energy / data.length) * 8));
    this.animation = requestAnimationFrame(() => this._drawLevel());
  }

  resetAnswer() {
    this.stopPlayback();
    this.answerPcm = [];
    this.nextPlayTime = 0;
  }

  playBase64(value) {
    const samples = base64ToFloat32(value);
    this.answerPcm.push(samples.slice(0));
    this._schedule(samples);
  }

  replay() {
    this.stopPlayback();
    this.nextPlayTime = 0;
    for (const samples of this.answerPcm) this._schedule(samples);
  }

  _ensureOutputContext() {
    if (!this.outputContext) {
      this.outputContext = new AudioContext({ sampleRate: OUTPUT_RATE });
    }
  }

  _schedule(samples) {
    if (!samples.length) return;
    this._ensureOutputContext();
    const buffer = this.outputContext.createBuffer(1, samples.length, OUTPUT_RATE);
    buffer.copyToChannel(samples, 0);
    const source = this.outputContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.outputContext.destination);
    const now = this.outputContext.currentTime;
    if (this.nextPlayTime < now + 0.02) {
      this.nextPlayTime = now + PLAY_LEAD_SECONDS;
    }
    source.start(this.nextPlayTime);
    this.nextPlayTime += buffer.duration;
    this.sources.add(source);
    this.onPlayback?.(true);
    source.onended = () => {
      this.sources.delete(source);
      if (!this.sources.size) this.onPlayback?.(false);
    };
  }

  stopPlayback() {
    for (const source of this.sources) {
      try { source.stop(); } catch (_) {}
    }
    this.sources.clear();
    this.onPlayback?.(false);
  }

  async close() {
    await this.abort();
    await this.outputContext?.close();
    this.outputContext = null;
  }
}
