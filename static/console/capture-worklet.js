class ConsoleCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.chunkSize = options.processorOptions?.chunkSize || 16000;
    this.buffer = new Float32Array(0);
    this.active = false;
    this.port.onmessage = ({ data }) => {
      if (data?.command === "start") {
        this.active = true;
        this.buffer = new Float32Array(0);
      } else if (data?.command === "stop") {
        this.active = false;
        if (this.buffer.length) {
          const chunk = this.buffer.slice(0);
          this.port.postMessage(
            { type: "chunk", audio: chunk, final: true },
            [chunk.buffer],
          );
        }
        this.buffer = new Float32Array(0);
        this.port.postMessage({ type: "stopped" });
      }
    };
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!this.active || !input?.length) return true;
    const combined = new Float32Array(this.buffer.length + input.length);
    combined.set(this.buffer);
    combined.set(input, this.buffer.length);
    this.buffer = combined;
    while (this.buffer.length >= this.chunkSize) {
      const chunk = this.buffer.slice(0, this.chunkSize);
      this.buffer = this.buffer.slice(this.chunkSize);
      this.port.postMessage({ type: "chunk", audio: chunk }, [chunk.buffer]);
    }
    return true;
  }
}

registerProcessor("console-capture", ConsoleCaptureProcessor);
