import { StreamingAudioIO } from "./audio-stream.js";

const $ = (selector) => document.querySelector(selector);
const micButton = $("#micButton");
const resetButton = $("#resetButton");
const connectionState = $("#connectionState");
const transcriptText = $("#transcriptText");
const partialTag = $("#partialTag");
const answerText = $("#answerText");
const evidenceList = $("#evidenceList");
const eventLog = $("#eventLog");
const toast = $("#toast");
const replayButton = $("#replayButton");
const audioState = $("#audioState");
const miniWave = $(".mini-wave");

let socket = null;
let reconnectTimer = null;
let ready = false;
let busy = false;
let recording = false;
let holding = false;
let preparingCapture = false;
let serverTurnStarted = false;
let sequence = 0;
let currentTurn = null;
let recordStarted = 0;
let recordTimer = null;
let stageTimes = {};
let firstPartialAt = null;
let asrFinalAt = null;
let firstTextAt = null;
let firstAudioAt = null;
let finalRetrievalAt = null;
let totalAt = null;
let hasAudio = false;
let lastRetrievalHits = [];
let lastInjectionWait = null;
const history = [];

const audio = new StreamingAudioIO({
  onChunk: (samples, final, encoded) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({
      type: "input.audio.append",
      sequence: sequence++,
      final,
      audio: encoded,
    }));
  },
  onLevel: (level) => {
    $("#levelBar").style.width = `${Math.round(level * 100)}%`;
  },
  onPlayback: (playing) => {
    miniWave.classList.toggle("playing", playing);
    audioState.textContent = playing ? "正在播放" : (hasAudio ? "播放完成" : "等待回答");
  },
});

function formatMs(value) {
  if (!Number.isFinite(value)) return "—";
  if (value < 1000) return `${Math.round(value)}ms`;
  return `${(value / 1000).toFixed(2)}s`;
}

function formatSeconds(value) {
  return `${value.toFixed(1)}s`;
}

function retrievalOptions() {
  const topK = Math.max(1, Math.min(10, Number($("#topKInput").value) || 3));
  const minScore = Math.max(0, Math.min(1, Number($("#minScoreInput").value) || 0));
  return {
    title_only: $("#retrievalMode").value === "title",
    top_k: topK,
    min_score: minScore,
    max_evidence_chars: 900,
    min_shared_bigrams: 2,
  };
}

function setConnection(label, mode) {
  connectionState.textContent = label;
  connectionState.className = `status-pill ${mode}`;
}

function showToast(message) {
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.hidden = true; }, 4200);
}

function setStage(name, state, label) {
  const node = document.querySelector(`.stage[data-stage="${name}"]`);
  if (!node) return;
  node.classList.remove("active", "done", "error");
  if (state) node.classList.add(state);
  if (label !== undefined) node.querySelector("em").textContent = label;
}

function resetStages() {
  for (const name of ["mic", "asr", "retrieval", "injection", "model"]) {
    setStage(name, "", "等待");
  }
}

function resetTurn() {
  currentTurn = null;
  sequence = 0;
  stageTimes = {};
  firstPartialAt = null;
  asrFinalAt = null;
  firstTextAt = null;
  firstAudioAt = null;
  finalRetrievalAt = null;
  totalAt = null;
  hasAudio = false;
  lastRetrievalHits = [];
  lastInjectionWait = null;
  transcriptText.textContent = "按住麦克风按钮开始提问";
  partialTag.hidden = true;
  answerText.textContent = "";
  evidenceList.innerHTML = `
    <div class="empty-state">
      <strong>等待问题</strong>
      <p>命中文档、分数和实际注入的证据会显示在这里。</p>
    </div>`;
  eventLog.innerHTML = '<div class="empty-state compact">事件将在这里按到达顺序追加。</div>';
  $("#retrievalState").textContent = "尚未检索";
  $("#turnId").textContent = "尚未开始";
  $("#latencySummary").textContent = "展开查看本轮时间线";
  for (const id of ["metricPartial", "metricAsr", "metricRetrieval", "metricFirstText", "metricFirstAudio", "metricTotal"]) {
    $(`#${id}`).textContent = "—";
  }
  document.querySelectorAll(".trace-row b").forEach((bar) => {
    bar.style.width = "0";
    bar.style.marginLeft = "0";
  });
  document.querySelectorAll(".trace-row em").forEach((node) => { node.textContent = "—"; });
  audio.resetAnswer();
  replayButton.disabled = true;
  resetStages();
}

function addHistory(event) {
  if (event.type === "answer.done") {
    history.unshift({
      query: transcriptText.textContent || "未识别",
      asr: asrFinalAt,
      retrieval: finalRetrievalAt,
      injection: lastInjectionWait,
      total: event.t_ms,
      hits: lastRetrievalHits.length,
    });
    history.splice(10);
    renderHistory();
  }
}

function renderHistory() {
  if (!history.length) {
    $("#historyList").innerHTML = '<div class="empty-state compact">完成一轮后显示 ASR、检索、注入和总耗时。</div>';
    return;
  }
  $("#historyList").innerHTML = "";
  history.forEach((item) => {
    const row = document.createElement("div");
    row.className = "history-row";
    const query = document.createElement("strong");
    query.textContent = `${item.query} · ${item.hits} hits`;
    const values = [
      formatMs(item.asr),
      formatMs(item.retrieval),
      formatMs(item.injection),
      formatMs(item.total),
    ];
    row.append(query, ...values.map((value) => {
      const node = document.createElement("span");
      node.textContent = value;
      return node;
    }));
    $("#historyList").append(row);
  });
}

function evidenceSections(text) {
  const result = new Map();
  for (const part of String(text || "").split(/\n{2,}(?=\[K\d+\])/)) {
    const match = part.match(/^\[(K\d+)\]/);
    if (match) result.set(match[1], part.replace(/^\[K\d+\]\s*/, ""));
  }
  return result;
}

function logEvent(event, label, hot = false) {
  if (eventLog.querySelector(".empty-state")) eventLog.innerHTML = "";
  const row = document.createElement("div");
  const time = document.createElement("span");
  time.className = hot ? "hot" : "signal";
  time.textContent = `+ ${formatMs(event.t_ms)}`;
  row.append(time, document.createTextNode(`  ${label}`));
  eventLog.append(row);
  eventLog.scrollTop = eventLog.scrollHeight;
}

function updateTrace(name, start, end) {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return;
  stageTimes[name] = { start, end };
  const maxEnd = Math.max(
    totalAt || 0,
    ...Object.values(stageTimes).map((item) => item.end),
    1,
  );
  const row = document.querySelector(`.trace-row[data-trace="${name}"]`);
  const bar = row.querySelector("b");
  bar.style.marginLeft = `${Math.max(0, (start / maxEnd) * 100)}%`;
  bar.style.width = `${Math.max(0.8, ((end - start) / maxEnd) * 100)}%`;
  row.querySelector("em").textContent = formatMs(end - start);
  $("#latencySummary").textContent = `${formatMs(maxEnd)} 已记录 ${Object.keys(stageTimes).length} 个阶段`;
}

function renderEvidence(event) {
  const exists = Array.isArray(event.hits) && event.hits.length;
  if (!exists) {
    evidenceList.innerHTML = `
      <div class="empty-state">
        <strong>无可用证据</strong>
        <p>检索未达到分数或词组门槛，本轮会明确标记为无证据。</p>
      </div>`;
    $("#retrievalState").textContent = "无证据";
    return;
  }
  evidenceList.innerHTML = "";
  const citations = new Map((event.citations || []).map((item) => [item.document_id, item]));
  const sections = evidenceSections(event.evidence);
  event.hits.forEach((hit, index) => {
    const citation = citations.get(hit.document_id) || {};
    const card = document.createElement("article");
    card.className = `evidence-card ${event.type === "retrieval.final" ? "final" : ""}`;
    const head = document.createElement("div");
    head.className = "evidence-head";
    const title = document.createElement("strong");
    title.textContent = hit.title || hit.document_id || `证据 ${index + 1}`;
    const score = document.createElement("span");
    score.className = "evidence-score";
    score.textContent = Number(hit.score).toFixed(3);
    head.append(title, score);
    const content = document.createElement("p");
    content.className = "evidence-content";
    content.textContent = sections.get(citation.id)
      || (event.type === "retrieval.final" ? "该命中未进入最终证据卡" : "候选命中，等待最终证据卡");
    const meta = document.createElement("div");
    meta.className = "evidence-meta";
    const source = document.createElement("span");
    source.textContent = hit.source || citation.source || hit.document_id || "";
    const kind = document.createElement("span");
    kind.textContent = event.type === "retrieval.final" ? "权威证据" : "partial 预检索";
    meta.append(source, kind);
    card.append(head, content, meta);
    evidenceList.append(card);
  });
  if (event.type === "retrieval.final" && event.evidence) {
    const details = document.createElement("details");
    details.className = "evidence-full";
    const summary = document.createElement("summary");
    summary.textContent = "完整注入证据";
    const pre = document.createElement("pre");
    pre.textContent = event.evidence;
    details.append(summary, pre);
    evidenceList.append(details);
  }
  $("#retrievalState").textContent = event.type === "retrieval.final" ? "最终命中" : "预检索";
}

function handleEvent(event) {
  if (event.turn_id && currentTurn && event.turn_id !== currentTurn) return;
  switch (event.type) {
    case "session.ready":
      ready = true;
      busy = false;
      micButton.disabled = false;
      resetButton.disabled = false;
      setConnection("链路在线", "ready");
      $("#knowledgeName").textContent = event.knowledge.split("/").pop();
      break;
    case "session.error":
      ready = false;
      micButton.disabled = true;
      setConnection("服务错误", "error");
      showToast(event.message);
      break;
    case "session.reset":
      resetTurn();
      setConnection("链路在线", "ready");
      break;
    case "turn.started":
      currentTurn = event.turn_id;
      busy = true;
      $("#turnId").textContent = event.turn_id;
      setConnection("处理中", "busy");
      resetButton.disabled = true;
      setStage("asr", "active", "开始");
      logEvent(event, "turn.started");
      break;
    case "asr.partial":
      transcriptText.textContent = event.text;
      partialTag.hidden = false;
      if (firstPartialAt === null) {
        firstPartialAt = event.t_ms;
        $("#metricPartial").textContent = formatMs(event.t_ms);
      }
      setStage("asr", "active", "partial");
      logEvent(event, `asr.partial ${event.text}`);
      break;
    case "asr.final":
      asrFinalAt = event.t_ms;
      transcriptText.textContent = event.text || "未检测到语音";
      partialTag.hidden = true;
      busy = event.text ? busy : false;
      $("#metricAsr").textContent = formatMs(event.t_ms);
      setStage("asr", "done", "final");
      updateTrace("asr", firstPartialAt || event.t_ms - (event.finished_s - event.started_s) * 1000, event.t_ms);
      logEvent(event, `asr.final ${event.text || "(empty)"}`);
      break;
    case "retrieval.partial":
      setStage("retrieval", "active", "预检索");
      renderEvidence(event);
      logEvent(event, `retrieval.partial ${event.hits?.length || 0} hits`);
      break;
    case "retrieval.final":
      finalRetrievalAt = event.t_ms;
      lastRetrievalHits = event.hits || [];
      $("#metricRetrieval").textContent = formatMs(event.t_ms);
      setStage("retrieval", "done", "final");
      renderEvidence(event);
      updateTrace("retrieval", event.t_ms - (event.finished_s - event.started_s) * 1000, event.t_ms);
      logEvent(event, `retrieval.final ${event.hits?.length || 0} hits`, true);
      break;
    case "audio.listen":
      if (event.phase === "waiting") {
        setStage("retrieval", "active", "等待 final");
      }
      break;
    case "rag.injected":
      lastInjectionWait = event.wait_to_inject_ms;
      setStage("injection", "done", formatMs(event.t_ms));
      setStage("model", "active", "生成中");
      updateTrace("injection", event.t_ms - (event.finished_s - event.started_s) * 1000, event.t_ms);
      logEvent(event, "rag.injected AUDIO / force_listen", true);
      break;
    case "answer.text.delta":
      if (!answerText.textContent) answerText.textContent = "";
      answerText.textContent += event.text;
      if (firstTextAt === null) {
        firstTextAt = event.t_ms;
        $("#metricFirstText").textContent = formatMs(event.t_ms);
      }
      setStage("model", "active", "文本");
      logEvent(event, `text.delta ${event.text}`);
      break;
    case "answer.audio.delta":
      hasAudio = true;
      replayButton.disabled = false;
      audio.playBase64(event.audio);
      if (firstAudioAt === null) {
        firstAudioAt = event.t_ms;
        $("#metricFirstAudio").textContent = formatMs(event.t_ms);
        audioState.textContent = "开始播放";
      }
      setStage("model", "active", "音频");
      logEvent(event, `audio.delta unit=${event.unit}`);
      break;
    case "answer.done":
      totalAt = event.t_ms;
      busy = false;
      micButton.disabled = false;
      resetButton.disabled = false;
      $("#metricTotal").textContent = formatMs(event.t_ms);
      setStage("model", "done", "完成");
      setConnection("链路在线", "ready");
      updateTrace("model", firstTextAt || event.t_ms, event.t_ms);
      if (firstAudioAt !== null) updateTrace("audio", firstAudioAt, event.t_ms);
      logEvent(event, "answer.done", true);
      addHistory(event);
      break;
    case "no_speech":
      busy = false;
      micButton.disabled = false;
      resetButton.disabled = false;
      transcriptText.textContent = "未检测到语音，请重试";
      setConnection("链路在线", "ready");
      showToast("未检测到语音，本轮未触发生成");
      logEvent(event, "no_speech", true);
      break;
    case "turn.error":
      busy = false;
      micButton.disabled = false;
      resetButton.disabled = false;
      setConnection("本轮错误", "error");
      setStage("model", "error", "错误");
      showToast(event.message || event.code || "处理失败");
      logEvent(event, `turn.error ${event.code || ""}`, true);
      break;
    case "turn.cancelled":
      busy = false;
      micButton.disabled = false;
      setConnection("已取消", "waiting");
      logEvent(event, "turn.cancelled", true);
      break;
    default:
      break;
  }
}

function connect() {
  clearTimeout(reconnectTimer);
  setConnection("正在连接", "waiting");
  micButton.disabled = true;
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/ws/console`);
  socket.onopen = () => {
    setConnection("握手中", "waiting");
  };
  socket.onmessage = ({ data }) => {
    try {
      handleEvent(JSON.parse(data));
    } catch (error) {
      showToast(`无法解析服务端事件: ${error.message}`);
    }
  };
  socket.onerror = () => {
    setConnection("连接错误", "error");
  };
  socket.onclose = () => {
    ready = false;
    busy = false;
    abortCapture();
    micButton.disabled = true;
    resetButton.disabled = true;
    setConnection("连接断开", "error");
    reconnectTimer = setTimeout(connect, 1800);
  };
}

async function startQuestion() {
  if (!ready || busy || recording || preparingCapture) return;
  holding = true;
  busy = true;
  preparingCapture = true;
  serverTurnStarted = false;
  sequence = 0;
  resetTurn();
  micButton.disabled = true;
  setStage("mic", "active", "准备麦克风");
  try {
    await audio.prepare();
    preparingCapture = false;
    if (!holding) {
      await audio.abort();
      busy = false;
      micButton.disabled = false;
      setStage("mic", "", "待机");
      return;
    }
    socket.send(JSON.stringify({
      type: "input.start",
      sample_rate: 16000,
      retrieval: retrievalOptions(),
    }));
    serverTurnStarted = true;
    audio.startCapture();
    recording = true;
    setStage("mic", "active", "采集中");
    micButton.classList.add("recording");
    recordStarted = performance.now();
    recordTimer = setInterval(() => {
      $("#recordTime").textContent = formatSeconds((performance.now() - recordStarted) / 1000);
      $("#turnClock").textContent = formatSeconds((performance.now() - recordStarted) / 1000);
      if (performance.now() - recordStarted >= 30000) {
        holding = false;
        stopQuestion();
      }
    }, 80);
  } catch (error) {
    preparingCapture = false;
    if (error?.name === "AbortError") {
      busy = false;
      micButton.disabled = false;
      setStage("mic", "", "待机");
      return;
    }
    if (serverTurnStarted && socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "input.cancel", reason: "capture_failed" }));
    }
    await audio.abort();
    recording = false;
    busy = false;
    micButton.disabled = false;
    micButton.classList.remove("recording");
    clearInterval(recordTimer);
    showToast(error.message);
    setStage("mic", "error", "失败");
    setConnection("麦克风错误", "error");
  }
}

async function stopQuestion() {
  holding = false;
  if (!recording) return;
  recording = false;
  micButton.classList.remove("recording");
  clearInterval(recordTimer);
  setStage("mic", "done", formatSeconds((performance.now() - recordStarted) / 1000));
  try {
    await audio.stop();
    if (socket?.readyState === WebSocket.OPEN && serverTurnStarted) {
      socket.send(JSON.stringify({ type: "input.end", sequence }));
    }
  } catch (error) {
    await audio.abort();
    if (socket?.readyState === WebSocket.OPEN && serverTurnStarted) {
      socket.send(JSON.stringify({ type: "input.cancel", reason: "capture_stop_failed" }));
    }
    showToast(error.message);
  }
}

async function abortCapture() {
  holding = false;
  recording = false;
  preparingCapture = false;
  clearInterval(recordTimer);
  recordTimer = null;
  micButton.classList.remove("recording");
  await audio.abort();
}

micButton.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  micButton.setPointerCapture?.(event.pointerId);
  startQuestion();
});
micButton.addEventListener("pointerup", (event) => {
  event.preventDefault();
  stopQuestion();
});
micButton.addEventListener("pointercancel", stopQuestion);
micButton.addEventListener("contextmenu", (event) => event.preventDefault());

resetButton.addEventListener("click", () => {
  if (busy || !socket || socket.readyState !== WebSocket.OPEN) return;
  audio.resetAnswer();
  socket.send(JSON.stringify({ type: "session.reset" }));
});

replayButton.addEventListener("click", () => {
  if (hasAudio) audio.replay();
});

window.addEventListener("beforeunload", () => {
  abortCapture();
  socket?.close();
});

resetTurn();
connect();
