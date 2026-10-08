const EXAMPLES = [
  "What are the main differences between RAG and fine-tuning, and when should each be used?",
  "How does TCP differ from UDP, and when would you choose each?",
  "What are the tradeoffs between SQL and NoSQL databases for a new app?",
];

const els = {
  question: document.getElementById("questionInput"),
  examples: document.getElementById("examples"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  btnLabel: document.querySelector(".btn-label"),
  btnSpinner: document.querySelector(".btn-spinner"),
  errorBanner: document.getElementById("errorBanner"),
  healthDot: document.getElementById("healthDot"),
  healthText: document.getElementById("healthText"),
  subQuestionsOut: document.getElementById("subQuestionsOut"),
  qaOut: document.getElementById("qaOut"),
  finalOut: document.getElementById("finalOut"),
  stages: {
    decompose: document.getElementById("stage-decompose"),
    answer: document.getElementById("stage-answer"),
    synthesize: document.getElementById("stage-final"),
  },
};

function renderExamples() {
  EXAMPLES.forEach((ex) => {
    const chip = document.createElement("button");
    chip.className = "example-chip";
    chip.type = "button";
    chip.textContent = ex.length > 60 ? ex.slice(0, 57) + "..." : ex;
    chip.title = ex;
    chip.addEventListener("click", () => {
      els.question.value = ex;
      els.question.focus();
    });
    els.examples.appendChild(chip);
  });
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.fire_key_configured) {
      els.healthDot.classList.add("ok");
      els.healthText.textContent = "Fire key configured";
    } else {
      els.healthDot.classList.add("bad");
      els.healthText.textContent = "Fire key missing on server";
    }
  } catch {
    els.healthDot.classList.add("bad");
    els.healthText.textContent = "Server unreachable";
  }
}

function setStageState(stage, state) {
  const el = els.stages[stage];
  el.classList.remove("active", "done", "error");
  if (state) el.classList.add(state);
  const statusEl = el.querySelector('[data-role="status"]');
  const labels = { active: "working...", done: "complete", error: "error", idle: "idle" };
  statusEl.textContent = labels[state] || "idle";
}

function resetUI() {
  els.errorBanner.hidden = true;
  els.subQuestionsOut.innerHTML = "";
  els.qaOut.innerHTML = "";
  els.finalOut.innerHTML = "";
  setStageState("decompose", "idle");
  setStageState("answer", "idle");
  setStageState("synthesize", "idle");
}

function showError(message) {
  els.errorBanner.hidden = false;
  els.errorBanner.textContent = message;
}

function renderSubQuestions(subQuestions) {
  els.subQuestionsOut.innerHTML = "";
  subQuestions.forEach((q, i) => {
    const div = document.createElement("div");
    div.className = "sub-q-item";
    div.style.animationDelay = `${i * 80}ms`;
    div.textContent = `${i + 1}. ${q}`;
    els.subQuestionsOut.appendChild(div);
  });
}

function renderQaPairs(qaPairs) {
  els.qaOut.innerHTML = "";
  qaPairs.forEach((pair, i) => {
    const div = document.createElement("div");
    div.className = "qa-item";
    div.style.animationDelay = `${i * 80}ms`;
    div.innerHTML = `<div class="q">${escapeHtml(pair.question)}</div><div class="a">${escapeHtml(pair.answer)}</div>`;
    els.qaOut.appendChild(div);
  });
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

function setLoading(isLoading) {
  els.analyzeBtn.disabled = isLoading;
  els.btnSpinner.hidden = !isLoading;
  els.btnLabel.textContent = isLoading ? "Running workflow..." : "Analyze Question";
}

async function runAnalysis() {
  const question = els.question.value.trim();
  if (!question) {
    showError("Please enter a question first.");
    return;
  }

  resetUI();
  setLoading(true);
  setStageState("decompose", "active");

  try {
    const res = await fetch("/api/analyze/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok || !res.body) {
      throw new Error(`Request failed (HTTP ${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sepIndex;
      while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);
        const line = rawEvent.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const payload = JSON.parse(line.slice(6));
        handleEvent(payload);
      }
    }
  } catch (err) {
    showError(err.message || "Something went wrong talking to the server.");
    setStageState("decompose", "error");
  } finally {
    setLoading(false);
  }
}

function handleEvent(payload) {
  if (payload.node === "error") {
    showError(payload.message || "The workflow hit an error.");
    ["decompose", "answer", "synthesize"].forEach((s) => {
      const el = els.stages[s];
      if (!el.classList.contains("done")) setStageState(s, "error");
    });
    return;
  }

  if (payload.node === "decompose") {
    setStageState("decompose", "done");
    setStageState("answer", "active");
    renderSubQuestions(payload.data.sub_questions || []);
  } else if (payload.node === "answer") {
    setStageState("answer", "done");
    setStageState("synthesize", "active");
    renderQaPairs(payload.data.qa_pairs || []);
  } else if (payload.node === "synthesize") {
    setStageState("synthesize", "done");
    els.finalOut.textContent = payload.data.final_answer || "";
  } else if (payload.node === "done") {
    if (payload.trace_id) {
      const t = document.createElement("div");
      t.className = "trace-ref";
      t.textContent = "Trace ID: " + payload.trace_id;
      els.finalOut.appendChild(t);
    }
  }
}

els.analyzeBtn.addEventListener("click", runAnalysis);
renderExamples();
checkHealth();
