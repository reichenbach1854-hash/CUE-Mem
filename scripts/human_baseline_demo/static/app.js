const state = {
  payload: null,
  answers: {},
  pointerStart: null,
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pointLabel(point) {
  const labels = {
    pref_img: "Long Pattern / image",
    pref_text: "Long Pattern / text",
    entity_img: "Entity Recall / image",
    entity_text: "Entity Recall / text",
    rec_img: "Recommendation / image",
    rec_text: "Recommendation / text",
    adversarial_text: "Answer Refusal",
  };
  return labels[point] || point || "unknown";
}

function currentVariant() {
  return $("variantSelect").value || "full_multimodal";
}

function currentProfile() {
  return new URLSearchParams(window.location.search).get("profile") || "";
}

function isTextVariant() {
  return currentVariant().endsWith("_text");
}

function isOracleVariant() {
  return currentVariant().startsWith("oracle_");
}

function variantName(id) {
  const item = (state.payload?.variants || []).find((variant) => variant.id === id);
  return item?.name || id;
}

function renderSessions() {
  $("historyTitle").textContent = isOracleVariant() ? "Oracle supporting clues" : "对话历史";
  const root = $("sessions");

  if (isOracleVariant()) {
    root.innerHTML = `
      <div class="empty-state">
        当前是 <strong>${esc(variantName(currentVariant()))}</strong>。完整历史已隐藏；每道题只展示它自己的 supporting clues。
      </div>
    `;
    return;
  }

  const filter = $("sessionFilter").value.trim().toLowerCase();
  const sessions = state.payload.sessions.filter((session) => {
    const blob = JSON.stringify(session).toLowerCase();
    return !filter || blob.includes(filter);
  });

  root.innerHTML = sessions.map((session, idx) => `
    <details class="session" ${idx < 2 ? "open" : ""}>
      <summary>${esc(session.session_id)} · ${esc(session.date || "")} · ${esc(session.task_id || "")}</summary>
      ${renderSessionMeta(session)}
      ${session.dialogues.map((turn) => renderTurn(turn)).join("")}
    </details>
  `).join("");
}

function renderSessionMeta(session) {
  if (!isTextVariant()) return "";
  const rows = [];
  if (session.scene_description) rows.push(`<div><strong>Scene:</strong> ${esc(session.scene_description)}</div>`);
  if (session.user_shared_image_description && session.user_shared_image_description !== "none") {
    rows.push(`<div><strong>Image caption:</strong> ${esc(session.user_shared_image_description)}</div>`);
  }
  if (session.background_audio_info && session.background_audio_info !== "none") {
    rows.push(`<div><strong>Background audio:</strong> ${esc(session.background_audio_info)}</div>`);
  }
  return rows.length ? `<div class="session-meta">${rows.join("")}</div>` : "";
}

function renderTurn(turn) {
  if (isTextVariant()) {
    const imageCaptions = (turn.images || [])
      .map((item) => item.caption ? `<div class="caption-box"><strong>${esc(item.id)} image:</strong> ${esc(item.caption)}</div>` : "")
      .join("");
    const audioCaptions = (turn.audios || [])
      .map((item) => item.caption ? `<div class="caption-box"><strong>${esc(item.id)} audio:</strong> ${esc(item.caption)}</div>` : "")
      .join("");
    return `
      <div class="turn text-only">
        <div class="round">${esc(turn.round)}</div>
        <div class="utterance"><strong>User:</strong> ${esc(turn.textual_user || turn.user || "")}</div>
        <div class="utterance assistant"><strong>Assistant:</strong> ${esc(turn.assistant || "")}</div>
        ${imageCaptions}${audioCaptions}
      </div>
    `;
  }

  const images = (turn.images || []).map((item) => `
    <div class="media-card">
      <div class="caption">${esc(item.id)}</div>
      ${item.url ? `<img src="${esc(item.url)}" alt="${esc(item.id)}" loading="lazy" />` : `<div class="bad">图片缺失</div>`}
    </div>
  `).join("");
  const audios = (turn.audios || []).map((item) => `
    <div class="media-card">
      <div class="caption">${esc(item.id)}</div>
      ${item.url ? `<audio controls preload="none" src="${esc(item.url)}"></audio>` : `<div class="bad">音频缺失</div>`}
    </div>
  `).join("");
  const media = images || audios ? `<div class="media-row">${images}${audios}</div>` : "";
  return `
    <div class="turn">
      <div class="round">${esc(turn.round)}</div>
      <div class="utterance"><strong>User:</strong> ${esc(turn.user || "")}</div>
      <div class="utterance assistant"><strong>Assistant:</strong> ${esc(turn.assistant || "")}</div>
      ${media}
    </div>
  `;
}

function renderFilters() {
  const points = [...new Set(state.payload.qas.map((qa) => qa.point).filter(Boolean))].sort();
  $("pointFilter").innerHTML = `<option value="">全部类型</option>` +
    points.map((point) => `<option value="${esc(point)}">${esc(pointLabel(point))}</option>`).join("");

  const types = [...new Set(state.payload.qas.map((qa) => qa.qa_type).filter(Boolean))].sort();
  $("qaTypeFilter").innerHTML = `<option value="">全部显隐式</option>` +
    types.map((type) => `<option value="${esc(type)}">${esc(type)}</option>`).join("");
}

function filteredQas() {
  const point = $("pointFilter").value;
  const type = $("qaTypeFilter").value;
  return state.payload.qas.filter((qa) => {
    if (point && qa.point !== point) return false;
    if (type && qa.qa_type !== type) return false;
    return true;
  });
}

function renderQas() {
  const qas = filteredQas();
  $("progressText").textContent = `${Object.keys(state.answers).length}/${state.payload.qas.length} answered`;
  $("qas").innerHTML = qas.map((qa, idx) => renderQa(qa, idx)).join("");
}

function renderQa(qa, idx) {
  const selected = state.answers[qa.key] || "";
  const oracle = isOracleVariant() ? renderOracleClues(qa) : "";
  return `
    <article class="qa-card">
      <div class="qa-head">
        <div>
          <div class="qa-id">#${idx + 1} ${esc(qa.qa_id)}</div>
          <div class="chips">
            <span class="chip">${esc(pointLabel(qa.point))}</span>
            <span class="chip">${esc(qa.qa_type || "unknown")}</span>
            ${qa.category ? `<span class="chip">${esc(qa.category)}</span>` : ""}
            ${qa.subcategory ? `<span class="chip">${esc(qa.subcategory)}</span>` : ""}
          </div>
        </div>
        <button class="secondary" onclick="scrollToSession('${esc(qa.session_id)}')">${esc(qa.session_id || "session")}</button>
      </div>
      <div class="question">${esc(qa.question_stem || qa.question)}</div>
      <div class="options">
        ${["A", "B", "C", "D"].map((choice) => renderOption(qa, choice, selected)).join("")}
      </div>
      ${oracle}
      <div class="qa-foot">
        <span class="clues">clues: ${(qa.clue || []).map(esc).join(", ")}</span>
        <span>${selected ? `当前选择：${selected}` : "未作答"}</span>
      </div>
    </article>
  `;
}

function renderOption(qa, choice, selected) {
  const option = qa.options[choice] || {};
  const img = !isTextVariant() && option.image_url ? `<img src="${esc(option.image_url)}" alt="${choice}" loading="lazy" />` : "";
  const text = option.text || (isTextVariant() ? (option.caption || option.description || "") : "");
  return `
    <div class="option ${selected === choice ? "selected" : ""}" data-qa-key="${esc(qa.key)}" data-choice="${choice}">
      <div class="choice">${choice}</div>
      <div>
        ${img}
        ${text ? `<div class="option-text">${esc(text)}</div>` : ""}
      </div>
    </div>
  `;
}

function renderOracleClues(qa) {
  const clues = qa.oracle_clues || [];
  const title = `Supporting clues (${clues.length})`;
  if (!clues.length) {
    return `
      <details class="oracle-clues">
        <summary>${esc(title)}</summary>
        <div class="oracle-clues-body">
          <div class="empty-state">无 clue</div>
        </div>
      </details>
    `;
  }
  return `
    <details class="oracle-clues">
      <summary>${esc(title)}</summary>
      <div class="oracle-clues-body">
        ${clues.map((clue) => renderOracleClue(clue)).join("")}
      </div>
    </details>
  `;
}

function renderOracleClue(clue) {
  const modality = clue.modality || "missing";
  if (isTextVariant()) {
    let body = "";
    if (modality === "text") {
      body = `<div><strong>User:</strong> ${esc(clue.text || "")}</div>${clue.assistant ? `<div><strong>Assistant:</strong> ${esc(clue.assistant)}</div>` : ""}`;
    } else if (modality === "image") {
      body = `<div>${esc(clue.caption || "[image clue: no caption]")}</div>`;
    } else if (modality === "audio") {
      body = `<div>${esc(clue.caption || "[audio clue: no caption]")}</div>`;
    } else {
      body = `<div class="bad">未能解析该 clue</div>`;
    }
    return `
      <div class="clue-card">
        <div class="clue-head">${esc(clue.clue)} · ${esc(modality)} · ${esc(clue.round || clue.session_id || "")}</div>
        <div class="clue-body">${body}</div>
      </div>
    `;
  }

  let body = "";
  if (modality === "text") {
    body = `<div><strong>User:</strong> ${esc(clue.text || "")}</div>${clue.assistant ? `<div><strong>Assistant:</strong> ${esc(clue.assistant)}</div>` : ""}`;
  } else if (modality === "image") {
    body = clue.url ? `<img class="clue-image" src="${esc(clue.url)}" alt="${esc(clue.clue)}" loading="lazy" />` : `<div class="bad">图片缺失</div>`;
  } else if (modality === "audio") {
    body = clue.url ? `<audio controls preload="none" src="${esc(clue.url)}"></audio>` : `<div class="bad">音频缺失</div>`;
  } else {
    body = `<div class="bad">未能解析该 clue</div>`;
  }
  return `
    <div class="clue-card">
      <div class="clue-head">${esc(clue.clue)} · ${esc(modality)} · ${esc(clue.round || clue.session_id || "")}</div>
      <div class="clue-body">${body}</div>
    </div>
  `;
}

function selectAnswer(qaId, choice) {
  state.answers[qaId] = choice;
  renderQas();
}

function optionFromEvent(event) {
  return event.target.closest(".option");
}

function handleOptionPointerDown(event) {
  const option = optionFromEvent(event);
  if (!option) return;
  state.pointerStart = {
    x: event.clientX,
    y: event.clientY,
    key: option.dataset.qaKey,
    choice: option.dataset.choice,
  };
}

function handleOptionPointerUp(event) {
  const option = optionFromEvent(event);
  const start = state.pointerStart;
  state.pointerStart = null;
  if (!option || !start) return;
  if (start.key !== option.dataset.qaKey || start.choice !== option.dataset.choice) return;

  const dx = Math.abs(event.clientX - start.x);
  const dy = Math.abs(event.clientY - start.y);
  if (dx > 8 || dy > 8) return;

  selectAnswer(start.key, start.choice);
}

function scrollToSession(sessionId) {
  if (isOracleVariant()) return;
  if (!sessionId) return;
  $("sessionFilter").value = sessionId;
  renderSessions();
  $("sessions").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function submitAnswers() {
  const resp = await fetch("/api/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      participant: $("participant").value,
      variant: currentVariant(),
      profile: currentProfile(),
      answers: state.answers,
    }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    alert(data.error || "submit failed");
    return;
  }
  const result = data.result;
  $("resultSummary").innerHTML = `
    <div><strong>Variant:</strong> ${esc(variantName(result.variant))}</div>
    <div><strong>Profile:</strong> ${esc(result.profile || state.payload.profile_id || "default")}</div>
    <div><strong>Participant:</strong> ${esc(result.participant)}</div>
    <div><strong>Total:</strong> ${result.total}</div>
    <div><strong>Answered:</strong> ${result.answered}</div>
    <div><strong>Correct:</strong> <span class="ok">${result.correct}</span></div>
    <div><strong>Accuracy:</strong> ${result.accuracy}%</div>
    <div><strong>Answered accuracy:</strong> ${result.answered_accuracy}%</div>
    <div><strong>Saved:</strong> ${esc(data.saved_to)}</div>
  `;
  $("resultDialog").showModal();
}

async function main() {
  const apiUrl = new URL("/api/data", window.location.href);
  const profile = currentProfile();
  if (profile) apiUrl.searchParams.set("profile", profile);
  const resp = await fetch(apiUrl);
  state.payload = await resp.json();
  if (!resp.ok) {
    throw new Error(state.payload.error || "failed to load data");
  }
  $("variantSelect").innerHTML = (state.payload.variants || []).map((variant) =>
    `<option value="${esc(variant.id)}">${esc(variant.name)}</option>`
  ).join("");
  const requestedVariant = new URLSearchParams(window.location.search).get("variant");
  if ((state.payload.variants || []).some((variant) => variant.id === requestedVariant)) {
    $("variantSelect").value = requestedVariant;
  }
  $("profileLine").textContent =
    `${state.payload.profile_id || "default"} · ${state.payload.profile?.name || "unknown"} · sessions=${state.payload.sessions.length} · QA=${state.payload.qas.length} · ${state.payload.data_file}`;
  renderFilters();
  renderSessions();
  renderQas();
}

$("sessionFilter").addEventListener("input", renderSessions);
$("variantSelect").addEventListener("change", () => {
  const url = new URL(window.location.href);
  url.searchParams.set("variant", currentVariant());
  const profile = currentProfile();
  if (profile) url.searchParams.set("profile", profile);
  history.replaceState(null, "", url);
  renderSessions();
  renderQas();
});
$("pointFilter").addEventListener("change", renderQas);
$("qaTypeFilter").addEventListener("change", renderQas);
$("qas").addEventListener("pointerdown", handleOptionPointerDown);
$("qas").addEventListener("pointerup", handleOptionPointerUp);
$("submitBtn").addEventListener("click", submitAnswers);
$("closeDialog").addEventListener("click", () => $("resultDialog").close());

main().catch((err) => {
  console.error(err);
  document.body.innerHTML = `<pre>${esc(err.stack || err)}</pre>`;
});
