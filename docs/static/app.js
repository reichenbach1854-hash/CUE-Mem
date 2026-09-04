const state = {
  cases: [],
  current: null,
  cards: [],
  arrows: [],
  nextButtons: [],
  previousButtons: [],
  visibleCount: 0,
  busy: false,
  runId: 0,
  qaCases: [],
  results: null,
  language: window.localStorage.getItem("cue-mem-language") || "zh",
};

const I18N = {
  zh: {
    benchmarkEyebrow: "MULTIMODAL MEMORY BENCHMARK", benchmarkIntro: "现有长期记忆评测多数只关注文本或显式多模态信息，对用户消息中反复出现的隐式线索关注不足：<br /><span class=\"benchmark-intro-list\">1、图片背景、边缘<br />2、语音消息的背景音</span><br />本研究构建 CUE-Mem Benchmark，旨在评估多模态智能体从隐式线索中检索并应用长期用户记忆的能力。", users: "users", qaPairs: "QA pairs", sessions: "sessions",
    dialogueTurns: "dialogue turns", images: "images", audioClips: "audio clips", paperSoon: "Paper · coming soon",
    navConstruction: "01 · 数据构造", navQa: "02 · QA 案例", navResults: "03 · 实验结果",
    dataConstruction: "数据构造", constructionTitle: "Benchmark 数据<br /><span>构造流程</span>",
    constructionDescription: "两个代表性案例展示了如何将用户画像偏好转化为结构化事件和多模态历史。<br />选择一个案例，使用步骤按钮查看完整的构造过程。",
    playFlow: "播放构造流程", nextStep: "下一步", reset: "重置", qaExamples: "QA EXAMPLES", qaTitle: "四类评测任务",
    qaDescription: "每个案例先展示问题和检索到的记忆线索。展开答案面板可查看标准答案及其考察的偏好。",
    experimentalResults: "实验结果 · DEMO V3", resultsTitle: "三个研究问题",
    resultsDescription: "结果整理自论文。切换 RQ1、RQ2 和 RQ3，查看主要对比、caption 质量权衡及原生多模态检索行为。",
    caseSources: ["Profile records", "Event annotations", "Dialogue history", "Media outputs"],
    memoryClues: "MEMORY CLUES", question: "QUESTION", viewAnswer: "查看答案", goldAnswer: "GOLD ANSWER", preference: "PREFERENCE EVALUATED",
    explicit: "Explicit", implicit: "Implicit", eiGap: "EI-Gap", experimentalConclusion: "EXPERIMENTAL CONCLUSION",
    caseLoading: "正在读取构造案例…", qaLoading: "正在读取 QA 案例…", resultLoading: "正在读取实验结果…", ready: "准备播放", complete: "流程已完成",
  },
  en: {
    benchmarkEyebrow: "MULTIMODAL MEMORY BENCHMARK", benchmarkIntro: "Most existing long-term memory evaluations focus on text or explicit multimodal information, paying limited attention to implicit cues that recur in user messages:<br /><span class=\"benchmark-intro-list\">1. Image backgrounds and peripheral details<br />2. Background sounds in voice messages</span><br />CUE-Mem is designed to evaluate whether multimodal agents can retrieve and apply long-term user memory from such implicit cues.", users: "users", qaPairs: "QA pairs", sessions: "sessions",
    dialogueTurns: "dialogue turns", images: "images", audioClips: "audio clips", paperSoon: "Paper · coming soon",
    navConstruction: "01 · Data construction", navQa: "02 · QA cases", navResults: "03 · Results",
    dataConstruction: "DATA CONSTRUCTION", constructionTitle: "Benchmark data<br /><span>construction process</span>",
    constructionDescription: "Two representative cases illustrate how profile-level preferences are transformed into structured events and multimodal history.<br />Select a case, then use the step controls to reveal the construction sequence.",
    playFlow: "Play construction flow", nextStep: "Next", reset: "Reset", qaExamples: "QA EXAMPLES", qaTitle: "Four evaluation settings",
    qaDescription: "Each example presents the question and retrieved memory clues first. Expand the answer panel to reveal the gold answer and the preference being evaluated.",
    experimentalResults: "EXPERIMENTAL RESULTS · DEMO V3", resultsTitle: "Three research questions",
    resultsDescription: "Results transcribed from the paper. Switch between RQ1, RQ2, and RQ3 to inspect the main comparison, caption-quality trade-off, and native multimodal retrieval behavior.",
    caseSources: ["Profile records", "Event annotations", "Dialogue history", "Media outputs"],
    memoryClues: "MEMORY CLUES", question: "QUESTION", viewAnswer: "View answer", goldAnswer: "GOLD ANSWER", preference: "PREFERENCE EVALUATED",
    explicit: "Explicit", implicit: "Implicit", eiGap: "EI-Gap", experimentalConclusion: "EXPERIMENTAL CONCLUSION",
    caseLoading: "Loading construction case…", qaLoading: "Loading QA cases…", resultLoading: "Loading experiment results…", ready: "Ready", complete: "Flow complete",
  },
};

function t(key) {
  return I18N[state.language][key] ?? I18N.en[key] ?? key;
}

function applyLanguage() {
  document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.innerHTML = t(element.dataset.i18n);
  });
  const toggle = document.getElementById("language-toggle");
  if (toggle) {
    toggle.textContent = state.language === "zh" ? "EN" : "中";
    toggle.setAttribute("aria-label", state.language === "zh" ? "Switch to English" : "切换到中文");
    toggle.title = toggle.getAttribute("aria-label");
  }
}

function toggleLanguage() {
  state.language = state.language === "zh" ? "en" : "zh";
  window.localStorage.setItem("cue-mem-language", state.language);
  applyLanguage();
  if (state.qaCases.length) renderQaCases(state.qaCases);
  if (state.results) renderResults(state.results);
  if (state.cases.length) {
    renderTabs();
    if (state.current) {
      renderSummary(state.current);
      renderFlow();
    }
  }
}

const refs = {
  tabs: document.getElementById("case-tabs"),
  summary: document.getElementById("case-summary"),
  title: document.getElementById("flow-title"),
  canvas: document.getElementById("flow-canvas"),
  play: document.getElementById("play-button"),
  next: document.getElementById("next-button"),
  reset: document.getElementById("reset-button"),
  progress: document.getElementById("progress-bar"),
  progressLabel: document.getElementById("progress-label"),
  qaGrid: document.getElementById("qa-grid"),
  resultsTabs: document.getElementById("results-tabs"),
  resultsPanel: document.getElementById("results-panel"),
  pageNav: document.getElementById("page-nav"),
  pages: document.querySelectorAll("[data-page-view]"),
  hero: document.querySelector(".hero"),
};

const themeColors = {
  mint: "#a4c9b8",
  lavender: "#b6afd1",
  peach: "#e5a98e",
  blue: "#7da7bf",
  coral: "#d87562",
};

function node(tag, className = "", text = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== "") element.textContent = text;
  return element;
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function showPage(pageId, updateHash = false) {
  const validPages = new Set(["construction", "qa", "results"]);
  const page = validPages.has(pageId) ? pageId : "construction";
  refs.pages.forEach((element) => {
    if (element.hasAttribute("data-construction-hero")) return;
    element.classList.toggle("is-active", element.dataset.pageView === page);
  });
  refs.hero.classList.toggle("construction-active", page === "construction");
  refs.pageNav.querySelectorAll("a").forEach((link) => {
    const active = link.dataset.page === page;
    link.classList.toggle("is-active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (updateHash && window.location.hash !== `#${page}`) {
    window.history.pushState({}, "", `#${page}`);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function formatResultNumber(value) {
  return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(1);
}

async function getJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const value = await response.json();
  const rewrite = (item) => {
    if (Array.isArray(item)) return item.map(rewrite);
    if (item && typeof item === "object") return Object.fromEntries(Object.entries(item).map(([key, value]) => [key, rewrite(value)]));
    if (typeof item === "string" && item.startsWith("/media/")) return `./media/${item.slice("/media/".length)}`;
    if (typeof item === "string" && item.startsWith("/static/")) return `./static/${item.slice("/static/".length)}`;
    return item;
  };
  return rewrite(value);
}

function renderTabs() {
  refs.tabs.replaceChildren();
  state.cases.forEach((item) => {
    const button = node("button", "case-tab");
    button.type = "button";
    button.dataset.caseId = item.id;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", "false");

    button.append(
      node("span", "case-tab-kicker", item.badge),
      node("span", "case-tab-title", item.title),
      node("span", "case-tab-subtitle", item.subtitle),
    );
    button.addEventListener("click", () => selectCase(item.id));
    refs.tabs.append(button);
  });
}

function renderSummary(caseInfo) {
  refs.summary.replaceChildren();
  const badge = node("span", "summary-badge", caseInfo.badge);
  const title = node("h3", "", caseInfo.title);
  const description = node("p", "", caseInfo.description);
  const sourceList = node("div", "summary-sources");
  t("caseSources").forEach((source) => {
    sourceList.append(node("span", "", source));
  });
  refs.summary.append(badge, title, description, sourceList);

  refs.tabs.querySelectorAll(".case-tab").forEach((tab) => {
    const active = tab.dataset.caseId === caseInfo.id;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
}

function renderValue(container, value) {
  if (Array.isArray(value)) {
    if (!value.length) {
      container.textContent = "—";
      return;
    }
    const isParagraph = value.some(
      (item) => typeof item === "string" && item.length > 45,
    );
    const list = node("ul", isParagraph ? "fact-list fact-paragraphs" : "fact-list");
    value.forEach((item) => {
      let text = item;
      if (typeof item === "object" && item !== null) {
        text = JSON.stringify(item, null, 2);
      }
      list.append(node("li", "", String(text)));
    });
    container.append(list);
    return;
  }
  if (value && typeof value === "object") {
    container.textContent = JSON.stringify(value, null, 2);
    return;
  }
  container.textContent = value === "" || value == null ? "—" : String(value);
}

function renderFacts(facts) {
  if (!facts || !facts.length) return null;
  const grid = node("div", "facts-grid");
  facts.forEach((item) => {
    const card = node("div", "fact-card");
    const rawValue = item.value;
    const stringLength = Array.isArray(rawValue)
      ? rawValue.join(" ").length
      : String(rawValue ?? "").length;
    if (stringLength > 190 || ["background_audio_info", "user_shared_image_description"].includes(item.label)) {
      card.classList.add("fact-wide");
    }
    card.append(node("span", "fact-label", item.label));
    const value = node("div", "fact-value");
    renderValue(value, rawValue);
    card.append(value);
    grid.append(card);
  });
  return grid;
}

function renderMedia(mediaItems, step) {
  if (!mediaItems || !mediaItems.length) return null;
  const onlyAudio = mediaItems.every((item) => item.kind === "audio");
  const wrapper = node("div", onlyAudio && mediaItems.length > 1 ? "media-grid audio-list" : "media-grid");

  mediaItems.forEach((media) => {
    const card = node("article", "media-card");
    if (media.label && media.label.toLowerCase().includes("final")) {
      card.classList.add("media-final");
    }
    if (media.kind === "audio") {
      card.append(createAudioPlayer(media.url, media.label));
    } else {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.src = media.url;
      image.alt = media.caption || media.label || "CUE-Mem media";
      image.loading = "lazy";
      image.title = state.language === "zh" ? "点击打开原图" : "Click to open original image";
      image.addEventListener("click", () => window.open(media.url, "_blank", "noopener"));
      figure.append(image);
      card.append(figure);
    }

    const body = node("div", "media-card-body");
    body.append(node("span", "media-label", media.label || media.kind));
    if (media.caption) body.append(node("p", "media-caption", media.caption));
    card.append(body);
    wrapper.append(card);
  });
  return wrapper;
}

function retryAudioSource(audio) {
  if (audio.dataset.retried === "true") return;
  const source = new URL(audio.src, window.location.origin);
  const eventPrefix = "/media/event/voice_mixed/";
  if (!source.pathname.startsWith(eventPrefix)) return;
  audio.dataset.retried = "true";
  source.pathname = source.pathname.replace(eventPrefix, "/media/voice_mixed/");
  audio.src = source.href;
  audio.load();
}

function formatAudioTime(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function createAudioPlayer(url, label) {
  const wrapper = node("div", "custom-audio");
  wrapper.title = label || "Audio clip";
  const button = node("button", "custom-audio-play", "");
  button.type = "button";
  button.setAttribute("aria-label", `Play ${label || "audio clip"}`);
  button.setAttribute("title", `Play ${label || "audio clip"}`);
  const progress = document.createElement("input");
  progress.type = "range";
  progress.className = "custom-audio-progress";
  progress.min = "0";
  progress.max = "100";
  progress.value = "0";
  const time = node("span", "custom-audio-time", "0:00 / 0:00");
  const audio = document.createElement("audio");
  audio.preload = "none";
  audio.src = url;
  audio.addEventListener("error", () => retryAudioSource(audio));
  const setButtonState = (playing, failed = false) => {
    button.classList.toggle("is-playing", playing);
    button.classList.toggle("is-error", failed);
    button.setAttribute("aria-label", failed ? "Audio unavailable" : `${playing ? "Pause" : "Play"} ${label || "audio clip"}`);
    button.title = failed ? "Audio unavailable" : `${playing ? "Pause" : "Play"} ${label || "audio clip"}`;
  };
  button.addEventListener("click", async () => {
    if (audio.paused) {
      try { await audio.play(); setButtonState(true); }
      catch (_error) { setButtonState(false, true); }
    } else { audio.pause(); setButtonState(false); }
  });
  audio.addEventListener("loadedmetadata", () => { time.textContent = `0:00 / ${formatAudioTime(audio.duration)}`; });
  audio.addEventListener("timeupdate", () => {
    const value = audio.duration ? audio.currentTime / audio.duration * 100 : 0;
    progress.value = String(value);
    progress.style.setProperty("--progress", `${value}%`);
    time.textContent = `${formatAudioTime(audio.currentTime)} / ${formatAudioTime(audio.duration)}`;
  });
  audio.addEventListener("ended", () => { setButtonState(false); progress.value = "0"; progress.style.setProperty("--progress", "0%"); });
  audio.addEventListener("pause", () => { if (!audio.ended) setButtonState(false); });
  audio.addEventListener("error", () => setButtonState(false, true));
  progress.addEventListener("input", () => { if (audio.duration) audio.currentTime = Number(progress.value) / 100 * audio.duration; });
  wrapper.append(button, progress, time, audio);
  return wrapper;
}

function renderDialogue(dialogue) {
  if (!dialogue || !dialogue.length) return null;
  const list = node("div", "dialogue-list");
  dialogue.forEach((item) => {
    const line = node("div", "dialogue-line");
    line.append(node("span", "dialogue-round", item.round || ""));
    const bubbles = node("div");
    if (item.user) {
      const userBubble = node("div", "dialogue-bubble");
      userBubble.append(node("span", "dialogue-role", "USER / voice message"));
      userBubble.append(node("span", "", item.user));
      bubbles.append(userBubble);
    }
    if (item.assistant) {
      const assistantBubble = node("div", "dialogue-bubble assistant");
      assistantBubble.append(node("span", "dialogue-role", "ASSISTANT"));
      assistantBubble.append(node("span", "", item.assistant));
      bubbles.append(assistantBubble);
    }
    line.append(bubbles);
    list.append(line);
  });
  return list;
}

function renderQaOptionImages(items) {
  if (!items || !items.length) return null;
  const grid = node("div", "qa-option-grid");
  items.forEach((item) => {
    const figure = node("figure", "qa-option");
    const image = document.createElement("img");
    image.src = item.media.url;
    image.alt = `Option ${item.option}`;
    image.loading = "lazy";
    image.addEventListener("click", () => window.open(item.media.url, "_blank", "noopener"));
    const caption = node("figcaption");
    caption.append(node("strong", "qa-option-letter", item.option));
    if (item.media.caption) caption.append(node("span", "", item.media.caption));
    figure.append(image, caption);
    grid.append(figure);
  });
  return grid;
}

function renderQaTextOptions(options) {
  if (!options || !Object.keys(options).length) return null;
  const list = node("div", "qa-text-options");
  Object.entries(options).forEach(([letter, value]) => {
    const item = node("div", "qa-text-option");
    item.append(node("strong", "qa-option-letter", letter), node("span", "", value));
    list.append(item);
  });
  return list;
}

function renderQaClues(mediaItems) {
  if (!mediaItems || !mediaItems.length) return null;
  const images = mediaItems.filter((item) => item.kind === "image");
  const audio = mediaItems.filter((item) => item.kind === "audio");
  const wrapper = node("div", "qa-clues");
  wrapper.append(node("p", "qa-subheading", t("memoryClues")));

  if (images.length) {
    const imageGrid = node("div", "qa-clue-image-grid");
    images.forEach((media) => {
      const figure = node("figure", "qa-clue-image");
      const image = document.createElement("img");
      image.src = media.url;
      image.alt = "Memory visual clue";
      image.loading = "lazy";
      image.addEventListener("click", () => window.open(media.url, "_blank", "noopener"));
      figure.append(image, node("figcaption", "", media.label || "Visual clue"));
      imageGrid.append(figure);
    });
    wrapper.append(imageGrid);
  }

  if (audio.length) {
    const audioGrid = node("div", "qa-clue-audio-grid");
    audio.forEach((media, index) => {
      const item = node("div", "qa-clue-audio");
      item.append(node("span", "qa-audio-label", media.label || `Audio clue ${index + 1}`));
      item.append(createAudioPlayer(media.url, media.label || `Audio clue ${index + 1}`));
      audioGrid.append(item);
    });
    wrapper.append(audioGrid);
  }
  return wrapper;
}

function renderQaCase(qaCase) {
  const article = node("article", "qa-card");
  const header = node("header", "qa-card-header");
  header.append(
    node("span", "qa-card-kicker", qaCase.category || "CUE-Mem QA"),
    node("h3", "", qaCase.title),
    node("p", "", qaCase.description),
  );
  article.append(header);

  const body = node("div", "qa-card-body");
  const meta = node("div", "qa-meta");
  if (qaCase.subcategory) meta.append(node("span", "", qaCase.subcategory));
  if (qaCase.qa_type) meta.append(node("span", "", qaCase.qa_type));
  body.append(meta);

  const question = node("div", "qa-question");
  question.append(node("span", "qa-subheading", t("question")));
  question.append(node("p", "qa-question-text", qaCase.question));
  const optionImages = renderQaOptionImages(qaCase.option_images);
  if (optionImages) question.append(optionImages);
  const textOptions = renderQaTextOptions(qaCase.options);
  if (textOptions) question.append(textOptions);
  body.append(question);

  const clues = renderQaClues(qaCase.clue_media);
  if (clues) body.append(clues);

  const answer = document.createElement("details");
  answer.className = "qa-answer";
  const summary = document.createElement("summary");
  summary.append(node("span", "", t("viewAnswer")), node("span", "qa-answer-toggle", "+"));
  answer.append(summary);
  const answerBody = node("div", "qa-answer-body");
  const answerValue = node("div", "qa-answer-value");
  answerValue.append(node("span", "qa-subheading", t("goldAnswer")), node("strong", "", qaCase.answer));
  const preference = node("div", "qa-preference");
  preference.append(node("span", "qa-subheading", t("preference")), node("p", "", qaCase.preference || "—"));
  answerBody.append(answerValue, preference);
  if (qaCase.answer_note) {
    const note = node("div", "qa-answer-note");
    note.append(node("span", "qa-subheading", "WHY THIS ANSWER"), node("p", "", qaCase.answer_note));
    answerBody.append(note);
  }
  if (qaCase.adversarial_type) {
    const refusalInfo = node("div", "qa-refusal-info");
    refusalInfo.append(
      node("span", "qa-subheading", "ANSWER REFUSAL · 8 SUBTYPES"),
      node("p", "qa-refusal-summary", state.language === "zh"
        ? `本任务共包含 8 个小类；本题属于：${qaCase.adversarial_type_label}（${qaCase.adversarial_type}）`
        : `This task contains 8 subtypes; this case is ${qaCase.adversarial_type} (${qaCase.adversarial_type_label}).`),
    );
    const subtypeList = node("div", "qa-subtype-list");
    qaCase.adversarial_subtypes.forEach(([key, value]) => {
      subtypeList.append(node("span", key === qaCase.adversarial_type ? "is-current" : "", `${value} · ${key}`));
    });
    refusalInfo.append(subtypeList);
    answerBody.append(refusalInfo);
  }
  if (qaCase.adversarial_answer) {
    const adversarial = node("div", "qa-adversarial-answer");
    adversarial.append(node("span", "qa-subheading", "ADVERSARIAL ANSWER"), node("p", "", qaCase.adversarial_answer));
    answerBody.append(adversarial);
  }
  if (qaCase.trap_reason) {
    const trap = node("div", "qa-trap-reason");
    trap.append(node("span", "qa-subheading", "TRAP REASON"), node("p", "", qaCase.trap_reason));
    answerBody.append(trap);
  }
  answer.append(answerBody);
  body.append(answer);
  article.append(body);
  return article;
}

function renderQaCases(cases) {
  refs.qaGrid.replaceChildren();
  cases.forEach((qaCase) => refs.qaGrid.append(renderQaCase(qaCase)));
}

function resultBar(value, tone = "implicit", scale = 100) {
  const wrap = node("div", "result-bar-wrap");
  const track = node("span", "result-bar-track");
  const fill = node("span", `result-bar-fill ${tone}`);
  const percent = Math.max(0, Math.min(100, ((Number(value) || 0) / scale) * 100));
  wrap.style.cssText = "display:flex;flex-direction:column;align-items:center;justify-content:flex-end;gap:4px;min-width:0;width:100%;height:72px;";
  track.style.cssText = "display:flex;flex:none;align-items:flex-end;width:34px;min-width:24px;height:52px;overflow:hidden;background:linear-gradient(to top,#e7edef 1px,transparent 1px);";
  fill.style.cssText = `display:block;width:100%;height:${percent}%;min-height:${percent > 0 ? "2px" : "0"};background:${tone === "explicit" ? "#8bb7aa" : tone === "gap" ? "#b36d57" : "#d28a70"};`;
  track.append(fill);
  wrap.append(track, node("strong", "result-number", formatResultNumber(value)));
  return wrap;
}

function resultTable(rows, columns) {
  const wide = columns.length > 4;
  const table = node("div", `result-table${wide ? " result-table-wide" : ""}`);
  const header = node("div", `result-row result-header${wide ? " result-row-wide" : ""}`);
  columns.forEach((column) => header.append(node("span", "", column.label)));
  table.append(header);
  rows.forEach((row) => {
    const line = node("div", `result-row${wide ? " result-row-wide" : ""}${row.oracle ? " oracle-row" : ""}`);
    columns.forEach((column) => {
      const value = row[column.key];
      if (column.bar) {
        line.append(resultBar(value, column.tone || "implicit", column.scale || 100));
      } else {
        line.append(node("span", column.className || "", String(value ?? "—")));
      }
    });
    table.append(line);
  });
  return table;
}

function renderModelChart(rows, labelFn = (row) => row.method) {
  const chart = node("div", "result-chart");
  const legend = node("div", "result-chart-legend");
  legend.append(
    node("span", "chart-legend-item explicit-legend", t("explicit")),
    node("span", "chart-legend-item implicit-legend", t("implicit")),
    node("span", "chart-legend-item gap-legend", t("eiGap")),
  );
  chart.append(legend);
  const plot = node("div", "result-chart-plot");
  rows.forEach((row) => {
    const group = node("div", "result-chart-group");
    const bars = node("div", "result-chart-bars");
    const explicit = node("div", "result-chart-column explicit-column");
    const explicitBar = resultBar(row.explicit, "explicit");
    explicitBar.classList.add("chart-bar");
    explicitBar.style.height = "118px";
    explicit.append(explicitBar);
    const implicit = node("div", "result-chart-column implicit-column");
    const implicitBar = resultBar(row.implicit, "implicit");
    implicitBar.classList.add("chart-bar");
    implicitBar.style.height = "118px";
    implicit.append(implicitBar);
    bars.append(explicit, implicit);
    const label = node("span", "result-chart-label", labelFn(row));
    const gap = node("span", "result-chart-gap", `Gap ${formatResultNumber(row.gap)}`);
    group.append(bars, label, gap);
    plot.append(group);
  });
  chart.append(plot);
  return chart;
}

function resultSectionIntro(result) {
  const intro = node("div", "result-intro");
  intro.append(node("p", "result-question", result.question));
  intro.append(node("h3", "", result.headline));
  intro.append(node("p", "result-callout", result.callout || ""));
  return intro;
}

function renderExperimentConclusion(text) {
  const card = node("article", "experiment-conclusion");
  card.append(node("span", "qa-subheading", t("experimentalConclusion")));
  const points = Array.isArray(text) ? text : [text];
  points.forEach((point) => card.append(node("p", "experiment-conclusion-point", point)));
  return card;
}

function renderRq1(result) {
  const section = node("div", "result-content");
  section.append(resultSectionIntro(result));
  const referenceLayout = node("div", "rq1-reference-layout");
  const human = node("article", "result-card result-human-card");
  human.append(node("div", "result-card-title", "Human reference"));
  human.append(renderModelChart(result.humans));
  referenceLayout.append(human);

  [
    ["Oracle Evidence", (row) => row.oracle],
    ["No Context", (row) => row.method === "No Context"],
  ].forEach(([title, predicate]) => {
    const card = node("article", "result-card rq1-reference-card");
    card.append(node("div", "result-card-title", title));
    const modelGrid = node("div", "rq1-reference-models");
    result.models.forEach((model) => {
      const rows = model.rows.filter(predicate);
      if (!rows.length) return;
      const submodel = node("div", "rq1-reference-model");
      submodel.append(node("div", "rq1-reference-model-title", model.name));
      submodel.append(renderModelChart(rows));
      modelGrid.append(submodel);
    });
    card.append(modelGrid);
    referenceLayout.append(card);
  });
  section.append(referenceLayout);

  const grid = node("div", "result-model-grid rq1-main-models");
  result.models.forEach((model) => {
    const card = node("article", "result-card");
    card.append(node("div", "result-card-title", model.name));
    card.append(renderModelChart(model.rows.filter((row) => !row.oracle && row.method !== "No Context")));
    grid.append(card);
  });
  section.append(grid);
  section.append(renderExperimentConclusion(state.language === "zh" ? [
    "RQ1 表明，文本化记忆系统在显式证据上的表现明显优于隐式证据；即使使用较强的记忆方法，隐式 Memory Avg. 仍显著落后于 Oracle Evidence。",
    "主要瓶颈不是回答问题本身，而是从长文本历史中保留、检索并整合分散的背景线索。",
  ] : [
    "RQ1 shows that textualized memory systems perform substantially better with explicit than implicit evidence; even strong memory methods remain far below Oracle Evidence on implicit Memory Avg.",
    "The main bottleneck is not answering the question itself, but preserving, retrieving, and integrating dispersed background cues from long textual histories.",
  ]));
  return section;
}

function seriesRow(task, values, tone = "implicit", scale = 100) {
  const row = node("div", "series-row");
  row.append(node("span", "series-label", task));
  values.forEach((value) => {
    const cell = node("div", "series-cell");
    cell.append(resultBar(value, tone, scale));
    row.append(cell);
  });
  return row;
}

function renderQualityPanel(label, data) {
  const card = node("article", "result-card quality-card");
  card.append(node("div", "result-card-title", label));
  const legend = node("div", "result-chart-legend");
  legend.append(
    node("span", "chart-legend-item explicit-legend", "Explicit"),
    node("span", "chart-legend-item implicit-legend", "Implicit"),
    node("span", "chart-legend-item gap-legend", "EI-Gap"),
  );
  card.append(legend);
  Object.entries(data.tasks).forEach(([task, values]) => {
    const taskBlock = node("div", "quality-task");
    taskBlock.append(node("div", "quality-task-name", task));
    const plot = node("div", "quality-chart-plot");
    data.levels.forEach((level, index) => {
      const group = node("div", "quality-chart-group");
      const bars = node("div", "quality-chart-bars");
      const explicitBar = resultBar(values.explicit[index], "explicit");
      explicitBar.style.height = "92px";
      const implicitBar = resultBar(values.implicit[index], "implicit");
      implicitBar.style.height = "92px";
      bars.append(explicitBar, implicitBar);
    const gap = values.explicit[index] == null || values.implicit[index] == null
        ? null : Math.max(0, values.explicit[index] - values.implicit[index]);
      group.append(bars, node("span", "quality-chart-label", level), node("span", "result-chart-gap", `Gap ${formatResultNumber(gap)}`));
      plot.append(group);
    });
    taskBlock.append(plot);
    card.append(taskBlock);
  });
  return card;
}

function renderRq2(result) {
  const section = node("div", "result-content");
  section.append(resultSectionIntro(result));
  const summary = node("div", "result-summary-strip");
  result.summary.forEach((item) => summary.append(node("span", "", item)));
  section.append(summary);
  const quality = node("div", "quality-grid");
  quality.append(renderQualityPanel("Image captions", result.caption_quality.image));
  const audioQuality = renderQualityPanel("Audio captions", result.caption_quality.audio);
  const audioNote = node("div", "rq2-audio-note");
  audioNote.append(
    node("strong", "", "Audio settings"),
    node("span", "", state.language === "zh" ? "ASR：只进行语音识别；Hint：用音频提示生成包含背景音的 caption；Split：将人声与背景音分离后分别识别和描述。" : "ASR: speech recognition only; Hint: audio prompting to generate captions that include background sounds; Split: separate speech from background audio and describe them independently."),
  );
  audioQuality.append(audioNote);
  quality.append(audioQuality);
  section.append(quality);
  const tradeoff = node("article", "result-card tradeoff-card");
  tradeoff.append(node("div", "result-card-title", "Retrieved context trade-off"));
  tradeoff.append(node("p", "result-card-note", "From Medium to Detailed captions, retrieved context grows sharply while accuracy improves unevenly."));
  result.tradeoff.forEach((item) => {
    const row = node("div", "tradeoff-row");
    row.append(node("strong", "tradeoff-label", item.label));
    row.append(node("span", "tradeoff-stat", `${item.tokens[0]}k → ${item.tokens[1]}k tokens / QA`));
    row.append(node("span", "tradeoff-stat implicit-stat", `Implicit ${item.implicit[0]} → ${item.implicit[1]}`));
    row.append(node("span", "tradeoff-stat explicit-stat", `Explicit ${item.explicit[0]} → ${item.explicit[1]}`));
    tradeoff.append(row);
  });
  const figure = document.createElement("figure");
  figure.className = "rq2-paper-figure";
  const image = document.createElement("img");
  image.src = "./static/figures/rq2_caption_quality.png";
  image.alt = "Effects and costs of cue-aware textualization from the CUE-Mem paper";
  image.loading = "lazy";
  image.addEventListener("click", () => window.open(image.src, "_blank", "noopener"));
  const caption = document.createElement("figcaption");
  caption.textContent = state.language === "zh" ? "论文 Figure：cue-aware textualization 的效果与成本；其中 (c) 展示 retrieved tokens 与准确率之间的 trade-off。" : "Paper figure: the effects and costs of cue-aware textualization; panel (c) shows the trade-off between retrieved tokens and accuracy.";
  figure.append(image, caption);
  tradeoff.append(figure);

  const interpretation = node("div", "rq2-figure-explanation");
  interpretation.append(
    node("span", "qa-subheading", "HOW TO READ THIS FIGURE"),
    node("p", "", state.language === "zh" ? "图中 (c) 表明，caption 越详细，检索到的上下文 token 数量增长越快：从 Medium 的约 1.9k tokens/QA 增加到 Detailed 的约 7.0k。与此同时，implicit accuracy 从 53.3 提升到 62.2，而 explicit accuracy 仅从 70.9 提升到 72.4。也就是说，更丰富的 caption 能保留更多隐式背景线索、改善隐式记忆，但需要付出显著的上下文和推理成本；这是一个 preservation–efficiency trade-off，而不是无代价的性能提升。" : "Panel (c) shows that more detailed captions sharply increase retrieved context, from about 1.9k tokens/QA for Medium to 7.0k for Detailed. Implicit accuracy rises from 53.3 to 62.2, while explicit accuracy increases only from 70.9 to 72.4. Richer captions preserve more background cues and improve implicit memory, but at a substantial context and reasoning cost: a preservation–efficiency trade-off rather than a free performance gain."),
  );
  tradeoff.append(interpretation);
  section.append(tradeoff);
  section.append(renderExperimentConclusion(state.language === "zh" ? [
    "RQ2 表明，更细致的图像 caption 和更充分的音频拆分能够恢复更多隐式线索，提升隐式记忆表现并缩小 EI-Gap。",
    "但这种收益并不均匀，而且会显著增加 retrieved context；文本化质量改善了线索保存，却无法消除效率与证据使用能力之间的权衡。",
  ] : [
    "RQ2 shows that more detailed image captions and more complete audio splitting recover more implicit cues, improve implicit memory, and narrow the EI-Gap.",
    "The gains are uneven and substantially increase retrieved context; better textualization preserves cues but cannot remove the trade-off between efficiency and evidence use.",
  ]));
  return section;
}

function renderRq3(result) {
  const section = node("div", "result-content");
  section.append(resultSectionIntro(result));
  const notation = node("div", "rq3-notation");
  notation.append(node("strong", "", "RQ3 notation"), node("span", "", state.language === "zh" ? "格式：Index / Use" : "Format: Index / Use"));
  const notationTable = document.createElement("table");
  notationTable.className = "rq3-notation-table";
  const head = document.createElement("tr");
  [state.language === "zh" ? "变体" : "Variant", "Index", "Use"].forEach((text) => head.append(node("th", "", text)));
  notationTable.append(head);
  [
    ["T / T", "Textualized", "Textualized"],
    ["T / M", "Textualized", "Multimodal"],
    ["M / T", "Multimodal", "Textualized"],
    ["M / M", "Multimodal", "Multimodal"],
  ].forEach((values) => {
    const row = document.createElement("tr");
    values.forEach((text, index) => row.append(node(index === 0 ? "th" : "td", index === 0 ? "rq3-variant-cell" : "", text)));
    notationTable.append(row);
  });
  notation.append(notationTable);
  section.append(notation);

  const backboneGrid = node("div", "rq3-backbone-grid");
  const variants = [
    ["T / T", "Text-Index + Text-Use"],
    ["T / M", "Text-Index + Multimodal-Use"],
    ["M / T", "MM-Index + Text-Use"],
    ["M / M", "MM-Index + Multimodal-Use"],
  ];
  result.systems.forEach((system) => {
    const card = node("article", "result-card rq3-backbone-card");
    card.append(node("div", "result-card-title", system.name));
    card.append(node("p", "result-card-note", state.language === "zh" ? "四种 Index / Use 组合的 Explicit、Implicit 与 EI-Gap 对比" : "Explicit, Implicit, and EI-Gap across four Index / Use combinations"));
    const variantGrid = node("div", "rq3-backbone-variants");
    variants.forEach(([short, description]) => {
      const row = system.rows.find((item) => {
        if (item.oracle) return false;
        const indexName = item.index === "Multimodal" ? "MM-Index" : "Text-Index";
        const useName = item.use === "Multimodal" ? "Multimodal-Use" : "Text-Use";
        return `${indexName} + ${useName}` === description;
      });
      if (!row) return;
      const variant = node("div", "rq3-backbone-variant");
      variant.append(node("span", "rq3-variant-badge", short));
      variant.append(renderModelChart([row], () => "Explicit / Implicit"));
      variantGrid.append(variant);
    });
    card.append(variantGrid);
    const oracleTitle = node("div", "rq3-oracle-title", "Oracle Evidence · reference");
    card.append(oracleTitle, renderModelChart(system.rows.filter((row) => row.oracle), (row) => `${row.index} / ${row.use}`));
    backboneGrid.append(card);
  });
  section.append(backboneGrid);

  const retrieval = node("article", "result-card retrieval-card");
  retrieval.append(node("div", "result-card-title", "Supporting-evidence retrieval"));
  retrieval.append(node("p", "result-card-note", state.language === "zh" ? "在 k = 25 时，MM-Index 将隐式召回率从 23.1 提升到 29.7，但将精确率从 12.4 降至 6.1。" : "At k = 25, MM-Index raises implicit recall from 23.1 to 29.7, but lowers precision from 12.4 to 6.1."));
  const table = node("div", "retrieval-table");
  const header = node("div", "retrieval-row retrieval-header");
  ["k", "Text-Index / R", "MM-Index / R", "Text-Index / P", "MM-Index / P"].forEach((label) => header.append(node("span", "", label)));
  table.append(header);
  result.retrieval.k.forEach((k, index) => {
    const row = node("div", "retrieval-row");
    row.append(node("strong", "", String(k)));
    row.append(node("span", "", `${result.retrieval.implicit["Text-Index"].recall[index]}`));
    row.append(node("span", "", `${result.retrieval.implicit["MM-Index"].recall[index]}`));
    row.append(node("span", "", `${result.retrieval.implicit["Text-Index"].precision[index]}`));
    row.append(node("span", "", `${result.retrieval.implicit["MM-Index"].precision[index]}`));
    table.append(row);
  });
  retrieval.append(table);
  section.append(retrieval);
  section.append(renderExperimentConclusion(state.language === "zh" ? [
    "RQ3 表明，原生多模态访问的收益取决于 backbone：MM-Use 在 Qwen 上能够改善隐式 Memory Avg.，但在 MiniCPM 上反而下降。",
    "MM-Index 虽然提高隐式召回，却明显降低精确率，引入的检索噪声抵消了收益。",
    "原生多模态访问能够保留被文本化遗漏的信息，但并不能自动解决检索和证据整合问题。",
  ] : [
    "RQ3 shows that the benefit of native multimodal access depends on the backbone: MM-Use improves implicit Memory Avg. on Qwen, but reduces it on MiniCPM.",
    "Although MM-Index improves implicit recall, it sharply lowers precision; the resulting retrieval noise offsets the recovered evidence.",
    "Native multimodal access preserves information lost in textualization, but does not automatically solve retrieval and evidence integration.",
  ]));
  return section;
}

function renderResults(results) {
  const definitions = [
    ["rq1", "RQ1", "Textualized memory"],
    ["rq2", "RQ2", "Caption quality"],
    ["rq3", "RQ3", "Native multimodality"],
  ];
  refs.resultsTabs.replaceChildren();
  const render = (id) => {
    refs.resultsPanel.replaceChildren();
    refs.resultsPanel.append(
      node("div", "result-panel-title", `${id.toUpperCase()} · ${results[id].title}`),
      id === "rq1" ? renderRq1(results[id]) : id === "rq2" ? renderRq2(results[id]) : renderRq3(results[id]),
    );
    refs.resultsTabs.querySelectorAll("button").forEach((button) => {
      const active = button.dataset.resultId === id;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
  };
  definitions.forEach(([id, label, subtitle]) => {
    const button = node("button", "result-tab");
    button.type = "button";
    button.dataset.resultId = id;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", "false");
    button.append(node("strong", "", label), node("span", "", subtitle));
    button.addEventListener("click", () => render(id));
    refs.resultsTabs.append(button);
  });
  render("rq1");
}

function renderStep(step, index, total) {
  const article = node("article", "flow-step is-hidden");
  const color = themeColors[step.theme] || themeColors.coral;
  article.style.setProperty("--step-color", color);
  article.dataset.stepNumber = step.number;

  const stepIndex = node("div", "step-index", step.number);
  const content = node("div", "step-content");
  content.append(
    node("p", "step-eyebrow", step.eyebrow || "PIPELINE STEP"),
    node("h3", "", step.title || ""),
  );
  if (step.intro) content.append(node("p", "step-intro", step.intro));

  const facts = renderFacts(step.facts);
  if (facts) content.append(facts);

  if (step.quote) {
    const quote = node("blockquote", "step-quote");
    quote.append(node("span", "quote-label", step.quote_label || "event text"));
    quote.append(node("p", "", step.quote));
    content.append(quote);
  }

  const dialogue = renderDialogue(step.dialogue);
  if (dialogue) content.append(dialogue);

  const media = renderMedia(step.media, step);
  if (media) content.append(media);

  const controls = node("div", "step-controls");
  const previousButton = node("button", "step-previous-button", state.language === "zh" ? "← 上一步" : "← Previous");
  previousButton.type = "button";
  previousButton.dataset.first = index === 0 ? "true" : "false";
  previousButton.disabled = index === 0;
  previousButton.addEventListener("click", () => {
    if (!state.busy && index > 0) {
      state.cards[index - 1]?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
  state.previousButtons.push(previousButton);
  controls.append(previousButton);

  const nextButton = node("button", "step-next-button", state.language === "zh" ? "下一步 →" : "Next →");
  nextButton.type = "button";
  nextButton.disabled = index === total - 1;
  nextButton.title = index === total - 1
    ? (state.language === "zh" ? "已经是最后一步" : "This is the last step")
    : (state.language === "zh" ? "展示下一步" : "Show next step");
  nextButton.addEventListener("click", () => {
    if (!state.busy) revealNext();
  });
  state.nextButtons.push(nextButton);
  controls.append(nextButton);
  if (index === total - 1) {
    const topButton = node("button", "step-top-button", state.language === "zh" ? "↑ 回到上方" : "↑ Back to top");
    topButton.type = "button";
    topButton.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
    controls.append(topButton);
  }
  content.append(controls);

  article.append(stepIndex, content);
  return article;
}

function renderArrow(label) {
  const arrow = node("div", "flow-arrow is-hidden");
  arrow.append(
    node("span", "arrow-line"),
    node("span", "arrow-head", "›"),
    node("span", "arrow-label", label || "next synthesis stage"),
  );
  return arrow;
}

function renderFlow() {
  refs.canvas.replaceChildren();
  state.cards = [];
  state.arrows = [];
  state.nextButtons = [];
  state.previousButtons = [];
  state.visibleCount = 0;

  const empty = node("div", "empty-state", state.language === "zh" ? "点击“播放构造流程”，按时间顺序展开 profile → event → artifact" : "Click “Play construction flow” to reveal profile → event → artifact in order");
  refs.canvas.append(empty);

  state.current.steps.forEach((step, index) => {
    const card = renderStep(step, index, state.current.steps.length);
    if (state.current.kind === "image" && step.number === "03") {
      card.classList.add("image-entity-step");
    }
    state.cards.push(card);
    refs.canvas.append(card);
    if (index < state.current.steps.length - 1) {
      const arrow = renderArrow(step.flow_label);
      state.arrows.push(arrow);
      refs.canvas.append(arrow);
    }
  });
  updateProgress();
}

function revealElement(element) {
  element.classList.remove("is-hidden");
  element.classList.add("is-new");
  window.requestAnimationFrame(() => {
    element.classList.add("is-visible");
  });
}

function updateProgress() {
  const total = state.current ? state.current.steps.length : 0;
  const percent = total ? (state.visibleCount / total) * 100 : 0;
  refs.progress.style.width = `${percent}%`;
  refs.next.disabled = state.busy || state.visibleCount >= total;
  refs.play.disabled = state.busy;
  state.nextButtons.forEach((button) => {
    button.disabled = state.busy || state.visibleCount >= total;
  });
  state.previousButtons.forEach((button) => {
    button.disabled = state.busy || button.dataset.first === "true";
  });
  if (!total || state.visibleCount === 0) {
    refs.progressLabel.textContent = t("ready");
  } else if (state.visibleCount >= total) {
    refs.progressLabel.textContent = t("complete");
  } else {
    refs.progressLabel.textContent = state.language === "zh"
      ? `已展示 ${state.visibleCount} / ${total}`
      : `${state.visibleCount} / ${total} shown`;
  }
}

async function revealNext(runId = state.runId) {
  if (!state.current || state.visibleCount >= state.current.steps.length) return false;
  const index = state.visibleCount;
  const empty = refs.canvas.querySelector(".empty-state");
  if (empty) empty.remove();

  if (index > 0) {
    revealElement(state.arrows[index - 1]);
    await sleep(330);
  }
  if (runId !== state.runId) return false;
  revealElement(state.cards[index]);
  state.visibleCount += 1;
  updateProgress();
  state.cards[index].scrollIntoView({ behavior: "smooth", block: "center" });
  return true;
}

async function playFlow() {
  if (state.busy || !state.current) return;
  resetFlow();
  state.busy = true;
  updateProgress();
  const runId = state.runId;
  while (runId === state.runId && state.visibleCount < state.current.steps.length) {
    await revealNext(runId);
    if (runId !== state.runId) return;
    if (state.visibleCount < state.current.steps.length) await sleep(2500);
  }
  if (runId === state.runId) {
    state.busy = false;
    updateProgress();
  }
}

function resetFlow() {
  state.runId += 1;
  state.busy = false;
  if (state.current) renderFlow();
  updateProgress();
}

async function selectCase(caseId) {
  state.runId += 1;
  state.busy = false;
  refs.canvas.replaceChildren(node("div", "loading-state", t("caseLoading")));
  try {
    state.current = await getJson(`./data/case-${encodeURIComponent(caseId)}.json`);
    refs.title.textContent = state.current.title;
    renderSummary(state.current);
    renderFlow();
  } catch (error) {
    refs.title.textContent = state.language === "zh" ? "无法读取案例" : "Unable to load case";
    refs.summary.replaceChildren(node("p", "", error.message));
    refs.canvas.replaceChildren(node("div", "error-state", state.language === "zh" ? `案例读取失败：${error.message}` : `Case loading failed: ${error.message}`));
  }
}

async function init() {
  applyLanguage();
  document.getElementById("language-toggle").addEventListener("click", toggleLanguage);
  refs.pageNav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => showPage(link.dataset.page, true));
  });
  window.addEventListener("hashchange", () => showPage(window.location.hash.slice(1)));
  showPage(window.location.hash.slice(1));
  refs.play.addEventListener("click", playFlow);
  refs.next.addEventListener("click", () => {
    if (!state.busy) revealNext();
  });
  refs.reset.addEventListener("click", resetFlow);
  try {
    const [constructionCases, qaCases, results] = await Promise.all([
      getJson("./data/cases.json"),
      getJson("./data/qa.json"),
      getJson("./data/results.json"),
    ]);
    state.cases = constructionCases;
    state.qaCases = qaCases;
    state.results = results;
    renderQaCases(qaCases);
    renderResults(results);
    renderTabs();
    if (state.cases.length) await selectCase(state.cases[0].id);
  } catch (error) {
    refs.title.textContent = "Demo server unavailable";
    refs.canvas.replaceChildren(node("div", "error-state", state.language === "zh" ? `后端连接失败：${error.message}` : `Backend connection failed: ${error.message}`));
    refs.resultsPanel.replaceChildren(node("div", "error-state", state.language === "zh" ? `实验结果读取失败：${error.message}` : `Experiment results failed to load: ${error.message}`));
  }
}

init();
