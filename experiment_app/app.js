const state = {
  protocol: null,
  participantId: "",
  sessionToken: "",
  trial: null,
  targetId: "",
  targetDepth: null,
  responseRecorded: false,
  visible: new Set(),
  observer: null,
  lastScrollEvent: 0
};

const screens = {
  consent: document.querySelector("#screen-consent"),
  baseline: document.querySelector("#screen-baseline"),
  instructions: document.querySelector("#screen-instructions"),
  intent: document.querySelector("#screen-intent"),
  task: document.querySelector("#screen-task"),
  survey: document.querySelector("#screen-survey"),
  complete: document.querySelector("#screen-complete"),
  withdrawn: document.querySelector("#screen-withdrawn")
};

function uuid() {
  return crypto.randomUUID();
}

async function post(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "请求失败");
  return body;
}

function sessionPayload(extra = {}) {
  return {
    participant_id: state.participantId,
    session_token: state.sessionToken,
    ...extra
  };
}

function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.hidden = false;
  window.setTimeout(() => { toast.hidden = true; }, 4200);
}

function showScreen(name) {
  Object.entries(screens).forEach(([key, element]) => {
    element.hidden = key !== name;
  });
  const progressMap = {
    consent: 0,
    baseline: 1,
    instructions: 2,
    intent: 2,
    task: 2,
    survey: 2,
    complete: 3,
    withdrawn: 3
  };
  const active = progressMap[name];
  document.querySelectorAll(".study-progress li").forEach((item, index) => {
    item.classList.toggle("active", index === active);
    item.classList.toggle("done", index < active);
  });
  document.querySelector("#withdraw-study").hidden =
    !state.participantId || ["complete", "withdrawn"].includes(name);
  window.scrollTo({ top: 0, behavior: "auto" });
}

function makeScale(container) {
  if (container.childElementCount) return;
  const min = Number(container.dataset.min);
  const max = Number(container.dataset.max);
  for (let value = min; value <= max; value += 1) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = container.id;
    input.value = String(value);
    input.required = true;
    const span = document.createElement("span");
    span.textContent = String(value);
    label.append(input, span);
    container.append(label);
  }
}

function selectedValue(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value ?? null;
}

function recordEvent(eventType, extra = {}) {
  if (!state.participantId || !state.trial || state.trial.complete) {
    return Promise.resolve();
  }
  return post("/api/event", sessionPayload({
    event_uuid: uuid(),
    event_type: eventType,
    trial_uuid: state.trial.trial_uuid,
    thread_id: state.trial.thread.thread_id,
    ...extra
  })).catch(reason => showToast(reason.message));
}

function clearTarget() {
  state.targetId = "";
  state.targetDepth = null;
  document.querySelector("#target-label").textContent = "尚未选择";
  document.querySelector("#response").placeholder = "先选择一条评论，再填写回复内容";
  document.querySelectorAll(".reply-target").forEach(button => {
    button.classList.remove("selected");
    button.textContent = "回复";
  });
}

function renderIntent(trial) {
  state.trial = trial;
  if (state.observer) state.observer.disconnect();
  document.querySelector("#intent-progress").textContent =
    `讨论 ${trial.trial_index + 1} / ${trial.total_trials}`;
  document.querySelector("#intent-community").textContent = trial.thread.community;
  document.querySelector("#intent-title").textContent = trial.thread.title;
  document.querySelector("#intent-prompt").textContent = trial.thread.prompt;
  document.querySelector("#intent-text").value = "";
  const choices = document.querySelector("#intent-choices");
  choices.replaceChildren();
  const options = trial.intent_choices?.length
    ? trial.intent_choices
    : state.protocol.intent_choices;
  options.forEach(option => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "intent-choice";
    input.value = option.id;
    input.required = true;
    const text = document.createElement("span");
    text.textContent = option.label;
    label.append(input, text);
    choices.append(label);
  });
  showScreen("intent");
}

function appendVoteSignal(meta, comment) {
  const votes = Number(comment.upvotes || 0) + Number(comment.downvotes || 0);
  const agreement = votes > 0
    ? Number(comment.upvotes || 0) / votes
    : Number(comment.agreement_ratio || 0.5);
  const block = document.createElement("span");
  block.className = "vote-signal";
  block.title = "支持票占全部投票的比例";
  const label = document.createElement("span");
  label.textContent = `${Math.round(agreement * 100)}% 支持`;
  const track = document.createElement("i");
  const fill = document.createElement("b");
  fill.style.width = `${Math.max(0, Math.min(100, agreement * 100))}%`;
  track.append(fill);
  block.append(label, track);
  meta.append(block);
}

function renderInteraction(trial) {
  state.trial = trial;
  state.responseRecorded = false;
  state.visible = new Set();
  clearTarget();
  document.querySelector("#trial-progress").textContent =
    `讨论 ${trial.trial_index + 1} / ${trial.total_trials}`;
  document.querySelector("#community").textContent = trial.thread.community;
  document.querySelector("#thread-title").textContent = trial.thread.title;
  document.querySelector("#thread-prompt").textContent = trial.thread.prompt;
  document.querySelector("#notice-author").textContent = trial.notice.author;
  document.querySelector("#notice-text").textContent = trial.notice.text;
  document.querySelector("#intervention-notice").dataset.type = trial.notice.type;
  document.querySelector("#response").value = "";
  document.querySelector("#character-count").textContent = "0";
  document.querySelector("#mechanism-question").textContent = trial.survey_item.text;
  const scale = document.querySelector("#mechanism-response");
  scale.dataset.left = trial.survey_item.left_anchor;
  scale.dataset.right = trial.survey_item.right_anchor;
  document.querySelector("#survey-form").reset();

  const container = document.querySelector("#thread");
  container.replaceChildren();
  trial.thread.comments.forEach((comment, position) => {
    const article = document.createElement("article");
    const depth = Number(comment.depth || 1);
    article.className = "comment";
    article.dataset.id = comment.comment_id;
    article.dataset.position = String(position + 1);
    article.dataset.depth = String(depth);
    article.style.setProperty("--depth", String(Math.max(0, depth - 1)));

    const meta = document.createElement("div");
    meta.className = "comment-meta";
    const author = document.createElement("span");
    author.textContent = `${comment.author} · 第 ${depth} 层`;
    const score = document.createElement("span");
    score.textContent = `${comment.score} 分`;
    meta.append(author, score);
    appendVoteSignal(meta, comment);

    const text = document.createElement("p");
    text.textContent = comment.text;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "reply-target";
    button.textContent = "回复";
    button.addEventListener("click", () => {
      clearTarget();
      state.targetId = comment.comment_id;
      state.targetDepth = depth;
      button.classList.add("selected");
      button.textContent = "已选择";
      document.querySelector("#target-label").textContent = comment.author;
      document.querySelector("#response").placeholder = `回复 ${comment.author}`;
      recordEvent("target_selected", {
        content_id: state.targetId,
        content_depth: depth,
        viewport_position: position + 1
      });
      document.querySelector("#response").focus();
    });
    article.append(meta, text, button);
    container.append(article);
  });

  if (state.observer) state.observer.disconnect();
  state.observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      const id = entry.target.dataset.id;
      if (entry.isIntersecting && !state.visible.has(id)) {
        state.visible.add(id);
        recordEvent("item_visible", {
          content_id: id,
          content_depth: Number(entry.target.dataset.depth),
          viewport_position: Number(entry.target.dataset.position)
        });
      }
    });
  }, { threshold: 0.65 });
  container.querySelectorAll(".comment").forEach(item => state.observer.observe(item));
  showScreen("task");
  recordEvent("thread_open");
}

function renderTrial(trial) {
  if (trial.complete) {
    state.trial = trial;
    showScreen("complete");
  } else if (trial.phase === "intent") {
    renderIntent(trial);
  } else {
    renderInteraction(trial);
  }
}

async function resumeStudy() {
  const stored = JSON.parse(localStorage.getItem("bdmtf-rct-session") || "null");
  if (!stored?.participant_id || !stored?.session_token) return false;
  state.participantId = stored.participant_id;
  state.sessionToken = stored.session_token;
  try {
    const result = await post("/api/resume", sessionPayload());
    renderTrial(result.trial);
    return true;
  } catch (reason) {
    localStorage.removeItem("bdmtf-rct-session");
    state.participantId = "";
    state.sessionToken = "";
    showToast(reason.message);
    return false;
  }
}

async function loadProtocol() {
  document.querySelectorAll(".segmented, .likert").forEach(makeScale);
  const response = await fetch("/api/protocol", { cache: "no-store" });
  state.protocol = await response.json();
  document.querySelector("#study-title").textContent = state.protocol.study_title;
  document.querySelector("#protocol-version").textContent = `方案版本 ${state.protocol.protocol_version}`;
  document.querySelector("#consent-summary").textContent = state.protocol.consent.summary;
  document.querySelector("#duration").textContent = `约 ${state.protocol.consent.duration_minutes} 分钟`;
  document.querySelector("#data-retention").textContent = state.protocol.consent.data_retention || "以知情同意书为准";
  document.querySelector("#compensation").textContent = state.protocol.consent.compensation || "以招募说明为准";
  document.querySelector("#contact").textContent = state.protocol.consent.contact || "请联系研究协调员";
  document.querySelector("#text-collection-note").textContent = state.protocol.response_text_collected
    ? "你提交的回复和初始想法将与匿名编号一起保存在本地结果中，用于行为与语义一致性分析。"
    : "系统只记录回复长度，不保存回复正文。";
  if (state.protocol.mode === "demo") {
    document.querySelector("#mode-badge").hidden = false;
    document.querySelector("#demo-notice").hidden = false;
  }
  if (!await resumeStudy()) showScreen("consent");
}

document.querySelector("#to-baseline").addEventListener("click", () => {
  if (!document.querySelector("#eligible").checked || !document.querySelector("#consent").checked) {
    return showToast("请确认参与资格并完成知情同意");
  }
  showScreen("baseline");
});
document.querySelector("#back-consent").addEventListener("click", () => showScreen("consent"));

document.querySelector("#baseline-form").addEventListener("submit", eventObject => {
  eventObject.preventDefault();
  if (!document.querySelector("#age-band").value || selectedValue("platform-use") === null || selectedValue("conflict-tolerance") === null) {
    return showToast("请完成全部三个问题");
  }
  showScreen("instructions");
});

document.querySelector("#begin-study").addEventListener("click", async () => {
  try {
    const result = await post("/api/enroll", {
      eligible: document.querySelector("#eligible").checked,
      consent: document.querySelector("#consent").checked,
      age_band: document.querySelector("#age-band").value,
      prior_platform_use: Number(selectedValue("platform-use")),
      baseline_conflict_tolerance: Number(selectedValue("conflict-tolerance"))
    });
    state.participantId = result.participant_id;
    state.sessionToken = result.session_token;
    localStorage.setItem("bdmtf-rct-session", JSON.stringify({
      participant_id: state.participantId,
      session_token: state.sessionToken
    }));
    renderTrial(result.trial);
  } catch (reason) {
    showToast(reason.message);
  }
});

document.querySelector("#intent-form").addEventListener("submit", async eventObject => {
  eventObject.preventDefault();
  const choice = selectedValue("intent-choice");
  if (choice === null) return showToast("请选择一个初始回复意图");
  try {
    const result = await post("/api/intent", sessionPayload({
      intent_uuid: uuid(),
      trial_uuid: state.trial.trial_uuid,
      intent_choice: choice,
      intent_text: document.querySelector("#intent-text").value.trim()
    }));
    renderTrial(result.trial);
  } catch (reason) {
    showToast(reason.message);
  }
});

document.querySelector("#response").addEventListener("input", eventObject => {
  document.querySelector("#character-count").textContent = String(eventObject.target.value.length);
});

async function saveResponse(skipped) {
  if (state.responseRecorded) return;
  const text = document.querySelector("#response").value.trim();
  if (!skipped && !state.targetId) return showToast("请先选择要回复的评论");
  if (!skipped && !text) return showToast("请输入回复内容，或选择不回复");
  try {
    await post("/api/response", sessionPayload({
      response_uuid: uuid(),
      trial_uuid: state.trial.trial_uuid,
      thread_id: state.trial.thread.thread_id,
      target_id: skipped ? "" : state.targetId,
      target_depth: skipped ? null : state.targetDepth,
      response_text: skipped ? "" : text,
      skipped
    }));
    state.responseRecorded = true;
    await recordEvent(skipped ? "reply_skipped" : "reply_submitted", {
      content_id: skipped ? "" : state.targetId,
      content_depth: skipped ? null : state.targetDepth,
      payload: { response_length: skipped ? 0 : text.length }
    });
    showScreen("survey");
  } catch (reason) {
    showToast(reason.message);
  }
}

document.querySelector("#reply-form").addEventListener("submit", eventObject => {
  eventObject.preventDefault();
  saveResponse(false);
});
document.querySelector("#skip-reply").addEventListener("click", () => saveResponse(true));

document.querySelector("#survey-form").addEventListener("submit", async eventObject => {
  eventObject.preventDefault();
  const response = selectedValue("mechanism-response");
  if (response === null) return showToast("请完成这道评价题");
  const payload = {
    survey_item_id: state.trial.survey_item.id,
    survey_response: Number(response)
  };
  try {
    await recordEvent("trial_complete", { payload });
    const result = await post("/api/trial-complete", sessionPayload({
      trial_uuid: state.trial.trial_uuid,
      ...payload
    }));
    renderTrial(result.trial);
  } catch (reason) {
    showToast(reason.message);
  }
});

document.querySelectorAll(".pause-action").forEach(button => {
  button.addEventListener("click", async () => {
    await recordEvent("exit");
    document.body.innerHTML = '<main class="screen"><p class="eyebrow">已保存</p><h1>可以关闭此页面</h1><p class="lead">再次启动本地研究程序时，将从当前任务继续。</p></main>';
  });
});

document.querySelector("#withdraw-study").addEventListener("click", async () => {
  if (!state.participantId) return;
  if (!window.confirm("确认撤回参与记录吗？本机保存的回复文本将被清除，且无法继续本次研究。")) return;
  try {
    await post("/api/withdraw", sessionPayload());
    localStorage.removeItem("bdmtf-rct-session");
    showScreen("withdrawn");
  } catch (reason) {
    showToast(reason.message);
  }
});

document.querySelector("#download-result").addEventListener("click", async () => {
  try {
    const bundle = await post("/api/participant-export", sessionPayload());
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${bundle.study_id}_${bundle.participant.participant_id}.json`;
    document.body.append(link);
    link.click();
    URL.revokeObjectURL(link.href);
    link.remove();
    document.querySelector("#result-status").textContent = "已下载";
    document.querySelector("#close-study").disabled = false;
  } catch (reason) {
    showToast(reason.message);
  }
});

document.querySelector("#close-study").addEventListener("click", async () => {
  try {
    await post("/api/shutdown", sessionPayload());
    localStorage.removeItem("bdmtf-rct-session");
    document.querySelector("#screen-complete").innerHTML = '<p class="eyebrow">已安全保存</p><h1>现在可以关闭浏览器</h1><p class="lead">请把刚下载的 JSON 文件发给研究者。</p>';
  } catch (reason) {
    showToast(reason.message);
  }
});

window.addEventListener("scroll", () => {
  if (state.trial?.phase !== "interaction") return;
  const now = Date.now();
  if (now - state.lastScrollEvent < 500) return;
  state.lastScrollEvent = now;
  const height = document.documentElement.scrollHeight - window.innerHeight;
  recordEvent("scroll", { scroll_depth: height > 0 ? Math.min(1, window.scrollY / height) : 1 });
});

loadProtocol().catch(reason => showToast(reason.message));
