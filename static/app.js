const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const DEMO_USER = "analyst";
const DEMO_PASS = "surgeshield2026";

let festivalData = null;
let costChart = null;
let selectedFestivalName = null;
let pendingQueue = {};      // "slice_id|window_start_ts" -> case detail, awaiting analyst review
let activeQueueKey = null;
let actionedKeys = new Set(); // rows already dismissed/watchlisted/escalated -- don't re-queue on loop repeat
let escalateTarget = null;  // {slice_id, window_start_ts} currently open in the modal

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function rowKey(slice_id, window_start_ts) { return slice_id + "|" + window_start_ts; }
function pct(x) { return x === null || x === undefined ? "n/a" : (x * 100).toFixed(1) + "%"; }
function inr(x) { return "Rs " + Math.round(x).toLocaleString("en-IN"); }

// ---------------- login ----------------
document.getElementById("loginForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const user = document.getElementById("loginUser").value.trim();
  const pass = document.getElementById("loginPass").value.trim();
  if (user === DEMO_USER && pass === DEMO_PASS) {
    sessionStorage.setItem("surgeshield_loggedIn", "1");
    document.getElementById("loginScreen").style.display = "none";
    document.getElementById("mainApp").style.display = "block";
    startOperations();
  } else {
    document.getElementById("loginError").style.display = "block";
  }
});

// Skip the login screen if this tab already authenticated (survives refresh, not new tabs)
if (sessionStorage.getItem("surgeshield_loggedIn") === "1") {
  document.getElementById("loginScreen").style.display = "none";
  document.getElementById("mainApp").style.display = "block";
  startOperations();
}

// ---------------- tabs ----------------
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "model" && !festivalData) initTimeline();
  });
});

// ---------------- operations: auto-starting, looping ticker ----------------
function feedEntry(c) {
  const div = document.createElement("div");
  div.className = "feed-entry";
  div.innerHTML = `<span>${c.slice_id} <span class="ts">${c.window_start_ts}</span></span>
                    <span class="badge badge-${c.action}">${c.action.replace("_", " ")}</span>`;
  return div;
}

function renderQueue() {
  const el = document.getElementById("queue");
  const keys = Object.keys(pendingQueue);
  document.getElementById("queueCount").textContent = keys.length;
  if (keys.length === 0) {
    el.innerHTML = `<div class="queue-empty">No cases waiting for review.</div>`;
    return;
  }
  el.innerHTML = "";
  keys.forEach(key => {
    const c = pendingQueue[key];
    const card = document.createElement("div");
    card.className = "queue-card" + (activeQueueKey === key ? " active" : "");
    card.innerHTML = `
      <div class="qc-top"><strong>${c.slice_id}</strong><span class="badge badge-${c.action}">${c.action.replace("_", " ")}</span></div>
      <div class="qc-signal">${c.dominant_signal.replace(/_/g, " ").toLowerCase()}, ${(c.calibrated_prob*100).toFixed(0)}% probability</div>
    `;
    card.addEventListener("click", () => openCase(key));
    el.appendChild(card);
  });
}

function openCase(key) {
  activeQueueKey = key;
  renderQueue();
  renderCaseDetail(pendingQueue[key]);
}

async function startOperations() {
  const streamRes = await fetch("/api/live-stream");
  if (!streamRes.ok) {
    document.getElementById("feed").innerHTML = `<div class="feed-entry">No live data available.</div>`;
    return;
  }
  const stream = await streamRes.json();

  // A disclosed, one-time replay of a real, chronologically coherent held-out window --
  // plays through exactly once (normal traffic + the curated escalations) and then stops,
  // rather than looping forever and re-showing already-resolved cases in the feed.
  for (const rowRef of stream.rows) {
    const key = rowKey(rowRef.slice_id, rowRef.window_start_ts);
    const c = await (await fetch("/api/case?" + new URLSearchParams(rowRef))).json(); // live inference

    const feedEl = document.getElementById("feed");
    feedEl.appendChild(feedEntry(c));
    feedEl.scrollTop = feedEl.scrollHeight;
    while (feedEl.children.length > 60) feedEl.removeChild(feedEl.firstChild); // keep the DOM bounded

    const needsReview = c.action === "HUMAN_REVIEW" || c.action === "URGENT_REVIEW";
    if (needsReview && !actionedKeys.has(key)) {
      pendingQueue[key] = c;
      renderQueue();
    }
    await sleep(c.action === "AUTO_CLEAR" ? 450 : 700);
  }

  const feedEl = document.getElementById("feed");
  const doneEl = document.createElement("div");
  doneEl.className = "feed-entry";
  doneEl.innerHTML = `<span style="color:var(--text-muted);font-style:italic;">End of replay window -- resolve any remaining cases in the queue above.</span>`;
  feedEl.appendChild(doneEl);
}

function renderCaseDetail(c) {
  const el = document.getElementById("caseDetail");
  el.classList.remove("empty");
  const maxAbs = Math.max(...c.shap_top_features.map(f => Math.abs(f.value)), 1e-6);
  const shapHtml = c.shap_top_features.map(f => `
    <div class="shap-row">
      <span class="fname">${f.feature}</span>
      <div class="shap-bar-track">
        <div class="shap-bar-fill ${f.value >= 0 ? "pos" : "neg"}" style="width:${(Math.abs(f.value)/maxAbs*100).toFixed(0)}%"></div>
      </div>
    </div>`).join("");

  const festivalNote = c.is_festival_window
    ? `<div class="festival-note"><strong>${c.festival_name}</strong> (${c.festival_phase}). Calendar-adjusted baseline already accounted for in this score.</div>`
    : "";
  const watchlistNote = c.is_slice_watchlisted
    ? `<div class="watchlist-note">This merchant is on the watchlist from a prior dismiss-and-watch decision.${c.watchlist_override_triggered ? " That is why this case was not auto-cleared." : ""}</div>`
    : "";

  el.innerHTML = `
    <h3><span class="badge badge-${c.action}">${c.action.replace("_", " ")}</span> ${c.slice_id}</h3>
    ${festivalNote}
    ${watchlistNote}
    <div class="detail-grid">
      <div><span class="label">Window</span><span class="value">${c.window_start_ts}</span></div>
      <div><span class="label">Category, region</span><span class="value">${c.category}, ${c.geo_region}</span></div>
      <div><span class="label">Calibrated probability</span><span class="value">${(c.calibrated_prob*100).toFixed(1)}%</span></div>
      <div><span class="label">Estimated exposure</span><span class="value">${inr(c.estimated_exposure_inr)}</span></div>
      <div><span class="label">Dominant signal</span><span class="value">${c.dominant_signal}</span></div>
      <div><span class="label">Decline rate</span><span class="value">${pct(c.decline_rate)}</span></div>
    </div>
    <div class="reasoning-box">
      <strong>Why this decision</strong>
      <ul>${c.reasoning.map(r => `<li>${r}</li>`).join("")}</ul>
      <div style="margin-top:8px;"><strong>Suggested response:</strong> ${c.response_hint}</div>
    </div>
    <div class="shap-bars"><strong>Model explanation, SHAP</strong>${shapHtml}</div>
    <div class="actions-row">
      <button class="action-btn confirm" onclick="openEscalateModal('${c.slice_id}', '${c.window_start_ts}')">Escalate</button>
      <button class="action-btn watchlist" onclick="takeAction('${c.slice_id}', '${c.window_start_ts}', 'Watchlist')">Dismiss and Watch</button>
      <button class="action-btn dismiss" onclick="takeAction('${c.slice_id}', '${c.window_start_ts}', 'Dismiss')">Dismiss</button>
    </div>
  `;
}

// ---------------- escalate modal ----------------
function openEscalateModal(slice_id, window_start_ts) {
  escalateTarget = { slice_id, window_start_ts };
  const c = pendingQueue[rowKey(slice_id, window_start_ts)];
  document.getElementById("escalateContext").innerHTML =
    `${slice_id} &middot; ${window_start_ts} &middot; ${c.dominant_signal.replace(/_/g," ").toLowerCase()} &middot; ${(c.calibrated_prob*100).toFixed(1)}% probability`;
  document.getElementById("escalateTo").value = "";
  document.getElementById("escalateNote").value = "";
  document.getElementById("escalateModal").style.display = "flex";
}

document.getElementById("escalateCancel").addEventListener("click", () => {
  document.getElementById("escalateModal").style.display = "none";
  escalateTarget = null;
});

document.getElementById("escalateSubmit").addEventListener("click", async () => {
  if (!escalateTarget) return;
  const escalated_to = document.getElementById("escalateTo").value.trim();
  const note = document.getElementById("escalateNote").value.trim();
  document.getElementById("escalateModal").style.display = "none";
  await submitAction(escalateTarget.slice_id, escalateTarget.window_start_ts, "Escalate", { escalated_to, note });
  escalateTarget = null;
});

async function takeAction(slice_id, window_start_ts, action) {
  await submitAction(slice_id, window_start_ts, action, {});
}

async function submitAction(slice_id, window_start_ts, action, extra) {
  await fetch("/api/case/action", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ slice_id, window_start_ts, action, ...extra }),
  });
  const key = rowKey(slice_id, window_start_ts);
  actionedKeys.add(key);
  delete pendingQueue[key];
  if (activeQueueKey === key) activeQueueKey = null;
  renderQueue();
  await refreshAuditTrail();

  const remaining = Object.keys(pendingQueue);
  if (remaining.length > 0) {
    openCase(remaining[0]);
  } else {
    document.getElementById("caseDetail").innerHTML = `<p class="hint">Logged: ${action}. No cases waiting for review.</p>`;
    document.getElementById("caseDetail").classList.add("empty");
  }
}

async function refreshAuditTrail() {
  const rows = await (await fetch("/api/audit-trail")).json();
  const body = document.getElementById("auditBody");
  body.innerHTML = rows.map(r => `
    <tr>
      <td>${r.logged_at}</td><td>${r.slice_id}</td><td>${r.window_start_ts}</td>
      <td><span class="badge badge-${r.system_action}">${r.system_action.replace("_"," ")}</span></td>
      <td>${r.dominant_signal}</td><td>${(r.calibrated_prob*100).toFixed(1)}%</td>
      <td>${inr(r.estimated_exposure_inr)}</td><td><strong>${r.analyst_action}</strong></td>
      <td>${r.note || ""}</td>
    </tr>`).join("");
  document.getElementById("auditEmpty").style.display = rows.length ? "none" : "block";
}

// ---------------- model tab ----------------
async function initTimeline() {
  const res = await fetch("/api/festivals");
  festivalData = await res.json();
  renderTimeline(festivalData);
}

function renderTimeline(data) {
  const start = new Date(data.overall_start);
  const end = new Date(data.overall_end);
  const totalMs = end - start;

  const monthsEl = document.getElementById("timeline-months");
  monthsEl.innerHTML = "";
  let cursor = new Date(start.getFullYear(), start.getMonth(), 1);
  const monthCursors = [];
  while (cursor < end) { monthCursors.push(new Date(cursor)); cursor = new Date(cursor.getFullYear(), cursor.getMonth() + 1, 1); }
  monthCursors.forEach((m, i) => {
    const pctPos = ((m - start) / totalMs) * 100;
    const label = document.createElement("span");
    if (i === 0) { label.style.left = "0%"; label.style.transform = "translateX(0)"; }
    else if (i === monthCursors.length - 1) { label.style.left = "100%"; label.style.transform = "translateX(-100%)"; }
    else { label.style.left = pctPos + "%"; label.style.transform = "translateX(-50%)"; }
    label.textContent = `${MONTHS[m.getMonth()]} ${m.getFullYear()}`;
    monthsEl.appendChild(label);
  });

  const markersEl = document.getElementById("timeline-markers");
  markersEl.innerHTML = "";
  data.festivals.forEach((f, i) => {
    const fStart = new Date(f.start_ts), fEnd = new Date(f.end_ts);
    const mid = new Date((fStart.getTime() + fEnd.getTime()) / 2);
    const pctPos = ((mid - start) / totalMs) * 100;

    const marker = document.createElement("div");
    marker.className = "marker " + (f.is_test_festival ? "heldout" : "train") + (i % 2 === 1 ? " row2" : "");
    marker.style.left = pctPos + "%";
    marker.title = f.name + (f.is_test_festival ? " (held out)" : " (trained on)");
    marker.innerHTML = `<div class="dot-el"></div><div class="m-label">${f.name.replace(" Sale", "").replace(" Mega", "")}</div>`;
    marker.addEventListener("click", () => selectFestival(f.name, marker));
    markersEl.appendChild(marker);
  });
}

async function selectFestival(name, markerEl) {
  document.querySelectorAll(".marker").forEach(m => m.classList.remove("selected"));
  markerEl.classList.add("selected");
  selectedFestivalName = name;

  const section = document.getElementById("tiers-section");
  section.style.display = "block";
  document.getElementById("tiers-heading").textContent = `Tier comparison, ${name}`;
  document.getElementById("tier-grid").innerHTML = "";
  document.getElementById("chart-card").style.display = "none";
  document.getElementById("tier-loading").style.display = "block";

  const [res] = await Promise.all([
    fetch(`/api/festival/${encodeURIComponent(name)}/metrics`),
    sleep(2200),
  ]);
  document.getElementById("tier-loading").style.display = "none";
  if (!res.ok) { console.error("metrics fetch failed", await res.text()); return; }
  const data = await res.json();
  renderTierCards(data);
  renderCostChart(data);
  document.getElementById("chart-card").style.display = "block";
}

const TIER_LABELS = { 0: "Tier 0, naive", 1: "Tier 1, rule based", 2: "Tier 2, full" };
const TIER_SUB = {
  0: "Raw volume only",
  1: "Fixed festival multiplier, no behavioral signal",
  2: "Calendar-adjusted plus UPI behavioral features, calibrated",
};

function renderTierCards(data) {
  const grid = document.getElementById("tier-grid");
  grid.innerHTML = "";
  [0, 1, 2].forEach(t => {
    const m = data.tiers[t];
    const card = document.createElement("div");
    card.className = "tier-card" + (t === 2 ? " featured" : "");
    card.innerHTML = `
      <h4>${TIER_LABELS[t]}</h4>
      <div class="tier-sub">${TIER_SUB[t]}</div>
      <div class="tier-metric"><span>Precision</span><span class="val">${pct(m.precision)}</span></div>
      <div class="tier-metric"><span>Recall</span><span class="val">${pct(m.recall)}</span></div>
      <div class="tier-metric"><span>False positive rate</span><span class="val">${pct(m.fpr)}</span></div>
      <div class="tier-metric"><span>Festive surge FPR</span><span class="val">${pct(m.festive_fpr)}</span></div>
      <div class="tier-metric"><span>Total cost, this range</span><span class="val">${inr(m.total_cost_inr)}</span></div>
    `;
    grid.appendChild(card);
  });
}

function renderCostChart(data) {
  const ctx = document.getElementById("costChart");
  const values = [data.no_detection_floor_cost_inr, data.tiers[0].total_cost_inr, data.tiers[1].total_cost_inr, data.tiers[2].total_cost_inr];
  if (costChart) costChart.destroy();
  costChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["No detection", "Tier 0", "Tier 1", "Tier 2"],
      datasets: [{ data: values, backgroundColor: ["#8A97A3", "#003087", "#C98A1E", "#0070BA"], borderRadius: 6 }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { type: "logarithmic", ticks: { callback: v => "Rs " + Number(v).toLocaleString("en-IN") } } },
    },
  });
}