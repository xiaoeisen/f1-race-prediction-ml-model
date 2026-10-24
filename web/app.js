const $ = (id) => document.getElementById(id);

function revealHeading(node, text = null) {
  if (text !== null) node.textContent = text;
  const label = node.textContent.replace(/\s+/g, " ").trim();
  node.setAttribute("aria-label", label);
  node.classList.add("letter-reveal");
  let index = 0;

  function splitText(container) {
    [...container.childNodes].forEach((child) => {
      if (child.nodeType === Node.TEXT_NODE) {
        const fragment = document.createDocumentFragment();
        for (const character of child.textContent) {
          const clip = document.createElement("span");
          clip.className = "letter-clip";
          clip.setAttribute("aria-hidden", "true");
          clip.style.setProperty("--letter-index", index++);
          const letter = document.createElement("span");
          letter.className = "letter-inner";
          letter.textContent = /\s/.test(character) ? "\u00a0" : character;
          clip.append(letter);
          fragment.append(clip);
        }
        child.replaceWith(fragment);
      } else if (child.nodeType === Node.ELEMENT_NODE) {
        splitText(child);
      }
    });
  }

  splitText(node);
}

const TEAM_INFO = window.F1_TEAM_INFO || {};
function normalizedKey(value) {
  return String(value || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]/g, "");
}
function getTeamInfo(name) {
  const label=String(name||"Unknown team");
  return TEAM_INFO[normalizedKey(label)]||{name:label,code:label.slice(0,3).toUpperCase(),color:"#8A8D91"};
}
function getDriverInfo(name,code) {
  return window.F1_DRIVER_INFO?.byName?.[normalizedKey(name)]||window.F1_DRIVER_INFO?.byCode?.[String(code||"").toUpperCase()]||{};
}
function driverVisual(name,code) {
  const info=getDriverInfo(name,code),team=getTeamInfo(info.team),mark=el("span",{className:"driver-visual"});
  mark.style.cssText="display:inline-flex;align-items:center;gap:7px;flex:none;padding:5px 8px;border:1px solid color-mix(in srgb,"+team.color+" 45%,transparent);border-left:3px solid "+team.color+";border-radius:8px;background:color-mix(in srgb,"+team.color+" 11%,#151619);white-space:nowrap";
  mark.style.setProperty("--team-color",team.color);
  if(info.flag)mark.append(el("span",{className:"driver-visual-flag",textContent:info.flag,"aria-hidden":"true"}));
  mark.append(el("strong",{className:"driver-visual-number",textContent:info.number?"#"+info.number:"?"}));
  const codeNode=el("span",{className:"driver-visual-code",textContent:String(code||info.code||"DRV").toUpperCase()});
  codeNode.style.cssText="font:700 10px/1 monospace;letter-spacing:.1em;color:"+team.color;
  mark.append(codeNode);return mark;
}
function teamVisual(name) {
  const info=getTeamInfo(name),chip=el("span",{className:"team-visual"});
  chip.style.cssText="display:inline-flex;align-items:center;gap:8px;padding:5px 9px;border:1px solid color-mix(in srgb,"+info.color+" 40%,transparent);border-radius:8px;background:color-mix(in srgb,"+info.color+" 9%,#151619);white-space:nowrap";
  chip.style.setProperty("--team-color",info.color);
  const code=el("strong",{className:"team-visual-code",textContent:info.code});
  code.style.cssText="font:700 10px/1 monospace;letter-spacing:.08em;color:"+info.color;
  chip.append(code,el("span",{textContent:info.name}));return chip;
}
const METRIC_LABELS = {
  spearman: ["Rank correlation", (v) => v.toFixed(2), true],
  winner: ["Winner correct", (v) => (v * 100).toFixed(0) + "%", true],
  podium: ["Podium overlap", (v) => (v * 100).toFixed(0) + "%", true],
  top10: ["Top-10 overlap", (v) => (v * 100).toFixed(0) + "%", true],
  ndcg5: ["Top-5 ranking quality (NDCG@5)", (v) => v.toFixed(3), true],
  mae: ["Avg. position error", (v) => v.toFixed(1), false],
  mae_fin: ["Avg. position error, finishers only", (v) => v.toFixed(1), false],
};


function animateCountUp(node) {
  if (node.dataset.countUpStarted) return;
  const match = node.textContent.match(/^(.*?)(-?\d+(?:\.\d+)?)(.*)$/);
  if (!match) return;
  const [, prefix, rawValue, suffix] = match;
  const target = Number(rawValue);
  if (!Number.isFinite(target)) return;

  node.dataset.countUpStarted = "true";
  node.classList.add("is-counting");
  const decimals = (rawValue.split(".")[1] || "").length;
  const started = performance.now();
  const duration = 1100;

  function frame(now) {
    const progress = Math.min(1, (now - started) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    node.textContent = prefix + (decimals ? value.toFixed(decimals) : Math.round(value)) + suffix;
    if (progress < 1) requestAnimationFrame(frame);
    else node.classList.remove("is-counting");
  }
  requestAnimationFrame(frame);
}

const countUpObserver = "IntersectionObserver" in window
  ? new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        observer.unobserve(entry.target);
        animateCountUp(entry.target);
      });
    }, { threshold: 0.2 })
  : null;

function observeCountUps(root) {
  const nodes = root.querySelectorAll(".card .big, .badge .pct, #model td:not(:first-child), .barrow > span:last-child");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  nodes.forEach((node) => {
    if (node.dataset.countUpObserved || !/\d/.test(node.textContent)) return;
    node.dataset.countUpObserved = "true";
    node.classList.add("count-up");
    if (reduceMotion || !countUpObserver) {
      if (!reduceMotion) animateCountUp(node);
      return;
    }
    countUpObserver.observe(node);
  });
}

async function api(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : "Something went wrong.");
  }
  return res.json();
}

function el(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  kids.forEach((k) => node.append(k));
  return node;
}

function setStatus(msg, isError = false, isLoading = false) {
  const s = $("status");
  s.replaceChildren();
  s.className = [isError ? "error" : "", isLoading ? "loading" : ""].filter(Boolean).join(" ");
  if (!msg) return;

  s.append(el("span", { textContent: msg }));
  if (isLoading) {
    const loader = el("span", { className: "loader" });
    loader.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i++) loader.append(el("span", { className: "bar" }));
    s.append(loader);
  }
}

/* ---------------------------------------------------------------- setup */
async function init() {
  try {
    const [seasons, model] = await Promise.all([api("/api/seasons"), api("/api/model")]);
    renderModel(model);
    const sel = $("season");
    seasons.seasons.forEach((y) => sel.append(el("option", { value: y, textContent: y })));
    sel.value = seasons.seasons[seasons.seasons.length - 1];
    await Promise.all([loadRaces(), loadStandings()]);
  } catch (e) {
    setStatus(e.message, true);
  }
}

async function loadRaces() {
  setStatus("Loading race calendar...");
  try {
    const data = await api("/api/races?year=" + $("season").value);
    const sel = $("race");
    sel.replaceChildren();
    data.races.forEach((r) => {
      const label = `R${r.round} \u00b7 ${r.name}${r.completed ? " (completed)" : " (upcoming)"}`;
      sel.append(el("option", { value: r.round, textContent: label }));
    });
    const next = [...data.races].reverse().find((r) => r.completed) || data.races[data.races.length - 1];
    if (next) sel.value = next.round;
    setStatus("");
  } catch (e) {
    setStatus(e.message, true);
  }
}

async function loadStandings() {
  const panel = $("standings-panel");
  try {
    const data = await api("/api/standings?year=" + $("season").value);
    $("standings-through").textContent = data.through_round
      ? "Points through Round " + data.through_round + " of " + data.year + "."
      : "No completed race results are available for " + data.year + " yet.";
    renderStandingsTable($("driver-standings"), "driver", data.drivers);
    renderStandingsTable($("team-standings"), "team", data.teams);
    panel.hidden = false;
  } catch (error) {
    $("standings-through").textContent = error.message;
    panel.hidden = false;
  }
}

function renderStandingsTable(container,kind,rows) {
  container.replaceChildren(); if(!rows.length)return;
  const isDriver=kind==="driver",table=el("table",{className:"standings-table"});
  const headings=isDriver?["Pos.","Driver","Team","Pts","Wins"]:["Pos.","Team","Pts","Wins"];
  table.append(el("thead",{},el("tr",{},...headings.map(text=>el("th",{textContent:text})))));
  const body=el("tbody");
  rows.forEach(row=>{
    const tr=el("tr");tr.append(el("td",{className:"standing-position",textContent:row.position}));
    if(isDriver){
      const driver=el("td",{},el("span",{className:"standing-driver"})),content=driver.firstChild;
      content.append(driverVisual(row.full_name,row.code),el("span",{textContent:row.full_name}));
      const team=getTeamInfo(row.team),teamCell=el("td",{className:"team",textContent:team.name});
      teamCell.style.color=team.color;tr.append(driver,teamCell);
    }else{const teamCell=el("td");teamCell.append(teamVisual(row.team));tr.append(teamCell);}
    tr.append(el("td",{className:"standing-number",textContent:row.points}),el("td",{className:"standing-number",textContent:row.wins}));
    body.append(tr);
  });
  table.append(body);container.append(table);
}

/* ---------------------------------------------------------------- predict */
async function predict() {
  const btn = $("predict");
  btn.disabled = true;
  setStatus("Calculating prediction. The first live fetch can take up to a minute.", false, true);

  const body = { year: +$("season").value, round: +$("race").value, grid: null, rain: null };
  const weather = $("weather").value;
  if (weather === "rain") body.rain = true;
  if (weather === "dry") body.rain = false;

  try {
    const data = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    render(data);
    setStatus("");
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    btn.disabled = false;
  }
}

function resetWhatIf() {
  $("weather").value = "auto";
  predict();
}

/* ---------------------------------------------------------------- render */
function note(kind, text) {
  return el("div", { className: "note " + kind, textContent: text });
}

function card(label, big, small, smallClass) {
  const s = el("div", { className: "small" + (smallClass ? " " + smallClass : ""), textContent: small || "" });
  return el("div", { className: "card" },
    el("div", { className: "label", textContent: label }),
    el("div", { className: "big", textContent: big }), s);
}

function render(data) {
  const { meta, rows } = data;

  const notes = $("notes");
  notes.replaceChildren();
  if (meta.in_sample) {
    notes.append(note("warn", "The model trained on this race, so this is not a fair test. Pick a more recent race or an upcoming one for an honest result."));
  } else if (meta.completed) {
    notes.append(note("info", "Honest backtest: the model had not seen this race when it made this prediction."));
  }
  if (!meta.completed && !meta.quali_available) {
    notes.append(note("warn", "Qualifying has not happened yet, so the starting grid is unknown and this prediction is less reliable."));
  }
  const wi = meta.what_if;
  if (wi.grid || wi.rain !== null) {
    const parts = [];
    if (wi.grid) parts.push("grid edited");
    if (wi.rain !== null) parts.push(wi.rain ? "forced rain" : "forced dry");
    notes.append(note("info", "What-if mode: " + parts.join(", ") + "."));
  }
  if (meta.forecast && wi.rain === null) {
    notes.append(note("info", `Race-day forecast used: ${Math.round(meta.forecast.rain_prob * 100)}% chance of rain, high of ${Math.round(meta.forecast.air_temp)}\u00b0C.`));
  }
  if (meta.grid_source === "openf1" || meta.grid_source === "apisports") {
    notes.append(note("info", `Starting grid: the official grid from ${meta.grid_source === "openf1" ? "OpenF1" : "API-Sports"}, so penalties are already included.`));
  }
  if (meta.penalties_applied) {
    notes.append(note("info", `${meta.penalties_applied} grid penalt${meta.penalties_applied === 1 ? "y" : "ies"} applied from penalties.csv.`));
  }
  $("reset").hidden = !(wi.grid || wi.rain !== null);

  const name = (r) => r.full_name || r.driver;
  const cards = $("summary");
  cards.replaceChildren();
  const w = rows[0];
  cards.append(card("Predicted winner", name(w), `${w.team} \u00b7 ${w["win_%"]}% chance`));
  cards.append(card("Predicted podium", rows.slice(0, 3).map((r) => r.driver).join("  \u00b7  "),
    "Most likely top three"));

  const hasActual = meta.completed && rows.some((r) => r.finish !== null && r.finish !== undefined);
  if (hasActual) {
    const actual = rows.filter((r) => r.finish).sort((a, b) => a.finish - b.finish);
    const aw = actual[0];
    const hits = rows.slice(0, 3).filter((r) => r.finish && r.finish <= 3).length;
    cards.append(card("Actual winner", name(aw), aw.driver === w.driver ? "Predicted correctly" : "Not predicted",
      aw.driver === w.driver ? "ok" : "no"));
    cards.append(card("Podium overlap", `${hits} of 3`, "Predicted podium drivers who finished top 3"));
  }
  cards.hidden = false;
  observeCountUps(cards);

  $("evt").textContent = `${meta.event} ${meta.year}`;
  document.querySelectorAll(".actual-col").forEach((c) => (c.hidden = !hasActual));
  const tbody = document.querySelector("#table tbody");
  tbody.replaceChildren();
  rows.forEach((r) => tbody.append(tableRow(r, hasActual)));
  $("results").hidden = false;
  renderSessionStandings(meta, rows);
}

function renderSessionStandings(meta, rows) {
  const panel = $("session-standings-panel");
  const tabs = $("session-tabs");
  const container = $("session-standing-table");
  const sessions = [
    { id: "fp1", label: "FP1", position: "fp1_pos", gap: "fp1_best_gap" },
    { id: "fp2", label: "FP2", position: "fp2_pos", gap: "fp2_best_gap" },
    { id: "fp3", label: "FP3", position: "fp3_pos", gap: "fp3_best_gap" },
    { id: "quali", label: "Qualifying", position: "quali_pos", gap: "q_best_gap" },
    { id: "grid", label: "Starting Grid", position: "grid", gap: null },
  ];
  tabs.replaceChildren();
  const initial = sessions.find(session => rows.some(row =>
    row[session.position] !== null && row[session.position] !== undefined && row[session.position] !== "")) || sessions[0];

  function showSession(selected) {
    tabs.querySelectorAll(".session-tab").forEach(button => {
      const active = button.dataset.session === selected.id;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    const results = rows.filter(row => {
      const position = row[selected.position];
      return position !== null && position !== undefined && position !== "" && Number.isFinite(Number(position));
    }).sort((a, b) => Number(a[selected.position]) - Number(b[selected.position]));
    container.replaceChildren();
    if (!results.length) {
      container.append(note("info", selected.label + " results are not available yet for this Grand Prix."));
      return;
    }

    const table = el("table", { className: "session-standing-table" });
    const header = el("tr");
    ["Pos.", "Driver", "Team"].concat(selected.gap ? ["Gap to fastest"] : []).forEach(label =>
      header.append(el("th", { textContent: label })));
    table.append(el("thead", {}, header));
    const body = el("tbody");
    results.forEach(row => {
      const tr = el("tr");
      tr.append(el("td", { className: "standing-position", textContent: "P" + Math.round(row[selected.position]) }));
      const driverCell = el("td");
      const driver=el("span",{className:"standing-driver"});
      driver.append(driverVisual(row.full_name||row.driver,row.driver),
        el("span",{textContent:row.full_name||row.driver}));
      driverCell.append(driver);
      const team = getTeamInfo(row.team);
      const teamCell = el("td");
      teamCell.append(teamVisual(row.team));
      teamCell.style.color = team.color;
      tr.append(driverCell, teamCell);
      if (selected.gap) {
        const gap = row[selected.gap];
        tr.append(el("td", { className: "standing-number", textContent:
          gap === null || gap === undefined || !Number.isFinite(Number(gap)) ? "—" : "+" + Number(gap).toFixed(2) + "%" }));
      }
      body.append(tr);
    });
    table.append(body);
    container.append(table);
  }

  sessions.forEach(session => {
    const button = el("button", { type: "button", className: "session-tab", role: "tab",
      textContent: session.label });
    button.dataset.session = session.id;
    button.setAttribute("aria-controls", "session-standing-table");
    button.addEventListener("click", () => showSession(session));
    tabs.append(button);
  });
  showSession(initial);
  panel.hidden = false;
}
function bar(value, strong) {
  const v = value === null || value === undefined ? 0 : value;
  const progress = Math.max(0, Math.min(100, v));
  const ring = el("div", {
    className: "prob-ring" + (strong ? " strong" : ""),
    style: "--progress:" + progress + "%",
    title: v.toFixed(1) + "% chance",
  }, el("span", { className: "val", textContent: v.toFixed(1) + "%" }));
  ring.setAttribute("role", "img");
  ring.setAttribute("aria-label", v.toFixed(1) + "% chance");
  return el("td", {}, ring);
}
function sessionCell(pos, gap) {
  const td = el("td");
  if (pos === null || pos === undefined) {
    td.textContent = "\u2013";
    td.title = "No data for this session";
    return td;
  }
  td.append(document.createTextNode("P" + Math.round(pos)));
  if (gap !== null && gap !== undefined && gap > 0) {
    td.append(el("span", { className: "fullname", textContent: "+" + gap.toFixed(2) + "%" }));
  }
  return td;
}

function tableRow(r,hasActual) {
  const tr=el("tr");tr.append(el("td",{className:"pos",textContent:"P"+r.pred_pos}));
  const driver=el("td",{className:"driver-cell"});
  driver.append(driverVisual(r.full_name,r.driver),el("span",{className:"fullname",textContent:r.full_name&&r.full_name!==r.driver?r.full_name:""}));
  tr.append(driver);
  const team=getTeamInfo(r.team),teamCell=el("td",{className:"team"});
  teamCell.append(teamVisual(r.team));teamCell.style.color=team.color;tr.append(teamCell);
  tr.append(sessionCell(r.grid,null));tr.append(sessionCell(r.quali_pos,r.q_best_gap));
  tr.append(sessionCell(r.fp1_pos,r.fp1_best_gap));tr.append(sessionCell(r.fp2_pos,r.fp2_best_gap));
  tr.append(el("td",{textContent:r.exp_pos.toFixed(1)}));tr.append(bar(r["win_%"],true));
  tr.append(bar(r["podium_%"],false));tr.append(bar(r["top10_%"],false));
  if(hasActual){
    const td=el("td",{className:"actual-col"});
    if(r.finish){
      td.append(document.createTextNode("P"+Math.round(r.finish)+" "));
      const diff=Math.round(r.pred_pos-r.finish);
      td.title=diff===0?"Prediction matched the actual finishing position":diff>0?
        "Finished "+diff+" place"+(diff===1?"":"s")+" ahead of the prediction":
        "Finished "+Math.abs(diff)+" place"+(diff===-1?"":"s")+" behind the prediction";
      if(diff!==0)td.append(el("span",{className:diff>0?"up":"down",textContent:(diff>0?"\u25b2":"\u25bc")+Math.abs(Math.round(diff))}));
    }
    tr.append(td);
  }
  return tr;
}

function renderModel(m) {
  const box = $("model");
  box.replaceChildren();
  if (!m.available) {
    box.append(el("p", { className: "hint", textContent: "No evaluation yet. Run python train.py to see how the model compares with simply guessing that drivers finish where they qualified." }));
    return;
  }
  box.append(el("p", { className: "hint",
    textContent: `Tested on the latest ${m.test_races} races${m.test_from ? ` (from ${m.test_from[0]} round ${m.test_from[1]})` : ` of ${m.test_year}`} that it never saw during training (trained on ${m.train_races} earlier races). Green is the best score in each row.` }));

  const names = Object.keys(m.results);
  const labels = { "grid order (baseline)": "Guess: finish = grid", "driver+team Elo": "Driver + team ratings only", "LightGBM ranker": "This model" };
  const table = el("table");
  const head = el("tr", {}, el("th", { textContent: "Measure" }));
  names.forEach((n) => head.append(el("th", { textContent: labels[n] || n })));
  table.append(el("thead", {}, head));
  const body = el("tbody");
  Object.entries(METRIC_LABELS).forEach(([key, [label, fmt, higherBetter]]) => {
    const vals = names.map((n) => m.results[n][key]).filter((v) => v !== undefined);
    if (!vals.length) return;
    const best = higherBetter ? Math.max(...vals) : Math.min(...vals);
    const row = el("tr", {}, el("td", { textContent: label }));
    vals.forEach((v) => row.append(el("td", { className: v === best ? "best" : "", textContent: fmt(v) })));
    body.append(row);
  });
  table.append(body);
  box.append(el("div", { className: "scroll" }, table));
  observeCountUps(box);
}

/* ---------------------------------------------------------------- events */
$("season").addEventListener("change", () => { loadRaces(); loadStandings(); });
$("predict").addEventListener("click", predict);
$("reset").addEventListener("click", resetWhatIf);
init();


/* ================================================================ Race Center (live) */
const STAGES = [["pre", "After practice"], ["quali", "After qualifying"]];
let liveRace = null;

const fmtTime = (s) => {
  if (!s) return "never";
  const d = new Date(/Z$|\+/.test(s) ? s : s.replace(" ", "T") + "Z");
  return isNaN(d) ? s : d.toLocaleString();
};

async function loadLive() {
  const pick = $("live-race");
  if (!liveRace) {
    pick.replaceChildren(el("option", { value: "", textContent: "Loading race calendar..." }));
    pick.disabled = true;
  }
  const q = liveRace ? "?year=" + liveRace.year + "&round=" + liveRace.round : "";
  try {
    renderLive(await api("/api/live" + q));
  } catch (e) {
    revealHeading($("live-title"), "Race Center unavailable");
    $("live-title").dataset.wipe = $("live-title").textContent;
    $("live-msg").textContent = e.message;
    pick.replaceChildren(el("option", { value: "", textContent: "Race calendar unavailable" }));
    pick.disabled = true;
  }
}
function pctClass(p) { return p >= 60 ? "high" : p >= 40 ? "mid" : "low"; }

function renderLive(d) {
  if (d.error) {
    revealHeading($("live-title"), "Race Center unavailable");
    $("live-title").dataset.wipe = $("live-title").textContent;
    $("live-msg").textContent = d.error;
    const pick = $("live-race");
    pick.replaceChildren(el("option", { value: "", textContent: "Race calendar unavailable" }));
    pick.disabled = true;
    return;
  }
  const ev = d.event;
  revealHeading($("live-title"), `${ev.name} ${ev.year}`);
  $("live-title").dataset.wipe = $("live-title").textContent;
  const statusText = {
    upcoming: "Upcoming. The first prediction appears automatically once practice data is published.",
    weekend: "Race weekend in progress. Predictions update by themselves as practice and qualifying data arrive.",
    race_started: "Race in progress. Predictions are locked; the real result and accuracy appear when it finishes.",
    finished: "Finished. Real result, accuracy and self-training are below.",
  }[ev.status] || "";
  $("live-sub").textContent = `Race start ${new Date(ev.race_start).toLocaleString()}. ${statusText}`;

  const pick = $("live-race");
  const want = `${ev.year}-${ev.round}`;
  pick.replaceChildren(...d.races.map((r) => el("option", { value: `${r.year}-${r.round}`, textContent: `${r.year} \u00b7 ${r.name}` })));
  pick.disabled = false;
  pick.value = want;

  $("live-sessions").replaceChildren(...ev.sessions.map((s) => el("span", {
    className: "chip" + (s.done ? " done" : ""), title: new Date(s.start).toLocaleString(),
    textContent: (s.done ? "\u2713 " : "") + s.label })));

  // --- predicted top 5 next to the real result
  const stages = STAGES.filter(([k]) => d.predictions[k]);
  const actual = d.actual;
  const panel = $("live-compare-panel");
  $("live-msg").textContent = "";
  if (!stages.length && !actual) {
    panel.hidden = true;
    $("live-msg").textContent = "No prediction stored for this race yet.";
  } else {
    panel.hidden = false;
    $("vs-actual").hidden = !actual;
    const head = el("tr", {}, el("th", { textContent: "Pos." }));
    stages.forEach(([k, label]) => {
      const p = d.predictions[k];
      head.append(el("th", { textContent: label, title: "Stored " + fmtTime(p.created) }));
    });
    if (actual) head.append(el("th", { textContent: "Actual result" }));
    $("live-table").querySelector("thead").replaceChildren(head);

    const actualTop = actual ? actual.top5.map((t) => t.driver) : [];
    const rows = [];
    for (let i = 0; i < 5; i++) {
      const tr = el("tr", {}, el("td", { className: "pos", textContent: "P" + (i + 1) }));
      stages.forEach(([k]) => {
        const t = d.predictions[k].top5[i];
        const td = el("td", { className: "cell" });
        if (t) {
          td.append(el("span", { className: "code", textContent: t.driver }));
          if (actual) {
            const exact = actualTop[i] === t.driver, near = actualTop.includes(t.driver);
            td.append(el("span", { className: "mark " + (exact ? "ok" : near ? "near" : "no"),
              textContent: exact ? "\u2713" : near ? "\u25cb" : "\u2717",
              title: exact ? "Right position" : near ? "In the top 5, wrong position" : "Not in the top 5" }));
          }
          td.append(el("span", { className: "sub2", textContent: `${t.name} \u00b7 win ${t.win}% \u00b7 podium ${t.podium}%` }));
        }
        tr.append(td);
      });
      if (actual) {
        const t = actual.top5[i];
        const td = el("td", { className: "cell" });
        if (t) {
          td.append(el("span", { className: "code", textContent: t.driver }),
            el("span", { className: "sub2", textContent: `${t.name} \u00b7 ${t.team}` }));
        }
        tr.append(td);
      }
      rows.push(tr);
    }
    $("live-table").querySelector("tbody").replaceChildren(...rows);

    $("live-badges").replaceChildren(...stages.filter(([k]) => d.predictions[k].metrics).map(([k, label]) => {
      const m = d.predictions[k].metrics;
      const b = el("div", { className: "badge " + pctClass(m.top5_pct) });
      b.append(el("div", { className: "stage", textContent: label }),
        el("div", { className: "pct", textContent: `${Math.round(m.top5_pct)}% top-5` }),
        el("div", { className: "sub2", textContent: `${m.top5_hits}/5 named \u00b7 exact ${m.exact}/5 \u00b7 winner ${m.winner ? "\u2713" : "\u2717"} \u00b7 score ${m.score}/100` }),
        el("div", { className: "sub2", textContent: m.baseline_top5_hits === null ? "" : `"Finish where you started" guess: ${m.baseline_top5_hits}/5` }));
      return b;
    }));
  }

  observeCountUps($("live-badges"));
  renderSeason(d.season);
  renderAuto(d.autopilot);
}

function renderSeason(s) {
  $("season-panel").hidden = !s || !s.n_races;
  if (!s || !s.n_races) return;
  const card = (label, big, small) => el("div", { className: "card" },
    el("div", { className: "label", textContent: label }), el("div", { className: "big", textContent: big }),
    el("div", { className: "small", textContent: small || "" }));
  $("season-cards").replaceChildren(
    card("Races scored", String(s.n_races), "predicted before the race, scored after"),
    card("Avg top-5 accuracy", s.avg_top5_pct + "%", s.baseline_top5_pct === null ? "" : `grid-order guess: ${s.baseline_top5_pct}%`),
    card("Last 10 races", s.last10_top5_pct + "%", "top-5 accuracy"),
    card("Winner hit rate", s.winner_pct + "%", `avg score ${s.avg_score}/100`));
  const bars = el("div", { className: "bars" });
  s.series.slice(-12).forEach((r) => {
    const track = el("div", { className: "track" }, el("div", { className: "fill", style: `width:${r.top5_pct}%` }));
    if (r.baseline_top5_pct !== null) track.append(el("div", { className: "tick", style: `left:${r.baseline_top5_pct}%`, title: "grid-order guess" }));
    bars.append(el("div", { className: "barrow" }, el("span", { textContent: `${r.year} R${r.round} ${r.event.replace(" Grand Prix", " GP")}` }),
      track, el("span", { textContent: `${Math.round(r.top5_pct)}%` })));
  });
  $("season-bars").replaceChildren(bars);
  observeCountUps($("season-panel"));
}

function renderAuto(a) {
  const body = $("auto-body");
  const dl = el("dl", { className: "kv" });
  const add = (k, v) => dl.append(el("dt", { textContent: k }), el("dd", { textContent: v }));
  add("Autopilot", a.enabled ? `on, checks every ${a.interval_min} min` : "off (set AUTOPILOT=1)");
  add("Last check", fmtTime(a.last_tick));
  if (a.last_error) add("Last problem", a.last_error);
  if (a.model) add("Model", `v${a.model.version}, trained ${fmtTime(a.model.trained_at)}, through ${a.model.trained_through.join(" R")} (${a.model.races} races)`);
  if (a.last_retrain) add("Last self-training", `${a.last_retrain.event}: ${a.last_retrain.gate}`);
  body.replaceChildren(dl);
  if (a.last_actions && a.last_actions.length) {
    body.append(el("p", { className: "hint", textContent: "Latest actions:" }),
      el("ul", { className: "log" }, ...a.last_actions.map((t) => el("li", { textContent: t }))));
  }
  if (a.training_log && a.training_log.length) {
    body.append(el("p", { className: "hint", textContent: "Training history:" }),
      el("ul", { className: "log" }, ...a.training_log.slice().reverse().map((t) => el("li", {
        textContent: `v${t.version} \u00b7 ${t.reason} \u00b7 ${t.promoted ? "promoted" : "rejected"} \u00b7 ${t.gate}` + (t.races_upweighted ? ` \u00b7 ${t.races_upweighted} missed race(s) weighted up` : "") }))));
  }
}

function clamp01(value) { return Math.max(0, Math.min(1, value)); }

function updateViewFromScroll() {
  const explore = $("explore");
  const exploreIsActive = explore.getBoundingClientRect().top <= window.innerHeight * 0.42;
  document.body.classList.toggle("view-live", !exploreIsActive);
  document.body.classList.toggle("view-explore", exploreIsActive);

  const stage = document.querySelector(".hero-stage");
  const stageRect = stage.getBoundingClientRect();
  const heroProgress = clamp01(-stageRect.top / Math.max(1, stageRect.height - window.innerHeight));
  document.documentElement.style.setProperty("--hero-shift", (-42 * heroProgress) + "px");
  document.documentElement.style.setProperty("--hero-opacity", 1 - heroProgress * .38);

  [["live-title", $("live")], ["explore-title", explore]].forEach(([id, section]) => {
    const title = $(id);
    const rect = section.getBoundingClientRect();
    const progress = clamp01((window.innerHeight * .88 - rect.top) / Math.max(1, window.innerHeight * .62));
    title.style.setProperty("--wipe", (progress * 100) + "%");
    title.dataset.wipe = title.textContent;
  });
}

let viewUpdateQueued = false;
window.addEventListener("scroll", () => {
  if (viewUpdateQueued) return;
  viewUpdateQueued = true;
  requestAnimationFrame(() => {
    viewUpdateQueued = false;
    updateViewFromScroll();
  });
}, { passive: true });
window.addEventListener("resize", updateViewFromScroll, { passive: true });
document.querySelectorAll(".hero-title, #live-title, #drivers-title, .explore-title, .panel h2, .panel h3").forEach((node) => revealHeading(node));
if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver((entries, observer) => entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("is-revealed");
      if (entry.target.classList.contains("letter-reveal")) entry.target.classList.add("letters-visible");
      observer.unobserve(entry.target);
    }
  }), { threshold: .12 });
  document.querySelectorAll(".panel, .section-kicker, .explore-title, .hero-title, #live-title, #drivers-title, .panel h2, .panel h3, footer").forEach((node, i) => {
    node.classList.add("scroll-reveal");
    node.style.setProperty("--reveal-delay", Math.min(i % 5, 4) * 55 + "ms");
    revealObserver.observe(node);
  });
}
$("live-race").addEventListener("change", (e) => {
  const [year, round] = e.target.value.split("-").map(Number);
  liveRace = { year, round };
  loadLive();
});
setInterval(loadLive, 60000);
setInterval(loadStandings, 5 * 60 * 1000);
loadLive();
try { updateViewFromScroll(); } catch (e) { console.warn("Scroll effect initialization failed", e); }
