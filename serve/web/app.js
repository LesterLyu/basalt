"use strict";
const $ = s => document.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const num = v => typeof v === "number" && Number.isFinite(v);
const fmt = (v, d = 0) => num(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "-";
const p2 = n => String(n).padStart(2, "0");
const clock = ms => { const d = new Date(ms); return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`; };
const gib = n => num(n) ? (n / 1024 ** 3).toFixed(1) : "-";
const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ESC[c]);
const drawn = new WeakMap();
const html = (el, h) => { if (drawn.get(el) !== h) { el.innerHTML = h; drawn.set(el, h); } };   // skip unchanged redraws
const text = (el, v) => { if (el.textContent !== v) el.textContent = v; };
const HIST = 60;
const kt = n => n >= 100000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);   // 41.2k, 262k
const lim = n => n >= 10000 ? kt(n) : fmt(n);   // token limit: 8,192, 41.2k, 262k
const tick = n => n >= 1000 ? `${+(n / 1000).toFixed(1)}k` : String(Math.round(n));
const API = { openai: "OpenAI", anthropic: "Anthropic" };
const shortGpu = s => String(s || "").replace(/^NVIDIA\s+/i, "").replace(/^GeForce\s+/i, "");
const shortCpu = s => String(s || "").replace(/\((R|TM)\)/gi, "").replace(/\s+\d+-Core Processor$/i, "")
  .replace(/\s+CPU\s+@.*$/i, "").replace(/\s+Processor$/i, "").replace(/\s+/g, " ").trim();

let apiKey = sessionStorage.getItem("basalt.key") || "";
let M = null;          // the last /metrics reading, with .slots from /slots

async function get(path) {
  const r = await fetch(path, { cache: "no-store", headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {} });
  if (r.status === 401) throw Object.assign(new Error("locked"), { locked: true });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/* ---- series and charts ---- */
// the last HIST samples, padded at the start with null (no line) so the newest sample sits at the right edge
const tail = a => { const v = (a || []).slice(-HIST).map(x => num(x) ? x : null); while (v.length < HIST) v.unshift(null); return v; };
const stats = a => {
  const v = a.filter(num);
  return v.length ? { min: Math.min(...v), max: Math.max(...v), avg: v.reduce((t, x) => t + x, 0) / v.length, last: v[v.length - 1] } : null;
};
const busyStats = a => stats(a.map(x => x > 0 ? x : null));     // over the samples that ran
const STEPS = [10, 20, 50, 100, 120, 150, 160, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 3000, 4000, 5000,
  8000, 10000, 16000, 20000, 32000, 50000];
const nice = v => STEPS.find(s => s >= v * 1.15) || Math.ceil(v * 1.15 / 10000) * 10000;
// polyline points in a 0..W x 0..100 box; a null sample breaks the line
function lines(a, y, W = 100) {
  const out = []; let cur = [];
  a.forEach((v, i) => {
    if (v == null) { if (cur.length) out.push(cur); cur = []; return; }
    cur.push(`${(i / (a.length - 1) * W).toFixed(2)},${y(v).toFixed(2)}`);
  });
  if (cur.length) out.push(cur);
  return out.map(c => c.length === 1 ? [c[0], c[0].replace(/^[\d.]+/, x => String(+x + .01))] : c);
}
const poly = (pts, attrs) => `<polyline points="${pts.join(" ")}" fill="none" vector-effect="non-scaling-stroke" stroke-linejoin="round" ${attrs}/>`;
const area = (pts, fill) => `<polygon points="${pts[0].split(",")[0]},100 ${pts.join(" ")} ${pts[pts.length - 1].split(",")[0]},100" fill="${fill}"/>`;
const baseLine = `<line x1="0" y1="100" x2="100" y2="100" stroke="var(--line)" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
let gid = 0;
// a sidebar chart: filled line between lo and hi; `limit` draws a dashed line at the top
function sbChart(a, lo, hi, accent, limit) {
  const id = `sbg${++gid}`, tint = accent ? "255,122,61" : "154,164,173";
  const segs = lines(a, v => clamp(100 - (v - lo) / (hi - lo) * 100, 1, 99));
  return `<div class="sb-ch"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="100%">`
    + `<stop offset="0" stop-color="rgba(${tint},.13)"/><stop offset="1" stop-color="rgba(${tint},.02)"/></linearGradient></defs>${baseLine}`
    + (limit ? `<line x1="0" y1="1" x2="100" y2="1" stroke="var(--basalt-500)" stroke-width="1" stroke-dasharray="2 3" vector-effect="non-scaling-stroke"/>` : "")
    + segs.map(p => area(p, `url(#${id})`) + poly(p, `stroke="${accent ? "var(--accent)" : "var(--fg-2)"}" stroke-width="1.5"`)).join("")
    + `</svg></div>`;
}
function spark(a, max) {
  const segs = lines(a.map(v => v == null ? null : v), v => 100 - Math.min(v, max) / max * 96 - 2);
  return `<div class="spark"><svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${baseLine}`
    + segs.map(p => area(p, "rgba(154,164,173,.08)") + poly(p, `stroke="var(--fg-2)" stroke-width="1.4"`)).join("") + `</svg></div>`;
}
const gy = t => t ? 100 - t * 92 - 4 : 4;             // gridline height for a fraction of the axis
const grid = fr => fr.map(t => `<line x1="0" x2="100" y1="${gy(t)}" y2="${gy(t)}" stroke="var(--line-soft)" vector-effect="non-scaling-stroke"/>`).join("");
// small tick labels at the right edge, at the top and middle gridlines, skipped where an end label sits
const ticks = (max, labels) => [[4, max], [50, max / 2]].filter(t => !labels.some(l => Math.abs(l - t[0]) < 12))
  .map(t => `<span class="tk" style="top:${t[0]}%">${tick(t[1])}</span>`).join("");

/* ---- top bar ---- */
function setState(s, label) { $("#state").dataset.s = s; text($("#state"), label); }

/* ---- sidebar ---- */
const sbRow = (k, v, u, st) => `<div class="sb-row"><span class="k">${k}</span>${st ? `<span class="st">${st}</span>` : ""}<span class="num">${v}${u ? `<span class="u">${u}</span>` : ""}</span></div>`;
const bar = f => `<div class="bar4"><i style="width:${(clamp(f || 0, 0, 1) * 100).toFixed(1)}%"></i></div>`;
const r0 = v => num(v) ? String(Math.round(v)) : "-";
function sbLoad(h, now, label = "Load") {
  const s = stats(h);
  return `<div class="sb-met grow">${sbRow(label, num(now) ? `${Math.round(now)}%` : "-", "", s ? `Min ${Math.round(s.min)} · Avg ${Math.round(s.avg)}%` : "")}${sbChart(h, 0, 100, true)}</div>`;
}
function sbPower(h, now, lim) {
  const s = stats(h), top = num(lim) && lim > 0 ? lim : Math.max(1, s ? s.max : 1);
  return `<div class="sb-met grow">${sbRow("Power", `${r0(now)}${num(lim) ? ` / ${Math.round(lim)}` : ""}`, "W", s ? `Avg ${Math.round(s.avg)} W` : "")}${sbChart(h, 0, top, false, num(lim))}</div>`;
}
function sbTemp(h, now) {
  const s = stats(h), lo = Math.min(40, s ? Math.floor(s.min) - 5 : 40), hi = Math.max(85, s ? Math.ceil(s.max) + 5 : 85);
  return `<div class="sb-met grow">${sbRow("Temp", r0(now), "°C", s ? `Min ${Math.round(s.min)} · Max ${Math.round(s.max)} °C` : "")}${sbChart(h, lo, hi, false)}</div>`;
}
const link = g => num(g.pcie_gen_max ?? g.pcie_gen) ? `Gen${g.pcie_gen_max ?? g.pcie_gen}${num(g.pcie_width) ? ` x${g.pcie_width}` : ""}` : "";
const vramRow = g => `<div class="sb-met">${sbRow("VRAM", num(g.mem_used) ? `${gib(g.mem_used)} / ${gib(g.mem_total)}` : "-", "GiB")}${bar(num(g.mem_total) ? g.mem_used / g.mem_total : 0)}</div>`;
function sbGpu(g, h, name, alone) {
  const right = alone ? `<span class="num">${esc(link(g))}</span>`
    : `<span class="num">${num(g.temp) ? `<b>${Math.round(g.temp)}</b><span class="u">°C</span>` : ""}${num(g.temp) && link(g) ? " · " : ""}${esc(link(g))}</span>`;
  return `<div class="sb-gpu"><div class="sb-gt"><span>${esc(name)}</span>${right}</div>${vramRow(g)}`
    + sbLoad(tail(h.util), g.util) + sbPower(tail(h.power), g.power, g.power_limit) + (alone ? sbTemp(tail(h.temp), g.temp) : "") + `</div>`;
}
function sbTable(G, H, names) {
  let h = `<div class="sb-tbl"><div class="h l">Load</div><div class="h">VRAM GiB</div><div class="h">Power W</div><div class="h">°C</div>`;
  G.forEach(g => {
    h += `<div class="t"><span>${esc(names(g))}</span><span class="num">${esc(link(g))}</span></div>`
      + `<div class="c l">${num(g.util) ? `${Math.round(g.util)}%` : "-"}</div>`
      + `<div class="c"><span>${gib(g.mem_used)}<span class="u">/${gib(g.mem_total)}</span></span></div>`
      + `<div class="c"><span>${r0(g.power)}<span class="u">/${r0(g.power_limit)}</span></span></div>`
      + `<div class="c">${r0(g.temp)}</div>`;
  });
  // the mean load and total power of all cards, sample by sample
  const series = k => tail(H[0] ? H[0][k] : []).map((_, i) => {
    const v = H.map(x => tail(x[k])[i]).filter(num);
    return v.length ? v.reduce((t, x) => t + x, 0) / (k === "util" ? v.length : 1) : null;
  });
  const sum = k => G.reduce((t, g) => t + (num(g[k]) ? g[k] : 0), 0);
  const load = G.filter(g => num(g.util)), mean = load.length ? load.reduce((t, g) => t + g.util, 0) / load.length : null;
  return h + `</div>`
    + `<div class="sb-agg"><div class="sb-gt"><span>All ${G.length} GPUs</span></div>`
    + sbLoad(series("util"), mean, "Mean load") + sbPower(series("power"), sum("power"), sum("power_limit") || null) + `</div>`;
}
function drawSide(hw, st, hist, eng, r, live) {
  const G = hw.gpus || [], H = Object.fromEntries((hist.gpus || []).map(x => [x.index, x]));
  const named = Object.fromEntries((st.gpus || []).map(g => [g.index, shortGpu(g.name)]));
  const title = g => `GPU ${g.index} · ${named[g.index] || "GPU"}`;
  html($("#gpus"), !G.length ? `<div class="sb-none">No GPU readings.</div>`
    : G.length >= 3 ? sbTable(G, G.map(g => H[g.index] || {}), title)
    : G.map(g => sbGpu(g, H[g.index] || {}, G.length === 1 ? named[g.index] || title(g) : title(g), G.length === 1)).join(""));

  const cpu = shortCpu(st.cpu_name);
  text($("#cpu-name"), cpu); $("#cpu-name").title = st.cpu_name || "";
  html($("#sys"), sbRow("CPU", num(hw.cpu) ? `${Math.round(hw.cpu)}%` : "-")
    + `<div class="sb-met">${sbRow("RAM", num(hw.ram_total) ? `${gib(hw.ram_used)} / ${gib(hw.ram_total)}` : "-", "GiB")}${bar(num(hw.ram_total) ? hw.ram_used / hw.ram_total : 0)}</div>`
    + sbRow("Disk read", num(hw.disk_read_mb) ? hw.disk_read_mb.toFixed(1) : "-", "MB/s")
    + (st.driver ? `<div class="sb-foot">Driver ${esc(st.driver)}${st.cuda ? ` · CUDA ${esc(st.cuda)}` : ""}</div>` : ""));

  const ds = eng.architecture === "deepseek41", routes = live.expert_routes || {};
  const running = (routes.primary || 0) + (routes.helper || 0) + (routes.ram || 0) > 0;
  text($("#expert-scope"), running ? "Active decode" : "Last request");
  let tiers, hitSummary = "";
  if (ds) {
    const current = running ? routes : r?.expert_routes || {};
    const primary = current.primary || 0, helper = current.helper || 0, ram = current.ram || 0;
    const total = primary + helper + ram;
    hitSummary = '<div class="sb-expert-summary">' + sbRow("VRAM hit", percent(primary + helper, total, 1)) + '</div>';
    tiers = [["Primary", "var(--fg)", primary], ["Peer", "var(--fg-3)", helper], ["RAM / CPU", "var(--basalt-500)", ram]]
      .map(([label, color, count]) => [label, color, total ? count / total : null]);
  } else {
    const peer = num(r?.pcie_share) ? r.pcie_share : null, hit = num(r?.hit_rate) ? r.hit_rate : null;
    tiers = [["VRAM", "var(--fg)", hit == null ? null : hit * (1 - (peer || 0))]];
    if (peer != null) tiers.push(["Peer", "var(--fg-3)", peer]);
    tiers.push(["CPU", "var(--basalt-500)", hit == null ? null : (1 - hit) * (1 - (peer || 0))]);
  }
  html($("#experts"), hitSummary + '<div class="sb-stack" aria-hidden="true">' + tiers.map(([,color,part]) =>
    '<i style="width:' + ((part || 0) * 100).toFixed(1) + '%;background:' + color + '"></i>').join("") + '</div>' +
    '<div class="sb-leg">' + tiers.map(([label, color, part]) => '<div style="--c:' + color + '"><span>' +
    label + '</span><b>' + percent(part, 1, 1) + '</b></div>').join("") + '</div>');
}

/* ---- Speed ---- */
const SC = ["#FF7A3D", "#FFB547", "#F3D3B5", "#D9582A", "#FFD08A", "#C98A6B", "#FF9A66", "#B8401A"], SD = ["", "7 3", "1.5 3", "8 3 1.5 3"];
const off = new Set();       // decode lines switched off: "t" (total) and slot numbers
let offFor = 0;              // the slot count `off` was set up for
function drawHero(live, hist, reqs, best, slots, total, eng) {
  const multi = total > 1, generating = live.state === "generating", reading = live.state === "reading";
  // the live rate while generating, else the newest finished request's that has one
  const r = reqs[0], rd = reqs.find(x => num(x.decode_tok_s)), last = rd ? rd.decode_tok_s : null;
  const liveDec = generating && num(live.tok_s), dec = liveDec ? live.tok_s : last;
  text($("#dec-k"), liveDec ? (multi ? "Decode · All slots" : "Decode") : last != null ? "Decode · Last request" : "Decode");
  html($("#dec-v"), num(dec) ? `${dec.toFixed(1)}<small>tok/s</small>` : "-");
  $("#dec-v").classList.toggle("idle", !liveDec);
  const draft = liveDec ? live : rd;
  text($("#dec-accept"), draftAcceptance(eng, draft));
  $("#dec-accept").hidden = !$("#dec-accept").textContent;
  const phases = live.phases || [];
  $("#dec-activity").hidden = !phases.length;
  html($("#dec-activity"), phases.map(s => '<span>' + esc(s[0].toUpperCase() + s.slice(1)) + '</span>').join(""));

  if (offFor !== total) { offFor = total; off.clear(); if (total >= 4) off.add("t"); }
  const ss = multi ? Array.from({ length: total }, (_, k) => tail((hist.slot_tok_s || [])[k])) : [tail(hist.tok_s)];
  const agg = tail(hist.tok_s);
  $("#chips").hidden = !multi;
  if (multi) {
    const items = [["t", "Total", "var(--fg)", ""]].concat(ss.map((_, k) => [String(k), `s${k + 1}`, SC[k % 8], SD[k % 4]]));
    html($("#chips"), `<div class="cks" role="group" aria-label="Decode lines">${items.map(x => `<button type="button" class="ck" data-k="${x[0]}" aria-pressed="${!off.has(x[0])}" style="color:${x[2]}"><i class="bx"></i>`
      + `<svg viewBox="0 0 12 6"><line x1="0" x2="12" y1="3" y2="3" stroke="${x[2]}" stroke-width="1.8" ${x[3] ? `stroke-dasharray="${x[3]}"` : ""}/></svg>${x[1]}</button>`).join("")}</div>`);
  }
  const showT = multi && !off.has("t"), vis = ss.map((_, k) => !multi || !off.has(String(k)));
  const peak = a => Math.max(0, ...a.filter(num));
  const max = nice(Math.max(showT ? peak(agg) : 0, ...ss.map((a, k) => vis[k] ? peak(a) : 0), 40));
  const y = v => 100 - 4 - Math.min(v, max) / max * 92;
  let svg = grid([0, .25, .5, .75]);
  const labels = [];
  ss.forEach((a, k) => {
    if (!vis[k]) return;
    lines(a, y).forEach(p => { svg += poly(p, `stroke="${SC[k % 8]}" stroke-width="${multi ? 1.3 : 1.7}" ${multi && SD[k % 4] ? `stroke-dasharray="${SD[k % 4]}"` : ""}`); });
  });
  const anyS = vis.some(Boolean) && multi;
  if (showT) lines(agg, y).forEach(p => { svg += poly(p, `stroke="${anyS ? "var(--fg)" : "var(--accent)"}" stroke-width="1.8"`); });
  let lab = "";
  const end = a => { const s = stats(a); return s ? y(s.last) : null; };
  if (showT && end(agg) != null) { labels.push(end(agg)); lab += `<span class="cl" style="top:${end(agg).toFixed(1)}%;color:${anyS ? "var(--fg)" : "var(--accent)"}">All</span>`; }
  if (!multi && end(ss[0]) != null) { labels.push(end(ss[0])); lab += `<span class="cl" style="top:${end(ss[0]).toFixed(1)}%;color:${SC[0]}">s1</span>`; }
  html($("#dec-chart"), `<svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Decode tokens per second, last 60 seconds">${svg}</svg>${lab}${ticks(max, labels)}`);

  // the live read rate while reading, else the newest finished request's that read part of its prompt
  const readOf = x => (x.prompt_tokens || 0) - (x.reused || 0);
  const rp = reqs.find(x => x.prompt_ms > 0 && readOf(x) > 0), lastPre = rp ? readOf(rp) / (rp.prompt_ms / 1000) : null;
  const livePre = reading && num(live.prefill_tok_s_mean) && live.prefill_tok_s_mean > 0;
  const pre = livePre ? live.prefill_tok_s_mean : lastPre;
  text($("#pre-k"), livePre ? "Prefill" : lastPre != null ? "Prefill · Last request" : "Prefill");
  html($("#pre-v"), num(pre) ? `${fmt(pre)}<small>tok/s</small>` : "-");
  const pa = tail(hist.prefill_tok_s_mean).map(v => v > 0 ? v : v === 0 ? 0 : null);
  const pmax = nice(Math.max(1000, ...pa.filter(num)));
  html($("#pre-chart"), `<svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Prefill tokens per second, last 60 seconds">${grid([0, .5])}`
    + lines(pa, v => 100 - 4 - Math.min(v, pmax) / pmax * 92).map(p => poly(p, `stroke="var(--fg)" stroke-width="1.6"`)).join("") + `</svg>${ticks(pmax, [])}`);
  html($("#pre-last"), r && num(r.prompt_tokens) ? `<b>${fmt(r.prompt_tokens)}</b> tok${num(r.prompt_ms) ? ` in <b>${(r.prompt_ms / 1000).toFixed(2)}</b> s` : ""}` : "-");
  text($("#pre-hit"), r && r.prompt_tokens > 0 && num(r.reused) ? `${Math.round(r.reused / r.prompt_tokens * 100)}%` : "-");

  const bd = best.decode, bp = best.prefill, busy = slots.filter(s => s.is_processing).length;
  html($("#foot-dec"), `${multi ? `${busy} of ${total} slots busy` : "Single slot"} · Best <b>${bd ? fmt(bd.tok_s, 1) : "-"}</b> tok/s`);
  html($("#foot-pre"), `Best <b>${bp ? fmt(bp.tok_s) : "-"}</b> tok/s`);
  $("#foot-dec").title = bd ? `Fastest whole request since the server started: ${bd.id}, ${fmt(bd.output_tokens)} tokens, ${clock(bd.time * 1000)}` : "";
  $("#foot-pre").title = bp ? `Fastest prompt read since the server started: ${bp.id}, ${fmt(bp.read_tokens)} tokens read, ${clock(bp.time * 1000)}` : "";
}
$("#chips").addEventListener("click", e => {
  const b = e.target.closest(".ck"); if (!b) return;
  off.has(b.dataset.k) ? off.delete(b.dataset.k) : off.add(b.dataset.k);
  draw();
});

/* ---- main card: Slots | Requests | Queue ---- */
let tab = sessionStorage.getItem("basalt.tab") || "slots";
$("#tabs").addEventListener("click", e => {
  const b = e.target.closest("button[data-t]"); if (!b) return;
  tab = b.dataset.t; try { sessionStorage.setItem("basalt.tab", tab); } catch (err) { /* private mode */ }
  draw();
});
const accept = s => s.drafts_offered > 20 ? Math.round(s.drafts_accepted / s.drafts_offered * 100) : null;
const prefixHit = s => num(s.reused) && s.prompt_tokens > 0 ? Math.round(s.reused / s.prompt_tokens * 100) : null;
const ACC_NOTE = "Accept: drafts accepted over offered, from the slot's last finished request";
// every slot keeps its row: an idle one shows its last figures dimmed, so the grid never changes shape
function slotRows(rows, series) {
  const dense = rows.length > 4, max = nice(Math.max(20, ...rows.flatMap(s => series[s.id].filter(num))));
  let h = `<div class="sl${dense ? " dense" : ""}"><div class="rw h"><span style="grid-column:1">Slot</span><span class="e" style="grid-column:2">tok/s</span><span class="sk" style="grid-column:3">Last 60 s</span><span class="e" style="grid-column:4">Output tok</span><span class="e cx" style="grid-column:5">Context</span><span class="e" style="grid-column:6" title="${ACC_NOTE}">Accept</span></div>`;
  rows.forEach(s => {
    const a = series[s.id], st = busyStats(a), api = API[s.api] || s.api || "", acc = accept(s), hit = prefixHit(s);
    const pre = s.state === "prefill", run = s.is_processing;
    const rate = !run ? "-" : pre ? (num(s.prefill_done) && s.prompt_tokens ? `${Math.min(100, Math.round(s.prefill_done / s.prompt_tokens * 100))}%` : "…") : fmt(s.tok_s);
    h += `<div class="rw${run ? "" : " idle"}"${pre ? ` title="Reading the prompt"` : ""}><span class="p nm" style="grid-column:1">Slot ${s.id + 1}${dense ? `<span class="u"> · ${run ? esc(api) : "Idle"}</span>` : ""}</span>`
      + `<span class="p rate e" style="grid-column:2">${rate}</span>${spark(a, max)}`
      + `<span class="p e" style="grid-column:4">${run ? `${fmt(s.generated)}${num(s.max_tokens) ? `<span class="u">/${lim(s.max_tokens)}</span>${bar(s.generated / s.max_tokens)}` : ""}` : "-"}</span>`
      + `<span class="p e cx" style="grid-column:5">${kt(s.ctx_used || 0)}<span class="u">/${kt(s.n_ctx || 0)}</span>${bar(s.n_ctx ? s.ctx_used / s.n_ctx : 0)}</span>`
      + `<span class="p e" style="grid-column:6">${acc != null ? `${acc}<span class="u">%</span>` : "-"}</span>`
      + (dense ? "" : `<span class="s" style="grid-column:1">${run ? esc(api) : "Idle"}</span>`
        + `<span class="s e" style="grid-column:2">${pre ? "Read" : st ? `Avg ${Math.round(st.avg)}` : ""}</span>`
        + `<span class="s e" style="grid-column:4">${num(s.ttft_s) ? `TTFT ${s.ttft_s.toFixed(1)} s` : ""}</span>`
        + `<span class="s e cx" style="grid-column:5">${hit != null ? `Hit ${hit}%` : ""}</span>`
        + `<span class="s e" style="grid-column:6">${num(s.draft_len) ? `Draft ${s.draft_len.toFixed(1)}` : ""}</span>`) + `</div>`;
  });
  return h + `</div>`;
}
function requestsPane(live, reqs, multi) {
  const rows = [];
  if (!multi && (live.state === "reading" || live.state === "generating"))
    rows.push({ t: Date.now() / 1000 - (live.elapsed_s || 0), id: "Running", path: "", out: live.generated, tps: live.tok_s, dur: live.elapsed_s, live: true });
  reqs.forEach(r => rows.push({ t: r.time, id: r.id, slot: r.slot, path: r.path, out: r.output_tokens, tps: r.decode_tok_s, dur: r.duration_s, finish: r.finish }));
  if (!rows.length) return `<div class="empty">No requests yet.</div>`;
  const ep = p => String(p || "-").replace("/v1/", "").replace("chat/completions", "chat");
  return `<table><thead><tr><th>Time</th><th>ID</th>${multi ? "<th>Slot</th>" : ""}<th>Endpoint</th><th class="r">Output</th><th class="r">Rate</th><th class="r">Duration</th></tr></thead><tbody>`
    + rows.map(r => {
      // a request that did not simply stop is marked on its id, with the reason
      const odd = r.finish && r.finish !== "stop" ? (r.finish === "error" ? "err" : "warn") : "";
      return `<tr${r.live ? ` class="live"` : ""}><td>${clock(r.t * 1000)}</td><td class="${odd}">${esc(r.id)}${odd ? ` · ${esc(r.finish)}` : ""}</td>`
        + (multi ? `<td class="dim">${num(r.slot) ? `s${r.slot + 1}` : "-"}</td>` : "")
        + `<td>${r.live ? "-" : esc(ep(r.path))}</td><td class="r">${num(r.out) ? fmt(r.out) : "-"}</td><td class="r">${num(r.tps) ? r.tps.toFixed(1) : "-"}</td>`
        + `<td class="r">${num(r.dur) ? `${r.dur.toFixed(1)} s` : "-"}</td></tr>`;
    }).join("") + `</tbody></table>`;
}
const mss = s => `${Math.floor(s / 60)}:${p2(Math.floor(s % 60))}`;
function queuePane(live) {
  const q = live.queue || [], n = live.queued || 0;
  if (!n && !q.length) return `<div class="empty">No request is waiting.</div>`;
  const toks = q.reduce((t, r) => t + (num(r.prompt_tokens) ? r.prompt_tokens : 0), 0);
  return `<div class="qsum"><span><b>${fmt(Math.max(n, q.length))}</b> prompt${Math.max(n, q.length) === 1 ? "" : "s"} waiting</span><span><b>${fmt(toks)}</b> tok to read</span></div>`
    + `<table><thead><tr><th>API</th><th class="r">Prompt tok</th><th class="r">Waiting</th></tr></thead><tbody>`
    + q.map(r => `<tr><td>${esc(API[r.api] || r.api || "-")}</td><td class="r">${num(r.prompt_tokens) ? fmt(r.prompt_tokens) : "-"}</td><td class="r">${mss(r.waited_s || 0)}</td></tr>`).join("")
    + `</tbody></table>`;
}
function drawMain(live, hist, reqs, slots, total) {
  const multi = total > 1, n = live.queued || 0;
  const T = (multi ? [["slots", "Slots"]] : []).concat([["req", "Requests"], ["queue", `Queue (${n})`]]);
  const cur = T.some(t => t[0] === tab) ? tab : T[0][0];
  html($("#tabs"), T.map(t => `<button type="button" role="tab" data-t="${t[0]}" aria-selected="${t[0] === cur}">${t[1]}</button>`).join(""));
  let h;
  if (cur === "slots") {
    const series = slots.map((_, k) => tail((hist.slot_tok_s || [])[k]));
    h = slotRows(slots, series);
  } else h = cur === "req" ? requestsPane(live, reqs, multi) : queuePane(live);
  html($("#pane"), h);
}

/* ---- Devices ---- */
const row = (label, value, unit = "", note = "") =>
  '<div class="metric-row"><span>' + esc(label) + (note ? '<small>' + esc(note) + '</small>' : '') +
  '</span><b class="num">' + esc(value) + (unit ? '<small>' + esc(unit) + '</small>' : '') + '</b></div>';
const percent = (n, d, digits = 0) => num(n) && d > 0 ? (n / d * 100).toFixed(digits) + "%" : "—";
const dash = "—";
const bytes = n => !num(n) ? [dash, ""] : n >= 1024 ** 3 ? [gib(n), "GiB"] : [fmt(n / 1048576, 1), "MiB"];
const memoryRow = (label, count, note) => row(label, ...bytes(count), note);
const legend = (label, value, color) => '<div class="memory-item" style="--swatch:' + color + '">' +
  memoryRow(label, value) + '</div>';
const colors = ["#ff8047", "#68b5aa", "#71a7bd", "#83a9e5", "#b18acf", "#e6b569", "#d88e9b", "#82b189",
  "#d59bcc", "#99b9c3", "#9ca9b8", "#837e9c"];
const grouped = (label, content) => '<section class="device-section"><h3>' + label + '</h3>' + content + '</section>';
const chart = (label, values, now, unit, ceiling, accent = false, decimals = 0, readout = null) => {
  const series = tail(values), s = stats(series);
  if (!num(now) && !s) return "";
  return '<div class="device-chart">' + row(label, readout ?? (num(now) ? fmt(now, decimals) : dash), unit) +
    sbChart(series, 0, ceiling > 0 ? ceiling : Math.max(1, (s?.max || 0) * 1.15), accent) + '</div>';
};
function usageBar(used, capacity, parts) {
  if (!num(used) || !(capacity > 0)) return "";
  const slices = parts && parts.length ? parts : [[used, "var(--accent)"]];
  return '<div class="usage-bar" aria-hidden="true">' + slices.map(([n, color]) =>
    '<i style="--share:' + (100 * clamp(n / capacity, 0, 1)).toFixed(4) + '%;--fill:' + color + '"></i>').join("") + '</div>';
}
function memoryMeter(label, used, total, free, parts) {
  const quantity = num(used) ? gib(used) + (num(total) ? ' <span>/ ' + gib(total) + '</span>' : '') + ' <small>GiB</small>' : dash;
  return '<div class="memory-meter"><div class="meter-label">' + label + ' <span>' + percent(used, total) +
    '</span></div><div class="meter-value num">' + quantity + '</div>' + usageBar(used, total, parts) +
    '<div class="meter-free">' + (num(free) ? gib(free) + ' GiB available' : 'Available memory unavailable') + '</div></div>';
}
const vramLabels = {weights: "Model weights", experts: "Expert cache", expert_helper: "Expert helper",
  kv_state: "KV cache & state", scratch: "Scratch", mmq_buffers: "MMQ buffers", dense_prefill: "Dense prefill",
  long_prefill: "32K prefill", dspark: "DSpark", vision: "Vision workspace",
  system_reserved: "System reserved", other_apps: "Other apps"};
const hostLabels = {resident_experts: "Resident experts", expert_homes: "Evicted / borrowed experts",
  checkpoints: "Conversation checkpoints", engram_workspace: "Engram workspace"};
function memoryParts(parts, labels) {
  return Object.entries(labels).flatMap(([key, label], i) =>
    num(parts?.[key]) ? [[label, parts[key], colors[i % colors.length]]] : []);
}
function gpuCard(g, history, st) {
  const b = g.vram_breakdown, entries = memoryParts(b?.components, vramLabels);
  if (b && !b.pending_engine) entries.push(["Unattributed", b.unattributed, "var(--fg-3)"]);
  const slices = b && num(b.unattributed) ? entries.filter(e => num(e[1])).map(e => e.slice(1)) : null;
  const gen = g.pcie_gen ?? g.pcie_gen_max;
  const link = num(gen) ? "Gen" + gen + (num(g.pcie_width) ? " ×" + g.pcie_width : "") : dash;
  const power = num(g.power_limit) ? (num(g.power) ? fmt(g.power) + " W" : dash) + " / " + fmt(g.power_limit) + " W" : null;
  const name = shortGpu(st?.name) || "NVIDIA GPU";
  return '<article class="card device"><header class="device-head"><div><p>GPU ' + fmt(g.index) +
    '</p><h2>' + esc(name) + '</h2></div><span class="badge">' + (num(g.temp) ? fmt(g.temp) + ' °C' : 'GPU') +
    '</span></header>' + memoryMeter("VRAM", g.mem_used, g.mem_total, g.mem_free, slices) +
    grouped("VRAM breakdown", entries.map(e => legend(...e)).join("") +
      (b?.pending_engine ? '<p class="hint">Waiting for engine allocations.</p>' :
        !entries.length ? '<p class="hint">Per-allocation data is not reported for this device.</p>' : "")) +
    grouped("Activity", '<div class="chart-grid">' +
      chart("Load", history.util, g.util, "%", 100, true) +
      chart("Power", history.power, g.power, power ? "" : "W", g.power_limit, false, 0, power) +
      chart("Temperature", history.temp, g.temp, "°C", 100) +
      chart("VRAM used", (history.mem_used || []).map(v => num(v) ? v / 1024 ** 3 : null), num(g.mem_used) ? g.mem_used / 1024 ** 3 : null, "GiB", g.mem_total / 1024 ** 3, false, 1) +
      '</div>' + row("PCIe link", link, "", num(g.pcie_gen_max) && g.pcie_gen_max !== gen ? "Max Gen" + g.pcie_gen_max : "") +
      '<div class="chart-grid pcie">' + chart("RX to GPU", history.pcie_rx_mb, g.pcie_rx_mb, "MiB/s", null, false, 1) +
      chart("TX from GPU", history.pcie_tx_mb, g.pcie_tx_mb, "MiB/s", null, false, 1) + '</div>' +
      (!num(g.pcie_rx_mb) && !num(g.pcie_tx_mb) ? '<p class="hint">Bandwidth counters unavailable.</p>' : "")) + '</article>';
}
function hostCard(hw, st, hist) {
  const ram = hw.ram_breakdown || {}, p = ram.processes || {}, parts = ram.engine_allocations || {};
  const entries = memoryParts(parts, hostLabels);
  const allocated = entries.reduce((sum, [, n]) => sum + n, 0);
  const used = hw.ram_used, total = hw.ram_total;
  const proc = [["engine", "Inference engine"], ["vision", "Vision process"], ["server", "API server"]];
  const resident = proc.filter(([key]) => num(p[key]?.rss)).map(([key, label]) => memoryRow(label, p[key].rss)).join("");
  const processStats = p.engine || {};
  return '<article class="card device"><header class="device-head"><div><p>Host</p><h2>System memory</h2></div><span class="badge">RAM</span></header>' +
    memoryMeter("Physical RAM", used, total, num(used) && num(total) ? total - used : null) +
    grouped("Engine allocations", entries.length ? '<div class="allocation-bar">' +
      usageBar(allocated, allocated, entries.map(e => e.slice(1))) + '</div>' +
      entries.map(e => legend(...e)).join("") +
      ((num(parts.pinned_experts) || num(parts.locked_experts)) ?
        '<div class="subrows">' + (num(parts.pinned_experts) ? memoryRow("CUDA-registered", parts.pinned_experts) : "") +
        (num(parts.locked_experts) ? memoryRow("OS locked", parts.locked_experts) : "") + '<p class="hint">Subsets of resident experts.</p></div>' : "") +
      '<p class="hint">Allocated bytes can include paged-out memory.</p>' : '<p class="hint">This engine does not report RAM allocations.</p>') +
    grouped("Process RAM · working set", (resident || '<p class="hint">Process counters unavailable.</p>') +
      (num(processStats.peak_rss) ? memoryRow("Engine peak working set", processStats.peak_rss) : "") +
      (num(processStats.private_commit) ? memoryRow("Engine private commit", processStats.private_commit) : "") +
      (resident ? '<p class="hint">Working sets can overlap. Private commit can include paged-out memory.</p>' : "")) +
    grouped("Host activity", row("Processor", shortCpu(st.cpu_name) || dash) +
      row("Cores / threads", (num(st.cores) ? fmt(st.cores) : dash) + " / " + (num(st.threads) ? fmt(st.threads) : dash)) +
      row("Disk read", num(hw.disk_read_mb) ? fmt(hw.disk_read_mb, 1) : dash, "MiB/s") +
      '<div class="chart-grid">' + chart("CPU load", hist.cpu, hw.cpu, "%", 100, true) +
      chart("RAM used", (hist.ram_used || []).map(v => num(v) ? v / 1024 ** 3 : null), num(used) ? used / 1024 ** 3 : null, "GiB", total / 1024 ** 3, false, 1) +
      '</div>') + '</article>';
}
function drawDevices(hw, st, hist) {
  const cards = Object.fromEntries((st.gpus || []).map(x => [x.index, x]));
  const series = Object.fromEntries((hist.gpus || []).map(x => [x.index, x]));
  text($("#device-driver"), [st.driver ? "Driver " + st.driver : "", st.cuda ? "CUDA " + st.cuda : ""].filter(Boolean).join(" · ") || "Last 60 seconds");
  const gpus = (hw.gpus || []).map(g => gpuCard(g, series[g.index] || {}, cards[g.index]));
  html($("#device-grid"), gpus.join("") + hostCard(hw, st, hist) +
    (!gpus.length ? '<div class="card empty">No GPU telemetry available.</div>' : ""));
}


function draftAcceptance(eng, sample) {
  return eng.architecture === "deepseek41" && sample?.drafts_offered > 0
    ? "DSpark accept " + percent(sample.drafts_accepted, sample.drafts_offered) : "";
}
function drawDeviceInference(live, eng, hist, reqs) {
  const active = live.state === "generating" && num(live.tok_s);
  const last = reqs.find(r => num(r.decode_tok_s));
  const rate = active ? live.tok_s : last?.decode_tok_s;
  const acceptance = draftAcceptance(eng, active ? live : last);
  const samples = tail(hist.tok_s), scale = nice(Math.max(40, ...samples.filter(num)));
  text($("#device-decode-source"), active ? "Live" : last ? "Last request" : "Idle");
  html($("#device-decode-rate"), '<span>' + (num(rate) ? fmt(rate, 1) + ' <small>tok/s</small>' : dash) + '</span>' +
    (acceptance ? '<small>' + acceptance + '</small>' : ''));
  html($("#device-decode-chart"), sbChart(samples, 0, scale, true));

  const phases = live.phases || [];
  const reading = live.state === "reading" || phases.includes("reading the prompt");
  const readOf = r => (r.prompt_tokens || 0) - (r.reused || 0);
  const lastPrompt = reqs.find(r => r.prompt_ms > 0 && readOf(r) > 0);
  const prefill = reading ? live.prefill_tok_s_mean : lastPrompt ? readOf(lastPrompt) * 1000 / lastPrompt.prompt_ms : null;
  const prefillSamples = tail(hist.prefill_tok_s_mean);
  text($("#device-prefill-source"), reading ? "Live" : lastPrompt ? "Last request" : "Idle");
  html($("#device-prefill-rate"), prefill > 0 ? fmt(prefill) + ' <small>tok/s</small>' : dash);
  html($("#device-prefill-chart"), sbChart(prefillSamples, 0, nice(Math.max(1000, ...prefillSamples.filter(num))), false));

  const actions = {
    reading: "Reading the prompt", generating: "Generating", unloaded: "Model unloaded", asleep: "Asleep", idle: "Idle",
    loading: live.load?.kind === "wake" ? "Waking up" : "Loading weights",
  };
  const action = phases.length ? phases.map(s => s.charAt(0).toUpperCase() + s.slice(1)).join(" · ") : actions[live.state] || "Idle";
  text($("#device-engine-action"), action);
}
const onDevices = () => location.hash === "#devices";
function showPage() {
  const devices = onDevices();
  $("#overview").hidden = devices;
  $("#devices").hidden = !devices;
  for (const a of document.querySelectorAll("[data-page]")) {
    if ((a.dataset.page === "devices") === devices) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  window.scrollTo(0, 0);
  $("#devices").scrollTop = 0;
  draw();
}

/* ---- the whole page ---- */
function draw() {
  if (!M) return;
  const live = M.live || {}, eng = M.engine || {}, hist = M.history || {}, reqs = M.requests || [];
  const total = (live.slots || {}).total || 1, slots = M.slots || [];
  const name = (live.state || "idle")[0].toUpperCase() + (live.state || "idle").slice(1);
  const ld = live.state === "loading" ? live.load || {} : null;
  const gib = ld ? Object.entries(ld.gib || {}).map(([k, v]) => `${k} ${v.toFixed(1)} GiB`).join(" · ") : "";
  setState(live.state || "idle", ld ? ld.kind === "wake" ? "Waking up" : `Loading weights${num(ld.percent) ? ` · ${ld.percent}%` : ""}`
    : live.queued > 0 ? `${name} · ${live.queued} queued` : name);
  text($("#detail"), gib); $("#detail").hidden = !gib;
  text($("#model"), eng.model || "");
  if (onDevices()) {
    drawDevices(M.hardware || {}, M.hardware_static || {}, hist);
    drawDeviceInference(live, eng, hist, reqs);
  } else {
    drawSide(M.hardware || {}, M.hardware_static || {}, hist, eng, reqs[0], live);
    drawHero(live, hist, reqs, M.best || {}, slots, total, eng);
    drawMain(live, hist, reqs, slots, total);
  }
}

/* ---- logs ---- */
const logEl = $("#log"), MAXLOG = 400;
let logCount = 0, follow = true;
function lineEl(e) {
  const d = document.createElement("div");
  if (e.lvl !== "info") d.className = e.lvl;
  const t = document.createElement("i"); t.textContent = clock(e.t * 1000);
  d.append(t, `  ${e.src === "srv" ? "[server] " : ""}${e.msg}`);
  return d;
}
function append(lines) {
  logEl.querySelector(".empty")?.remove();
  logEl.append(...lines.map(lineEl));
  while (logEl.childElementCount > MAXLOG) logEl.firstChild.remove();
  logCount += lines.length; text($("#log-count"), `${fmt(logCount)} lines`);
  if (follow) logEl.scrollTop = logEl.scrollHeight;
}
logEl.addEventListener("scroll", () => { follow = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 12; });
function clearLog(note) {
  logCount = 0; text($("#log-count"), "");
  const d = document.createElement("div"); d.className = "empty"; d.textContent = note; logEl.replaceChildren(d);
}

/* ---- polling ---- */
let lastLog = 0;
async function pollMetrics() {
  try {
    const m = await get("metrics");
    if (!onDevices()) m.slots = await get("slots").catch(() => []);
    M = m; $("#auth").hidden = true; draw();
  } catch (e) {
    $("#auth").hidden = !e.locked; M = null;
    setState("offline", e.locked ? "Locked" : "Offline");
    text($("#device-engine-action"), e.locked ? "Locked" : "Offline");
  }
  setTimeout(pollMetrics, 1000);
}
async function pollLog() {
  if (onDevices()) { setTimeout(pollLog, 1000); return; }
  try {
    const r = await get(`api/logs?after=${lastLog}`);
    if (r.last < lastLog) { lastLog = 0; clearLog("The server restarted."); }
    else if (r.lines.length) { lastLog = r.last; append(r.lines); }
    else if (!logCount && !logEl.firstChild) clearLog("No log lines yet.");
  } catch (e) { /* the metrics poll reports the connection */ }
  setTimeout(pollLog, 1000);
}
$("#auth").onsubmit = e => {
  e.preventDefault(); apiKey = $("#key").value.trim(); sessionStorage.setItem("basalt.key", apiKey);
};

$("#host").textContent = location.host;
fetch("health").then(r => r.json()).then(h => { $("#keyed").hidden = !h.api_key; }).catch(() => {});
window.addEventListener("hashchange", showPage);
showPage();
pollMetrics(); pollLog();
