"use strict";
const ExpertHotspotReport = (() => {
  const phases = {decode: "Decode / verify", prefill: "Prefill", combined: "Combined"};
  function rows(s, phase) {
    const overrides = new Map((s.overrides || []).map(r => [`${r[0]}:${r[1]}`, r.slice(2)]));
    const result = [], totals = Array(s.layers).fill(0);
    for (let l = 0; l < s.layers; ++l) for (let e = 0; e < s.experts; ++e) {
      const i = l * s.experts + e, pp = s.prefill[i] || 0, tg = s.decode[i] || 0;
      const count = phase === "decode" ? tg : phase === "prefill" ? pp : pp + tg;
      const replacement = overrides.get(`${l}:${e}`), format = replacement || s.formats[l] || [];
      totals[l] += count;
      result.push({layer: l, expert: e, count, prefill: pp, decode: tg, gate_up: format[0] || "unknown",
        down: format[1] || "unknown", bytes: format[2] || 0, overlay: !!replacement});
    }
    const total = totals.reduce((a, b) => a + b, 0);
    result.sort((a, b) => b.count - a.count || a.layer - b.layer || a.expert - b.expert);
    let cumulative = 0;
    return result.map((r, i) => {
      cumulative += r.count;
      return {...r, rank: i + 1, share: total ? r.count / total : 0,
        layer_share: totals[r.layer] ? r.count / totals[r.layer] : 0,
        relative: totals[r.layer] ? r.count * s.experts / totals[r.layer] : 0,
        coverage: total ? cumulative / total : 0};
    });
  }
  function csv(s, phase, label) {
    const cell = v => typeof v === "number" ? String(v) : '"' + String(v).replace(/^([=+@-])/, "'$1").replace(/"/g, '""') + '"';
    const header = ["model", "label", "captured_utc", "engine_session", "capture_epoch", "phase", "rank", "layer", "expert",
      "count", "prefill_count", "decode_verify_count", "route_share", "layer_share", "layer_mean_multiple", "cumulative_share",
      "gate_up_format", "down_format", "bytes", "overlay", "scope"];
    const prefix = [s.model, label, new Date(s.captured_at * 1000).toISOString(), s.engine_session, s.epoch, phase];
    return [header, ...rows(s, phase).map(r => [...prefix, r.rank, r.layer, r.expert, r.count, r.prefill, r.decode,
      r.share, r.layer_share, r.relative, r.coverage, r.gate_up, r.down, r.bytes, r.overlay, s.scope])]
      .map(r => r.map(cell).join(",")).join("\r\n") + "\r\n";
  }
  function selected(ranked, limit) {
    const active = ranked.filter(r => r.count);
    return limit === null ? active : active.slice(0, Math.max(1, Math.floor(limit)));
  }
  function markdown(s, phase, label, limit = 100) {
    const safe = v => String(v || "").replace(/[\r\n]/g, " ").replace(/[|`<>]/g, "");
    const ranked = rows(s, phase), shown = selected(ranked, limit), activeCount = ranked.filter(r => r.count).length;
    const total = ranked.reduce((a, r) => a + r.count, 0);
    return `# Expert hotspot report\n\nModel: ${safe(s.model)}\n\nWorkload: ${safe(label) || "Unlabelled"}\n\n` +
      `Captured: ${new Date(s.captured_at * 1000).toISOString()} · ${s.recording ? "Recording" : "Paused"}\n\n` +
      `Engine session: ${safe(s.engine_session)} · capture epoch ${s.epoch}\n\n` +
      `View: ${phases[phase]} · ${total.toLocaleString("en-US")} evaluated expert routes · zero-based layer/expert IDs\n\n` +
      `${s.scope}\n\nLayer-relative heat = count / mean count in that layer. Frequency identifies candidates; it does not establish domain specialization or quality improvement. Compare against a control workload and validate upgraded experts separately.\n\n` +
      `Showing ${shown.length} of ${activeCount} active experts, ranked by count; CSV exports every expert and both phase counts.\n\n` +
      "| Rank | Layer | Expert | Count | Layer-relative | Cumulative routes | Gate/up | Down | Overlay |\n" +
      "|---:|---:|---:|---:|---:|---:|---|---|---|\n" + shown.map(r =>
        `| ${r.rank} | ${r.layer} | ${r.expert} | ${r.count} | ${r.relative.toFixed(2)}× | ${(r.coverage * 100).toFixed(2)}% | ${r.gate_up} | ${r.down} | ${r.overlay ? "Yes" : "No"} |`).join("\n") + "\n";
  }
  return {rows, selected, csv, markdown};
})();
