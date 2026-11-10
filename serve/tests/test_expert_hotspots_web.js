"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/hotspots.js"), "utf8") + "\nthis.report = ExpertHotspotReport;", context);
const report = context.report;
const s = {supported: true, recording: false, model: "test", captured_at: 1700000000,
  engine_session: "1:1", epoch: 1, layers: 2, experts: 3, scope: "All slots, rejected rows included.",
  formats: [["iq1_s", "iq2_xxs", 100], ["iq1_m", "iq1_s", 120]],
  overrides: [[0, 1, "iq2_xxs", "iq2_xxs", 150]],
  prefill: [0, 0, 20, 1, 0, 0], decode: [2, 6, 0, 2, 0, 0]};
const rows = report.rows(s, "decode");
assert.equal(rows[0].expert, 1);
assert.equal(rows[0].layer, 0);
assert.equal(rows[0].relative, 2.25);
assert.equal(rows[0].coverage, .6);
assert.equal(rows[0].overlay, true);
assert.equal(rows[0].gate_up, "iq2_xxs");
assert.equal(report.rows(s, "prefill")[0].expert, 2);
assert.equal(report.rows(s, "combined")[0].count, 20);
assert.equal(rows.at(-1).coverage, 1);
const csv = report.csv(s, "decode", '=formula,"test"');
assert.equal(csv.trim().split("\r\n").length, 7);
assert.ok(csv.includes('"\'=formula,""test"""'));
assert.ok(csv.includes('"prefill_count","decode_verify_count"'));
const md = report.markdown(s, "decode", "geth\n|test|");
assert.ok(md.includes("# Expert hotspot report"));
assert.ok(md.includes("zero-based"));
assert.ok(md.includes("| 1 | 0 | 1 | 6 | 2.25× | 60.00% | iq2_xxs | iq2_xxs | Yes |"));
assert.equal(report.selected(rows, 2).length, 2);
assert.equal(report.selected(rows, null).length, 3);
assert.equal(report.selected(rows, 1000).length, 3);
assert.equal(report.selected(rows, 1.9).length, 1);
const limited = report.markdown(s, "decode", "", 2);
assert.ok(limited.includes("Showing 2 of 3 active experts"));
assert.ok(!limited.includes("| 3 |"));
assert.ok(report.markdown(s, "decode", "", null).includes("Showing 3 of 3 active experts"));
const empty = {...s, prefill: [], decode: []};
assert.equal(report.rows(empty, "combined")[0].relative, 0);
assert.ok(!report.markdown(empty, "decode", "").includes("NaN"));
console.log("Expert hotspot exports: OK");
