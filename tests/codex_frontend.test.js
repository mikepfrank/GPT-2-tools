"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../src/gpt2_local/web/codex.html"), "utf8");
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(match => match[1]);
const context = vm.createContext({URL});
vm.runInContext(scripts[0] + "\nglobalThis.helpers = CodexChatHelpers;", context);
const helpers = context.helpers;
const plain = value => JSON.parse(JSON.stringify(value));

test("both browser scripts parse before loading any session", () => {
  assert.equal(scripts.length, 2);
  for (const script of scripts) assert.doesNotThrow(() => new vm.Script(script));
});

test("API and saved-session scope stay under the private chat URL", () => {
  const privateUrl = "https://codex-test.gpt2-chat.com/closed-beta/";
  assert.equal(helpers.apiBase(privateUrl), privateUrl + "api/");
  assert.equal(helpers.apiBase(privateUrl + "?shared=1#history"), privateUrl + "api/");
  assert.equal(helpers.apiBase("https://codex-test.gpt2-chat.com/closed-beta"), privateUrl + "api/");
  assert.equal(helpers.storageKey(privateUrl), helpers.storageKey(privateUrl + "?shared=1#history"));
  assert.notEqual(helpers.storageKey(privateUrl), helpers.storageKey("https://codex-test.gpt2-chat.com/another/"));
  assert.notEqual(helpers.storageKey(privateUrl), helpers.storageKey("https://gpt2-chat.com/closed-beta/"));
});

test("role coloring preserves exact prompt whitespace and delimiter text", () => {
  const blocks = [{role: "context", text: "Header: three participants.\n"},
    {role: "Human", text: "\n\nHuman> Hello 😀"},
    {role: "Codex", text: "\n\nCodex> One paragraph.\nWith a line break."},
    {role: "GPT-2", text: "\n\nGPT-2> A response."},
    {role: "GPT-2", text: "\n\nGPT-2>"}];
  const state = {context_text: blocks.map(block => block.text).join(""), context_blocks: blocks};
  assert.deepEqual(plain(helpers.contextBlocks(state)), blocks);
  assert.equal(helpers.contextBlocks(state).map(block => block.text).join(""), state.context_text);
});

test("mismatched or unknown role blocks fall back to the authoritative complete prompt", () => {
  const exact = "Header\n\nHuman> <img src=x onerror=alert(1)>\n\nGPT-2> ";
  const broken = [{role: "Human", text: exact.trimEnd()}];
  for (const blocks of [broken, [{role: "Tool", text: exact}], null, [null]]) {
    assert.deepEqual(plain(helpers.contextBlocks({context_text: exact, context_blocks: blocks})), [{role: "context", text: exact}]);
  }
});

test("an older poll cannot overwrite newer messages or an in-progress reply", () => {
  const current = {session_id: "a", revision: 22};
  assert.equal(helpers.acceptsState(current, {session_id: "a", revision: 21}), false);
  assert.equal(helpers.acceptsState(current, {session_id: "a", revision: 22}), true);
  assert.equal(helpers.acceptsState(current, {session_id: "a", revision: 23}), true);
  assert.equal(helpers.acceptsState(current, {session_id: "new-chat", revision: 1}), true);
  assert.equal(helpers.acceptsState(null, {session_id: "a", revision: 1}), true);
  assert.equal(helpers.acceptsState(current, {session_id: "a"}), true);
});

test("a lost message acknowledgment is reconciled from durable Human request IDs", () => {
  const pending = {session_id: "chat", request_id: "request-7", message: "hello", target: "auto"};
  assert.equal(helpers.pendingAccepted(pending, {session_id: "chat", last_request_id: "request-7"}), true);
  // Another browser tab can send a later request before this tab reconnects.
  const afterRestart = {session_id: "chat", last_request_id: "request-8", messages: [
    {role: "Human", request_id: "request-7", content: "hello"}, {role: "Codex", content: "Hi."},
    {role: "Human", request_id: "request-8", content: "a later message"}]};
  assert.equal(helpers.pendingAccepted(pending, afterRestart), true);
  assert.equal(helpers.pendingAccepted(pending, {...afterRestart, session_id: "different"}), false);
  assert.equal(helpers.pendingAccepted(pending, {session_id: "chat", messages: [{role: "GPT-2", request_id: "request-7"}]}), false);
  assert.equal(helpers.pendingAccepted(null, afterRestart), false);
});

test("polling is responsive during work and backs off during interruptions", () => {
  for (const status of ["queued", "thinking", "generating"]) assert.equal(helpers.pollDelay({status}), 400);
  for (const status of ["idle", "error"]) assert.equal(helpers.pollDelay({status}), 2000);
  assert.equal(helpers.pollDelay({status: "generating"}, 1), 2000);
  assert.equal(helpers.pollDelay({status: "generating"}, 2), 4000);
  assert.equal(helpers.pollDelay({status: "generating"}, 6), 30000);
  assert.equal(helpers.pollDelay(null, 100), 30000);
});

test("status explains autonomous waiting, pause and active turns", () => {
  assert.equal(helpers.describeStatus({status: "idle", paused: true, next_idle_seconds: 20}), "Autonomous activity paused");
  assert.equal(helpers.describeStatus({status: "thinking", paused: true}), "Codex is thinking…");
  assert.equal(helpers.describeStatus({status: "idle", next_idle_seconds: 59.2}), "Ready · next autonomous turn in 60s");
  assert.equal(helpers.describeStatus({status: "idle", next_idle_seconds: null, autonomous_turns: 3}), "Ready · waiting for you");
  assert.equal(helpers.describeStatus({status: "error", error: "budget exhausted"}), "Conversation needs attention");
});

test("daily allowance displays conservative admission charges independently of reported estimates", () => {
  const budget = {utc_day: "2026-10-05", daily_limit_usd: 10, admitted_usd: 1.235,
    remaining_usd: 8.765, reported_estimate_usd: .43, requests: 3, unknown_usage_requests: 1};
  assert.equal(helpers.budgetText(budget), "Daily API allowance: $1.24 / $10.00");
  assert.equal(helpers.budgetText({...budget, admitted_usd: 0}), "Daily API allowance: $0.00 / $10.00");
  assert.equal(helpers.budgetText({...budget, admitted_usd: "unknown"}), "Daily API allowance: — / $10.00");
  assert.match(helpers.budgetTooltip(budget), /2026-10-05 \(UTC\)/);
  assert.match(helpers.budgetTooltip(budget), /conservative allowance before execution, including interrupted requests/);
  assert.match(helpers.budgetTooltip(budget), /Reported usage estimate: \$0\.43/);
  assert.doesNotMatch(helpers.budgetText(budget), /estimate|reserved|spent/);
});

test("server Unix-second timestamps and ISO timestamps display the same local time", () => {
  const milliseconds = Date.UTC(2026, 9, 5, 21, 37, 15);
  const iso = new Date(milliseconds).toISOString();
  assert.equal(helpers.messageTime(milliseconds / 1000), helpers.messageTime(iso));
  assert.equal(helpers.messageTime(milliseconds / 1000, true), helpers.messageTime(iso, true));
  assert.match(helpers.messageTime(milliseconds / 1000, true), /2026/);
  for (const invalid of [null, undefined, "", "not a date", Infinity]) assert.equal(helpers.messageTime(invalid), "");
});

test("temperature rejects blank, nonfinite and out-of-range values", () => {
  assert.equal(helpers.validateTemperature("0"), 0);
  assert.equal(helpers.validateTemperature(" 0.8 "), .8);
  assert.equal(helpers.validateTemperature("2"), 2);
  for (const value of ["", " ", "NaN", "Infinity", "-1", "2.01"]) assert.throws(() => helpers.validateTemperature(value), /between 0 and 2/);
});
