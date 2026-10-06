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

test("new chats are unseeded by default and each creation reads the visible example checkbox", async () => {
  const checkbox = html.match(/<input\s[^>]*id="include-example"[^>]*>/)[0];
  assert.match(checkbox, /type="checkbox"/);
  assert.doesNotMatch(checkbox, /\bchecked\b/);
  assert.match(html, />Include an example round<\/label>/);
  const createSource = scripts[1].match(/    async function createSession\(\) \{[\s\S]*?\n    \}/)[0];
  const calls = [];
  const ui = {"include-example": {checked: false}, message: {value: "", focus() {}}};
  const creator = vm.createContext({ui, clearTimeout() {}, refreshControls() {}, refreshNotice() {},
    save() {}, schedule() {}, applyState() {}, presence: async () => {},
    request: async (route, body) => { calls.push({route, body: plain(body)}); return {session_id: "new"}; }});
  vm.runInContext("let mutation=false,timer=null,actionError='',sessionId=null,sessionEpoch=0,pollSequence=0,state=null,historySignature='',lastContext=null,pending=null,pendingContinue=null,connectionError='',failures=0;\n"+createSource, creator);
  await vm.runInContext("createSession()", creator);
  ui["include-example"].checked = true;
  await vm.runInContext("createSession()", creator);
  ui["include-example"].checked = false;
  await vm.runInContext("createSession()", creator);
  assert.deepEqual(calls, [{route: "sessions", body: {example_rounds: 0}},
    {route: "sessions", body: {example_rounds: 1}}, {route: "sessions", body: {example_rounds: 0}}]);
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

test("context recolors when role blocks change even if the streamed text is identical", () => {
  const renderSource = scripts[1].match(/    function renderContext\(incoming\) \{[\s\S]*?\n    \}/)[0];
  const renders = [];
  const contextNode = {scrollTop: 0, scrollHeight: 700, clientHeight: 250,
    replaceChildren(fragment) { renders.push(fragment.children); }};
  const document = {
    createDocumentFragment: () => ({children: [], appendChild(node) { this.children.push(node); }}),
    createElement: () => ({className: "", textContent: ""})
  };
  const renderer = vm.createContext({helpers, ui: {context: contextNode}, document, atBottom: () => false});
  vm.runInContext("let lastContext = null;\n" + renderSource, renderer);
  const blocks = [{role: "context", text: "Header"}, {role: "Human", text: "\n\nHuman> Hello"},
    {role: "Codex", text: "\n\nCodex> Hi."}, {role: "GPT-2", text: "\n\nGPT-2> A reply."}];
  const exact = blocks.map(block => block.text).join("");
  renderer.incoming = {context_text: exact, context_blocks: [{role: "context", text: exact}]};
  vm.runInContext("renderContext(incoming)", renderer);
  assert.equal(renders.length, 1);
  assert.equal(renders[0][0].className, "");
  contextNode.scrollTop = 28;
  renderer.incoming = {context_text: exact, context_blocks: blocks};
  vm.runInContext("renderContext(incoming)", renderer);
  assert.equal(renders.length, 2);
  assert.deepEqual(renders[1].map(node => node.className), ["", "role-human", "role-codex", "role-gpt2"]);
  assert.equal(renders[1].map(node => node.textContent).join(""), exact);
  assert.equal(contextNode.scrollTop, 28, "recoloring must preserve a reader's scroll position");
  vm.runInContext("renderContext(incoming)", renderer);
  assert.equal(renders.length, 2, "an identical snapshot does not rebuild the context");
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

function continuationClient(storage = new Map(), request = async () => { throw new Error("Connection interrupted."); }, uuid = () => "continue-request-0001") {
  const privateUrl = "https://codex-test.gpt2-chat.com/closed-beta/";
  const key = helpers.storageKey(privateUrl);
  if (!storage.has(key + ":session")) storage.set(key + ":session", "chat");
  const requests = [];
  const nodes = new Map();
  const ui = new Proxy({}, {get: (_, name) => {
    if (!nodes.has(name)) nodes.set(name, {textContent: "", value: "", disabled: false, hidden: false,
      style: {}, classList: {toggle() {}}, setAttribute() {}, focus() {}});
    return nodes.get(name);
  }});
  ui.message.value = "An unsent Human draft";
  ui.target.value = "gpt2";
  const source = ["readSaved", "save", "endpoint", "refreshControls", "refreshNotice", "applyState", "acknowledgePending", "acknowledgeContinue", "continueConversation", "boot"]
    .map(name => scripts[1].match(name === "endpoint" ? /^    function endpoint[^\n]+/m :
      new RegExp(`    (?:async )?function ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n    \\}`))[0]).join("\n");
  const waiting = {session_id: "chat", revision: 1, context_text: "Header", context_tokens: 4,
    status: "idle", paused: false, autonomous_turns: 3, next_idle_seconds: null,
    messages: [{id: "human-1", role: "Human", content: "Existing conversation"}]};
  const client = vm.createContext({helpers, ui, key, waiting, crypto: {randomUUID: uuid},
    localStorage: {getItem: name => storage.get(name) ?? null,
      setItem: (name, value) => storage.set(name, value), removeItem: name => storage.delete(name)},
    renderContext() {}, renderHistory() {}, schedule() {},
    idleIndicator: helpers.createIdleWait({now: () => 0, setInterval: () => 1, clearInterval() {}, announce() {},
      render: frame => { ui["idle-wait"].hidden = !frame.visible; ui["wait-rest"].hidden = !frame.waitingForHuman; }}),
    presence: async () => {}, poll: async () => {}, createSession: async () => { throw new Error("Unexpected new chat"); },
    request: async (route, body) => { requests.push({route, body: plain(body)}); return request(route, body); }});
  vm.runInContext(source + "\nlet sessionId=readSaved('session'),state=waiting,pending=null,pendingContinue=helpers.savedContinue(readSaved('pending-continue',true),sessionId),mutation=false,stopped=false,sessionEpoch=0,actionError='',connectionError='',failures=0;", client);
  return {run: code => vm.runInContext(code, client), client, ui, storage, key, requests, waiting};
}

test("Continue keeps the draft and target intact, persists an uncertain ID, and reuses it after reload", async () => {
  const first = continuationClient();
  first.run("refreshControls()");
  assert.equal(first.ui.continue.disabled, false);
  assert.match(html, /id="wait-rest"[\s\S]*?<button id="continue"/);
  await first.run("continueConversation()");
  const saved = JSON.parse(first.storage.get(first.key + ":pending-continue"));
  assert.deepEqual(saved, {session_id: "chat", request_id: "continue-request-0001"});
  assert.equal(first.ui.continue.textContent, "Retry continue");
  assert.match(first.run("actionError"), /delivery is unconfirmed/);
  assert.equal(first.ui.message.value, "An unsent Human draft");
  assert.equal(first.ui.target.value, "gpt2");
  assert.equal(first.run("state.messages.length"), 1);

  let attempt = 0;
  const reloaded = continuationClient(first.storage, async () => {
    if (++attempt === 1) { const error = new Error("Server interrupted"); error.status = 500; throw error; }
    return {...first.waiting, revision: 2, status: "queued", autonomous_turns: 0,
      last_continue_request_id: "continue-request-0001"};
  }, () => { throw new Error("Retry must reuse the persisted ID"); });
  await reloaded.run("boot()");
  assert.equal(reloaded.requests.length, 0, "a reload only reconciles state; it does not post Continue");
  await reloaded.run("continueConversation()");
  assert.equal(JSON.parse(reloaded.storage.get(reloaded.key + ":pending-continue")).request_id, saved.request_id);
  await reloaded.run("continueConversation()");
  assert.equal(reloaded.storage.has(reloaded.key + ":pending-continue"), false);
  assert.equal(reloaded.run("pendingContinue"), null);
  assert.equal(reloaded.run("actionError"), "");
  assert.equal(reloaded.ui.continue.disabled, true);
  assert.equal(reloaded.ui.message.value, "An unsent Human draft");
  assert.equal(reloaded.ui.target.value, "gpt2");
  assert.deepEqual([...first.requests, ...reloaded.requests], Array(3).fill(null).map(() => ({
    route: "sessions/chat/continue", body: {request_id: "continue-request-0001"}})));
  assert.equal(reloaded.run("state.messages.length"), 1, "Continue adds no Human message");
});

test("polling acknowledges a saved Continue even after the agent starts, without posting or touching a draft", async () => {
  const storage = new Map();
  const key = helpers.storageKey("https://codex-test.gpt2-chat.com/closed-beta/");
  storage.set(key + ":pending-continue", JSON.stringify({session_id: "chat", request_id: "continue-request-0002"}));
  const client = continuationClient(storage);
  client.client.received = {...client.waiting, revision: 7, status: "thinking", autonomous_turns: 0,
    last_continue_request_id: "continue-request-0002"};
  client.run("actionError='delivery unconfirmed'; applyState(received)");
  assert.equal(client.run("pendingContinue"), null);
  assert.equal(storage.has(key + ":pending-continue"), false);
  assert.equal(client.run("actionError"), "");
  assert.equal(client.requests.length, 0);
  assert.equal(client.ui.message.value, "An unsent Human draft");
  assert.equal(client.ui.target.value, "gpt2");
  assert.equal(helpers.continueAccepted({session_id: "different", request_id: "continue-request-0002"}, client.client.received), false);
});

test("Continue cannot overlap a busy turn, pause, Human delivery or mutation, and definite rejection clears its ID", async () => {
  const client = continuationClient(undefined, async () => { const error = new Error("A Human message is pending"); error.status = 409; error.accepted = false; throw error; });
  for (const condition of ["state.status='thinking'", "state.status='queued'", "state.status='generating'",
    "state.paused=true", "pending={session_id:'chat',request_id:'human-pending'}", "mutation=true"]) {
    client.run("state={...waiting};pending=null;mutation=false;" + condition + ";refreshControls()");
    assert.equal(client.ui.continue.disabled, true);
    await client.run("continueConversation()");
    assert.equal(client.requests.length, 0);
  }
  client.run("state={...waiting};pending=null;mutation=false");
  await client.run("continueConversation()");
  assert.equal(client.requests.length, 1);
  assert.equal(client.run("pendingContinue"), null);
  assert.equal(client.storage.has(client.key + ":pending-continue"), false);
  assert.equal(client.ui.message.value, "An unsent Human draft");
  assert.equal(client.ui.target.value, "gpt2");
  const value = {session_id: "chat", request_id: "continue-request-0001"};
  assert.equal(helpers.savedContinue(value, "another-chat"), null);
  assert.equal(helpers.savedContinue({...value, request_id: "invalid request"}, "chat"), null);
  assert.equal(helpers.savedContinue({...value, request_id: "continue:request"}, "chat"), null);
  assert.equal(helpers.savedContinue({...value, request_id: "continue.request"}, "chat"), null);
  assert.equal(helpers.savedContinue({session_id: null, request_id: value.request_id}, null), null);
});

test("Continue allows explicitly retryable failures and blocks budget, configuration and unclassified errors", () => {
  const idle = {status: "idle", paused: false, autonomous_turns: 3, next_idle_seconds: null};
  for (const error_code of ["action_timeout", "process_exited", "connection_closed"]) {
    const retryable = {...idle, error: "The turn stopped; the conversation is saved.", error_code, error_retryable: true};
    assert.equal(helpers.canContinue(retryable), true);
    for (const status of ["queued", "thinking", "generating"]) assert.equal(helpers.canContinue({...retryable, status}), false);
    assert.equal(helpers.canContinue({...retryable, paused: true}), false);
    assert.equal(helpers.canContinue({...retryable, autonomous_turns: 2}), false);
    assert.equal(helpers.canContinue({...retryable, next_idle_seconds: 60}), false);
  }
  for (const failure of [
    {error_code: "daily_budget_exceeded", error_retryable: false},
    {error_code: "configuration_error", error_retryable: false},
    {error_code: "action_timeout"}, {},
    {error_code: "action_timeout", error_retryable: "true"},
    {error_code: "action_timeout", error_retryable: 1}
  ]) assert.equal(helpers.canContinue({...idle, error: "The turn could not finish.", ...failure}), false);
});

test("a timeout keeps its exact notice visible beside manual Continue and retries without a Human message", async () => {
  const error = "Codex action timed out; the conversation is saved.";
  const client = continuationClient(undefined, async (_, body) => ({...client.waiting, revision: 2,
    status: "queued", autonomous_turns: 0, error: null, error_code: null, error_retryable: false,
    last_continue_request_id: body.request_id}));
  client.client.failure = {...client.waiting, error, error_code: "action_timeout", error_retryable: true};
  client.run("applyState(failure)");
  await client.run("boot()");
  assert.equal(client.requests.length, 0, "showing or reloading a timeout never retries automatically");
  assert.equal(client.ui.notice.hidden, false);
  assert.equal(client.ui["notice-text"].textContent, error);
  assert.equal(client.ui.status.textContent, "Conversation needs attention");
  assert.equal(client.ui["wait-rest"].hidden, false);
  assert.equal(client.ui.continue.disabled, false);
  assert.equal(client.ui["idle-wait"].hidden, true, "a failure has no autonomous countdown");
  await client.run("continueConversation()");
  assert.deepEqual(client.requests, [{route: "sessions/chat/continue", body: {request_id: "continue-request-0001"}}]);
  assert.equal(client.ui.notice.hidden, true);
  assert.equal(client.ui["wait-rest"].hidden, true);
  assert.equal(client.ui.message.value, "An unsent Human draft");
  assert.equal(client.ui.target.value, "gpt2");
  assert.deepEqual(plain(client.run("state.messages")), client.waiting.messages);
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
  assert.equal(helpers.describeStatus({status: "idle", next_idle_seconds: 59.2}), "Ready · autonomous turn scheduled");
  assert.equal(helpers.describeStatus({status: "idle", next_idle_seconds: null, autonomous_turns: 3}), "Ready · waiting for you");
  assert.equal(helpers.describeStatus({status: "error", error: "budget exhausted"}), "Conversation needs attention");
});

function fakeWaitIndicator(reduced = false) {
  let clock = 1000, serial = 0;
  const timers = new Map(), frames = [], announcements = [];
  const indicator = helpers.createIdleWait({now: () => clock,
    setInterval: (callback, delay) => { assert.equal(delay, 250); const id = ++serial; timers.set(id, callback); return id; },
    clearInterval: id => timers.delete(id), reducedMotion: () => reduced,
    render: frame => frames.push(plain(frame)), announce: text => announcements.push(text)});
  return {indicator, announcements, timers, frame: () => frames.at(-1),
    advance: milliseconds => { clock += milliseconds; for (const callback of [...timers.values()]) callback(); }};
}

const scheduledWait = {session_id: "chat", status: "idle", paused: false, error: null,
  autonomous_turns: 0, next_idle_seconds: 52, idle_wait_seconds: 60, idle_wait_until: 1791234567};

test("the wait drains between polls and stale polls or changed history do not restart it", () => {
  const clock = fakeWaitIndicator();
  clock.indicator.update(scheduledWait);
  assert.equal(clock.frame().label, "Next turn in 0:52");
  assert.equal(clock.timers.size, 1);
  assert.equal(clock.frame().fraction, 52 / 60);
  assert.equal(clock.frame().smooth, false, "a newly shown wait does not animate into place");
  clock.advance(7250);
  assert.equal(clock.frame().label, "Next turn in 0:45");
  assert.equal(clock.frame().fraction, 44.75 / 60);
  assert.equal(clock.frame().smooth, true);
  clock.indicator.update({...scheduledWait, next_idle_seconds: 53, messages: [{role: "Human", content: "history changed"}]});
  assert.equal(clock.frame().label, "Next turn in 0:45", "polling must keep the existing monotonic deadline");
  assert.equal(clock.frame().fraction, 44.75 / 60);
  clock.advance(45000);
  assert.equal(clock.frame().label, "Turn due…");
  assert.equal(clock.frame().fraction, 0);
  assert.equal(clock.timers.size, 0, "a due turn does not leave an active ticker running");
  clock.indicator.update({...scheduledWait, next_idle_seconds: 0});
  assert.equal(clock.timers.size, 0);
  assert.deepEqual(clock.announcements, ["Autonomous turn scheduled.", "Autonomous turn due."]);
});

test("new server wait deadlines and different chats receive fresh full-period anchors", () => {
  const clock = fakeWaitIndicator();
  clock.indicator.update(scheduledWait);
  clock.advance(51000);
  assert.equal(clock.frame().label, "Next turn in 0:01");
  clock.indicator.update({...scheduledWait, idle_wait_until: scheduledWait.idle_wait_until + 120,
    next_idle_seconds: 120, idle_wait_seconds: 120, autonomous_turns: 1});
  assert.equal(clock.frame().label, "Next turn in 2:00");
  assert.equal(clock.frame().fraction, 1);
  assert.equal(clock.frame().smooth, false);
  assert.equal(clock.timers.size, 1, "updating a wait must not duplicate tickers");
  clock.advance(1000);
  assert.equal(clock.frame().label, "Next turn in 1:59");
  clock.indicator.update({...scheduledWait, session_id: "different-chat"});
  assert.equal(clock.frame().label, "Next turn in 0:52");
  clock.indicator.dispose();
  assert.equal(clock.frame().visible, false);
  assert.equal(clock.timers.size, 0);
});

test("busy, paused, disconnected, erroneous and permanent waits stop the indicator", () => {
  const clock = fakeWaitIndicator();
  for (const [state, connected] of [
    [{...scheduledWait, status: "queued"}, true], [{...scheduledWait, status: "thinking"}, true],
    [{...scheduledWait, status: "generating"}, true], [{...scheduledWait, paused: true}, true],
    [{...scheduledWait, error: "budget unavailable"}, true], [scheduledWait, false],
    [{...scheduledWait, next_idle_seconds: null, idle_wait_until: null, idle_wait_seconds: null, autonomous_turns: 3}, true],
    [{...scheduledWait, next_idle_seconds: null}, true], [{...scheduledWait, idle_wait_until: null}, true],
    [null, true]]) {
    clock.indicator.update(scheduledWait);
    assert.equal(clock.timers.size, 1);
    clock.indicator.update(state, connected);
    assert.equal(clock.frame().visible, false);
    assert.equal(clock.timers.size, 0);
    const stopped = clock.frame();
    clock.advance(5000);
    assert.equal(clock.frame(), stopped, "stopping a wait removes all timer-driven rendering");
  }
  clock.indicator.update({...scheduledWait, next_idle_seconds: null, autonomous_turns: 3});
  assert.equal(clock.frame().waitingForHuman, true, "the footer distinguishes waiting for a Human from an autonomous delay");
  clock.indicator.update({...scheduledWait, next_idle_seconds: null, autonomous_turns: 1});
  assert.equal(clock.frame().waitingForHuman, false, "immediately eligible continuation does not pretend to wait for a Human");
});

test("reduced motion retains an accurate clock and phase announcements without bar animation or countdown spam", () => {
  const clock = fakeWaitIndicator(true);
  clock.indicator.update(scheduledWait);
  for (let second = 1; second <= 4; second++) {
    clock.advance(1000);
    assert.equal(clock.frame().label, `Next turn in 0:${52-second}`);
    assert.equal(clock.frame().smooth, false);
  }
  assert.deepEqual(clock.announcements, ["Autonomous turn scheduled."]);
  clock.advance(48000);
  assert.equal(clock.frame().label, "Turn due…");
  assert.equal(clock.frame().smooth, false);
  clock.indicator.update({...scheduledWait, status: "thinking"});
  assert.deepEqual(clock.announcements, ["Autonomous turn scheduled.", "Autonomous turn due.", ""]);
  assert.match(html, /prefers-reduced-motion:[\s\S]*?\.wait-fill\s*\{\s*transition: none !important/);
  assert.equal(helpers.describeStatus({...scheduledWait, next_idle_seconds: 52}),
    helpers.describeStatus({...scheduledWait, next_idle_seconds: 1}), "the main live region also avoids announcing every poll");
});

test("daily allowance uses settled estimates plus unresolved reservations instead of gross historical admissions", () => {
  const budget = {utc_day: "2026-10-05", daily_limit_usd: 10, admitted_usd: 9.82,
    charged_usd: .735, unsettled_reserved_usd: .305, remaining_usd: 9.265,
    reported_estimate_usd: .43, requests: 3, unknown_usage_requests: 1};
  assert.equal(helpers.budgetText(budget), "Daily API allowance: $0.73 / $10.00");
  assert.equal(helpers.budgetText({...budget, charged_usd: 0}), "Daily API allowance: $0.00 / $10.00");
  assert.equal(helpers.budgetText({...budget, admitted_usd: "unknown"}), "Daily API allowance: $0.73 / $10.00");
  assert.match(helpers.budgetTooltip(budget), /2026-10-05 \(UTC\)/);
  assert.match(helpers.budgetTooltip(budget), /Unused reserved allowance is released when complete usage is reported/);
  assert.match(helpers.budgetTooltip(budget), /Known usage estimate: \$0\.43/);
  assert.match(helpers.budgetTooltip(budget), /Unresolved reservations: \$0\.30/);
  assert.match(helpers.budgetTooltip(budget), /Unknown or interrupted requests keep their full reservations/);
  assert.match(helpers.budgetTooltip(budget), /conservative, not an invoice/);
  assert.doesNotMatch(helpers.budgetText(budget), /estimate|reserved|spent/);
});

test("allowance rendering remains compatible with a server that has not exposed settlement fields yet", () => {
  const old = {utc_day: "2026-10-05", daily_limit_usd: 10, admitted_usd: 9.82, reported_estimate_usd: .43};
  assert.equal(helpers.budgetText(old), "Daily API allowance: $9.82 / $10.00");
  assert.match(helpers.budgetTooltip(old), /Conservative admission allowances are shown until settled usage is available/);
  assert.doesNotMatch(helpers.budgetTooltip(old), /Unresolved reservations:/);
  for (const unavailable of [null, {}, {...old, admitted_usd: "unknown"}, {...old, daily_limit_usd: null}]) {
    assert.equal(helpers.budgetText(unavailable), "Daily API allowance: — / $10.00");
  }
});

test("the allowance tooltip explains an operator reset while the visible counter starts at zero", () => {
  const budget = {utc_day: "2026-10-05", daily_limit_usd: 10, charged_usd: 0,
    admitted_usd: 0, unsettled_reserved_usd: 0, reported_estimate_usd: 0,
    reset_at: Date.UTC(2026, 9, 5, 16, 30) / 1000};
  assert.equal(helpers.budgetText(budget), "Daily API allowance: $0.00 / $10.00");
  assert.match(helpers.budgetTooltip(budget), /Counter reset at 2026-10-05T16:30:00\.000Z; earlier requests remain in the audit history/);
  assert.doesNotThrow(() => helpers.budgetTooltip({...budget, reset_at: 1e100}));
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
