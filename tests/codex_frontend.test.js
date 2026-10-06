"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../src/gpt2_local/web/codex.html"), "utf8");
const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(match => match[1]);
const context = vm.createContext({URL, Blob});
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
  vm.runInContext("let mutation=false,timer=null,actionError='',infoNotice='',sessionId=null,sessionEpoch=0,pollSequence=0,state=null,historySignature='',lastContext=null,pending=null,pendingContinue=null,connectionError='',failures=0;\n"+createSource, creator);
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
    generationIndicator: helpers.createGenerationActivity({announce() {},
      render: frame => { ui["gpt2-wait"].hidden = !frame.visible; }}),
    presence: async () => {}, poll: async () => {}, createSession: async () => { throw new Error("Unexpected new chat"); },
    request: async (route, body) => { requests.push({route, body: plain(body)}); return request(route, body); }});
  vm.runInContext(source + "\nlet sessionId=readSaved('session'),state=waiting,pending=null,pendingContinue=helpers.savedContinue(readSaved('pending-continue',true),sessionId),mutation=false,stopped=false,sessionEpoch=0,actionError='',infoNotice='',connectionError='',failures=0;", client);
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

test("Continue can shorten a scheduled wait without a Human message, draft change or exhausted-wait label", async () => {
  const client = continuationClient(undefined, async (_, body) => ({...client.waiting, revision: 2,
    status: "queued", autonomous_turns: 1, last_continue_request_id: body.request_id}));
  client.client.scheduled = {...client.waiting, autonomous_turns: 1, next_idle_seconds: 90,
    idle_wait_seconds: 120, idle_wait_until: 1791234567};
  client.run("applyState(scheduled)");
  assert.equal(client.ui["idle-wait"].hidden, false);
  assert.equal(client.ui["wait-rest"].hidden, true);
  assert.equal(client.ui.continue.hidden, false);
  assert.equal(client.ui.continue.disabled, false);
  await client.run("continueConversation()");
  assert.deepEqual(client.requests, [{route: "sessions/chat/continue", body: {request_id: "continue-request-0001"}}]);
  assert.equal(client.run("state.autonomous_turns"), 1);
  assert.equal(client.ui.continue.hidden, true);
  assert.equal(client.ui["idle-wait"].hidden, true);
  assert.equal(client.ui.message.value, "An unsent Human draft");
  assert.equal(client.ui.target.value, "gpt2");
  assert.deepEqual(plain(client.run("state.messages")), client.waiting.messages);
  for (const invalid of [
    {paused: true}, {connected: false}, {status: "generating"}, {error: "Budget exceeded", error_retryable: true},
    {autonomous_turns: 3}, {idle_wait_until: null}, {idle_wait_seconds: 0}, {next_idle_seconds: -1}
  ]) assert.equal(helpers.canContinue({...client.client.scheduled, ...invalid}), false);
  assert.equal(helpers.canContinue({...client.waiting, connected: false}), false);
  assert.equal([...html.matchAll(/id="continue"/g)].length, 1, "both wait modes share one button and request ID flow");
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

const preparingGpt2 = {...scheduledWait, status: "generating", active: {role: "GPT-2", content: " "},
  messages: [{id: "human-1", role: "Human", content: "Hello"}]};

test("GPT-2 preparation stays distinct from visible generation and an indeterminate indicator never claims percent complete", () => {
  const frames = [], announcements = [];
  const indicator = helpers.createGenerationActivity({render: frame => frames.push(plain(frame)),
    announce: text => announcements.push(text)});
  for (const content of ["", " ", "\n\t", null]) indicator.update({...preparingGpt2, active: {role: "GPT-2", content}});
  assert.equal(frames.at(-1).phase, "preparing");
  assert.equal(frames.at(-1).label, "GPT-2 · preparing reply…");
  assert.match(frames.at(-1).title, /first text can take a while/);
  assert.equal(frames.at(-1).animated, true);
  assert.equal("fraction" in frames.at(-1), false);
  assert.equal("percent" in frames.at(-1), false);
  assert.deepEqual(announcements, ["GPT-2 is preparing a reply. First text can take a while."]);
  for (const content of [" A", " A reply", " "]) indicator.update({...preparingGpt2, active: {role: "GPT-2", content}});
  assert.equal(frames.at(-1).phase, "generating", "decoder buffering does not rewind an already visible reply to preparation");
  assert.deepEqual(announcements, ["GPT-2 is preparing a reply. First text can take a while.", "GPT-2 is generating its reply."]);
  indicator.update({...preparingGpt2, messages: [...preparingGpt2.messages, {id: "gpt2-1", role: "GPT-2", content: "A reply"}]});
  assert.equal(frames.at(-1).phase, "preparing", "a new query starts its own preparation phase");
  indicator.update({...preparingGpt2, session_id: "another-chat"});
  assert.equal(frames.at(-1).phase, "preparing");
  indicator.dispose();
  assert.equal(frames.at(-1).visible, false);
  assert.equal(frames.at(-1).animated, false);
});

test("GPT-2 activity hides for stale, paused, failed and disconnected work and stays static under reduced motion", () => {
  for (const [state, connected] of [
    [null, true], [preparingGpt2, false], [{...preparingGpt2, connected: false}, true], [{...preparingGpt2, paused: true}, true],
    [{...preparingGpt2, error: "The turn stopped"}, true], [{...preparingGpt2, status: "idle"}, true],
    [{...preparingGpt2, status: "thinking"}, true], [{...preparingGpt2, status: "queued"}, true],
    [{...preparingGpt2, active: null}, true], [{...preparingGpt2, active: {role: "Codex", content: "Hello"}}, true]
  ]) assert.deepEqual(plain(helpers.generationActivity(state, connected)), {visible: false, phase: "off", animated: false});
  assert.equal(helpers.generationActivity(preparingGpt2, true, true).visible, true);
  assert.equal(helpers.generationActivity(preparingGpt2, true, true).animated, false);
  assert.match(html, /prefers-reduced-motion:[\s\S]*?\.gpt2-wait\.is-animated \.gpt2-sweep\s*\{\s*animation: none;/);
  assert.match(html, /id="gpt2-announcement"[^>]*aria-live="polite"[^>]*aria-atomic="true"/);
});

function realBrowserClient(initial, reducedMotion = false) {
  const nodes = new Map(), listeners = new Map(), storage = new Map(), requests = [], createdElements = [];
  const timers = new Map(); let serial = 0, incoming = initial, fail = false, responseError = null;
  function node() {
    const classes = new Set(), events = new Map(), attributes = new Map();
    let text = "";
    return {hidden: false, disabled: false, value: "", checked: false, files: [], clicks: 0, style: {}, children: [], textWrites: 0,
      scrollTop: 0, scrollHeight: 1200, clientHeight: 300,
      get textContent() { return text; }, set textContent(value) { text = value; this.textWrites++; },
      classList: {toggle(name, on) { if (on) classes.add(name); else classes.delete(name); }, contains: name => classes.has(name)},
      setAttribute: (name, value) => attributes.set(name, value), getAttribute: name => attributes.get(name),
      appendChild(child) { child.parentNode = this; this.children.push(child); }, append(...children) { for (const child of children) this.appendChild(child); },
      replaceChildren(...children) { this.children = children; }, addEventListener: (name, callback) => events.set(name, callback),
      remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(child => child !== this); this.parentNode = null; this.removed = true; },
      focus() {}, click() { this.clicks++; this.clickedWhileAttached = Boolean(this.parentNode); }, events};
  }
  for (const match of html.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const element = node(); element.hidden = /\bhidden\b/.test(match[0]); nodes.set(match[1], element);
  }
  const href = "https://codex-test.gpt2-chat.com/closed-beta/";
  const body = node();
  storage.set(helpers.storageKey(href) + ":session", initial.session_id);
  const timer = callback => { const id = ++serial; timers.set(id, callback); return id; };
  const client = vm.createContext({URL, Blob, console, location: {href}, performance: {now: () => 0},
    crypto: {randomUUID: () => "request-for-browser-test"},
    document: {body, getElementById: id => nodes.get(id),
      createElement: tagName => { const element = node(); element.tagName = tagName; createdElements.push(element); return element; },
      createDocumentFragment: node},
    window: {matchMedia: () => ({matches: reducedMotion}), addEventListener: (name, callback) => listeners.set(name, callback)},
    navigator: {sendBeacon: () => true}, setInterval: timer, setTimeout: timer,
    clearInterval: id => timers.delete(id), clearTimeout: id => timers.delete(id),
    localStorage: {getItem: name => storage.get(name) ?? null, setItem: (name, value) => storage.set(name, value), removeItem: name => storage.delete(name)},
    fetch: async (url, options = {}) => {
      requests.push({route: new URL(url).pathname, method: options.method || "GET", body: options.body ? JSON.parse(options.body) : undefined});
      if (new URL(url).pathname.endsWith("/presence")) return {ok: true, json: async () => ({})};
      if (fail) throw new Error("Offline");
      if (responseError) return {ok: false, status: responseError.status, json: async () => ({error: responseError.message})};
      return {ok: true, json: async () => incoming};
    }});
  vm.runInContext(scripts[0], client);
  vm.runInContext(scripts[1].replace(/    boot\(\);\s*$/, "    globalThis.bootReady = boot();"), client);
  return {ui: Object.fromEntries(nodes), listeners, storage, requests, createdElements, body, ready: client.bootReady,
    run: code => vm.runInContext(code, client), receive: value => { incoming = value; }, disconnect: () => { fail = true; },
    reject: (status, message) => { responseError = {status, message}; }};
}

test("real browser scripts wire preparation, streamed text, timer Continue and waiting phases without touching draft or scroll", async () => {
  const client = realBrowserClient({...preparingGpt2, revision: 1, context_text: "Header\n\nGPT-2>", context_tokens: 8});
  await client.ready;
  const ui = client.ui;
  assert.equal(ui["gpt2-wait"].hidden, false);
  assert.equal(ui["gpt2-wait"].classList.contains("is-animated"), true);
  assert.equal(ui["gpt2-wait-label"].textContent, "GPT-2 · preparing reply…");
  assert.equal(ui["idle-wait"].hidden, true);
  assert.equal(ui["wait-rest"].hidden, true);
  assert.equal(ui.continue.hidden, true);
  ui.message.value = "Unsent draft"; ui.target.value = "gpt2"; ui.context.scrollTop = 42; ui.history.scrollTop = 77;
  const textWrites = ui["gpt2-announcement"].textWrites;
  await client.run("poll()");
  assert.equal(ui["gpt2-announcement"].textWrites, textWrites, "repeated polls do not rewrite a live announcement");
  client.receive({...preparingGpt2, revision: 2, context_text: "Header\n\nGPT-2> A reply", context_tokens: 10,
    active: {role: "GPT-2", content: " A reply"}});
  await client.run("poll()");
  assert.equal(ui["gpt2-wait-label"].textContent, "GPT-2 · generating…");
  assert.equal(ui["gpt2-announcement"].textContent, "GPT-2 is generating its reply.");
  assert.equal(ui.context.scrollTop, 42); assert.equal(ui.history.scrollTop, 77);
  client.receive({...scheduledWait, revision: 3, context_text: "Header", context_tokens: 4, messages: preparingGpt2.messages});
  await client.run("poll()");
  assert.equal(ui["gpt2-wait"].hidden, true);
  assert.equal(ui["gpt2-wait"].classList.contains("is-animated"), false);
  assert.equal(ui["idle-wait"].hidden, false);
  assert.equal(ui.continue.hidden, false); assert.equal(ui.continue.disabled, false);
  assert.equal(ui["wait-rest"].hidden, true);
  client.receive({...scheduledWait, revision: 4, context_text: "Header", context_tokens: 4, messages: preparingGpt2.messages,
    autonomous_turns: 3, next_idle_seconds: null, idle_wait_until: null, idle_wait_seconds: null});
  await client.run("poll()");
  assert.equal(ui["idle-wait"].hidden, true); assert.equal(ui["wait-rest"].hidden, false);
  assert.equal(ui.continue.hidden, false); assert.equal(ui.continue.disabled, false);
  assert.equal(ui.message.value, "Unsent draft"); assert.equal(ui.target.value, "gpt2");
});

test("real browser generation animation stops on error, pause, new chat, disconnect and pagehide; reduced motion keeps its text", async () => {
  const initial = {...preparingGpt2, revision: 1, context_text: "Header", context_tokens: 4};
  const client = realBrowserClient(initial, true); await client.ready;
  const ui = client.ui;
  assert.equal(ui["gpt2-wait"].hidden, false);
  assert.equal(ui["gpt2-wait"].classList.contains("is-animated"), false);
  assert.equal(ui["gpt2-wait-label"].textContent, "GPT-2 · preparing reply…");
  let revision = 1;
  for (const stopping of [{error: "Generation stopped"}, {paused: true}, {connected: false}, {status: "thinking"}, {active: null}]) {
    client.receive({...initial, revision: ++revision, ...stopping}); await client.run("poll()");
    assert.equal(ui["gpt2-wait"].hidden, true);
    assert.equal(ui["gpt2-wait"].classList.contains("is-animated"), false);
    client.receive({...initial, revision: ++revision}); await client.run("poll()");
    assert.equal(ui["gpt2-wait"].hidden, false);
  }
  client.receive({...initial, session_id: "new-chat", revision: 1, status: "idle", active: null, messages: []});
  await client.run("createSession()");
  assert.equal(ui["gpt2-wait"].hidden, true);
  client.receive({...initial, session_id: "new-chat", revision: 2}); await client.run("poll()");
  assert.equal(ui["gpt2-wait"].hidden, false);
  client.disconnect(); await client.run("poll()");
  assert.equal(ui["gpt2-wait"].hidden, true);
  assert.equal(ui["gpt2-wait"].classList.contains("is-animated"), false);
  assert.equal(ui.continue.hidden, true);
  assert.match(ui.status.textContent, /Connection interrupted/);
  client.listeners.get("pagehide")();
  assert.equal(ui["gpt2-wait"].hidden, true);
  assert.equal(ui["gpt2-announcement"].textContent, "");
  assert.equal(ui.import.disabled, true);
  assert.equal(ui["import-file"].disabled, true);
});

const exportedArchive = {format: "gpt2-codex-chat", version: 1, session_id: "0123456789abcdef0123456789abcdef",
  conversation: {header: "An imported header.", context_start: 0,
    messages: [{role: "Human", content: " Hello\nwith a line break", generation: null},
      {role: "GPT-2", content: " A reply", generation: {output_token_ids: [1, 2], output_tokens: 2}}]},
  metadata: [{id: 1, created_at: 1791234567, example: false}, {id: 2, created_at: 1791234568, example: false}],
  settings: {temperature: .7, max_new_tokens: 120}, codex_thread_id: "old-native-id"};
const jsonFile = (document, prefix = "") => {
  const text = prefix + JSON.stringify(document);
  return {size: new Blob([text]).size, text: async () => text};
};

test("imported JSON is bounded, parsed as data and checked for the supported archive format", async () => {
  assert.deepEqual(plain(await helpers.readImportFile(jsonFile(exportedArchive))), exportedArchive);
  assert.deepEqual(plain(await helpers.readImportFile(jsonFile(exportedArchive, "\uFEFF"))), exportedArchive);
  let read = false;
  await assert.rejects(helpers.readImportFile({size: helpers.importMaxBytes + 1, text: async () => { read = true; return "{}"; }}), /up to 10 MiB/);
  assert.equal(read, false, "oversized files are rejected before they are read into memory");
  for (const file of [null, {size: -1}, {size: 100}, {size: Infinity, text() {}}])
    await assert.rejects(helpers.readImportFile(file), /Choose an exported chat JSON file/);
  await assert.rejects(helpers.readImportFile({size: 0, text: async () => ""}), /file is empty/);
  await assert.rejects(helpers.readImportFile({size: 1, text: async () => "{"}), /not valid JSON/);
  await assert.rejects(helpers.readImportFile({size: 10, text: async () => { throw new Error("file-system details"); }}), /could not be read/);
  for (const document of [null, [], 1, {}, {...exportedArchive, format: "another-app"}, {...exportedArchive, version: 2}])
    await assert.rejects(helpers.readImportFile(jsonFile(document)), /not a supported/);
  const hugeArchive = {...exportedArchive, metadata: {text: "x".repeat(helpers.importMaxBytes)}};
  await assert.rejects(helpers.readImportFile({size: 10, text: async () => JSON.stringify(hugeArchive)}), /up to 10 MiB/,
    "the serialized request is bounded as well as the source file");
  const evidence = {...exportedArchive, conversation: {...exportedArchive.conversation,
    messages: exportedArchive.conversation.messages.map(message => message.role === "GPT-2" ?
      {...message, generation: {...message.generation, provenance: {overflow: 0}}} : message)}};
  const overflow = JSON.stringify(evidence).replace('"overflow":0', '"overflow":1e999');
  await assert.rejects(helpers.readImportFile({size: new Blob([overflow]).size, text: async () => overflow}), /outside the browser's exact numeric range/,
    "overflowing numbers cannot silently become null when the archive is serialized for upload");
  const unsafeInteger = JSON.stringify(evidence).replace('"overflow":0', '"overflow":9007199254740993');
  await assert.rejects(helpers.readImportFile({size: new Blob([unsafeInteger]).size, text: async () => unsafeInteger}), /outside the browser's exact numeric range/,
    "64-bit provenance integers cannot silently round when the archive is parsed in the browser");
  const preciseSeeds = {...exportedArchive, conversation: {...exportedArchive.conversation,
    messages: exportedArchive.conversation.messages.map(message => message.role === "GPT-2" ?
      {...message, generation: {...message.generation, seed: "9223372036854775807", requested_seed: "9223372036854775807",
        provenance: {largest_safe_integer: Number.MAX_SAFE_INTEGER}}} : message)}};
  assert.deepEqual(plain(await helpers.readImportFile(jsonFile(preciseSeeds))), preciseSeeds,
    "decimal seed strings and safe integer evidence retain exact values");
});

test("the actual Export click requests a same-origin HTTP attachment while retaining its user gesture and the selected chat", async () => {
  const original = {...scheduledWait, revision: 1, context_text: "Current context", messages: [{role: "Human", content: " A message"}]};
  const client = realBrowserClient(original); await client.ready;
  const ui = client.ui;
  ui.message.value = "Keep this draft"; ui.target.value = "gpt2"; ui.context.scrollTop = 12; ui.history.scrollTop = 34;
  const priorRequests = client.requests.length;
  client.run("globalThis.Blob=function(){throw new Error('Export must not create a Blob URL')}");
  ui.export.events.get("click")();
  const anchors = client.createdElements.filter(element => element.tagName === "a");
  assert.equal(anchors.length, 1);
  const anchor = anchors[0];
  assert.equal(anchor.href, "https://codex-test.gpt2-chat.com/closed-beta/api/sessions/chat/export/download");
  assert.match(anchor.download, /^gpt2-codex-chat-\d{4}-\d{2}-\d{2}\.json$/);
  assert.equal(anchor.hidden, true);
  assert.equal(anchor.clicks, 1); assert.equal(anchor.clickedWhileAttached, true);
  assert.equal(anchor.removed, true); assert.equal(client.body.children.length, 0);
  assert.equal(client.requests.length, priorRequests, "Export uses the browser attachment route directly, without an async fetch or duplicate request");
  assert.equal(client.run("sessionId"), "chat");
  assert.deepEqual(plain(client.run("state.messages")), original.messages);
  assert.equal(ui.message.value, "Keep this draft"); assert.equal(ui.target.value, "gpt2");
  assert.equal(ui.context.scrollTop, 12); assert.equal(ui.history.scrollTop, 34);
  assert.equal(ui["notice-text"].textContent, "Export download requested.");
  assert.equal(ui.notice.classList.contains("error"), false);
  for (const condition of ["mutation=true", "stopped=true", "state=null"]) {
    client.run("mutation=false;stopped=false;" + condition + ";refreshControls()");
    assert.equal(ui.export.disabled, true);
    ui.export.events.get("click")();
    assert.equal(client.createdElements.filter(element => element.tagName === "a").length, 1);
  }
});

test("the actual import button opens the JSON picker, loads a separate paused chat and preserves draft and target", async () => {
  const original = {...scheduledWait, status: "idle", autonomous_turns: 3, next_idle_seconds: null,
    session_id: "original", revision: 1, context_text: "Original context", messages: [{id: "old-1", role: "Human", content: "Original message"}]};
  const imported = {...original, session_id: "imported", revision: 1, paused: true, active: null,
    context_text: "An imported header.\n\nHuman> Hello\nwith a line break\n\nGPT-2> A reply",
    messages: exportedArchive.conversation.messages.map((message, index) => ({...message, id: "import-" + index})),
    settings: exportedArchive.settings};
  const client = realBrowserClient(original); await client.ready;
  const ui = client.ui;
  ui.message.value = "Keep this draft"; ui.target.value = "gpt2";
  assert.equal(ui.import.disabled, false);
  assert.match(html, /<button id="export"[^>]*>[\s\S]*?<button id="import"/);
  assert.match(html, /<input id="import-file"[^>]*type="file"[^>]*accept="\.json,application\/json"[^>]*hidden/);
  ui["import-file"].value = "previous-file.json";
  ui.import.events.get("click")();
  assert.equal(ui["import-file"].value, "", "the same filename can be selected again");
  assert.equal(ui["import-file"].clicks, 1);
  let finishRead;
  ui["import-file"].files = [{size: jsonFile(exportedArchive).size, text: () => new Promise(resolve => { finishRead = resolve; })}];
  ui["import-file"].value = "archive.json";
  const loading = ui["import-file"].events.get("change")();
  assert.equal(ui.import.disabled, true);
  assert.equal(client.run("sessionId"), "original", "the current chat stays selected during validation and upload");
  client.receive(imported); finishRead(JSON.stringify(exportedArchive)); await loading;
  const uploads = client.requests.filter(request => request.route.endsWith("/sessions/import"));
  assert.deepEqual(uploads.map(({method, body}) => ({method, body})), [{method: "POST", body: {archive: exportedArchive}}]);
  assert.equal(client.run("sessionId"), "imported");
  assert.equal(client.run("state.paused"), true);
  assert.equal(client.run("state.status"), "idle");
  assert.deepEqual(plain(client.run("state.messages")), imported.messages);
  assert.equal(client.run("state.context_text"), imported.context_text);
  assert.equal(ui.message.value, "Keep this draft"); assert.equal(ui.target.value, "gpt2");
  assert.equal(ui.temperature.value, "0.7");
  assert.equal(ui.pause.textContent, "Resume activity");
  assert.equal(ui.notice.hidden, false); assert.equal(ui.notice.classList.contains("error"), false);
  assert.match(ui["notice-text"].textContent, /Chat imported\. Activity is paused/);
  assert.equal(ui["import-file"].value, "");
  assert.equal(client.requests.some(request => /\/(message|continue|control)$/.test(request.route)), false,
    "import never sends a message, resumes activity or starts a model turn");
  assert.deepEqual(client.requests.filter(request => request.route.endsWith("/presence")).map(request => request.body),
    [{connected: true}, {connected: false}, {connected: true}]);
});

test("cancelled, invalid, rejected and interrupted imports preserve the existing conversation, draft and scroll without automatic retry", async () => {
  const original = {...scheduledWait, session_id: "original", revision: 1, status: "idle", autonomous_turns: 3,
    next_idle_seconds: null, context_text: "Original context", messages: [{role: "Human", content: "Original message"}]};
  for (const failure of ["cancel", "invalid-json", "oversized", "server-rejection", "network", "incomplete-response", "malformed-messages"]) {
    const client = realBrowserClient(original); await client.ready;
    const ui = client.ui;
    ui.message.value = "Keep my draft"; ui.target.value = "codex";
    ui.context.scrollTop = 27; ui.history.scrollTop = 56;
    if (failure === "server-rejection") client.reject(400, "This archive has invalid message metadata");
    if (failure === "network") client.disconnect();
    if (failure === "incomplete-response") client.receive({session_id: "new", status: "idle", paused: true});
    if (failure === "malformed-messages") client.receive({...original, session_id: "new", paused: true, messages: [null]});
    ui["import-file"].files = failure === "cancel" ? [] : [failure === "invalid-json" ? {size: 1, text: async () => "{"} :
      failure === "oversized" ? {size: helpers.importMaxBytes + 1, text: async () => { throw new Error("must not read"); }} : jsonFile(exportedArchive)];
    await ui["import-file"].events.get("change")();
    assert.equal(client.run("sessionId"), "original", failure);
    assert.deepEqual(plain(client.run("state.messages")), original.messages, failure);
    assert.equal(client.run("state.context_text"), original.context_text, failure);
    assert.equal(ui.message.value, "Keep my draft", failure); assert.equal(ui.target.value, "codex", failure);
    assert.equal(ui.context.scrollTop, 27, failure); assert.equal(ui.history.scrollTop, 56, failure);
    if (failure !== "cancel") assert.equal(ui.notice.classList.contains("error"), true, failure);
    if (failure === "server-rejection") assert.equal(ui["notice-text"].textContent, "This archive has invalid message metadata");
    const uploads = client.requests.filter(request => request.route.endsWith("/sessions/import"));
    assert.equal(uploads.length, ["cancel", "invalid-json", "oversized"].includes(failure) ? 0 : 1);
    assert.equal(client.requests.filter(request => request.route.endsWith("/presence")).length, 1,
      "only initial presence is sent; an unsuccessful import never disconnects the selected chat");
    await client.run("poll()");
    assert.equal(client.requests.filter(request => request.route.endsWith("/sessions/import")).length, uploads.length,
      "reconnection checks the selected session and never posts an import again");
  }
});

test("imports cannot overlap active work, uncertain deliveries, a mutation or a stopped page", async () => {
  const original = {...scheduledWait, revision: 1, context_text: "Original context", messages: []};
  const client = realBrowserClient(original); await client.ready;
  const ui = client.ui;
  ui["import-file"].files = [jsonFile(exportedArchive)];
  for (const condition of ["state.status='queued'", "state.status='thinking'", "state.status='generating'",
    "pending={request_id:'human-1'}", "pendingContinue={request_id:'continue-1'}", "mutation=true", "stopped=true"]) {
    client.run("state.status='idle';pending=null;pendingContinue=null;mutation=false;stopped=false;" + condition + ";refreshControls()");
    assert.equal(ui.import.disabled, true, condition);
    ui.import.events.get("click")();
    await ui["import-file"].events.get("change")();
    assert.equal(ui["import-file"].clicks, 0, condition);
    assert.equal(client.requests.filter(request => request.route.endsWith("/sessions/import")).length, 0, condition);
  }
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
