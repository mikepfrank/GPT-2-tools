"use strict";

// Model-free browser lifecycle tests. Run: node --test tests/public_frontend.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const html = fs.readFileSync(path.join(__dirname, "../src/gpt2_local/web/public.html"), "utf8");
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(source); // Also catch syntax errors before any browser mocks.

function state(context = "Context: test", reply = false) {
  const settings = {temperature: 0.8, max_new_tokens: 120, seed: "9223372036854775807"};
  const generation = reply ? {
    prompt_text: "exact last prompt", elapsed_seconds: 3, input_tokens: 15,
    output_tokens: 12, stop_reason: "max_new_tokens", seed: settings.seed,
    generated_token_ids: [1, 2, 3]
  } : null;
  return {
    context_text: context, context_tokens: 15, example_rounds: 2,
    retained_example_rounds: 2, dropped_rounds: 0,
    context_blocks: [{role: "context", text: context}], settings,
    turns: reply ? [{human: "Hello", assistant: "world"}] : [],
    last_generation: generation, notes: [], can_regenerate: reply,
    chat: {
      format: "gpt2-local-chat", schema_version: 2, exported_at: "2026-10-02T00:00:00Z",
      prompt_header: "Context: test", example_rounds: 2, settings: {
        ...settings, top_k: 50, top_p: 0.95, repetition_penalty: 1
      },
      model: {id: "openai-community/gpt2-xl"},
      protocol: {human_label: "Human", model_label: "GPT-2"},
      blocks: reply ? [
        {role: "human", text: "Hello", example: false, generation: null},
        {role: "model", text: "world", example: false, generation,
          previous_responses: [{text: "earlier", generation: {generated_token_ids: [8, 9], seed: "42"}}]}
      ] : [],
      context: {text: context, first_retained_round: 0}, notes: []
    }
  };
}

function storage(values = new Map()) {
  return {getItem: (key) => values.get(key) || null, setItem: (key, value) => values.set(key, value), values};
}

function fakeElement(id) {
  const element = {
    id, value: "", textContent: "", disabled: false, hidden: false, children: [],
    files: [], listeners: {}, scrollTop: 0, scrollHeight: 100, clientHeight: 100,
    classList: {toggle() {}}, setAttribute() {},
    addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); },
    appendChild(child) { this.children.push(child); },
    replaceChildren(fragment) {
      this.children = fragment.children;
      this._textContent = "";
    },
    focus() {}, click() {}, remove() {}
  };
  Object.defineProperty(element, "textContent", {
    get() { return (this._textContent || "") + this.children.map((child) => child.textContent).join(""); },
    set(value) { this._textContent = String(value); this.children = []; }
  });
  return element;
}

function indexedDB(records, failWrites = false) {
  const db = {
    objectStoreNames: {contains: () => true}, createObjectStore() {}, close() {},
    transaction() {
      const transaction = {
        objectStore() {
          return {
            get(key) {
              const request = {};
              queueMicrotask(() => { request.result = structuredClone(records.get(key)); request.onsuccess(); });
              return request;
            },
            put(value, key) {
              queueMicrotask(() => {
                if (failWrites) { transaction.onerror(); return; }
                records.set(key, structuredClone(value));
                transaction.oncomplete();
              });
            }
          };
        }
      };
      return transaction;
    }
  };
  return {open() {
    const request = {result: db};
    queueMicrotask(() => request.onsuccess());
    return request;
  }};
}

function channelFactory() {
  const channels = new Set();
  return class BroadcastChannel {
    constructor() { channels.add(this); }
    postMessage(data) {
      for (const channel of channels) {
        if (channel !== this) queueMicrotask(() => channel.onmessage && channel.onmessage({data}));
      }
    }
    close() { channels.delete(this); }
  };
}

function browser(options = {}) {
  const records = options.records || new Map();
  const session = options.session || storage();
  const local = options.local || storage();
  const nodes = new Map();
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(match[1], fakeElement(match[1]));
  for (const id of ["last-prompt", "archive-notes", "storage-warning", "pending-note", "import-file", "progress-info"]) {
    nodes.get(id).hidden = true;
  }
  nodes.get("temperature").value = "0.8";
  nodes.get("seed").value = "";
  nodes.get("reply-limit").value = "120";
  nodes.get("example-rounds").value = "2";
  const calls = [];
  let uid = 0;
  const document = {
    body: fakeElement("body"), visibilityState: "visible",
    getElementById: (id) => nodes.get(id),
    createElement: () => fakeElement("span"),
    createDocumentFragment: () => fakeElement("fragment"),
    addEventListener() {}
  };
  const window = {
    sessionStorage: session, localStorage: local, indexedDB: options.noStorage ? undefined : indexedDB(records, options.failWrites),
    crypto: {randomUUID: () => "tab-" + (options.id || "A") + "-" + (++uid)},
    listeners: {},
    addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); },
    prompt: () => "export.json"
  };
  if (options.channels) window.BroadcastChannel = options.channels;
  if (options.throwStorageAccess) {
    Object.defineProperty(window, "sessionStorage", {get() { throw new Error("SecurityError"); }});
    Object.defineProperty(window, "localStorage", {get() { throw new Error("SecurityError"); }});
  }
  const timers = new Set();
  const context = vm.createContext({
    window, document, AbortController, Uint32Array, Uint8Array, TextDecoder, Blob, URL, console,
    setTimeout(callback, ms) {
      // Accelerate UI polling, but do not accidentally expire fetch aborts.
      const handle = setTimeout(callback, ms === 20000 ? (options.watchdogMs || 20000) : Math.min(ms, 2));
      timers.add(handle);
      return handle;
    },
    clearTimeout(handle) { clearTimeout(handle); timers.delete(handle); },
    async fetch(url, params = {}) {
      const body = params.body ? JSON.parse(params.body) : undefined;
      calls.push({url, body, method: params.method || "GET"});
      const response = options.respond ? await options.respond(url, body, params) :
        url === "/api/health" ? {limits: {max_reply_tokens: 256}} : state();
      if (response instanceof Error) throw response;
      if (response.streamResponse) return response.streamResponse;
      return {ok: !response.httpStatus, status: response.httpStatus || 200,
        json: async () => response.payload || response};
    }
  });
  vm.runInContext(source.replace("void initialize();", "globalThis.boot = initialize();"), context);
  return {
    nodes, records, session, local, calls, context, boot: context.boot,
    run: (expression) => vm.runInContext(expression, context),
    stop() {
      vm.runInContext("pageLeaving = true; if (activeStreamAbort) activeStreamAbort.abort();", context);
      for (const timer of timers) clearTimeout(timer);
    }
  };
}

function snapshot(initial, overrides = {}) {
  return {
    version: 1, updated_at: "2026-10-02T00:00:00Z", state: initial, draft: "unsent draft",
    controls: {temperature: "0.65", seed: "9223372036854775807", replyLimit: "80", exampleRounds: "1"},
    pending: null, ...overrides
  };
}

async function until(condition) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (condition()) return;
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
  throw new Error("Browser did not reach the expected state.");
}

test("new chat stores a full archive without a server session ID", async () => {
  const b = browser();
  try {
    await b.boot;
    const stored = [...b.records.values()][0];
    assert.deepEqual(stored.state.chat, state().chat);
    assert.equal(stored.draft, "");
    assert.equal(b.calls.filter((call) => call.url === "/api/new").length, 1);
    assert.equal(b.calls.some((call) => call.url.startsWith("/api/state")), false);
    assert.equal(b.nodes.get("seed").value, "9223372036854775807");
  } finally { b.stop(); }
});

test("reload after a server restart restores transcript, draft, controls and revalidates by archive", async () => {
  const original = state("full retained context", true);
  const records = new Map([["saved", snapshot(original)]]);
  const b = browser({
    records, session: storage(new Map([["gpt2-public-chat-tab-v1", "saved"]])),
    respond: (url, body) => url === "/api/health" ? {limits: {max_reply_tokens: 256}} :
      url === "/api/import" ? (assert.deepEqual(body.chat, original.chat), structuredClone(original)) : new Error("unexpected")
  });
  try {
    await b.boot;
    assert.equal(b.nodes.get("context").textContent, "full retained context");
    assert.equal(b.nodes.get("message").value, "unsent draft");
    assert.equal(b.nodes.get("temperature").value, "0.65");
    assert.equal(b.nodes.get("reply-limit").value, "80");
    assert.equal(b.calls.some((call) => call.url === "/api/new"), false);
    assert.equal(b.run("committed.chat.blocks[1].previous_responses[0].generation.seed"), "42");
  } finally { b.stop(); }
});

test("offline reload and repeated offline reload preserve local transcript and export without network", async () => {
  const original = state("<script>alert('model')</script>", true);
  const records = new Map([["saved", snapshot(original)]]);
  const session = storage(new Map([["gpt2-public-chat-tab-v1", "saved"]]));
  const local = storage();
  const make = () => browser({records, session, local, respond: () => new Error("offline")});
  const b = make();
  try {
    await b.boot;
    assert.equal(b.nodes.get("context").textContent, original.context_text);
    assert.equal(b.nodes.get("export-chat").disabled, false);
    const before = b.calls.length;
    const archive = JSON.parse(b.run("JSON.stringify(exportArchive())"));
    assert.equal(b.calls.length, before);
    assert.deepEqual(archive.blocks, original.chat.blocks);
    assert.equal(archive.settings.temperature, 0.65);
    assert.equal(archive.settings.seed, "9223372036854775807");
    assert.notEqual(archive.exported_at, original.chat.exported_at);
    assert.equal(b.run("committed.chat.exported_at"), original.chat.exported_at);
  } finally { b.stop(); }
  const reloaded = make();
  try {
    await reloaded.boot;
    assert.equal(reloaded.nodes.get("context").textContent, original.context_text);
    assert.equal(reloaded.nodes.get("message").value, "unsent draft");
  } finally { reloaded.stop(); }
});

test("full-range seeds remain decimal strings and blank export uses the active seed", async () => {
  const b = browser();
  try {
    await b.boot;
    b.nodes.get("seed").value = "0009223372036854775807";
    assert.equal(b.run("settings().seed"), "9223372036854775807");
    b.nodes.get("seed").value = "9223372036854775808";
    assert.throws(() => b.run("settings()"), /Seed must be from/);
    b.nodes.get("seed").value = "";
    assert.equal(b.run("exportArchive().settings.seed"), "9223372036854775807");
    b.nodes.get("reply-limit").value = "1023";
    assert.throws(() => b.run("settings()"), /1 to 256/);
    assert.equal(b.run("exportArchive().settings.max_new_tokens"), 1023);
  } finally { b.stop(); }
});

test("successful export commits future controls, so a later blank seed uses the newly exported active seed", async () => {
  const original = state("retained reply", true);
  const b = browser({respond: (url) => url === "/api/health" ?
    {limits: {max_reply_tokens: 256}} : original});
  try {
    await b.boot;
    b.nodes.get("seed").value = "9223372036854775806";
    b.nodes.get("temperature").value = "0.7";
    await b.run("exportChat()");
    b.nodes.get("seed").value = "";
    assert.equal(b.run("exportArchive().settings.seed"), "9223372036854775806");
    assert.equal(b.run("committed.settings.seed"), "9223372036854775806");
    assert.equal(b.run("committed.chat.settings.temperature"), 0.7);
    assert.deepEqual(JSON.parse(b.run("JSON.stringify(committed.chat.blocks)")), original.chat.blocks);
    assert.deepEqual([...b.records.values()].at(-1).state.chat.blocks, original.chat.blocks);
    assert.equal([...b.records.values()].at(-1).state.chat.settings.seed, "9223372036854775806");
  } finally { b.stop(); }
});

test("queue rejection rolls the preview back and preserves draft and committed archive", async () => {
  const original = state("committed context", true);
  const b = browser({respond: (url) => {
    if (url === "/api/health") return {limits: {max_reply_tokens: 256}};
    if (url === "/api/new") return original;
    if (url === "/api/preview") return {...state("pending context"), chat: undefined};
    if (url === "/api/message") return {httpStatus: 429, payload: {error: "Queue is full."}};
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "keep this draft";
    await b.run("generate()");
    assert.equal(b.nodes.get("context").textContent, "committed context");
    assert.equal(b.nodes.get("message").value, "keep this draft");
    assert.equal(b.run("pending"), null);
    assert.equal(b.run("busy"), false);
    assert.deepEqual(JSON.parse(b.run("JSON.stringify(committed.chat)")), original.chat);
    assert.match(b.nodes.get("status").textContent, /Queue is full/);
  } finally { b.stop(); }
});

test("explicitly unaccepted HTTP503 queue rejection rolls back without an unknown-outcome lock", async () => {
  const original = state("committed context", true);
  const b = browser({respond: (url) => {
    if (url === "/api/health") return {limits: {max_reply_tokens: 256}};
    if (url === "/api/new") return original;
    if (url === "/api/preview") return state("pending context");
    if (url === "/api/message") return {
      httpStatus: 503, payload: {error: "The reply queue is full; try again shortly", accepted: false}
    };
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "keep this draft";
    await b.run("generate()");
    assert.equal(b.nodes.get("context").textContent, original.context_text);
    assert.equal(b.nodes.get("message").value, "keep this draft");
    assert.equal(b.run("pending"), null);
    assert.equal(b.run("busy"), false);
    assert.equal(b.nodes.get("pending-note").hidden, true);
    assert.match(b.nodes.get("status").textContent, /reply queue is full/);
  } finally { b.stop(); }
});

test("ambiguous HTTP500 submission remains blocked and does not automatically retry", async () => {
  const original = state("committed context", true);
  const b = browser({respond: (url) => {
    if (url === "/api/health") return {limits: {max_reply_tokens: 256}};
    if (url === "/api/new") return original;
    if (url === "/api/preview") return state("pending context");
    if (url === "/api/message") return {httpStatus: 500, payload: {error: "Unexpected server failure"}};
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "preserved draft";
    await b.run("generate()");
    assert.equal(b.nodes.get("context").textContent, original.context_text);
    assert.equal(b.nodes.get("message").value, "preserved draft");
    assert.equal(b.run("pending.job_id"), null);
    assert.equal(b.run("busy"), true);
    assert.equal(b.nodes.get("pending-note").hidden, false);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
  } finally { b.stop(); }
});

test("near-limit Unicode archives export within the 4MiB import limit without dropping history", async () => {
  const b = browser();
  try {
    await b.boot;
    // Lots of short Unicode records make pretty indentation materially larger
    // than compact UTF-8. JS string length alone would undercount these bytes.
    const archive = state("retained reply", true).chat;
    const entry = {role: "human", text: "汉字🙂".repeat(10), example: false, generation: null};
    const compactEntryBytes = new Blob([JSON.stringify(entry) + ","]).size;
    const baseBytes = new Blob([JSON.stringify(archive)]).size;
    const count = Math.floor((4 * 1024 * 1024 - baseBytes - 200) / compactEntryBytes);
    archive.blocks = Array.from({length: count}, () => structuredClone(entry));
    assert.ok(new Blob([JSON.stringify(archive)]).size <= 4 * 1024 * 1024);
    assert.ok(new Blob([JSON.stringify(archive, null, 2)]).size > 4 * 1024 * 1024);
    b.context.largeArchive = archive;
    const contents = b.run("exportContents(largeArchive)");
    assert.ok(new Blob([contents]).size <= 4 * 1024 * 1024);
    assert.deepEqual(JSON.parse(contents), archive);
    const captured = [];
    b.context.window.showSaveFilePicker = async () => ({
      name: "large.json", createWritable: async () => ({
        write: async (data) => captured.push(data), close: async () => {}
      })
    });
    b.run("nativeSaveAvailable = true; committed.chat = largeArchive;");
    await b.run("exportChat()");
    assert.equal(captured.length, 1);
    assert.ok(new Blob([captured[0]]).size <= 4 * 1024 * 1024);
    const importCallsBefore = b.calls.filter((call) => call.url === "/api/import").length;
    b.context.file = {size: new Blob([captured[0]]).size, name: "large.json", text: async () => captured[0]};
    await b.run("importChat(file)");
    assert.equal(b.calls.filter((call) => call.url === "/api/import").length, importCallsBefore + 1);
    assert.equal(b.calls.at(-1).body.chat.blocks.length, count);
    b.run("largeArchive.blocks[0].text += 'x'.repeat(4 * 1024 * 1024 - new Blob([JSON.stringify(largeArchive)]).size + 1);");
    assert.throws(() => b.run("exportContents(largeArchive)"), /exceeds the 4 MiB/);
  } finally { b.stop(); }
});

test("queued generation polls to completion and commits exactly once", async () => {
  let polls = 0;
  const final = state("accepted reply", true);
  const b = browser({respond: (url) => {
    if (url === "/api/health") return {limits: {max_reply_tokens: 256}};
    if (url === "/api/new") return state("original");
    if (url === "/api/preview") return state("pending human message");
    if (url === "/api/message") return {job_id: "job-one", status: "queued", position: 2};
    if (url === "/api/jobs/job-one") {
      polls++;
      return polls === 1 ? {status: "queued", position: 2} :
        polls === 2 ? {status: "running"} : {status: "complete", state: final};
    }
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "Hello";
    await b.run("generate()");
    assert.equal(polls, 3);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
    assert.equal(b.nodes.get("context").textContent, final.context_text);
    assert.equal(b.nodes.get("message").value, "");
    assert.equal(b.run("pending"), null);
    assert.match(b.nodes.get("status").textContent, /4.00 tokens\/s/);
    assert.deepEqual([...b.records.values()].at(-1).state.chat, final.chat);
  } finally { b.stop(); }
});

test("reloaded pending generation polls its saved job without re-submitting", async () => {
  const original = state("before queue");
  const final = state("after recovered generation", true);
  const records = new Map([["saved", snapshot(original, {
    draft: "changed since submission",
    pending: {job_id: "existing-job", message: "submitted draft", regenerated: false}
  })]]);
  const b = browser({
    records, session: storage(new Map([["gpt2-public-chat-tab-v1", "saved"]])),
    respond: (url) => url === "/api/health" ? {limits: {max_reply_tokens: 256}} :
      url === "/api/jobs/existing-job" ? {status: "complete", state: final} : new Error("Unexpected POST")
  });
  try {
    await b.boot;
    assert.equal(b.nodes.get("context").textContent, final.context_text);
    assert.equal(b.nodes.get("message").value, "changed since submission");
    assert.equal(b.calls.filter((call) => call.method === "POST").length, 0);
    assert.equal(b.run("pending"), null);
  } finally { b.stop(); }
});

test("transient polling errors retry only GET; expired jobs preserve the original draft", async () => {
  let polls = 0;
  const original = state("before queue");
  const records = new Map([["saved", snapshot(original, {
    pending: {job_id: "existing-job", message: "unsent draft", regenerated: false}
  })]]);
  const b = browser({
    records, session: storage(new Map([["gpt2-public-chat-tab-v1", "saved"]])),
    respond: (url) => {
      if (url === "/api/health") return {limits: {max_reply_tokens: 256}};
      if (url === "/api/jobs/existing-job") return ++polls === 1 ? new Error("offline") :
        {httpStatus: 404, payload: {error: "Job unavailable."}};
      throw new Error("Unexpected POST");
    }
  });
  try {
    await b.boot;
    assert.equal(polls, 2);
    assert.equal(b.calls.filter((call) => call.method === "POST").length, 0);
    assert.equal(b.nodes.get("context").textContent, original.context_text);
    assert.equal(b.nodes.get("message").value, "unsent draft");
    assert.equal(b.run("busy"), false);
    assert.match(b.nodes.get("status").textContent, /expired or the server restarted/);
  } finally { b.stop(); }
});

test("lost POST acknowledgement remains blocked across reload and never auto retries", async () => {
  const original = state("before uncertain request");
  const b = browser({respond: (url) => {
    if (url === "/api/health") return {limits: {max_reply_tokens: 256}};
    if (url === "/api/new") return original;
    if (url === "/api/preview") return state("pending");
    if (url === "/api/message") return new Error("lost response");
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "preserved draft";
    await b.run("generate()");
    assert.equal(b.run("pending.job_id"), null);
    assert.equal(b.run("busy"), true);
    assert.equal(b.nodes.get("pending-note").hidden, false);
    assert.equal(b.nodes.get("context").textContent, original.context_text);
    assert.equal(b.nodes.get("export-chat").disabled, false);
  } finally { b.stop(); }
  const reloaded = browser({
    records: b.records, session: b.session, local: b.local,
    respond: (url) => url === "/api/health" ? {limits: {max_reply_tokens: 256}} : new Error("Must not retry")
  });
  try {
    await reloaded.boot;
    assert.equal(reloaded.run("busy"), true);
    assert.equal(reloaded.nodes.get("message").value, "preserved draft");
    assert.equal(reloaded.calls.filter((call) => call.method === "POST").length, 0);
    reloaded.nodes.get("dismiss-pending").listeners.click[0]();
    assert.equal(reloaded.run("busy"), false);
    assert.equal(reloaded.run("pending"), null);
  } finally { reloaded.stop(); }
});

test("failed regeneration and failed import leave transcript, history, controls and draft intact", async () => {
  const original = state("original reply", true);
  const b = browser({respond: (url) => {
    if (url === "/api/health") return {limits: {max_reply_tokens: 256}};
    if (url === "/api/new") return original;
    if (url === "/api/regenerate-preview") return {...state("exact previous input"), settings: {...original.settings, seed: "0"}};
    if (url === "/api/regenerate") return {job_id: "failed-job", status: "running"};
    if (url === "/api/jobs/failed-job") return {status: "failed", error: "CPU timeout"};
    if (url === "/api/import") return {httpStatus: 400, payload: {error: "Invalid archive"}};
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "do not lose this";
    await b.run("generate(true)");
    assert.equal(b.nodes.get("context").textContent, original.context_text);
    assert.equal(b.nodes.get("seed").value, "9223372036854775807");
    assert.equal(b.nodes.get("message").value, "do not lose this");
    b.context.file = {size: 2, name: "bad.json", text: async () => "{}"};
    await b.run("importChat(file)");
    assert.equal(b.nodes.get("message").value, "do not lose this");
    assert.deepEqual(JSON.parse(b.run("JSON.stringify(committed.chat)")), original.chat);
  } finally { b.stop(); }
});

test("duplicate tabs fork durable snapshots and neither tab overwrites the other's draft", async () => {
  const channels = channelFactory();
  const records = new Map();
  const local = storage();
  const first = browser({records, local, channels, id: "first"});
  try {
    await first.boot;
    first.nodes.get("message").value = "first tab draft";
    await first.run("persist()");
    const originalId = first.session.getItem("gpt2-public-chat-tab-v1");
    const second = browser({
      records, local, channels, id: "second",
      session: storage(new Map([["gpt2-public-chat-tab-v1", originalId]]))
    });
    try {
      await second.boot;
      const forkId = second.session.getItem("gpt2-public-chat-tab-v1");
      assert.notEqual(forkId, originalId);
      assert.equal(second.nodes.get("message").value, "first tab draft");
      second.nodes.get("message").value = "second tab draft";
      await second.run("persist()");
      assert.equal(records.get(originalId).draft, "first tab draft");
      assert.equal(records.get(forkId).draft, "second tab draft");
    } finally { second.stop(); }
  } finally { first.stop(); }
});

test("unavailable, denied and full browser storage show a persistent export warning", async () => {
  for (const options of [{noStorage: true}, {throwStorageAccess: true}, {failWrites: true}]) {
    const b = browser(options);
    try {
      await b.boot;
      assert.equal(b.nodes.get("storage-warning").hidden, false);
      assert.match(b.nodes.get("storage-warning").textContent, /Export JSON regularly/);
      assert.equal(b.nodes.get("export-chat").disabled, false);
      assert.equal(b.run("exportArchive().settings.seed"), "9223372036854775807");
    } finally { b.stop(); }
  }
});

function wire(type, seq, values = {}, ending = "\n") {
  const payload = {version: 1, seq, ...values};
  return "id: " + seq + ending + "event: " + type + ending +
    "data: " + JSON.stringify(payload) + ending + ending;
}

function progress(seq, text, count = seq) {
  return wire("progress", seq, {
    status: "running", position: 0, generated_tokens: count, reply_text: text
  });
}

function streamed(parts, onRead) {
  let index = 0;
  return {streamResponse: {
    ok: true, status: 200, headers: {get: () => "text/event-stream; charset=utf-8"},
    body: {getReader: () => ({
      async read() {
        if (onRead) await onRead(index);
        if (index === parts.length) return {done: true};
        const item = typeof parts[index] === "function" ? await parts[index++]() : parts[index++];
        return {done: false, value: typeof item === "string" ? new TextEncoder().encode(item) : item};
      },
      cancel: async () => {}, releaseLock() {}
    })}
  }};
}

function displayFor(value) {
  return {
    context_text: value.context_text, context_blocks: structuredClone(value.context_blocks),
    context_tokens: value.context_tokens, retained_example_rounds: value.retained_example_rounds,
    dropped_rounds: value.dropped_rounds
  };
}

test("SSE parser accepts every UTF-8 byte split, CRLF/LF/CR framing and comments", async () => {
  const b = browser();
  try {
    await b.boot;
    const expected = {version: 1, seq: 7, status: "running", position: 0,
      generated_tokens: 7, reply_text: " 漢字🙂\n<script>literal</script>"};
    for (const ending of ["\n", "\r\n", "\r"]) {
      const bytes = new TextEncoder().encode(": heartbeat" + ending + ending +
        wire("progress", 7, expected, ending));
      for (let split = 0; split <= bytes.length; split++) {
        b.run("globalThis.events = []; globalThis.parser = createSseParser((event) => events.push(event), 4096);");
        b.context.first = bytes.slice(0, split);
        b.context.second = bytes.slice(split);
        b.run("parser.push(first); parser.push(second); parser.finish();");
        assert.deepEqual(JSON.parse(b.run("JSON.stringify(events)")), [{type: "progress", payload: expected}]);
      }
    }
  } finally { b.stop(); }
});

test("SSE byte bound applies per unfinished frame, allowing multiple valid frames in an oversized read chunk", async () => {
  const b = browser();
  try {
    await b.boot;
    const text = "x".repeat(600000);
    b.context.bytes = new TextEncoder().encode(progress(1, text, 1) + progress(2, text, 2));
    assert.ok(b.context.bytes.byteLength > 1024 * 1024);
    b.run("globalThis.events = []; globalThis.parser = createSseParser((event) => events.push(event)); parser.push(bytes);");
    assert.equal(b.run("events.length"), 2);
    b.context.bytes = new TextEncoder().encode(progress(3, "漢".repeat(400000), 3));
    assert.throws(() => b.run("parser.push(bytes)"), /1 MiB/);
    b.context.bytes = new TextEncoder().encode("event: complete\nid: 4\ndata: {\"version\":1,\"seq\":5}\n\n");
    b.run("globalThis.parser = createSseParser(() => {});");
    assert.throws(() => b.run("parser.push(bytes)"), /invalid event/);
    b.context.bytes = new Uint8Array([100, 97, 116, 97, 58, 32, 0xff, 10]);
    b.run("globalThis.parser = createSseParser(() => {});");
    assert.throws(() => b.run("parser.push(bytes)"));
  } finally { b.stop(); }
});

test("streamed reply remains provisional and exportable archive unchanged until final GET commits", async () => {
  const original = state("committed");
  const preview = state("exact submitted prompt");
  const final = state("canonical completed reply", true);
  let release;
  const terminal = new Promise((resolve) => { release = resolve; });
  const b = browser({respond: (url, body) => {
    if (url === "/api/health") return {capabilities: {streaming: true}, limits: {max_reply_tokens: 256}};
    if (url === "/api/new") return original;
    if (url === "/api/preview") return preview;
    if (url === "/api/message") {
      const recorded = [...b.records.values()].at(-1).pending;
      assert.equal(recorded.job_id, null);
      assert.deepEqual(recorded.submitted_settings, body.settings);
      assert.deepEqual(Object.keys(recorded.display).sort(),
        ["context_blocks", "context_text", "context_tokens", "dropped_rounds", "retained_example_rounds"]);
      assert.equal(recorded.submitted_settings.temperature, 0.6);
      assert.equal(recorded.submitted_settings.max_new_tokens, 16);
      return {job_id: "stream-one", status: "running"};
    }
    if (url === "/api/jobs/stream-one/events") {
      assert.equal([...b.records.values()].at(-1).pending.job_id, "stream-one");
      return streamed([progress(1, " partial 漢🙂", 1), () => terminal]);
    }
    if (url === "/api/jobs/stream-one") return {status: "complete", state: final};
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "Hello";
    b.nodes.get("temperature").value = "0.6";
    b.nodes.get("reply-limit").value = "16";
    const running = b.run("generate()");
    await until(() => b.run("provisionalText") === " partial 漢🙂");
    assert.equal(b.nodes.get("context").textContent, preview.context_text + " partial 漢🙂");
    assert.equal(b.run("committed.context_text"), original.context_text);
    assert.deepEqual(JSON.parse(b.run("JSON.stringify(exportArchive().blocks)")), original.chat.blocks);
    assert.deepEqual([...b.records.values()].at(-1).state.chat, original.chat);
    assert.equal("reply_text" in [...b.records.values()].at(-1).pending, false);
    assert.match(b.nodes.get("progress-info").textContent, /not yet saved/);
    release(wire("complete", 2));
    await running;
    assert.equal(b.nodes.get("context").textContent, final.context_text);
    assert.equal(b.run("pending"), null);
    assert.equal(b.nodes.get("progress-info").hidden, true);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
    assert.equal(b.calls.filter((call) => call.url === "/api/jobs/stream-one").length, 1);
  } finally { b.stop(); }
});

test("reconnect preserves partial text, uses advisory cursor, ignores duplicates and stale revisions, and retries terminal GET only", async () => {
  let streams = 0;
  let polls = 0;
  const final = state("reconnected final", true);
  const b = browser({respond: (url, body, params) => {
    if (url === "/api/health") return {capabilities: {streaming: true}};
    if (url === "/api/new") return state("before");
    if (url === "/api/preview") return state("prepared");
    if (url === "/api/message") return {job_id: "reconnect", status: "running"};
    if (url === "/api/jobs/reconnect/events") {
      if (++streams === 1) return streamed([progress(2, " A", 2)]);
      assert.equal(params.headers["Last-Event-ID"], "2");
      assert.equal(b.nodes.get("context").textContent, "prepared A");
      return streamed([
        progress(1, "stale", 1), progress(2, "duplicate altered text", 2),
        progress(4, " AB", 3), wire("complete", 5)
      ]);
    }
    if (url === "/api/jobs/reconnect") {
      if (++polls === 1) return {status: "running", progress: {
        version: 1, seq: 2, status: "running", position: 0, generated_tokens: 2, reply_text: " A"
      }};
      if (polls === 2) {
        assert.equal(b.run("provisionalText"), " AB");
        return new Error("temporary result retrieval failure");
      }
      return {status: "complete", state: final};
    }
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "Hello";
    await b.run("generate()");
    assert.equal(streams, 2);
    assert.equal(polls, 3);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
    assert.equal(b.nodes.get("context").textContent, final.context_text);
  } finally { b.stop(); }
});

test("reload accepts current snapshot despite an older saved cursor and freezes input separately from UI controls", async () => {
  const original = state("committed old reply", true);
  const prepared = state("recorded regeneration input");
  const records = new Map([["saved", snapshot(original, {
    pending: {
      job_id: "restore-stream", regenerated: true, message: "unsent draft", seq: 999,
      display: {...displayFor(prepared), turns: ["must not be retained"]},
      submitted_settings: {temperature: 0.7, max_new_tokens: 32, seed: "9223372036854775807"},
      regeneration_settings: {temperature: 0.7, max_new_tokens: 32, seed: "0"}
    }
  })]]);
  let release;
  const terminal = new Promise((resolve) => { release = resolve; });
  const b = browser({
    records, session: storage(new Map([["gpt2-public-chat-tab-v1", "saved"]])),
    respond: (url, body, params) => {
      if (url === "/api/health") return {capabilities: {streaming: true}};
      if (url === "/api/jobs/restore-stream/events") {
        assert.equal(params.headers["Last-Event-ID"], undefined);
        return streamed([progress(2, " alternative", 2), () => terminal]);
      }
      if (url === "/api/jobs/restore-stream") return {status: "failed", error: "deadline"};
      throw new Error("Unexpected POST");
    }
  });
  try {
    await until(() => b.run("provisionalText") === " alternative");
    assert.equal(b.nodes.get("context").textContent, prepared.context_text + " alternative");
    assert.equal(b.nodes.get("temperature").value, "0.65");
    assert.match(b.nodes.get("progress-info").textContent, /seed 0/);
    assert.equal(b.nodes.get("seed").value, "9223372036854775807");
    assert.equal("turns" in [...records.values()].at(-1).pending.display, false);
    release(wire("failed", 3));
    await b.boot;
    assert.equal(b.nodes.get("context").textContent, original.context_text);
    assert.equal(b.nodes.get("message").value, "unsent draft");
    assert.deepEqual(JSON.parse(b.run("JSON.stringify(committed.chat)")), original.chat);
    assert.equal(b.nodes.get("seed").value, "9223372036854775807");
    assert.equal(b.calls.filter((call) => call.method === "POST").length, 0);
  } finally { b.stop(); }
});

test("legacy or invalid pending display remains final-only even when streaming is available", async () => {
  for (const display of [undefined, {context_text: "wrong", context_tokens: 1}]) {
    let polls = 0;
    const original = state("committed original");
    const records = new Map([["saved", snapshot(original, {
      pending: {job_id: "legacy-job", regenerated: false, message: "unsent draft", display}
    })]]);
    const b = browser({
      records, session: storage(new Map([["gpt2-public-chat-tab-v1", "saved"]])),
      respond: (url) => {
        if (url === "/api/health") return {capabilities: {streaming: true}};
        if (url === "/api/jobs/legacy-job") {
          assert.equal(b.nodes.get("context").textContent, original.context_text);
          return ++polls === 1 ? {status: "running", progress: {
            version: 1, seq: 1, status: "running", position: 0, generated_tokens: 1, reply_text: "wrong-context reply"
          }} : {status: "complete", state: state("legacy final", true)};
        }
        throw new Error("Must not open a stream");
      }
    });
    try {
      await b.boot;
      assert.equal(polls, 2);
      assert.equal(b.calls.some((call) => call.url.endsWith("/events")), false);
      assert.equal(b.run("pending"), null);
    } finally { b.stop(); }
  }
});

test("subscription capacity and malformed frames fall back to progress polling for the same job", async () => {
  for (const badStream of [
    {httpStatus: 503, payload: {error: "subscription capacity"}},
    streamed(["data: not-json\nevent: progress\n\n"]),
    streamed([": " + "x".repeat(1024 * 1024) + "\n\n"])
  ]) {
    let polls = 0;
    let streams = 0;
    const b = browser({respond: (url) => {
      if (url === "/api/health") return {capabilities: {streaming: true}};
      if (url === "/api/new") return state("committed");
      if (url === "/api/preview") return state("prepared");
      if (url === "/api/message") return {job_id: "fallback", status: "running"};
      if (url === "/api/jobs/fallback/events") { streams++; return badStream; }
      if (url === "/api/jobs/fallback") {
        if (++polls === 1) return {status: "running", progress: {
          version: 1, seq: 3, status: "running", position: 0, generated_tokens: 3, reply_text: " <script>literal</script>"
        }};
        assert.equal(b.nodes.get("context").textContent, "prepared <script>literal</script>");
        return {status: "complete", state: state("fallback final", true)};
      }
      throw new Error("Unexpected " + url);
    }});
    try {
      await b.boot;
      b.nodes.get("message").value = "Hello";
      await b.run("generate()");
      assert.equal(streams, 1);
      assert.equal(polls, 2);
      assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
      assert.equal(b.run("pending"), null);
    } finally { b.stop(); }
  }
});

test("stream idle watchdog aborts stalled subscription and retrieves the same final result", async () => {
  let stalled = false;
  let streamAborted = false;
  const b = browser({watchdogMs: 10, respond: (url, body, params) => {
    if (url === "/api/health") return {capabilities: {streaming: true}};
    if (url === "/api/new") return state("committed");
    if (url === "/api/preview") return state("prepared");
    if (url === "/api/message") return {job_id: "stall", status: "running"};
    if (url === "/api/jobs/stall/events") return {streamResponse: {
      ok: true, status: 200, headers: {get: () => "text/event-stream"}, body: {getReader: () => ({
        read: () => new Promise((resolve, reject) => {
          stalled = true;
          params.signal.addEventListener("abort", () => {
            streamAborted = true; reject(new Error("aborted"));
          }, {once: true});
        }),
        cancel: async () => {}, releaseLock() {}
      })}
    }};
    if (url === "/api/jobs/stall") return {status: "complete", state: state("after idle watchdog", true)};
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "Hello";
    await b.run("generate()");
    assert.equal(stalled, true);
    assert.equal(streamAborted, true);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
    assert.equal(b.nodes.get("context").textContent, "after idle watchdog");
  } finally { b.stop(); }
});

test("stream missing job restores the committed view and draft without resubmission", async () => {
  const original = state("committed");
  const b = browser({respond: (url) => {
    if (url === "/api/health") return {capabilities: {streaming: true}};
    if (url === "/api/new") return original;
    if (url === "/api/preview") return state("prepared");
    if (url === "/api/message") return {job_id: "expired-stream", status: "running"};
    if (url === "/api/jobs/expired-stream/events") return streamed([progress(1, " unfinished", 1)]);
    if (url === "/api/jobs/expired-stream") return {httpStatus: 404, payload: {error: "Reply expired"}};
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "keep draft";
    await b.run("generate()");
    assert.equal(b.nodes.get("context").textContent, original.context_text);
    assert.equal(b.nodes.get("message").value, "keep draft");
    assert.equal(b.run("pending"), null);
    assert.equal(b.nodes.get("progress-info").hidden, true);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
  } finally { b.stop(); }
});

test("heartbeat bytes reset idle watchdog during a stream longer than the idle limit", async () => {
  let index = 0;
  let interrupted = false;
  const b = browser({watchdogMs: 30, respond: (url, body, params) => {
    if (url === "/api/health") return {capabilities: {streaming: true}};
    if (url === "/api/new") return state("committed");
    if (url === "/api/preview") return state("prepared");
    if (url === "/api/message") return {job_id: "heartbeat", status: "running"};
    if (url === "/api/jobs/heartbeat/events") return {streamResponse: {
      ok: true, status: 200, headers: {get: () => "text/event-stream"}, body: {getReader: () => ({
        read: () => new Promise((resolve, reject) => {
          const onAbort = () => { interrupted = true; clearTimeout(timer); reject(new Error("aborted")); };
          const timer = setTimeout(() => {
            params.signal.removeEventListener("abort", onAbort);
            resolve({done: false, value: new TextEncoder().encode(++index < 5 ? ": heartbeat\n\n" : wire("complete", 1))});
          }, 12);
          params.signal.addEventListener("abort", onAbort, {once: true});
        }),
        cancel: async () => {}, releaseLock() {}
      })}
    }};
    if (url === "/api/jobs/heartbeat") return {status: "complete", state: state("after heartbeats", true)};
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "Hello";
    await b.run("generate()");
    assert.equal(index, 5);
    assert.equal(interrupted, false);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
  } finally { b.stop(); }
});

test("pagehide aborts only the subscription, preserving the acknowledged job for reload", async () => {
  let waiting = false;
  let cancelled = false;
  const b = browser({respond: (url, body, params) => {
    if (url === "/api/health") return {capabilities: {streaming: true}};
    if (url === "/api/new") return state("committed");
    if (url === "/api/preview") return state("prepared");
    if (url === "/api/message") return {job_id: "pagehide", status: "running"};
    if (url === "/api/jobs/pagehide/events") return {streamResponse: {
      ok: true, status: 200, headers: {get: () => "text/event-stream"}, body: {getReader: () => ({
        read: () => new Promise((resolve, reject) => {
          waiting = true;
          params.signal.addEventListener("abort", () => { cancelled = true; reject(new Error("page hidden")); }, {once: true});
        }),
        cancel: async () => {}, releaseLock() {}
      })}
    }};
    throw new Error("Must not submit or poll after pagehide");
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "preserved message";
    const running = b.run("generate()");
    await until(() => waiting);
    b.context.window.listeners.pagehide[0]();
    await running;
    await b.run("saveChain");
    assert.equal(cancelled, true);
    assert.equal(b.run("pending.job_id"), "pagehide");
    assert.equal([...b.records.values()].at(-1).pending.job_id, "pagehide");
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
    assert.equal(b.calls.some((call) => call.url === "/api/jobs/pagehide"), false);
  } finally { b.stop(); }
});

test("provisional text follows the reader only when already near the bottom", async () => {
  const b = browser();
  try {
    await b.boot;
    b.context.displayFixture = displayFor(state("prepared"));
    b.run("pending = {job_id: 'scroll', display: displayFixture, submitted_settings: {max_new_tokens: 32}};");
    b.nodes.get("context").scrollHeight = 1000;
    b.nodes.get("context").clientHeight = 100;
    b.nodes.get("context").scrollTop = 0;
    b.context.progressFixture = {version: 1, seq: 1, status: "running", position: 0, generated_tokens: 1, reply_text: " one"};
    b.run("applyProgress(progressFixture, 'scroll');");
    assert.equal(b.nodes.get("context").scrollTop, 0);
    b.nodes.get("context").scrollTop = 900;
    b.context.progressFixture = {...b.context.progressFixture, seq: 2, generated_tokens: 2, reply_text: " one two"};
    b.run("applyProgress(progressFixture, 'scroll');");
    assert.equal(b.nodes.get("context").scrollTop, 1000);
  } finally { b.stop(); }
});

test("back-forward cache restoration resumes the same acknowledged job without overlapping subscriptions", async () => {
  let streams = 0;
  let waiting = false;
  const b = browser({respond: (url, body, params) => {
    if (url === "/api/health") return {capabilities: {streaming: true}};
    if (url === "/api/new") return state("committed");
    if (url === "/api/preview") return state("prepared");
    if (url === "/api/message") return {job_id: "back-forward", status: "running"};
    if (url === "/api/jobs/back-forward/events") {
      if (++streams > 1) return streamed([wire("complete", 1)]);
      return {streamResponse: {
        ok: true, status: 200, headers: {get: () => "text/event-stream"}, body: {getReader: () => ({
          read: () => new Promise((resolve, reject) => {
            waiting = true;
            params.signal.addEventListener("abort", () => reject(new Error("left page")), {once: true});
          }),
          cancel: async () => {}, releaseLock() {}
        })}
      }};
    }
    if (url === "/api/jobs/back-forward") return {status: "complete", state: state("back-forward final", true)};
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "Hello";
    const firstMonitor = b.run("generate()");
    await until(() => waiting);
    b.context.window.listeners.pagehide[0]();
    await firstMonitor;
    b.context.window.listeners.pageshow[0]({persisted: true});
    await until(() => b.run("pending") === null);
    assert.equal(streams, 2);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
    assert.equal(b.nodes.get("context").textContent, "back-forward final");
  } finally { b.stop(); }
});

test("terminal result timeout includes receiving the JSON body and retries only that result GET", async () => {
  let results = 0;
  let abortedBody = false;
  const b = browser({watchdogMs: 10, respond: (url, body, params) => {
    if (url === "/api/health") return {capabilities: {streaming: true}};
    if (url === "/api/new") return state("committed");
    if (url === "/api/preview") return state("prepared");
    if (url === "/api/message") return {job_id: "result-body", status: "running"};
    if (url === "/api/jobs/result-body/events") return streamed([wire("complete", 1)]);
    if (url === "/api/jobs/result-body") {
      if (++results === 1) return {streamResponse: {
        ok: true, status: 200, json: () => new Promise((resolve, reject) => {
          params.signal.addEventListener("abort", () => {
            abortedBody = true; reject(new Error("incomplete response body"));
          }, {once: true});
        })
      }};
      return {status: "complete", state: state("complete body retrieved", true)};
    }
    throw new Error("Unexpected " + url);
  }});
  try {
    await b.boot;
    b.nodes.get("message").value = "Hello";
    await b.run("generate()");
    assert.equal(abortedBody, true);
    assert.equal(results, 2);
    assert.equal(b.calls.filter((call) => call.url.endsWith("/events")).length, 1);
    assert.equal(b.calls.filter((call) => call.url === "/api/message").length, 1);
    assert.equal(b.nodes.get("context").textContent, "complete body retrieved");
  } finally { b.stop(); }
});
