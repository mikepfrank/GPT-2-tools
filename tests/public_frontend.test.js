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
  return {
    id, value: "", textContent: "", disabled: false, hidden: false, children: [],
    files: [], listeners: {}, scrollTop: 0, scrollHeight: 100,
    classList: {toggle() {}}, setAttribute() {},
    addEventListener(type, handler) { (this.listeners[type] ||= []).push(handler); },
    appendChild(child) { this.children.push(child); },
    replaceChildren(fragment) {
      this.children = fragment.children;
      this.textContent = fragment.children.map((child) => child.textContent).join("");
    },
    focus() {}, click() {}, remove() {}
  };
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
  for (const id of ["last-prompt", "archive-notes", "storage-warning", "pending-note", "import-file"]) {
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
    addEventListener() {}, prompt: () => "export.json"
  };
  if (options.channels) window.BroadcastChannel = options.channels;
  if (options.throwStorageAccess) {
    Object.defineProperty(window, "sessionStorage", {get() { throw new Error("SecurityError"); }});
    Object.defineProperty(window, "localStorage", {get() { throw new Error("SecurityError"); }});
  }
  const timers = new Set();
  const context = vm.createContext({
    window, document, AbortController, Uint32Array, Blob, URL, console,
    setTimeout(callback, ms) {
      // Accelerate UI polling, but do not accidentally expire fetch aborts.
      const handle = setTimeout(callback, ms === 20000 ? 20000 : Math.min(ms, 2));
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
      return {ok: !response.httpStatus, status: response.httpStatus || 200,
        json: async () => response.payload || response};
    }
  });
  vm.runInContext(source.replace("void initialize();", "globalThis.boot = initialize();"), context);
  return {
    nodes, records, session, local, calls, context, boot: context.boot,
    run: (expression) => vm.runInContext(expression, context),
    stop() { for (const timer of timers) clearTimeout(timer); }
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
