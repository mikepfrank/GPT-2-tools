# Streaming replies: implementation and previews

Status: **implemented; laptop and hosted previews verified and owner-accepted;
production streaming active, October 3 CDT / October 4 UTC,
2026**.
The initial streaming suites passed 197 Python tests and 30 JavaScript tests. Real-model
greedy/sample parity, near-context generation, and browser streaming/reload
checks passed. This work lives in `.worktrees/gpt2-streaming`, on
`codex/gpt2-streaming`, based on public-service commit
`e5971db3b79110cb62e92575a2054a87ba5b98c4`. The owner chose a **local laptop
preview first** and accepted it. The owner has now authorized a temporary
GPT-2-only production pause for the hosted preview, accepted the hosted
incremental display, and explicitly requested production rollout.

The later automatic seed advancement change is now deployed as the accepted
frozen `78ebbdc` release. See [SEED-UPDATE.md](SEED-UPDATE.md) for current
deployment/rollback and `VERIFICATION.md` for its tests and live seed sequence.

The goal is to display a reply as GPT-2 generates it while preserving the
current model, sampling behavior, chat archives, and recovery rules. Keep the
pinned GPT-2 XL checkpoint, CPU FP32, one resident model, and one inference
worker. Idle browser chats need no resident KV cache. The active generation
continues to own its existing cache, then releases it as it does today.

Streaming can shorten the wait for visible reply text. Queueing, prompt
prefill, and decoding still take time; total throughput is a separate
measurement. Unicode characters and possible speaker boundaries sometimes
require several tokens before any additional text is safe to show.

## Preserve the existing generation path

Add an optional, keyword-only observer to `Gpt2Runner.generate_result()` in
[`runtime.py`](../src/gpt2_local/runtime.py):

```python
@dataclass(frozen=True)
class SampledToken:
    token_id: int
    output_tokens: int
    elapsed_seconds: float

generate_result(..., on_token: Callable[[SampledToken], None] | None = None)
```

When present, a small custom Transformers `BaseStreamer` adapter skips the
initial prompt batch and reports each subsequently generated token. Batch size
remains one. Its elapsed time starts immediately before `model.generate()` and
includes prefill. Keep the same generation arguments, stopping criteria, seed,
raw output IDs, final decoding, accounting checks, and `GenerationResult`.
Calls with no observer keep their existing argument behavior.

The installed Transformers 5.18.0 CPU path evaluates stopping criteria before
calling the token streamer. Normal completion calls `streamer.end()`, but an
exception, including the cooperative deadline, can bypass that notification
and the latest token callback. The worker must own terminal status; streamer
completion alone cannot commit a chat. The built-in text streamer buffers
words, so use the raw-token hook for per-token progress.

Thread this optional observer through `_generate_reply()`, `ChatSession.reply()`
and `ChatSession.regenerate()` in [`chat.py`](../src/gpt2_local/chat.py).
`_PublicRunner` forwards it while retaining its tokenizer lock and deadline.
The observer updates bounded job progress and notifies subscribers. It performs
no socket writes, RNG calls, model calls, or thread-count changes. An observer
error fails the job through the existing worker error path.

## Decode a safe visible prefix

Keep presentation decoding in a small, independently tested helper. Use the
pinned Tokenizers 0.23.2 `DecodeStream(skip_special_tokens=True)` with the
runner's backend tokenizer, under the existing tokenizer lock. Preserve the
current `clean_up_tokenization_spaces=False` behavior. Retain raw generated
token IDs independently of presentation text.

GPT-2 uses byte-level BPE. One emoji can span multiple tokens; decoding each
token independently can expose temporary replacement characters. A
tokenizer-only probe against the cached pinned tokenizer confirmed that
`DecodeStream.step()` returns no chunk while those bytes are incomplete and
releases the complete character when possible. This probe loaded no model
weights and establishes no generation performance.

Next, withhold any suffix that might become the existing chat stop pattern:

```python
r"\n\n[^\s>]+>"
```

For example, hold a trailing newline, `"\n\n"`, or `"\n\nHuman"`.
Release the held text when whitespace invalidates a speaker-label candidate,
while checking overlapping newline prefixes. If `>` completes the boundary,
discard the boundary and everything after it before publishing. Speaker labels
have no fixed character limit, so a fixed-size suffix buffer would leak some
labels. The reply-token limit bounds this holdback instead.

At successful completion, use the existing `GenerationResult.text` as the
authority. Release incomplete boundary prefixes that were ordinary reply text,
such as `"\n\nHu"` at the token cap. `DecodeStream` has no final flush method;
a trailing incomplete byte sequence can legitimately become a replacement
character in final decoding. Reconcile with that result without changing
sampling, IDs, or stopping rules. Every provisional visible string must be a
prefix of the final accepted reply on successful jobs. Test this property.

## Keep submission and streaming separate

The current generation POST remains unchanged:

```text
POST /api/message or /api/regenerate
    -> 202 {job_id, status, position}
```

After that acknowledgement, subscribe to:

```text
GET /api/jobs/<job_id>/events
    -> Content-Type: text/event-stream; charset=utf-8
```

Use `fetch()` and a `ReadableStream` SSE parser. This provides explicit HTTP
error handling, aborts, reconnect control, and fallback to the existing
`GET /api/jobs/<job_id>` polling endpoint. The SSE wire format is defined by the
[HTML standard](https://html.spec.whatwg.org/multipage/server-sent-events.html).
Fetch consumers must implement parsing and reconnection themselves.

Send cumulative snapshots, rather than append-only text deltas:

```text
id: 7
event: progress
data: {"version":1,"seq":7,"status":"running","position":0,"generated_tokens":5,"reply_text":" I like"}

```

`generated_tokens` counts tokens observed by the callback, including withheld
speaker markers and special tokens. It is distinct from visible characters
and from the final authoritative output-token count. Sequence numbers advance
when text, token count, status, or queue position changes. Heartbeats do not
advance them. Queued snapshots have empty reply text and a positive position;
running snapshots have position zero.

Retain one latest snapshot per job, under the existing condition lock. Do not
retain an event history or a growing collection of partial strings. HTTP
handlers copy the snapshot under the lock, then serialize and write outside
it. Slow subscribers may skip revisions and receive the latest complete
prefix; they must never delay inference.

Every subscription immediately receives the latest snapshot. A reconnect can
send a validated decimal `Last-Event-ID` header; its sequence is advisory,
because the latest cumulative snapshot suffices without replay. The browser
ignores stale or duplicate revisions within its current view. A newly loaded
page accepts the initial snapshot even if an older saved cursor exists.

After generation and existing final-state/archive validation succeed, publish
a terminal event:

```text
id: 19
event: complete
data: {"version":1,"seq":19}

```

The browser then fetches the canonical result from `GET /api/jobs/<job_id>`
and uses the existing commit path. A `failed` terminal event prompts the same
GET to obtain the existing bounded error. Do not send a potentially 4 MiB
archive through SSE. If that GET fails temporarily, retain the pending job ID
and retry GET; terminal notification is not yet a successful browser commit.

Add the latest progress snapshot to queued/running JSON job responses so
polling fallback can still display partial text. Completed responses retain
their existing canonical state. Expose the streaming capability through
health, controlled initially by a proposed `--enable-streaming` flag.

## Bound connections and recovery

Initial limits for the preview and first release:

| Item | Choice |
| --- | --- |
| Concurrent inference | Existing single worker |
| Streams | Four globally, two per client IP |
| HTTP handlers | Existing maximum of 16; streams occupy at most four |
| Heartbeat | SSE comment every five seconds, including queueing/prefill |
| Stream lease | 60 seconds, then reconnect to the same job |
| Browser idle watchdog | 20 seconds, reset by received bytes/heartbeats |
| Socket I/O timeout | Existing 15 seconds; stalled readers release their slot |
| Maximum SSE frame/client parser buffer | 1 MiB of UTF-8 bytes per unfinished frame; reject oversized frames and fall back to polling |
| Jobs, queue, replies, deadlines, result retention | Existing limits unchanged |

Prove the frame limit sufficient for the configured reply cap with the actual
pinned tokenizer; never truncate canonical replies to fit transport bounds.
Frames contain bounded progress, with no input archive or repeated full
context. Consume complete frames incrementally: one network chunk may contain
several valid frames whose combined size exceeds the per-frame cap. The bound
is not a total-stream limit or a JavaScript string-length limit. Keep the
existing 256-token production cap, 300-second generation
deadline, 300-second queue expiry, and 600-second completed-result retention.
The deadline is cooperative and cannot interrupt one hung model forward.

Reserve/release stream slots atomically. Return a retryable 503 before stream
headers when slots are full; the browser polls its already accepted job.
Release slots on disconnect, timeout, terminal status, expiry, and shutdown.
Use a separate HTTP streaming response path: omit `Content-Length`, flush
frames, retain no-store/security headers, and close the response on exit. The
current HTTP/1.0 connection-close framing supports this; avoid a global HTTP
protocol change. After headers are sent, handle errors inside the streaming
path instead of trying to append a normal JSON HTTP response.

Preserve exact Host/Origin validation, request rates, trusted-proxy client IP
handling, CSP, no-referrer, and private job IDs. A job ID remains an unguessable
bearer capability; subscribing does not newly bind it to an IP address.
Do not log stream URLs, prompts, or reply payloads.

The browser reconnects with bounded backoff. On EOF or a transport/parser error,
check the existing job by GET, then reconnect or poll if still pending. Never
repeat the generation POST automatically. A missing/expired job or server
restart restores committed state and the draft. Closing a tab ends only its
subscription; generation can finish within the existing deadline. An explicit
Stop button and cancellation semantics are separate work.

## Display partial replies without saving them as chats

In [`public.html`](../src/gpt2_local/web/public.html), retain `committed` and
`pending`. Save the exact prepared input-display snapshot and immutable
submitted controls in IndexedDB with `job_id: null` before the generation
POST, preserving the existing uncertain-request protection. After the 202
acknowledgement, persist its job ID before subscribing. The display snapshot
contains context blocks/counts, not a second full transcript.

Whitelist `pending.display` fields: `context_text`, `context_blocks`,
`context_tokens`, `retained_example_rounds`, and `dropped_rounds`.
Copy only those fields from the prepared preview; exclude its full `turns` and
chat archive. Store submitted settings separately, and retain regeneration's
effective preview settings when displaying its provisional seed. Ordinary
`preview.settings` currently describes the old session, so it cannot substitute
for the submitted controls. Validate restored fields before rendering.

Keep provisional reply text, its sequence, and generated-token count in a
separate ephemeral view. After reload, the saved input display and latest job
snapshot reconstruct that view. Do not recalculate the pending prompt using
settings that may have changed since submission. Older pending records lacking
the display snapshot use final-only polling, with provisional rendering
disabled: progress text alone cannot reconstruct the corresponding input.

Update one provisional GPT-2-colored span through `textContent`. Keep the
prepared prompt visible and label the reply as in progress. Show the generated
count separately; do not treat the evolving reply as an authoritative retained
context token count. Render available text immediately, without a synthetic
typing delay. Follow new text only when the reader was already near the bottom.
Accessibility announcements should report status changes, not every token.

Keep the last provisional view during transient connection failures. Export
continues to use the committed archive, and stream updates do not write partial
replies into it. At completion, fetch and validate the canonical state, then
persist through the established commit path. Share terminal handling between
streaming and polling, including the existing pending-job identity check.

During regeneration, retain the old candidate and history in `committed` while
showing the replacement provisionally. Success replaces it and preserves the
old candidate as today. Failure restores it and the draft. Reconnection does
not advance the seed or regenerate. Preserve independent tab storage,
import/export format version 2, seeded examples, and Enter/Shift+Enter behavior.

### Next-generation seed policy in the local development preview

The seed control is now the seed for the next generation. After a successful
new reply or regeneration, canonical session/archive settings store its
successor modulo `2**63`. The completed reply's generation record keeps the
seed actually used. Regeneration uses the displayed/stored next seed as-is,
so the sequence advances once per successful generation rather than twice
when replacing a reply. Manual seed changes are honored; blank uses the
stored next seed. Failed/rejected jobs, previews, provisional stream tokens,
reconnects and repeated result GETs leave committed seed settings intact.
Greedy decoding still advances the field even though it ignores seed values.

The existing completion handlers apply the authoritative returned settings;
the browser does no seed arithmetic. Export/import/reload preserve the next
seed along with historical actual seeds, using schema version 2. Older
archives retain their saved seed as the next seed without a hidden migration.
This change is local-only pending owner testing; the public streaming build
and its deployment/rollback state remain unchanged.

## Implement and verify in stages

1. **Runtime observer and decoder.** Add the optional hook and presentation
   helper, with model-free callback/stopping/decode tests. Cover prompt skipping,
   EOS, deadline/callback exceptions, split emoji/CJK/combining marks, leading
   spaces, incomplete bytes, arbitrary/long speaker labels, overlapping
   newlines, false delimiter prefixes, and incomplete prefixes at token limit.
   Use the real pinned tokenizer for decoder tests, without loading weights.
2. **Backend transport.** Extend jobs, JSON progress, capability reporting, and
   the SSE handler. Test early flush, heartbeats during a blocked fake prefill,
   reconnects, expiry/shutdown, slow readers, subscriber limits, and concurrent
   health/submission requests. Assert no model-callback socket writes and no
   growing event backlog. Retain all origin, request, queue, and archive checks.
3. **Browser integration.** Test byte-fragmented UTF-8/SSE frames, duplicate and
   stale sequences, sequence gaps, malformed/oversized frames, multiple valid
   frames in one oversized network chunk, literal HTML display, and capacity
   fallback. Test reload/reconnect with exactly one
   generation POST, terminal GET failures, export while pending, regeneration
   rollback/history, seed wrapping, tab isolation, and unchanged uncertain-POST
   handling. Use a deterministic fake runner for the first browser preview.
4. **Local real-model parity and performance.** Review the fixed test prompts
   and settings before running generation. Compare streaming off/on on the
   same WSL runtime, first greedy and then sampled with fixed seeds, starting
   each comparison from the same exported chat. Require
   identical prompt/output IDs, final text, stopping reason, effective seed,
   settings, and archive structure, excluding elapsed measurements,
   `exported_at`, `cold_start_included`, and transient delivery measurements. Use short
   32-token trials and one near-full-context trial, not a sustained load test.
   Record source revision, prompt, settings, model/runtime provenance, and
   cold/warm status in ignored private evidence files.
5. **Owner preview.** Make the loopback laptop preview available for new chats
   and imported copies of existing chats. Exercise visible incremental output,
   regeneration, reload midway, and exports. Address findings and retain a
   concrete reviewed candidate before requesting production approval.

Existing test suites must remain passing. The runtime, transport, and browser
implementation is present and the automated suites pass. Local real-model
checks passed, and the owner confirmed that the laptop preview works. Keep reproduced results in
[`VERIFICATION.md`](../VERIFICATION.md) when they exist.

Measure queue wait from admission to worker start. Callback time to first token
starts immediately before `model.generate()` and includes prefill and the
stopping-check overhead before the hook; it does not isolate prefill alone.
Record time to the first published safe prefix separately, leaving it null if
no safe text was streamed before completion.

For `N` observed generated tokens at callback times `t1` through `tN`, report
decoding TPS as `(N - 1) / (tN - t1)` only when `N >= 2` and the denominator is
positive. Retain the existing total TPS, which includes prefill. Measure browser
receipt/first-paint latency on its own clock, with a recorded start point such
as pressing Send; do not subtract unsynchronized browser/server timestamps.
Record CPU utilization, process peak RSS, and cgroup memory events where
applicable. Compare streaming overhead on the same machine, and do not infer
production latency or sustained burst behavior from laptop TPS.

## The selected laptop preview

Use **<http://localhost:8891/>**, separate from existing chats and the public
origin. Reuse `~/.venvs/gpt2-xl` and its pinned cache in Ubuntu WSL, selecting this
worktree through `PYTHONPATH`; do not reinstall the shared editable package.
Use loopback, offline mode, and two Torch threads for the first trials.

For a future foreground run, open PowerShell in `.worktrees/gpt2-streaming`
and start the tracked convenience launcher:

```powershell
.\streaming-preview.ps1
```

Keep that terminal open. Once the ready message appears, open
<http://localhost:8891/>. **Press Ctrl+C in that terminal to stop the preview**;
wait for the process to exit before starting another model. The launcher uses
this checkout's source through `PYTHONPATH`, resolves Ubuntu's home dynamically,
and runs the existing pinned environment without an editable reinstall. It
uses CPU FP32, offline mode, two Torch threads, loopback port 8891, streaming,
the normal 256-token service cap, and the 8 GiB available-memory guard.

The current seed-policy preview is a detached WSL service at PID 1355,
replacing the verified idle parity harness's PID 341. Its private log and PID
are `outputs/deploy/seed-preview-20261004.log` and `seed-preview.pid`.
Do not run the foreground launcher while that process is loading or serving
on 8891. The convenience launcher starts the implemented service directly and
does not repeat the historical private parity trials.

The equivalent command inside Ubuntu WSL is:

```bash
PYTHONPATH='/mnt/c/Users/MikeFrank/Documents/=== FILES ===/-- AI --/GPT-2/.worktrees/gpt2-streaming/src' \
  ~/.venvs/gpt2-xl/bin/python -m gpt2_local.public_chat \
  --host 127.0.0.1 --port 8891 --offline --threads 2 \
  --max-reply-tokens 256 --minimum-available-memory-gib 8 \
  --enable-streaming
```

Set the browser reply allowance to 32 for the short trials. Keep the server cap
at 256: the existing fresh-chat default is 120, and a server cap of 32 would
reject initialization unless the interface also changed. Recheck available
memory, model processes, and port 8891 on both WSL
and Windows immediately before loading. If another laptop model is running,
coordinate its shutdown/export instead of starting another resident copy.

Read-only checks before this preview found Python 3.12.3, Torch 2.14.1+cpu,
Transformers 5.18.0, and Tokenizers 0.23.2 in that WSL environment, approximately
14.68 GiB available memory, and no identified GPT-2 Python process or listener
on the candidate port. These are snapshots, not future capacity guarantees.
The durable harness completed exact greedy/sample parity and a 992-token
context trial, then reused that single resident model for the service at
**http://localhost:8891/**. The browser showed provisional tokens, resumed the
same pending regeneration after reload, and committed the final reply. Details
and timing limits are recorded in [`VERIFICATION.md`](../VERIFICATION.md).

## Accepted hosted preview and authorized production rollout

The hosted candidate at `test.gpt2-chat.com` passed incremental display,
same-job reload, canonical token/settings checks, archive transfers,
regeneration history, query/origin checks, and independent service/memory
verification. The owner accepted the hosted incremental display and explicitly
requested production rollout. Detailed browser/API evidence and timing limits
are in [`VERIFICATION.md`](../VERIFICATION.md).

**Production streaming is active.** The application stays frozen at
`5389a30` in `/home/gpt-2/GPT-2-streaming-preview`; production's original
`/home/gpt-2/GPT-2-tools` checkout remains `e5971db`. Amazon Linux preparation
passed at **2026-10-04 04:47:57.130018 UTC**, after 14 promotion and 11 guard
tests. Durable unit `gpt2-production-promote-20261004-0448` began its full
600-second preview-result retention wait at **04:48:13.198694 UTC** and emitted
`production_streaming_active` at **2026-10-04 04:58:27.430940 UTC**. Production
PID **58714** is active/enabled; preview is inactive/PID 0. Independent checks
at **04:59:00.504526 UTC** passed canonical HTTPS streaming-ready health,
test-domain 307/`no-store`, exact source/override/snapshot preservation,
sole-model ownership, and unchanged chess/Caddy PIDs. Two certified canonical
HTTPS jobs passed incremental SSE/disconnect/reconnect, canonical token IDs and
settings, archive transfers and regeneration checks, plus exact-origin/query/
redirect checks. Independent after-verification at **05:01:30.491948 UTC**
passed with no warnings, unchanged PIDs, and the service idle.

The controlled handoff keeps one resident Lightsail model. It preserves the
old production unit/env/checkout and installs only its owned drop-in to select
the exact tested source through WorkingDirectory and a process `PYTHONPATH`
wrapper, with streaming enabled on canonical port 8899. The final Caddy file
preserves the original public/chess blocks and adds a temporary test-domain
307/`no-store` redirect. Actual canonical SSE delivery and reconnect passed;
HTTPS readiness alone would not establish first-event delivery.

See [`PRODUCTION-STREAMING.md`](PRODUCTION-STREAMING.md) for the exact prepare,
durable promote and rollback commands, stage paths, readiness criteria, and
failure recovery. **The old preview helper's `--restore` is obsolete after
promotion; use the promotion helper's `--rollback`.** Both directions gate
generation, drain accepted jobs, and retain completed GET results for 600
seconds before stopping a model. Source/configuration drift or undrainable
work requires operator review; no force stop is permitted.

There is no demonstrated combined budget for two FP32 inference processes on
this shared host. Reclaimable file cache is not independently spare RAM while
preserving a model's working set. Recheck the 8 GiB startup floor and recursive
cgroup/exclusive-model guards on each handoff. Starting two
`PublicChatApplication` objects around one runner creates two inference workers;
sharing a model across independent frontends requires a separate broker.
The laptop preview uses a different machine. Test and canonical origins keep
separate browser storage; move chats through JSON export/import.

## Source basis

The design uses the checked-in runtime, chat transaction, public worker, and
browser code at `e5971db`, plus read-only inspection of installed generation
hooks and the pinned tokenizer. Relevant primary references are the
[Transformers 5.18.0 generation source](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/generation/utils.py),
the [SSE standard](https://html.spec.whatwg.org/multipage/server-sent-events.html),
and the [Caddy streaming documentation](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#streaming).
