# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Always make sure that the SSD streaming, CUDA, distributed inference, Metal default inference are not affected by fixes to other parts of the code.
- Keep model loading mmap-backed for the Metal default case; do not eagerly copy the full GGUF. Keep the model loading for SSD streaming of routed experts explicit: allocated buffers, fast reads from disk, always try to hide loading of missing routed experts by loading them while performing the inference of the shared expert and routed experts already in RAM. Always try to hide loading of layers for prefill in SSD streaming mode using the inference time of the current layer as the next one is loaded.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write elegant code in a state of grace. Don't settle for the first thing that comes to mind, try to find the most minimal and better working design. Don't introduce slop: very fragile code that just patches specific cases, dead code, useless code and code ways more complicated of how it should be.
- Comment important inference code where the model mechanics, cache lifetime, memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is intentional.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

This list is not complete, check the files for more info.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.

### TCIM/XH2 remote testing

- Standing user authorization (2026-09-05): project source code, build
  artifacts, and files needed for testing may be transferred to `hq50`
  for this project. This authorization applies to subsequent tests and
  other sessions of this project; do not request transfer permission again.
- Use `ssh hq50` to access the target host. Run `hm_smi -a` there to inspect
  the available XH2 devices.
- `make test-tcim` runs CLI/link checks followed by real-device runtime,
  tensor and synthetic static-span smoke on logical device 0; no GGUF is
  required. Serialize it with other device tests, for example with
  `flock -n /tmp/ds4-xh2.lock make -j2 test-tcim` on hq50.
- Before remote builds or tests, refresh `tcim/*.hex` locally with HDPL
  (hq50 only has the HAL SDK). Sync the worktree first, then the ignored hex
  files separately:

  ```sh
  rsync -az --exclude='.git/' --filter=':- .gitignore' ./ hq50:/home/sky/ds4/
  rsync -az tcim/*.hex hq50:/home/sky/ds4/tcim/
  ```

At every major change where one of the following could be affected, make sure to:

1. Test the normal Metal path and that speed is still at the level it was.
2. Test the SSD streaming path.
3. Test the distributed inference if it could be affected, but ask the user before doing so.
4. Check if CUDA could be broken after the change, and ask the user to give you access to the CUDA machine to actually test if everything is still fine.

## Agent skills

- Issue tracker: Issues and specifications are tracked in GitHub Issues for `fixedpointworks/ds4`. See `docs/agents/issue-tracker.md`.
- Triage labels: Use the five default canonical triage labels. See `docs/agents/triage-labels.md`.
- Domain docs: This repository uses a single-context domain documentation layout. See `docs/agents/domain.md`.
