# audreyt/pi-ds4

`pi` provider extension for running the [cyberneurova abliterated DeepSeek V4
Flash](https://huggingface.co/cyberneurova/CyberNeurova-DeepSeek-V4-Flash-abliterated-GGUF)
GGUF on Apple Silicon, end-to-end and out-of-the-box.

This is **a personal fork of [mitsuhiko/pi-ds4](https://github.com/mitsuhiko/pi-ds4)**.
Same UX (one-line `pi install`, on-demand `ds4-server`, per-process lease,
watchdog shutdown), with two changes:

1. **Pulls [`audreyt/ds4`](https://github.com/audreyt/ds4) `main`** instead of
   `antirez/ds4` `main`. That branch already contains:
   * the `support-q8_0-token-embd` PR (sent upstream to antirez/ds4) — adds
     stock-recipe loader support for the Q8_0 small tensors and F32 router
     that cyberneurova ships, including a `kernel_mul_mm_f32_f32` Metal
     kernel and CPU-side dequant for the compressor APE path; and
   * [ivanfioravanti's PR #15](https://github.com/antirez/ds4/pull/15) — Metal
     4 / M5 prefill optimizations (Q8_0 MPP, attention-output MPP, staged
     routed-MoE MPP, fused six-expert sum kernel; ~1.5x prefill on M5 Max).
2. **Replaces `download_model.sh` with `prepare_model.sh`**, which downloads
   the cyberneurova `Q2_K` file (~99 GB, resumable) and symlinks
   `ds4flash.gguf` to it. No harmonization, no Python venv, no extra disk
   beyond the source file — `audreyt/ds4` `main` accepts and runs the
   unmodified Q8_0 file directly on M-series Metal.

## Install

```sh
pi remove   github.com/mitsuhiko/pi-ds4   # if you had the upstream extension
pi install  github.com/audreyt/pi-ds4
```

On first launch, `pi` will:

1. Clone `audreyt/ds4` `main` into `~/.pi/ds4/support/`
2. `make ds4-server`
3. Run `prepare_model.sh`:
   * download `cyberneurova-DeepSeek-V4-Flash-abliterated-Q2_K.gguf` (~99 GB)
   * symlink `ds4flash.gguf` to it
4. Spawn `ds4-server` and register `ds4/deepseek-v4-flash` with `pi`.

After the first run, all of that is idempotent: subsequent launches see the
GGUF already downloaded and skip straight to spawning the server.

**Disk needed:** ~99 GB for the GGUF. Set `HF_TOKEN` if your HuggingFace
download benefits from auth.

## Local development install

If you have a checkout of this repo and a checkout of `audreyt/ds4` (or any
ds4 fork), wire `pi` to use them directly:

```sh
./install-pi-extension-local.sh /path/to/audreyt-ds4-checkout
```

If `~/.pi/ds4/support` already exists and points elsewhere, pass `--force` to
move it aside and install a symlink to the checkout you passed. Existing
`gguf/*.gguf` files (and resumable `.gguf.part` downloads) are preserved into
the new checkout first, using APFS clone-on-write copies on macOS when
available.

After install, restart `pi` or run `/reload`.

## What the upstream extension does (and this fork preserves)

Everything the [upstream `mitsuhiko/pi-ds4`](https://github.com/mitsuhiko/pi-ds4)
README documents still applies:

* On-demand `ds4-server` lifecycle managed via per-process leases in
  `~/.pi/ds4/clients/<pid>.json`, with a bundled `ds4-watchdog.sh` that stops
  the server when no leases remain.
* Single shared inference backend across all `pi` processes.
* HTTP API on `127.0.0.1:8000`, OpenAI-compatible.
* Logs at `~/.pi/ds4/log`; KV disk cache at `~/.pi/ds4/kv` with
  `--kv-disk-space-mb 8192` by default.
* `/ds4` inside `pi` shows the live ds4 log.

The only differences are the two fork-specific ones above (the ds4 source it
pulls and the model it downloads).

## Runtime layout

Runtime state under `~/.pi/ds4`:

* `support/` — shallow checkout of `audreyt/ds4` (`main` by default)
* `support/gguf/` — downloaded source GGUF
* `support/ds4flash.gguf` — symlink to the GGUF (consumed by `ds4-server`)
* `kv/` — on-disk KV cache
* `clients/` — active pi process leases
* `log` — build/download/server/watchdog log

## Configuration

Same env vars as upstream, plus a couple of fork-specific ones:

* `DS4_SUPPORT_REPO` — git URL of the ds4 fork to use. Default
  `https://github.com/audreyt/ds4`. Set to `https://github.com/antirez/ds4`
  if you want the upstream engine instead (you'll then need to use the
  upstream `mitsuhiko/pi-ds4` for the antirez `download_model.sh` flow, or
  override `DS4_PREPARE_SCRIPT`).
* `DS4_SUPPORT_BRANCH` — branch to clone. Default `main`. Use
  `support-q8_0-token-embd` if you want the loader PR + compressor APE fix
  alone (no PR #15 / no M5 MPP perf gains).
* `DS4_PREPARE_SCRIPT` — absolute path to the model-prep script. Default is
  the bundled `prepare_model.sh`.
* `DS4_RUNTIME_DIR` — use an existing ds4 checkout instead of `~/.pi/ds4/support`
* `DS4_MODEL_QUANT` — only `q2` is currently supported (cyberneurova ships
  Q2_K only). Default is auto-detected from RAM (≥128 GB → `q2`).
* `DS4_READY_TIMEOUT_MS` — server startup timeout.
* `DS4_SERVER_BINARY` — custom `ds4-server` binary path.
* `HF_TOKEN` — passed through to `curl` for HuggingFace downloads if set.

## Acknowledgements

* **[mitsuhiko/pi-ds4](https://github.com/mitsuhiko/pi-ds4)** — the upstream
  extension this fork is based on. All of the lifecycle / watchdog / lease
  machinery is Armin Ronacher's work.
* **[antirez/ds4](https://github.com/antirez/ds4)** — Salvatore Sanfilippo's
  DeepSeek V4 Flash inference engine, and its
  [llama.cpp-deepseek-v4-flash](https://github.com/antirez/llama.cpp-deepseek-v4-flash)
  converter that produced the cyberneurova GGUFs.
* **[ivanfioravanti's PR #15](https://github.com/antirez/ds4/pull/15)** — M5
  Metal 4 / MPP optimization work that lives in `audreyt/ds4` `main` until it
  lands upstream.
* **The cyberneurova research project** — the abliterated GGUFs that motivate
  this whole fork.

## License

MIT, matching upstream. See `LICENSE`.
