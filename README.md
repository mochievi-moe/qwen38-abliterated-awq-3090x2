# Running Abliterated Qwen3.8-27B AWQ on Two RTX 3090s (vLLM TP=2)

## Overview

This recipe documents observed inference results for the abliterated Qwen3.8-27B AWQ artifact served with [vLLM](https://github.com/vllm-project/vllm) tensor parallelism (TP=2) across two NVIDIA RTX 3090 GPUs (24 GB each). It is a reproducibility guide and measurement record, not a model release or a weight-distribution mechanism.

The measured quantization artifact is [`twolven/Qwen3.8-27B-abliterated-AWQ-MTP`](https://huggingface.co/twolven/Qwen3.8-27B-abliterated-AWQ-MTP). The Hugging Face revision used in the 2026-08-31 benchmark wave was not recorded in the primary measurement report; verify the snapshot you download locally before serving. This repository does not redistribute model weights.

The documented engine is vLLM **0.27.1**. The baseline configuration measured here uses **MTP disabled** (no speculative decoding). A separate MTP3 + PIECEWISE candidate was not measured because FlashInfer compilation failed with a missing `ninja` binary; see [MTP not measured](#mtp-not-measured) below.

## Hardware and prerequisites

| Item | Documented configuration |
|---|---:|
| GPUs | 2x NVIDIA RTX 3090 (24 GB VRAM each) |
| Parallelism | vLLM TP=2 |
| Weight format | AWQ (`compressed-tensors`) |
| Engine | vLLM 0.27.1 |
| Context length | 131072 tokens |
| Max concurrent sequences | 4 (`--max-num-seqs 4`) |
| Max batched tokens | 4096 (`--max-num-batched-tokens 4096`, BTOK=4096) |
| GPU memory utilization | 0.70 |
| MTP / speculative decoding | Off (baseline) |

Before deployment, confirm compatible NVIDIA drivers, CUDA toolkit, and Python for vLLM 0.27.1 on your host. Driver and CUDA versions were not recorded in the primary benchmark report; check vLLM release notes and your `nvidia-smi` output locally.

You also need sufficient disk space for the Hugging Face snapshot, a Python virtual environment with vLLM 0.27.1, and no competing heavyweight GPU workload on both cards.

## Setup

### 1. Create a Python environment with vLLM 0.27.1

```bash
python3 -m venv <VENV_DIR>
source <VENV_DIR>/bin/activate
pip install "vllm==0.27.1"
vllm --version
```

Confirm the reported version is `0.27.1` before loading weights.

### 2. Download the model snapshot (do not redistribute)

Fetch the approved artifact from Hugging Face into `<MODEL_DIR>`. This recipe does not include weights.

```bash
# Example using the Hugging Face CLI; use any equivalent fetch method you trust.
huggingface-cli download twolven/Qwen3.8-27B-abliterated-AWQ-MTP \
  --local-dir <MODEL_DIR>
```

Record the revision or commit you actually downloaded. The 2026-08-31 wave did not publish a pinned revision in its report.

### 3. Optional: pin power limits

The measured host ran both GPUs at a 250 W power limit during the benchmark wave. Adjust with your platform tools if you want to match that condition; this is optional and host-specific.

## Launch recipe

Bind to localhost unless you deliberately expose the server behind your own reverse proxy or tunnel.

```bash
source <VENV_DIR>/bin/activate

vllm serve <MODEL_DIR> \
  --host 127.0.0.1 \
  --port <PORT> \
  --served-model-name qwen \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --quantization compressed-tensors \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Fixed baseline flags for comparisons with the measurements below:

- TP=2
- `max-model-len=131072`
- `max-num-seqs=4`
- `max-num-batched-tokens=4096` (BTOK=4096)
- MTP off (no `--speculative-config` / no draft model)

## Health check

After the server finishes loading, confirm the process is listening and responding:

```bash
curl -sS "http://127.0.0.1:<PORT>/health"
curl -sS "http://127.0.0.1:<PORT>/v1/models"
```

Expect HTTP 200 and a served model id of `qwen`.

## Smoke generation

```bash
curl -sS "http://127.0.0.1:<PORT>/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Reply with only the number for 1+1."}],
    "max_tokens": 16,
    "temperature": 1.0,
    "top_p": 0.95
  }'
```

The benchmark wave used thinking off, `temperature=1.0`, `top_p=0.95`, `top_k=20`, `repetition_penalty=1.05`, and `max_tokens=512` for throughput tests.

## Measured results

These are observed results from the 2026-08-31 benchmark wave, not vendor projections. Two measurement methodologies are used and must not be treated as interchangeable:

1. **Individual pure decode (tok/s per request)** — per-call decode interval and that call's `completion_tokens`.
2. **HTTP wave wall aggregate (tok/s)** — `sum(completion_tokens) / wall-clock batch duration` for concurrent requests launched in one wave.

Do not compute speedup ratios between these two methodologies.

### Throughput (2026-08-31, baseline, MTP off)

| Metric | Concurrency | Methodology | Result |
|---|---:|---|---:|
| Individual generation speed | C1 (1 concurrent request) | pure decode per request | **61.1 tok/s** (61.1267 measured) |
| Individual generation speed | C8 (8 concurrent requests) | median pure decode per request | **43.1 tok/s** (43.115 measured) |
| HTTP effective total throughput | C8 | HTTP wave wall aggregate | **154.1 tok/s** (154.088 measured) |

C8 was the **maximum concurrency measured in this wave**. It is not documented here as a hard implementation ceiling; higher concurrency was not tested.

C2/C4 aggregate values from an earlier aggregation formula were withdrawn and must not be cited.

### Visualizations

![Throughput summary](visualizations/qwen38-3090x2-throughput-x-card.png)

The chart shows per-request pure-decode speeds at C1/C2/C4/C6/C8 and the C8 HTTP wave wall aggregate as separate panels. Aggregate and per-request numbers are not directly comparable.

### 131072-token admission (single request)

A single streaming request was admitted with chat-template tokenization adjusted to **131008 prompt tokens** and **`max_tokens=64`** (requested total 131072). Result: **PASS** (HTTP 200; `usage.prompt_tokens=131008`, `completion_tokens=26`).

This is a **single-request admission probe**, not a guarantee that all workloads or concurrency levels can sustain 128K context.

### Limited quality probes

Representative probes on the same baseline server:

| Probe | Result |
|---|---|
| Uncensored fixture (3 prompts) | refusal-pattern matched **0/3** |
| JSON-constrained prompt | JSON parse **PASS** (`{"answer":4}`) |
| Tool-call prompt | `get_weather` **PASS** with arguments `{"city":"東京"}` |

These are limited smoke checks, not a full safety or capability certification.

## Benchmark commands (generalized)

The primary report used a fixed prompt SHA-256 (`e1ec74f3e8a5970ba44223caf71bb57e7e70dcd7ca77feabb2293cf9fc6aca0a`), warmup plus three formal batches per concurrency level, and concurrent HTTP clients. Reproduce with your own harness; keep methodologies separated.

**Individual pure decode** — for each completed request, compute `(completion_tokens - 1) / (last_content_timestamp - first_content_timestamp)` from streamed chunks.

**HTTP wave wall aggregate** — for a batch of `C` concurrent requests launched together, compute `sum(completion_tokens) / (batch_end - batch_start)` using wall-clock time covering the whole wave.

Example single-client timing check (not a full benchmark harness):

```bash
# Replace with your streaming client that records first/last content timestamps.
python3 <YOUR_BENCH_CLIENT>.py \
  --url "http://127.0.0.1:<PORT>/v1/chat/completions" \
  --model qwen \
  --concurrency 1 \
  --max-tokens 512
```

## MTP not measured

An MTP3 + PIECEWISE configuration was attempted after the baseline wave. FlashInfer compilation failed with `FileNotFoundError: ninja`. No dependency install, download, or retry was performed for that path in the measured run. MTP acceptance rates and MTP speed are **unknown**. Treat MTP as an optional future verification item only; this recipe makes no speed-up claim for MTP.

## Stopping the server

Identify the vLLM API server parent process and stop it gracefully:

```bash
# Find the listener (example)
ss -H -ltnp "sport = :<PORT>"

# Send SIGTERM to the vLLM parent PID you verified
kill -TERM <PID>

# Wait until the port is free; escalate only if the process survives
while ss -H -ltnp "sport = :<PORT>" | grep -q .; do sleep 2; done
```

If child worker processes remain, terminate the parent first; avoid killing unrelated GPU jobs. Confirm both GPUs release memory with `nvidia-smi`.

## Recovery and rollback

1. **Graceful restart** — stop the server as above, then relaunch the [fixed baseline command](#launch-recipe).
2. **Configuration rollback** — if you changed launch flags, restore the last known-good command line from your own notes or process manager config, then restart.
3. **Model rollback** — if you tested another snapshot, switch `<MODEL_DIR>` back to the verified baseline snapshot and restart.
4. **Post-recovery checks** — `/health` returns 200, `/v1/models` lists `qwen`, and a short smoke completion succeeds.

## Known pitfalls and troubleshooting

1. **Do not mix measurement methodologies.** Per-request pure decode and HTTP wave wall aggregate answer different questions; never compare them with a single speedup ratio.
2. **Do not cite withdrawn C2/C4 aggregate numbers.** Only per-request pure-decode values are valid for C2/C4.
3. **C8 is a measured upper bound in this wave, not proven max concurrency.** Higher `C` was not tested.
4. **128K admission is a single-request probe.** Do not describe it as full-context guarantee for all traffic patterns.
5. **Check free memory and swap before loading.** Abort or reduce concurrency if swap grows unexpectedly during load or benchmark waves.
6. **Do not overlap heavyweight GPU work.** Avoid simultaneous large downloads, second model loads, or unrelated training jobs on the same GPUs.
7. **Verify tool and JSON probes locally** after any change to parsers (`qwen3_coder`, `qwen3` reasoning parser) or chat-template kwargs.
8. **MTP requires separate validation.** Missing build tools (for example `ninja`) can fail FlashInfer compilation; fix the build environment before attempting MTP, and measure independently.

## Credits and License

- The measured quantization artifact is credited to [twolven](https://huggingface.co/twolven) and [`twolven/Qwen3.8-27B-abliterated-AWQ-MTP`](https://huggingface.co/twolven/Qwen3.8-27B-abliterated-AWQ-MTP). Verify the applicable model license on Hugging Face before use.
- The underlying Qwen model family is credited to the original model authors. Verify upstream license terms separately.
- The serving engine is [vLLM](https://github.com/vllm-project/vllm); verify the applicable engine license from its upstream source.
- Benchmark measurements cited here come from model-lab PR #94 (`reports/out/3090_qwen38_speed_wave_20260831_0620/report.md`, 2026-08-31).

This recipe documentation does not redistribute model weights. Select a documentation license before public release if you fork this README.
