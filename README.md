# Multilingual Streaming ASR

[![tests](https://github.com/akakarantzas/multilingual-streaming-asr/actions/workflows/tests.yml/badge.svg)](https://github.com/akakarantzas/multilingual-streaming-asr/actions/workflows/tests.yml)

![Side-by-side streaming demo (mock data)](docs/assets/demo-hero.png)

*Demo UI preview, mock data. Real results pending.*

*Not affiliated with or endorsed by NVIDIA. NVIDIA and Nemotron are trademarks of NVIDIA Corporation.*

A benchmark and evaluation toolkit for NVIDIA Nemotron 3.5 ASR, focused on Greek batch and streaming workflows with language-appropriate accuracy metrics, latency, GPU memory, and multi-stream concurrency profiling. Cantonese (Yue) exploration and adaptation are planned, pending data and model validation.

Target model: `nvidia/nemotron-3.5-asr-streaming-0.6b`

The model weights are not included in this repository. They are downloaded from [Hugging Face](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) and are covered by NVIDIA's own license, not this repository's MIT license; see the model card for the terms.

Target hardware: NVIDIA DGX Spark. Development also happens on a consumer NVIDIA GPU (RTX 4060 Ti) and on CPU; none of these setups are verified yet. Each batch evaluation records the device and GPU it ran on.

## Status

Pre-baseline. Environment checks and pipeline scaffolding are present; model revision, hardware results, and fine-tuning results are not yet recorded. Detailed milestones are in `docs/milestone_checklist.md`.

- [x] Repository scaffold
- [x] Environment verification script
- [x] Model smoke-load script
- [x] Batch evaluation scaffold
- [x] Streaming demo scaffold
- [x] Profiling scaffold
- [ ] Confirm CUDA, PyTorch, and NeMo versions on target hardware
- [ ] Record exact model revision
- [ ] Run Greek baseline
- [ ] Validate Cantonese (Yue) dataset readiness
- [ ] Validate Cantonese (Yue) model behavior
- [ ] Finalize Greek fine-tuning recipe
- [ ] Finalize Cantonese (Yue) adaptation recipe
- [ ] Publish benchmark results

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (recommended). Scripts default to `--device auto`, which falls back to CPU when CUDA is unavailable; CPU runs are slow.
- NeMo is Linux-first; on Windows, WSL2 is the likely route (not yet tested here)
- CUDA: TBD
- PyTorch: TBD
- NVIDIA NeMo: TBD
- Hugging Face access to `nvidia/nemotron-3.5-asr-streaming-0.6b`

If the model is gated, accept the model terms on Hugging Face and authenticate locally:

```bash
hf auth login
```

## Setup

Confirm CUDA, PyTorch, and NVIDIA NeMo compatibility for the target environment before installing dependencies.

Linux / Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows / PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Supported Languages

- Greek: baseline evaluation and fine-tuning scaffolding in progress with WER
- Cantonese (Yue): exploration and adaptation planned; metric TBD, likely CER or mixed WER/CER depending on transcript format

## Usage

The manifest paths below are examples. Replace them with your local manifest paths.

Validate setup:

```bash
python scripts/verify_environment.py --json-output reports/environment.json
python scripts/smoke_load_model.py
```

`smoke_load_model.py` also accepts `--device` and `--revision`.

Transcribe a single file:

```bash
python -m src.asr.infer --audio path/to/audio.wav --output experiments/baseline/single_file_result.json
```

Run evaluation:

```bash
python scripts/run_batch_eval.py --manifest data/manifests/el.jsonl --language el --target-lang el-GR --output experiments/baseline/el_results.csv
```

`--language` and `--target-lang` must agree: the only supported pair is `el` with `el-GR`. The streaming demo and single-file inference also accept `--target-lang auto`.

Run the streaming demo:

```bash
python scripts/run_streaming_demo.py --language el --target-lang el-GR
```

Profile and plot:

```bash
python scripts/profile_inference.py --manifest data/manifests/el.jsonl --language el --target-lang el-GR --output-dir experiments/baseline/profile_el
python scripts/make_benchmark_plots.py --input-dir experiments/baseline/profile_el --output-dir reports/figures
```

Validate fine-tuning inputs (does not start training):

```bash
python experiments/finetuning/validate_finetuning_inputs.py --train-manifest data/manifests/el_gr_train.jsonl --val-manifest data/manifests/el_gr_val.jsonl
```

Run tests (no GPU needed; `requirements-test.txt` is enough, and CI uses it on Python 3.11 and 3.12):

```bash
python -m pip install -r requirements-test.txt
python -m pip install ruff
python -m ruff check .
pytest
```

Preview the comparison demo page (base vs. fine-tuned, side by side). It currently runs on a mock data source with invented Greek sentences, so no server, model, or GPU is needed:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory src/app/static
```

Then open http://localhost:8765.

## Data

Evaluation manifests are JSONL files:

```json
{"audio_filepath": "data/samples/el_gr_sample.wav", "text": "Reference transcript.", "duration": 3.2}
```

Single-file inference accepts `.wav` or `.flac`. Validation for sample rate and channels is not finalized yet and should be settled before benchmark results are treated as final.

See `data/manifests/README.md` for the full manifest format.

## Outputs

Output directories are created automatically by the scripts where needed.

- `reports/environment.json`: environment and dependency check
- `experiments/baseline/*_results.csv`: per-file transcripts and language-appropriate accuracy metrics
- `experiments/baseline/profile_*`: latency, memory, and concurrency profiles
- `reports/figures/`: generated benchmark plots

Benchmark results depend on hardware, audio format, model revision, batch size, chunking, and concurrency settings.

## Results

No benchmark results are published yet. Results will be added after model revision, environment versions, audio validation, and baselines on the target hardware are finalized.

## Fine-Tuning

Greek fine-tuning support is scaffolding only. Cantonese (Yue) exploration and adaptation are planned. Training recipes, datasets, and results are not finalized.

## Project Structure

```text
src/             Core ASR, audio, evaluation, and profiling code
src/app/static/  Comparison demo page (mock data for now)
scripts/         CLI entry points
data/            Manifests and sample placeholders
experiments/     Baseline and fine-tuning outputs
reports/         Result schemas and figures
docs/            Milestone checklist and images
tests/           Unit tests
.github/         CI workflow
```

## Contact

Questions, partnership inquiries, or feedback about this project:

- LinkedIn: [Apostolos Kakarantzas](https://www.linkedin.com/in/akakarantzas/)
- X / Twitter: [@aevagnarly](https://x.com/aevagnarly)
- Email: [hello@akakarantzas.com](mailto:hello@akakarantzas.com)

## License

MIT
