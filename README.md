# Multilingual Streaming ASR

A benchmark and evaluation toolkit for NVIDIA Nemotron 3.5 ASR, focused on English and Greek batch and streaming workflows with language-appropriate accuracy metrics, latency, GPU memory, and multi-stream concurrency profiling. Cantonese (Yue) exploration and adaptation are planned, pending data and model validation.

Target model: `nvidia/nemotron-3.5-asr-streaming-0.6b`

Target hardware: NVIDIA DGX Spark. Other CUDA systems are unverified.

Status: pre-baseline. Environment checks and pipeline scaffolding are present; model revision, hardware results, and fine-tuning results are not yet recorded.

## Status

- [x] Repository scaffold
- [x] Environment verification script
- [x] Model smoke-load script
- [x] Batch evaluation scaffold
- [x] Streaming demo scaffold
- [x] Profiling scaffold
- [ ] Confirm CUDA, PyTorch, and NeMo versions on target hardware
- [ ] Record exact model revision
- [ ] Run English baseline
- [ ] Run Greek baseline
- [ ] Validate Cantonese (Yue) dataset readiness
- [ ] Validate Cantonese (Yue) model behavior
- [ ] Finalize Greek fine-tuning recipe
- [ ] Finalize Cantonese (Yue) adaptation recipe
- [ ] Publish benchmark results

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- CUDA: TBD
- PyTorch: TBD
- NVIDIA NeMo: TBD
- Hugging Face access to `nvidia/nemotron-3.5-asr-streaming-0.6b`

If the model is gated, accept the model terms on Hugging Face and authenticate locally:

```bash
huggingface-cli login
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

- English: baseline evaluation scaffolded with WER
- Greek: baseline evaluation and fine-tuning scaffolding in progress with WER
- Cantonese (Yue): exploration and adaptation planned; metric TBD, likely CER or mixed WER/CER depending on transcript format

## Usage

The manifest paths below are examples. Replace them with your local manifest paths.

Validate setup:

```bash
python scripts/verify_environment.py --json-output reports/environment.json
python scripts/smoke_load_model.py
```

Run evaluation:

```bash
python scripts/run_batch_eval.py --manifest data/manifests/en.jsonl --language en --target-lang en-US --output experiments/baseline/en_results.csv
python scripts/run_batch_eval.py --manifest data/manifests/el.jsonl --language el --target-lang el-GR --output experiments/baseline/el_results.csv
```

Run the streaming demo:

```bash
python scripts/run_streaming_demo.py --language en --target-lang en-US
python scripts/run_streaming_demo.py --language el --target-lang el-GR
```

Profile and plot:

```bash
python scripts/profile_inference.py --manifest data/manifests/en.jsonl --language en --target-lang en-US --output-dir experiments/baseline/profile_en
python scripts/make_benchmark_plots.py --input-dir experiments/baseline/profile_en --output-dir reports/figures
```

Run tests:

```bash
pytest
```

## Data

Evaluation manifests are JSONL files:

```json
{"audio_filepath": "data/samples/en_us_sample.wav", "text": "Reference transcript.", "duration": 3.2}
```

Audio format expectations are currently TBD. Validation for sample rate, channels, and supported formats should be finalized before benchmark results are treated as final.

See `data/manifests/README.md` for the full manifest format.

## Outputs

Output directories are created automatically by the scripts where needed.

- `reports/environment.json`: environment and dependency check
- `experiments/baseline/*_results.csv`: per-file transcripts and language-appropriate accuracy metrics
- `experiments/baseline/profile_*`: latency, memory, and concurrency profiles
- `reports/figures/`: generated benchmark plots

Benchmark results depend on hardware, audio format, model revision, batch size, chunking, and concurrency settings.

## Results

No benchmark results are published yet. Results will be added after model revision, environment versions, audio validation, and DGX Spark baselines are finalized.

## Fine-Tuning

Greek fine-tuning support is scaffolding only. Cantonese (Yue) exploration and adaptation are planned. Training recipes, datasets, and results are not finalized.

## Project Structure

```text
src/             Core ASR, audio, evaluation, and profiling code
scripts/         CLI entry points
data/            Manifests and sample placeholders
experiments/     Baseline and fine-tuning outputs
reports/         Result schemas and figures
tests/           Unit tests
```

## License

MIT
