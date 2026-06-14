# Milestone Checklist

## Milestone 0 - Environment Verification

- [ ] NeMo installed
- [ ] PyTorch CUDA works
- [ ] nvidia-smi works
- [ ] exact ASR checkpoint identified
- [ ] real model load smoke script runs
- [ ] single-file inference confirmed
- [ ] memory footprint measured

## Milestone 1 - Evaluation Infrastructure

- [ ] WER implemented and tested
- [ ] CER implemented and tested
- [ ] manifest validator implemented
- [ ] English test manifest created
- [ ] Greek test manifest created and labeled adaptation-ready/exploratory
- [ ] normalized baseline WER table generated
- [ ] raw/cased/punctuated scoring TODO documented

## Milestone 2 - Streaming Demo

- [ ] microphone capture works
- [ ] chunked fallback ASR works
- [ ] latency and RTF logged
- [ ] terminal demo works
- [ ] native streaming support status documented

## Milestone 3 - Cantonese/Yue Evaluation, Deferred Extension

- [ ] do not start until English/Greek pipeline and Greek fine-tuning are complete
- [ ] model/tokenizer compatibility checked
- [ ] recording consent documented
- [ ] sentence list prepared
- [ ] recordings collected
- [ ] transcripts QA'd by native speaker
- [ ] CER computed

## Milestone 4 - Profiling and Benchmarking

- [ ] latency tracker implemented
- [ ] GPU snapshot implemented
- [ ] profiling script runs
- [ ] concurrency simulation runs
- [ ] benchmark plots generated
- [ ] bottleneck documented

## Milestone 5 - Greek Fine-Tuning

- [ ] train/val/test manifests prepared
- [ ] Greek adaptation-ready status documented
- [ ] fine-tuning config validated
- [ ] training run completed or infeasibility documented
- [ ] base vs fine-tuned WER compared

## Milestone 6 - Deployment Optimization, Stretch

- [ ] ONNX/TensorRT/Triton feasibility checked
- [ ] at least one optimization benchmark run if time allows

## Milestone 7 - Portfolio Packaging

- [ ] README complete with English baseline vs Greek adaptation framing
- [ ] technical report complete
- [ ] plots saved
- [ ] demo video recorded
- [ ] resume bullets updated with real numbers only

## Commit Discipline

- One prompt per commit where practical.
- Commit messages should reference the milestone.
- Never commit raw private audio without consent.
- Never commit large checkpoints.
