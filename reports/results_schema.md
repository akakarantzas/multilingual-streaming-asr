# Results Schema

Planned per-file CSV fields:
- `audio_filepath`
- `reference_text`
- `hypothesis_text`
- `language`
- `language_readiness`
- `wer`
- `cer`
- `latency_ms`
- `gpu_memory_allocated_mb`
- `model_id_or_path`
- `device`
- `warnings`

Planned summary JSON fields:
- `model_id_or_path`
- `model_revision`
- `language`
- `manifest_path`
- `num_files`
- `wer`
- `cer`
- `total_words`
- `total_chars`
- `substitutions`
- `deletions`
- `insertions`
- `latency_ms_mean`
- `latency_ms_p50`
- `latency_ms_p95`
- `gpu_memory_allocated_mb_max`
- `normalization`
- `warnings`

Planned environment check JSON fields:
- `python_version`
- `platform`
- `torch_version`
- `torchaudio_version`
- `nemo_version`
- `cuda_available`
- `torch_cuda_version`
- `nvidia_smi_available`
- `gpu_name`
- `driver_version`
- `checks`
- `warnings`
