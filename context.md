# Project Context

Status: Pre-Milestone 0

Hardware: NVIDIA DGX Spark

Target model: NVIDIA Nemotron 3.5 ASR

Target model ID: nvidia/nemotron-3.5-asr-streaming-0.6b

Target source: Hugging Face model ID or local .nemo export/download, to be confirmed during Milestone 0

Primary languages: English (en-US/en-GB, WER), Greek (el-GR, WER + fine-tuning)

Committed extension: Cantonese/Yue (yue-HK, CER + possible fine-tuning/adaptation using collaborator-recorded and native-speaker-QA'd data after English/Greek pipeline is complete)

Pipeline: Audio → Preprocessor → Nemotron 3.5 ASR → Transcript + Logger

Pipeline end point: transcript only. No LLM. No RAG. No TTS.

Current milestone: 0 — environment verification

Model name: NVIDIA Nemotron 3.5 ASR

Model ID: nvidia/nemotron-3.5-asr-streaming-0.6b

Model version/revision: TBD — record exact revision/hash after download

Model memory footprint: TBD

Confirmed single-file inference: TBD

Confirmed target_lang support: TBD

Confirmed native streaming/cache-aware inference support: TBD

Confirmed chunked fallback support: TBD

Confirmed Greek fine-tuning path: TBD
