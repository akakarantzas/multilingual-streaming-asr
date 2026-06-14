# JSONL Manifests

Evaluation manifests use one JSON object per line.

Required fields:
- `audio_filepath`: non-empty string path to an audio file.
- `text`: non-empty ground-truth transcript string.

Optional fields:
- `duration`: numeric duration in seconds, when known.

Example English row:

```json
{"audio_filepath": "data/samples/en_us_sample.wav", "text": "This is an English reference transcript.", "duration": 3.2}
```

Example Greek row:

```json
{"audio_filepath": "data/samples/el_gr_sample.wav", "text": "Αυτό είναι ένα ελληνικό κείμενο αναφοράς.", "duration": 4.1}
```

Greek `el-GR` is adaptation-ready/exploratory until fine-tuning and evaluation results are measured.

Future Cantonese/Yue example, not part of the current English/Greek core milestone:

```json
{"audio_filepath": "data/samples/yue_hk_sample.wav", "text": "呢個係粵語參考文本。", "duration": 3.8}
```

Data must be collected with consent and documented with source, speaker, language/locale, recording conditions, and permitted use.
