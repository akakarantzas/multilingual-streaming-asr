# Greek Fine-Tuning Preparation

Goal: compare exploratory base-model Greek output against Greek fine-tuned model output for `el-GR`.

Required data:
- Train manifest
- Validation manifest
- Held-out test manifest for final reporting

Expected manifest format is JSONL with one object per line:

```json
{"audio_filepath": "data/samples/el_gr_train_001.wav", "text": "Αυτό είναι ένα ελληνικό κείμενο αναφοράς.", "duration": 3.2}
```

Required fields:
- `audio_filepath`: non-empty path to the audio file
- `text`: non-empty Greek reference transcript

Optional fields:
- `duration`: numeric duration in seconds

Data consent reminder: every recording must have documented consent, permitted use, speaker/language metadata, and recording conditions.

Honest-results rule:
- Do not smooth results.
- Do not cherry-pick examples.
- Do not invent WER improvements.
- Report base-model and fine-tuned model results on the same held-out test manifest.

Greek readiness rule: do not claim production Greek support unless measured fine-tuned results justify it. Until then, Greek `el-GR` remains adaptation-ready/exploratory.
