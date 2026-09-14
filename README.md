# Hugging Face Model Exploration

Comparing 3 pretrained Hugging Face models for a healthcare text pipeline:
**summarization**, **classification**, and **question answering**.

## Models Used

| Task | Model | Architecture |
|---|---|---|
| Summarization | [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) | BART (encoder-decoder) |
| Zero-shot Classification | [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) | BART (encoder-decoder, NLI) |
| Question Answering | [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2) | RoBERTa (extractive QA) |

## What it does

The script (`model_exploration.py`) runs all three models against a single
synthetic clinical note and prints:

1. A short generated summary of the note.
2. The most likely document category out of `consultation`, `investigation`,
   `prescription`, `follow-up` (via zero-shot classification, no training required).
3. Direct answers to factual questions about the note (medication, diagnosis, test ordered).

No real patient data is used — the note is a synthetic example written for
demonstration purposes only.

## Setup

```bash
pip install -r requirements.txt
python model_exploration.py
```

First run will download the model weights (~1.5GB total across all three
models), so it may take a few minutes depending on your connection.

## Example Output

```
Summary: Patient has fever, sore throat, and body ache for 3 days.
Diagnosed with suspected streptococcal pharyngitis and started on
Amoxicillin 500mg three times daily for 7 days.

Top label: prescription | score: 0.41

Answer: Amoxicillin 500mg | confidence: 0.87
```

## Notes

- Zero-shot classification confidence can be moderate since the model isn't
  fine-tuned on clinical documentation categories specifically — a labeled
  dataset and fine-tuning would likely improve accuracy for production use.
- Extractive QA (`roberta-base-squad2`) only returns answers that are stated
  verbatim in the source text; it cannot infer or reason beyond what's written.
- Summarization output should be reviewed by a human before any clinical use,
  as abstractive models can occasionally add phrasing not strictly present
  in the source.

## Part of

This repo accompanies the "Hugging Face Model Exploration" assignment task,
comparing pretrained models suitable for summarization, classification, and
question answering on healthcare text.
