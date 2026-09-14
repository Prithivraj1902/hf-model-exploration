# Hugging Face Model Exploration

## Objective
Identify and compare three pretrained Hugging Face Transformer models suitable for **summarization, classification, and question answering**.

## Models Selected

| Task | Hugging Face Model | Architecture | Main Use |
|---|---|---|---|
| Summarization | `facebook/bart-large-cnn` | BART encoder-decoder | Generate concise summaries |
| Classification | `facebook/bart-large-mnli` | BART + NLI | Zero-shot text classification |
| Question Answering | `deepset/roberta-base-squad2` | RoBERTa | Extract answers from context |

## Approach and Methodology

1. Prepare a synthetic sample clinical note.
2. Load each pretrained model using the Hugging Face `pipeline` API.
3. Pass the same note to the summarization and classification pipelines.
4. Pass questions plus the note to the question-answering pipeline.
5. Compare the generated summary, predicted category, and extracted answers.

No real patient data is used.

## Dataset / Sample Data

A synthetic clinical note is used instead of a public patient dataset. It contains symptoms, examination findings, a test order, diagnosis, medication, and follow-up advice. This makes it possible to demonstrate all three tasks with one controlled example.

## Implementation

The complete implementation is in [`model_exploration.py`](./model_exploration.py). Dependencies are listed in [`requirements.txt`](./requirements.txt).

```bash
pip install -r requirements.txt
python model_exploration.py
```

The first run downloads the pretrained model weights.

## Expected Output

The program prints:

- A concise summary of the clinical note.
- The highest-scoring document category and scores for all candidate labels.
- Answers to questions about the medication, diagnosis, and test ordered.

Example output format:

```text
=== 1. Summarization (facebook/bart-large-cnn) ===
Summary: ...

=== 2. Zero-shot Classification (facebook/bart-large-mnli) ===
Top label: ... | score: ...

=== 3. Question Answering (deepset/roberta-base-squad2) ===
Question: What medication was prescribed?
Answer: Amoxicillin 500mg | confidence: ...
```

## Results and Observations

- **BART-large-CNN** produces a compact abstractive summary and is suitable when a longer document needs to be reduced to its key information.
- **BART-large-MNLI** can classify text into user-defined labels without task-specific training, which is useful for quick zero-shot experiments.
- **RoBERTa-base-SQuAD2** performs extractive question answering by locating answer spans in the supplied context.
- The models are pretrained and are not specifically fine-tuned for clinical use, so the outputs should be treated as a demonstration rather than medical advice.

## Comparison

| Criterion | BART-large-CNN | BART-large-MNLI | RoBERTa-base-SQuAD2 |
|---|---|---|---|
| Task | Summarization | Classification | Question Answering |
| Output | Generated text | Labels + scores | Answer span + score |
| Training needed for demo | No | No | No |
| Best feature | Concise summaries | Flexible labels | Direct context-based answers |
| Limitation | May paraphrase information | Zero-shot scores may be moderate | Cannot answer beyond supplied context |

## Conclusion

The exploration demonstrates that Hugging Face provides ready-to-use pretrained models for different NLP tasks through a common pipeline interface. BART-large-CNN is effective for summarization, BART-large-MNLI is convenient for zero-shot classification, and RoBERTa-base-SQuAD2 is suitable for extractive question answering. Together, they provide a simple foundation for experimenting with modern NLP without training models from scratch.

## GitHub Repository

https://github.com/Prithivraj1902/hf-model-exploration
