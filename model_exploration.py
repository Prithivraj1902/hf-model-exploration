"""
Hugging Face Model Exploration
--------------------------------
Compares 3 pretrained Hugging Face models for a healthcare text use case:
  1. Summarization        -> facebook/bart-large-cnn
  2. Zero-shot Classification -> facebook/bart-large-mnli
  3. Question Answering    -> deepset/roberta-base-squad2

Usage:
    pip install -r requirements.txt
    python model_exploration.py
"""

from transformers import pipeline

# Sample synthetic clinical note (not from a real patient)
note = """Patient presents with a 3-day history of fever, sore throat, and
mild body ache. No history of similar episodes. On examination, throat is
erythematous with tonsillar exudate. Rapid strep test ordered. Clinical
impression: suspected streptococcal pharyngitis. Started on Amoxicillin
500mg three times daily for 7 days. Advised to return if symptoms worsen
or persist beyond 5 days, or sooner if difficulty swallowing/breathing
develops."""


def run_summarization(text):
    print("\n=== 1. Summarization (facebook/bart-large-cnn) ===")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(text, max_length=40, min_length=10, do_sample=False)
    summary = result[0]["summary_text"]
    print("Summary:", summary)
    return summary


def run_classification(text, labels):
    print("\n=== 2. Zero-shot Classification (facebook/bart-large-mnli) ===")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(text, candidate_labels=labels)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    print(f"Top label: {top_label} | score: {top_score:.2f}")
    print("All scores:")
    for label, score in zip(result["labels"], result["scores"]):
        print(f"  {label}: {score:.3f}")
    return result


def run_qa(text, question):
    print("\n=== 3. Question Answering (deepset/roberta-base-squad2) ===")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa(question=question, context=text)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']} | confidence: {result['score']:.2f}")
    return result


if __name__ == "__main__":
    print("Sample clinical note:")
    print(note)

    run_summarization(note)

    candidate_labels = ["consultation", "investigation", "prescription", "follow-up"]
    run_classification(note, candidate_labels)

    run_qa(note, "What medication was prescribed?")
    run_qa(note, "What is the diagnosis?")
    run_qa(note, "What test was ordered?")

    print("\nDone.")
