!pip install transformers torch -q

from transformers import pipeline

note = """Patient presents with a 3-day history of fever, sore throat, and
mild body ache. Rapid strep test ordered. Diagnosis: suspected
streptococcal pharyngitis. Started on Amoxicillin 500mg three times
daily for 7 days."""

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("Summary:", summarizer(note, max_length=40, min_length=10)[0]["summary_text"])

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["consultation", "investigation", "prescription", "follow-up"]
result = classifier(note, candidate_labels=labels)
print("Top label:", result["labels"][0], "| score:", result["scores"][0])

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
answer = qa(question="What medication was prescribed?", context=note)
print("Answer:", answer["answer"], "| confidence:", answer["score"])
