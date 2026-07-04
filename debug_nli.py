import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

m = "roberta-large-mnli"
tok = AutoTokenizer.from_pretrained(m)
mod = AutoModelForSequenceClassification.from_pretrained(m).eval()
print("id2label:", mod.config.id2label)

pairs = [
    ("The cat sat on the mat. It was sunny.", "The cat sat on the mat."),   # obvious entailment
    ("The cat sat on the mat.", "The cat did not sit on the mat."),          # obvious contradiction
    ("The cat sat on the mat.", "The stock market rose today."),             # unrelated/neutral
]
enc = tok([p for p, _ in pairs], [h for _, h in pairs],
          padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(mod(**enc).logits, dim=-1)
for (p, h), pr in zip(pairs, probs):
    named = {mod.config.id2label[i]: round(float(v), 3) for i, v in enumerate(pr)}
    print(f"H: {h!r}\n   -> {named}\n")