# download_nsina.py
# Downloads NSina Sinhala news corpus

from datasets import load_dataset
import pandas as pd

print("Downloading NSina corpus...")
dataset = load_dataset("sinhala-nlp/NSINA", split="train")

df = pd.DataFrame(dataset)
print(f"Total articles: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample article:\n{df['content'][0][:300]}")

# Save locally
df.to_csv("data/nsina_corpus.csv", index=False)
print("\n✓ Saved to data/nsina_corpus.csv")