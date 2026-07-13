import pandas as pd
from collections import Counter
from wordfreq import zipf_frequency

INPUT_PATH = "data/raw/NLP_Dataset_2026.xlsx"
TEXT_COL = "Discrepancy"

df = pd.read_excel(INPUT_PATH, usecols=[TEXT_COL])
series = df[TEXT_COL].dropna().astype(str)

# Small manual stopword list (optional)
STOPWORDS = {
    "the","and","for","with","that","this","from","into","over","under","to","of","in","on","at",
    "a","an","is","are","was","were","be","been","by","as","or","if","it","its","not","no",
    "we","you","they","he","she","his","her","their","our","us"
}

def is_common_english(tok: str) -> bool:
    # Zipf >= 3.0 means common word; adjust threshold as needed
    return zipf_frequency(tok, "en") >= 3.0

counter = Counter()

for text in series:
    for tok in text.lower().split(" "):
        if not tok.isalpha():
            continue
        if not (2 <= len(tok) <= 4):
            continue
        if tok in STOPWORDS:
            continue
        if is_common_english(tok):
            continue
        counter[tok] += 1

out = pd.DataFrame(counter.most_common(), columns=["token", "count"])
out.to_csv("abbrev_candidates.csv", index=False)

print(out.head(50))