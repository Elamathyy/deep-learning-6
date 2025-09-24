import spacy
import pandas as pd

# Load spaCy English NER model
nlp = spacy.load("en_core_web_sm")

def sentence_to_bio(sentence):
    doc = nlp(sentence)

    tokens = [token.text for token in doc]
    bio_tags = ["O"] * len(tokens)  # initialize all tags as "O"

    for ent in doc.ents:
        start = ent.start
        bio_tags[start] = "B-" + ent.label_
        for i in range(start + 1, ent.end):
            bio_tags[i] = "I-" + ent.label_

    # Create DataFrame for tabular output
    df = pd.DataFrame({"Word": tokens, "BIO Tag": bio_tags})
    return df

# Example sentence
sentence = "John lives in New York"
df = sentence_to_bio(sentence)
print(df)


output:
screenshot:(<img width="651" height="517" alt="Screenshot 2025-09-24 102336" src="https://github.com/user-attachments/assets/e2f6bda3-91b8-49a3-ba5b-cb7ee53f247c" />
)
