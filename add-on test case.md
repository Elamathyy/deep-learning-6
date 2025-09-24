import spacy
import pandas as pd

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def sentence_to_bio(sentence):
    doc = nlp(sentence)

    tokens = [token.text for token in doc]
    bio_tags = ["O"] * len(tokens)

    for ent in doc.ents:
        start = ent.start
        bio_tags[start] = "B-" + ent.label_
        for i in range(start + 1, ent.end):
            bio_tags[i] = "I-" + ent.label_

    return tokens, bio_tags

# Test cases
test_sentences = [
    ("Elon Musk founded SpaceX", ["B-PER", "I-PER", "O", "B-ORG"]),
    ("Google is in California", ["B-ORG", "O", "O", "B-LOC"])
]

results = []
for sentence, expected in test_sentences:
    tokens, bio_tags = sentence_to_bio(sentence)
    correct = "Y" if bio_tags == expected else "N"
    results.append([sentence, " ".join(bio_tags), correct])

# Convert to DataFrame for neat table
df = pd.DataFrame(results, columns=["Input Sentence", "Output Tags", "Correct (Y/N)"])
print(df.to_string(index=False))



output:
screenshot:(<img width="618" height="579" alt="Screenshot 2025-09-24 105547" src="https://github.com/user-attachments/assets/47297497-3052-41cd-b860-02f0e5c4fbf4" />)
