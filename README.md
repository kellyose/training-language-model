# Lab: Preprocessing Text Datasets for Small Language Model (SLM) Training

This repository features a completed Python implementation and notebook detailing the fundamental data preprocessing pipeline required before training a Transformer-based Small Language Model (SLM). 

Neural networks cannot process raw strings directly; they require structured numerical inputs. This project demonstrates how to ingest a text corpus and construct an end-to-end tokenization and indexing pipeline from scratch.

## ⚙️ Core Pipeline Implemented

1. **Tokenization Engine:** Utilizes regular expressions (`re.split(r" +", text)`) to isolate words and symbols while safely handling varying whitespace frequencies.
2. **Vocabulary Extraction:** Isolates unique tokens across the dataset to define the strict categorical boundaries of the model's language features.
3. **Index Mapping:** Constructs dual-lookup dictionaries to maintain consecutive numerical token IDs spanning $0$ to $k-1$ (where $k$ is the total vocabulary size):
   - `token_to_index`: Encodes text strings into mathematical tensors.
   - `index_to_token`: Decodes model logit outputs back into human-readable strings.
4. **Object-Oriented Packaging:** Consolidates the entire workflow into a clean, production-ready `SimpleWordTokenizer` class capable of initializing from an arbitrary text corpus or a pre-defined vocabulary list.

## 📊 Dataset Insights (Africa Galore Corpus)
- **Total Tokens Processed:** 19,065
- **Unique Vocabulary Size ($k$):** 5,260

## 🚀 Usage Example

```python
# Initialize the tokenizer with your dataset
tokenizer = SimpleWordTokenizer(corpus=dataset)

# Encode a raw string into numerical Token IDs
encoded_ids = tokenizer.encode("The Lagos air was thick with humidity")
print(encoded_ids) 
# Output: [5173, 2480, 4192, 360, 1942, 613, 1054]

# Decode Token IDs back to human-readable text
decoded_text = tokenizer.decode(encoded_ids)
print(decoded_text)
# Output: "The Lagos air was thick with humidity"
Note: This lab is built subject to the license and terms set out in the Google DeepMind AI Research Foundations repository.
