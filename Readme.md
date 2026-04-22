# Faithfulness-QA: A Counterfactual Entity Substitution Dataset for Training Context-Faithful RAG Models

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dataset: 99K](https://img.shields.io/badge/Dataset-99K_samples-blue.svg)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)]()

## 📖 Overview

**Faithfulness-QA** is a large-scale dataset of **99,094** question–answer pairs designed to train and evaluate the faithfulness of Retrieval-Augmented Generation (RAG) models to retrieved context.

The core idea is **counterfactual entity substitution**: for each QA sample, we replace the answer-bearing entity in the context with a type-consistent alternative, creating a controlled conflict between the context and the model's parametric knowledge. A faithful model should output the *replacement* entity (from context), not the *original* entity (from memory).

<p align="center">
  <img src="https://img.shields.io/badge/SQuAD-49%2C094_samples-blue" alt="SQuAD"/>
  <img src="https://img.shields.io/badge/TriviaQA-50%2C000_samples-red" alt="TriviaQA"/>
  <img src="https://img.shields.io/badge/Entity_Bank-76%2C953_entities-green" alt="Entity Bank"/>
  <img src="https://img.shields.io/badge/Quality-100%25_pass-brightgreen" alt="Quality"/>
</p>

## 🔑 Key Features

- **99K counterfactual QA samples** from SQuAD (49,094) and TriviaQA (50,000)
- **8 entity types** covered: PERSON, ORG, GPE, DATE, CARDINAL, NORP, LOC, EVENT
- **76,953 unique entities** in the curated entity bank
- **100% quality pass rate** on 200-sample automated audits
- **Ready-to-use train/dev/test splits** (80/10/10)
- **Fully automated pipeline** — reproducible from source datasets

## 📊 Dataset Statistics

### Overall

| Source | Input | Output | Success Rate | Train | Dev | Test |
|--------|-------|--------|-------------|-------|-----|------|
| SQuAD | 87,599 | 49,094 | 56.0% | 39,275 | 4,909 | 4,910 |
| TriviaQA | 87,041 | 50,000 | 57.4% | 40,000 | 5,000 | 5,000 |
| **Total** | **174,640** | **99,094** | **56.7%** | **79,275** | **9,909** | **9,910** |

### Entity Type Distribution

| Type | SQuAD | SQuAD % | TriviaQA | TriviaQA % |
|------|-------|---------|----------|------------|
| PERSON | 9,775 | 19.9% | 22,871 | 45.7% |
| ORG | 10,114 | 20.6% | 8,186 | 16.4% |
| DATE | 9,997 | 20.4% | 2,057 | 4.1% |
| GPE | 6,292 | 12.8% | 11,810 | 23.6% |
| CARDINAL | 6,568 | 13.4% | 1,348 | 2.7% |
| NORP | 4,004 | 8.2% | 1,247 | 2.5% |
| LOC | 1,629 | 3.3% | 1,952 | 3.9% |
| EVENT | 715 | 1.5% | 529 | 1.1% |

> **Complementarity**: TriviaQA is dominated by PERSON entities (45.7%), while SQuAD provides balanced coverage across ORG, DATE, PERSON, and CARDINAL. Combining both yields broad entity-type diversity.

### Entity Bank

| Type | Count |
|------|-------|
| ORG | 25,378 |
| PERSON | 20,292 |
| DATE | 10,613 |
| GPE | 6,769 |
| CARDINAL | 6,636 |
| LOC | 2,977 |
| NORP | 2,849 |
| EVENT | 1,439 |
| **Total** | **76,953** |

## 📁 Repository Structure

```
faithfulness-qa-dataset/
├── code/
│   ├── entity_bank.py          # Entity bank construction & management
│   ├── build_dataset.py        # Main pipeline (SQuAD processing)
│   ├── build_triviaqa.py       # TriviaQA pipeline (streaming mode)
│   └── quality_analysis.py     # Quality analysis & statistics
├── data/
│   ├── entity_bank/            # Typed entity bank (76,953 entities)
│   │   ├── PERSON.json         # 20,292 person entities
│   │   ├── ORG.json            # 25,378 organization entities
│   │   ├── GPE.json            # 6,769 geo-political entities
│   │   ├── DATE.json           # 10,613 date entities
│   │   ├── CARDINAL.json       # 6,636 cardinal number entities
│   │   ├── NORP.json           # 2,849 nationality/group entities
│   │   ├── LOC.json            # 2,977 location entities
│   │   ├── EVENT.json          # 1,439 event entities
│   │   └── summary.json        # Entity bank statistics
│   ├── faithfulness_qa_squad_raw.jsonl      # SQuAD raw (49,094)
│   ├── faithfulness_qa_squad_train.jsonl    # SQuAD train (39,275)
│   ├── faithfulness_qa_squad_dev.jsonl      # SQuAD dev (4,909)
│   ├── faithfulness_qa_squad_test.jsonl     # SQuAD test (4,910)
│   ├── faithfulness_triviaqa_raw.jsonl      # TriviaQA raw (50,000)
│   ├── faithfulness_triviaqa_train.jsonl    # TriviaQA train (40,000)
│   ├── faithfulness_triviaqa_dev.jsonl      # TriviaQA dev (5,000)
│   ├── faithfulness_triviaqa_test.jsonl     # TriviaQA test (5,000)
│   ├── build_stats.json                     # SQuAD build statistics
│   └── triviaqa_build_stats.json            # TriviaQA build statistics
└── README.md
```

## 📋 Data Format

Each sample is a JSON object in JSONL format:

```json
{
  "id": "5733be284776f41900661182",
  "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
  "original_context": "...the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858...",
  "modified_context": "...the Virgin Mary reputedly appeared to Kiyomori in 1858...",
  "original_answer": "Saint Bernadette Soubirous",
  "faithful_answer": "Kiyomori",
  "original_entity": "Saint Bernadette Soubirous",
  "replacement_entity": "Kiyomori",
  "entity_type": "PERSON",
  "source": "squad"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Original sample ID from the source dataset |
| `question` | string | The question text |
| `original_context` | string | Unmodified context from the source dataset |
| `modified_context` | string | Context after counterfactual entity substitution |
| `original_answer` | string | Ground-truth answer from the source dataset |
| `faithful_answer` | string | Correct answer given the modified context (= replacement entity) |
| `original_entity` | string | The named entity that was replaced |
| `replacement_entity` | string | The new entity substituted in |
| `entity_type` | string | NER type: `PERSON`, `GPE`, `ORG`, `DATE`, `CARDINAL`, `NORP`, `LOC`, or `EVENT` |
| `source` | string | Source dataset: `squad` or `triviaqa` |

## 🚀 Quick Start

### Loading the Dataset

```python
import json

# Load training data
train_data = []
with open("data/faithfulness_qa_squad_train.jsonl") as f:
    for line in f:
        train_data.append(json.loads(line))

print(f"Loaded {len(train_data)} training samples")

# Example: create RAG-style training input
sample = train_data[0]
prompt = f"""Context: {sample['modified_context']}

Question: {sample['question']}

Answer:"""

target = sample['faithful_answer']  # Model should output this
```

### Evaluating Model Faithfulness

```python
def compute_faithfulness_rate(model, test_data):
    """Measure how often the model follows context over parametric memory."""
    faithful_count = 0
    parametric_count = 0

    for sample in test_data:
        prediction = model.generate(
            context=sample['modified_context'],
            question=sample['question']
        )
        if sample['faithful_answer'].lower() in prediction.lower():
            faithful_count += 1
        elif sample['original_answer'].lower() in prediction.lower():
            parametric_count += 1

    n = len(test_data)
    print(f"Faithfulness Rate: {faithful_count/n:.1%}")
    print(f"Parametric Rate:   {parametric_count/n:.1%}")
```

### Reproducing the Dataset

```bash
# Install dependencies
pip install spacy transformers datasets pandas tqdm
python -m spacy download en_core_web_lg

# Step 1: Build entity bank from SQuAD contexts
python code/entity_bank.py --data_dir data/

# Step 2: Build Faithfulness-QA from SQuAD
python code/build_dataset.py --data_dir data/ --source squad

# Step 3: Build Faithfulness-QA from TriviaQA (streaming)
python code/build_triviaqa.py --data_dir data/ --target 50000

# Step 4: Run quality analysis
python code/quality_analysis.py --data_dir data/
```

## 🔬 Methodology

### Pipeline Overview

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Stage 1:       │    │   Stage 2:       │    │   Stage 3:       │    │   Stage 4:       │
│   Entity Bank    │───▶│   NER & Answer   │───▶│  Counterfactual  │───▶│   Quality        │
│   Construction   │    │   Entity Match   │    │  Substitution    │    │   Filtering      │
└──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
Extract entities        Match answer to NER     Replace entity with     6 quality checks +
from SQuAD contexts     entity via 3-strategy   same-type alternative   80/10/10 split
(SpaCy en_core_web_lg)  cascade                 from entity bank
```

**Stage 1**: Extract all named entities from 19,035 unique SQuAD contexts using SpaCy NER → 76,953 entity bank.

**Stage 2**: For each QA sample, match the answer to a recognized entity using:
1. Exact match (case-insensitive)
2. Substring match (overlap ≥ 3 chars)
3. Positional overlap (≥ 50% character overlap)

**Stage 3**: Sample a type-consistent replacement entity, re-sample up to 5× for length compatibility (0.3–3.0× ratio), replace all occurrences in context.

**Stage 4**: Apply 6 quality filters (presence, change, length, novelty, frequency, entity length), then split 80/10/10.

## ✅ Quality Validation

| Check | Pass Rate | Target |
|-------|-----------|--------|
| Replacement entity present in modified context | 200/200 (100%) | 100% |
| Original entity removed from modified context | 200/200 (100%) | 100% |
| Context actually changed | 200/200 (100%) | 100% |
| Context length ratio within [0.5, 2.0] | 200/200 (100%) | ≥90% |

## 💡 Intended Use Cases

1. **Faithfulness-aware fine-tuning**: Train with `(modified_context, question) → faithful_answer` to teach models to follow context over parametric memory.
2. **Attention-based faithfulness loss**: Supervise cross-attention weights to ensure models attend to retrieved context.
3. **Faithfulness evaluation**: Measure the rate at which models output the faithful answer (context-grounded) vs. the original answer (parametric).
4. **Knowledge conflict research**: Study LLM behavior when retrieved context contradicts parametric knowledge.

## ⚠️ Known Limitations

- **No coreference resolution**: Pronominal references to replaced entities are not updated.
- **No NLI-based filtering**: Quality checks are rule-based; no semantic consistency verification via NLI models.
- **Semantic plausibility**: Some substitutions are syntactically valid but semantically implausible (e.g., non-US city in "born in [City], Illinois").
- **English only**: Current pipeline and entity bank support English only.

## 📄 Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{zhang2026faithfulnessqa,
  title={Faithfulness-QA: A Counterfactual Entity Substitution Dataset for Training Context-Faithful RAG Models},
  author={Zhang, Qiang},
  year={2026},
  url={https://github.com/qzhangFDU/faithfulness-qa-dataset}
}
```

## 📚 Related Work

- **Self-RAG** (Asai et al., ICLR 2024): Self-reflective retrieval-augmented generation
- **FaithfulRAG** (Zhang et al., ACL 2025): Fact-level conflict modeling for context-faithful RAG
- **FaithEval** (Ming et al., ICLR 2025): Faithfulness evaluation benchmark (4.9K samples)
- **CounterFact** (Meng et al., NeurIPS 2022): Counterfactual dataset for knowledge editing
- **Knowledge Conflicts Survey** (Xu et al., EMNLP 2024): Comprehensive survey of LLM knowledge conflicts

## 📜 License

This project is released under the [MIT License](LICENSE).

The source datasets (SQuAD, TriviaQA) are used under their respective licenses.
