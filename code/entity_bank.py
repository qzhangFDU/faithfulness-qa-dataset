"""
Entity Bank Builder for Faithfulness-QA
Extracts entities from SQuAD contexts using SpaCy NER,
builds a typed entity bank for counterfactual substitution.
"""

import json
import os
import random
import logging
from collections import defaultdict
from pathlib import Path

import spacy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Entity types we care about for substitution
ENTITY_TYPES = ['PERSON', 'GPE', 'ORG', 'DATE', 'CARDINAL', 'NORP', 'LOC', 'EVENT']

class EntityBank:
    """Manages a typed entity bank for counterfactual substitution."""
    
    def __init__(self, bank_dir: str):
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.entities = defaultdict(set)
    
    def add_entity(self, text: str, label: str):
        """Add an entity to the bank."""
        text = text.strip()
        if len(text) < 2 or len(text) > 100:
            return
        # Skip entities that are just numbers or single chars
        if text.isdigit() and label not in ('CARDINAL', 'DATE'):
            return
        self.entities[label].add(text)
    
    def build_from_texts(self, texts: list, nlp, batch_size: int = 100):
        """Extract entities from a list of texts using SpaCy NER."""
        logger.info(f"Processing {len(texts)} texts for entity extraction...")
        
        for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, disable=["parser", "lemmatizer"]),
                       total=len(texts), desc="Extracting entities"):
            for ent in doc.ents:
                if ent.label_ in ENTITY_TYPES:
                    self.add_entity(ent.text, ent.label_)
        
        for label in sorted(self.entities.keys()):
            logger.info(f"  {label}: {len(self.entities[label])} unique entities")
    
    def save(self):
        """Save entity bank to disk."""
        for label, ents in self.entities.items():
            filepath = self.bank_dir / f"{label}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sorted(list(ents)), f, ensure_ascii=False, indent=2)
        
        # Save summary
        summary = {label: len(ents) for label, ents in self.entities.items()}
        with open(self.bank_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Entity bank saved to {self.bank_dir}")
    
    def load(self):
        """Load entity bank from disk."""
        for filepath in self.bank_dir.glob("*.json"):
            if filepath.name == "summary.json":
                continue
            label = filepath.stem
            with open(filepath, 'r', encoding='utf-8') as f:
                self.entities[label] = set(json.load(f))
        logger.info(f"Loaded entity bank: {dict({k: len(v) for k, v in self.entities.items()})}")
    
    def sample(self, label: str, exclude: str = None) -> str:
        """Sample a random entity of the given type, excluding a specific one."""
        candidates = self.entities.get(label, set())
        if exclude:
            candidates = candidates - {exclude}
        if not candidates:
            return None
        return random.choice(list(candidates))
    
    def get_stats(self) -> dict:
        return {label: len(ents) for label, ents in self.entities.items()}


def build_entity_bank_from_squad(data_dir: str, max_samples: int = None):
    """Build entity bank from SQuAD dataset."""
    from datasets import load_dataset
    
    logger.info("Loading SQuAD dataset...")
    ds = load_dataset('rajpurkar/squad', split='train')
    
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    # Get unique contexts (SQuAD has many questions per context)
    contexts = list(set(ds['context']))
    logger.info(f"Found {len(contexts)} unique contexts from {len(ds)} QA pairs")
    
    # Load SpaCy
    logger.info("Loading SpaCy model...")
    nlp = spacy.load('en_core_web_lg')
    
    # Build bank
    bank = EntityBank(os.path.join(data_dir, 'entity_bank'))
    bank.build_from_texts(contexts, nlp)
    bank.save()
    
    return bank


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    
    bank = build_entity_bank_from_squad(args.data_dir, args.max_samples)
    stats = bank.get_stats()
    print(f"\nEntity Bank Statistics:")
    for label, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    print(f"  TOTAL: {sum(stats.values())}")
