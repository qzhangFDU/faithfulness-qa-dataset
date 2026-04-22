"""
Faithfulness-QA Dataset Builder (v2 - improved success rate)
Constructs counterfactual entity-substituted QA pairs from SQuAD/TriviaQA.
"""

import json
import os
import re
import random
import logging
from pathlib import Path
from collections import Counter

import spacy
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entity_bank import EntityBank

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(42)

class FaithfulnessQABuilder:
    """Builds Faithfulness-QA dataset via counterfactual entity substitution."""
    
    def __init__(self, entity_bank_dir: str):
        logger.info("Loading SpaCy model...")
        self.nlp = spacy.load('en_core_web_lg')
        
        logger.info("Loading entity bank...")
        self.entity_bank = EntityBank(entity_bank_dir)
        self.entity_bank.load()
        
        self.stats = Counter()
    
    def find_answer_entity(self, context: str, answer_text: str):
        """Find the NER entity in context that matches the answer."""
        doc = self.nlp(context)
        
        # Strategy 1: Exact match
        for ent in doc.ents:
            if ent.text.lower().strip() == answer_text.lower().strip():
                return ent.text, ent.label_, ent.start_char, ent.end_char
        
        # Strategy 2: Answer is contained in entity or entity in answer
        best_match = None
        best_overlap = 0
        for ent in doc.ents:
            ent_lower = ent.text.lower().strip()
            ans_lower = answer_text.lower().strip()
            if ans_lower in ent_lower or ent_lower in ans_lower:
                overlap = min(len(ent_lower), len(ans_lower))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (ent.text, ent.label_, ent.start_char, ent.end_char)
        
        if best_match and best_overlap >= 3:
            return best_match
        
        # Strategy 3: Answer text exists literally in context, try to match NER
        # even if NER boundaries differ
        ans_start = context.lower().find(answer_text.lower())
        if ans_start >= 0:
            ans_end = ans_start + len(answer_text)
            for ent in doc.ents:
                # Check overlap
                overlap_start = max(ent.start_char, ans_start)
                overlap_end = min(ent.end_char, ans_end)
                if overlap_end > overlap_start:
                    overlap_len = overlap_end - overlap_start
                    if overlap_len >= len(answer_text) * 0.5:
                        return ent.text, ent.label_, ent.start_char, ent.end_char
        
        return None
    
    def replace_entity_in_context(self, context: str, original_entity: str, new_entity: str) -> str:
        """Replace all occurrences of the original entity in context."""
        pattern = re.compile(re.escape(original_entity), re.IGNORECASE)
        new_context = pattern.sub(new_entity, context)
        return new_context
    
    def quality_check(self, original_context: str, modified_context: str, 
                      original_entity: str, new_entity: str, query: str) -> tuple:
        """Check quality of substitution. Returns (passed, reason)."""
        
        # Check 1: New entity appears in modified context
        if new_entity not in modified_context:
            return False, "new_entity_not_in_context"
        
        # Check 2: Context actually changed
        if original_context == modified_context:
            return False, "context_unchanged"
        
        # Check 3: Modified context not too short
        if len(modified_context) < 50:
            return False, "context_too_short"
        
        # Check 4: The replacement shouldn't already exist in original
        if new_entity.lower() in original_context.lower():
            return False, "new_entity_already_in_original"
        
        # Check 5: Count occurrences - replacement should be reasonable
        pattern = re.compile(re.escape(original_entity), re.IGNORECASE)
        occurrence_count = len(pattern.findall(original_context))
        if occurrence_count > 10:
            return False, "too_many_occurrences"
        
        # Check 6: Length of entity should be reasonable
        if len(new_entity) < 2:
            return False, "replacement_too_short"
        
        return True, "ok"
    
    def process_squad_sample(self, sample: dict) -> dict:
        """Process a single SQuAD sample."""
        self.stats['total'] += 1
        
        context = sample['context']
        question = sample['question']
        answers = sample['answers']
        
        if not answers['text']:
            self.stats['skip_no_answer'] += 1
            return None
        
        answer_text = answers['text'][0]
        
        # Skip very short answers (likely not entities)
        if len(answer_text) < 2:
            self.stats['skip_answer_too_short'] += 1
            return None
        
        # Find answer entity via NER
        result = self.find_answer_entity(context, answer_text)
        if result is None:
            self.stats['skip_no_entity_match'] += 1
            return None
        
        entity_text, entity_label, start_char, end_char = result
        
        # Sample replacement entity
        new_entity = self.entity_bank.sample(entity_label, exclude=entity_text)
        if new_entity is None:
            self.stats['skip_no_replacement'] += 1
            return None
        
        # Try to match entity length roughly (prefer similar length replacements)
        # Try up to 5 times to find a reasonable length match
        for _ in range(5):
            candidate = self.entity_bank.sample(entity_label, exclude=entity_text)
            if candidate and 0.3 <= len(candidate) / max(len(entity_text), 1) <= 3.0:
                new_entity = candidate
                break
        
        # Replace in context
        modified_context = self.replace_entity_in_context(context, entity_text, new_entity)
        
        # Quality check
        passed, reason = self.quality_check(
            context, modified_context, entity_text, new_entity, question)
        if not passed:
            self.stats[f'skip_quality_{reason}'] += 1
            return None
        
        self.stats['success'] += 1
        self.stats[f'entity_type_{entity_label}'] += 1
        
        return {
            'id': sample.get('id', ''),
            'question': question,
            'original_context': context,
            'modified_context': modified_context,
            'original_answer': answer_text,
            'faithful_answer': new_entity,
            'original_entity': entity_text,
            'replacement_entity': new_entity,
            'entity_type': entity_label,
            'source': 'squad'
        }
    
    def process_triviaqa_sample(self, sample: dict) -> dict:
        """Process a single TriviaQA sample."""
        self.stats['total'] += 1
        
        question = sample['question']
        answer = sample['answer']
        answer_text = answer.get('value', '')
        
        entity_pages = sample.get('entity_pages', {})
        wiki_contexts = entity_pages.get('wiki_context', [])
        
        if not wiki_contexts or not answer_text:
            self.stats['skip_no_context'] += 1
            return None
        
        # Find context containing the answer
        context = None
        for wc in wiki_contexts:
            if answer_text.lower() in wc.lower():
                context = wc
                break
        
        if context is None:
            # Try aliases
            aliases = answer.get('aliases', [])
            for alias in aliases:
                for wc in wiki_contexts:
                    if alias.lower() in wc.lower():
                        context = wc
                        answer_text = alias
                        break
                if context:
                    break
        
        if context is None:
            self.stats['skip_answer_not_in_context'] += 1
            return None
        
        # Truncate long contexts
        if len(context) > 2000:
            idx = context.lower().find(answer_text.lower())
            start = max(0, idx - 800)
            end = min(len(context), idx + len(answer_text) + 800)
            context = context[start:end]
        
        if len(answer_text) < 2:
            self.stats['skip_answer_too_short'] += 1
            return None
        
        result = self.find_answer_entity(context, answer_text)
        if result is None:
            self.stats['skip_no_entity_match'] += 1
            return None
        
        entity_text, entity_label, start_char, end_char = result
        
        new_entity = self.entity_bank.sample(entity_label, exclude=entity_text)
        if new_entity is None:
            self.stats['skip_no_replacement'] += 1
            return None
        
        for _ in range(5):
            candidate = self.entity_bank.sample(entity_label, exclude=entity_text)
            if candidate and 0.3 <= len(candidate) / max(len(entity_text), 1) <= 3.0:
                new_entity = candidate
                break
        
        modified_context = self.replace_entity_in_context(context, entity_text, new_entity)
        
        passed, reason = self.quality_check(
            context, modified_context, entity_text, new_entity, question)
        if not passed:
            self.stats[f'skip_quality_{reason}'] += 1
            return None
        
        self.stats['success'] += 1
        self.stats[f'entity_type_{entity_label}'] += 1
        
        return {
            'id': sample.get('question_id', ''),
            'question': question,
            'original_context': context,
            'modified_context': modified_context,
            'original_answer': answer_text,
            'faithful_answer': new_entity,
            'original_entity': entity_text,
            'replacement_entity': new_entity,
            'entity_type': entity_label,
            'source': 'triviaqa'
        }
    
    def build_from_squad(self, output_path: str, max_samples: int = None):
        """Build dataset from SQuAD."""
        from datasets import load_dataset
        
        logger.info("Loading SQuAD training set...")
        ds = load_dataset('rajpurkar/squad', split='train')
        
        if max_samples:
            indices = list(range(len(ds)))
            random.shuffle(indices)
            ds = ds.select(indices[:max_samples])
        
        logger.info(f"Processing {len(ds)} SQuAD samples...")
        results = []
        
        for i in tqdm(range(len(ds)), desc="Building Faithfulness-QA (SQuAD)"):
            result = self.process_squad_sample(ds[i])
            if result:
                results.append(result)
        
        self._save_results(results, output_path)
        return results
    
    def build_from_triviaqa(self, output_path: str, max_samples: int = None):
        """Build dataset from TriviaQA."""
        from datasets import load_dataset
        
        logger.info("Loading TriviaQA training set...")
        ds = load_dataset('trivia_qa', 'rc', split='train')
        
        if max_samples:
            indices = list(range(len(ds)))
            random.shuffle(indices)
            ds = ds.select(indices[:max_samples])
        
        logger.info(f"Processing {len(ds)} TriviaQA samples...")
        results = []
        
        for i in tqdm(range(len(ds)), desc="Building Faithfulness-QA (TriviaQA)"):
            result = self.process_triviaqa_sample(ds[i])
            if result:
                results.append(result)
        
        self._save_results(results, output_path)
        return results
    
    def _save_results(self, results: list, output_path: str):
        """Save results to JSONL."""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(results)} samples to {output_path}")
    
    def print_stats(self):
        """Print processing statistics."""
        print("\n=== Processing Statistics ===")
        for key, val in sorted(self.stats.items()):
            print(f"  {key}: {val}")
        if self.stats['total'] > 0:
            rate = self.stats['success'] / self.stats['total'] * 100
            print(f"\n  Success rate: {rate:.1f}%")


def split_dataset(input_path: str, output_dir: str, 
                  train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/dev/test."""
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))
    
    splits = {
        'train': data[:train_end],
        'dev': data[train_end:dev_end],
        'test': data[dev_end:]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    for name, split_data in splits.items():
        path = os.path.join(output_dir, f'faithfulness_qa_{name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"  {name}: {len(split_data)} samples -> {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--source', type=str, default='squad', choices=['squad', 'triviaqa', 'both'])
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--pilot', action='store_true', help='Run pilot with 500 samples')
    args = parser.parse_args()
    
    if args.pilot:
        args.max_samples = 500
    
    entity_bank_dir = os.path.join(args.data_dir, 'entity_bank')
    builder = FaithfulnessQABuilder(entity_bank_dir)
    
    all_results = []
    
    if args.source in ('squad', 'both'):
        output = os.path.join(args.data_dir, 'faithfulness_qa_squad_raw.jsonl')
        results = builder.build_from_squad(output, max_samples=args.max_samples)
        all_results.extend(results)
    
    if args.source in ('triviaqa', 'both'):
        output = os.path.join(args.data_dir, 'faithfulness_qa_triviaqa_raw.jsonl')
        results = builder.build_from_triviaqa(output, max_samples=args.max_samples)
        all_results.extend(results)
    
    builder.print_stats()
    
    if all_results:
        combined_path = os.path.join(args.data_dir, 'faithfulness_qa_all_raw.jsonl')
        with open(combined_path, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        if not args.pilot:
            split_dataset(combined_path, args.data_dir)
    
    stats_path = os.path.join(args.data_dir, 'build_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(dict(builder.stats), f, indent=2)
