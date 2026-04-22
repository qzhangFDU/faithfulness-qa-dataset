"""
Faithfulness-TriviaQA Builder (streaming mode to avoid disk cache)
Processes TriviaQA samples one-by-one in streaming mode.
Stops once target number of successful samples is reached.
"""

import json
import os
import random
import logging
from collections import Counter

import spacy
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entity_bank import EntityBank
from build_dataset import FaithfulnessQABuilder, split_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(42)


def build_triviaqa_streaming(data_dir: str, target_samples: int = 50000):
    """Build Faithfulness-TriviaQA using streaming to avoid disk cache issues."""
    from datasets import load_dataset
    
    entity_bank_dir = os.path.join(data_dir, 'entity_bank')
    builder = FaithfulnessQABuilder(entity_bank_dir)
    
    logger.info(f"Loading TriviaQA in streaming mode (target: {target_samples} samples)...")
    ds = load_dataset('trivia_qa', 'rc', split='train', streaming=True)
    
    results = []
    processed = 0
    
    for sample in tqdm(ds, desc="Building Faithfulness-TriviaQA"):
        processed += 1
        result = builder.process_triviaqa_sample(sample)
        if result:
            results.append(result)
        
        # Stop once we have enough
        if len(results) >= target_samples:
            logger.info(f"Reached target of {target_samples} samples after processing {processed} inputs")
            break
        
        # Save intermediate results every 10K successes
        if len(results) > 0 and len(results) % 10000 == 0:
            logger.info(f"  Progress: {len(results)} successful / {processed} processed")
    
    # Save results
    output_path = os.path.join(data_dir, 'faithfulness_triviaqa_raw.jsonl')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(results)} samples to {output_path}")
    
    # Split into train/dev/test
    random.shuffle(results)
    n = len(results)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)
    
    splits = {
        'train': results[:train_end],
        'dev': results[train_end:dev_end],
        'test': results[dev_end:]
    }
    
    for name, split_data in splits.items():
        path = os.path.join(data_dir, f'faithfulness_triviaqa_{name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"  {name}: {len(split_data)} samples -> {path}")
    
    builder.print_stats()
    
    # Save stats
    stats_path = os.path.join(data_dir, 'triviaqa_build_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(dict(builder.stats), f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--target', type=int, default=50000)
    args = parser.parse_args()
    
    results = build_triviaqa_streaming(args.data_dir, args.target)
    print(f"\nTotal samples generated: {len(results)}")
