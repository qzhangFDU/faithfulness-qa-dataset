"""Quality analysis and statistics for Faithfulness-QA dataset."""

import json
import random
import os
from collections import Counter
from pathlib import Path

random.seed(42)

def analyze_dataset(data_dir: str):
    """Comprehensive quality analysis of the generated dataset."""
    
    # Load all data
    all_path = os.path.join(data_dir, 'faithfulness_qa_all_raw.jsonl')
    with open(all_path) as f:
        data = [json.loads(l) for l in f]
    
    print(f"{'='*60}")
    print(f"Faithfulness-QA Dataset Quality Report")
    print(f"{'='*60}")
    print(f"\n## 1. Dataset Size")
    print(f"  Total samples: {len(data)}")
    
    # Split sizes
    for split in ['train', 'dev', 'test']:
        path = os.path.join(data_dir, f'faithfulness_qa_{split}.jsonl')
        if os.path.exists(path):
            with open(path) as f:
                n = sum(1 for _ in f)
            print(f"  {split}: {n}")
    
    # Entity type distribution
    print(f"\n## 2. Entity Type Distribution")
    types = Counter(s['entity_type'] for s in data)
    for t, c in types.most_common():
        print(f"  {t}: {c:,} ({c/len(data)*100:.1f}%)")
    
    # Context length stats
    print(f"\n## 3. Context Length Statistics")
    ctx_lens = [len(s['original_context']) for s in data]
    print(f"  Mean: {sum(ctx_lens)/len(ctx_lens):.0f} chars")
    print(f"  Min: {min(ctx_lens)} chars")
    print(f"  Max: {max(ctx_lens)} chars")
    ctx_lens_sorted = sorted(ctx_lens)
    print(f"  Median: {ctx_lens_sorted[len(ctx_lens_sorted)//2]} chars")
    
    # Answer length stats
    print(f"\n## 4. Answer/Entity Length Statistics")
    orig_lens = [len(s['original_entity']) for s in data]
    repl_lens = [len(s['replacement_entity']) for s in data]
    print(f"  Original entity - Mean: {sum(orig_lens)/len(orig_lens):.1f}, Median: {sorted(orig_lens)[len(orig_lens)//2]}")
    print(f"  Replacement entity - Mean: {sum(repl_lens)/len(repl_lens):.1f}, Median: {sorted(repl_lens)[len(repl_lens)//2]}")
    
    # Quality checks on random sample
    print(f"\n## 5. Quality Spot Checks (200 random samples)")
    sample = random.sample(data, min(200, len(data)))
    
    checks = {
        'entity_in_modified': 0,
        'original_entity_not_in_modified': 0,
        'context_changed': 0,
        'reasonable_length': 0,
    }
    
    for s in sample:
        # Check 1: Replacement entity exists in modified context
        if s['replacement_entity'] in s['modified_context']:
            checks['entity_in_modified'] += 1
        
        # Check 2: Original entity NOT in modified context (replaced successfully)
        if s['original_entity'].lower() not in s['modified_context'].lower():
            checks['original_entity_not_in_modified'] += 1
        
        # Check 3: Context actually changed
        if s['original_context'] != s['modified_context']:
            checks['context_changed'] += 1
        
        # Check 4: Context length is reasonable (within 2x of original)
        ratio = len(s['modified_context']) / max(len(s['original_context']), 1)
        if 0.5 <= ratio <= 2.0:
            checks['reasonable_length'] += 1
    
    n_sample = len(sample)
    print(f"  Replacement entity present in modified context: {checks['entity_in_modified']}/{n_sample} ({checks['entity_in_modified']/n_sample*100:.1f}%)")
    print(f"  Original entity removed from modified context: {checks['original_entity_not_in_modified']}/{n_sample} ({checks['original_entity_not_in_modified']/n_sample*100:.1f}%)")
    print(f"  Context actually changed: {checks['context_changed']}/{n_sample} ({checks['context_changed']/n_sample*100:.1f}%)")
    print(f"  Reasonable length ratio: {checks['reasonable_length']}/{n_sample} ({checks['reasonable_length']/n_sample*100:.1f}%)")
    
    # Show example samples
    print(f"\n## 6. Example Samples")
    for i, s in enumerate(sample[:5]):
        print(f"\n--- Example {i+1} ({s['entity_type']}) ---")
        print(f"  Q: {s['question']}")
        print(f"  Original answer: {s['original_answer']}")
        print(f"  Faithful answer: {s['faithful_answer']}")
        print(f"  Entity: \"{s['original_entity']}\" -> \"{s['replacement_entity']}\"")
        # Show context snippet
        idx = s['modified_context'].find(s['replacement_entity'])
        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(s['modified_context']), idx + len(s['replacement_entity']) + 50)
            snippet = s['modified_context'][start:end]
            print(f"  Context snippet: ...{snippet}...")
    
    # Save statistics JSON
    stats = {
        'total_samples': len(data),
        'entity_type_distribution': dict(types.most_common()),
        'context_length': {
            'mean': sum(ctx_lens)/len(ctx_lens),
            'min': min(ctx_lens),
            'max': max(ctx_lens),
            'median': ctx_lens_sorted[len(ctx_lens_sorted)//2]
        },
        'quality_checks': {k: f"{v}/{n_sample} ({v/n_sample*100:.1f}%)" for k, v in checks.items()},
        'build_stats_file': os.path.join(data_dir, 'build_stats.json')
    }
    
    stats_path = os.path.join(data_dir, '..', 'results', 'statistics.json')
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n\nStatistics saved to {stats_path}")
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    analyze_dataset(args.data_dir)
