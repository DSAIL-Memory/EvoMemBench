"""Sample 170 tasks from webwalkerqa_main.jsonl with random seed 42."""

import json
import random
import os


def sample_webwalkerqa(
    input_file='webwalkerqa_main.jsonl',
    output_file='webwalkerqa_subset_170.jsonl',
    sample_size=170,
    random_seed=42
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    random.seed(random_seed)
    
    # Read data
    print(f"Reading {input_file}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"✓ Loaded {len(data)} tasks")
    
    if sample_size > len(data):
        raise ValueError(f"Sample size ({sample_size}) exceeds total size ({len(data)})")
    
    # Sample
    print(f"Sampling {sample_size} tasks (seed={random_seed})...")
    sampled_data = random.sample(data, sample_size)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(sampled_data)} tasks to {output_file}")
    
    # Preview
    print("\nFirst 3 samples:")
    for i, item in enumerate(sampled_data[:3], 1):
        question = item.get('question', 'N/A')
        preview = question[:80] + '...' if len(question) > 80 else question
        print(f"  {i}. {preview}")


if __name__ == "__main__":
    try:
        sample_webwalkerqa()
    except Exception as e:
        print(f"\n Error: {e}")
        exit(1)
