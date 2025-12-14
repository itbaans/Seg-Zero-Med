# Quick test to verify dataset loading
from datasets import load_from_disk
import json

# Load dataset
dataset = load_from_disk("d:/data_building/Seg-Zero/data/vqa_mcq_840")

print(f"Dataset loaded: {len(dataset)} samples")
print(f"Features: {dataset.features}")

# Show first sample
sample = dataset[0]
print("\n" + "="*80)
print("SAMPLE 1:")
print("="*80)
print(f"ID: {sample['id']}")
print(f"\nPROBLEM:")
print(sample['problem'])
print(f"\nSOLUTION:")
solution = json.loads(sample['solution'])
print(f"  Correct option: {solution['correct_option']}")
print(f"  Bboxes: {solution['bboxes']}")
print(f"  Partial scores: {solution['partial_scores']}")
print(f"\nImage: {sample['image'].size}")

# Show second sample
sample2 = dataset[1]
print("\n" + "="*80)
print("SAMPLE 2:")
print("="*80)
print(f"ID: {sample2['id']}")
print(f"\nPROBLEM (first 300 chars):")
print(sample2['problem'][:300] + "...")
solution2 = json.loads(sample2['solution'])
print(f"\nSOLUTION:")
print(f"  Correct option: {solution2['correct_option']}")
print(f"  Number of bboxes: {len(solution2['bboxes'])}")
print(f"  Partial scores: {solution2['partial_scores']}")
