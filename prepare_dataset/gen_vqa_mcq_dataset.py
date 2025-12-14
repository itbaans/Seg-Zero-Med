import json
import cv2
import numpy as np
from datasets import Dataset, Features, Value, Image as HFImage
from PIL import Image
import os

# Question template
QUESTION_TEMPLATE = """Question: {question_text}

Options:
{options_list}

Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D) inside <answer>...</answer> tags.
3. Output the bounding box(es) for all polyps in <bbox></bbox> tags as a JSON array.
4. No extra information outside these tags.

Example format:
<think>reasoning process here</think>
<answer>A</answer>
<bbox>[{{"bbox_2d": [x1, y1, x2, y2]}}]</bbox>
"""


def format_question_prompt(item):
    """
    Generate formatted question prompt from MCQ item.
    
    Args:
        item: Dict with 'question' and 'options' keys
        
    Returns:
        Formatted prompt string
    """
    # Format options in sorted order
    options_text = "\n".join([
        f"{opt['option_id']}: {opt['text']}" 
        for opt in sorted(item['options'], key=lambda x: x['option_id'])
    ])
    
    return QUESTION_TEMPLATE.format(
        question_text=item['question'],
        options_list=options_text
    )


def load_and_resize_image(image_path, target_size=840):
    """
    Load image and resize to target size.
    
    Args:
        image_path: Path to image file
        target_size: Target dimension (will be square)
        
    Returns:
        PIL Image
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(img)


def get_image_size(image_path):
    """Get original image dimensions."""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    return (height, width)


def scale_bboxes(bboxes, original_size, target_size=(840, 840)):
    """
    Scale bbox coordinates from original to target size.
    
    Args:
        bboxes: List of [x1, y1, x2, y2] bboxes
        original_size: (height, width) of original image
        target_size: (height, width) of target image
        
    Returns:
        List of scaled bboxes
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    
    x_factor = target_w / orig_w
    y_factor = target_h / orig_h
    
    scaled = []
    for bbox in bboxes:
        scaled.append([
            int(bbox[0] * x_factor + 0.5),
            int(bbox[1] * y_factor + 0.5),
            int(bbox[2] * x_factor + 0.5),
            int(bbox[3] * y_factor + 0.5)
        ])
    return scaled


def convert_vqa_to_segzero(mcq_json_path, output_dir, image_resize=840):
    """
    Convert MCQ VQA JSON to Seg-Zero HuggingFace dataset format.
    
    Args:
        mcq_json_path: Path to diverse_compound_mcq_dataset.json
        output_dir: Output directory for dataset
        image_resize: Target image size (default 840)
        
    Returns:
        HuggingFace Dataset
    """
    # Load MCQ data
    print(f"Loading MCQ data from {mcq_json_path}...")
    with open(mcq_json_path, 'r') as f:
        mcq_data = json.load(f)
    
    print(f"Found {len(mcq_data)} samples")
    
    # Prepare lists for dataset
    id_list = []
    problem_list = []
    solution_list = []
    image_list = []
    img_height_list = []
    img_width_list = []
    
    for idx, item in enumerate(mcq_data):
        print(f"Processing {idx+1}/{len(mcq_data)}: {item['image_id']}")
        
        # Format problem with question template
        problem = format_question_prompt(item)
        
        # Get original image size for bbox scaling
        original_size = get_image_size(item['image_path'])
        
        # Scale bboxes to target size
        scaled_bboxes = scale_bboxes(
            item['bbox'], 
            original_size, 
            (image_resize, image_resize)
        )
        
        # Format solution with correct option + bboxes (NO points)
        solution = {
            "correct_option": item['correct_option'],
            "bboxes": scaled_bboxes,
            "partial_scores": {
                opt['option_id']: opt['partial_score'] 
                for opt in item['options']
            }
        }
        
        # Load and resize image
        image = load_and_resize_image(item['image_path'], image_resize)
        
        # Add to lists
        id_list.append(item['image_id'])
        problem_list.append(problem)
        solution_list.append(json.dumps(solution))
        image_list.append(image)
        img_height_list.append(image_resize)
        img_width_list.append(image_resize)
    
    # Create HuggingFace dataset
    print("\nCreating HuggingFace dataset...")
    dataset = Dataset.from_dict(
        {
            'id': id_list,
            'problem': problem_list,
            'solution': solution_list,
            'image': image_list,
            'img_height': img_height_list,
            'img_width': img_width_list
        },
        features=Features({
            'id': Value('string'),
            'problem': Value('string'),
            'solution': Value('string'),
            'image': HFImage(),
            'img_height': Value('int64'),
            'img_width': Value('int64')
        })
    )
    
    # Save to disk
    print(f"\nSaving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    
    print(f"✓ Dataset saved successfully!")
    print(f"  Total samples: {len(dataset)}")
    
    return dataset


if __name__ == "__main__":
    # Configuration
    MCQ_JSON_PATH = "d:/data_building/datasets/diverse_compound_mcq_dataset_regenerated.json"
    OUTPUT_DIR = "d:/data_building/Seg-Zero/data/vqa_mcq_840"
    IMAGE_RESIZE = 840
    
    # Convert dataset
    dataset = convert_vqa_to_segzero(
        mcq_json_path=MCQ_JSON_PATH,
        output_dir=OUTPUT_DIR,
        image_resize=IMAGE_RESIZE
    )
    
    # Print sample
    print("\n" + "="*60)
    print("SAMPLE ENTRY:")
    print("="*60)
    sample = dataset[0]
    print(f"ID: {sample['id']}")
    print(f"\nPROBLEM:\n{sample['problem'][:500]}...")
    print(f"\nSOLUTION:\n{sample['solution']}")
    print(f"\nImage size: {sample['img_height']}x{sample['img_width']}")
