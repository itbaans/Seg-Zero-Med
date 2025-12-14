"""
VQA MCQ Reward Function for Seg-Zero

Computes rewards for 3-tag format:
<think>...</think>
<answer>A</answer>
<bbox>[{"bbox_2d": [x1,y1,x2,y2]}]</bbox>

Total Reward = Format (3.0) + Option (3.0) + Bbox (3.0) = Max 9.0
"""

import re
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def vqa_mcq_compute_score(predict_str: str, ground_truth: str) -> float:
    """
    Compute reward for VQA MCQ with 3-tag format.
    
    Args:
        predict_str: Model output string
        ground_truth: JSON string with correct_option, bboxes, partial_scores
        
    Returns:
        Total reward score (max 9.0)
    """
    format_reward = vqa_mcq_format_reward(predict_str)
    option_reward = vqa_mcq_option_reward(predict_str, ground_truth)
    bbox_reward = vqa_mcq_bbox_reward(predict_str, ground_truth)
    
    total = format_reward + option_reward + bbox_reward
    return total


def vqa_mcq_format_reward(predict_str: str) -> float:
    """
    Validate 3-tag format structure.
    
    Rewards:
    - <think> tag present: 1.0
    - <answer> tag with single letter: 1.0  
    - <bbox> tag with valid JSON: 1.0
    
    Max: 3.0 points
    """
    reward = 0.0
    
    # Check for <think> tag (1.0)
    think_pattern = r'<think>.*?</think>'
    if re.search(think_pattern, predict_str, re.DOTALL):
        reward += 1.0
    
    # Check for <answer> tag with single letter A-D (1.0)
    answer_pattern = r'<answer>\s*([A-D])\s*</answer>'
    if re.search(answer_pattern, predict_str, re.DOTALL):
        reward += 1.0
    
    # Check for <bbox> tag with valid JSON array (1.0)
    bbox_pattern = r'<bbox>\s*(\[.*?\])\s*</bbox>'
    bbox_match = re.search(bbox_pattern, predict_str, re.DOTALL)
    if bbox_match:
        try:
            bboxes = json.loads(bbox_match.group(1))
            # Validate bbox structure: array of objects with bbox_2d
            if isinstance(bboxes, list) and len(bboxes) > 0:
                valid_count = 0
                for bbox_obj in bboxes:
                    if isinstance(bbox_obj, dict) and 'bbox_2d' in bbox_obj:
                        bbox = bbox_obj['bbox_2d']
                        if isinstance(bbox, list) and len(bbox) == 4:
                            # Check all coordinates are numbers
                            if all(isinstance(x, (int, float)) for x in bbox):
                                valid_count += 1
                
                # Award proportional reward (at least one valid bbox gets full point)
                if valid_count > 0:
                    reward += 1.0
        except (json.JSONDecodeError, ValueError):
            pass
    
    return min(reward, 3.0)


def vqa_mcq_option_reward(predict_str: str, ground_truth: str) -> float:
    """
    Reward for selecting correct/partially correct option.
    Uses partial correctness scores for fine-grained feedback.
    
    Max: 3.0 points
    """
    try:
        gt_data = json.loads(ground_truth)
        partial_scores = gt_data['partial_scores']
        
        # Extract predicted option letter
        answer_match = re.search(r'<answer>\s*([A-D])\s*</answer>', predict_str, re.DOTALL)
        if answer_match:
            predicted_option = answer_match.group(1)
            
            # Get partial score for selected option (0.0 to 1.0)
            score = partial_scores.get(predicted_option, 0.0)
            
            # Scale to max 3.0
            return score * 3.0
    except (json.JSONDecodeError, KeyError):
        pass
    
    return 0.0


def vqa_mcq_bbox_reward(predict_str: str, ground_truth: str) -> float:
    """
    Reward for bbox predictions using IoU and L1 distance.
    Uses Hungarian algorithm for optimal matching of multiple polyps.
    
    Criteria per bbox:
    - IoU > 0.5: +1.0
    - L1 distance < 10: +1.0
    Total per bbox: 2.0, scaled to max 3.0 overall
    
    Max: 3.0 points
    """
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = gt_data['bboxes']  # List of [x1,y1,x2,y2]
        
        # Extract predicted bboxes
        bbox_match = re.search(r'<bbox>\s*(\[.*?\])\s*</bbox>', predict_str, re.DOTALL)
        if not bbox_match:
            return 0.0
        
        pred_data = json.loads(bbox_match.group(1))
        pred_bboxes = [item['bbox_2d'] for item in pred_data]
        
        # Limit to reasonable number
        MAX_OBJECTS = 10
        if len(pred_bboxes) > MAX_OBJECTS:
            pred_bboxes = pred_bboxes[:MAX_OBJECTS]
        if len(gt_bboxes) > MAX_OBJECTS:
            gt_bboxes = gt_bboxes[:MAX_OBJECTS]
        
        # Convert to numpy
        pred_bboxes = np.array(pred_bboxes, dtype=float)
        gt_bboxes = np.array(gt_bboxes, dtype=float)
        
        # Compute IoU and L1 distance matrices
        iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # Shape: (M, N)
        l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)  # Shape: (M, N)
        
        # Reward criteria (binary: pass or fail)
        iou_reward = (iou_matrix > 0.5).astype(float)  # 1.0 if IoU > 0.5
        l1_reward = (l1_matrix < 10).astype(float)  # 1.0 if L1 < 10
        
        # Cost matrix for Hungarian matching (minimize cost)
        # Max reward per match is 2.0, so cost = 2.0 - (iou_reward + l1_reward)
        cost_matrix = 2.0 - (iou_reward + l1_reward)
        
        # Hungarian algorithm finds optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Calculate total reward from matched pairs
        total_reward = len(row_indices) * 2.0 - cost_matrix[row_indices, col_indices].sum()
        
        # Normalize by max(#pred, #gt) to penalize missing/extra predictions
        max_count = max(len(pred_bboxes), len(gt_bboxes))
        avg_reward = total_reward / max_count if max_count > 0 else 0.0
        
        # Scale from [0, 2.0] to [0, 3.0]
        return min(avg_reward * 1.5, 3.0)
        
    except (json.JSONDecodeError, KeyError, ValueError, IndexError):
        return 0.0


def batch_iou(boxes1, boxes2):
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: numpy array of shape (M, 4) - [x1, y1, x2, y2]
        boxes2: numpy array of shape (N, 4) - [x1, y1, x2, y2]
        
    Returns:
        IoU matrix of shape (M, N)
    """
    # Split into coordinates
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # Each (M, 1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # Each (N, 1)
    
    # Compute intersection
    xA = np.maximum(x11, np.transpose(x21))  # (M, N)
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    
    # Compute union
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M, 1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N, 1)
    unionArea = box1Area + np.transpose(box2Area) - interArea
    
    # IoU
    iou = interArea / unionArea  # (M, N)
    return iou


def batch_l1_distance(boxes1, boxes2):
    """
    Compute L1 distance matrix (mean absolute error per coordinate).
    
    Args:
        boxes1: numpy array of shape (M, 4)
        boxes2: numpy array of shape (N, 4)
        
    Returns:
        L1 distance matrix of shape (M, N)
    """
    boxes1 = boxes1[:, np.newaxis, :]  # (M, 1, 4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1, N, 4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M, N)


# Test function
if __name__ == "__main__":
    # Test case 1: Perfect prediction
    predict_perfect = """<think>The polyp appears red-white and medium-sized in the center region.</think>
<answer>A</answer>
<bbox>[{"bbox_2d": [150, 200, 450, 500]}]</bbox>"""
    
    gt_perfect = json.dumps({
        "correct_option": "A",
        "bboxes": [[150, 200, 450, 500]],
        "partial_scores": {"A": 1.0, "B": 0.7, "C": 0.0, "D": 0.6}
    })
    
    score = vqa_mcq_compute_score(predict_perfect, gt_perfect)
    print(f"Test 1 - Perfect prediction: {score:.2f} / 9.0")
    print(f"  Format: {vqa_mcq_format_reward(predict_perfect):.2f}")
    print(f"  Option: {vqa_mcq_option_reward(predict_perfect, gt_perfect):.2f}")
    print(f"  Bbox: {vqa_mcq_bbox_reward(predict_perfect, gt_perfect):.2f}")
    
    # Test case 2: Partial correctness
    predict_partial = """<think>It looks pinkish</think>
<answer>B</answer>
<bbox>[{"bbox_2d": [140, 190, 460, 510]}]</bbox>"""
    
    score2 = vqa_mcq_compute_score(predict_partial, gt_perfect)
    print(f"\nTest 2 - Partial correctness (B selected, score=0.7): {score2:.2f} / 9.0")
    print(f"  Format: {vqa_mcq_format_reward(predict_partial):.2f}")
    print(f"  Option: {vqa_mcq_option_reward(predict_partial, gt_perfect):.2f}")
    print(f"  Bbox: {vqa_mcq_bbox_reward(predict_partial, gt_perfect):.2f}")
    
    # Test case 3: Wrong option, good bbox
    predict_wrong = """<think>Yellow polyp</think>
<answer>C</answer>
<bbox>[{"bbox_2d": [148, 198, 452, 502]}]</bbox>"""
    
    score3 = vqa_mcq_compute_score(predict_wrong, gt_perfect)
    print(f"\nTest 3 - Wrong option (C), good bbox: {score3:.2f} / 9.0")
    print(f"  Format: {vqa_mcq_format_reward(predict_wrong):.2f}")
    print(f"  Option: {vqa_mcq_option_reward(predict_wrong, gt_perfect):.2f}")
    print(f"  Bbox: {vqa_mcq_bbox_reward(predict_wrong, gt_perfect):.2f}")
    
    # Test case 4: Multiple polyps
    predict_multi = """<think>Two polyps visible</think>
<answer>A</answer>
<bbox>[{"bbox_2d": [100, 50, 250, 180]}, {"bbox_2d": [150, 200, 450, 500]}]</bbox>"""
    
    gt_multi = json.dumps({
        "correct_option": "A",
        "bboxes": [[100, 50, 250, 180], [150, 200, 450, 500]],
        "partial_scores": {"A": 1.0, "B": 0.6, "C": 0.6, "D": 0.0}
    })
    
    score4 = vqa_mcq_compute_score(predict_multi, gt_multi)
    print(f"\nTest 4 - Multiple polyps: {score4:.2f} / 9.0")
    print(f"  Format: {vqa_mcq_format_reward(predict_multi):.2f}")
    print(f"  Option: {vqa_mcq_option_reward(predict_multi, gt_multi):.2f}")
    print(f"  Bbox: {vqa_mcq_bbox_reward(predict_multi, gt_multi):.2f}")
