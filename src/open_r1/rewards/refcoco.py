import os
import re
from math_verify import parse, verify
import json
from datetime import datetime
from PIL import Image
def iou_reward(completions, solution, **kwargs):
    """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
    import re
    import os
    from datetime import datetime
    import json
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    def resize_bbox(bbox, input_height, input_width, image_height, image_width):
        bbox[0] = bbox[0] / input_width * image_width
        bbox[1] = bbox[1] / input_height * image_height
        bbox[2] = bbox[2] / input_width * image_width
        bbox[3] = bbox[3] / input_height * image_height
        return bbox
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    image_width_tag_pattern = r'<image_width>(.*?)</image_width>'
    image_height_tag_pattern = r'<image_height>(.*?)</image_height>'
    print("kwargs", kwargs)
    for i, (content, sol) in enumerate(zip(contents, solution)):
        image_grid_thw = kwargs.get("image_grid_thw")[i]
        image_path = kwargs.get("image_path")[i]
        image = Image.open(image_path)
        image_width, image_height = image.size
        # image_width = float(re.findall(image_width_tag_pattern, sol, re.DOTALL)[-1])
        # image_height = float(re.findall(image_height_tag_pattern, sol, re.DOTALL)[-1])
        
        input_height = int(image_grid_thw[1]*14)
        input_width = int(image_grid_thw[2]*14)
        
        sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
        sol = json.loads(sol.strip())
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                    # if iou(bbox, sol) > 0.5:
                    #     reward = 1.0
                    reward = iou(bbox, sol)
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            image_path = kwargs.get("image_path")[i] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[i][0]["content"]
            if reward <= 1.0:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n") 
    return rewards
        
def format_answer_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_think_reward_rec(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

if __name__=="__main__":
    
# Test examples
    examples = [
        {"completion": "<answer>42</answer>", "expected": 1.0},
        {"completion": "<answer>Hello world</answer>", "expected": 1.0},
        {"completion": "<think>abc</think><answer>Test</answer>", "expected": 0.0},
        {"completion": "<answer>Missing closing tag", "expected": 0.0},
        {"completion": "No tags", "expected": 0.0},
        {"completion": "<answer>Multi\nLine\nContent</answer>", "expected": 1.0},
    ]

    # Run tests and show results
    for ex in examples:
        reward = format_answer_reward([[{"content": ex["completion"]}]])[0]
        status = "PASS" if reward == ex["expected"] else "FAIL"
        print(f"Content: {ex['completion']!r}\n  -> Reward: {reward}, Expected: {ex['expected']} [{status}]\n")    