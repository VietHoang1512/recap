import os
import re
from math_verify import parse, verify
import json
from datetime import datetime

def extract_bbox(response: str):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            #print(content_str_corrected)
            #print("JSON error!")
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

# list1: groundtruth
# list2: student
def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    
    matched_list1_indices = set()

    # start from the highest-confidence one, traverse student bboxs
    for bbox2 in list2_sorted:
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            # meaning that the index is not already matched to a previous bbox
            if i not in matched_list1_indices:
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
        else:
            iou_results.append((0, bbox2['Confidence']))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)]
    return iou_results

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes

def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0: # mean that this is not matched, should be less confident
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt 
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward





def accuracy_iou_coco6k_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    #print("I am writing: ", contents[0], ", gt: ", solution, " I am done.")
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        
        student_answer_bbox = []
        ground_truth_bbox = []
        content_match = []
        student_answer = []
        iou_results = []
        show_flage = 0
        
        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                show_flage = 1
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                # fix format error
                student_answer = student_answer.replace("[[",'[')  
                student_answer = student_answer.replace("]]",']')  
                student_answer = student_answer.replace("\n",'')  
                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                # get a list of bbox using json
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.0
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    ### new iou reward
                    reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
                    if reward>1:
                        reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        
        # import pdb; pdb.set_trace()
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage==1:
                    f.write(f"content_match: {content_match}\n")
                    f.write(f"student_answer: {student_answer}\n")
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
        
    return rewards

        
def iou_bbox_reward(completions, solution, **kwargs):
    """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
    import re
    import os
    from datetime import datetime
    import json


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
        
        input_height = int(image_grid_thw[1]*14)
        input_width = int(image_grid_thw[2]*14)
        # image_width = float(re.findall(image_width_tag_pattern, sol, re.DOTALL)[-1])
        # image_height = float(re.findall(image_height_tag_pattern, sol, re.DOTALL)[-1])
        sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
        
        sol = json.loads(sol.strip())
        reward = 0.0
        bbox=None
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)) / input_width, int(bbox_match.group(2)) / input_height, int(bbox_match.group(3)) / input_width, int(bbox_match.group(4)) / input_height]
                    # bbox = resize_bbox(bbox, input_height, input_width, image_height, image_width)
                    # if iou(bbox, sol) > 0.5:
                    #     reward = 1.0
                    reward = calculate_iou(bbox, sol)
        except Exception:
            pass  # Continue to next verification method if this fails
        if reward > 0.5:       
            rewards.append(1.)
        else:
            rewards.append(0.)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            problem = kwargs.get("problem")[i][0]["content"]
            if reward <= 1.0:  # this condition can be changed for debug
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} IOU reward: {reward} -------------\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution bbox: {bbox}\n") 
                    f.write(f"Solution: {sol}\n") 
    return rewards

if __name__=="__main__":
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

    ans = [0.1956521739130435, 0.29411764705882354, 0.7872670807453416, 0.3634453781512605]
    sol =  [0.20203804347826088, 0.2775065104166667, 0.78338623046875, 0.3458658854166667]
    print(calculate_iou(ans, sol))
    print(iou(ans, sol))
    