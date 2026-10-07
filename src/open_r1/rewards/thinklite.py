from datetime import datetime
from math_verify import parse, verify
import re
import json
import os
from .grader import grade_answer

def extract_letters(text): # for RAVEN
    pattern = r'(^|\s|\[|\()([A-H])(\s|\]|\)|$)'
    matches = re.findall(pattern, text)
    return [match[1] for match in matches]

def accuracy_thinklite_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # for content, sol in zip(contents, solution):
    for i, (content, sol) in enumerate(zip(contents, solution)):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # student_answer = re.sub(r"\s*(<|>|/)\s*", r"\1", student_answer) # borrow from PAPO, check this
                if student_answer == ground_truth:
                    reward = 1.0

                if extract_letters(student_answer)[-1] == ground_truth:
                    reward = 1.0
                if grade_answer(answer, ground_truth):
                    reward = 1.0
                # Compare the extracted answers

            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            problem = kwargs.get("problem")[i][0]["content"]
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"problem: {problem}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def format_think_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
