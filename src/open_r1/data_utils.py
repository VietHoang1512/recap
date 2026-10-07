
import functools
from typing import Any, Callable, Optional, Sequence, TypeVar, Union
from transformers import PreTrainedTokenizerBase


def is_conversational(example: dict[str, Any]) -> bool:
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages,
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True
    return False

def apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    r"""
    Apply a chat template to a conversational example along with the schema for a list of functions in `tools`.

    For more details, see [`maybe_apply_chat_template`].
    """
    # Check that the example has the correct keys
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "label"]
    # TODO: Branch for DPO/SFT/RL
    example = { k: v for k, v in example.items() if v is not None}
    example_keys = {key for key in example.keys() if key in supported_keys}
    if example_keys not in [
        {"messages"},  # language modeling
        {"prompt"},  # prompt-only
        {"prompt", "completion"},  # prompt-completion
        {"prompt", "chosen", "rejected"},  # preference
        {"chosen", "rejected"},  # preference with implicit prompt
        {"prompt", "completion", "label"},  # unpaired preference
    ]:
        raise KeyError(f"Invalid keys in the example: {example_keys}")
    
    # Apply the chat template to the whole conversation
    if "messages" in example:
        messages = tokenizer.apply_chat_template(example["messages"], tools=tools, tokenize=False)

    # Apply the chat template to the prompt, adding the generation prompt
    if "prompt" in example:
        last_role = example["prompt"][-1]["role"]
        if last_role == "user":
            add_generation_prompt = True
            continue_final_message = False
        elif last_role == "assistant":
            add_generation_prompt = False
            continue_final_message = True
        else:
            raise ValueError(f"Invalid role in the last message: {last_role}")
        prompt = tokenizer.apply_chat_template(
            example["prompt"],
            tools=tools,
            continue_final_message=continue_final_message,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Apply the chat template to the entire prompt + completion
    if "prompt" in example:  # explicit prompt and prompt-completion case
        if "chosen" in example and example["chosen"]:
            prompt_chosen = tokenizer.apply_chat_template(
                example["prompt"] + example["chosen"], tools=tools, tokenize=False
            )
            chosen = prompt_chosen[len(prompt) :]
        if "rejected" in example and "prompt" in example:  # explicit prompt
            prompt_rejected = tokenizer.apply_chat_template(
                example["prompt"] + example["rejected"], tools=tools, tokenize=False
            )
            rejected = prompt_rejected[len(prompt) :]
        if "completion" in example:
            prompt_completion = tokenizer.apply_chat_template(
                example["prompt"] + example["completion"], tools=tools, tokenize=False
            )
            completion = prompt_completion[len(prompt) :]
    else:  # implicit prompt case
        if "chosen" in example:
            chosen = tokenizer.apply_chat_template(example["chosen"], tools=tools, tokenize=False)
        if "rejected" in example:
            rejected = tokenizer.apply_chat_template(example["rejected"], tools=tools, tokenize=False)

    # Ensure that the prompt is the initial part of the prompt-completion string
    if "prompt" in example:
        error_message = (
            "The chat template applied to the prompt + completion does not start with the chat template applied to "
            "the prompt alone. This can indicate that the chat template is not supported by TRL."
            "\n**Prompt**:\n{}\n\n**Prompt + Completion**:\n{}"
        )
        if "chosen" in example and not prompt_chosen.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_chosen))
        if "rejected" in example and not prompt_rejected.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_rejected))
        if "completion" in example and not prompt_completion.startswith(prompt):
            raise ValueError(error_message.format(prompt, prompt_completion))

    # Extract the completion by removing the prompt part from the prompt-completion string
    output = {}
    if "messages" in example:
        output["text"] = messages
    if "prompt" in example:
        output["prompt"] = prompt
    if "chosen" in example:
        output["chosen"] = chosen
    if "rejected" in example:
        output["rejected"] = rejected
    if "completion" in example:
        output["completion"] = completion
    if "label" in example:
        output["label"] = example["label"]

    return output



def maybe_apply_chat_template(
    example: dict[str, list[dict[str, str]]],
    tokenizer: PreTrainedTokenizerBase,
    tools: Optional[list[Union[dict, Callable]]] = None,
) -> dict[str, str]:
    if is_conversational(example):
        return apply_chat_template(example, tokenizer, tools)
    else:
        return example
    



if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )
    tools = [{"name": "dummy_tool", "description": "A placeholder", "parameters": {}}]

    examples = [
        {"messages": [{"role": "user", "content": "Hello, model!"}]},
        
        # 1. Language modeling (messages)
        {"messages": [{"role": "user", "content": [  {"type": "image"},{"type": "text", "text": "Hello, model!"}]}]},
        # 2. Prompt-only
        {"prompt": [{"role": "user", "content": "What is the capital of France?"}]},
        # 3. Prompt + Completion
        {
            "prompt": [{"role": "user", "content": "Tell me a joke."}],
            "completion": [{"role": "assistant", "content": "Why did the chicken cross the road? To get to the other side!"}]
        },
        # 4. Preference (prompt + chosen + rejected)
        {
            "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text":"????"}]}],
            "chosen": [{"role": "assistant", "content": "Hello there!"}],
            "rejected": [{"role": "assistant", "content": "Yo!"}]
        },
        # 5. Implicit prompt preference (chosen + rejected)
        {
            "chosen": [{"role": "assistant", "content": "Affirmative."}],
            "rejected": [{"role": "assistant", "content": "Nope."}]
        },
        # 6. Unpaired preference (prompt + completion + label)
        {
            "prompt": [{"role": "user", "content": "Rate this response."}],
            "completion": [{"role": "assistant", "content": "It was excellent."}],
            "label": 1
        },
        {"prompt":[{"role": "user", "content":[{'type': 'image'}, {'text': '/nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags.', 'type': 'text'}]}]}
    ]

    for idx, ex in enumerate(examples):
        formatted = apply_chat_template(ex, tokenizer, tools=tools)
        print(f"\nExample {idx} formatted output:")
        for k, v in formatted.items():
            print(f"->  {k}: {v}")


    from transformers import AutoProcessor

    # 1) Load the Qwen‑VL‑2.5 processor
    processor = AutoProcessor.from_pretrained(        "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )

    # 2) Prepare a 'messages' example
    example = {
        "messages": [
            {"role": "user",      "content": "Hi, can you help me plan a trip to Kyoto?"},
            {"role": "assistant", "content": "Sure! When are you thinking of going?"},
        ]
    }

    # 3) Apply the chat template
    features = processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)

    # 4) Inspect the result
    print(features)
    example = {
        "messages": [
            {"role": "user",      "content": [{"type": "text", "text":"Hi, can you help me plan a trip to Kyoto?"}]},
            {"role": "assistant", "content": [{"type": "text", "text":"Sure! When are you thinking of going?"}]},
        ]
    }
    # 3) Apply the chat template
    features = processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)

    # 4) Inspect the result
    print(features)