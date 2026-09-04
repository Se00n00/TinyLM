from ast import Pass
import math

def get_lr(step, max_steps, learning_rate, warmup_steps=100, min_lr_ratio=0.1):
    """
    Cosine learning rate decay with linear warmup.
    """
    min_lr = learning_rate * min_lr_ratio
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def preprocess_text_generation(example, text_column):
    return {"text": example[text_column]}

# def preprocess_ift(example):
#     return {
#         "messages": [
#             {"role": "system", "content": example["text"]},
#             {"role": "user", "content": example["text"]},
#             {"role": "assistant", "content": example["text"]},
#         ],
#     }

def process_ift(batch):
    new_messages = []
    
    # Iterate through each multi-turn row in the dataset
    for messages_list in batch["messages"]:
        temp_message = []
        for message in messages_list:
            temp_message.append({"role": message["role"], "content": message['content']})
            if message["role"] == 'assistant':
                new_messages.append(temp_message)
                temp_message = []
    return {"messages": new_messages}

def flatten_conversations(example):
    new_messages = []
    temp_message = []

    for message in example["messages"]:
        temp_message.append({
            "role": message["role"],
            "content": message["content"],
        })

        if message["role"] == "assistant":
            new_messages.append(temp_message)
            temp_message = []

    return new_messages
    
def filter_valid(example):
    return example["valid"]
    
# def process_ift_dataset(dataset):
#     for example in dataset:
#         conversations = flatten_conversations(example)

#         for conversation in conversations:
#             yield {"messages": conversation}
            
def process_ift_dataset(dataset):
    for example in dataset:
        messages = example.get("messages")

        if not messages:
            continue

        yield {
            "messages": [
                {
                    "role": message["role"],
                    "content": message.get("content", ""),
                }
                for message in messages
            ]
        }
    
def preprocess_rft(example):
    answer = example['reasoning']
    if len(answer.split("<think>", 1)) > 1:
        reasoning = answer.split("<think>", 1)[1].split("</think>", 1)[0]
        answer = answer.split("<think>", 1)[1].split("</think>", 1)[0]
    else:
        reasoning = ''
        
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": f"<|THINK|>{reasoning}<|/THINK|>{answer}"},
        ],
    }

def preprocess_tc(example):
    return {
        "messages": [
            {"role": "available_tools", "content": example["tools"]},
            {"role": "system", "content": example["query"]},
            {"role": "assistant", "content": f"<|TOOL_CALLS|>{example['answers']}<|/TOOL_CALLS|>"},
        ],
    }

def preprocess_rtc(example):
    return {
        "messages": [
            {"role": "available_tools", "content": example["text"]},
            {"role": "system", "content": example["text"]},
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": f"<|THINK|>{example['text']}<|/THINK|><|TOOL_CALLS|>{example['text']}<|/TOOL_CALLS|>{example['text']}"},
        ],
    }

def process_iftc(example):
    from_replc = {'system':'system', 'human': 'user', 'gpt':'assistant'}
    return {
        "messages": [
            {"role":from_replc.get(msg['from'], 'assistant'), "content": msg.get("value", '')}
            for msg in example['conversations']
        ]
    }
    

    
import json


def process_tc(example):
    messages = json.loads(example["messages"])
    functions = json.loads(example["functions"])

    processed = []
    {
        "role": "available_tools",
        "content": json.dumps(
            functions,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            processed.append({
                "role": "system",
                "content": content,
            })

        elif role == "user":
            processed.append({
                "role": "user",
                "content": content,
            })

        elif role == "function_call":
            processed.append({
                "role": "assistant",
                "content": (
                    "<|TOOL_CALLS|>"
                    + content
                    + "<|/TOOL_CALLS|>"
                ),
            })

        elif role == "function_response":
            processed.append({
                "role": "tool",
                "content": content,
            })

        elif role == "assistant":
            processed.append({
                "role": "assistant",
                "content": content,
            })

    return {
        "messages": processed
    }
    
def parse_content(content):
    if not isinstance(content, str):
        return content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            import ast
            return ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return None


def process_warmup(example):
    change = {
        "tool_results": "tool",
        "tool": "tool",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
        "available_tools": "available_tools",
    }

    tools = next(
        (
            msg.get("content")
            for msg in example["messages"]
            if msg.get("role") == "available_tools"
        ),
        None,
    )

    available_tools = parse_content(tools)

    if tools is not None and available_tools is None:
        return {"valid": False}
        
    available_tools = json.dumps(
        available_tools,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return {
        "messages": [
            {
                "role": change[msg["role"]],
                "content": (
                    available_tools
                    if msg["role"] == "available_tools"
                    else msg["content"]
                ),
            }
            for msg in example["messages"]
        ],
        "valid": True,
    }

def process_cot(example):
    return {
        "messages": [
            {"role": "system", "content": "You are TinyLM2, created by Se00n00. You are a helpful assistant"},
            {"role": "user", "content": example['question']},
            {"role": "assistant", "content": f"<|THINK|>{example['thinking_trajectories'][0]}<|/THINK|>{example['solution']}"},
        ]
    }

def process_cot2(example):
    return {
        "messages": [
            {"role": "system", "content": "You are TinyLM2, created by Se00n00. Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions."},
            
        ]+ [{
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in example["messages"]]
    }