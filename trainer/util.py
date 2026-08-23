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
    return {
        "messages": {"text": example[text_column]},
    }

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
    

def process_ift_dataset(dataset):
    for example in dataset:
        conversations = flatten_conversations(example)

        for conversation in conversations:
            yield {"messages": conversation}
    
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
            {"role": "available_tools", "content": example["text"]},
            {"role": "system", "content": example["text"]},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": f"<|TOOL_CALLS|>{example['text']}<|/TOOL_CALLS|>{example['text']}"},
        ],
    }

def preprocess_rtc(example):
    return {
        "messages": [
            {"role": "available_tools", "content": example["text"]},
            {"role": "system", "content": example["text"]},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": f"<|THINK|>{example['text']}<|/THINK|><|TOOL_CALLS|>{example['text']}<|/TOOL_CALLS|>{example['text']}"},
        ],
    }