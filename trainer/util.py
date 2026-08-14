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


def preprocess_text_generation(example):
    return {
        "messages": {"text": example["text"]},
    }

def preprocess_ift(example):
    return {
        "messages": [
            {"role": "system", "content": example["text"]},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": example["text"]},
        ],
    }

def preprocess_rft(example):
    return {
        "messages": [
            {"role": "system", "content": example["text"]},
            {"role": "user", "content": example["text"]},
            {"role": "assistant", "content": f"<|THINK|>{example['text']}<|THINK|>{example['text']}"},
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