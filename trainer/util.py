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