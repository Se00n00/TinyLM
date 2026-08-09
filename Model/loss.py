import torch
import torch.nn.functional as F

# for Pretraining & Instruction fine-tunning:
# X: [B, L, VOCAB_SIZE] --> [B*L, VOCAB_SIZE], Y: [B, L] --> [B*L]
# 
# Pretraining >
# loss = torch.nn.functional.cross_entropy(X, Y, ignore_index = None)
# 
# Instruction Fine-tunning >
# loss = torch.nn.functional.cross_entropy(X, Y, ignore_index = prompt_tokens_index)

def get_cross_entropy_loss(X:torch.tensor, Y:torch.tensor, pad_index = 60000):
    # X:[B, L, VOCAB_SIZE] --> [B*L, VOCAB_SIZE], Y: [B, L] --> [B*L]

    loss = F.cross_entropy(
        X.view(-1, X.size(-1)), Y.view(-1), ignore_index=pad_index
    )
    
    return loss
    
def token_level_kd_loss(teacher_output:torch.tensor, student_output:torch.tensor, loss_coefficient=0.3, temperature = 1):

    vocab_size = student_output.size(-1)
    teacher_logits = teacher_output.view(-1, vocab_size)
    student_logits = student_output.view(-1, vocab_size)
    
    # 1. Soft targets (KL Divergence)
    # Scale logits by temperature before applying functions
    teacher_soft_prob = F.softmax(teacher_logits / temperature, dim=-1)
    student_soft_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    
    # Compute KL Div 
    kl_div = F.kl_div(student_soft_log_prob, teacher_soft_prob, reduction='batchmean')
    
    # Scale the KL loss back up to match the magnitude of standard gradients
    kl_loss = kl_div * (temperature ** 2)
    
    # 2. Hard targets (Cross Entropy from Teacher Distribution)
    # PyTorch's cross_entropy expects raw unnormalized logits for the input (student)
    # and soft probabilities for the target (teacher)
    teacher_hard_prob = F.softmax(teacher_logits, dim=-1)
    cross_entropy_loss = F.cross_entropy(student_logits, teacher_hard_prob)
    
    # 3. Combine losses
    return (1 - loss_coefficient) * cross_entropy_loss + loss_coefficient * kl_loss