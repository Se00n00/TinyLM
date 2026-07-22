# 🦜 TinyLM-1-70M: Deep-Dive Custom Transformer Framework

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](#)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](#)
[![GPU Accelerated](https://img.shields.io/badge/CUDA-Accelerated-green.svg?logo=nvidia&logoColor=white)](#)
[![Pipeline: PT | IFT | PFT](https://img.shields.io/badge/Pipeline-PT%20%7C%20IFT%20%7C%20PFT-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](file:///run/media/se00n00/P/LittleParrot/GPT/LICENSE)

**TinyLM-1-70M** is a modular, high-performance, and feature-rich framework written in PyTorch for training, aligning, and deploying custom Generative Pre-trained Transformer (GPT) models. Spanning the entire life cycle of modern Large Language Models (LLMs), it provides PyTorch-native implementations of autoregressive pre-training, supervised instruction fine-tuning, vocabulary expansion, and direct preference alignment.

---

## 📐 Mathematical & Algorithmic Architecture

TinyLM-1-70M leverages a pre-normalization architecture with gated feedforward networks and optional Mixture of Experts. Below is the integrated breakdown of each layer, combining mathematical formulations, your original algorithmic pseudocode, and references to the active code.

### 1. Learned Embeddings

Combines token identities with absolute learned position encoding to form continuous representations.

#### Mathematical Formulation
$$\text{Embedding}(X) = E(X) + P(\text{Positions})$$

Where $E \in \mathbb{R}^{\text{Vocab} \times D}$ is the token embedding matrix and $P \in \mathbb{R}^{\text{MaxSeqLen} \times D}$ is the spatial positional matrix.

#### Algorithmic Pseudocode
```python
LearnedEmbedding(X:[B, L]): --> [B, L, D]
  Require: 
    E: [VOCAB_SIZE, D] # Embedding Matrix
    P: [MAX_LEN, D] # Position Matrix
  
  Steps:
    I = [0, 1, ..., L-1]
    X_pos = P[I] # X_pos: [L, D]
    X_tok = E[X] # X_emb: [B, L, D]
    
    return X_pos + X_tok
```

#### Active Code Location
* Class: [EmbeddingLayer](file:///run/media/se00n00/P/LittleParrot/GPT/model.py#L122) in [model.py](file:///run/media/se00n00/P/LittleParrot/GPT/model.py).

---

### 2. Normalization Layers (RMSNorm)

Performs scale-normalization on layer inputs. The model uses RMSNorm to omit the mean-centering step for faster training throughput:

#### Mathematical Formulation
$$\text{RMSNorm}(X) = \frac{X}{\sqrt{\frac{1}{d} \sum_{i=1}^d X_i^2 + \epsilon}} \odot W$$

Where $W \in \mathbb{R}^D$ is the learnable scaling parameter.

#### Algorithmic Pseudocode
```python
RMSNorm(X:[B, L, D]):  --> [B, L, D]
  Require: 
    y: [D]
  
  Steps:
    rms = sqrt(mean(X**2 + e, dim=-1)/D)
    X^ = X / rms
    
    return x^ @ y  # y:[D] --> [B, l, D] Broadcasting across seq length and batches
```

#### Active Code Location
* Class: [RMSNorm](file:///run/media/se00n00/P/LittleParrot/GPT/model.py#L36) in [model.py](file:///run/media/se00n00/P/LittleParrot/GPT/model.py). (Standard [LayerNorm](file:///run/media/se00n00/P/LittleParrot/GPT/model.py#L25) is also available).

---

### 3. Causal Multi-Head Attention

Allows tokens to query and incorporate information from preceding tokens.

#### Mathematical Formulation
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}} + M\right) V$$

$$\text{MHA}(X) = \text{Attention}(Q, K, V) W_O$$

Where $M_{ij} = 0 \text{ if } j \le i \text{ else } -\infty$ masks out future tokens to preserve causal autoregression.

#### Algorithmic Pseudocode
```python
Multi_Head_Attention(X:[B, L, D]): --> [B, L, D]
  Require: 
    W_Q: [D, D] # Query Matrix
    W_K: [D, D] # Key Matrix
    W_V: [D, D] # Key Matrix
    W_O: [D, D] # Key Matrix
  
  Steps:
    Q = W_Q @ X
    K = W_K @ X
    V = W_V @ X
    
    for n in Num_heads:
        Scores = softmax((Qn@Kn.T)/sqrt(Dn) + M) # [B, L, L]
        #                   -- Here M is causal mask :[L, L] where Mij = 0 if j<=(L-i) else -inf
        Values = Scores @ Vn # [B, L, Dn]
    Attention = Values.concat()
    
    return Attention @ W_O
```

#### Active Code Location
* Class: [Attention](file:///run/media/se00n00/P/LittleParrot/GPT/model.py#L72) in [model.py](file:///run/media/se00n00/P/LittleParrot/GPT/model.py).

---

### 4. Gated FeedForward (SwiGLU)

Projects representation dimensions up, applies gating logic, and projects back down.

#### Mathematical Formulation
$$\text{FFN}_{\text{gated}}(X) = \text{Dropout}\Big( \big(\text{SiLU}(X W_{\text{up}}) \odot (X W_{\text{gate}}) \big) W_{\text{down}} \Big)$$

#### Algorithmic Pseudocode
```python
FeedForward(X:[B, L, D]): --> [B, L, D]
  Require: 
    Wup: [D, D_H] # Up Matrix
    Wgate: [D, D_H] # Gate Matrix
    Wdown: [D_H, D] # Down Matrix
  
  Steps:
    temp = SiLU(Wup @ X) * Wgate @ X
    y = Wdown @ temp

    return temp
```

#### Active Code Location
* Class: [FeedForward](file:///run/media/se00n00/P/LittleParrot/GPT/model.py#L48) in [model.py](file:///run/media/se00n00/P/LittleParrot/GPT/model.py).

---

### 5. Transformer Block

Combines Attention, FeedForward, Normalization, and Residual connections into a single pre-normalized layer block.

#### Mathematical Formulation
$$X_1 = X + \text{Attention}(\text{RMSNorm}(X))$$

$$X_2 = X_1 + \text{FeedForward}(\text{RMSNorm}(X_1))$$

#### Algorithmic Pseudocode
```python
Block(X:[B, L, D]): --> [B, L, D]
  Steps:
    X = X + Attention(LayerNorm(X)) # Pre-normalization
    X = X + FeedForward(LayerNorm(X))

    return X
```

#### Active Code Location
* Class: [Block](file:///run/media/se00n00/P/LittleParrot/GPT/model.py#L134) in [model.py](file:///run/media/se00n00/P/LittleParrot/GPT/model.py).

---

### 6. Consolidated Model

Stacks multiple blocks sequentially and projects the final output representations back to the vocabulary dimension.

#### Mathematical Formulation
$$X_0 = \text{Embedding}(X)$$

$$X_k = \text{Block}_k(X_{k-1}), \quad \text{for } k=1, \dots, N_{\text{layers}}$$

$$\text{Output} = X_N W_{\text{head}}$$

#### Algorithmic Pseudocode
```python
Model(X:[B, L]): --> [B, L, D]
  Require:
    W_head: [D, VOCAB_SIZE]
    
  Initialization:
    Model.layers = normal_distribution(layers.weight, mean=0, std=0.02)
    
  Steps:
    X = LearnedEmbedding(X)
    for block in Blocks*Num_Layers:
      X = block(X)
    
    output = W_head(X)
    return output
```

#### Active Code Location
* Class: [Model](file:///run/media/se00n00/P/LittleParrot/GPT/model.py#L149) in [model.py](file:///run/media/se00n00/P/LittleParrot/GPT/model.py).

---

### 7. Mixture of Experts (MoE) - Experimental

Directs tokens dynamically to expert FFN subnetworks to scale model parameters while maintaining active compute efficiency.

#### Mathematical Formulation
$$\text{Router}(X) = \text{softmax}(\text{TopK}(X W_{\text{expert\_gate}}, k))$$

$$\text{MoE}(X) = \sum_{i \in \text{TopK}} \text{Router}(X)_i \cdot \text{Expert}_i(X)$$

#### Active Code Location
* Class: [MoE](file:///run/media/se00n00/P/LittleParrot/GPT/Model/layers.py#L137) in [Model/layers.py](file:///run/media/se00n00/P/LittleParrot/GPT/Model/layers.py).

---

## Model Architecture

The schematic sequence mapping input text processing through the networks is illustrated below:

```
 [OUTPUT]
    │
    + ──────────┐
    |   ┌──────────────────────────┐  
    │   |        FEEDFORWARD       │
    │   └──────────────────────────┘
    |──[RMS-NORM]───┘
    + ──────────┐
    │   ┌──────────────────────────┐   
    │   |   MULTI-HEAD ATTENTION   │
    │   └──────────────────────────┘
    └──[RMS-NORM]───┘
    │
 [INPUT] + ──[LEARNED ENCODING]
```

---

## 📂 Project Directory Hierarchy

Below is the workspace layout highlighting the role of each module:

* [model.py](file:///run/media/se00n00/P/LittleParrot/GPT/model.py) / [Model/](file:///run/media/se00n00/P/LittleParrot/GPT/Model) — Model files:
  * [Model](file:///run/media/se00n00/P/LittleParrot/GPT/Model/models.py#L8) — Main decoder-only wrapper.
  * [layers.py](file:///run/media/se00n00/P/LittleParrot/GPT/Model/layers.py) — Low-level layers (`Attention`, `RMSNorm`, `FeedForward`, `MoE`).
  * [loss.py](file:///run/media/se00n00/P/LittleParrot/GPT/Model/loss.py) — Objective functions ([token_level_kd_loss](file:///run/media/se00n00/P/LittleParrot/GPT/Model/loss.py#L14)).
* [Trainer/](file:///run/media/se00n00/P/LittleParrot/GPT/Trainer) — Training infrastructure:
  * [gpu.py](file:///run/media/se00n00/P/LittleParrot/GPT/Trainer/gpu.py) — Device telemetry checks ([check_and_cooldown_gpu](file:///run/media/se00n00/P/LittleParrot/GPT/Trainer/gpu.py#L19), [check_vram_limit](file:///run/media/se00n00/P/LittleParrot/GPT/Trainer/gpu.py#L34)).
  * [dataset_preparation.py](file:///run/media/se00n00/P/LittleParrot/GPT/Trainer/dataset_preparation.py) — Tokenization and binary exporting pipeline.
  * [text_dataset.py](file:///run/media/se00n00/P/LittleParrot/GPT/Trainer/text_dataset.py) — Memory-mapped file stream dataset.
  * [util.py](file:///run/media/se00n00/P/LittleParrot/GPT/Trainer/util.py) — Linear warmup / cosine decay scheduler.
* [Datasets/](file:///run/media/se00n00/P/LittleParrot/GPT/Datasets) — Tokenizers and data instructions:
  * [tokenizer.py](file:///run/media/se00n00/P/LittleParrot/GPT/Datasets/tokenizer.py) — Custom BPETokenizer class.
  * [dataset.md](file:///run/media/se00n00/P/LittleParrot/GPT/Datasets/dataset.md) — Raw documentation on data prep setup.
* [train.py](file:///run/media/se00n00/P/LittleParrot/GPT/train.py) — Advanced trainer logic equipped with self-healing Memory Guard and Thermal Guard.
* [dpo_trainer.py](file:///run/media/se00n00/P/LittleParrot/GPT/dpo_trainer.py) — Direct Preference Optimization trainer with memory efficiency routines.
* [expand_model.py](file:///run/media/se00n00/P/LittleParrot/GPT/expand_model.py) — Weight transfer utility for vocab extensions.
* [sft_dataset.py](file:///run/media/se00n00/P/LittleParrot/GPT/sft_dataset.py) — Script to mask out user prompts for Supervised Fine-Tuning.
* [inference.py](file:///run/media/se00n00/P/LittleParrot/GPT/inference.py) — Asynchronous streaming generator CLI tool.
* [requirements.txt](file:///run/media/se00n00/P/LittleParrot/GPT/requirements.txt) — Dependency list.

---

## 🚀 Installation & Data Preparation

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Tokenize Datasets
Refer to [Datasets/dataset.md](file:///run/media/se00n00/P/LittleParrot/GPT/Datasets/dataset.md) to download wikitext, then tokenize to binary (`uint16`) blocks:
```python
from Trainer.dataset_preparation import prepare_datasets

DATAPATH = {
    "train": {"token_path": "train_tokens.bin", "dataset_path": "Datasets/train"},
    "validation": {"token_path": "val_tokens.bin", "dataset_path": "Datasets/validation"},
    "test": {"token_path": "test_tokens.bin", "dataset_path": "Datasets/test"},
}

prepare_datasets(DATAPATH, tokenizer_dir="Datasets/tokenizer.json")
```

---

## 🛠️ The 3-Stage LLM Pipeline

| Stage | Mode Flag | Description | Loss Metric |
| :--- | :--- | :--- | :--- |
| **1. Pre-Training (PT)** | `--pipeline PT` | Learns next-token probability distributions from raw corpora. | Standard Cross Entropy |
| **2. Supervised Fine-Tuning (SFT)** | `--pipeline IFT` | Tunes the model on chat instructions, masking input prompts. | Masked Cross-Entropy (Prompt token label = `60000`) |
| **3. Direct Preference Optimization (DPO)**| `--pipeline PFT` | Aligns the policy model predictions using chosen vs. rejected pairs. | DPO Log-Sigmoid Relative Loss |

### Stage 1: Pre-Training (PT)
Train model from scratch using next-token causal prediction:
```bash
python train.py \
  --training_name pt_run \
  --pipeline PT \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --max_seq_len 512 \
  --learning_rate 3e-4 \
  --eval_interval 200 \
  --checkpoint_dir checkpoints
```

### Stage 2: Supervised Fine-Tuning (IFT / SFT)
1. Write masked SFT label arrays:
   ```bash
   python sft_dataset.py
   ```
2. Fine-tune your pre-trained model:
   ```bash
   python train.py \
     --training_name sft_run \
     --pipeline IFT \
     --resume checkpoints/TinyLM-1-70M_PT.pt \
     --batch_size 4 \
     --grad_accum_steps 8 \
     --max_seq_len 512 \
     --learning_rate 1e-4
   ```

### Stage 3: Preference Fine-Tuning (PFT / DPO)
Align model responses using paired preference datasets:
```bash
python train.py \
  --training_name dpo_run \
  --pipeline PFT \
  --resume checkpoints/TinyLM-1-70M_IFT.pt \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --max_seq_len 512
```

> [!TIP]
> **VRAM-Saving Reference Logic**: Computing DPO loss typically requires both the active policy model and the frozen reference model in VRAM. TinyLM-1-70M saves memory by dynamically loading the reference model, extracting log probabilities, moving reference weights back to the CPU, and flushing the CUDA cache before the policy backward pass.

---

## 🛡️ Hardware Safeguards

### Memory Guard (OOM Self-Healing)
If PyTorch raises a CUDA Out of Memory exception at step execution:
1. The trainer catches the error, flushes the cache, and halts the current pass.
2. If the current micro-batch size $B > 1$, it sets $B \leftarrow \lfloor B/2 \rfloor$ and adjusts the gradient accumulation steps to keep the effective batch size constant.
3. If $B = 1$ already, the trainer dynamically enables **gradient checkpointing** on the model parameters to trade compute for memory.
4. The step is re-attempted without failing the training run.

### Thermal Guard
To protect hardware from prolonged heat stress, [gpu.py](file:///run/media/se00n00/P/LittleParrot/GPT/Trainer/gpu.py) checks GPU temperature:
* Threshold triggering pause: `--max_temp` (default: `75°C`).
* Resuming target temperature: `--cooldown_temp` (default: `60°C`).

---

## 📈 Vocabulary Expansion

When transitioning from pre-training tokenizers to SFT tokenizers (which introduce chat-centric tokens like `<|USER|>` and `<|ASSISTANT|>`), the vocabulary size changes. Use [expand_model.py](file:///run/media/se00n00/P/LittleParrot/GPT/expand_model.py) to map weights from a smaller vocab size to an expanded one:

```bash
python expand_model.py
```

This replicates pre-trained embedding and projection weights for existing token indices, while randomly initializing row entries matching new vocab additions, preserving downstream performance.

---

## 💬 Asynchronous Streaming Inference

Test your trained model using the interactive streaming CLI. It runs sampling asynchronously and prints tokens in real-time:

```bash
python inference.py \
  --checkpoint checkpoints/TinyLM-1-70M_IFT.pt \
  --prompt "Explain the concept of neural networks in simple terms." \
  --temperature 0.7 \
  --top_k 40 \
  --top_p 0.9
```

---

## 📊 Model Evaluation

Evaluate checkpoints using standard language model benchmarks with the custom `lm_eval` wrapper in the [Evaluation](file:///run/media/se00n00/P/LittleParrot/GPT/Evaluation) directory.

### Custom LM Harness Wrapper
The evaluation suite contains a PyTorch-native adapter `CustomGPTLM` that hooks our custom transformer models directly into the `lm-evaluation-harness` (v0.4.x) framework:
- **Loading & Alignment**: Automatically detects the vocabulary size from checkpoint state dicts to initialize the model layers, handling the tokenizer expansions (e.g., from `32768` to `32771` tokens).
- **OOM Resilience**: Automatically reduces batch sizes or falls back to CPU if a CUDA Out of Memory error is raised during validation.
- **Save Formats**: Full sample-level logs are outputted to JSON formats for downstream debugging and trace analyses.

### Run Evaluation
To evaluate a model checkpoint:
```bash
python Evaluation/eval.py --checkpoint checkpoints/TinyLM-1-70M_IFT.pt
```

To run a comparative evaluation of all checkpoints in the workspace:
```bash
python Evaluation/eval.py --checkpoint all
```

Parameters:
- `--checkpoint`: Path to model checkpoint file (or `'all'`).
- `--tasks`: Comma-separated benchmarks (default: `hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu,sciq,openbookqa,commonsense_qa,blimp,gsm8k,triviaqa,wikitext`).
- `--batch_size`: Evaluation batch size (default: `8`).
- `--limit`: Limit the number of samples per task (for quick testing).

### Benchmark Comparison Results

All benchmark evaluations below were executed using `lm-evaluation-harness` (`lm-eval`). For every reference model listed, we evaluated them ourselves using `lm-eval` under identical evaluation conditions to ensure direct, fair, and reproducible comparison with **TinyLM-1-70M**.

#### 1. Pretrained Models Evaluation

Comparative zero-shot and few-shot evaluation results for pretrained base models:

| Benchmark Task | Category | Shots | Metric | TinyLM-1-70M (Base, 70M) | GPT-2 (Base, 124M)* | Supra-50M (Base, 50M)* | SmolLM2-135M (Base, 135M)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HellaSwag** | Commonsense Reasoning | 0-shot | AN* | 0.2568 | 0.3114 | 0.3178 | 0.4322 |
| **ARC-Easy** | Science Question Answering | 0-shot | AN* | 0.3013 | 0.3948 | 0.4600 | 0.5871 |
| **ARC-Challenge** | Complex Scientific Reasoning | 0-shot | AN* | 0.2637 | 0.2270 | 0.2500 | 0.2952 |
| **PiQA** | Physical Commonsense Reasoning | 0-shot | AN* | 0.5550 | 0.6251 | 0.6208 | 0.6861 |
| **SciQ** | General Science Knowledge | 0-shot | AN* | 0.2310 | 0.6440 | 0.6810 | 0.7860 |
| **OpenBookQA** | Multi-hop Open Book QA | 0-shot | AN* | 0.3520 | 0.2720 | 0.3060 | 0.3260 |
| **Winogrande** | Pronoun Disambiguation & Logic | 0-shot | Acc* | 0.5036 | 0.5162 | 0.5099 | 0.5304 |
| **MMLU** (57 subjects) | Multi-task Domain Knowledge | 0-shot | Acc* | 0.2295 | 0.2292 | 0.2301 | 0.2410 |
| **CommonsenseQA** | Conceptual Commonsense | 0-shot | Acc* | 0.1957 | 0.1957 | 0.1966 | 0.1933 |
| **BLiMP** | Linguistic & Syntax Probe | 0-shot | Acc* | 0.7538 | 0.8215 | 0.7632 | 0.8007 |
| **GSM8K** | Mathematical Problem Solving | 5-shot | EM* | 0.0015 | 0.0068 | 0.0235 | 0.0220 |
| **TriviaQA** | Open-Domain Fact Retrieval | 0-shot | EM* | 0.0000 | 0.0030 | 0.0041 | 0.0495 |
| **WikiText** | Language Modeling Perplexity | 0-shot | PPL* ↓ | 84.56 | 37.37 | 44.95 | 21.06 |

`* Short-form metric definitions:`
- **AN***: `acc_norm` (Length-normalized Accuracy)
- **Acc***: `acc` (Standard Raw Accuracy)
- **EM***: `exact_match` (Exact Match string accuracy)
- **PPL***: `word_perplexity` (Perplexity on test set; lower score indicates better language modeling)
- `*`: *Denotes reference models evaluated locally by us using `lm-eval`.*

---

#### 2. Instruction-Tuned Models Evaluation

Comparative evaluation results for instruction-fine-tuned (IFT) models:

| Benchmark Task | Category | Shots | Metric | TinyLM-1-70M (Instruct, 70M) | SmolLM2-135M (Instruct, 135M)* |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HellaSwag** | Commonsense Reasoning | 0-shot | AN* | 0.2762 | 0.4031 |
| **ARC-Easy** | Science Question Answering | 0-shot | AN* | 0.2929 | 0.4571 |
| **ARC-Challenge** | Complex Scientific Reasoning | 0-shot | AN* | 0.2321 | 0.2858 |
| **PiQA** | Physical Commonsense Reasoning | 0-shot | AN* | 0.5506 | 0.6725 |
| **SciQ** | General Science Knowledge | 0-shot | AN* | 0.2260 | 0.6960 |
| **OpenBookQA** | Multi-hop Open Book QA | 0-shot | AN* | 0.2920 | 0.3280 |
| **Winogrande** | Pronoun Disambiguation & Logic | 0-shot | Acc* | 0.5036 | 0.5280 |
| **MMLU** (57 subjects) | Multi-task Domain Knowledge | 0-shot | Acc* | 0.2295 | 0.2470 |
| **CommonsenseQA** | Conceptual Commonsense | 0-shot | Acc* | 0.1957 | 0.2105 |
| **BLiMP** | Linguistic & Syntax Probe | 0-shot | Acc* | 0.6979 | 0.8039 |
| **GSM8K** | Mathematical Problem Solving | 5-shot | EM* | 0.0182 | 0.0144 |
| **TriviaQA** | Open-Domain Fact Retrieval | 0-shot | EM* | 0.0000 | 0.0035 |
| **WikiText** | Language Modeling Perplexity | 0-shot | PPL* ↓ | 147.29 | 24.11 |

`* Short-form metric definitions:`
- **AN***: `acc_norm` (Length-normalized Accuracy)
- **Acc***: `acc` (Standard Raw Accuracy)
- **EM***: `exact_match` (Exact Match string accuracy)
- **PPL***: `word_perplexity` (Perplexity on test set; lower score indicates better language modeling)
- `*`: *Denotes reference models evaluated locally by us using `lm-eval`.*

Evaluation details and full raw output predictions are saved under the [Evaluation/](file:///run/media/se00n00/P/LittleParrot/GPT/Evaluation) directory.

---

## 📜 License

This codebase is licensed under the [MIT License](file:///run/media/se00n00/P/LittleParrot/GPT/LICENSE). Feel free to modify and adapt it for research or commercial applications.
