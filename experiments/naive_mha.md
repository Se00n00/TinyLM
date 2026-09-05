# This Document Serves as Experiments Concluded In this Repository

## Current Best
```
-----------------------------------------------------------------------------------

 [OUTPUT]                                 Architecture: Naive MHA
    │                                      .
    + ──────────┐                           \_ [72M Parameters]
    |   ┌──────────────────────────┐       .
    │   |        FEEDFORWARD       │        \_ [528 D_model]
    │   └──────────────────────────┘       .
    |──[RMS-NORM]───┘                       \_ [528 Context Length]
    + ──────────┐
    │   ┌──────────────────────────┐   
    │   |   MULTI-HEAD ATTENTION   │
    │   └──────────────────────────┘
    └──[RMS-NORM]───┘
    │
 [INPUT] + ──[LEARNED ENCODING]

-----------------------------------------------------------------------------------
```

#### 1. Pretrained Models Evaluation

Comparative zero-shot and few-shot evaluation results for pretrained base models:

| Category | Benchmark Task (Metric) | TinyLM-1-70M (Base, 70M) | GPT-2 (Base, 124M)* | Supra-50M (Base, 50M)* | SmolLM2-135M (Base, 135M)* |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Commonsense & Logic** | **HellaSwag** (AN*) | 0.2568 | 0.3114 | 0.3178 | **0.4322** |
| | **CommonsenseQA** (Acc*) | 0.1957 | 0.1957 | **0.1966** | 0.1933 |
| | **PiQA** (AN*) | 0.5550 | 0.6251 | 0.6208 | **0.6861** |
| | **Winogrande** (Acc*) | 0.5036 | 0.5162 | 0.5099 | **0.5304** |
| **Language Modeling** | **WikiText** (PPL* ↓) | 84.56 | 37.37 | 44.95 | **21.06** |
| **Linguistic & Syntax** | **BLiMP** (Acc*) | 0.7538 | **0.8215** | 0.7632 | 0.8007 |
| **Mathematical Reasoning** | **GSM8K** (5-shot, EM*) | 0.0015 | 0.0068 | **0.0235** | 0.0220 |
| **Open-Domain Fact Retrieval** | **TriviaQA** (EM*) | 0.0000 | 0.0030 | 0.0041 | **0.0495** |
| **Science & Domain Knowledge** | **ARC-Easy** (AN*) | 0.3013 | 0.3948 | 0.4600 | **0.5871** |
| | **ARC-Challenge** (AN*) | 0.2637 | 0.2270 | 0.2500 | **0.2952** |
| | **SciQ** (AN*) | 0.2310 | 0.6440 | 0.6810 | **0.7860** |
| | **OpenBookQA** (AN*) | **0.3520** | 0.2720 | 0.3060 | 0.3260 |
| | **MMLU** (57 subjects, Acc*) | 0.2295 | 0.2292 | 0.2301 | **0.2410** |

`* Short-form metric definitions:`
- **AN***: `acc_norm` (Length-normalized Accuracy)
- **Acc***: `acc` (Standard Raw Accuracy)
- **EM***: `exact_match` (Exact Match string accuracy)
- **PPL***: `word_perplexity` (Perplexity on test set; lower score indicates better language modeling)
- `*`: *Denotes reference models evaluated locally by us using `lm-eval`.*

---

#### 2. Instruction-Tuned Models Evaluation

Comparative evaluation results for instruction-fine-tuned (IFT) models:

| Category | Benchmark Task (Metric) | TinyLM-1-70M (Instruct, 70M) | SmolLM2-135M (Instruct, 135M)* |
| :--- | :--- | :---: | :---: |
| **Commonsense & Logic** | **HellaSwag** (AN*) | 0.2762 | **0.4031** |
| | **CommonsenseQA** (Acc*) | 0.1957 | **0.2105** |
| | **PiQA** (AN*) | 0.5506 | **0.6725** |
| | **Winogrande** (Acc*) | 0.5036 | **0.5280** |
| **Instruction Following** | **IFEval** (Acc*) | 0.1232 | **0.2990** |
| **Language Modeling** | **WikiText** (PPL* ↓) | 147.29 | **24.11** |
| **Linguistic & Syntax** | **BLiMP** (Acc*) | 0.6979 | **0.8039** |
| **Mathematical Reasoning** | **GSM8K** (5-shot, EM*) | **0.0182** | 0.0144 |
| **Open-Domain Fact Retrieval** | **TriviaQA** (EM*) | 0.0000 | **0.0035** |
| **Science & Domain Knowledge** | **ARC-Easy** (AN*) | 0.2929 | **0.4571** |
| | **ARC-Challenge** (AN*) | 0.2321 | **0.2858** |
| | **SciQ** (AN*) | 0.2260 | **0.6960** |
| | **OpenBookQA** (AN*) | 0.2920 | **0.3280** |
| | **MMLU** (57 subjects, Acc*) | 0.2295 | **0.2470** |

`* Short-form metric definitions:`
- **AN***: `acc_norm` (Length-normalized Accuracy)
- **Acc***: `acc` (Standard Raw Accuracy)
- **EM***: `exact_match` (Exact Match string accuracy)
- **PPL***: `word_perplexity` (Perplexity on test set; lower score indicates better language modeling)
- `*`: *Denotes reference models evaluated locally by us using `lm-eval` (IFEval for TinyLM-1-70M reflects prompt strict: `0.0832`, inst strict: `0.1631`, avg: `0.1232`; SmolLM2-135M-Instruct IFEval avg: `0.2990`).*

Evaluation details and full raw output predictions are saved under the [Evaluation/](file:///run/media/se00n00/P/LittleParrot/GPT/Evaluation) directory.


---

## The 2-Stage LLM Pipeline

| Stage | Mode Flag | Description | Loss Metric |
| :--- | :--- | :--- | :--- |
| **1. Pre-Training (PT)** | `--pipeline PT` | Learns next-token probability distributions from raw corpora. | Standard Cross Entropy |
| **2. Supervised Fine-Tuning (SFT)** | `--pipeline IFT` | Tunes the model on chat instructions, masking input prompts. | Masked Cross-Entropy (Prompt token label = `60000`) |