---
title: >-
  [Paper Note] Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference
description: >-
  [ACL 2025][LLM Efficiency][encoder-only] ModernBERT is proposed, systematically introducing modern LLM architectural optimizations (RoPE, GeGLU, alternating local/global attention, and unpadding) into encoder-only models. Trained on 2T tokens and natively supporting a context length of 8192, it outperforms BERT/RoBERTa/DeBERTaV3 across classification and retrieval tasks while achieving significantly faster inference speeds and superior memory efficiency.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "encoder-only"
  - "BERT"
  - "inference efficiency"
  - "long context"
  - "information retrieval"
  - "Flash Attention"
  - "RoPE"
date: 2026-05-08
content_hash: e3ffccd76559ba5b
---

# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference

**Conference**: ACL 2025  
**arXiv**: [2412.13663](https://arxiv.org/abs/2412.13663)  
**Code**: [AnswerDotAI/ModernBERT](https://github.com/AnswerDotAI/ModernBERT)  
**Area**: LLM Efficiency  
**Keywords**: encoder-only, BERT, inference efficiency, long context, information retrieval, Flash Attention, RoPE

## TL;DR

ModernBERT is proposed, systematically introducing modern LLM architectural optimizations (RoPE, GeGLU, alternating local/global attention, and unpadding) into encoder-only models. Trained on 2T tokens and natively supporting a context length of 8192, it outperforms BERT/RoBERTa/DeBERTaV3 across classification and retrieval tasks while achieving significantly faster inference speeds and superior memory efficiency.

## Background & Motivation

**Encoder Models Remain the Production Workhorses**: Although LLMs have achieved remarkable success, encoder-only models (such as BERT) remain widely used for non-generative tasks like retrieval (as the core component of RAG pipelines), classification, and NER due to their lightweight inference and high throughput. More than half of the top 100 most downloaded models on Hugging Face are encoder-based retrieval models.

**Severe Limitations of Legacy Models**: Existing pipelines rely heavily on the original BERT directly, facing multiple bottlenecks: a sequence length of only 512, unoptimized model architectures, outdated vocabularies (lacking code symbols), and small, single-domain training datasets.

**Incomplete Prior Improvements**: MosaicBERT and CrammingBERT focus only on training efficiency; NomicBERT and GTE-en-MLM extend the context length but fail to optimize inference efficiency or classification performance, and they still use legacy data recipes, leading to poor performance on code tasks.

**Core Problem**: Can modern architectural improvements and training strategies accumulated in decoder-only LLMs in recent years be systematically transferred to encoder-only models to achieve Pareto improvements in both performance and efficiency?

## Method

### Architectural Modernization (Modern Transformer)

| Component | Improvement | Motivation |
|---|---|---|
| **Biases** | Remove all biases except for the final decoder linear layer | Concentrate parameter budget on the linear layers |
| **Position Embeddings** | Absolute position embeddings → RoPE | Effective for both short and long contexts, easy to scale |
| **Normalization** | Pre-Norm + post-embedding LayerNorm | Stabilizes training, removes redundant LN in the first layer |
| **Activation Function** | GeLU → GeGLU (GLU variant) | Empirically proven to outperform original GeLU |

### Efficiency Optimization

- **Alternating Attention**: Global attention (RoPE theta=160,000) is used every 3 layers, while the remaining layers use 128-token sliding-window local attention (RoPE theta=10,000). Inspired by efficient long-context models like Gemma, this design significantly reduces the computational overhead for long sequences.

- **Full-Throughput Unpadding**: Padding tokens are removed prior to token embedding, concatenating all sequences in a batch into a single long sequence for processing. Implemented via variable-length Flash Attention and RoPE, this approach is 10-20% faster than previous unpadding schemes (such as the internal unpad/repad in MosaicBERT).

- **Hybrid Flash Attention**: Flash Attention 3 (optimized for H100) is utilized for global attention layers, while Flash Attention 2 (supporting sliding windows) is used for local attention layers.

- **torch.compile**: Compiles all compatible modules, yielding an additional throughput increase of approximately 10% with negligible compilation overhead.

### Hardware-Aware Model Design

- Employs a **Deep & Narrow** strategy: deeper and narrower layers yield better downstream performance than fewer and wider layers.
- ModernBERT-base: 22 layers, hidden=768, GLU expansion=2304, with 149M total parameters.
- ModernBERT-large: 28 layers, hidden=1024, GLU expansion=5248, with 395M total parameters.
- Maximizes utilization on a set of common GPUs (T4, A10, L4, RTX 3090/4090) through small-scale ablation studies.
- Parameter dimensions are meticulously selected to ensure alignment with optimal Tensor Core tiling, achieving highly efficient computation across varying streaming multiprocessor counts on different GPUs.

### Loss & Training

- **Data**: 2T tokens, primarily English, containing web documents, code, and scientific literature.
- **Tokenizer**: A modern BPE tokenizer based on OLMo with a vocabulary size of 50,368 (a multiple of 64), retaining BERT's [CLS]/[SEP] special tokens for backward compatibility.
- **Objective Function**: MLM only (30% masking rate), removing the ineffective Next-Sentence Prediction.
- **Optimizer**: StableAdamW = AdamW + Adafactor-style per-parameter LR clipping, which is more stable than standard gradient clipping.
- **Learning Rate Schedule**: Warmup-Stable-Decay (WSD) + 1-sqrt decay, which performs better than linear and cosine decays.
- **Batch Size Schedule**: Progressively scales up from a small batch size (base: 768 to 4608, large: 448 to 4928) to accelerate early training.
- **Weight Initialization**: ModernBERT-base uses Megatron initialization; ModernBERT-large is tile-initialized from the trained base weights (reminiscent of the Phi series), significantly accelerating initial loss descent.
- **Context Expansion**: The model is first trained on a sequence length of 1024 for 1.7T tokens, and then the global attention RoPE theta is scaled to 160k to continue training on a sequence length of 8192 for an additional 300B tokens (250B with a constant low LR + 50B with a 1-sqrt decay, upsampling high-quality data sources).

## Key Experimental Results

### Table 1: Main Task Performance Overview

| Model | IR-DPR BEIR | IR-ColBERT BEIR | MLDR OOD | MLDR ID | GLUE | CSN | SQA |
|---|---|---|---|---|---|---|---|
| **Base** | | | | | | | |
| BERT-base | 38.9 | 49.0 | 23.9 | 32.2 | 84.7 | 41.2 | 59.5 |
| RoBERTa-base | 37.7 | 48.7 | 22.9 | 32.8 | 86.4 | 44.3 | 59.6 |
| DeBERTaV3-base | 20.2 | 47.1 | 5.4 | 13.4 | 88.1 | 17.5 | 18.6 |
| NomicBERT | 41.0 | 49.9 | 26.7 | 30.3 | 84.0 | 41.6 | 61.4 |
| GTE-en-MLM-base | 41.4 | 48.2 | 34.3 | 44.4 | 85.6 | 44.9 | 71.4 |
| **ModernBERT-base** | **41.6** | **51.3** | 27.4 | 44.0 | **88.4** | **56.4** | **73.6** |
| **Large** | | | | | | | |
| BERT-large | 38.9 | 49.5 | 23.3 | 31.7 | 85.2 | 41.6 | 60.8 |
| RoBERTa-large | 41.4 | 49.8 | 22.6 | 36.1 | 88.9 | 47.3 | 68.1 |
| DeBERTaV3-large | 25.6 | 46.7 | 7.1 | 19.2 | **91.4** | 21.2 | 19.7 |
| GTE-en-MLM-large | 42.5 | 50.7 | 36.4 | 48.9 | 87.6 | 40.5 | 66.9 |
| **ModernBERT-large** | **44.0** | **52.4** | 34.3 | 48.6 | 90.4 | **59.5** | **83.9** |

**Key Findings**:

- ModernBERT-base outperforms DeBERTaV3-base on GLUE for the first time (88.4 vs 88.1), making it the first model to achieve this using MLM only.
- Code tasks exhibit the largest performance gains: CodeSearchNet +11.5, StackQA +2.2 over GTE-en-MLM.
- In ColBERT long-context retrieval, ModernBERT leads other long-context models by at least 9 nDCG@10 on MLDR OOD.

### Table 2: Inference Efficiency (RTX 4090, thousand tokens/sec)

| Model | Parameters | Short Max BS | Short Fixed | Short Var | Long Fixed | Long Var |
|---|---|---|---|---|---|---|
| BERT-base | 110M | 1096 | 180.4 | 90.2 | - | - |
| DeBERTaV3-base | 183M | 236 | 70.2 | 35.1 | - | - |
| NomicBERT | 137M | 588 | 117.1 | 58.5 | 46.1 | 23.1 |
| GTE-en-MLM-base | 137M | 640 | 123.7 | 61.8 | 46.8 | 23.4 |
| **ModernBERT-base** | 149M | **1604** | **148.1** | **147.3** | **123.7** | **133.8** |
| GTE-en-MLM-large | 435M | 472 | 38.7 | 19.3 | 16.2 | 8.1 |
| **ModernBERT-large** | 395M | **770** | **52.3** | **52.9** | **46.8** | **49.8** |

**Key Findings**:

- The maximum batch size of ModernBERT-base (1604) is more than 2x larger than other models, demonstrating leading GPU memory efficiency.
- The processing speed for long text (8192) is 2.65x to 3x faster than the second-fastest model.
- ModernBERT-large's long-text speed (46.8k tok/s) is close to GTE-base (47.5k) and significantly outperforms GTE-large (16.5k).
- ModernBERT's advantage is even more pronounced under variable-length inputs, outperforming GTE by 14.5% to 118.8%, thanks to unpadding and local attention.

## Highlights & Insights

1. **Systematic Pareto Optimization**: For the first time, a full suite of modern architectural improvements from decoder-only LLMs (RoPE, GeGLU, alternating attention, Flash Attention, unpadding, and torch.compile) is ported to an encoder-only model, setting a new SOTA in both downstream performance and inference efficiency.

2. **First MLM Model to Outperform DeBERTaV3-base**: While DeBERTaV3 relies on the RTD (Replaced Token Detection) objective to lead on GLUE, ModernBERT surpasses it at the base scale using MLM alone, breaking the assumption that RTD is necessary to achieve top classification performance.

3. **Breakthrough Coding Capability**: As the only encoder model to incorporate code data during pre-training, paired with a code-aware tokenizer (OLMo-based), it leads significantly on CodeSearchNet and StackQA, offering immense value for code retrieval scenarios.

4. **Solid Engineering Execution**: Features highly polished engineering details including a hardware-aware Deep & Narrow design, full-throughput unpadding (removing padding tokens prior to embedding), a hybrid FA2/FA3 strategy, and Tensor Core tiling optimizations.

5. **Open-Source Friendly**: Releases the modular FlexBERT framework and all intermediate training checkpoints (reminiscent of Pythia), facilitating community research and replication.

## Limitations & Future Work

1. **English Only**: The 2T tokens consist entirely of English data, rendering the model unsuitable for multilingual scenarios and less friendly to low-resource languages.

2. **Ceiling of MLM-Only Objective**: DeBERTaV3-large still performs slightly better on GLUE (91.4 vs 90.4). Joint training with RTD and MLM might further improve classification performance, which the authors explicitly target as a future direction.

3. **Weaker Out-of-Distribution (OOD) DPR in Long Contexts compared to GTE**: Performance in the MLDR OOD setting is noticeably weaker than GTE-en-MLM (27.4 vs 34.3). While local attention boosts efficiency, it may compromise the generalization of zero-shot long-context single-vector retrieval.

4. **Insufficient Exploration of Model Scales**: Only base (149M) and large (395M) sizes have been released, leaving the scaling behavior of larger (1B+) or smaller (tiny/mini) encoder models unexplored.

5. **Web Data Bias**: A substantial portion of the training data is sourced from the web; consequently, the model representations inevitably incorporate societal biases from web data, which the authors do not quantitatively analyze.

## Related Work & Insights

| Dimension | BERT / RoBERTa | DeBERTaV3 | NomicBERT | GTE-en-MLM | **ModernBERT** |
|---|---|---|---|---|---|
| Context Length | 512 | 512 | 8192 | 8192 | **8192** |
| Training Data Volume | 16B / 33B | ~64B | ~32B | ~30B | **2T** |
| Code Data | None | None | None | None | **Yes** |
| Inference Efficiency Optimization | None | None | Partial | unpadding | **Full Suite** |
| GLUE (base) | 84.7 / 86.4 | 88.1 | 84.0 | 85.6 | **88.4** |
| BEIR DPR (base) | 38.9 / 37.7 | 20.2 | 41.0 | 41.4 | **41.6** |
| Position Encoding | Absolute | Relative disentangled | RoPE | RoPE | **RoPE** |
| Attention Type | Global | Global | Global + Local | Global | **Alternating Global/Local** |

## Rating

- Novelty: ⭐⭐⭐ — The architectural modifications are drawn from existing techniques; the core contribution lies in systematic integration and engineering optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Broadly covers GLUE, BEIR, MLDR, CodeSearchNet, StackQA, etc., with efficiency benchmarks and ablations included.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with detailed engineering specifications and a transparent contribution statement.
- Value: ⭐⭐⭐⭐⭐ — Presents a comprehensive generational upgrade for encoder-only models, ready to directly replace legacy BERT in production pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts](epman_episodic_memory_attention_for_generalizing_to_longer_contexts.md)
- [\[NeurIPS 2025\] SkyLadder: Better and Faster Pretraining via Context Window Scheduling](../../NeurIPS2025/llm_efficiency/skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)
- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)
- [\[ICML 2025\] EasyInv: Toward Fast and Better DDIM Inversion](../../ICML2025/llm_efficiency/easyinv_toward_fast_and_better_ddim_inversion.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)

</div>

<!-- RELATED:END -->
