---
title: >-
  [Paper Note] Liger: Linearizing Large Language Models to Gated Recurrent Structures
description: >-
  [ICML2025][Physics & Scientific Computing][LLM Linearization] Liger converts pretrained Transformer LLMs into gated linear recurrent structures without adding extra parameters by reusing the Key projection matrix to construct the gating mechanism. It recovers up to 93% of the original model's performance using only 0.02% of the pretraining tokens, while achieving linear-time inference and constant memory overhead.
tags:
  - "ICML2025"
  - "Physics & Scientific Computing"
  - "LLM Linearization"
  - "Gated Linear Recurrent"
  - "Linear Attention"
  - "LoRA"
  - "Sliding Window Attention"
date: 2026-05-08
content_hash: d085db4ad3643baf
---

# Liger: Linearizing Large Language Models to Gated Recurrent Structures

**Conference**: ICML2025  
**arXiv**: [2503.01496](https://arxiv.org/abs/2503.01496)  
**Code**: [OpenSparseLLMs/Linearization](https://github.com/OpenSparseLLMs/Linearization)  
**Area**: Model Compression  
**Keywords**: LLM Linearization, Gated Linear Recurrent, Linear Attention, LoRA, Sliding Window Attention

## TL;DR
Liger converts pretrained Transformer LLMs into gated linear recurrent structures without adding extra parameters by reusing the Key projection matrix to construct the gating mechanism. It recovers up to 93% of the original model's performance using only 0.02% of the pretraining tokens, while achieving linear-time inference and constant memory overhead.

## Background & Motivation

The softmax attention of Transformers scales quadratically ($O(T^2)$) with sequence length, and the KV cache grows linearly, severely limiting the speed and memory efficiency during long-sequence inference. Gated linear recurrent models (e.g., Linear Attention, GLA, Mamba) offer $O(T)$ training and $O(1)$ inference memory, but pretraining them from scratch is prohibitively expensive.

**Linearization** is an emerging direction to transfer pretrained Transformer weights into linear recurrent architectures, obtaining highly efficient models at a very low cost. However, existing methods suffer from two core limitations:

**Architectural Overhead**: Methods like SUPRA and LoLCATs require introducing extra feature maps or gating modules that cannot reuse pretrained weights and must be trained from scratch.

**Optimization Vulnerability**: LoLCATs relies on a two-stage training process (attention transfer followed by LoRA fine-tuning), which is complex and prevents end-to-end optimization.

In addition, existing linearization methods overlook the **gating mechanism** in SOTA linear recurrent models, which is crucial for controlling memory retention/forgetting and enhancing sequence modeling expressivity.

## Method

### Core Idea: Key Projection Reuse for Gating

The parameter space of LLMs exhibits structural redundancy. The core insight of Liger is to **assign a dual role to the Key projection matrix $\mathbf{W_K}$**—it performs the original linear transformation and simultaneously derives the gating signal through a parameter-free Pooling operation:

$$\mathbf{G}_t = f(\boldsymbol{k}_t) = f(\boldsymbol{x}_t \mathbf{W}_K)$$

where $f(\cdot)$ is a Pooling operation (e.g., mean pooling) that requires no additional trainable parameters. This parameter sharing strategy ensures compatibility with pretrained weights while avoiding the introduction of auxiliary gating modules.

### Unified Gated Linear Recurrent Framework

Liger unifies various gated linear recurrent structures into a general formulation:

$$\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \phi(\boldsymbol{k}_t^\top) \boldsymbol{v}_t$$
$$\boldsymbol{o}_t = \phi(\boldsymbol{q}_t) \mathbf{S}_t$$

where $\mathbf{G}_t$ is generated via Pooling($\boldsymbol{k}_t$), and $\phi(\cdot)$ directly applies Softmax normalization (instead of learning feature maps). All parameters $\mathbf{W_Q}, \mathbf{W_K}, \mathbf{W_V}$ are inherited from the pretrained LLM without any extra modules.

By using different gating parameterizations, Liger can adapt to various gated linear recurrent structures (such as GLA, Mamba2, mLSTM, HGRN2, RWKV6, etc.).

### Liger Attention: Intra-Layer Hybrid Attention

Liger proposes an intra-layer weighted hybrid of **Gated Recurrent Modeling (GRM)** and **Sliding Window Attention (SWA)**:

$$\boldsymbol{o}_t = \alpha \cdot \text{GRM}(\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t) + \beta \cdot \text{SWA}(\boldsymbol{q}_t, \boldsymbol{k}_t, \boldsymbol{v}_t)$$

where $\alpha + \beta = 1$ (defaulting to 0.5 each), and the SWA window size is $w=64$. GRM performs global long-range modeling, while SWA preserves local softmax non-linearity. The overall complexity is $O(TWD + TD^2)$, maintaining linearity.

### Overall Architecture

- **Intra-layer Hybrid**: Each layer utilizes Liger Attention (GRM + SWA).
- **Inter-layer Hybrid**: A standard softmax attention block is inserted after every few (e.g., 7) Liger blocks.
- Standard components like Pre-Norm, MLP, and residual connections are preserved.
- End-to-end fine-tuning is performed on $\mathbf{W_Q}, \mathbf{W_K}, \mathbf{W_V}$ using LoRA (rank=8, alpha=8), training only 0.085% of the parameters.
- Training data: 50K cleaned Alpaca instruction data (approximately 0.02B tokens), trained for 2 epochs.

## Key Experimental Results

### Comparison of Linearization Methods (Llama-3-8B)

| Model | Training Tokens(B) | PiQA | ARC-e | ARC-c | HellaSwag | WinoGrande | MMLU | Avg |
|------|------------|------|-------|-------|-----------|------------|------|-----|
| Llama-3-8B (Original) | 15000 | 79.4 | 80.1 | 53.2 | 79.2 | 72.9 | 65.3 | 71.7 |
| SUPRA | 20 | 78.9 | 75.1 | 46.5 | 71.7 | 65.8 | 40.9 | 63.2 |
| LoLCATs (Two-stage) | 0.04 | 80.1 | 80.4 | 53.5 | 63.4 | 72.9 | 42.1 | 65.4 |
| **Liger-GLA (Ours)** | **0.02** | **80.3** | **81.1** | **52.5** | **76.3** | **72.0** | **43.4** | **67.6** |

### Comparison with Pretrained Linear Models

| Model | Training Tokens(B) | Avg (no MMLU) |
|------|------------|---------------|
| Mamba-7B | 1200 | 71.0 |
| RWKV-6-7B | 1420 | 69.4 |
| Griffin-7B | 300 | 71.1 |
| Zamba2-7B (Hybrid) | 2100 | 75.3 |
| **Liger-GLA-8B** | **0.02** | **72.4** |

Liger outperforms linear models trained from scratch on hundreds of billions of tokens using only 0.02B tokens.

### Scaling Analysis

| Model Scale | Llama-3 | Liger-GLA | Recovery Rate |
|---------|---------|-----------|--------|
| 1B | 59.9 | 59.0 | 98.5% |
| 3B | 68.1 | 66.5 | 97.7% |
| 8B | 73.0 | 72.4 | 99.2% |

The larger the model scale, the better the performance recovery (the gap narrows from 4.8% to 1.8% from 1B to 8B).

### Ablation Study

| Variant | PPL↓ | Avg (no MMLU)↑ |
|------|------|----------------|
| Liger-GLA (Full) | 2.96 | 72.4 |
| Randomly Initialized Gate | 3.16 | 68.8 |
| w/o SWA | 3.75 | 60.2 |
| w/o LoRA | 3.23 | 68.1 |
| Pure Linear Attention (No Gate) | 3.00 | 71.5 |
| Extra Feature Mapping Module | 9.04 | 40.2 |

SWA contributes the most (removing it results in a 12.2 point drop in performance); introducing an extra feature mapping severely degrades performance instead.

## Highlights & Insights

- **Zero Extra Parameters**: Gating is constructed using the Pooling of Key projections, completely reusing pretrained weights. This represents the simplest linearization scheme.
- **Extremely Low Cost**: Requires only 0.02B tokens (0.02% of the pretraining volume), enabling linearization execution on a single A800 GPU.
- **Unified Framework**: A single methodology compatible with various gated recurrent structures such as GLA, HGRN2, and GSA.
- **Linear Inference**: The linearized model displays $O(T)$ decoding latency and constant activation memory, offering significant advantages at 32K sequence lengths.
- **Elegant Design of Liger Attention**: SWA retains local softmax non-linear information, while GRM manages global modeling, creating a complementary synergy.

## Limitations & Future Work

- Although the Avg (no MMLU) performance is recovered to 99%, the **MMLU recovery rate is relatively low** (43.4 vs 65.3), indicating a notable gap in knowledge-intensive tasks.
- The sliding window size $w=64$ is a fixed hyperparameter; dynamic or adaptive window sizes have not been explored.
- Validated only on Llama-3 and Mistral, without coverage of newer architectures (e.g., Qwen, Gemma).
- The training data only leverages the Alpaca instruction set; data quality and diversity may constrain the upper bound of recovery.
- Real-world performance on long-context tasks (e.g., 128K) has not been fully evaluated.

## Rating
- Novelty: ⭐⭐⭐⭐ — The idea of reusing Key projections to construct gating is elegant and simple, avoiding extra parameters.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Complete evaluations across model scales, structural variants, efficiency analyses, and ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations and intuitive framework diagrams.
- Value: ⭐⭐⭐⭐ — Provides a low-cost linearization path for LLM deployment with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] L2D: Large Language Models to Diffusion Finetuning](large_language_models_to_diffusion_finetuning.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](../../ICML2026/physics/softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](../../NeurIPS2025/physics/encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[ICML 2025\] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)
- [\[ICML 2025\] Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)

</div>

<!-- RELATED:END -->
