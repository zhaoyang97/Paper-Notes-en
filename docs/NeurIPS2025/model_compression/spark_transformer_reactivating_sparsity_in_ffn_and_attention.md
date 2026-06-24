---
title: >-
  [Paper Note] Spark Transformer: Reactivating Sparsity in FFN and Attention
description: >-
  [NeurIPS 2025][Model Compression][activation sparsity] This paper proposes the Spark Transformer architecture, which simultaneously achieves high-level activation sparsity in both FFN and attention mechanisms (only 8% of neurons activated in FFN; each token attends to at most 256 tokens) via a Statistical Top-k operator. The approach achieves a 2.5× FLOPs reduction and up to 1.79× inference speedup while maintaining quality comparable to Gemma-2.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "activation sparsity"
  - "Statistical Top-k"
  - "inference acceleration"
  - "FFN sparsity"
  - "attention sparsity"
date: 2026-05-08
content_hash: 510899616956b4aa
---

# Spark Transformer: Reactivating Sparsity in FFN and Attention

**Conference**: NeurIPS 2025
**arXiv**: [2506.06644](https://arxiv.org/abs/2506.06644)  
**Code**: None (Google internal project using Gemma-2 and gemma.cpp)  
**Area**: Model Compression / Efficient Transformers
**Keywords**: activation sparsity, Statistical Top-k, inference acceleration, FFN sparsity, attention sparsity

## TL;DR
This paper proposes the Spark Transformer architecture, which simultaneously achieves high-level activation sparsity in both FFN and attention mechanisms (only 8% of neurons activated in FFN; each token attends to at most 256 tokens) via a Statistical Top-k operator. The approach achieves a 2.5× FLOPs reduction and up to 1.79× inference speedup while maintaining quality comparable to Gemma-2.

## Background & Motivation

Activation sparsity is a key technique for reducing computational costs in large models. Early ReLU-based Transformers (e.g., T5, ViT) naturally exhibit a "lazy neuron" phenomenon—the vast majority of FFN neurons are inactive for any given token. Prior work has successfully exploited this sparsity to achieve practical acceleration on CPUs, GPUs, and TPUs.

The core challenge, however, is that **modern state-of-the-art Transformers have abandoned ReLU in favor of non-ReLU gated activation functions such as GELU** (e.g., Gemma, LLaMA, Mistral), which produce activations with little to no natural sparsity. The open question is: how can high-level sparsity be reintroduced without degrading model quality?

Existing attempts face three major challenges:
- **Challenge #1 (Quality Degradation)**: Naively switching back to ReLU or applying simple top-k masking reduces model quality.
- **Challenge #2 (Training Slowdown)**: Standard top-k requires sorting, which can cause up to 10× training slowdown on accelerators such as TPUs.
- **Challenge #3 (Extra Parameters/Complexity)**: Introducing sparsity predictors typically increases parameter count and training pipeline complexity.

## Method

### Overall Architecture

Spark Transformer comprises two components: Spark FFN and Spark Attention. Both are built on a unified design philosophy—treating FFN and attention as **key-value lookup tables**, using a low-rank predictor to identify active entries, and then performing full computation only for the active subset. The core technical tool is the Statistical Top-k operator, which replaces conventional sort-based top-k.

### Key Designs

1. **Spark FFN (Core Architectural Innovation)**:

    - The input $q$ is partitioned along the feature dimension into $q[:r]$ and $q[r:]$.
    - $q[:r]$ is multiplied by $K_1$ as a low-rank predictor; Top-k selects the $k$ most important neurons.
    - $q[r:]$ is multiplied by the sparse columns of $K_2$, computing only the $k$ selected columns.
    - $V$ computes only the corresponding $k$ rows.
    - Formula: $\text{Spark-FFN}(q) = V \cdot (\sigma(\text{Top}_k(K_1^\top \cdot q[:r])) \odot (K_2^\top \cdot q[r:]))$
    - **FLOPs Analysis**: With $r \approx d_\text{model}/2$, total FLOPs are approximately $d_\text{model} \cdot d_\text{ff} + 3 \cdot d_\text{model} \cdot k$; for small $k$, this is roughly $1/4$ of a standard FFN.
    - **No Parameter Increase**: $K_1$ and $K_2$ are obtained by partitioning the original $K$ along the feature dimension, so total parameter count equals that of a Gated FFN.

2. **Spark Attention (Unified Sparse Framework)**:

    - Attention has the same form as FFN (Eq. 6 vs. Eq. 1), enabling the same strategy to be applied.
    - Key vectors $K$ are partitioned into $K_1$ (for prediction) and $K_2$ (for computation).
    - $\sigma_1 = \text{softmax}$, $\sigma_2 = \text{softplus}$ (chosen empirically).
    - Each token attends to at most $k_\text{attn} = 256$ tokens.
    - FLOPs are approximately $d_\text{model} \cdot n_\text{ctx} + 3 \cdot d_\text{model} \cdot \min\{k_\text{attn}, n_\text{ctx}\}$; when $k_\text{attn} \ll n_\text{ctx}$, this approaches a 4× reduction.

3. **Statistical Top-k (Core Technical Tool)**:

    - **Motivation**: Standard top-k requires $O(d \log d)$ sorting, which is extremely slow on accelerators; moreover, hard thresholding is non-differentiable.
    - **Core Idea**: Assuming activation scores approximately follow a Gaussian distribution, the sample mean and standard deviation are used to estimate a threshold $\theta$ such that approximately $k$ entries exceed it.
    - Threshold formula: $\theta(x, k) = \text{mean}(x) + \text{std}(x) \cdot Q(1 - k/d)$, where $Q$ is the standard normal quantile function.
    - A **soft-threshold operator** (rather than a hard threshold) is then applied: $\text{Soft-Threshold}(x, \theta) = \max\{x - \theta \cdot \mathbf{1},\ 0\}$.
    - **Theoretical Guarantee (Theorem 3.1)**: The deviation between the actual number of selected entries and $k$ is $O(\sqrt{\log d / d})$.
    - **Computational Complexity**: Requires only $2d$ FLOPs (analogous to LayerNorm), far below the $O(d \log d)$ cost of sorting.
    - **Differentiability (Theorem 3.2)**: The operator becomes continuously differentiable after smoothing with a Huber function.

4. **Relationship to Gated FFN**: Spark FFN is structurally similar to Gated FFN (as used in Gemma)—both employ two linear projections in the first layer and one in the second. The key differences are: (1) Top-k is introduced to enforce structured sparsity; (2) the input is partitioned along the feature dimension rather than shared.

### Loss & Training

The training data and procedure are identical to those of Gemma-2 2B (2T tokens of English text). The elegance of Spark Transformer's design lies in:
- No additional parameters (FFN parameters and attention key embeddings are reused via dimension partitioning).
- No multi-stage training (all parameters are trained in a single stage).
- The linear complexity and approximate differentiability of Statistical Top-k avoid training slowdowns.

## Key Experimental Results

### Main Results

| Model | Train Loss (relative) | FLOPs/token (relative) | Downstream Tasks | Notes |
|---|---|---|---|---|
| Gemma-2 2B | 1.00 | 1.00 | Baseline | No sparsity |
| ReLU replacement | ~1.02 | ~0.65 | Large quality drop | Natural sparsity, poor quality |
| ReLU² replacement | ~1.005 | ~0.80 | Slight quality drop | Insufficient FLOPs reduction |
| Top-k + GELU | ~1.02 | ~0.55 | Large quality drop | Top-k without predictor |
| **Spark FFN** | ~1.005 | **~0.55** | Near-identical quality | Predictor improves both quality and efficiency |
| **Spark Transformer** | **~1.00** | **~0.40** | **On par with Gemma-2** | FFN + Attention sparsity |

### Ablation Study

| Platform | Gemma-2 Decode | Spark Decode | Speedup |
|---|---|---|---|
| 4-Core CPU | 141 ms/token | 86 ms/token | 1.64× |
| 16-Core CPU ($L=512$) | baseline | — | 1.79× |
| NVIDIA T4 GPU ($L=4096$) | baseline | — | 1.40× |

| CPU Prefill | Gemma-2 | Spark | Speedup |
|---|---|---|---|
| 4-Core (4096 tokens) | 28 ms/token | 15 ms/token | 1.86× |

### Key Findings
- **Surprising Effect of the Predictor**: The only difference between Spark FFN and naive Top-k+GELU is the introduction of the low-rank predictor, yet this not only reduces FLOPs but unexpectedly **improves model quality**.
- Statistical Top-k stably maintains the target sparsity level (8% FFN, ≤256 attention) throughout training.
- Sparsity reduces not only FLOPs but also memory bandwidth requirements, yielding speedups even in memory-bound scenarios (e.g., decode).
- The combination of Spark Attention and Spark FFN further improves quality, suggesting complementary effects between the two forms of sparsity.

## Highlights & Insights
- **Elegant Design of Statistical Top-k**: By leveraging a Gaussian assumption and quantile functions, top-k is approximated in $O(d)$ time. Its LayerNorm-like computation runs efficiently on accelerators, avoiding the sorting bottleneck that has long hindered this research area.
- **Parameter Reuse via Dimension Partitioning**: Partitioning the FFN key matrix along the feature dimension into a "predictor" and a "computer" enables sparse prediction without any parameter overhead.
- **Unified View of FFN and Attention**: Treating both as key-value lookup tables enables a single framework to introduce sparsity into both components.
- The technique may also be applicable to MoE routing mechanisms, avoiding the $O(d \log d)$ sorting overhead therein.

## Limitations & Future Work
- Validation is currently limited to Gemma-2 2B; effectiveness at larger scales (e.g., 7B/70B) remains to be demonstrated.
- The Gaussian assumption underlying Statistical Top-k may become less precise after training, although experiments suggest it remains approximately valid.
- Hardware-efficient implementations on CPUs and GPUs may require platform-specific optimizations.
- The FFN sparsity level of 8% and the attention cap of 256 are fixed; adaptive adjustment may yield further improvements.
- Experiments are limited to language modeling; performance on vision and multimodal models is unknown.

## Related Work & Insights
- **vs. ReLU Sparsity (Li et al. 2022)**: The original lazy neuron phenomenon is confined to ReLU-based Transformers. Spark achieves controllable sparsity on GELU-based models via top-k, offering greater flexibility.
- **vs. DejaVu (Liu et al. 2023)**: DejaVu uses an auxiliary MLP predictor, increasing parameter count and training complexity. Spark avoids these issues through dimension partitioning.
- **vs. Soft Top-k (Lei et al. 2023) / SparseK (Lou et al. 2024)**: These methods define top-k via optimization problems requiring iterative solvers. Statistical Top-k admits a closed-form solution and guarantees approximately $k$ nonzero outputs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Both Statistical Top-k and dimension-partitioned predictors are elegant innovations; the unified framework is highly convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full Gemma-2 pretraining validation and multi-platform inference evaluation, though experiments are limited to a single model scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, complete theoretical treatment, and excellent figures.
- Value: ⭐⭐⭐⭐⭐ Addresses a core pain point of activation sparsity in modern Transformers with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[NeurIPS 2025\] SpecAttn: Speculating Sparse Attention](specattn_speculating_sparse_attention.md)
- [\[NeurIPS 2025\] Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)
- [\[NeurIPS 2025\] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)

</div>

<!-- RELATED:END -->
