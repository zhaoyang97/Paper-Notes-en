---
title: >-
  [Paper Note] Where and What Matters: Sensitivity-Aware Task Vectors for Many-Shot Multimodal In-Context Learning
description: >-
  [AAAI 2026][Reinforcement Learning][Multimodal In-Context Learning] This paper proposes the STV framework, which identifies attention head positions sensitive to in-context information via activation deltas…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Multimodal In-Context Learning"
  - "Task Vectors"
  - "Attention Head Sensitivity"
  - "Activation Space Modulation"
date: 2026-05-08
content_hash: 45e6997dd3cbe541
---

# Where and What Matters: Sensitivity-Aware Task Vectors for Many-Shot Multimodal In-Context Learning

**Conference**: AAAI 2026
**arXiv**: [2511.08246](https://arxiv.org/abs/2511.08246)  
**Code**: [https://github.com/AMAP-ML/STV](https://github.com/AMAP-ML/STV)  
**Area**: Reinforcement Learning
**Keywords**: Multimodal In-Context Learning, Task Vectors, Attention Head Sensitivity, Reinforcement Learning, Activation Space Modulation

## TL;DR

This paper proposes the STV framework, which identifies attention head positions sensitive to in-context information via activation deltas, and leverages reinforcement learning to select optimal task vectors from a pre-clustered activation bank for insertion—enabling efficient many-shot multimodal in-context learning without increasing input length.

## Background & Motivation

### Problem Introduction

Large multimodal models (LMMs) have demonstrated strong in-context learning (ICL) capabilities, yet scaling to many-shot settings faces two core challenges:

**Context length constraints**: Models such as Qwen-VL support a maximum of 8,192 tokens, while encoding a single image consumes 256 tokens, severely limiting the number of accommodatable demonstrations.

**Substantial inference overhead**: More demonstrations entail greater memory consumption and inference latency.

### Limitations of Prior Work

Task vector-based methods have been proposed as a promising solution, compressing many-shot demonstrations into compact representations inserted into model activations. However, existing approaches exhibit notable shortcomings:

- **Value Estimation methods**: ICV applies PCA over hidden states to compute compact vectors but inserts them at fixed positions, ignoring the critical impact of insertion location on performance, and generalizes poorly to complex multimodal tasks.
- **Location Selection methods**: MTV employs a policy network to optimize insertion positions but uses mean activations as fixed task vectors, causing loss of task information. Furthermore, its sampling-based policy is unstable, yielding inconsistent positions across runs.

### Core Findings and Motivation

The authors identify a key phenomenon: **activation deltas between query–context pairs exhibit consistent structural patterns**. Specifically:
- Under the same model and task, the distribution of positions with large activation deltas is consistent (e.g., on OK-VQA with Qwen-VL).
- Across different models (e.g., Qwen-VL vs. Idefics-2) and different tasks (e.g., VizWiz vs. OK-VQA), sensitive positions differ significantly.

This indicates that activation deltas serve as reliable indicators for identifying the processing pathways of in-context information, simultaneously addressing the two key questions of *where* and *what* to insert.

## Method

### Overall Architecture

The STV (Sensitivity-aware Task Vector insertion) framework consists of two stages:
1. **Sensitive Position Identification**: Activation deltas with and without context are computed to localize attention heads sensitive to in-context information.
2. **Task Vector Selection**: A pre-clustered activation bank is constructed for each sensitive position, and reinforcement learning selects the optimal task vector for insertion.

### Key Designs

#### 1. **Activation Delta Computation and Position Selection**

**Mechanism**: Quantify the sensitivity of each attention head to in-context information.

For query $q$, the activation at attention head $(l,h)$ is $A_q^{(l,h)}$; after concatenating $S$ in-context demonstrations with the query, the activation becomes $A_c^{(l,h)}$. Sensitivity is measured via L2 activation delta:

$$\bar{\Delta}^{(l,h)} = \frac{1}{T} \sum_{t=1}^{T} \|A_{c_t}^{(l,h)} - A_{q_t}^{(l,h)}\|_2$$

where $T$ is the number of sampled query–context pairs; averaging ensures estimation stability. The resulting matrix $\bar{\Delta} \in \mathbb{R}^{L \times H}$ reflects the context-sensitivity distribution across all attention heads. The Top-K most sensitive positions are selected:

$$\Lambda = \text{TopK}(\bar{\Delta}, K)$$

**Design Motivation**: Compared to MTV's position search requiring 6,000 seconds, this approach requires only 88 seconds (a 98.53% reduction) while producing more stable and reliable results.

#### 2. **Pre-Clustered Activation Bank Construction**

For each sensitive position $(l_k, h_k)$, activations from multiple query–context forward passes are collected and clustered via k-means into $M$ cluster centers:

$$\text{ClusterBank}[(l_k, h_k)] = \{\mathbf{v}_1^{(k)}, \ldots, \mathbf{v}_M^{(k)}\}$$

**Design Motivation**: Compared to mean activations (e.g., MTV), clustering preserves richer semantic diversity and avoids information loss through averaging. Experiments confirm that increasing the number of cluster centers steadily improves performance, saturating around 32.

#### 3. **Reinforcement Learning-Based Task Vector Selection**

Vector selection is formulated as a discrete optimization problem. A categorical distribution is defined for each position:

$$\mathbf{p}^{(k)} = \text{softmax}(\boldsymbol{\alpha}^{(k)})$$

where $\boldsymbol{\alpha}^{(k)} \in \mathbb{R}^M$ are learnable logits. During training, cluster indices are sampled from the distribution, inserted into the model, and a reward is computed:

$$r_i = -\mathcal{L}_{\text{task}}(F(x_i; \Lambda, \mathcal{V}_i), y_i)$$

The REINFORCE algorithm updates the policy:

$$\mathcal{L}_{\text{policy}} = -\sum_{i=1}^{N} \sum_{k=1}^{K} \log p_{i_k}^{(k)} \cdot \frac{r_i - \bar{r}}{\sigma_r + \epsilon}$$

Upon convergence, the cluster center with the highest probability at each position is used as the final task vector.

**Design Motivation**: RL efficiently searches the discrete candidate space without exhaustive enumeration. Variance-normalized baselines reduce gradient variance and accelerate convergence.

### Inference Strategy

Inference requires only a single forward pass: when the test input is processed by the model, activations at the designated positions are replaced with the corresponding task vectors. This introduces no additional input length or parameters, yielding inference overhead nearly identical to zero-shot inference.

## Key Experimental Results

### Main Results

Performance on 5 vision-language benchmarks (Qwen-VL-7B):

| Dataset | Metric | STV (Ours) | MTV (Prev. SOTA) | Zero-shot | 4-shot ICL |
|---------|--------|-----------|-----------------|-----------|------------|
| VizWiz | Acc | **58.30** | 45.60 | 35.21 | 42.00 |
| OK-VQA | Acc | **61.94** | 60.51 | 57.76 | 54.62 |
| DTD | Acc | **80.45** | 76.50 | 55.07 | 55.50 |
| Flowers | Acc | **81.51** | 78.10 | 55.24 | 54.67 |
| CUB | Acc | **82.33** | 80.00 | 56.50 | 56.16 |
| **Average** | Acc | **72.11** | 68.94 | 51.96 | 52.59 |

State-of-the-art results are also achieved on Idefics2-8B: average 76.04% vs. MTV 73.65%.

### Efficiency Comparison

| Metric | MTV | STV | Change |
|--------|-----|-----|--------|
| Position search time | 6000s | 88s | ↓98.53% |
| GPU memory | 19.8GB | 19.8GB | Unchanged |
| Inference time | 0.49s | 0.49s | Unchanged |
| VizWiz performance | 45.6% | 58.3% | ↑12.7% |

### Ablation Study

| Configuration | VizWiz Acc | Notes |
|---------------|-----------|-------|
| Zero-shot baseline | 35.2% | No augmentation |
| + Sensitive position selection (K=300) | 49.2% | Location contribution: +14.0% |
| + Clustered vector selection (N=32) | 58.3% | Vector contribution: +9.1% |
| STV + F.L. high-quality samples | 61.9% | Sample quality yields further gains |

Comparison with parameter fine-tuning methods:

| Method | VizWiz | OK-VQA | Notes |
|--------|--------|--------|-------|
| SFT | 62.0 (+26.8) | 25.1 (-33.5) | Severe overfitting |
| LoRA | 44.3 (+9.1) | 57.7 (-0.9) | Limited improvement |
| **STV** | **58.3 (+23.1)** | **61.9 (+3.3)** | No parameter updates required |

### Key Findings

1. Activation deltas exhibit task-dependent yet intra-model-consistent structural patterns, providing a reliable basis for position selection.
2. Increasing the number of cluster centers steadily improves performance (saturating around 32), confirming that clustering preserves richer semantic content than mean activations.
3. STV handles 400 samples without OOM, whereas standard ICL encounters OOM at 64-shot.
4. The impact of cross-domain noisy samples on STV (−0.7%) is smaller than on ICL (−1.0%), demonstrating greater robustness.

## Highlights & Insights

1. **Core insight is minimal yet effective**: Identifying context-sensitive positions via activation deltas is both intuitive and empirically verifiable.
2. **Two-stage decoupled design**: Treating "where" and "what" separately, each addressed by the most appropriate method (statistical analysis vs. RL), yields greater stability than end-to-end approaches.
3. **Strong practical utility**: The full pipeline runs on a single GPU (<20 GB), requires no model fine-tuning, and achieves inference speed equivalent to zero-shot inference.
4. **Dramatic search efficiency improvement**: Position search time is reduced from ~100 minutes to ~1.5 minutes, a 98.5% reduction.

## Limitations & Future Work

1. The Top-K selection of sensitive positions requires the hyperparameter $K$ to be predetermined; the optimal $K$ may vary across tasks.
2. Clustering is performed offline in a one-shot manner and cannot dynamically adapt to new data distributions.
3. Validation is limited to open-source models at the 7B/8B scale; effectiveness on larger models remains to be confirmed.
4. Performance degrades when the number of in-context samples is excessively large (overly large TT or shot count), suggesting information redundancy.

## Related Work & Insights

- **Relationship to MTV**: STV directly addresses MTV's two core issues—unstable sampling-based policy and information loss from mean activations.
- **Comparison with LoRA/SFT**: Demonstrates the unique advantage of activation-level modulation—achieving cross-task generalization without any parameter updates.
- **Inspiration**: Activation deltas as a general "structural probe" may be applicable to understanding internal mechanisms in other models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The observation underlying sensitive position discovery is novel, though the RL selection component is relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 5 datasets × 2 models, complete ablations, and clear efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically structured, richly illustrated, with well-motivated explanations.
- **Value**: ⭐⭐⭐⭐⭐ — Minimal resource requirements, plug-and-play deployment, and strong applicability to industrial settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)
- [\[ICLR 2026\] Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs](../../ICLR2026/reinforcement_learning/chain-of-context_learning_dynamic_constraint_understanding_for_multi-task_vrps.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](../../NeurIPS2025/reinforcement_learning/zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[ICLR 2026\] Scalable In-Context Q-Learning](../../ICLR2026/reinforcement_learning/scalable_in-context_q-learning.md)
- [\[AAAI 2026\] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)

</div>

<!-- RELATED:END -->
