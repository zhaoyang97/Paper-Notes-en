---
title: >-
  [Paper Note] Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability
description: >-
  [NeurIPS 2025][Robotics][Adversarial transferability] By systematically exploiting data-level and model-level computation redundancy in ViTs, this paper proposes five techniques—attention sparsification…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Adversarial transferability"
  - "ViT computation redundancy"
  - "attention sparsification"
  - "Ghost MoE"
  - "robust tokens"
date: 2026-05-08
content_hash: a1c4cfca77c6dec5
---

# Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability

**Conference**: NeurIPS 2025
**arXiv**: [2504.10804](https://arxiv.org/abs/2504.10804)  
**Code**: None  
**Area**: Robotics
**Keywords**: Adversarial transferability, ViT computation redundancy, attention sparsification, Ghost MoE, robust tokens

## TL;DR
By systematically exploiting data-level and model-level computation redundancy in ViTs, this paper proposes five techniques—attention sparsification, attention head permutation, clean token regularization, Ghost MoE diversification, and robust tokens—combined with an online learning strategy that dynamically selects operations. The method achieves an average fooling rate of 86.9% on ImageNet-1K, substantially outperforming all baselines.

## Background & Motivation

1. **Adversarial transferability advantage of ViTs**: Prior work has shown that adversarial examples generated from ViTs transfer more effectively than those from CNNs, yet the underlying reasons remain poorly understood.
2. **Insight into computation redundancy**: ViTs contain substantial redundancy at both the data level (tokens can be heavily pruned without performance degradation) and the model level (redundant attention heads, and neuron redundancy in FFNs induced by dropout training).
3. **Core hypothesis**: Such redundancy is not wasteful but can be repurposed to enhance adversarial transferability—by diversifying the model's intermediate representations to reduce overfitting to the surrogate model.

## Core Problem
How to systematically harness the inherent computation redundancy of ViTs to improve the transferability of adversarial examples from surrogate models to black-box victim models?

## Method

### Technique 1: Attention Sparsification
A random binary mask is applied to attention logits with a drop ratio $r$:

$$\text{MHA}(\mathbf{z}) = \text{softmax}\left(\left(\frac{\mathbf{z}\mathbf{W}_Q(\mathbf{z}\mathbf{W}_K)^\top}{\sqrt{d_k}}\right) \odot \mathbf{M}\right)\mathbf{z}\mathbf{W}_V$$

where $\mathbf{M}\in\{0,1\}^{N\times N}$ is a random mask with drop ratio $r$. Results show that white-box success rates remain nearly unchanged for $r\leq 0.4$, while black-box transferability improves significantly.

### Technique 2: Attention Head Permutation
The QK weights across different attention heads are randomly shuffled while V weights remain fixed:

$$\text{MHA}(\mathbf{z}) = \text{Concat}\left(\text{softmax}\left(\pi\left(\frac{\mathbf{Q}_1\mathbf{K}_1^\top}{\sqrt{d_k}},...,\frac{\mathbf{Q}_H\mathbf{K}_H^\top}{\sqrt{d_k}}\right)\right)[\mathbf{V}_1,...,\mathbf{V}_H]^T\right)$$

Each layer is selected with probability $p$, and a fraction $r$ of heads in selected layers are randomly permuted. This validates that redundant heads learn similar attention patterns.

### Technique 3: Clean Token Regularization
A small number of tokens from clean samples are inserted into each Transformer block as stable anchors to regularize adversarial representations. A sampling ratio of $r=0.3\sim0.5$ yields the best performance.

### Technique 4: Ghost MoE Diversification
Multiple "ghost experts" are instantiated from the same FFN using different dropout masks:

$$\text{MoE}(\mathbf{z}) = \frac{1}{q}\sum_{e=1}^q \text{FFN}_{\theta_e}(\mathbf{z}), \quad q\sim\mathcal{U}(1,E)$$

The optimal configuration uses a drop rate of 0.3, and performance improves consistently as the number of experts increases.

### Technique 5: Robust Tokens
$N_r$ learnable tokens are appended after patch embedding and optimized via test-time adversarial training:

$$\min_{\mathbf{z}_r}\max_\delta \mathcal{L}(f(x+\delta;\mathbf{z}_r), y)$$

Both the dynamic variant (per-sample optimization) and the global variant (offline pretraining on a calibration set) are effective, yielding improvements exceeding 14%.

### Online Learning Strategy
A sampling matrix $\mathbf{M}\in\mathbb{R}^{L\times O}$ is initialized, and at each layer $l$, operation $\phi_o$ is selected with a learned probability. The operation distribution is optimized via REINFORCE:

$$\max_\mathbf{M}\;\mathbb{E}_{\phi\sim\mathbf{M}}[\mathcal{L}(f(x+\delta(\phi)),y)]$$

This adaptively selects the optimal redundancy exploitation strategy for each Transformer block.

## Key Experimental Results

### ViT-B/16 as Surrogate Model

| Method | RN-50 | VGG-16 | MN-V2 | Inc-v3 | ViT | PiT | Vis-S | Swin | Avg. |
|--------|-------|--------|-------|--------|-----|-----|-------|------|------|
| MI-FGSM | 39.4 | 58.4 | 57.9 | 42.2 | 97.4 | 40.4 | 42.0 | 55.0 | 54.1 |
| PGN | 68.9 | 75.7 | 76.3 | 72.4 | 97.6 | 75.6 | 75.5 | 80.0 | 77.8 |
| TGR | 53.4 | 72.5 | 72.4 | 55.5 | 97.7 | 59.2 | 61.8 | 74.5 | 68.4 |
| **Ours** | **77.7** | **90.6** | **91.1** | **79.9** | **99.7** | **78.9** | **83.5** | **93.5** | **86.9** |

### PiT-B / Swin-T as Surrogate

| Surrogate | Method | Avg. Fooling Rate |
|-----------|--------|-------------------|
| PiT-B | PGN | 75.1% |
| PiT-B | **Ours** | **87.4%** |
| Swin-T | PGN | 85.3% |
| Swin-T | **Ours** | **88.9%** |

### Attacking VLLMs (ViT surrogate → LLaVA/Qwen/InternVL/DeepSeek)
Average improvement of 2.2%, with a 5.5% gain over the runner-up on Qwen.

## Highlights & Insights
- **Systematic analysis**: The first work to unify the understanding of ViT adversarial transferability from the perspective of "computation redundancy."
- **Five complementary techniques**: Comprehensive coverage from data level (tokens) to model level (FFN/MHA).
- **Online learning strategy**: Adaptive operation combination that eliminates manual hyperparameter tuning.
- **Cross-architecture effectiveness**: Validated across CNNs, ViTs, and VLLMs.
- Outperforms the strongest baseline PGN by an average margin of 9+ percentage points.

## Limitations & Future Work
- Each of the five techniques has its own hyperparameters (drop ratio, number of experts, token count, etc.), resulting in a large joint search space.
- The per-sample optimization variant of robust tokens incurs considerable computational overhead.
- The online learning strategy relies on the REINFORCE estimator, which may exhibit high variance.
- Robustness against defense methods (e.g., adversarially trained models) is not thoroughly validated.

## Rating
- Novelty: ⭐⭐⭐⭐ — The "redundancy as resource" perspective is novel and the technical combination is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 target models, 3 surrogate models, VLLMs, and extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with a coherent progression from analysis to design.
- Value: ⭐⭐⭐⭐ — A significant advance in the field of adversarial transfer attacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets](../../AAAI2026/robotics/when_hallucination_costs_millions_benchmarking_ai_agents_in_high-stakes_adversar.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](learning_spatial-aware_manipulation_ordering.md)
- [\[NeurIPS 2025\] Can Agents Fix Agent Issues?](can_agents_fix_agent_issues.md)
- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)

</div>

<!-- RELATED:END -->
