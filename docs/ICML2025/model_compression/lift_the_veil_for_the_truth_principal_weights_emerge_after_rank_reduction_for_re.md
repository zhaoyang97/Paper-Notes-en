---
title: >-
  [Paper Note] LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reliable Model Merging
description: >-
  [ICML 2025][Model Compression][sparse fine-tuning] It is discovered that the weights with the largest magnitude after low-rank approximation (Principal Weights) are the critical parameters for fine-tuning. LIFT is proposed, which updates only the top 5% of Principal Weights to outperform full-parameter fine-tuning on reasoning tasks while maintaining LoRA-level memory efficiency.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "sparse fine-tuning"
  - "rank reduction"
  - "principal weights"
  - "LoRA"
  - "reasoning"
date: 2026-05-08
content_hash: 5bc244cc8e191de9
---

# LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reliable Model Merging

**Conference**: ICML 2025  
**arXiv**: [2506.00772](https://arxiv.org/abs/2506.00772)  
**Code**: [github.com/zihanghliu/LIFT](https://github.com/zihanghliu/LIFT)  
**Area**: Model Compression / Fine-tuning  
**Keywords**: sparse fine-tuning, rank reduction, principal weights, LoRA, reasoning

## TL;DR
It is discovered that the weights with the largest magnitude after low-rank approximation (Principal Weights) are the critical parameters for fine-tuning. LIFT is proposed, which updates only the top 5% of Principal Weights to outperform full-parameter fine-tuning on reasoning tasks while maintaining LoRA-level memory efficiency.

## Background & Motivation
**Background**: Fine-tuning (SFT) with small amounts of high-quality data can achieve strong reasoning capabilities, but Full FT is computationally expensive and prone to overfitting/forgetting.

**Limitations of Prior Work**: Sparse fine-tuning lags behind in the LLM era—it is difficult to identify parameters critical for reasoning. In particular, magnitude-based sparse fine-tuning performs poorly on LLMs.

**Key Challenge**: LoRA saves memory but falls short of Full FT in reasoning performance; Full FT performs well but tends to forget source domain knowledge.

**Goal**: Find a method that outperforms Full FT while maintaining memory efficiency and preserving source domain knowledge.

**Key Insight**: Low-rank approximation can "reveal" critical weights—those with the largest magnitude in the residual are the Principal Weights.

**Core Idea**: Magnitude-based sparse fine-tuning itself is poor, but applying low-rank approximation first and then selecting by magnitude is highly effective. Low-rank approximation removes the general structure, and the large-magnitude weights in the residual are the critical components for task specialization.

## Method

### Overall Architecture
LIFT pipeline: SVD truncation $\rightarrow$ residual calculation $\rightarrow$ select top 5% magnitude positions as a mask $\rightarrow$ update only weights within the mask.

### Key Designs
1. **Low-rank approximation reveals key weights**: The main structure of the pre-trained weights is captured by the first few singular values (general knowledge). The large-magnitude elements in the residual $\mathbf{R} = \mathbf{W} - \mathbf{W}_r$ represent the "specialized structures" that do not align with the low-rank structure. Design Motivation: Direct magnitude-based selection fails because large-magnitude weights can be part of the general structure. Only after removing the low-rank approximation do the truly critical specialized weights emerge.

2. **Sparsity selection**: A 5% update rate provides LoRA-level memory efficiency and avoids the expressiveness limitations imposed by LoRA's low-rank constraints.

3. **Source domain knowledge preservation**: Since 95% of the weights remain unchanged, pre-trained knowledge is naturally preserved. Full FT updates all parameters, leading to easy forgetting, while LoRA has fewer parameters but its update mechanism can still disturb the original structure.

### Loss & Training
Standard SFT cross-entropy loss is used, where gradients only apply to the masked regions of Principal Weights. The mask is determined before training and remains fixed.

## Key Experimental Results

### Main Results

| Method | GSM8K | MATH | ARC | Avg | Source Retention |
|------|-------|------|-----|------|---------|
| Full FT | High | High | High | Baseline | -20% |
| LoRA | Medium | Medium | Medium | Below FT | Medium |
| **LIFT (5%)** | **Highest** | **Highest** | **Highest** | **Outperforms FT** | **Best** |
| Direct Mag FT | Low | Low | Low | Poor | Medium |

### Ablation Study

| Configuration | Reasoning Performance | Source Retention | Description |
|------|---------|---------|------|
| LIFT 5% | Best | Best (+20%) | Full method |
| W/o low-rank (Direct Mag) | Poor | Medium | Validates the criticality of low-rank approximation |
| Different sparsity (1/5/10%) | 5% Optimal | Smaller is better | 5% is the sweet spot |
| Different rank r | Insensitive | - | Good robustness |

### Key Findings
- **Surprising Finding**: Magnitude-based sparse fine-tuning becomes highly effective after low-rank approximation.
- Updating 5% of parameters outperforms 100% updates, demonstrating that precise selection is more important than brute-force updates.
- Source domain preservation is 20% better than Full FT.

## Highlights & Insights
- Reveals the collaborative effect of "low-rank approximation + magnitude selection."
- Extremely simple method (SVD + threshold) with no extra trainable parameters.
- Provides a new perspective for understanding weight structures: low-rank = general, high-value residual = specialized.

## Limitations & Future Work
- SVD incurs a one-time overhead; extremely large models may require approximations.
- Only tested on reasoning tasks; performance on other tasks is unknown.
- Limited comparison with other PEFT methods.

## Related Work & Insights
- Complementary to LoRA: Reveals two "parameter-efficient" paradigms: low-rank vs. sparsity.
- Future work can consider combining LIFT + LoRA.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery that low-rank approximation reveals key weights is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-task ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation.
- Value: ⭐⭐⭐⭐⭐ Direct guidance for practical LLM fine-tuning.

---

## Supplementary Reflections

### Relationship with Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: (1) the increasing demand for an in-depth understanding of LLM internal mechanisms; (2) the rising importance of model efficiency and accessibility; and (3) AI safety and reliability becoming core concerns. From a methodological perspective, this work represents a paradigm shift from "black-box usage" to "white-box understanding."

### Specific Suggestions for Future Work
1. The core idea of this paper can be combined with other modalities (vision, audio).
2. Consider validating the generalizability of the findings on larger-scale models and datasets.
3. Explore the possibility of integration with reinforcement learning and online learning.
4. Develop automated evaluation and optimization toolchains.


---

## Supplementary Reflections

### Relationship with Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: model capability evaluation and reliability assurance, parameter-efficient fine-tuning and model compression, and AI safety and alignment. From a methodological perspective, this work represents an exploration of the deeper mechanisms of LLMs, helping to drive the paradigm shift from empirical-driven to theory-driven research.

### Specific Suggestions for Future Work
1. Combine the core idea with other modalities (vision, audio, multimodal) to verify cross-modal generalizability.
2. Validate the conclusions on larger-scale models (70B+) and newer architectures (e.g., Mixture-of-Experts).
3. Explore the possibility of integration with reinforcement learning and online learning to achieve dynamic adaptation.
4. Develop automated evaluation and optimization tools to lower the barrier of entry for the method.
5. Consider the intersection with LLM alignment research to explore the collaborative optimization of safety and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](../../NeurIPS2025/model_compression/accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[ICML 2025\] From Low Rank Gradient Subspace Stabilization to Low-Rank Weights: Observations, Theories, and Applications](from_low_rank_gradient_subspace_stabilization_to_low-rank_weights_observations_t.md)
- [\[ICML 2025\] BECAME: BayEsian Continual Learning with Adaptive Model MErging](became_bayesian_continual_learning_with_adaptive_model_merging.md)
- [\[ICML 2025\] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](bring_reason_to_vision_understanding_perception_and_reasoning_through_model_merg.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](../../ICLR2026/model_compression/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)

</div>

<!-- RELATED:END -->
