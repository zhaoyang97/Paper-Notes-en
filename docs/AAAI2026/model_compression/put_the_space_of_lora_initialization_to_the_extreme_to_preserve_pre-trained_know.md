---
title: >-
  [Paper Note] Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge
description: >-
  [AAAI2026][Model Compression][LoRA] This paper proposes LoRA-Null, which initializes LoRA within the null space of pre-trained input activations (rather than the null space of weights). From an information-theoretic perspective, the effective rank of activations is much lower than that of weights, meaning their null space encodes less pre-trained knowledge, thereby substantially mitigating catastrophic forgetting during fine-tuning.
tags:
  - "AAAI2026"
  - "Model Compression"
  - "LoRA"
  - "catastrophic forgetting"
  - "knowledge preservation"
  - "null space"
  - "activation-aware initialization"
date: 2026-05-08
content_hash: af71141d51df6fda
---

# Put the Space of LoRA Initialization to the Extreme to Preserve Pre-trained Knowledge

**Conference**: AAAI2026
**arXiv**: [2503.02659](https://arxiv.org/abs/2503.02659)  
**Code**: [HungerPWAY/LoRA-Null](https://github.com/HungerPWAY/LoRA-Null)  
**Area**: Model Compression
**Keywords**: LoRA, catastrophic forgetting, knowledge preservation, null space, activation-aware initialization

## TL;DR
This paper proposes LoRA-Null, which initializes LoRA within the null space of pre-trained input activations (rather than the null space of weights). From an information-theoretic perspective, the effective rank of activations is much lower than that of weights, meaning their null space encodes less pre-trained knowledge, thereby substantially mitigating catastrophic forgetting during fine-tuning.

## Background & Motivation
Although LoRA is parameter-efficient, it still suffers from significant catastrophic forgetting. Existing LoRA initialization methods for alleviating forgetting follow two lines:

**Making the residual weight $\mathbf{W}_0'$ close to the pre-trained weight $\mathbf{W}_0$**: Both CorDA and MiLoRA pursue this objective.

**Initializing LoRA in a subspace orthogonal to pre-trained knowledge**: MiLoRA uses the null space of $\mathbf{W}_0$.

A key observation is that LoRA adapters undergo relatively small changes after fine-tuning ($\|\mathbf{A}^* - \mathbf{A}_0\|_F / \|\mathbf{A}_0\|_F$ is small), making the initialization subspace of the adapter more critical than the residual weight. Moreover, vanilla LoRA freezes the complete $\mathbf{W}_0$ yet still suffers severe forgetting, indicating that "keeping the residual weight close" is not the decisive factor.

Core insight: The input activation $\mathbf{X}_\text{pre}$ aggregates information from all preceding layers and input data (whereas $\mathbf{W}_0$ contains only the current layer's information), and its effective rank is far smaller than that of $\mathbf{W}_0$ (information is more concentrated), so its null space carries less pre-trained knowledge.

## Method

### LoRA-Null Pipeline
1. **Collect calibration data**: Randomly sample 256 examples from a dataset representative of pre-trained knowledge (e.g., NQ Open) and perform a forward pass to obtain per-layer input activations $\mathbf{X}_\text{pre} \in \mathbb{R}^{d_\text{in} \times BL}$.
2. **Extract the null space**: Apply SVD to $\mathbf{X}_\text{pre}$ and take the left singular vectors corresponding to the smallest $r$ singular values, $\mathbf{U}_\text{null} \in \mathbb{R}^{d_\text{in} \times r}$, satisfying $\mathbf{U}_\text{null}^\top \mathbf{X}_\text{pre} \approx \mathbf{0}$.
3. **Project and initialize**: Project $\mathbf{W}_0$ onto the null space: $\mathbf{W}_0 \mathbf{U}_\text{null} \mathbf{U}_\text{null}^\top$, then apply SVD to the projected result to initialize $\mathbf{A}$ and $\mathbf{B}$.
4. **Residual weight**: $\mathbf{W}_0' = \mathbf{W}_0 - \mathbf{BA}$ (no constraint requiring proximity to $\mathbf{W}_0$).
5. Only $\mathbf{A}$ and $\mathbf{B}$ are updated during fine-tuning.

### Theoretical Analysis
- **Theorems 1 & 2**: MiLoRA and CorDA are respectively shown to be the solutions minimizing $\|\mathbf{W}_0' - \mathbf{W}_0\|_F$ and $\|\mathbf{W}_0' \mathbf{X}_\text{pre} - \mathbf{W}_0 \mathbf{X}_\text{pre}\|_F$.
- **Theorem 3**: LoRA-Null satisfies neither of the above objectives — it entirely abandons residual weight constraints and maximizes null space orthogonality.
- **Effective rank comparison**: The eRank of $\mathbf{X}_\text{pre}$ is far lower than that of $\mathbf{W}_0$ (e.g., for LLaMA-3.2-3B layer 0 k-proj: 101 vs. 548), confirming that information is more concentrated in the principal subspace.

## Key Experimental Results

Evaluations are conducted on LLaMA-2-7B and LLaMA-3.2-3B across three tasks: Math, Code, and Instruction Following.

### LLaMA-2-7B Math Task

| Method | TriviaQA | NQ Open | WebQS | Avg1(Per) | GSM8k | Math | GM |
|---|---|---|---|---|---|---|---|
| LoRA | 45.95 | 1.16 | 6.64 | 64.56% | 42.99 | 6.26 | 21.00 |
| MiLoRA | 47.02 | 3.66 | 6.10 | 69.66% | 41.47 | 6.20 | 21.24 |
| CorDA | 48.99 | 7.15 | 5.76 | 76.24% | 41.47 | 8.22 | 22.64 |
| **LoRA-Null** | **50.02** | **7.98** | **6.55** | **79.21%** | **44.43** | **8.80** | **23.93** |

- Knowledge retention (Avg1 Per) improves by approximately **3%** over CorDA and **10%** over MiLoRA.
- LoRA-Null simultaneously achieves the best downstream performance on GSM8k and Math, demonstrating a dual gain in knowledge retention and task performance.
- Consistent improvements are observed on LLaMA-3.2-3B as well.

### Hyperparameter Analysis
- **Calibration data size**: LoRA-Null is less sensitive to data volume than CorDA (CorDA's knowledge retention drops to 69% with 64 samples, while LoRA-Null maintains 95%).
- **LoRA rank**: CorDA's knowledge retention degrades more rapidly as rank increases; LoRA-Null remains more stable.

## Highlights & Insights
- **Clear core insight**: "The orthogonality of the LoRA initialization subspace matters more than the proximity of the residual weight" — a concise and compelling finding.
- **Activation vs. weight null space**: The superiority of the activation null space is argued from two angles: information volume (effective rank) and information source (all layers vs. a single layer).
- **Theory–experiment alignment**: Theorems 1–3 precisely characterize the positioning of each method, and the projection analysis in Figure 4 visually confirms that LoRA-Null adapters reside exclusively within the null space.
- **Hyperparameter robustness**: LoRA-Null demonstrates greater stability than CorDA with respect to variations in calibration data size and rank.

## Limitations & Future Work
- Additional forward passes are required to collect activations (256 samples × 1024 sequence length), incurring non-trivial computational overhead.
- The "approximate" nature of the null space relies on the tail singular values being sufficiently small, which may not hold for certain layers or models.
- Experiments are limited to the LLaMA family; generalization to other architectures (e.g., Mistral, Qwen) remains unverified.
- Downstream evaluation covers only three task categories (Math, Code, Instruction Following); broader scenarios such as dialogue and summarization are absent.
- Compatibility with other PEFT methods (e.g., DoRA, AdaLoRA) is not discussed.

## Rating
- Novelty: ⭐⭐⭐⭐ — The activation null space perspective is insightful and well-supported by theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two models, three tasks, and ablations over rank and calibration size; broader scenarios would further strengthen the work.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly developed, theory and experiments are tightly integrated, and the figures are well-designed.
- Value: ⭐⭐⭐⭐ — Directly informative for research on LoRA knowledge preservation; the method is simple, effective, and easy to reproduce.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](../../ICCV2025/model_compression/efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)
- [\[ICML 2025\] Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics](../../ICML2025/model_compression/beyond_zero_initialization_investigating_the_impact_of_non-zero_initialization_o.md)
- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[ICML 2026\] LoRA-DA: Data-Aware Initialization for Low-Rank Adaptation via Asymptotic Analysis](../../ICML2026/model_compression/lora-da_data-aware_initialization_for_low-rank_adaptation_via_asymptotic_analysi.md)

</div>

<!-- RELATED:END -->
