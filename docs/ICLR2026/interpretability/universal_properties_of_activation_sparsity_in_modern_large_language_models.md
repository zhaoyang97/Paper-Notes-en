---
title: >-
  [Paper Note] Universal Properties of Activation Sparsity in Modern Large Language Models
description: >-
  [ICLR 2026][Interpretability][activation sparsity] This paper presents a systematic study of activation sparsity in modern LLMs (GLU architecture + SiLU/GELU)…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "activation sparsity"
  - "LLM acceleration"
  - "GLU architecture"
  - "critical sparsity"
  - "top-p sparsification"
  - "diffusion LLM"
date: 2026-05-08
content_hash: 56e14a7708beffcf
---

# Universal Properties of Activation Sparsity in Modern Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.00454](https://arxiv.org/abs/2509.00454)  
**Code**: [GitHub](https://github.com/fszatkowski/activation-sparsity-benchmarking)  
**Area**: Interpretability
**Keywords**: activation sparsity, LLM acceleration, GLU architecture, critical sparsity, top-p sparsification, diffusion LLM

## TL;DR
This paper presents a systematic study of activation sparsity in modern LLMs (GLU architecture + SiLU/GELU), proposes a universal top-p sparsification framework and a critical sparsity metric, demonstrates that activation sparsity increases monotonically with model scale, identifies input sparsification as the most practical training-free acceleration scheme, and provides the first empirical evidence that diffusion-based LLMs also exhibit significant activation sparsity.

## Background & Motivation

**Historical context of activation sparsity**: ReLU networks naturally produce exact zero activations, and a large body of work has exploited this property for efficiency optimization, robustness improvement, and interpretability analysis.

**The problem with modern LLMs**: Mainstream LLMs (Gemma3, LLaMA3, Qwen2.5) adopt GLU architectures with SiLU/GELU activations, which do not produce strictly zero values—methods developed for the ReLU era cannot be directly transferred.

**Fragmentation of existing approaches**:
   - **Replacement approaches** (substituting SiLU with ReLU) require additional training and may degrade model quality.
   - **Approximate sparsification approaches** lack the principled guarantees of ReLU's exact zeros, require threshold calibration, and may overfit to the calibration set.
   - Different methods target the input, gate, or intermediate activations of FFN layers without unified design guidance.

**Goal**: To establish a universal, simple, training-free framework for systematically studying and exploiting activation sparsity in modern LLMs.

## Method

### Top-p Sparsification Rule

For an arbitrary activation vector $v \in \mathbb{R}^n$, retain the entries with the largest absolute values such that their L1 norm accounts for a fraction $p$ of the total:

$$\text{top-p}(v) = m_p \odot v; \quad m_p = \arg\min_m \|m\|_0 \quad \text{s.t.} \quad \|m \odot v\|_1 \geq p \cdot \|v\|_1, \quad m \in \{0,1\}^n$$

The induced sparsity is: $S_p(v) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}(m_p^{(i)} = 0)$

**Advantages**:
- Applicable to any FFN module without architectural assumptions or additional training.
- No calibration overfitting—no auxiliary calibration dataset is required.
- Simple and interpretable, enabling fair comparison across models and modules.

### Critical Sparsity

Defined as the maximum sparsity level at which a model retains ≥99% of its original performance. This provides a quantitative metric anchored to practical performance constraints, enabling direct comparison of sparsification tolerance across different models and modules.

### Four Activation Vector Types in GLU FFN

For the GLU architecture $\mathcal{FFN}(x) = W_d((W_u x) \odot \sigma(W_g x))$, four activation types are defined:

| Activation Type | Definition | Description |
|-----------------|------------|-------------|
| Input $x$ | FFN input vector | Can accelerate all three linear layers |
| Up-projection $u$ | $W_u x$ | Linear projection without activation function |
| Gate $g$ | $\sigma(W_g x)$ | Gate signal after activation function |
| Intermediate $i$ | $(W_u x) \odot \sigma(W_g x)$ | Intermediate representation after element-wise product |

### Comparison of Three Acceleration Strategies

| Strategy | Target Activation | Advantages | Disadvantages |
|----------|-------------------|------------|---------------|
| Input sparsification | $x$ | No predictor needed; accelerates all FFN modules | No natural sparsity in inputs |
| Gate sparsification | $g$ | Activation function naturally compresses values | Computing the gate itself costs ~1/3 of FFN computation |
| Predictor-based | $i$ | Theoretically highest acceleration | Requires training a predictor; introduces approximation error |

## Key Experimental Results

### Model Scale and Critical Sparsity (Gemma3 Family)

| Model | Parameters | Intermediate Sparsity | Input Sparsity | Gate Sparsity |
|-------|------------|-----------------------|----------------|---------------|
| Gemma3-1B | 1B | ~50% | ~35% | ~35% |
| Gemma3-4B | 4B | ~55% | ~40% | ~40% |
| Gemma3-12B | 12B | ~62% | ~48% | ~48% |
| Gemma3-27B | 27B | ~70% | ~55% | ~55% |

**Core Finding**: Critical sparsity increases monotonically with model scale—larger models have more redundant neurons that can be safely skipped.

### Effective Rank Analysis

The effective rank of activations consistently decreases with model scale, indicating that larger models produce lower-rank, more redundant representations. However, the effective rank of gate activations is comparable to that of intermediate activations, even though gate activations tolerate sparsification less well empirically—demonstrating that effective rank alone is insufficient to fully characterize sparsification robustness.

### Cross-Family Trends

| Model Family | Scale Range | Critical Sparsity Trend |
|--------------|-------------|-------------------------|
| Gemma3 | 1B–27B | Most pronounced linear growth |
| LLaMA3.1/3.2 | 1B–70B | Consistent growth; relatively balanced width/depth scaling |
| Qwen2.5 | 0.5B–72B | Overall growth but more volatile; uneven dimension scaling |

### Effect of Training Paradigm

| Model Variant | Change in Critical Sparsity |
|---------------|-----------------------------|
| Pretraining → Instruction Tuning | IT models exhibit higher sparsity at larger scales |
| Qwen3-4B Instruct vs. Thinking | Reasoning models are more robust on GSM8K but degrade faster on MMLU |

### First Analysis of Diffusion LLMs (LLaDA-8B)

| Task | Intermediate Critical Sparsity | All-Inputs Critical Sparsity |
|------|-------------------------------|------------------------------|
| MMLU | 69.46% | 62.72% |
| HumanEval | 81.25% | 77.89% |
| HellaSwag | 71.21% | 67.92% |
| MBPP | 66.67% | 59.18% |
| **Average** | **68.13%** | **56.79%** |

LLaDA-8B achieves substantially higher critical sparsity than the comparably sized autoregressive LLaMA3.1-8B—the denoising nature of diffusion models renders them more robust to the noise introduced by sparsification.

### Temporal Stability Across Diffusion Steps

- Jaccard similarity between consecutive diffusion steps is stable but not high (~0.6–0.7).
- Drift similarity relative to the initial step decreases rapidly—sparse patterns evolve progressively during denoising.
- Conclusion: Sparse masks in diffusion LLMs **cannot be reused across steps** (unlike autoregressive models, where masks can be reused after the prompt phase).

### MoE Model Analysis (Qwen3-30B-A3B)

The average per-layer critical sparsity is stable, but individual experts exhibit sparsity far above the mean. Among 128 experts, outlier experts surpass the sparsity of comparably sized dense models—MoE experts universally display activation sparsity.

## Highlights & Insights
- **"Functional sparsity is a universal property of LLMs"**: This holds consistently across architectures (GLU/MoE), training paradigms (PT/IT/Thinking), and generation paradigms (autoregressive/diffusion).
- **Input sparsification is the most practical approach**: It requires no predictor and no gate computation, yet accelerates all FFN modules—gate sparsification offers no advantage at the scales studied.
- **Risks of calibration**: Critical sparsity varies substantially across tasks; threshold methods based on calibration datasets carry overfitting risk, motivating truly data-free sparsification approaches.
- **Potential of diffusion LLMs**: This work provides the first empirical evidence that diffusion LLMs exhibit higher activation sparsity than autoregressive models, though dedicated methods tailored to diffusion-specific characteristics are required.

## Limitations & Future Work
- **FFN-only scope**: Activation sparsity in multi-head attention is not analyzed, though FFN layers dominate computation outside long-context settings.
- **Limited acceleration ceiling**: Activation sparsity yields approximately 1.3–1.5× speedup, far below speculative decoding (~4×); it should be positioned as a complementary technique.
- **Top-p as a lower bound**: More sophisticated layer-wise or module-specific methods may achieve higher sparsity.
- **No concrete acceleration implementation**: The paper focuses on characterizing sparsity rather than deployment-level optimization.

## Related Work & Insights
- **vs. Mirzadeh et al. (2024)**: Prior ReLU replacement approaches require additional training; this paper demonstrates that training-free top-p sparsification already achieves practical performance levels.
- **vs. Liu et al. (2025a/b)**: The empirical premises underlying input sparsification acceleration methods are systematically validated in this work.
- **vs. Song et al. (2024a) / Lee et al. (2024)**: Gate sparsification does **not** outperform input sparsification at the scales studied—an important practical guideline.
- **Implication**: As models continue to scale, activation sparsity continues to grow; frontier models may naturally possess ≥70% exploitable sparsity (Gemma3n has already begun integrating sparsity-aware layers into its architecture).

## Rating
- Novelty: ⭐⭐⭐⭐ Unified framework + critical sparsity definition + first sparsity analysis of diffusion LLMs, though the core method (top-p) is simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers Gemma3/LLaMA3/Qwen2.5 at multiple scales + PT/IT/Thinking + MoE + diffusion models across 9 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, informative figures, and well-defined conclusions.
- Value: ⭐⭐⭐⭐ Provides a comprehensive foundational reference for LLM activation sparsity acceleration with strong practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training](zerotuning_unlocking_the_initial_tokens_power_to_enhance_large_language_models_w.md)
- [\[ACL 2026\] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](../../ACL2026/interpretability/model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/interpretability/compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](../../NeurIPS2025/interpretability/the_trilemma_of_truth_in_large_language_models.md)

</div>

<!-- RELATED:END -->
