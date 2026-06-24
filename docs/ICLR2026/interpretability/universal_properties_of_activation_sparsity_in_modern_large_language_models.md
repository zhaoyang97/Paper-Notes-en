---
title: >-
  [Paper Note] Universal Properties of Activation Sparsity in Modern Large Language Models
description: >-
  [ICLR2026][Interpretability][activation sparsity] This paper provides a systematic study of activation sparsity in modern LLMs (GLU architecture + SiLU/GELU). It proposes a universal top-p sparsification framework and a "critical sparsity" metric, finding that activation sparsity increases monotonically with model scale. Input sparsification is identified as the most practical training-free acceleration scheme, and the authors demonstrate for the first time that diffusion-bas…
tags:
  - "ICLR2026"
  - "Interpretability"
  - "activation sparsity"
  - "LLM acceleration"
  - "GLU architecture"
  - "critical sparsity"
  - "top-p sparsification"
  - "diffusion LLM"
date: 2026-05-08
content_hash: f9a02553b43fa5f7
---

# Universal Properties of Activation Sparsity in Modern Large Language Models

**Conference**: ICLR2026  
**arXiv**: [2509.00454](https://arxiv.org/abs/2509.00454)  
**Code**: [GitHub](https://github.com/fszatkowski/activation-sparsity-benchmarking)  
**Area**: Interpretability  
**Keywords**: activation sparsity, LLM acceleration, GLU architecture, critical sparsity, top-p sparsification, diffusion LLM

## TL;DR
This paper provides a systematic study of activation sparsity in modern LLMs (GLU architecture + SiLU/GELU). It proposes a universal top-p sparsification framework and a "critical sparsity" metric, finding that activation sparsity increases monotonically with model scale. Input sparsification is identified as the most practical training-free acceleration scheme, and the authors demonstrate for the first time that diffusion-based LLMs also exhibit significant activation sparsity.

## Background & Motivation

**History of Activation Sparsity**: ReLU networks naturally produce exact zero activations. Extensive work has utilized this property for efficiency optimization, robustness enhancement, and interpretability analysis.

**Modern LLM Challenges**: Mainstream LLMs (Gemma3, LLaMA3, Qwen2.5) use GLU architectures with SiLU/GELU activations, which do not produce strict zeros. Consequently, methods from the ReLU era cannot be directly migrated.

**Fragmentation of Existing Solutions**:
   - **Modification Schemes**: Replacing SiLU with ReLU requires additional training and may degrade model quality.
   - **Approximate Sparsity Schemes**: These lack the principled guarantees of ReLU's strict zeros and require threshold calibration, risking overfitting to calibration sets.
   - Design choices for FFN input, gating, or intermediate activations lack unified guidance.

**Goal**: To establish a universal, simple, and training-free framework to systematically study and exploit activation sparsity in modern LLMs.

## Method

### Overall Architecture

The paper does not propose a new sparsification algorithm but establishes a unified, training-free measurement toolkit. It uses an architecture-agnostic top-p rule to sparsify any activation vector and employs "critical sparsity"—a metric linked to actual performance—to compare how much sparsity different models, FFN modules, and generation paradigms can tolerate. All conclusions are built upon the GLU architecture $$\mathcal{FFN}(x) = W_d\big((W_u x) \odot \sigma(W_g x)\big)$$, which is the shared structure of mainstream LLMs like Gemma3, LLaMA3, and Qwen2.5.

### Key Designs

**1. Top-p Sparsification Rule: Replacing Strict Zeros with Energy Ratios**

Since SiLU/GELU do not produce exact zeros like ReLU, traditional zero-based sparsification fails. Top-p defines sparsity from an energy perspective: for any activation vector $v \in \mathbb{R}^n$, it retains only the items with the largest absolute values such that their L1 energy ratio reaches $p$. Formally, $\text{top-p}(v) = m_p \odot v$, where the mask $m_p = \arg\min_m \|m\|_0$ is the sparsest solution under the constraint $\|m \odot v\|_1 \geq p \cdot \|v\|_1$ and $m \in \{0,1\}^n$. This induces a sparsity $S_p(v) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}(m_p^{(i)} = 0)$. This rule is effective because it makes no architectural assumptions and requires no training or calibration data, avoiding overfitting. Empirical evidence also shows it is more interpretable and degrades more smoothly with model scale than top-k.

**2. Critical Sparsity: Anchoring "How Much to Sparsify" to Performance Constraints**

Sparsity at a fixed $p$ cannot determine how much computation a model can safely skip. This paper defines "critical sparsity" as the maximum sparsity a model can reach while maintaining $\geq 99\%$ of its original performance. By binding the metric to performance, it allows direct comparison of redundancy across different model families and scales. Core findings, such as "sparsity increases monotonically with scale," are measured using this indicator.

**3. Four Activation Types and Three Acceleration Paths: Defining Action and Cost**

Within a GLU FFN, four types of activations can be sparsified, each with different benefits and costs:

| Activation Type | Definition | Description |
|----------|------|------|
| Input $x$ | FFN input vector | Sparsification accelerates all three linear layers simultaneously |
| Up-projection $u$ | $W_u x$ | Linear projection without an activation function |
| Gating $g$ | $\sigma(W_g x)$ | Gating signal after the activation function |
| Intermediate $i$ | $(W_u x) \odot \sigma(W_g x)$ | Intermediate representation after element-wise product |

These correspond to three mutually exclusive acceleration paths. The paper systematically compares their trade-offs: **Input sparsification** acts on $x$ and accelerates three linear layers without a predictor, though $x$ itself lacks natural sparsity. **Gating sparsification** leverages the compression in $g$ post-activation, but calculating $g$ accounts for ~1/3 of FFN costs. **Predictor methods** directly predict the mask for $i$, offering high theoretical acceleration but requiring additional training and risking approximation errors. This mapping supports the argument that input sparsification is the most practical.

## Key Experimental Results

### Model Scale and Critical Sparsity (Gemma3 Series)

| Model | Parameters | Intermediate Sparsity | Input Sparsity | Gating Sparsity |
|------|--------|---------------|-----------|-----------|
| Gemma3-1B | 1B | ~50% | ~35% | ~35% |
| Gemma3-4B | 4B | ~55% | ~40% | ~40% |
| Gemma3-12B | 12B | ~62% | ~48% | ~48% |
| Gemma3-27B | 27B | ~70% | ~55% | ~55% |

**Core Finding**: Critical sparsity increases monotonically with model scale—larger models have more redundant neurons that can be safely skipped.

### Effective Rank Analysis

Effective rank consistently decreases as model scale increases, indicating that activations in larger models are more low-rank and redundant. However, while the effective rank of gating activations is similar to intermediate ones, their empirical sparsity tolerance is lower, suggesting effective rank does not fully characterize sparsification robustness.

### Trends Across Model Families

| Model Family | Scale Range | Critical Sparsity Trend |
|---------|---------|---------------|
| Gemma3 | 1B–27B | Most prominent linear growth |
| LLaMA3.1/3.2 | 1B–70B | Consistent growth; uniform width/depth scaling |
| Qwen2.5 | 0.5B–72B | Overall growth but more volatile; non-uniform dimension growth |

### Impact of Training Method

| Model Variant | Change in Critical Sparsity |
|---------|--------------|
| Pre-trained → Instruction Tuned | IT models show higher sparsity at larger scales |
| Qwen3-4B Instruct vs Thinking | Thinking models are more robust on GSM8K but degrade faster on MMLU |

### First Analysis of Diffusion LLM (LLaDA-8B)

| Task | Intermediate Critical Sparsity | All-Inputs Critical Sparsity |
|------|------------------|---------------------|
| MMLU | 69.46% | 62.72% |
| HumanEval | 81.25% | 77.89% |
| HellaSwag | 71.21% | 67.92% |
| MBPP | 66.67% | 59.18% |
| **Average** | **68.13%** | **56.79%** |

LLaDA-8B's critical sparsity is significantly higher than the autoregressive LLaMA3.1-8B of the same scale. The denoising nature of diffusion models makes them more robust to noise introduced by sparsification.

### Temporal Stability within Diffusion Steps

- Jaccard similarity between consecutive diffusion steps is stable but not high (~0.6–0.7).
- Drift similarity relative to the initial step drops rapidly—sparsity patterns change gradually during denoising.
- Conclusion: Sparse masks in diffusion LLMs **cannot be reused across steps** (unlike autoregressive models where masks can be reused after the prompt).

### MoE Model Analysis (Qwen3-30B-A3B)

While average critical sparsity within a layer is stable, sparsity in individual experts varies significantly. Outliers among the 128 experts show higher sparsity than dense models of equivalent size, indicating that MoE experts also exhibit widespread activation sparsity.

## Highlights & Insights
- **"Functional sparsity is a universal property of LLMs"**: This holds true across architectures (GLU/MoE), training methods (PT/IT/Thinking), and generation paradigms (Autoregressive/Diffusion).
- **Input Sparsification is the Most Practical**: It accelerates all FFN modules without predictors or gating calculations. Within the studied scales, gating offers no advantage.
- **Risk of Calibration**: Critical sparsity varies greatly across tasks. Threshold methods based on calibration sets risk overfitting; researchers should pursue truly data-free sparsification.
- **Potential of Diffusion LLMs**: First empirical evidence shows that diffusion-based LLMs have higher activation sparsity than autoregressive models, though methods must be designed specifically for diffusion properties.

## Limitations & Future Work
- **FFN Focus**: The study does not analyze activation sparsity in multi-head attention, though FFNs dominate computation outside of long-context scenarios.
- **Limited Acceleration Ceiling**: Activation sparsity provides ~1.3–1.5x acceleration, which is lower than speculative decoding (~4x). It should be viewed as a complementary technology.
- **Top-p as a Lower Bound**: More complex layer-specific or module-specific methods might achieve higher sparsity.
- **Lack of Implementation**: The paper focuses on characterization rather than deployment optimization.

## Related Work & Insights
- **vs. Mirzadeh et al. (2024)**: Previous ReLU modification schemes required extra training; this work proves training-free top-p is already practical.
- **vs. Liu et al. (2025a/b)**: The empirical premise of input sparsification acceleration is systematically verified here.
- **vs. Song et al. (2024a) / Lee et al. (2024)**: Within the studied scales, gating sparsification is **not superior** to input sparsification—a key practical guideline.
- **Insight**: As models continue to grow, activation sparsity grows with them. Frontier models may naturally possess over 70% exploitable sparsity (Gemma3 series has already begun integrating sparsity-aware layers).

## Rating
- Novelty: ⭐⭐⭐⭐ Unified framework + critical sparsity definition + first diffusion LLM sparsity analysis, though the core method (top-p) is simple.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers Gemma3/LLaMA3/Qwen2.5 scales + PT/IT/Thinking + MoE + Diffusion models across 9 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, informative charts, and explicit conclusions.
- Value: ⭐⭐⭐⭐ Provides a comprehensive foundation for LLM activation sparsity acceleration with strong practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spilled Energy in Large Language Models](spilled_energy_in_large_language_models.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ICLR 2026\] To Sink or Not to Sink: Visual Information Pathways in Large Vision-Language Models](to_sink_or_not_to_sink_visual_information_pathways_in_large_vision-language_mode.md)
- [\[ICLR 2026\] Medical Interpretability and Knowledge Maps of Large Language Models](medical_interpretability_and_knowledge_maps_of_large_language_models.md)
- [\[ACL 2026\] Model Internal Sleuthing: Finding Lexical Identity and Inflectional Features in Modern Language Models](../../ACL2026/interpretability/model_internal_sleuthing_finding_lexical_identity_and_inflectional_features_in_m.md)

</div>

<!-- RELATED:END -->
