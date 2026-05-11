---
title: >-
  [Paper Note] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners
description: >-
  [NeurIPS 2025][Reinforcement Learning][Multilingual Reasoning] Inspired by cognitive neuroscience (the relative independence of reasoning and language processing in the human brain)…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Multilingual Reasoning"
  - "Language-Reasoning Disentanglement"
  - "Causal Intervention"
  - "Activation Space"
  - "Training-Free"
date: 2026-05-08
content_hash: 84dad2402a268f5f
---

# When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners

**Conference**: NeurIPS 2025
**arXiv**: [2505.15257](https://arxiv.org/abs/2505.15257)
**Code**: [GitHub](https://github.com/MuyuenLP/Language-Reasoning-Disentangle)
**Area**: Reinforcement Learning
**Keywords**: Multilingual Reasoning, Language-Reasoning Disentanglement, Causal Intervention, Activation Space, Training-Free

## TL;DR

Inspired by cognitive neuroscience (the relative independence of reasoning and language processing in the human brain), this work identifies and removes language-specific components in the activation space of LLMs to disentangle language from reasoning, achieving consistent improvements in multilingual reasoning performance without any training.

## Background & Motivation

Reasoning-oriented LLMs (e.g., DeepSeek-R1, QwQ) perform strongly on high-resource languages such as English and Chinese, but exhibit substantially degraded reasoning capabilities on low- and mid-resource languages. This **multilingual reasoning gap** limits the global applicability of LLMs and exacerbates cross-lingual inequality in AI capabilities.

The central question is: what are the underlying causes of this multilingual reasoning performance gap?

The paper's hypothesis draws from findings in **cognitive neuroscience** — brain regions responsible for language comprehension and production are **largely inactive** during reasoning tasks, as human language evolved for communication rather than reasoning. Inspired by this, the authors hypothesize that reasoning capabilities and language processing in LLMs are similarly **separable**. If language-specific representational components interfere with the reasoning process, **removing** them should improve non-English reasoning performance — enabling reasoning capabilities to "transfer freely" from high-resource to low-resource languages.

The elegance of this hypothesis lies in its implication: if valid, it suggests that expensive multilingual post-training is unnecessary, and that cross-lingual reasoning can be enhanced through simple representational intervention at inference time.

## Method

### Overall Architecture

The method consists of three steps: (1) identifying the language-specific subspace; (2) eliminating language-specific components via projection during inference; and (3) applying elimination only at intermediate layers while preserving language signals at higher layers to maintain output language fidelity.

### Key Designs

1. **Language-Specific Subspace Identification**: For a model processing inputs in $L$ languages, the mean representation at each layer is computed as $\boldsymbol{m}_l = \frac{1}{n}\sum_{i=1}^{n}\boldsymbol{e}_l^i$ (using the final token embedding), and these are concatenated into a matrix $\boldsymbol{M} \in \mathbb{R}^{d \times L}$. Orthogonal decomposition splits $\boldsymbol{M}$ into a **language-agnostic subspace** $\boldsymbol{M}_a$ (shared cross-lingual semantics) and a **language-specific subspace** $\boldsymbol{M}_s$ (encoding linguistic variation). The decomposition is solved efficiently via SVD, with the objective:

$$\min_{\boldsymbol{M}_a, \boldsymbol{M}_s, \boldsymbol{\Gamma}} \|\boldsymbol{M} - \boldsymbol{M}_a \mathbbm{1}^\top - \boldsymbol{M}_s \boldsymbol{\Gamma}^\top\|_F^2 \quad \text{s.t.} \quad \text{Span}(\boldsymbol{M}_a) \perp \text{Span}(\boldsymbol{M}_s)$$

2. **Activation Ablation**: At inference time, each hidden representation $\boldsymbol{h}$ is projected to remove its component along the language-specific subspace $\boldsymbol{M}_s$:

$$\hat{\boldsymbol{h}} = \boldsymbol{h} - \lambda \boldsymbol{M}_s^\top \boldsymbol{M}_s \boldsymbol{h}$$

where $\lambda$ controls the ablation strength. This removes language-idiosyncratic variation, allowing the residual representation $\hat{\boldsymbol{h}}$ to better reflect language-agnostic reasoning processes. The **Design Motivation** follows directly from hypothesis validation — if language components do interfere with reasoning, projection-based removal should yield performance gains.

3. **Layer-Wise Intervention Strategy**: Empirical results show that applying intervention across all layers degrades output language fidelity (the model tends to respond in English). The key insight is that **language-specific signals in higher layers are critical for maintaining target-language output**. Accordingly, the method adopts a strategy of "elimination at intermediate layers, preservation at higher layers" — removing language-specific components in lower-to-middle layers to enable cross-lingual transfer of reasoning capabilities, while reintroducing them at higher layers to preserve output language consistency.

### Validation Experiments

Two experiments confirm that the removed components indeed correspond to language signals: (1) PCA visualizations show that non-English representations converge toward English clusters after ablation; and (2) language fidelity metrics demonstrate that stronger ablation increasingly causes the model to output English regardless of the input language.

## Key Experimental Results

### Main Results

Average accuracy across 11 languages on MGSM (mathematical reasoning):

| Model | Original AVG | +Disentangle AVG | Gain | Notes |
|-------|-------------|-----------------|------|-------|
| Qwen-2.5-3B-Instruct | 56.36 | **58.51** | +2.15 | Small non-reasoning model |
| Qwen-2.5-7B-Instruct | 69.82 | **70.76** | +0.94 | General model |
| Qwen-3-8B-Thinking | 84.15 | **85.42** | +1.27 | Effective on reasoning models too |
| R1-Distill-Qwen-14B | 65.24 | **67.24** | +2.00 | Distilled reasoning model |
| QwQ-32B | 83.31 | **84.62** | +1.31 | 32B reasoning model also benefits |

Improvements are more pronounced on XWinograd (commonsense reasoning) and M-MMLU (knowledge-intensive QA).

### Detailed Cross-Benchmark Comparison (XWinograd / M-MMLU)

| Model | XWinograd Original | XWinograd +Disentangle | M-MMLU Original | M-MMLU +Disentangle |
|-------|--------------------|------------------------|-----------------|----------------------|
| Qwen-2.5-3B | 65.07 | **70.18** (+5.11) | 52.62 | **56.63** (+4.01) |
| Qwen-2.5-7B | 68.13 | **74.00** (+5.87) | 61.25 | **63.88** (+2.63) |
| Qwen-3-8B-Think | 85.14 | **87.99** (+2.85) | 73.94 | **76.19** (+2.25) |

### Key Findings

- **Consistent Gains**: Improvements are observed across all 10 tested models (reasoning and general-purpose), 3 benchmarks, and 11 languages, demonstrating the universality of the hypothesis.
- **Low-Resource Languages Benefit More**: Swahili, for instance, gains over 10 percentage points on multiple models, with performance more than doubling in some cases.
- **Training-Free vs. Post-Training**: The training-free intervention is competitive with or superior to multilingual post-training approaches (SFT/RL) at negligible computational cost.
- **Language Interference Strength vs. Reasoning Accuracy**: Analysis reveals that stronger language-specific signals in hidden states correlate with lower reasoning accuracy, directly supporting the causal mechanism of "language signals interfering with reasoning."

## Highlights & Insights

- **Interdisciplinary Inspiration**: The hypothesis transfer from cognitive neuroscience (language-reasoning independence) to AI is both elegant and compelling.
- **Training-Free Method**: No fine-tuning or additional data is required; inference-time matrix projection incurs virtually no deployment overhead.
- The "Less is More" philosophy — removing language signals to enhance reasoning — suggests that the entanglement of language and reasoning representations in current LLMs may be a fundamental bottleneck for multilingual performance.
- The layer-wise intervention finding (elimination at intermediate layers, preservation at higher layers) provides empirical evidence for understanding the functional specialization of Transformer layers.

## Limitations & Future Work

- Although both Chinese and English are high-resource languages, Chinese representations also converge toward English clusters — the authors speculate this relates to the proportion of English in pretraining data, but the precise mechanism remains unclear.
- The ablation strength $\lambda$ and the range of intervened layers currently require per-model tuning.
- The method assumes linear separability of language and reasoning (via SVD), which may be insufficient for more complex non-linear entanglement.
- Gains are limited for extremely low-resource languages with very little training data (e.g., Swahili shows marginal improvements on some models).

## Related Work & Insights

- Theoretical connections exist with multilingual representation alignment research (language-neutral representations, cross-lingual transfer).
- The work is related to representation engineering and activation steering, though with a different objective (enhancing reasoning rather than controlling behavior).
- The method is complementary to multilingual post-training approaches (SFT/RL) and can be used in conjunction with them.
- This work inspires the broader question of whether LLMs contain additional disentanglable "functional subspaces," such as knowledge vs. creativity, or facts vs. style.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Unique hypothesis origin (cognitive neuroscience), elegant validation via causal intervention
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 10 models × 11 languages × 3 benchmarks, large-scale and consistent conclusions
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear narrative arc, progressing systematically from hypothesis to validation to analysis
- **Value**: ⭐⭐⭐⭐⭐ — Training-free multilingual reasoning improvement with both theoretical insights and practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[NeurIPS 2025\] RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](rl_tango_reinforcing_generator_and_verifier_together_for_lan.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[NeurIPS 2025\] Horizon Reduction Makes RL Scalable](horizon_reduction_makes_rl_scalable.md)

</div>

<!-- RELATED:END -->
