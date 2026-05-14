---
title: >-
  [Paper Note] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data
description: >-
  [CVPR 2026][Medical Imaging][Data Selection] This paper proposes the Difficulty-Influence Quadrant (DIQ) data selection strategy, which jointly considers sample difficulty and gradient influence to enable VLM language ba…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Data Selection"
  - "Medical Reasoning"
  - "Large Language Models"
  - "SFT"
  - "Gradient Influence"
date: 2026-05-08
content_hash: 01a9d2b9644c8065
---

# Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data

**Conference**: CVPR 2026
**arXiv**: [2508.01450](https://arxiv.org/abs/2508.01450)
**Code**: [GitHub](https://github.com/mihara-bot/DIQ)
**Area**: Medical Imaging
**Keywords**: Data Selection, Medical Reasoning, Large Language Models, SFT, Gradient Influence

## TL;DR

This paper proposes the Difficulty-Influence Quadrant (DIQ) data selection strategy, which jointly considers sample difficulty and gradient influence to enable VLM language backbones to match full-data SFT performance using only 1% of curated data, and to surpass full-data training with just 10%.

## Background & Motivation

The standard approach for adapting LLMs to medical reasoning tasks is supervised fine-tuning (SFT); however, existing practices suffer from several issues:

- **Data Redundancy**: Large-scale datasets contain substantial low-quality or duplicated samples, incurring high computational costs with marginal performance gains.
- **Limitations of Single-Dimension Selection**:
  - Selecting by **difficulty** alone → retrieves overly noisy samples with weak gradient signals, leading to unstable training.
  - Selecting by **gradient influence** alone → favors easily optimizable samples with shallow reasoning chains.
- These two dimensions are in fundamental tension; neither alone is optimal.

Through pilot experiments on the FineMed dataset—training separately on four quadrants partitioned by difficulty and influence—the authors find that high-influence + low-difficulty ($\mathcal{Q}_2$) data yields better benchmark performance than low-influence + high-difficulty ($\mathcal{Q}_3$) data, yet produces lower reasoning quality. This confirms that achieving the best of both worlds requires samples that are simultaneously high-difficulty and high-influence.

## Method

### Overall Architecture

DIQ projects each training sample into a two-dimensional space: (1) a **difficulty score**—model-agnostic, predicted by a BiomedBERT classifier on a 5-point Likert scale; and (2) an **influence score (Dot)**—model-dependent, computed as the inner product between a training sample's gradient and the mean gradient of the validation set. Data are partitioned into four quadrants along these two dimensions, and samples are selected in priority order $\mathcal{Q}_1 \to \mathcal{Q}_2 \to \mathcal{Q}_3 \to \mathcal{Q}_4$ until the target retention ratio is reached.

### Key Designs

1. **Difficulty Estimation**: A BiomedBERT classifier fine-tuned on multiple medical QA datasets evaluates difficulty across three dimensions: Knowledge, Reasoning, and Overall. One dimension is chosen as the scalar score $D(z) \triangleq D_\phi(z)$, with a percentile threshold $\tau_d$ used to partition samples into high/low difficulty. This score is model-agnostic and can be computed once and reused.

2. **Dot-Product Influence**: The influence of a training sample $z$ is defined as the inner product of its gradient with the mean validation gradient:
$$\text{Dot}(z) \triangleq g(z; \hat{\boldsymbol{\theta}})^\top \bar{g}_{\text{val}}(\hat{\boldsymbol{\theta}})$$
This represents a first-order approximation of the one-step reduction in mean validation loss:
$$\Delta \bar{\ell}_{\text{val}} = -\eta \cdot \text{Dot}(z) + O(\eta^2)$$
In practice, Johnson–Lindenstrauss random projection reduces gradient dimensionality to 4096, yielding complexity $O(|\mathcal{D}_{\text{val}}| + |\mathcal{D}|)$ without requiring Hessian computation.

3. **Quadrant-Priority Selection**: Data are partitioned into four quadrants using difficulty threshold $\tau_d$ and influence median $m_{\text{dot}}$. $\mathcal{Q}_1$ (high difficulty + high influence) receives the highest priority, as it encompasses complex clinical reasoning and strong gradient signals. Within each quadrant, samples are ranked by Dot in descending order, with difficulty used as a tiebreaker.

### Loss & Training

- LoRA fine-tuning (rank=8, target modules QKV), learning rate $1 \times 10^{-4}$, cosine decay, 3 epochs.
- Maximum context length: 8192 tokens.
- Validation set: 20 randomly sampled examples per downstream task by default (180 total).
- DIQ scoring is a one-time upfront cost reusable across experiments.

## Key Experimental Results

### Main Results

Fine-tuning Llama3.1-8B-Instruct on the Huatuo dataset; averages across 9 benchmarks:

| Data Size | Method | AvgS ↑ | AvgC ↑ | AvgA ↑ |
|-----------|--------|--------|--------|--------|
| Full (19k) | — | 54.77 | 37.77 | 43.44 |
| 1% | Random | 51.31 | 33.47 | 39.42 |
| 1% | LESS | 54.97 | 33.32 | 40.54 |
| 1% | **DIQ** | **56.54** | **35.91** | **42.78** |
| 10% | Similarity | 54.13 | 35.53 | 41.73 |
| 10% | **DIQ** | **58.11** | **37.00** | **44.04** |

**1% DIQ nearly matches full-data SFT (42.78 vs. 43.44); 10% DIQ surpasses it (44.04 vs. 43.44).**

### Ablation Study

| Selection Strategy | AvgA (1%) | AvgA (10%) | Notes |
|--------------------|-----------|------------|-------|
| Influence only | ~41.89 | ~42.45 | Biased toward easy samples |
| Reasoning only | ~41.89 | ~43.16 | Strongest single-dimension baseline |
| Knowledge only | ~41.05 | ~42.36 | High single-dimension bias |
| **DIQ (Full)** | **42.78** | **44.04** | Two-dimensional complementarity optimal |

### Key Findings

- **Improved Clinical Reasoning Quality**: LLM-as-judge evaluation shows that data selected by DIQ-1% scores +0.80 higher on differential diagnosis (DDx), +0.35 on safety checks (SC), and +0.46 on evidence citation (EC) compared to the remaining data.
- **Efficiency**: DIQ's computational cost is only 1/1.85 of a single full-data SFT run on Llama3.1-8B, and scores are reusable.
- **Cross-Model Generalization**: Influence scores computed on Llama transfer effectively to the Qwen3 series, yielding gains in 6 of 9 settings.
- **Compatibility with DPO**: 1% DIQ + DPO outperforms full-data SFT + DPO by 1.00 AvgA.
- Validation sets of 360–450 samples suffice to stabilize influence rankings.

## Highlights & Insights

- **Strong Empirical Validation of "Less is More"**: The paper explicitly characterizes the tension between difficulty and influence in data selection and provides an actionable resolution.
- The DIQ framework is simple and efficient: gradients are computed in a single forward/backward pass without requiring Hessian computation; random projection preserves ranking order.
- A direct connection is established between data selection and clinical reasoning quality—DIQ improves not only benchmark scores but also reasoning alignment.
- The quadrant-based selection strategy is intuitive and interpretable.

## Limitations & Future Work

- Influence scores are computed once prior to training and do not account for dynamic changes during the training process.
- Experiments are limited to models with at most 32B parameters; performance on 70B+ models remains untested.
- The task composition and distribution of the validation set may affect the quality of Dot scores.
- Biases inherent in the difficulty classifier (BiomedBERT) may propagate to the selected data.
- Despite the title referencing VLMs, the experiments predominantly involve pure-text LLM fine-tuning.

## Related Work & Insights

- Compared to LESS (TracIn-based), DIQ is simpler and more efficient (no Hessian required) and additionally incorporates difficulty information.
- The principle of prioritizing the "high difficulty + high influence" quadrant is generalizable to data selection in other domains.
- Gradient inner products as a measure of sample value are concise and effective, with solid theoretical grounding via first-order Taylor expansion.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Difficulty and influence have been studied independently; the two-dimensional quadrant selection framework is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 9 benchmarks, 6 datasets, multiple models, comprehensive ablations, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic, rich experiments, and motivation supported by pilot studies.
- **Value**: ⭐⭐⭐⭐ — Highly practical for medical LLM fine-tuning, though the VLM component claimed in the title is insufficiently validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[CVPR 2026\] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)
- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](../../AAAI2026/medical_imaging/g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [\[ICLR 2026\] Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](../../ICLR2026/medical_imaging/fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)

</div>

<!-- RELATED:END -->
