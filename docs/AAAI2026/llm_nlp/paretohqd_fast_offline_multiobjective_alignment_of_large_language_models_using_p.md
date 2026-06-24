---
title: >-
  [Paper Note] ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models Using Pareto High-Quality Data
description: >-
  [AAAI 2026][LLM (Other)][multi-objective alignment] This paper proposes ParetoHqD, which represents human preferences as preference directions in objective space (rather than linear scalarization), and performs two-stage SFT on high-quality data selected near the Pareto front. Using only 42% of the GPU time, it achieves multi-objective LLM alignment performance superior to five baselines.
tags:
  - "AAAI 2026"
  - "LLM (Other)"
  - "multi-objective alignment"
  - "Pareto optimality"
  - "SFT"
  - "preference direction"
  - "data selection"
date: 2026-05-08
content_hash: 2362c3193dc6f07e
---

# ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models Using Pareto High-Quality Data

**Conference**: AAAI 2026
**arXiv**: [2504.16628](https://arxiv.org/abs/2504.16628)  
**Code**: None  
**Area**: LLM/NLP
**Keywords**: multi-objective alignment, Pareto optimality, SFT, preference direction, data selection

## TL;DR

This paper proposes ParetoHqD, which represents human preferences as preference directions in objective space (rather than linear scalarization), and performs two-stage SFT on high-quality data selected near the Pareto front. Using only 42% of the GPU time, it achieves multi-objective LLM alignment performance superior to five baselines.

## Background & Motivation

LLM alignment typically requires satisfying multiple objectives simultaneously (e.g., helpfulness, harmlessness, humor), which often conflict with one another. User preferences over these objectives also vary—some prioritize safety while others prioritize usefulness.

**Three problems with existing offline multi-objective alignment**:

**Linear scalarization misrepresents preferences**: Mainstream methods use $\omega \cdot r$ to convert multi-objective problems into a single objective, but this cannot distinguish two data points lying on the same iso-value contour with entirely different preference trade-offs, and cannot handle non-convex Pareto fronts.

**Data imbalance**: Data with high scores across all objectives is extremely scarce; training on the full dataset biases the model toward mediocre score combinations.

**Low training efficiency**: Training on the full dataset is time-consuming, yet alignment can in principle be achieved with a small amount of high-quality data (the LIMA hypothesis).

**Mechanism**: Since alignment requires only a small amount of data, the key is selecting the right data. Preferences are represented as directions (geometric rays) in objective space, and data nearest to the Pareto front and closest to the specified direction are selected for SFT.

## Method

### Overall Architecture

ParetoHqD consists of two stages. Stage 1 selects high-quality data from the Pareto front of the original dataset that matches the preference direction and performs SFT. Stage 2 uses the Stage 1 model to generate new data, then selects data from the Pareto front of the newly generated data for a second round of SFT (data augmentation to prevent overfitting).

### Key Designs

1. **Preference direction representation (replacing linear scalarization)**

    - In objective space, preference $\omega$ is defined as a ray originating from the ideal point $r^{\max}$ and pointing toward the compromise point $W$
    - Compromise point: $W = r^{\min} + \omega \odot (r^{\max} - r^{\min})$
    - Advantage: Data along the same direction maintains a fixed ratio of objective values, faithfully reflecting the user's trade-off intent; also capable of covering non-convex Pareto fronts

2. **Pareto high-quality data selection**

    - $M$ reward models are used to score the entire dataset; data from the top few Pareto layers are extracted to form $\mathcal{D}^{\text{Pareto}}$
    - For each preference $\omega_i$, the $k=100$ data points in $\mathcal{D}^{\text{Pareto}}$ closest to preference direction $\mathcal{P}_i$ are selected for SFT
    - Only a minimal number of high-quality samples (100) are needed per preference, with training completing in 200 steps

3. **Two-stage training with data augmentation**

    - Stage 1: original dataset → Pareto high-quality data → SFT
    - Stage 2: $M{+}1$ representative Stage 1 models (each emphasizing one objective plus a balanced preference) generate new responses for 10,000 random prompts; Pareto high-quality data are then extracted from these responses
    - Stage 2 fine-tunes each preference using $k/2=50$ samples to mitigate overfitting caused by the small data volume in Stage 1

### Loss & Training

Both stages use the standard SFT loss $\mathcal{L}_{\text{SFT}} = -\mathbb{E}[\sum_i \log \pi(y_i | x, y_{<i})]$. Each preference is trained for 200 steps with batch size 8. The base model is Llama-2 7B.

## Key Experimental Results

### Main Results

| Method | Hypervolume ↑ | Collapse Rate CR% ↓ | GPU Time (h) ↓ |
|--------|--------------|---------------------|----------------|
| MORLHF (5 preferences) | 0.3777 | 35.39% | 2272.84 |
| Rewarded Soups | 0.3605 | 31.29% | 923.68 |
| RiC (SOTA) | ~second best | ~moderate | ~moderate |
| **ParetoHqD** | **0.7526** | **7.03%** | **55.87** |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| Stage 1 only | Effective but exhibits overfitting |
| Stage 1 + Stage 2 | Overfitting mitigated; Pareto front quality improved |
| Linear scalarization replacing preference direction | Diversity drops sharply; non-convex front not covered |
| Full dataset replacing Pareto high-quality data | Training time increases substantially; performance degrades |

### Key Findings

- **All baseline methods exhibit language collapse** (repetitive phrases or extremely short responses), with collapse rates above 30%; ParetoHqD achieves only 7.03%—attributable to avoiding the conflicting learning patterns induced by linear scalarization
- **A single preference can be aligned using only 100 data points and 200 training steps**, validating the LIMA hypothesis in the multi-objective setting
- **Scalable to three objectives**: strong performance is maintained on the helpfulness + harmlessness + humor tri-objective setting
- GPU time is only 2.5% of MORLHF and approximately 42% of RiC

## Highlights & Insights

- **Preference direction as a replacement for linear scalarization**: Representing preferences as geometric rays in objective space fundamentally resolves two inherent problems of linear scalarization—the inability to distinguish different trade-off points on the same iso-value contour, and the inability to cover non-convex Pareto fronts
- **"Small but precise" data selection strategy**: A multi-objective extension of the LIMA hypothesis—alignment does not require large amounts of data; selecting the right data is what matters. Data near the Pareto front is inherently high-quality
- **Extremely low training cost**: 55 GPU hours to complete dual-objective alignment across 11 preferences, making personalized multi-objective alignment practically deployable

## Limitations & Future Work

- Validation is limited to Llama-2 7B; performance on larger models remains unknown
- Two-stage SFT does not support fine-grained preference optimization that methods such as DPO or RLHF may provide
- The construction of preference directions depends on the accuracy of reward models; inaccurate reward models directly degrade data selection quality
- Pareto front data becomes sparser with more than three objectives; scalability analysis is insufficient

## Related Work & Insights

- **vs. RiC (Yang et al. 2024b)**: RiC embeds reward scores into prompts for conditional generation training, still relying on linear scalarization; ParetoHqD selects data via preference directions from a geometric perspective, yielding greater diversity
- **vs. MODPO (Zhou et al. 2024)**: MODPO integrates preference values into the DPO loss for preference-aware training, but still requires large amounts of data and is constrained by linear scalarization
- **vs. MORLHF (Li et al. 2021)**: MORLHF applies online RL to optimize linearly scalarized rewards, incurring extremely high computational costs (2,272 GPU hours vs. 56 hours)

## Rating

- Novelty: ⭐⭐⭐⭐ Replacing linear scalarization with preference direction representation is a theoretically grounded innovation
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers dual- and tri-objective settings, five baseline comparisons, collapse analysis, and efficiency comparisons
- Writing Quality: ⭐⭐⭐⭐ Figures are clear (Fig. 2 intuitively illustrates the problem with linear scalarization)
- Value: ⭐⭐⭐⭐ Provides an efficient and theoretically sound approach to personalized LLM alignment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](../../ACL2025/llm_nlp/gapo_multi_objective_alignment.md)
- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](profuser_progressive_fusion_of_large_language_models.md)
- [\[ACL 2025\] DeAL: Decoding-time Alignment for Large Language Models](../../ACL2025/llm_nlp/deal_decoding_time_alignment.md)
- [\[ACL 2025\] Diversity-oriented Data Augmentation with Large Language Models](../../ACL2025/llm_nlp/diversity_data_augmentation.md)
- [\[ACL 2025\] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment](../../ACL2025/llm_nlp/from_neurons_to_semantics_evaluating_cross-linguistic_alignment_capabilities_of_.md)

</div>

<!-- RELATED:END -->
