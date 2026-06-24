---
title: >-
  [Paper Note] Understanding and Mitigating Memorization in Diffusion Models for Tabular Data
description: >-
  [ICML2025][Text Generation][Tabular data generation] This work presents the first systematic study of the memorization phenomenon in tabular diffusion models, finding that memorization intensifies as training epochs increase and is strongly correlated with dataset size. It proposes TabCutMix/TabCutMixPlus to mitigate memorization via feature-segment swapping while maintaining generation quality.
tags:
  - "ICML2025"
  - "Text Generation"
  - "Tabular data generation"
  - "Diffusion models"
  - "Memorization"
  - "Data augmentation"
  - "Privacy protection"
date: 2026-05-08
content_hash: 9a8ee44956d0df90
---

# Understanding and Mitigating Memorization in Diffusion Models for Tabular Data

**Conference**: ICML2025  
**arXiv**: [2412.11044](https://arxiv.org/abs/2412.11044)  
**Code**: [GitHub - TabCutMix](https://github.com/fangzy96/TabCutMix)  
**Area**: Image Generation  
**Keywords**: Tabular data generation, Diffusion models, Memorization, Data augmentation, Privacy protection

## TL;DR
This work presents the first systematic study of the memorization phenomenon in tabular diffusion models, finding that memorization intensifies as training epochs increase and is strongly correlated with dataset size. It proposes TabCutMix/TabCutMixPlus to mitigate memorization via feature-segment swapping while maintaining generation quality.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Diffusion models have achieved breakthroughs in tabular data generation (e.g., TabDDPM, TabSyn), but the issue of memorization has been severely neglected. Memorization refers to the model unintentionally copying training data, which leads to privacy leakage and degraded generalization capabilities.

### Key Challenge

**Key Challenge**: While memorization in image and text generation has been widely studied, the mixed-type features (numerical + categorical) of tabular data prevent existing detection and mitigation methods from being directly transferred.

### Three Key Questions

1. Do tabular diffusion models exhibit memorization?
2. What factors influence the level of memorization?
3. How can it be effectively mitigated?

## Method

### Memorization Detection Criteria
A "relative distance ratio" criterion is adopted: a generated sample $x$ is considered memorized when its distance to the nearest training sample is significantly smaller than its distance to the second-nearest training sample. A composite distance metric is designed for mixed-type data: normalized L2 distance for numerical features and indicator functions for categorical features.

### Empirical Findings
- Memorization increases with training epochs (reaching up to 80% on the Magic dataset).
- Memorization is more severe on smaller datasets (prominent at 0.1% downsampling).
- TabSyn converges faster, but its final memorization rate is comparable to TabDDPM.

### Theoretical Analysis
It is proved that under ideal conditions (perfect score function approximation + perfect SDE solving), generated samples in the latent space will exactly replicate training samples. In practice, due to model capacity limitations and the randomness of VAE decoding, memorization does not reach 100%, but the underlying trend remains.

### TabCutMix: Feature-Segment Swapping Augmentation
Randomly swap feature segments between samples of the same class:
- Generate a binary mask $M$ according to a Bernoulli distribution.
- New sample = $M * \text{sample A} + (1-M) * \text{sample B}$.
- Constraint: Only swap features between samples of the same class.

### TabCutMixPlus: Correlation-Aware Feature Augmentation
Incorporate feature clustering on top of TabCutMix:
- Numerical features: Pearson correlation coefficient.
- Categorical features: Cramer's V.
- Mixed pairs: Squared ETA coefficient.
- Group correlated features together and swap the entire group to avoid disrupting feature dependencies.

## Key Experimental Results

### Main Results: Memorization Rate and Generation Quality

| Method | Default Memorization Rate (%) | MLE (%) | alpha-Precision (%) | Shape Score (%) |
|------|-----------------|--------|-------------------|---------------|
| TabSyn | 22.4 | 79.5 | 92.3 | 94.1 |
| TabSyn+Mixup | 19.8 | 78.2 | 89.5 | 90.3 |
| TabSyn+SMOTE | 18.5 | 77.1 | 88.7 | 89.8 |
| TabSyn+TabCutMix | 12.3 | 79.1 | 91.8 | 93.5 |
| TabSyn+TabCutMixPlus | **10.1** | **79.3** | **92.1** | **93.8** |

### Impact of Dataset Scale (TabSyn on Magic)

| Dataset Scale | Memorization Rate (%) |
|-----------|----------|
| 0.1% | ~90% |
| 1% | ~60% |
| 10% | ~30% |
| 50% | ~12% |
| 100% | ~8% |

### Key Findings
1. TabCutMixPlus consistently reduces the memorization rate across all datasets and models.
2. Generation quality (MLE/Precision/Shape Score) is virtually unaffected.
3. Feature clustering effectively avoids the generation of out-of-distribution (OOD) samples.
4. The proposed method is effective for both TabDDPM and TabSyn.

## Highlights & Insights

1. This work presents the first systematic study of memorization in tabular diffusion models, filling an important gap in the literature.
2. The composite distance metric cleverly handles mixed numerical and categorical features.
3. The theoretical analysis aligns with empirical observations, explaining the root cause of memorization.
4. The feature clustering design of TabCutMixPlus elegantly balances memorization mitigation and feature consistency.
5. The method is simple, effective, plug-and-play, and does not alter the underlying model architecture.

## Limitations & Future Work

1. Only two diffusion models (TabSyn and TabDDPM) were tested; more architectures remain to be validated.
2. Feature clustering may exhibit lower efficiency on extremely high-dimensional data.
3. The distance threshold (1/3) originates from empirical findings in the image domain; the optimal threshold for tabular data might differ.
4. Integration with differential privacy methods was not discussed.
5. Adaptation to regression tasks remains unexplored.

## Related Work & Insights

- Memorization mitigation methods in the image domain (such as Concept Ablation or AMG) are not directly applicable to tabular data.
- The concept of CutMix image augmentation is cleverly transferred to the tabular domain.
- Insight: TabCutMixPlus can be extended to federated tabular data generation scenarios.

## Rating
- Novelty: 4.5/5 — First systematic study + theoretical analysis + practical method
- Experimental Thoroughness: 4.5/5 — Comprehensive evaluation across multiple datasets and models
- Writing Quality: 4.0/5
- Value: 5.0/5 — Fills an important gap with a highly practical approach

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs](../../ICLR2026/nlp_generation/rainbow_padding_mitigating_early_termination_in_instruction-tuned_diffusion_llms.md)
- [\[ICLR 2026\] Planner Aware Path Learning in Diffusion Language Models Training](../../ICLR2026/nlp_generation/planner_aware_path_learning_in_diffusion_language_models_training.md)
- [\[ACL 2025\] Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems](../../ACL2025/nlp_generation/dehumanizing_machines_anthropomorphic.md)
- [\[ICCV 2025\] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](../../ICCV2025/nlp_generation/beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)
- [\[ACL 2025\] Theme-Explanation Structure for Table Summarization Using Large Language Models](../../ACL2025/nlp_generation/theme-explanation_structure_for_table_summarization_using_large_language_models_.md)

</div>

<!-- RELATED:END -->
