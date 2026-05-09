---
title: >-
  [Paper Note] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation
description: >-
  [ICLR 2026][Social Computing][Test-Time Adaptation] This paper introduces Tsallis entropy (a generalization of Shannon entropy) into Test-Time Adaptation for vision-language models, and further develops Adaptive Debiasing Tsallis Entropy (ADTE), which customizes a per-class debiasing parameter $q^l$ to select more reliable high-confidence views than Shannon entropy without distribution-specific hyperparameter tuning. ADTE surpasses the state of the art on ImageNet and its 5 variants as well as 10 cross-domain benchmarks.
tags:
  - ICLR 2026
  - Social Computing
  - Test-Time Adaptation
  - Tsallis entropy
  - CLIP
  - debiasing
  - uncertainty estimation
date: 2026-05-08
content_hash: ef12944f26118624
---

# Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation

**Conference**: ICLR 2026
**arXiv**: [2602.11743](https://arxiv.org/abs/2602.11743)
**Code**: [https://github.com/Jinx630/ADTE](https://github.com/Jinx630/ADTE)
**Area**: Social Computing
**Keywords**: Test-Time Adaptation, Tsallis entropy, CLIP, debiasing, uncertainty estimation

## TL;DR
This paper introduces Tsallis entropy (a generalization of Shannon entropy) into Test-Time Adaptation for vision-language models, and further develops Adaptive Debiasing Tsallis Entropy (ADTE), which customizes a per-class debiasing parameter $q^l$ to select more reliable high-confidence views than Shannon entropy without distribution-specific hyperparameter tuning. ADTE surpasses the state of the art on ImageNet and its 5 variants as well as 10 cross-domain benchmarks.

## Background & Motivation

**Background**: TTA methods improve the out-of-distribution performance of VLMs such as CLIP by selecting high-confidence augmented views. Representative methods including TPT and Zero employ Shannon entropy to measure uncertainty and filter low-entropy views.

**Limitations of Prior Work**: CLIP is pre-trained on imbalanced web-crawled data, leading to overconfidence on head classes and underconfidence on tail classes. Shannon entropy applies a uniform formula $-p\log p$ to all classes, making it unable to distinguish the degree of bias across different classes. As a result, the entropy estimate itself is biased, which degrades the quality of high-confidence view selection.

**Key Challenge**: Shannon entropy assumes an unbiased probability distribution (the extensivity assumption), whereas CLIP's predictive distribution exhibits systematic bias (non-extensivity) that SE cannot characterize.

**Goal**: How can the influence of VLM prediction bias on entropy estimation be corrected during TTA?

**Key Insight**: Tsallis entropy is a generalization of Shannon entropy that captures statistical dependencies among probability distributions via the non-extensive parameter $q$. When $q<1$, TE tends to select more reliable high-confidence views.

**Core Idea**: Replace Shannon entropy with Tsallis entropy for high-confidence view selection, and adaptively compute a debiasing parameter $q^l$ for each class.

## Method

### Overall Architecture
ADTE serves as a plug-and-play replacement for Shannon entropy in TTA methods such as Zero and TPT. The pipeline proceeds as follows: a test image is augmented into $N$ views → ADTE computes the uncertainty of each view → low-entropy, high-confidence views are selected → predictions are aggregated. The key distinction lies in the entropy computation and the use of class-specific parameters.

### Key Designs

1. **Tsallis Entropy as a Replacement for Shannon Entropy**:

   - **Function**: Replace SE $\mathbf{H}_{SE} = -\sum_l P_l \log P_l$ with TE $\mathbf{H}_{TE} = \frac{\sum_l P_l^q - 1}{1-q}$.
   - **Mechanism**: Theoretically, TE reduces to SE as $q \to 1$ (lower-bound property). When $q < 1$, views selected by TE exhibit higher Top-K cumulative reliability (TcrK). For $0 < q < 1$, TE naturally mitigates the influence of VLM prediction bias.
   - **Design Motivation**: SE is sensitive to bias in tail classes (where probabilities approach zero). By replacing $p\log p$ with $p^q$, TE alters the treatment of small probabilities.

2. **Adaptive Debiasing Tsallis Entropy (ADTE)**:

   - **Function**: Customize a class-specific parameter $q^l$ for each class $l$ without manual tuning.
   - **Mechanism**: (1) A memory bank is maintained to estimate the per-class prior probability $\tilde{p}_l$ (solved via Jacobi iteration, approximated with pseudo-labels). (2) The estimated bias is mapped to $[\alpha, \beta] = [0.01, 0.9]$ via min-max normalization to yield $q^l$—classes with larger bias receive smaller $q^l$, implying stronger correction.
   - **Design Motivation**: Manual tuning of $q$ is infeasible across varying test distributions, and different classes are affected by bias to different extents (head vs. tail classes).

3. **Integration with Logit Adjustment**:

   - ADTE integrates seamlessly with logit adjustment strategies: the estimated bias is first used to adjust logits, after which ADTE selects high-confidence views.
   - The entire process requires no additional training or distribution-specific hyperparameter tuning.

### Loss & Training
No training is required. ADTE is a purely inference-time method that directly replaces Shannon entropy in the TTA pipeline. The memory bank stores 10 samples per class.

## Key Experimental Results

### Main Results (ImageNet + 5 Variants, CLIP ViT-B/16)

| Method | IN | IN-A | IN-R | IN-K | Average | OOD Avg |
|--------|----|------|------|------|---------|---------|
| CLIP | 68.7 | 50.6 | 77.7 | 48.3 | 61.5 | 59.7 |
| Zero | 70.9 | 64.0 | 80.8 | 50.3 | 66.2 | 65.0 |
| BCA | 70.2 | 61.1 | 80.7 | 50.9 | 65.6 | 64.4 |
| **ADTE** | **71.8** | **65.5** | **81.4** | **53.5** | **67.5** | **66.5** |

### Cross-Domain Benchmarks (Best Average across 10 Datasets)

| Metric | Description |
|--------|-------------|
| ADTE Average Accuracy | Highest average performance across 10 cross-domain benchmarks |
| Model-agnostic | Outperforms SOTA on both ViT-B/16 and ViT-L/14 |
| Prompt-agnostic | Effective with both hand-crafted templates and CuPL-generated text |

### Key Findings
- TE consistently outperforms SE when $q < 1$ (SE is a special case of TE at $q=1$), though the optimal $q$ varies across test distributions.
- ADTE eliminates the need for manual hyperparameter search via adaptive $q^l$, yielding robust performance across all test distributions.
- The largest gain is observed on ImageNet-K (48.3→53.5), the variant with the most severe distribution shift.
- ADTE can directly replace entropy computation in any SE-based TTA method without further modification.

## Highlights & Insights
- **First systematic analysis of Shannon entropy bias in VLM TTA**: The extensivity assumption implicitly embedded in SE does not hold in this setting—a long-overlooked issue.
- **Tsallis entropy as a principled drop-in replacement**: Theoretically elegant (SE is a lower bound) and practically effective; any TTA method using SE can directly substitute TE/ADTE.
- **Adaptive parameter estimation**: The per-class debiasing parameter $q^l$ is derived from existing bias estimation techniques (from Frolic), reusing available tools without additional overhead.

## Limitations & Future Work
- The memory bank size is fixed at 10 samples per class, which may be insufficient when the label space is large (e.g., 1,000 classes in ImageNet).
- Bias estimation relies on pseudo-label quality; early-stage pseudo-labels may be unreliable.
- The normalization interval $[\alpha, \beta] = [0.01, 0.9]$ remains a manually specified hyperparameter.
- Validation is limited to classification tasks; dense prediction tasks such as detection and segmentation are not covered.

## Related Work & Insights
- **vs. Zero/TPT**: ADTE is a direct upgrade—replacing only the entropy computation yields consistent improvements without modifying other components.
- **vs. Frolic**: Frolic applies logit adjustment for bias correction, whereas ADTE corrects bias at the entropy estimation level; the two approaches are complementary.
- **vs. Prior Use of Tsallis Entropy in Domain Adaptation**: Previous work optimizes TE for pseudo-label generation in source-domain adaptation; ADTE is the first to apply it to view selection in online TTA.

## Rating
- Novelty: ⭐⭐⭐⭐ — Applying Tsallis entropy to VLM TTA represents a novel theoretical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — ImageNet + 5 variants, 10 cross-domain benchmarks, two model scales, two prompt types.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, though notation is dense.
- Value: ⭐⭐⭐⭐ — A plug-and-play replacement for Shannon entropy with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scalable Multi-Task Low-Rank Model Adaptation](scalable_multi-task_low-rank_model_adaptation.md)
- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)
- [\[ICLR 2026\] SAGE: Spatial-visual Adaptive Graph Exploration for Efficient Visual Place Recognition](sage_spatial-visual_adaptive_graph_exploration_for_efficient_visual_place_recogn.md)
- [\[AAAI 2026\] SceneJailEval: A Scenario-Adaptive Multi-Dimensional Framework for Jailbreak Evaluation](../../AAAI2026/social_computing/scenejaileval_a_scenario-adaptive_multi-dimensional_framework_for_jailbreak_eval.md)
- [\[ICCV 2025\] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](../../ICCV2025/social_computing/no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)

</div>

<!-- RELATED:END -->
