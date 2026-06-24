---
title: >-
  [Paper Note] Improving Low-Resource Morphological Inflection via Self-Supervised Objectives
description: >-
  [ACL 2025][Self-Supervised Learning][Morphological Inflection] This paper systematically explores the effectiveness of 13 self-supervised auxiliary objectives (Autoencoding, CMLM, T5-style, etc.) in extremely low-resource morphological inflection tasks. It finds that autoencoding is optimal when unlabeled data is extremely scarce, whereas character-level MLM is better when data increases. Mask sampling based on morpheme boundaries represents the most promising direction.
tags:
  - "ACL 2025"
  - "Self-Supervised Learning"
  - "Morphological Inflection"
  - "Low-Resource Languages"
  - "Masked Language Modeling"
  - "Character-Level Sequence-to-Sequence"
date: 2026-05-08
content_hash: f90c5e2e12928b01
---

# Improving Low-Resource Morphological Inflection via Self-Supervised Objectives

**Conference**: ACL 2025  
**arXiv**: [2506.05227](https://arxiv.org/abs/2506.05227)  
**Code**: None  
**Area**: Self-Supervised Learning / Morphology  
**Keywords**: Morphological Inflection, Self-Supervised Learning, Low-Resource Languages, Masked Language Modeling, Character-Level Sequence-to-Sequence

## TL;DR
This paper systematically explores the effectiveness of 13 self-supervised auxiliary objectives (Autoencoding, CMLM, T5-style, etc.) in extremely low-resource morphological inflection tasks. It finds that autoencoding is optimal when unlabeled data is extremely scarce, whereas character-level MLM is better when data increases. Mask sampling based on morpheme boundaries represents the most promising direction.

## Background & Motivation
**Background**: Morphological inflection is a character-level seq2seq task (e.g., try + PST $\rightarrow$ tried), which is crucial for language documentation. Advances in NLP mainly rely on large-scale pre-training data, but many languages lack such resources.

**Limitations of Prior Work**: Morphological inflection in low-resource settings (200-600 supervised samples) suffers from poor performance. Existing works majorly improve performance via architectural inductive biases; self-supervised objectives, though successful in high-resource NLP, are under-explored in character-level tasks.

**Key Challenge**: Since character-level task models are small and data is scarce, conventional large-scale MLM pre-training is inapplicable, necessitating targeted self-supervised strategies.

**Goal**: Which self-supervised objective is best suited for extremely low-resource morphological inflection? How do different masking strategies, objective functions, and data filtering methods affect performance?

**Key Insight**: Systematically compare masking strategies (iid/suffix/prefix) $\times$ objectives (CMLM/T5) $\times$ deletion vs. masking $\times$ morpheme boundary masking within a multi-task learning framework.

**Core Idea**: A systematic comparison of character-level MLM variants and morpheme boundary masking serves as the optimal self-supervised direction for low-resource morphological inflection.

## Method

### Overall Architecture
An encoder-decoder Transformer (7.4M parameters) trained via multi-task learning: the main task is morphological inflection (lemma + tag $\rightarrow$ inflected form), and the auxiliary task is self-supervised denoising (corruption + `[TASK]` $\rightarrow$ original word). Both tasks share the model, and their losses are directly summed.

### Key Designs

1. **Masking Objective Variants**:

    - CMLM: Dynamic sampling masking of 25% of characters (80% replaced with `[MASK]`, 10% replaced with random characters, 10% kept unchanged)
    - T5-style: Merges adjacent masked characters into a single span token `<X><Y>` after 25% sampling
    - Autoencoding (AE): Directly copies input to output with zero noise

2. **Masking Sampling Strategies**:

    - iid: Uniformly distributed sampling
    - suffix: 95% probability allocated to the last 1/3 characters (simulating suffix change, which is typologically most common)
    - prefix: 95% probability allocated to the first 1/3 characters

3. **Character Deletion vs. Masking**:

    - The deletion mode directly removes characters instead of replacing them with `[MASK]`, simulating the additive behavior of seq2seq (e.g., bake $\rightarrow$ baked)

4. **Morpheme Boundary Masking (Segment Masking)**:

    - Utilizes known morpheme boundaries (oracle) to sample masks by entire morpheme segments (e.g., walk-ing $\rightarrow$ walk`[MASK]` $\rightarrow$ walking)

### Loss & Training
Standard seq2seq cross-entropy loss, where the main task loss and auxiliary task loss are summed. Training takes approximately 1 hour per model on an A100 GPU.

## Key Experimental Results

### Main Results (Average Accuracy across 19 Languages)

| Dataset | Baseline | AE (Autoencoding) | cmlm-iid | cmlm-suff | t5-iid | t5-suff |
|--------|----------|------------|----------|-----------|--------|---------|
| ud-1k (1k supervised) | 64.39 | **75.83** | 74.67 | 74.07 | 74.39 | 73.43 |
| ud-200 (200 supervised) | 5.16 | **47.48** | 42.92 | 42.76 | 41.04 | 41.34 |
| ud-wl-NR (deduplicated) | 5.16 | 50.49 | **51.68** | 50.51 | 51.26 | 49.19 |

### Ablation Study

| Dimension | Conclusion |
|---------|------|
| Masking vs. Deletion | Masking consistently outperforms deletion (average +2-3 pp) |
| suffix vs. iid vs. prefix | iid and suffix are close, prefix is the worst |
| CMLM vs. T5 | CMLM is slightly better (more flexible masking details) |
| Morpheme boundary masking | Consistent improvement across 5 languages (the most promising direction) |

### Key Findings
- **Autoencoding is strongest when data is extremely scarce**: On ud-200, $AE = 47.48$ vs. $\text{cmlm-iid} = 42.92$, with a gap of 4.56 percentage points (pp). Explanation: Autoencoding enhances the copy bias, which perfectly aligns with the "copy-and-modify" nature of inflection tasks.
- **MLM overtakes when data increases**: On ud-wl-NR, $\text{cmlm-iid} = 51.68 > AE = 50.49$. Data diversity allows MLM to leverage its advantages.
- **Objectives with strong inductive biases are not necessarily better**: Although the suffix strategy intuitively matches suffixation inflection better, it underperforms compared to the neutral iid strategy.
- **Morpheme boundary masking is an exception**: Sampling masks using ground-truth morpheme boundaries yields consistent improvements, making it the most promising direction.

## Highlights & Insights
- **Systematic exploration of self-supervised learning for low-resource character-level tasks**: With 13 objectives $\times$ 6 datasets $\times$ 19 languages, the experiments are comprehensive and provide clear practical guidance for this field.
- **Unexpected success of autoencoding**: In extremely low-resource settings, the simplest autoencoding (without noise) is actually optimal. This contradicts the dominance of MLM in high-resource NLP, demonstrating that inductive bias is more important in low-resource scenarios.
- **Promise of morpheme boundary masking**: Although it requires oracle boundary information, it yields consistent improvements, suggesting that future work could substitute this with unsupervised morpheme segmentation.

## Limitations & Future Work
- Morpheme boundary masking requires oracle information; practical applications need to explore unsupervised segmentation methods.
- Experiments are only conducted on UD corpora; other low-resource data sources have not been tested.
- The impact of auxiliary task ratios (main task vs. self-supervision) has not been explored in depth.
- The model has only 7.4M parameters; whether larger models exhibit different trends remains unknown.

## Related Work & Insights
- **vs. Purushothama et al. (2024)**: They found that AE auxiliary tasks are effective. Building upon this, this study systematically compares 13 variants and finds that the advantage of AE is limited to extremely small datasets.
- **vs. ByT5 (Xue et al. 2022)**: ByT5 is a large byte-level pre-trained model, whereas this paper focuses on training from scratch in extremely low-resource settings, making them complementary rather than competitive.
- **vs. SIGMORPHON shared tasks**: Comparable to the low-resource settings in shared tasks (~700 samples), providing a practical solution for auxiliary task enhancement.

## Rating
- Novelty: ⭐⭐⭐ The method itself is not new (MLM variants + multi-task), but the systematic comparison is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 13 objectives $\times$ 6 datasets $\times$ 19 languages $\times$ masking/deletion/strategy ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-organized, with a systematic experimental design.
- Value: ⭐⭐⭐⭐ Provides clear guidance for selecting self-supervised strategies in low-resource character-level tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)
- [\[ACL 2025\] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning](whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] ReSA: Clustering Properties of Self-Supervised Learning](../../ICML2025/self_supervised/clustering_properties_of_self-supervised_learning.md)
- [\[CVPR 2025\] AutoSSVH: Automated Frame Sampling for Self-Supervised Video Hashing](../../CVPR2025/self_supervised/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_.md)

</div>

<!-- RELATED:END -->
