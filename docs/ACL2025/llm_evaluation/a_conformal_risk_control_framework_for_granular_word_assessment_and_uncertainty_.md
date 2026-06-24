---
title: >-
  [Paper Note] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates
description: >-
  [LLM Evaluation] A conformal risk control framework is proposed for granular word-level error detection and uncertainty calibration of CLIPScore. By generating a score distribution through simple attention mask sampling, this method provides formal risk control guarantees while remaining model-agnostic.
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: bbe619053552769d
---

# A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates

- **Conference**: ACL 2025
- **arXiv**: [2504.01225](https://arxiv.org/abs/2504.01225)
- **Code**: Not provided
- **Area**: LLM Evaluation / Image-Text Evaluation / Uncertainty Quantification
- **Keywords**: CLIPScore, Conformal Risk Control, Foil Word Detection, Uncertainty Calibration, Image Captioning Evaluation

## TL;DR

A conformal risk control framework is proposed for granular word-level error detection and uncertainty calibration of CLIPScore. By generating a score distribution through simple attention mask sampling, this method provides formal risk control guarantees while remaining model-agnostic.

## Background & Motivation

- **Problem**: Existing image captioning evaluation metrics, such as CLIPScore, only provide single-point quality estimates and lack two crucial capabilities: **(1)** the ability to locate specific error words (foil words) in the caption, and **(2)** uncertainty quantification, making it difficult to judge the reliability of a single score.
- **Limitations of Prior Work**: Deep Ensembles require training multiple models, Monte Carlo Dropout is not applicable to CLIP (as it lacks dropout layers), and gradient attribution methods (GAE), though effective, depend on specific model architectures.
- **Core Motivation**: There is a need for a **model-agnostic**, **simple and effective** method that can generate an output distribution of CLIPScore to quantify uncertainty, while also providing users with controllable error detection guarantees through a formal framework.

## Method

### Overall Architecture

The system is divided into three stages:
1. **Distribution Generation**: Generate the distribution of CLIPScore values through Attention Mask Sampling.
2. **Foil Word Detection**: Calibrate the threshold $\lambda$ based on conformal risk control to detect foil words in the description.
3. **Confidence Interval Construction**: Fit a truncated Gaussian distribution to the CLIPScore distribution to generate calibrated confidence intervals.

### Key Designs

1. **Attention Mask Sampling Strategy**: Randomly mask $\xi_i\%$ of patches on the image side (in the self-attention layer) and $\xi_t\%$ of words with specific part-of-speech (POS) tags (nouns, verbs, adjectives, etc.) on the text side. A distribution is constructed using $I$ image versions $\times$ $T$ text versions = $I \times T$ different CLIPScore values. The key lies in mapping subword tokens back to word-level POS tags to ensure semantic consistency in masking.

2. **Word-level Error Score Derivation**: For each word $w_j$, its contribution is estimated by the change in CLIPScore when it is masked. A positive difference indicates that the word has a negative impact on the original score (i.e., it is likely a foil word). After aggregating multiple samples, a normalized error score $f_v[j]$ is obtained via a sigmoid transformation.

3. **Conformal Risk Control Calibration**: The Hoeffding-Bentkus combined concentration inequality is used to construct the Upper Confidence Bound (UCB) of risk. The optimal threshold $\hat{\lambda}$ is found through the calibration set, ensuring the formal guarantee under the user-defined risk tolerance $\alpha$ and error rate $\delta$: $P(R(\hat{\lambda}) < \alpha) \geq 1-\delta$. FDR is controlled for multi-class tasks, and FPR is controlled for multi-label tasks.

### Loss & Training

Model training is not involved. The core optimization objective is: find the minimum $\lambda$ on the calibration set such that the UCB risk corresponding to all $\lambda' \geq \lambda$ is lower than the target tolerance $\alpha$. For the confidence interval task, the Learn Then Test (LTT) method is used to handle non-monotonic risk functions and optimize the Uncertainty Pearson Score (UPS).

## Experiments

### Main Results

| Method | FOIL-it AP | FOIL-it LA | FOIL-nocaps LA (Overall) |
|------|-----------|-----------|-------------------------|
| CHAIR | 92.5 | 79.0 | 14.4 |
| ALOHa | 61.4 | 40.0 | 45.2 |
| GAE_B (ViT-B/32) | 71.4 | 73.2 | 60.3 |
| GAE_H (ViT-H/14) | 80.6 | 83.6 | 71.6 |
| **Ours (ViT-B/32)** | 59.7 | 40.2 | 54.9 |
| **Ours (ViT-H/14)** | 63.4 | 51.4 | 60.3 |

Rich-HF Multi-label Benchmark:

| Method | Precision | Recall | F1 |
|------|-----------|--------|-----|
| ALOHa | 34.4 | 31.1 | 38.5 |
| Rich-HF (Fine-tuned) | 43.9 | 61.3 | 34.1 |
| GAE_H | 42.7 | 36.5 | 51.6 |
| **Ours (ViT-H/14)** | 32.0 | **64.2** | **42.7** |

### Ablation Study

| Risk Tolerance $\alpha$ | Calibration Set FDR | Test Set FDR | Test Set F1 | Test Set LA |
|-------------|-----------|-----------|---------|---------|
| 10% | 9.69 | 10.10 | 61.93 | 34.39 |
| 20% | 19.58 | 20.20 | **63.76** | 41.92 |
| 30% | 29.52 | 30.24 | 62.81 | 47.06 |
| 50% | 49.47 | 50.27 | 56.68 | 54.88 |

### Key Findings

1. **Formal guarantees are valid**: The FDR/FPR tolerances set on the calibration set are highly consistent with the actual values observed on the test set, remaining conservative yet effective even under distribution shifts across datasets (FOIL-it $\rightarrow$ FOIL-nocaps).
2. **Simple method + calibration $\approx$ complex method**: Although attention mask sampling is simple, after conformal calibration, it outperforms Rich-HF (which requires fine-tuning) and the LLM-based ALOHa on multi-label tasks.
3. **Confidence interval calibration significantly improves UPS**: After LTT calibration, the correlation between prediction error and uncertainty estimation (UPS) is improved across all datasets, while maintaining the correlation with human judgments (Kendall-$\tau$) unchanged.

## Highlights & Insights

- **Model-agnostic + Formal Guarantees**: The method can be applied to any CLIP model or other learned evaluation metrics without retraining, while providing user-defined risk control guarantees.
- **Methodological Innovation**: Conformal risk control is introduced from traditional classification/regression domains to vision-language evaluation, addressing the issue of non-monotonic risk functions (using the LTT method).
- **High Practical Value**: Users can adjust the risk tolerance according to the scenario—lowering $\alpha$ for high-precision scenarios, and raising $\alpha$ for high-recall scenarios.

## Limitations & Future Work

- Compared to gradient attribution methods like GAE, pure attention mask sampling still has a detection accuracy gap, especially in multi-class single-error scenarios (FOIL-it LA).
- The quality of calibration depends on the size and distributional representativeness of the calibration set (Rich-HF has only 955 samples, leading to a conservative UCB estimate).
- The masking rates $\xi_i$ and $\xi_t$, as well as the number of sampling steps $I$ and $T$, are hyperparameters that may require adjustment for different datasets.
- Only English description scenarios were validated; cross-lingual generalizability remains unknown.

## Related Work & Insights

- **Uncertainty Quantification**: MC Dropout (Gal & Ghahramani, 2016), Deep Ensembles (Kendall & Gal, 2017)
- **Conformal Prediction**: Conformal risk control frameworks by Angelopoulos & Bates (2021), Bates et al. (2021)
- **Image-Text Evaluation Metrics**: CLIPScore (Hessel et al., 2021), BERTScore
- **Foil Word Detection**: FOIL-it (Shekhar et al., 2017), ALOHa (Petryk et al., 2024), GAE (Nam et al., 2024), Rich-HF (Liang et al., 2024)

## Rating

- **Novelty**: 7/10 — Introducing conformal risk control into CLIPScore uncertainty quantification is a novel cross-disciplinary application.
- **Technical Depth**: 8/10 — Rigorous theoretical derivation (Hoeffding-Bentkus bound + LTT) and complete formal guarantees.
- **Experimental Thoroughness**: 8/10 — Three foil detection benchmarks + four confidence interval datasets, with ablations over various $\alpha$ values.
- **Writing Quality**: 7/10 — The extensive mathematical derivations make it somewhat difficult to follow, but the framework diagram and experimental tables are clearly organized.
- **Overall Score**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ICML 2025\] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](../../ICML2025/llm_evaluation/the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)
- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](androidlab_autonomous_agent.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)
- [\[ACL 2025\] Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)

</div>

<!-- RELATED:END -->
