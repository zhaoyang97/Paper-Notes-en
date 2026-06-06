---
title: >-
  [Paper Note] Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment
description: >-
  [ICML 2026][Self-Supervised Learning][TabPFN] The authors propose posterior correction for "tabular foundation models" like TabPFN, which feed training sets directly into attention mechanisms as in-context data. Identify…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "TabPFN"
  - "label shift"
  - "posterior adjustment"
  - "temperature scaling"
  - "plug-in correction"
date: 2026-05-08
content_hash: 8dd72fed5e53b202
---

# Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment

**Conference**: ICML 2026  
**arXiv**: [2605.04363](https://arxiv.org/abs/2605.04363)  
**Code**: https://github.com/seunghan96/DistPFN (Available)  
**Area**: Tabular Foundation Models / In-Context Learning / Test-Time Adaptation / Label Shift  
**Keywords**: TabPFN, label shift, posterior adjustment, temperature scaling, plug-in correction

## TL;DR
The authors propose posterior correction for "tabular foundation models" like TabPFN, which feed training sets directly into attention mechanisms as in-context data. Identifying a severe over-fitting to the training set's majority class, they introduce DistPFN: a posterior reweighting via $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$. Across 253 OpenML datasets, this method improves TabPFN-v2 accuracy from 72.7% to 76.9% under strong label shift ($\beta=5$) without retraining, test prior estimation, or architectural modifications.

## Background & Motivation
**Background**: Tabular classification has long been dominated by tree models such as XGBoost, LightGBM, and CatBoost. However, TabPFN (2023) introduced the paradigm of "tabular foundation models" by feeding entire training sets as prompts to a pre-trained Transformer, obtaining all test predictions in a single forward pass. TabPFN-v2 (Nature 2025) reached SOTA scale and generalization through dual-axis attention, inspiring derivatives like LoCalPFN, TabICL, TabFlex, and MixturePFN.

**Limitations of Prior Work**: The authors identify a widely overlooked flaw in this family of models: **oversensitivity to training class priors**. On imbalanced datasets like CostaMadre1, TabPFN-v2 predicts the majority class for 98.3% of test samples, even when training and test distributions are identical. In their study of 253 OpenML datasets, 84.6% were imbalanced, meaning this flaw affects most real-world tabular tasks; performance drops sharply as soon as the test label distribution deviates from the training one (label shift).

**Key Challenge**: Classical label shift correction methods (EME, BBE, Logit Adjustment, Balanced Softmax) either require retraining or the estimation of **test set** label priors. Retraining destroys the zero-shot advantage of TabPFN, while test priors are rarely available in real-world deployments. Furthermore, applying these methods in standard (no-shift) settings often degrades performance (EME/BBE drop 1.5/1.1 points respectively on LoCalPFN w/o shift). Thus, the root problem is: **existing methods require data, require training, or degrade standard performance**.

**Goal**: (1) Provide a **training-free** plug-in posterior correction, (2) **without needing to estimate test priors**, (3) while maintaining the base model's original performance w/o shift, and (4) providing increasing gains as drift intensity increases w/ shift.

**Key Insight**: The authors observe a fundamental difference between TabPFN-like models and traditional models: **the training distribution is explicitly encoded into the attention mechanism rather than implicitly into model weights**. Consequently, the training prior $p_{train}(y)$ is an explicit, observable quantity in TabPFN, whereas classical methods must estimate it indirectly. Once this distinction is clear, the solution becomes straightforward.

**Core Idea**: Divide the model output posterior $\hat{p}(y)$ by the training prior $p_{train}(y)$, and normalize using $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$. This effectively "dampens the pull of the training distribution and amplifies the evidence from the test sample itself."

## Method

### Overall Architecture
The method consists of three components: (1) DistPFN, a single-line posterior adjustment formula; (2) DistPFN-T, a temperature-scaled version that adaptively controls adjustment intensity via cross-entropy; and (3) an inverse-frequency resampling benchmark to quantify the "drift intensity $\beta$ vs. accuracy" curve on OpenML. The pipeline involves taking logits from a TabPFN forward pass $\rightarrow$ softmax $\rightarrow$ multiplying/dividing by an adjustment factor $\alpha$ $\rightarrow$ renormalization $\rightarrow$ output. The adjustment occurs entirely at **inference time** as a plug-in.

### Key Designs

1.  **DistPFN: Posterior/Prior Ratio as Adjustment Factor**:
    - **Function**: Corrects the biased posterior $\hat{p}_{TabPFN}(y)$ toward a distribution less biased by the majority class.
    - **Mechanism**: $\tilde{p}_{DistPFN}(y) = \mathrm{Norm}\!\left(\hat{p}_{TabPFN}(y) \cdot \frac{\hat{p}_{TabPFN}(y)}{p_{train}(y)}\right) = \mathrm{Norm}\!\left(\frac{\hat{p}_{TabPFN}(y)^2}{p_{train}(y)}\right)$, where $p_{train}(y)$ is the training class frequency. This is a variant of the "prior removal" idea by Saerens et al. (2002) ($p(y|x) \propto p(y|x)/p(y)$), but uses $\hat{p}^2$ in the numerator to retain the model's predictive information, preventing over-correction (termed "partial correction").
    - **Design Motivation**: Classical prior correction assumes $\hat{p}(y) \approx p_{train}(y)$ to fully remove the training prior, but in practice, $\hat{p}(y)$ does not fully collapse to $p_{train}(y)$. Thus, the $\hat{p}^2/p_{train}$ form provides a compromise that oracle experiments show is "near-optimal."

2.  **DistPFN-T: Adaptive Temperature Scaling via KL/CE**:
    - **Function**: Dynamically adjusts correction intensity based on the deviation between model predictions and training priors—larger deviations indicate higher distribution shift and warrant more aggressive adjustment.
    - **Mechanism**: Temperature is defined as $\tau = \mathrm{CE}(\hat{p}_{TabPFN}(y), p_{train}(y))$. Predictions are temperature-scaled as $\hat{p}_{TabPFN\text{-}T}(y=c) = \mathrm{softmax}(\hat{p}_{TabPFN}(y=c)/\tau)$, then adjusted via $\tilde{p}_{DistPFN\text{-}T}(y) = \mathrm{Norm}\!\left(\hat{p}_{TabPFN}(y) \cdot \frac{\hat{p}_{TabPFN\text{-}T}(y)}{p_{train}(y)}\right)$.
    - **Design Motivation**: Fixed DistPFN might over-correct under strong shift. Using $\tau$ as a self-monitoring signal ensures that (a) high deviations lead to higher $\tau$, smoothing the scaling and keeping the adjustment resilient, and (b) it counterbalances biases by magnifying minority classes in majority-biased cases and vice-versa.

3.  **Inverse-Frequency Resampling Benchmark**:
    - **Function**: Systematically measures the "drift intensity vs. accuracy" curve across 253 OpenML datasets by controlling label shift with a scalar $\beta \geq 0$ without modifying test sets.
    - **Mechanism**: Sampling weights $w_k = (1/p(y=c_k))^\beta$ are assigned to each class $c_k$. Training sets are oversampled according to normalized weights $\tilde{w}_k$. $\beta = 0$ corresponds to uniform resampling, while increasing $\beta$ shifts the distribution further toward rare classes.
    - **Design Motivation**: Existing benchmarks lack continuous drift intensity metrics. This allows large-scale controlled evaluation across $\beta \in \{0, 0.1, 0.5, 1, 2, 5\}$.

### Loss & Training
No training or fine-tuning is required. The method is entirely composed of inference-time probability reweighting.

## Key Experimental Results

Evaluated on 253 OpenML datasets (50/50 train/test split, average of 5 seeds) across 6 $\beta$ levels. Comparison with 16 baselines (LogReg, SVM, RF, LightGBM, CatBoost, FT-Transformer, TabM, etc.).

### Main Results

| Method | $\beta=0$ | $\beta=0.1$ | $\beta=0.5$ | $\beta=1$ | $\beta=2$ | $\beta=5$ | Avg (w/ shift) |
|---|---|---|---|---|---|---|---|
| CatBoost | 0.803 | 0.774 | 0.771 | 0.751 | 0.718 | 0.665 | 0.717 |
| RealMLP | 0.794 | 0.760 | 0.758 | 0.745 | 0.720 | 0.677 | 0.717 |
| TabPFN-v2 (base) | **0.818** | 0.797 | 0.796 | 0.790 | 0.782 | 0.759 | 0.775 |
| + DistPFN | 0.818 | 0.799 | 0.797 | 0.795 | 0.791 | 0.783 | 0.789 |
| **+ DistPFN-T** | **0.818** | **0.799** | **0.798** | **0.797** | **0.796** | **0.789** | **0.792** |
| + DistPFN-Oracle (Upper) | 0.818 | 0.803 | 0.802 | 0.800 | 0.797 | 0.792 | 0.796 |
| TabICL (base) | 0.806 | 0.783 | 0.781 | 0.770 | 0.747 | 0.704 | 0.742 |
| TabICL + DistPFN-T | 0.806 | 0.786 | 0.786 | 0.783 | 0.780 | 0.771 | 0.777 |
| LoCalPFN (base) | 0.816 | 0.794 | 0.793 | 0.788 | 0.778 | 0.753 | 0.771 |
| LoCalPFN + DistPFN-T | 0.816 | 0.798 | 0.797 | 0.796 | 0.794 | 0.787 | 0.791 |

Key Observation: Under $\beta=5$, TabPFN-v2 + DistPFN-T pushed the base from 75.9% to 78.9%; TabICL from 70.4% to 77.1% (+6.7pp). Gains are consistent across three different FMs, demonstrating model-agnosticism.

### Ablation Study

| Configuration | w/o shift | w/ shift (Avg) | Description |
|---|---|---|---|
| TabPFN-v2 (base) | 0.818 | 0.775 | Baseline |
| + EME (Saerens 2002) | 0.801 | 0.786 | -1.7pp w/o shift |
| + BBE (Lipton 2018) | 0.805 | 0.789 | -1.3pp w/o shift |
| + DistPFN | 0.818 | 0.789 | **No loss w/o shift** |
| **+ DistPFN-T** | **0.818** | **0.792** | **No loss w/o shift + Max gain w/ shift** |
| + DistPFN-Oracle (True $p_{test}$) | 0.818 | 0.796 | Performance upper bound |
| TableShift Diabetes OOD | base 0.589 → DistPFN-T 0.600 | — | Gains on real OOD |

### Key Findings
- **Gains scale with drift**: As the train-test KL divergence increases, the accuracy gain of DistPFN-T rises monotonically, confirming it directly combats label shift rather than acting as a random regularizer.
- **Approaching Oracle**: DistPFN-T (78.9% at $\beta=5$) even slightly outperforms "hard dividing" by true test priors (78.4%) because temperature scaling provides smoother adjustment.
- **zero performance loss w/o shift**: Unlike EME/BBE which drop 1–2pp when there is no shift, DistPFN-T strictly maintains base performance when $\beta=0$.
- **Majority-class bias is systemic**: 84.6% of OpenML datasets are imbalanced, indicating this is a systemic issue for TabPFN rather than a corner case.

## Highlights & Insights
- **Explicit Prior Observation**: The pivot of the paper is the observation that TabPFN explicitly encodes training sets. This bypasses the need to estimate priors, a major hurdle in classical label shift literature.
- **"Partial Correction" with $\hat{p}^2/p_{train}$**: This engineering choice prevents over-correction by retaining model confidence.
- **Self-monitoring Design**: DistPFN-T uses only internal signals (the model's own output and known training priors), making it entirely self-contained.
- **Inverse-frequency Benchmark**: Provides a reusable methodology for future label shift evaluations to generate continuous performance curves.

## Limitations & Future Work
- Designed specifically for explicit-prior models (TabPFN family + kNN); benefits for tree models are less clear.
- It remains a partial correction; a 0.4pp gap to the absolute oracle remains.
- Only classification was tested; label distribution shift in regression was not explored.
- In extreme long-tail cases (e.g., 100+ classes), adjustment factors might be overly sensitive to overconfident predictions.

## Related Work & Insights
- **vs. EME/BBE**: These require iterative estimation of test priors and degrade performance without shift; Ours is non-iterative and lossless w/o shift.
- **vs. Logit Adjustment/Balanced Softmax**: These require loss modification during training; Ours is an inference-time plug-in for frozen foundation models.
- **vs. TTA (Test-Time Training)**: Most TTA requires backpropagation on test samples; Ours has near-zero computational overhead.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combines the insight of explicit priors with posterior adjustment and temperature scaling in a simple yet sharp way.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High-density assessment across 253 datasets, multiple $\beta$ levels, seeds, and foundation models.
- **Writing Quality**: ⭐⭐⭐⭐ Clear contrast between explicit/implicit models and logical flow.
- **Value**: ⭐⭐⭐⭐ High industrial utility as a one-line plug-in for any TabPFN-v2/TabICL deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)
- [\[CVPR 2026\] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](../../CVPR2026/self_supervised/redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)
- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](../../NeurIPS2025/self_supervised/hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[ICCV 2025\] To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models](../../ICCV2025/self_supervised/to_label_or_not_to_label_palm_-_a_predictive_model_for_evaluating_sample_efficie.md)
- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)

</div>

<!-- RELATED:END -->
