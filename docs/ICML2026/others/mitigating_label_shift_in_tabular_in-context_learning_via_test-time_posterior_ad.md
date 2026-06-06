---
title: >-
  [Paper Note] Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment
description: >-
  [ICML 2026][TabPFN] For TabPFN-type "tabular foundation models" that feed the training set directly as in-context input to attention…
tags:
  - "ICML 2026"
  - "TabPFN"
  - "label shift"
  - "posterior adjustment"
  - "temperature scaling"
  - "plug-in correction"
date: 2026-05-08
content_hash: 451d3ded2e3922cc
---

# Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment

**Conference**: ICML 2026  
**arXiv**: [2605.04363](https://arxiv.org/abs/2605.04363)  
**Code**: https://github.com/seunghan96/DistPFN (available)  
**Area**: Tabular Foundation Models / In-Context Learning / Test-Time Adaptation / Label Shift  
**Keywords**: TabPFN, label shift, posterior adjustment, temperature scaling, plug-in correction

## TL;DR
For TabPFN-type "tabular foundation models" that feed the training set directly as in-context input to attention, this work proposes posterior correction—finding that such models severely overfit the majority class in the training set. The authors introduce DistPFN: a one-line posterior reweighting $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$, which lifts TabPFN-v2 accuracy under strong label shift ($\beta=5$) from 72.7% to 76.9% on 253 OpenML datasets—without retraining, estimating test priors, or modifying the architecture.

## Background & Motivation
**Background**: Tabular classification has long been dominated by tree models like XGBoost/LightGBM/CatBoost. In 2023, TabPFN introduced the paradigm of "feeding the entire training set as a prompt to a pretrained Transformer, producing all test predictions in a single forward pass," bringing in-context learning to tabular data. TabPFN-v2 (Nature 2025) further achieved SOTA scale and generalization via dual-axis attention, spawning a series of "tabular foundation models" (Tabular FM) such as LoCalPFN, TabICL, TabFlex, MixturePFN, etc.

**Limitations of Prior Work**: The authors identify a critical, widely overlooked flaw in this family of models—**extreme sensitivity to the class prior in the training set**. On imbalanced datasets like CostaMadre1, TabPFN-v2 predicts 98.3% of test samples as the majority class, even when train and test distributions are identical. Among the 253 OpenML datasets examined, 84.6% are imbalanced, meaning this flaw affects most real-world tabular tasks. Moreover, even slight label shift between train and test distributions causes a dramatic performance drop.

**Key Challenge**: Classical label shift correction methods (EME, BBE, Logit Adjustment, Balanced Softmax) either require retraining or estimating the **test set** label prior—compromising TabPFN's zero-shot advantage or being infeasible in real deployments. Moreover, enabling these methods in standard (no shift) settings actually degrades performance (EME/BBE on LoCalPFN drop 1.5/1.1 points w/o shift). Thus, the core issue is: **existing methods require extra data, retraining, or harm standard performance**.

**Goal**: (1) Provide a **completely training-free** plug-in posterior correction, (2) **no need to estimate test priors**, (3) preserve base model performance w/o shift, (4) deliver increasing gains as shift strength grows.

**Key Insight**: The fundamental difference between TabPFN-type models and traditional models is that **the training set distribution is explicitly encoded in attention, not implicitly in model weights**. Thus, $p_{train}(y)$ is directly accessible and computable in TabPFN, whereas classical label shift methods must estimate priors from weights. Recognizing this, the solution becomes straightforward.

**Core Idea**: Divide the model output posterior $\hat{p}(y)$ by the training prior $p_{train}(y)$, then normalize: $\tilde{p}(y) \propto \hat{p}(y)^2 / p_{train}(y)$—"dampening the pull of the training distribution, amplifying evidence from the test sample itself."

## Method

### Overall Architecture
The method comprises three compact components: (1) a one-line posterior adjustment formula, DistPFN; (2) a temperature scaling variant, DistPFN-T, which adaptively controls adjustment strength via cross-entropy; (3) an inverse-frequency resampling benchmark construction to systematically quantify "shift strength $\beta$ vs accuracy" curves on OpenML. The pipeline: obtain logits from a single TabPFN forward pass → softmax → multiply/divide by adjustment factor $\alpha$ → renormalize → output. All adjustment occurs **at inference**, fully plug-in, with no changes to TabPFN parameters or architecture.

### Key Designs

1. **DistPFN: Posterior/Prior Ratio as Adjustment Factor**:

    - **Function**: Corrects the model's biased posterior $\hat{p}_{TabPFN}(y)$, which is pulled by the training distribution, toward a less majority-skewed distribution.
    - **Mechanism**: $\tilde{p}_{DistPFN}(y) = \mathrm{Norm}\!\left(\hat{p}_{TabPFN}(y) \cdot \frac{\hat{p}_{TabPFN}(y)}{p_{train}(y)}\right) = \mathrm{Norm}\!\left(\frac{\hat{p}_{TabPFN}(y)^2}{p_{train}(y)}\right)$, where $p_{train}(y)$ is the class frequency in the training set, and $\mathrm{Norm}(\cdot)$ denotes class-wise normalization. The intuition is classic—this is a variant of the "removing training prior" idea from Saerens et al. 2002 ($p(y|x) \propto p(y|x)/p(y)$), but uses $\hat{p}^2$ in the numerator to retain model prediction information and avoid overcorrection ("partial correction" per the paper).
    - **Design Motivation**: Classical prior correction assumes $\hat{p}(y) \approx p_{train}(y)$ and fully divides out the training prior; in practice, $\hat{p}(y)$ does not collapse entirely to $p_{train}(y)$, so full correction overcompensates. The $\hat{p}^2/p_{train}$ form is empirically validated by oracle experiments as "near-optimal" compromise.

2. **DistPFN-T: Adaptive Temperature Scaling via KL/CE**:

    - **Function**: Dynamically adjusts correction strength based on the "deviation between model prediction and training prior"—greater deviation indicates greater train-test distribution mismatch, warranting more aggressive adjustment, but first smooths overconfident predictions to prevent overcorrection.
    - **Mechanism**: Set temperature $\tau = \mathrm{CE}(\hat{p}_{TabPFN}(y), p_{train}(y))$ (cross-entropy between training prior and current prediction), then apply temperature scaling: $\hat{p}_{TabPFN\text{-}T}(y=c) = \mathrm{softmax}(\hat{p}_{TabPFN}(y=c)/\tau)$, finally compute $\tilde{p}_{DistPFN\text{-}T}(y) = \mathrm{Norm}\!\left(\hat{p}_{TabPFN}(y) \cdot \frac{\hat{p}_{TabPFN\text{-}T}(y)}{p_{train}(y)}\right)$.
    - **Design Motivation**: Fixed $\hat{p}^2/p_{train}$ in DistPFN suffices for weak shift but can overcorrect under strong shift. Using $\tau$ as a self-monitoring signal: (a) greater deviation from training prior → larger $\tau$ → smoother predictions after scaling → milder but sustained adjustment; (b) further amplifies minority in majority cases, and softens in minority cases, providing "counterbalance the bias" in both directions.

3. **Inverse-Frequency Resampling Benchmark**:

    - **Function**: Controls label shift strength via a scalar $\beta \geq 0$ by modifying only the training set, enabling systematic "shift strength vs accuracy" curves on 253 OpenML datasets.
    - **Mechanism**: Assign each class $c_k$ a sampling weight $w_k = (1/p(y=c_k))^\beta$, normalize $\tilde{w}_k = w_k / \sum_j w_j$, and oversample (not undersample) the training set according to $\tilde{w}_k$ to avoid information loss. $\beta = 0$ yields uniform resampling; increasing $\beta$ increasingly favors rare classes, making the training distribution diverge from the test distribution.
    - **Design Motivation**: Existing label shift benchmarks (e.g., TableShift) offer only a few real shift points, insufficient for continuous curves; this inverse-frequency approach enables large-scale, controlled evaluation at $\beta \in \{0, 0.1, 0.5, 1, 2, 5\}$ × 253 datasets × 5 seeds, providing cleaner signals than real OOD benchmarks.

### Loss & Training
No training or fine-tuning is required; the entire method is inference-time probability reweighting. The only "hyperparameter" is whether to use DistPFN-T (i.e., enable temperature scaling).

## Key Experimental Results

253 OpenML datasets (50/50 train/test split, averaged over 5 seeds), 6 $\beta$ levels, reporting both w/o shift and w/ shift averages. Compared against 16 baselines (including LogReg/SVM/MLP/kNN/RF/LightGBM/CatBoost/FT-Transformer/TabM/RealMLP/LoCalPFN/TabICL/TabPFN-v2, etc.).

### Main Results

| Method | $\beta=0$ | $\beta=0.1$ | $\beta=0.5$ | $\beta=1$ | $\beta=2$ | $\beta=5$ | Mean (w/ shift) |
|---|---|---|---|---|---|---|---|
| CatBoost | 0.803 | 0.774 | 0.771 | 0.751 | 0.718 | 0.665 | 0.717 |
| RealMLP | 0.794 | 0.760 | 0.758 | 0.745 | 0.720 | 0.677 | 0.717 |
| TabPFN-v2 (base) | **0.818** | 0.797 | 0.796 | 0.790 | 0.782 | 0.759 | 0.775 |
| + DistPFN | 0.818 | 0.799 | 0.797 | 0.795 | 0.791 | 0.783 | 0.789 |
| **+ DistPFN-T** | **0.818** | **0.799** | **0.798** | **0.797** | **0.796** | **0.789** | **0.792** |
| + DistPFN-Oracle (upper bound) | 0.818 | 0.803 | 0.802 | 0.800 | 0.797 | 0.792 | 0.796 |
| TabICL (base) | 0.806 | 0.783 | 0.781 | 0.770 | 0.747 | 0.704 | 0.742 |
| TabICL + DistPFN-T | 0.806 | 0.786 | 0.786 | 0.783 | 0.780 | 0.771 | 0.777 |
| LoCalPFN (base) | 0.816 | 0.794 | 0.793 | 0.788 | 0.778 | 0.753 | 0.771 |
| LoCalPFN + DistPFN-T | 0.816 | 0.798 | 0.797 | 0.796 | 0.794 | 0.787 | 0.791 |

Key observations: At $\beta=5$, TabPFN-v2 + DistPFN-T lifts base from 75.9% to 78.9%; TabICL from 70.4% to 77.1% (+6.7pp); LoCalPFN from 75.3% to 78.7%. The consistent gains across three different FMs indicate model-agnostic effectiveness.

### Ablation Study

| Configuration | w/o shift | w/ shift (mean) | Notes |
|---|---|---|---|
| TabPFN-v2 (base) | 0.818 | 0.775 | Baseline |
| + EME (Saerens 2002, EM for test prior) | 0.801 | 0.786 | Drops 1.7pp w/o shift |
| + BBE (Lipton 2018, black-box test prior) | 0.805 | 0.789 | Drops 1.3pp w/o shift |
| + DistPFN | 0.818 | 0.789 | **No loss w/o shift** |
| **+ DistPFN-T** | **0.818** | **0.792** | **No loss w/o shift + largest gain w/ shift** |
| + DistPFN-Oracle (true $p_{test}(y)$) | 0.818 | 0.796 | Upper bound, only 0.4pp gap |
| TableShift Diabetes OOD | base 0.589 → DistPFN-T 0.600 | — | Real OOD also gains |
| TableShift Acsincome OOD | base 0.795 → DistPFN-T 0.799 | — | — |

### Key Findings
- **Greater shift, greater gain**: As train-test KL divergence increases, DistPFN-T yields monotonically increasing accuracy improvements per dataset, confirming its direct effectiveness against label shift rather than incidental regularization.
- **Approaches oracle**: Using true test prior (DistPFN-Oracle) achieves 78.4% at $\beta=5$; using predicted posterior (DistPFN-T) achieves 78.9%—the latter is **slightly higher** due to temperature scaling being smoother and less prone to overcorrection than "hard division by true $p_{test}$".
- **No loss in no-shift is the biggest selling point**: EME/BBE both drop 1–2pp w/o shift, making practitioners hesitant to enable them in deployment; DistPFN/DistPFN-T strictly preserve base performance w/o shift (since for $\beta=0$, $\hat{p}/p_{train} \approx 1$, so the adjustment factor is unity), making them safe to enable by default.
- **Single vs multiple instance nearly identical**: TabPFN supports both single-sample and batch modes; adjustment factors computed per-sample or as test-set averages yield nearly identical results, indicating robustness to implementation details.
- **84.6% of OpenML datasets are inherently imbalanced** (minority/majority ratio < 1.0), so even in $\beta=0$ "no shift" settings, the majority-class bias is a systemic issue for TabPFN-type models, not a corner case.

## Highlights & Insights
- **The observation that "training prior is explicitly visible" is the paper's fulcrum**—once it is clear that TabPFN encodes the entire training set in-context, $p_{train}(y)$ no longer needs to be estimated, bypassing the entire engineering effort of "test prior estimation" in classical label shift literature. Table 2 cleanly separates explicit/implicit models, making this "paradigm difference" perspective highly instructive.
- **The "partial correction" $\hat{p}^2/p_{train}$ is an engineeringly tasteful choice**: Full prior correction ($\hat{p}/p_{train}$) overcorrects in practice, as models do not truly encode the training distribution in $\hat{p}$; squaring the numerator retains model confidence, making the correction "gentle but not extreme".
- **DistPFN-T's use of $\tau = \mathrm{CE}(\hat{p}, p_{train})$ as temperature** is an elegant self-monitoring design: it relies solely on model outputs and known training priors, being fully self-contained.
- **The inverse-frequency oversample benchmark** is a reusable methodological contribution—future label shift evaluations can directly use this $\beta$ controller to draw continuous curves on fixed test sets, offering more controllability than real OOD data.

## Limitations & Future Work
- The method is theoretically designed for explicit-prior models (TabPFN family + kNN); although Table 2 claims it is "technically applicable to tree models," $p_{train}$ is less "explicit" there, and actual gains are not shown in the main results, suggesting less clean applicability.
- The paper acknowledges this is partial, not full, correction; even with temperature scaling, there remains a 0.4pp gap to oracle. Fully closing this gap may require some lightweight online test prior estimation, which would compromise the "zero training, zero estimation" appeal.
- Additional limitations: (1) Only classification is tested, not regression label distribution shift; (2) Adjustment occurs at the softmax output—if the model's logits are poorly calibrated (deep model overconfidence), the adjustment factor may be exaggerated; (3) No analysis for extreme class counts (e.g., 100+ classes, long-tail); experiments focus on moderate class counts (≤10) in tabular tasks; (4) Temperature $\tau$ computed via CE may overflow under extreme distributions, requiring numerical clamping in deployment.
- Potential improvements: Extend this idea to calibration (using priors to correct confidence), apply to RAG-LLM output distribution correction, or combine with conformal prediction for uncertainty-aware, shift-robust predictions.

## Related Work & Insights
- **vs EME (Saerens 2002) / BBE (Lipton 2018)**: Both require iterative test prior estimation and inevitably degrade performance w/o shift; this work does not estimate test priors, strictly preserves performance w/o shift, and delivers larger gains w/ shift—a clear advance in the same line.
- **vs Logit Adjustment / Balanced Softmax**: These require loss modification during training, unsuitable for frozen FMs like TabPFN-v2; this work is an inference-time plug-in, usable with any pretrained FM.
- **vs Drift-Resilient TabPFN (Helli 2024)**: That method targets temporal shift during pretraining and requires retraining; this work targets label shift via test-time adaptation, making the two orthogonal and combinable.
- **vs General TTA (test-time training)**: Most TTA methods require backpropagation on test samples, incurring high computational cost and risk; this work is pure forward pass + probability reweighting, with negligible extra computation.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation that "TabPFN explicitly encodes training priors" + posterior/prior ratio + temperature scaling is a simple yet sharp combination, previously unexplored.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 253 OpenML datasets × 6 $\beta$ × 5 seeds × 3 FMs × real TableShift OOD × EME/BBE comparison × oracle upper bound—extremely dense evaluation.
- Writing Quality: ⭐⭐⭐⭐ Table 2's explicit/implicit split and Table 3's method comparison are clear; the derivation is logically smooth.
- Value: ⭐⭐⭐⭐ One-line plug-in, directly deployable on any TabPFN-v2/LoCalPFN/TabICL, high industrial value, and opens up the research direction of "leveraging explicit priors in FMs".

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TabMGP: Martingale Posterior with TabPFN](tabmgp_martingale_posterior_with_tabpfn.md)
- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](../../CVPR2026/others/mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](private_and_stable_test-time_adaptation_with_differential_privacy.md)

</div>

<!-- RELATED:END -->
