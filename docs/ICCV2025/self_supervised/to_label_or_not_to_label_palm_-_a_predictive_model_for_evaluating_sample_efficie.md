---
title: >-
  [Paper Note] To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models
description: >-
  [ICCV 2025][Self-Supervised Learning][Active Learning] This paper proposes PALM — a unified mathematical model that characterizes active learning trajectories using four interpretable parameters (maximum accuracy $A_{\ma…
tags:
  - "ICCV 2025"
  - "Self-Supervised Learning"
  - "Active Learning"
  - "Sample Efficiency"
  - "Learning Curve Prediction"
  - "Coverage Efficiency"
  - "Self-Supervised Representations"
date: 2026-05-08
content_hash: e48b633c30c2c240
---

# To Label or Not to Label: PALM – A Predictive Model for Evaluating Sample Efficiency in Active Learning Models

**Conference**: ICCV 2025
**arXiv**: [2507.15381](https://arxiv.org/abs/2507.15381)  
**Code**: [github.com/juliamachnio/PALM](https://github.com/juliamachnio/PALM)  
**Area**: Self-Supervised Learning / Active Learning
**Keywords**: Active Learning, Sample Efficiency, Learning Curve Prediction, Coverage Efficiency, Self-Supervised Representations

## TL;DR

This paper proposes PALM — a unified mathematical model that characterizes active learning trajectories using four interpretable parameters (maximum accuracy $A_{\max}$, coverage efficiency $\delta$, initial learning offset $\alpha$, and scalability $\beta$). The model predicts complete learning curves from limited labeled data, enabling quantitative and fair comparison of active learning strategies.

## Background & Motivation

### Core Problem

Active Learning (AL) reduces annotation costs by iteratively selecting the most informative samples for labeling. However, existing AL evaluation methods focus solely on final accuracy under a fixed labeling budget, neglecting the full picture of the learning process — including early-stage learning efficiency, spatial coverage, and the scalability of learning gains.

### Limitations of Prior Work

**Lack of predictive evaluation**: Traditional methods compare accuracy and AUC only at fixed budgets, providing no means to predict future performance or estimate the annotation volume required to reach a target accuracy.

**Absence of a unified model**: No general predictive framework exists for evaluating and comparing AL methods across different strategies, datasets, and annotation budgets.

**Neglect of learning dynamics**: Existing protocols do not account for behavioral differences between strategies at early vs. late stages of learning, nor do they quantify the impact of self-supervised representations on AL efficiency.

### Core Idea

Drawing on the probabilistic principles of the random coverage problem, the AL process is modeled as a coverage process over the data space — each labeled sample covers a proportion $\delta$ of the space, and as the number of labeled samples increases, the coverage grows exponentially toward 1. Parameters $\alpha$ (initial offset) and $\beta$ (scalability) are introduced to enhance model flexibility.

## Method

### Overall Architecture

PALM is a purely mathematical model (not a neural network). It fits four parameters from partial observations of AL experiments via nonlinear least squares, then uses the fitted model to predict complete learning curves and compare different AL strategies.

### Key Designs

#### 1. **Expected Coverage Formula**
- **Function**: Models the expected proportion of the data space covered by $B$ labeled samples.
- **Mechanism**: Assuming each labeled sample independently covers a proportion $\delta$ of the space, the expected coverage of $B$ samples is:

$$\mathbb{E}_C = 1 - (1-\delta)^B$$

- **Design Motivation**: A larger $\delta$ indicates higher per-sample coverage efficiency; differences in $\delta$ across AL strategies directly reflect their sampling quality.

#### 2. **Generalized Accuracy Function (Core PALM Formula)**
- **Function**: Expresses test accuracy as a function of the annotation budget.
- **Mechanism**:

$$A = A_{\max}\left(1 - (1-\delta)^{\left(\frac{B}{b}+\alpha\right)^\beta}\right)$$

where:
  - $A_{\max}$: maximum achievable accuracy (asymptotic upper bound)
  - $\delta$: coverage efficiency (proportion of the space covered per labeled sample)
  - $b$: average number of labels per iteration (normalization factor)
  - $\alpha$: initial learning offset (reflects the contribution of uncovered regions to generalization)
  - $\beta$: scalability of learning gain (controls the rate at which accuracy grows with annotation budget)
- **Design Motivation**: The base coverage model cannot distinguish early-stage learning behavior or scaling characteristics across methods; the introduction of $\alpha$ and $\beta$ enables the model to capture the "inflection point" and "plateau" behavior of AL curves.

#### 3. **Strategy Comparison Criterion**
- **Function**: Directly compares the relative merit of two AL methods based on fitted parameters.
- **Mechanism**: Given budget $B$, comparing $A_1$ and $A_2$ suffices. At a finer granularity:
    - High $\delta$ → better sampling efficiency
    - Low $\alpha$ → faster initial learning
    - High $\beta$ → better late-stage scalability
    - High $A_{\max}$ → higher asymptotic performance
- **Design Motivation**: Provides a quantifiable and interpretable comparison framework that is independent of dataset and initialization conditions.

### Loss & Training

- Parameters are fitted via nonlinear least squares with computational complexity approximately $\mathcal{O}(\log(B))$.
- Constraints are imposed on parameters to ensure stable and interpretable fitting.

## Key Experimental Results

### Main Results (CIFAR-10/100 AL Curve Fitting)

| Dataset | Method | $\delta$ | $\alpha$ | $\beta$ | Note |
|-------|------|---------|---------|--------|------|
| CIFAR-10 | Margin (no embedding) | 0.094 | - | - | Low coverage efficiency |
| CIFAR-10 | Margin (SimCLR embedding) | 0.535 | - | - | Embedding significantly boosts $\delta$ |
| CIFAR-100 | Random (no embedding) | 0.048 | - | - | Low efficiency |
| CIFAR-100 | Random (SimCLR embedding) | 0.318 | - | - | 5.7× coverage improvement |
| CIFAR-100 | Margin (no embedding) | - | 10.643 | - | High $\alpha$ = delayed learning |
| CIFAR-100 | Margin (SimCLR embedding) | - | 0.068 | - | Embedding substantially reduces $\alpha$ |

### Predictive Capability Validation (Predicting Full Curves from Partial Data)

| Dataset | Min. Labels Required | Fraction of Dataset | Prediction Quality |
|-------|-----------|-------------|---------|
| CIFAR-10 | 1,000 samples | 2% | Accurately predicts full dynamics |
| CIFAR-10 (best case) | 200–300 samples | 0.4–0.6% | Reasonable prediction |
| CIFAR-100 | 5,000–10,000 samples | 10–20% | Reliable prediction |
| CIFAR-100 + TypiClust | 1,000 samples | 2% | Accurate prediction |

### ImageNet Ablation (PALM Parameters Across Different SSL Embeddings)

| Embedding | Relative $A_{\max}$ | $\delta$ Level | Characteristics |
|---------|-------------------|--------------|------|
| MoCov3 | Highest | Highest | Best representations, fastest learning |
| SimCLR | Competitive | Moderate | Sensitive to AL strategy |
| BYOL | Low | Very low | Slow learning, near-linear growth |
| MoCov2+ | Weak | Weak | Generally inferior |

### Key Findings

1. **SSL embeddings have a profound impact on AL efficiency**: Methods using pretrained embeddings achieve $\delta$ values up to 5–6× higher than those without embeddings.
2. **PALM can predict the full learning trajectory from very limited data**: On CIFAR-10, as little as 2% of the data suffices to predict the complete learning curve.
3. **Methods with similar final accuracy may exhibit dramatically different learning trajectories**: PALM's parameterization reveals differences that are obscured by endpoint-only evaluation.
4. **Prediction error remains within 2%**: Even during late stages when the labeled pool approaches exhaustion.

## Highlights & Insights

1. **Elegant mathematical formulation**: A closed-form expression is derived from the random coverage problem with clear physical interpretation; each of the four parameters carries well-defined semantics.
2. **Practically useful predictive capability**: In real-world settings, only a small number of samples need to be labeled to assess the full behavioral profile of different AL strategies, substantially reducing annotation costs.
3. **Reveals interaction patterns between SSL and AL**: Performance differences across SSL methods (MoCov3 > SimCLR > MoCov2+ > BYOL) in AL settings can be precisely characterized via PALM parameters.
4. **Methodological contribution**: Advances AL evaluation from "point evaluation" to "curve evaluation," introducing a new evaluation paradigm.

## Limitations & Future Work

1. **Assumes labeled samples are independent**: In practice, AL involves sequential decision-making with dependencies between selected samples.
2. **Assumes an infinite unlabeled pool**: When the labeled pool approaches exhaustion, minor fitting deviations arise.
3. **Validated only on classification tasks**: Extension to detection, segmentation, and other vision tasks has not been explored.
4. **Inflection point issue on CIFAR-10**: Sharp transitions in learning curves on simpler datasets are difficult to fit precisely.
5. **Annotation quality and noisy labels not considered**: Real-world annotation quality is often inconsistent.

## Related Work & Insights

- TypiClust is an AL method targeting small-budget regimes and demonstrates extremely high annotation efficiency within the PALM framework.
- Quality differences among SSL methods (SimCLR, BYOL, MoCov2+, MoCov3) are amplified in the context of AL evaluation.
- PALM's mathematical framework can be viewed as analogous to a "scaling law" for AL — describing complex learning behavior through a concise parameterized model.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First closed-form predictive model for AL learning curves, with elegant theoretical derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers CIFAR-10/100 and ImageNet subsets across diverse AL strategies and SSL methods, though downstream task validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and the logical flow is clear.
- **Value**: ⭐⭐⭐⭐ — Provides the AL community with a unified evaluation tool and offers practical guidance for annotation budget planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](../../NeurIPS2025/self_supervised/hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[ICML 2026\] Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](../../ICML2026/self_supervised/mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[ICCV 2025\] A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)](a_tokenlevel_text_image_foundation_model_for_document_unders.md)
- [\[ICCV 2025\] Improving Large Vision and Language Models by Learning from a Panel of Peers](improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)

</div>

<!-- RELATED:END -->
