---
title: >-
  [Paper Note] Foster Adaptivity and Balance in Learning with Noisy Labels
description: >-
  [ECCV 2024][noisy labels] Proposes the SED method, which addresses the label noise problem through an adaptive and class-balanced sample selection and re-weighting mechanism. It achieves SOTA performance on both synthetic and real-world noisy datasets without requiring prior knowledge such as predefined thresholds.
tags:
  - "ECCV 2024"
  - "noisy labels"
  - "sample selection"
  - "sample re-weighting"
  - "class balance"
  - "self-adaptive"
date: 2026-05-08
content_hash: e5bc3ebd52f80817
---

# Foster Adaptivity and Balance in Learning with Noisy Labels

**Conference**: ECCV 2024  
**arXiv**: [2407.02778](https://arxiv.org/abs/2407.02778)  
**Code**: [GitHub](https://github.com/NUST-Machine-Intelligence-Laboratory/SED)  
**Area**: Others  
**Keywords**: noisy labels, sample selection, sample re-weighting, class balance, self-adaptive

## TL;DR
Proposes the SED method, which addresses the label noise problem through an adaptive and class-balanced sample selection and re-weighting mechanism. It achieves SOTA performance on both synthetic and real-world noisy datasets without requiring prior knowledge such as predefined thresholds.

## Background & Motivation

**Background**: Deep neural networks have achieved remarkable success in tasks such as image classification, but they rely heavily on large-scale, high-quality annotated data. In practice, annotations obtained through crowdsourcing or web-scraping often contain noisy labels, which severely hurts the generalization performance of models.

**Two paradigms of existing methods**:
   - **Label Correction**: Rectifies incorrect labels using noise transition matrices or model predictions. However, transition matrices are difficult to estimate accurately, and prediction-based methods are prone to imbalanced label correction (samples are more easily corrected to simpler classes).
   - **Sample Selection/Re-weighting**: Treats small-loss samples as clean samples. However, this line of work requires prior knowledge such as predefined drop rates or thresholds, and neglects the class balance issue.

**Key Challenge**:
   - Existing sample selection methods highly depend on dataset-specific prior knowledge (e.g., predefined thresholds), making them difficult to generalize to different datasets.
   - Most methods overlook the class balance issue, causing the model to bias towards easier classes.

**Key Insight**: Designs a unified framework that remains free of prior knowledge while maintaining class balance, integrating three strategies: sample selection, label correction, and sample re-weighting.

**Core Idea**: Achieves class-balanced sample selection via a global-and-local adaptive threshold mechanism, followed by adaptive re-weighting using a truncated normal distribution.

## Method

### Overall Architecture
The SED framework consists of four stages: (1) partitioning the training set into a clean subset $D_c$ and a noisy subset $D_n$ based on global and local thresholds; (2) generating corrected labels for noisy samples using a mean-teacher model; (3) adaptively assigning weights to noisy samples based on correction confidence; (4) applying a consistency regularization loss on clean samples to enhance model robustness. The final loss is formulated as $\mathcal{L} = \mathcal{L}_{D_c} + \mathcal{L}_{D_n} + \mathcal{L}_{reg}$.

### Key Designs

1. **Self-adaptive Class-balanced Sample Selection (SCS)**:

    - **Function**: Determines whether a sample is clean based on its prediction probability $p^{y_i}(x_i, \theta)$ on the given label.
    - **Mechanism**: Designs a combination of global and local thresholds. The global threshold is dynamically updated via EMA:
    $T_t = m T_{t-1} + (1-m) \frac{1}{N} \sum_{i=1}^{N} p^{y_i}(x_i, \theta)$
      The local threshold reflects the learning status of each class, obtained by normalizing the expected class predictions:
    $\tilde{T}_t(c) = \frac{\tilde{E}_t(c)}{\max\{\tilde{E}_t(c : c \in [C])\}} T_t$
      where $\tilde{E}_t(c)$ is the EMA prediction expectation of class $c$. The initial global threshold is set to $T_0 = 1/C$, making the entire process completely data-driven without requiring prior knowledge.
    - **Design Motivation**: The global threshold ensures sufficient clean samples are identified and increases naturally as training progresses (consistent with the memorization effect). The local threshold establishes different criteria for classes of varying difficulty, preventing simple classes from dominating the selection results.

2. **Mean-Teacher Label Correction**:

    - **Function**: Generates reliable pseudo-labels for samples identified as noisy.
    - **Mechanism**: Employs an exponential moving average (EMA) model $\theta^*$ to generate pseudo-labels:
    $\theta_{t'}^* = \alpha \theta_{t'-1}^* + (1-\alpha) \theta_{t'}$
    $y_i^{corr} = \arg\max_{j=1,...,C} p^j(x_i, \theta^*)$
    - **Design Motivation**: Introducing historical model information enhances the reliability of label correction and alleviates error propagation.

3. **Self-adaptive Class-balanced Sample Re-weighting (SCR)**:

    - **Function**: Allocates different weights to noisy samples according to their label correction confidence.
    - **Mechanism**: Assumes the sample weights follow a dynamic truncated normal distribution and calculates weights based on the deviation of correction confidence from the mean:
    $\lambda(x_i) = \begin{cases} \lambda_m \exp\left(\frac{(p^{y_i^{corr}}(x_i, \theta) - \mu_t)^2}{-2\sigma_t^2}\right), & p^{y_i^{corr}} < \mu_t \\ \lambda_m, & \text{otherwise} \end{cases}$
      where $\mu_t(c)$ and $\sigma_t^2(c)$ are estimated class-wise and updated through EMA.
    - **Design Motivation**: Highly confident samples are less likely to be incorrectly corrected, thus receiving larger weights; estimating distribution parameters on a per-class basis mitigates the class imbalance problem in label correction.

4. **Consistency Regularization (CR)**:

    - **Function**: Computes an additional weighted classification loss on clean samples.
    - **Mechanism**: Evaluates the loss on strongly augmented views of clean samples using corrected labels and adaptive weights:
    $\mathcal{L}_{reg} = -\frac{1}{|D_c|} \sum_{(x,y) \in D_c} \lambda(x) y^{corr} \log p(\hat{x}, \theta)$
    - **Design Motivation**: Implicitly encourages prediction consistency between weakly and strongly augmented views, further enhancing model robustness.

### Loss & Training

The final loss function consists of three components:
- Classification loss $\mathcal{L}_{D_c}$ on the clean subset (using original given labels)
- Weighted corrected classification loss $\mathcal{L}_{D_n}$ on the noisy subset (using corrected labels + adaptive weights)
- Consistency regularization loss $\mathcal{L}_{reg}$ on the clean subset (using corrected labels + adaptive weights + strong augmentation)

**Training Strategy**: SGD optimizer (momentum=0.9), EMA coefficients $m=0.99$, $\alpha=0.95$, and $\lambda_m=1.0$. The training process includes 20 warm-up epochs, totaling 100 epochs.

## Key Experimental Results

### Main Results

| Dataset | Noise Type | Ours(SED) | Prev. SOTA | Gain |
|--------|----------|-----------|----------|------|
| CIFAR100N | Sym-20% | **66.50** | 60.28 (DISC) | +6.22 |
| CIFAR100N | Sym-80% | **38.15** | 35.23 (NCE) | +2.92 |
| CIFAR100N | Asym-40% | **58.29** | 52.28 (Co-LDL) | +6.01 |
| CIFAR80N | Sym-20% | **69.10** | 65.83 (Jo-SRC) | +3.27 |
| CIFAR80N | Sym-80% | **42.57** | 39.34 (NCE) | +3.23 |
| CIFAR80N | Asym-40% | **60.87** | 56.40 (NCE) | +4.47 |
| Web-Aircraft | Real Noise | **86.62** | 85.27 (DISC) | +1.35 |
| Web-Bird | Real Noise | **82.00** | 81.20 (UNICON) | +0.80 |
| Web-Car | Real Noise | **88.88** | 88.31 (DISC) | +0.57 |

### Ablation Study

| Configuration | Key Metric (Acc%) | Description |
|------|----------------|------|
| Standard (baseline) | 34.10 | No processing |
| +SCS w/o local threshold | 53.36 | Global threshold only |
| +SCS w/o global threshold | 55.64 | Local threshold only |
| +SCS (full) | 58.21 | Global + local threshold |
| +SCS+label correction+SCR | Higher | Re-weighting mechanism added |
| +SCS+label correction+SCR+CR | Highest | Complete SED |

### Key Findings

- SED demonstrates the most significant advantage under heavy noise (Sym-80%) and challenging noise (Asym-40%) scenarios.
- On the CIFAR80N dataset, which contains both open-set and closed-set noise, SED still exhibits strong robustness.
- The class precision comparison plot for sample selection shows that SCS achieves a more balanced performance compared to the small-loss and GMM selection strategies.
- Without incorporating extra techniques like Mixup or dual-network co-training, the single network of SED outperforms methods utilizing these techniques.

## Highlights & Insights

- **Completely self-adaptive**: The global threshold is initialized to $1/C$ and updated via EMA, discarding any dataset-dependent prior parameter tuning.
- **Implicitly aligns with the memorization effect**: The global threshold increases naturally as training progresses, allowing the learning of more samples in the early phase and applying stricter selection in the later phase.
- **Truncated normal distribution re-weighting**: Sweeter and smoother than hard 0/1 selection or simple linear weighting, automatically tightening as training progresses ($\mu$ increases while $\sigma$ decreases).
- **Unity of three strategies**: The integration of selection, correction, and re-weighting is more effective than any single strategy alone.

## Limitations & Future Work

- Validated only on classification tasks; downstream tasks such as detection or segmentation were not considered.
- Evaluated mainly using a 7-layer CNN as the backbone, without validation on larger architectures (e.g., ViT).
- EMA coefficients $m$ and $\alpha$ still require manual configuration.
- Instance-dependent noise scenarios were not taken into consideration.

## Related Work & Insights

- **vs Co-teaching**: Co-teaching requires a predefined drop rate and dual-network cross-training, whereas SED's single-network design with an adaptive threshold is more concise and efficient.
- **vs DivideMix**: DivideMix relies on a GMM for selection and MixMatch for semi-supervised learning. In contrast, SED utilizes a data-driven threshold and truncated normal distribution weighting, bypassing GMM's Gaussian mixture assumption.
- **vs DISC**: DISC selects samples based on memorization strength, while SED operates on prediction probability and class-balanced thresholds, delivering superior performance in most scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of global-and-local thresholds and truncated normal distribution re-weighting is innovative, although the overall paradigm (selection + correction + re-weighting) is relatively classical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of synthetic and real-world noise datasets, extensive ablation studies, and rich visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with complete mathematical derivations, though some formulas involve heavy notations and require close reading.
- Value: ⭐⭐⭐⭐ High practical value due to its prior-free and effective nature; however, it is currently limited to small-scale classification experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Debiased Sample Selection for Learning with Noisy Labels](../../CVPR2026/others/debiased_sample_selection_for_learning_with_noisy_labels.md)
- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICLR 2026\] Learning in Prophet Inequalities with Noisy Observations](../../ICLR2026/others/learning_in_prophet_inequalities_with_noisy_observations.md)
- [\[ECCV 2024\] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion](hpff_hierarchical_locally_supervised_learning_with_patch_feature_fusion.md)
- [\[ECCV 2024\] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)

</div>

<!-- RELATED:END -->
