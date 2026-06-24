---
title: >-
  [Paper Note] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score
description: >-
  [CVPR 2026][Optimization][Conformal prediction] The SemiCP framework is proposed to integrate unlabeled data into the conformal prediction calibration process via Nearest Neighbor Matching (NNM) scores. This reduces the average coverage gap by up to 77% when labeled data is extremely scarce, while simultaneously shrinking the size of prediction sets.
tags:
  - "CVPR 2026"
  - "Optimization"
  - "Conformal prediction"
  - "semi-supervised learning"
  - "uncertainty quantification"
  - "prediction sets"
  - "nearest neighbor matching"
date: 2026-05-08
content_hash: 2bf0199c42ab982c
---

# Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score

**Conference**: CVPR 2026  
**arXiv**: [2505.21147](https://arxiv.org/abs/2505.21147)  
**Code**: Yes (Integrated into TorchCP library)  
**Area**: Optimization  
**Keywords**: Conformal prediction, semi-supervised learning, uncertainty quantification, prediction sets, nearest neighbor matching

## TL;DR

The SemiCP framework is proposed to integrate unlabeled data into the conformal prediction calibration process via Nearest Neighbor Matching (NNM) scores. This reduces the average coverage gap by up to 77% when labeled data is extremely scarce, while simultaneously shrinking the size of prediction sets.

## Background & Motivation

**Core Value of Conformal Prediction (CP)**: CP is a model-agnostic, distribution-free framework for uncertainty quantification. It generates prediction sets containing the true label with finite-sample coverage guarantees, which is critical in high-risk scenarios such as medical diagnosis and financial decision-making.

**Dependency of Split CP on Labeled Data**: Standard Split CP requires a labeled hold-out calibration set to estimate thresholds. However, labeled data is often scarce in practice, leading to highly unstable coverage across different runs.

**Theoretical Flaws of Small Calibration Sets**: When the calibration set size $n$ is small, the coverage follows a Beta distribution with a variance of approximately $\alpha(1-\alpha)/(n+2)$. For instance, with $n=10, \alpha=0.1$, there is a 10.7% probability that the actual coverage falls below 80%.

**Limitations of Prior Work**: Interpolation and p-value modification methods are heuristic and lack finite-sample guarantees. Few-shot CP relies on sets of exchangeable tasks, which limits practical utility.

**Richness of Unlabeled Data**: In many scenarios, the volume of unlabeled data far exceeds labeled data, representing a natural resource. However, prior work has not utilized this for CP calibration.

**Key Challenge in Conditional Coverage**: In class-conditional or group-conditional coverage scenarios, each subgroup requires independent calibration data. For example, ImageNet with 1000 classes and 100 samples per class would require $10^5$ labeled points, far exceeding practical availability.

## Method

### Overall Architecture

SemiCP expands the calibration set to $\mathcal{D} = \mathcal{D}_{\text{labeled}} \cup \mathcal{D}_{\text{unlabeled}}$, with $n$ labeled samples and $N$ unlabeled samples. Standard nonconformity scores $s_i = S(\mathbf{x}_i, y_i)$ (e.g., THR/APS/RAPS) are used for labeled data, while specially designed unlabeled scores $\tilde{s}_i = \tilde{S}(\tilde{\mathbf{x}}_i)$ are used for unlabeled data. The combined scores are used to calculate the quantile threshold:

$$\hat{\tau}_{\text{SemiCP}} = \text{Quantile}\left(\{\tilde{s}_i\}_{i=1}^N \cup \{s_i\}_{i=1}^n, \frac{\lceil(n+N+1)(1-\alpha)\rceil}{n+N}\right)$$

At test time, the prediction set for a test sample $\mathbf{x}_{\text{test}}$ is constructed as $\mathcal{C}(\mathbf{x}_{\text{test}}) = \{y : S(\mathbf{x}_{\text{test}}, y) \le \hat{\tau}_{\text{SemiCP}}\}$. The pipeline merges the "labeled path" (direct standard scores) and the "unlabeled path" (NNM-corrected scores) into a single calibration pool to estimate the threshold:

```mermaid
mermaid
flowchart TD
    L["Labeled Calibration Set (n samples, with ground truth)"] --> LS["Standard Nonconformity Score S<br/>THR / APS / RAPS"]
    U["Unlabeled Calibration Set (N samples)"] --> P["Pseudo-label ŷ = argmax softmax"]
    P --> PS["Pseudo-scores S(·,ŷ) are systematically low<br/>Define Pseudo-bias Δ"]
    PS --> NNM["NNM Score: Match nearest labeled samples in pseudo-score space<br/>Correct using their observable real bias"]
    LS --> MERGE["SemiCP Framework: Merge n+N scores<br/>Calculate quantile for threshold"]
    NNM --> MERGE
    MERGE --> C["Construct Test Prediction Set<br/>Retain labels with scores ≤ threshold"]
```

### Key Designs

**1. SemiCP Framework: Merging Unlabeled Scores to Balance Variance Reduction and Bias**

Standard Split CP estimates thresholds using only $n$ labeled scores; a small $n$ leads to high coverage variance. SemiCP aggregates $N$ unlabeled scores and $n$ labeled scores into a unified calibration pool of size $n+N$ (see threshold formula). Theorem 1 provides a coverage lower bound $1-\alpha+\epsilon_{n,N}$, and Theorem 2 proves that the average coverage gap shrinks at a rate of $\mathcal{O}(1/\sqrt{N})$. The cost is a bias term $\epsilon_{n,N}=\frac{N}{N+n}\big(F_S(\hat{\tau})-F_{\tilde{S}}(\hat{\tau})\big)$, determined by the difference between the estimated and true score CDFs. This defines the goal for unlabeled scoring: as the unlabeled score distribution approaches the true distribution ($\epsilon_{n,N}\to 0$), the framework achieves variance reduction without introducing significant bias.

**2. Pseudo-bias: Why Naive Pseudo-label Scores are Systematically Low**

The most intuitive unlabeled scoring method assigns pseudo-labels $\hat{y}_i = \arg\max_j f_j(\tilde{\mathbf{x}}_i)$ and computes scores (the naive method). Since pseudo-labels coincide with the model's most confident predictions, the resulting scores are systematically low, underestimating the threshold and leading to under-coverage. This increases the bias term $\epsilon_{n,N}$. To address this, the bias is quantified as the difference between the true score and the pseudo-score: $\Delta(\tilde{\mathbf{x}}_i) = S(\tilde{\mathbf{x}}_i, \tilde{y}_i) - S(\tilde{\mathbf{x}}_i, \hat{y}_i)$. Since $\tilde{y}_i$ is unknown, NNM is used for estimation.

**3. Nearest Neighbor Matching (NNM) Score: Correcting Pseudo-scores via Labeled Bias**

The bias cannot be calculated directly for unlabeled samples. NNM finds the labeled sample $j$ whose pseudo-score is closest to that of the unlabeled sample in the pseudo-score space: $j = \arg\min_{j \in \{1,...,n\}} |S(\tilde{\mathbf{x}}_i, \hat{y}_i) - S(\mathbf{x}_j, \hat{y}_j)|$. The observable real bias of this labeled sample is then used to correct the unlabeled pseudo-score: $\tilde{S}_{\text{nnm}}(\tilde{\mathbf{x}}_i) = S(\tilde{\mathbf{x}}_i, \hat{y}_i) + S(\mathbf{x}_j, y_j) - S(\mathbf{x}_j, \hat{y}_j)$. The Key Insight is that "samples with similar pseudo-scores have similar pseudo-biases." Theorem 3 proves that as $n$ increases, the NNM score CDF asymptotically converges to the true score CDF. Empirical results (Fig. 3) show the NNM score PDF aligns closely with the true distribution, effectively suppressing $\epsilon_{n,N}$.

### Loss & Training

This method is **training-free** and requires no additional optimization. Specifically:

- It utilizes the softmax output of pre-trained classifiers without accessing the original training data.
- It is compatible with any nonconformity score function (THR, APS, RAPS, SAPS, etc.).
- it can be seamlessly integrated into conditional coverage (class-conditional, group-conditional) settings.
- It can be combined with existing methods like Interpolation or ClusterCP.

The theoretical guarantees (coverage lower bounds and convergence rates) ensure reliability.

## Key Experimental Results

### Main Results

| Dataset | Labeled $n$ | Unlabeled $N$ | Method | CovGap ↓ | AvgSize ↓ |
|---------|-----------|-------------|------|----------|-----------|
| CIFAR-10 | 20 | 4000 | Standard | 4.8 | 1.45 |
| CIFAR-10 | 20 | 4000 | **SemiCP** | **1.1** | **1.37** |
| CIFAR-10 | 10 | 4000 | Standard | 6.4 | 1.60 |
| CIFAR-10 | 10 | 4000 | **SemiCP** | **1.1** | **1.27** |
| ImageNet | 50 | 20000 | Standard | ~3.3 | ~75 |
| ImageNet | 50 | 20000 | **SemiCP** | **~2.1** | **~70.3** |

| Setting | Dataset | $n_{\text{avg}}$ | Method | CovGap ↓ | AvgSize ↓ |
|------|--------|-----------------|------|----------|-----------|
| Class-Cond | CIFAR-100 | 10 | Standard | 7.75 | 18.9 |
| Class-Cond | CIFAR-100 | 10 | **SemiCP** | **6.29** | **17.0** |
| Group-Cond | CIFAR-100 | 10 | Standard | High | Large |
| Group-Cond | CIFAR-100 | 10 | **SemiCP** | Significant Reduction | Significant Shrinkage |

### Ablation Study

- **Impact of Unlabeled Data Volume** (ImageNet, $n=50$): As $N$ grows from 10 to 20,000, CovGap and AvgSize consistently decrease. Even with $N=10$ unlabeled samples, CovGap and AvgSize show measurable improvement.
- **Combination with Interpolation** (CIFAR-100): While Interpolation alone can be unstable, SemiCP+Interpolation reduces CovGap from 9 to 3.9 at $n=10$. For $n>40$, AvgSize approaches the Oracle.
- **Combination with ClusterCP**: SemiCP+ClusterCP further reduces CovGap and AvgSize across all $n_{\text{avg}}$ levels.
- **NNM vs. Naive**: Naive pseudo-label scores show a systematically low PDF, while NNM scores match the true score distribution.

### Key Findings

1. **Coverage Gap Reduction up to 77%**: On CIFAR-10 (20 labeled + 4000 unlabeled), CovGap fell from 4.8 to 1.1, with a 5.7% reduction in AvgSize.
2. **Robustness Across Architectures**: Effective across 10 architectures (ResNet, MobileNet, EfficientNet, ViT, etc.), reducing average CovGap from 3.3 to 2.1.
3. **Greater Gains in Conditional Coverage**: Improvements in class-conditional and group-conditional settings exceed those in marginal coverage.
4. **Data Efficiency**: Significant improvements are observed even with minimal unlabeled data ($N=10$).

## Highlights & Insights

- **Training-free Design**: Plug-and-play implementation leveraging pre-trained model outputs without additional training, unlike concurrent works requiring $N \times K$ matrix optimization.
- **Mechanism of NNM**: Estimating bias via nearest neighbor matching in the pseudo-score space is an elegant exploitation of the empirical observation that similar pseudo-scores imply similar biases.
- **Theory-Experiment Synergy**: Theorems 1-3 provide a rigorous foundation for coverage guarantees and score consistency, which are closely reflected in experimental results.
- **Extensive Compatibility**: Acts as a modular plugin for THR/APS/RAPS, Interpolation, ClusterCP, and various conditional CP methods.

## Limitations & Future Work

- Theoretical analysis relies on the i.i.d. assumption, which is stricter than the standard exchangeability assumption in CP.
- Currently validated only for classification tasks; extension to regression is pending.
- NNM matching precision may degrade when labeled data is extremely sparse (e.g., $n < 5$).
- The assumption of similar pseudo-biases for similar pseudo-scores may not hold under significant distribution shifts.

## Related Work & Insights

- **Split CP Variants**: THR [Sadinle+ 2019], APS [Romano+ 2020], RAPS [Angelopoulos+ 2020], and SAPS [Huang+ 2023] focus on score design for efficiency but depend on sufficient labeled data.
- **Small Calibration Set Handling**: Interpolation [Johansson+ 2015], ClusterCP [Ding+ 2023], and Few-shot CP [Fisch+ 2021] address small $n$ but do not utilize unlabeled samples.
- **Prediction-Powered Inference (PPI)**: [Angelopoulos+ 2023] uses model predictions to tighten confidence intervals for self-supervised inference rather than CP calibration.
- **Unsupervised Calibration**: [Mazuelas 2025] minimizes IPM to estimate label weights but requires expensive $N \times K$ matrix optimization.
- **Position of SemiCP**: The first work to utilize unlabeled data to estimate nonconformity scores and improve CP calibration stability.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce semi-supervised learning to CP calibration with a theoretically sound NNM score design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely comprehensive evaluation across 3 datasets, 3 score functions, 10 architectures, and 1000 repetitions.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete derivations, and systematic presentation of results.
- Value: ⭐⭐⭐⭐ — High practical utility as a training-free plugin, though currently limited to classification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/optimization/conformal_prediction_as_bayesian_quadrature.md)
- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](../../ICML2025/optimization/on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](../../NeurIPS2025/optimization/conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/optimization/one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](../../CVPR2025/optimization/conformal_prediction_for_zero-shot_models.md)

</div>

<!-- RELATED:END -->
