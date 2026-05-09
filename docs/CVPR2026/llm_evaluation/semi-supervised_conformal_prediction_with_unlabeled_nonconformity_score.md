---
title: >-
  [Paper Note] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score
description: >-
  [CVPR 2026][LLM Evaluation][Conformal Prediction] This paper proposes SemiCP, a framework that incorporates unlabeled data into the conformal prediction calibration pipeline via a Nearest Neighbor Matching (NNM) score. Under extremely limited labeled data, SemiCP reduces the average coverage gap by up to 77% while simultaneously shrinking prediction set sizes.
tags:
  - CVPR 2026
  - LLM Evaluation
  - Conformal Prediction
  - Semi-Supervised Learning
  - Uncertainty Quantification
  - Prediction Sets
  - Nearest Neighbor Matching
date: 2026-05-08
content_hash: c52da716e8abe486
---

# Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score

**Conference**: CVPR 2026
**arXiv**: [2505.21147](https://arxiv.org/abs/2505.21147)
**Code**: Available (integrated into TorchCP library)
**Area**: LLM Evaluation
**Keywords**: Conformal Prediction, Semi-Supervised Learning, Uncertainty Quantification, Prediction Sets, Nearest Neighbor Matching

## TL;DR

This paper proposes SemiCP, a framework that incorporates unlabeled data into the conformal prediction calibration pipeline via a Nearest Neighbor Matching (NNM) score. Under extremely limited labeled data, SemiCP reduces the average coverage gap by up to 77% while simultaneously shrinking prediction set sizes.

## Background & Motivation

**Core value of Conformal Prediction (CP)**: CP is a model-agnostic, distribution-free uncertainty quantification framework that produces prediction sets with finite-sample coverage guarantees, making it critical in high-stakes applications such as medical diagnosis and financial decision-making.

**Dependence of Split CP on labeled data**: Standard Split CP requires a labeled hold-out calibration set to estimate the threshold. In practice, labeled data is often scarce, causing coverage rates to be highly unstable across different runs.

**Theoretical deficiency of small calibration sets**: When calibration set size $n$ is small, the coverage follows a Beta distribution with variance approximately $\alpha(1-\alpha)/(n+2)$; for example, with $n=10$ and $\alpha=0.1$, there is a 10.7% probability that coverage falls below 80%.

**Limitations of Prior Work**: Interpolation-based methods and modified p-value approaches are heuristic and lack finite-sample guarantees; Few-shot CP relies on exchangeable task collections, limiting its practicality.

**Abundance of unlabeled data**: In many settings, unlabeled data vastly outnumbers labeled data and constitutes a natural resource to exploit, yet no prior work has leveraged it for CP calibration.

**Greater challenge under conditional coverage**: Under class-conditional or group-conditional coverage settings, each subgroup requires independent calibration data. For instance, calibrating ImageNet across 1,000 classes with 100 samples per class would require $10^5$ labeled points—far beyond what is practically obtainable.

## Method

### Overall Architecture

SemiCP extends the calibration set to $\mathcal{D} = \mathcal{D}_{\text{labeled}} \cup \mathcal{D}_{\text{unlabeled}}$, comprising $n$ labeled and $N$ unlabeled samples. Standard nonconformity scores $s_i = S(\mathbf{x}_i, y_i)$ (e.g., THR/APS/RAPS) are applied to labeled data, while a specially designed unlabeled score $\tilde{s}_i = \tilde{S}(\tilde{\mathbf{x}}_i)$ is applied to unlabeled data. The two score sets are merged to compute the quantile threshold:

$$\hat{\tau}_{\text{SemiCP}} = \text{Quantile}\left(\{\tilde{s}_i\}_{i=1}^N \cup \{s_i\}_{i=1}^n, \frac{\lceil(n+N+1)(1-\alpha)\rceil}{n+N}\right)$$

At test time, the prediction set for a test sample $\mathbf{x}_{\text{test}}$ is constructed as $\mathcal{C}(\mathbf{x}_{\text{test}}) = \{y : S(\mathbf{x}_{\text{test}}, y) \le \hat{\tau}_{\text{SemiCP}}\}$.

### Key Designs: Nearest Neighbor Matching (NNM) Score

1. **Problem with the naive approach**: Directly substituting pseudo-labels $\hat{y}_i = \arg\max_j f_j(\tilde{\mathbf{x}}_i)$ into the score function introduces a systematic downward bias (pseudo bias), since pseudo-labels always correspond to the model's most confident class, causing scores to be underestimated, thresholds to be deflated, and coverage to be insufficient.
2. **Pseudo-bias definition**: $\Delta(\tilde{\mathbf{x}}_i) = S(\tilde{\mathbf{x}}_i, \tilde{y}_i) - S(\tilde{\mathbf{x}}_i, \hat{y}_i)$, i.e., the difference between the true score and the pseudo-label score.
3. **Nearest Neighbor Matching strategy**: For each unlabeled sample $\tilde{\mathbf{x}}_i$, the labeled sample $\mathbf{x}_j$ whose pseudo-label score is closest is identified in the pseudo-score space: $j = \arg\min_{j \in \{1,...,n\}} |S(\tilde{\mathbf{x}}_i, \hat{y}_i) - S(\mathbf{x}_j, \hat{y}_j)|$.
4. **NNM score computation**: The true bias of the matched labeled sample is used to correct the pseudo-score of the unlabeled sample: $\tilde{S}_{\text{nnm}}(\tilde{\mathbf{x}}_i) = S(\tilde{\mathbf{x}}_i, \hat{y}_i) + S(\mathbf{x}_j, y_j) - S(\mathbf{x}_j, \hat{y}_j)$.
5. **Core assumption**: Samples with similar pseudo-scores exhibit similar pseudo-bias distributions. Experiments confirm that the empirical distribution of NNM scores closely approximates the true score distribution.

### Loss & Training

The method is **training-free** and requires no additional training or optimization. Specifically:

- It directly utilizes the softmax outputs of a pretrained classifier without accessing training data.
- It is compatible with any labeled score function (THR, APS, RAPS, SAPS, etc.).
- It integrates seamlessly into conditional coverage settings (class-conditional, group-conditional).
- It can be combined with existing methods such as Interpolation and ClusterCP.

**Theoretical Guarantees**:

- **Theorem 1**: The coverage lower bound is $1-\alpha + \epsilon_{n,N}$, where the bias term $\epsilon_{n,N} = \frac{N}{N+n}(F_S(\hat{\tau}) - F_{\tilde{S}}(\hat{\tau}))$ is controlled by the CDF discrepancy between true and estimated scores.
- **Theorem 2**: The average coverage gap converges at rate $\mathcal{O}(1/\sqrt{N})$; increasing the amount of unlabeled data continuously reduces coverage bias.
- **Theorem 3**: The CDF of NNM scores asymptotically converges to the CDF of true scores, with the convergence rate governed by the number of labeled samples $n$.

## Key Experimental Results

### Main Results

| Dataset | Labeled $n$ | Unlabeled $N$ | Method | CovGap ↓ | AvgSize ↓ |
|---------|------------|--------------|--------|----------|-----------|
| CIFAR-10 | 20 | 4000 | Standard | 4.8 | 1.45 |
| CIFAR-10 | 20 | 4000 | **SemiCP** | **1.1** | **1.37** |
| CIFAR-10 | 10 | 4000 | Standard | 6.4 | 1.60 |
| CIFAR-10 | 10 | 4000 | **SemiCP** | **1.1** | **1.27** |
| ImageNet | 50 | 20000 | Standard | ~3.3 | ~75 |
| ImageNet | 50 | 20000 | **SemiCP** | **~2.1** | **~70.3** |

| Setting | Dataset | $n_{\text{avg}}$ | Method | CovGap ↓ | AvgSize ↓ |
|---------|---------|-----------------|--------|----------|-----------|
| Class-conditional | CIFAR-100 | 10 | Standard | 7.75 | 18.9 |
| Class-conditional | CIFAR-100 | 10 | **SemiCP** | **6.29** | **17.0** |
| Group-conditional | CIFAR-100 | 10 | Standard | High | Large |
| Group-conditional | CIFAR-100 | 10 | **SemiCP** | Significantly reduced | Significantly reduced |

### Ablation Study

- **Effect of unlabeled data volume** (ImageNet, $n=50$): As $N$ grows from 10 to 20,000, CovGap decreases and AvgSize shrinks consistently. Even with as few as $N=10$ unlabeled samples, CovGap decreases by approximately 0.1 and AvgSize by approximately 0.2.
- **Combination with Interpolation** (CIFAR-100): Interpolation alone yields a larger CovGap than Standard (unstable), but SemiCP+Interpolation reduces CovGap from 9 to 3.9 at $n=10$, and AvgSize approaches the Oracle level for $n>40$.
- **Combination with ClusterCP**: SemiCP+ClusterCP further reduces both CovGap and AvgSize across all values of $n_{\text{avg}}$.
- **NNM vs. Naive**: The naive pseudo-label score PDF is systematically lower than the true distribution, whereas the NNM score distribution closely matches the true score distribution (Fig. 3).

### Key Findings

1. **Up to 77% reduction in coverage gap**: On CIFAR-10 with 20 labeled and 4,000 unlabeled samples, CovGap drops from 4.8 to 1.1 while AvgSize simultaneously decreases by 5.7%.
2. **Cross-architecture robustness**: Consistent improvements are observed across 10 different architectures (ResNet/MobileNet/ConvNet/EfficientNet/ViT, etc.), with average CovGap reduced from 3.3 to 2.1.
3. **Larger gains under conditional coverage**: SemiCP yields greater improvements in class-conditional and group-conditional settings than in the marginal coverage setting.
4. **High data efficiency**: Even a very small amount of unlabeled data ($N=10$) produces meaningful improvements.

## Highlights & Insights

- **Training-free design**: Requires no additional training or optimization; directly exploits pretrained model outputs as a plug-and-play module. This contrasts sharply with the concurrent work [34], which requires optimizing an $N \times K$ weight matrix.
- **Elegant intuition behind NNM**: Bias estimation via nearest neighbor matching in pseudo-score space leverages the empirical observation that samples with similar pseudo-scores exhibit similar biases—a simple yet effective design.
- **Closed loop between theory and experiments**: Theorems 1–3 comprehensively characterize coverage guarantees, convergence rates, and score consistency, with experimental results closely matching theoretical predictions.
- **Strong compatibility**: SemiCP functions as a plug-in compatible with arbitrary combinations of THR/APS/RAPS, Interpolation, ClusterCP, and conditional CP variants.

## Limitations & Future Work

- The theoretical analysis relies on the i.i.d. assumption, which is stronger than the exchangeability assumption underlying standard CP.
- Validation is currently limited to classification tasks; extension to regression settings remains unexplored.
- NNM matching accuracy may be insufficient when labeled data is extremely scarce (e.g., $n < 5$).
- The core assumption of NNM (similar pseudo-scores → similar biases) may not hold under significant distribution shift.

## Related Work & Insights

- **Split CP and variants**: THR [Sadinle+ 2019], APS [Romano+ 2020], RAPS [Angelopoulos+ 2020], and SAPS [Huang+ 2023] design labeled score functions to improve prediction set efficiency, but all require sufficient labeled calibration data.
- **Handling small calibration sets**: Interpolation [Johansson+ 2015] interpolates thresholds, ClusterCP [Ding+ 2023] shares calibration across class clusters, and Few-shot CP [Fisch+ 2021] uses meta-learning—none exploit unlabeled data.
- **Prediction-Powered Inference (PPI)**: [Angelopoulos+ 2023] uses model predictions to tighten confidence intervals; the spirit is related but targets semi-supervised inference rather than CP calibration.
- **Unsupervised calibration**: [Mazuelas 2025] estimates label weights by minimizing IPM, but requires optimizing an $N \times K$ matrix, incurring high computational cost and limited scalability.
- **Positioning of SemiCP**: This is the first work to leverage unlabeled data for estimating nonconformity scores to improve CP calibration stability, and is complementary to the above methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce semi-supervised learning into CP calibration; the NNM score design is concise and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely comprehensive: 3 datasets × 3 score functions × 10 architectures × 1,000 repetitions × multiple CP variants.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clear, theoretical derivations are complete, and experimental presentation is systematic.
- Value: ⭐⭐⭐⭐ — Highly practical, training-free, and compatible with existing methods, though currently limited to classification settings.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Conformal Prediction Adaptive to Unknown Subpopulation Shifts](../../ICLR2026/llm_evaluation/conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/llm_evaluation/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](../../AAAI2026/llm_evaluation/dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/llm_evaluation/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[CVPR 2026\] HeSS: Head Sensitivity Score for Sparsity Redistribution in VGGT](hess_head_sensitivity_score_for_sparsity_redistribution_in_vggt.md)

<!-- RELATED:END -->
