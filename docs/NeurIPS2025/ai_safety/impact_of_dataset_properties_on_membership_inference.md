---
title: >-
  [Paper Note] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning
description: >-
  [NeurIPS 2025][AI Safety][membership inference attack] This paper theoretically derives and empirically validates a power-law relationship between membership inference attack (MIA) vulnerability and the number of samples per class in deep transfer learning: $\log(\text{tpr}-\text{fpr}) = -\beta_S \log(S) - \beta_0$. It finds that increasing data volume reduces both average and worst-case vulnerability, but protecting the most vulnerable samples requires an extremely large amount of data.
tags:
  - NeurIPS 2025
  - AI Safety
  - membership inference attack
  - privacy
  - transfer learning
  - power law
  - differential privacy
date: 2026-05-08
content_hash: aba55d38b373ff43
---

# Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning

**Conference**: NeurIPS 2025
**arXiv**: [2402.06674](https://arxiv.org/abs/2402.06674)
**Code**: None
**Area**: AI Security
**Keywords**: membership inference attack, privacy, transfer learning, power law, differential privacy

## TL;DR

This paper theoretically derives and empirically validates a power-law relationship between membership inference attack (MIA) vulnerability and the number of samples per class in deep transfer learning: $\log(\text{tpr}-\text{fpr}) = -\beta_S \log(S) - \beta_0$. It finds that increasing data volume reduces both average and worst-case vulnerability, but protecting the most vulnerable samples requires an extremely large amount of data.

## Background & Motivation

1. **MIA as a Privacy Metric**: Membership inference attacks (MIAs) evaluate privacy leakage by determining whether a sample was included in the training set, providing a complementary empirical lower bound on privacy relative to the theoretical upper bounds offered by differential privacy (DP).

2. **Threat Model Differences between MIA and DP**: DP assumes an extremely powerful adversary (with access to all training data except the target point) and provides worst-case guarantees; MIA assumes a more realistic adversary (with access only to the data distribution) but yields statistical evaluations that cannot provide universal guarantees.

3. **Known but Unquantified Phenomena**: Prior work has observed that models with more classes, fewer training samples, or minority classes exhibit greater vulnerability, yet none have established **quantitative relationships** between these factors and vulnerability.

4. **Privacy Significance of Transfer Learning**: Fine-tuning pretrained models is increasingly common in privacy-sensitive, data-limited settings (e.g., medical imaging), making it critical to understand the factors driving MIA vulnerability in such scenarios.

5. **Absence of Worst-Case Analysis**: Existing work focuses on average vulnerability, with insufficient systematic study of worst-case vulnerability at the individual sample level—even a small number of samples being inferred with high confidence constitutes a serious privacy risk.

6. **Need for Predictable Vulnerability Models**: Practitioners require tools to assess privacy risk before training; a quantitative power-law relationship can be directly applied to risk prediction and data requirement estimation.

## Method

### Theoretical Analysis Framework

#### Simplified Model Construction

The paper constructs a simplified model of membership inference to facilitate theoretical derivation:

1. **Data Generation**: For each class, a true class mean $\bm{m}_c$ is sampled (orthogonal vectors on a high-dimensional unit sphere), and $2S$ samples $\bm{x}_c \sim \mathcal{N}(\bm{m}_c, \Sigma)$ are generated per class.
2. **Target Model**: $CS$ samples are randomly selected from all samples to compute per-class means $\hat{\bm{m}}_c$, and classification is performed via inner products $\langle \bm{x}, \hat{\bm{m}}_c \rangle$.
3. **Adversary**: The LiRA likelihood ratio test is used to determine whether the target point belongs to the training set.

This model corresponds to the linear classifier (head) commonly used in transfer learning, where the inner product scores follow a normal distribution; LiRA is therefore the optimal attack under this model in the sense of the Neyman–Pearson lemma.

#### Core Theoretical Results

**Lemma 1 (Per-Sample LiRA Vulnerability)**: LiRA vulnerability is expressed in terms of the parameters of a location-scale family distribution:

$$\text{tpr}_{\text{LiRA}}(\bm{x}) = 1 - F_Z\left(F_Z^{-1}(1-\text{fpr}) - \frac{\mu_{\text{in}}(\bm{x}) - \mu_{\text{out}}(\bm{x})}{\sigma(\bm{x})}\right)$$

**Theorem 2 (Per-Sample Power Law)**: Under the simplified model, the following is derived:

$$\log(\text{tpr} - \text{fpr}) = -\frac{1}{2}\log S - \frac{1}{2}\Phi^{-1}(\text{fpr})^2 + \log\frac{|\langle \bm{x}, \bm{x} - \bm{m}_{\bm{x}} \rangle|}{\sqrt{\bm{x}^T \Sigma \bm{x}} \sqrt{2\pi}} + \log(1+\xi(S))$$

where $\xi(S) = O(1/\sqrt{S})$. The key insight is that the slope $\beta_S = 1/2$ (in log-log coordinates), and vulnerability decays with increasing $S$ according to a power law.

**Corollary 4 (Average-Case Power Law)**: After taking expectations over the data distribution, average vulnerability follows the same power law, and the Cauchy–Schwarz inequality provides an upper bound on the worst case, showing that as long as $\|\bm{x} - \bm{m}_{\bm{x}}\|$ is bounded, the vulnerability of all samples can be reduced by increasing $S$.

### Econometric Regression Model

Motivated by theory, the following linear regression prediction model is constructed:

$$\log_{10}(\text{tpr} - \text{fpr}) = \beta_S \log_{10}(S) + \beta_C \log_{10}(C) + \beta_0$$

Trained on ViT-B (Head) data, this achieves $R^2 = 0.930$ (fpr=0.001) and generalizes to ResNet-50 and FiLM fine-tuning.

### Extension to RMIA

The theoretical framework also applies to RMIA attacks; analogous power-law proofs are provided in the appendix.

## Key Experimental Results

### Experimental Setup

- **Feature Extractors**: ViT-Base-16 (pretrained on ImageNet-21k), ResNet-50
- **Fine-tuning Methods**: Head (linear layer), FiLM (parameter-efficient)
- **Attack Methods**: LiRA (256 shadow models), RMIA
- **Datasets**: Subset of the VTAB benchmark (datasets with test accuracy >80%), including Patch Camelyon, EuroSAT, CIFAR-100, etc.
- **Evaluation Metric**: tpr at fixed fpr (fpr = $10^{-1}$ to $10^{-5}$)

### Main Results

**Power-Law Validation (Figure 1)**: Across all tested datasets, $\log(\text{tpr}-\text{fpr})$ exhibits a clear linear relationship with $\log(S)$, holding across different fpr levels.

| fpr Level | Regression Slope $\beta_S$ | Theoretical Value | $R^2$ |
|:---:|:---:|:---:|:---:|
| 0.1 | ≈ -0.5 | -0.5 | 0.930 |
| 0.01 | ≈ -0.5 | -0.5 | High |
| 0.001 | ≈ -0.5 | -0.5 | 0.930 |

### Data Requirements to Protect the Most Vulnerable Samples (Table 1 — Predicted Minimum $S$, $C=2$)

| DP Parameter $\epsilon$ | Average Case (fpr=0.001) | Worst Case (fpr=0.1) |
|:---:|:---:|:---:|
| 0.25 | 320,000 | $5.5 \times 10^9$ |
| 0.50 | 88,000 | $2.6 \times 10^8$ |
| 0.75 | 38,000 | $3.5 \times 10^7$ |
| 1.00 | 19,000 | $7.0 \times 10^6$ |

**Key Findings**:
- **Power law holds**: $\beta_S \approx -0.5$ is in strong agreement with theoretical predictions.
- **Weak effect of class count**: Increasing $C$ slightly raises vulnerability, but the trend is less pronounced than that of $S$.
- **Individual sample analysis (Figure 6)**: Slopes at quantile levels range from approximately -0.48 to -0.57, close to the theoretical value; however, the slope for the most vulnerable samples (max) is only -0.27, indicating substantially slower decay.
- **Generalization**: The regression model trained on ViT-B (Head) predicts vulnerability for R-50 (Head) ($R^2=0.790$) and R-50 (FiLM) well.
- **Training from scratch yields higher vulnerability**: The model underestimates vulnerability for models trained from scratch (i.e., without transfer learning).

## Highlights & Insights

- **Strong theory–experiment agreement**: The power law $\beta_S = -0.5$ derived from the simplified model is precisely validated in real deep transfer learning experiments, demonstrating strong predictive capability.
- **Practical vulnerability prediction tool**: The regression model predicts MIA vulnerability using only dataset properties ($S$, $C$), enabling privacy risk assessment prior to training.
- **In-depth worst-case analysis**: The paper reveals that protecting the most vulnerable samples is orders of magnitude harder than protecting average samples (requiring 2–4 orders of magnitude more data), providing an important practical warning.
- **Bridging to DP guarantees**: By employing the conversion theorem of Kairouz et al., the paper draws meaningful comparisons between empirical MIA results and formal DP guarantees.

## Limitations & Future Work

- **Simplified model assumptions**: The theory relies on Gaussian distributions and orthogonal class means; it may not apply to heavy-tailed distributions or data with complex manifold structure.
- **Limited to transfer learning**: The power-law relationship may not hold for training from scratch (explicitly noted in Remark 3), and experiments confirm that training from scratch results in higher vulnerability.
- **Statistical rather than guaranteed**: MIA evaluation is inherently statistical and cannot provide the universal privacy guarantees offered by DP.
- **Restricted adversary assumptions**: The adversary is assumed to know only the target point while the rest of the dataset is random; a stronger adversary with partial knowledge of the training data may invalidate the power law.

## Related Work & Insights

| Dimension | Ours | Carlini et al. (2022) LiRA |
|:---|:---|:---|
| Focus | Quantitative impact of dataset properties ($S$, $C$) on MIA vulnerability | Proposing the LiRA attack itself |
| Theoretical Contribution | Deriving the power-law relationship and providing a predictive model | No quantitative theory relating vulnerability to data volume |
| Experimental Scope | Systematic variation of $S$ and $C$ across multiple datasets and fine-tuning methods | Limited results for training from scratch |
| Practical Guidance | Provides minimum $S$ estimates required for protection | No data requirement estimation |

| Dimension | Ours | Feldman & Zhang (2020) |
|:---|:---|:---|
| Research Focus | Quantitative relationship between MIA vulnerability and dataset size | Whether memorization is necessary for high utility |
| Core Finding | Power-law decay: $\text{tpr} - \text{fpr} \propto S^{-1/2}$ | Training from scratch requires extensive memorization; fine-tuning substantially reduces it |
| Analysis Granularity | Average + worst case + individual quantiles | Overall degree of memorization |
| Theoretical Framework | Simplified Gaussian model + Neyman–Pearson optimality | Memorization argument based on long-tail distributions |

## Rating

- ⭐⭐⭐⭐ **Novelty**: First work to establish a precise quantitative relationship (power law) between MIA vulnerability and dataset properties.
- ⭐⭐⭐⭐⭐ **Technical Depth**: Rigorous theoretical derivation (from simplified model to regression prediction) with strong theory–experiment agreement.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Covers multiple feature extractors, fine-tuning methods, datasets, and attack methods, with detailed individual vulnerability analysis.
- ⭐⭐⭐⭐ **Value**: Provides directly applicable vulnerability prediction models and data requirement estimates, offering practical guidance for privacy-preserving practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)
- [\[NeurIPS 2025\] Enabling Differentially Private Federated Learning for Speech Recognition: Benchmarks, Adaptive Optimizers and Gradient Clipping](enabling_differentially_private_federated_learning_for_speech_recognition_benchm.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)

</div>

<!-- RELATED:END -->
