---
title: >-
  [Paper Note] Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration
description: >-
  [NeurIPS 2025][Interpretability][Perturbation-based Explanations] This paper reveals a fundamental connection between uncertainty calibration (the alignment between model confidence and actual accuracy) and the quality o…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Perturbation-based Explanations"
  - "Uncertainty Calibration"
  - "ReCalX"
  - "Shapley Values"
  - "LIME"
  - "Temperature Scaling"
date: 2026-05-08
content_hash: 08486fb5292b4521
---

# Improving Perturbation-based Explanations by Understanding the Role of Uncertainty Calibration

**Conference**: NeurIPS 2025
**arXiv**: [2511.10439](https://arxiv.org/abs/2511.10439)  
**Code**: [GitHub](https://github.com/thomdeck/recalx)  
**Area**: Interpretability
**Keywords**: Interpretability, Perturbation-based Explanations, Uncertainty Calibration, ReCalX, Shapley Values, LIME, Temperature Scaling

## TL;DR

This paper reveals a fundamental connection between uncertainty calibration (the alignment between model confidence and actual accuracy) and the quality of perturbation-based explanation methods. It demonstrates that miscalibration of models on perturbed inputs directly degrades the quality of both global and local explanations, and proposes ReCalX, which applies perturbation-level-adaptive temperature scaling to substantially improve the robustness and fidelity of explanations.

## Background & Motivation

Perturbation-based explanation methods (e.g., SHAP, LIME) are among the most widely used model interpretability techniques in practice. Their core principle is to systematically modify input features and observe changes in model outputs to quantify feature importance. However, a fundamental challenge exists:

**Out-of-Distribution Problem**: Perturbed inputs deviate significantly from the training distribution, making model predictions on these "synthetic" inputs unreliable.

**Misleading Probabilities**: Models may output high-confidence yet incorrect probabilities for perturbed samples; aggregating such misleading predictions to construct explanations severely distorts the results.

**Explanation Instability**: This may also account for the commonly observed instability of perturbation-based explanations.

Core Problem: **If the underlying predictions used to construct explanations are themselves unreliable, how can the explanations be reliable?**

Existing calibration methods (e.g., standard temperature scaling) are optimized only on clean, unperturbed data, neglecting the specific perturbation scenarios encountered during the explanation process. Preliminary evidence suggests that calibration may benefit interpretability, but rigorous theoretical analysis has been lacking.

## Method

### Overall Architecture

A three-step approach: (1) theoretically prove how calibration error degrades explanation quality; (2) empirically verify that models are indeed severely miscalibrated under perturbation; (3) propose ReCalX for perturbation-specific recalibration.

### Theoretical Analysis I: Impact on Global Explanations

The predictive power of a feature subset $S$ is defined as the improvement in model performance when only features in $S$ are observed:

$$v_f^\pi(S) = \mathbb{E}[\mathcal{L}(f_\emptyset^\pi(X), Y)] - \mathbb{E}[\mathcal{L}(f_S^\pi(X), Y)]$$

**Theorem 3.2** (Decomposition of Predictive Power): For cross-entropy loss:

$$v_f^\pi(S) = \underbrace{D_{\text{KL}}(P_Y \| f_\emptyset^\pi(X))}_{\text{Perturbation Baseline Bias}} + \underbrace{I(f_S^\pi(X), Y)}_{\text{Information Content}} - \underbrace{CE_{\text{KL}}(f_S^\pi)}_{\text{Calibration Error}}$$

Interpretation of the three components:
- First term: Baseline bias introduced by the perturbation strategy.
- Second term: Mutual information between the model's predictions (given only features in $S$) and $Y$ — the ideal predictive power.
- Third term: **Calibration error directly reduces predictive power**, causing feature importance to be under- or over-estimated.

**Corollary 3.3**: If the model is perfectly calibrated under all subset perturbations, then $v_f^\pi(S) = I(f_S^\pi(X), Y)$.

### Theoretical Analysis II: Impact on Local Explanations

**Theorem 3.4**: The discrepancy between local explanation $\phi(x)$ and the ideal explanation $\phi^*(x)$ under perfect calibration is bounded by:

$$\frac{1}{d} \|\phi(x) - \phi^*(x)\|_2^2 \leq 2 \cdot CE_{\text{KL}}^{\max_S} + \sqrt{8 \log(1/\delta)}$$

with probability at least $1-\delta$. Here $CE_{\text{KL}}^{\max_S}$ denotes the maximum calibration error across all perturbed subsets. This implies that improving local explanations requires reducing calibration error across **all** perturbation levels; calibrating only on the original data is insufficient.

### Key Design: ReCalX

Standard temperature scaling uses a single global temperature $T$, which cannot adapt to varying perturbation intensities. The core idea of ReCalX is **adaptive temperature conditioning on perturbation level**.

**Perturbation Level Definition**: $\lambda(S) = (d - |S|) / d \in [0, 1]$, i.e., the proportion of perturbed features.

**Binned Temperature Learning**: The interval $[0,1]$ is divided into $B$ equal-width bins; an independent temperature $T_b$ is learned for each bin $b$. Given a validation set, $T_b$ is optimized by minimizing cross-entropy loss on perturbed samples within each bin.

**Inference-Time Application**:

$$f_{\text{ReCalX}}^\pi(x, S; \{T_b\}_{b=1}^B)_k = \frac{\exp(z_k(\pi(x,S)) / T(S))}{\sum_{j=1}^K \exp(z_j(\pi(x,S)) / T(S))}$$

where $T(S)$ is selected based on the bin corresponding to $\lambda(S)$.

**Information Preservation**: Temperature scaling is a componentwise strictly monotone function and satisfies information preservation (Proposition 4.2) — it does not alter prediction rankings or mutual information $I(f_S^\pi(X), Y)$, ensuring that explanations still target the original model behavior.

### Implementation Details

- Validation set: 200 randomly selected samples per dataset, 10 perturbed instances generated per perturbation level, yielding 2,000 samples per bin.
- Evaluation: A consistent and asymptotically unbiased KL calibration error estimator is used; at least 5,000 test samples per setting.
- Number of bins $B$: Default is 10; more bins yield monotone improvement with diminishing returns.

## Key Experimental Results

### Main Results I: Calibration Error on Tabular Data (Mean Replacement Perturbation)

| Dataset | Model | Uncalibrated $CE^{\max}$ | Temp. Scaling $CE^{\max}$ | **ReCalX** $CE^{\max}$ | Reduction↓ |
|--------|------|-------------------|---------------------|----------------------|------|
| Electricity | MLP | 0.1534 | 0.1664 | **0.0163** | 89.4% |
| Covertype | MLP | 0.0797 | 0.1115 | **0.0061** | 92.3% |
| Credit | MLP | 0.4763 | 0.5961 | **0.0533** | 88.8% |
| Pol | MLP | 0.6735 | 0.6521 | **0.1679** | 75.1% |
| Covertype | ResNet | 0.0963 | 0.1413 | **0.0080** | 91.7% |
| Pol | ResNet | 0.8633 | 1.0173 | **0.0910** | 89.5% |

Standard temperature scaling often **exacerbates** miscalibration under perturbation (e.g., MLP+Credit: 0.4763→0.5961), whereas ReCalX achieves 75–92% reduction.

### Main Results II: Calibration Error on Image Models (ImageNet)

| Model | Perturbation | Uncalibrated $CE^{\max}$ | Temp. Scaling $CE^{\max}$ | **ReCalX** $CE^{\max}$ | Reduction↓ |
|------|---------|-------------------|---------------------|----------------------|------|
| ResNet50 | Zero | 0.4177 | 0.1810 | **0.0128** | 96.9% |
| DenseNet121 | Zero | 0.3769 | 0.2640 | **0.0098** | 97.4% |
| ViT | Zero | 0.2618 | 0.3057 | **0.0078** | 97.0% |
| SigLIP | Zero | 0.2013 | 0.1476 | **0.0300** | 85.1% |
| ResNet50 | Blur | 0.4158 | 0.1659 | **0.0139** | 96.7% |
| ViT | Blur | 0.0365 | 0.0559 | **0.0072** | 80.3% |

ReCalX achieves up to 97.4% reduction in calibration error on image models. Temperature scaling on ViT even worsens calibration (0.2618→0.3057).

### Explanation Robustness (ImageNet, Average Sensitivity $S_{\text{AVG}}$↓)

| Model | LIME (Orig.)→LIME (ReCalX) | KernelSHAP (Orig.)→KernelSHAP (ReCalX) | FeatureAblation (Orig.)→FeatureAblation (ReCalX) |
|------|---------------------|----------------------------------|-------------------------------------------|
| ResNet50 | 1.349→**1.190** | 1.434→**1.364** | 0.965→**0.825** |
| DenseNet121 | 1.174→**0.952** | 1.465→**1.125** | 0.716→**0.602** |
| ViT | 1.498→**1.155** | 1.399→**1.279** | 1.041→**0.880** |
| SigLIP | 1.215→**0.963** | 1.434→**1.222** | 1.140→**0.922** |

ReCalX consistently improves explanation robustness across all models × all explanation methods × both perturbation types.

### Ablation Study & Key Findings

1. **Remove-and-Retrain Fidelity**: Removing features in decreasing order of ReCalX-enhanced Shapley importance leads to steeper performance degradation (e.g., on the Electricity dataset, removing the top-3 features yields a 33% loss increase vs. 24% for the uncalibrated baseline), indicating that calibrated explanations more accurately identify truly important features.
2. **Miscalibration Increases Monotonically with Perturbation Level**: On tabular data, miscalibration increases nearly monotonically with perturbation level; image models exhibit more varied patterns (ResNet50 shows worse calibration at low perturbation levels).
3. **Few Validation Samples Suffice**: ReCalX achieves most of its calibration improvement with only a few hundred validation samples.
4. **Cross-Perturbation-Type Correlation**: Strong correlations exist between miscalibration patterns across different perturbation types.

## Highlights & Insights

1. **Filling a Theoretical Gap**: This work provides the first rigorous proof of how calibration error directly degrades the quality of both global and local perturbation-based explanations, establishing a quantitative link between calibration and interpretability.
2. **Revealing a Counterintuitive Phenomenon**: Standard temperature scaling (optimized on original data) can actually **worsen** miscalibration in perturbation scenarios, explaining why naive calibration offers limited benefit for explanations.
3. **A Simple Yet Effective Method**: ReCalX essentially applies different temperatures for different perturbation levels — it is straightforward to implement with near-zero inference overhead (millisecond-level temperature lookup).
4. **Information Preservation Guarantee**: The method theoretically ensures that ReCalX does not alter the original model's predictive behavior (rank-preserving), which is crucial for the principle of "explaining the original model."
5. **Cross-Domain Applicability**: Effective across tabular data (MLP, ResNet) and images (ResNet50, DenseNet121, ViT, SigLIP), including zero-shot models.

## Limitations & Future Work

1. **Applicable Only to Perturbation-based Explanations**: Not directly applicable to gradient-based (e.g., Integrated Gradients) or counterfactual explanation methods, though the underlying principle may generalize.
2. **Discrete Bin Approximation**: Discretizing perturbation levels into bins may miss fine-grained variation, though experiments show 10 bins are sufficient.
3. **Requires Labeled Validation Data**: Calibration requires a labeled validation set and is thus inapplicable in purely unsupervised settings.
4. **Limitations of Temperature Scaling**: Temperature scaling assumes calibration error can be corrected by global logit rescaling, which may be insufficient for severely miscalibrated models.
5. **No Large-Scale LLM Validation**: Experiments focus on CV classification and tabular models; NLP/LLM settings remain untested.

## Related Work & Insights

- Relationship to SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro et al., 2016): ReCalX is a post-processing enhancement to these methods and does not modify the explanation algorithms themselves.
- The calibration literature (Guo et al., 2017) first documented the systematic overconfidence of deep networks; this paper extends those findings to perturbation scenarios.
- Work on out-of-distribution detection and calibration (Ovadia et al., 2019) has shown that model calibration collapses on OOD inputs; perturbations constitute a specific form of OOD inputs.
- Inspiration: The analogous idea of "condition-grouped calibration" could be extended to adversarial example detection, model monitoring, and related settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Solid theoretical contributions; establishes a new "calibration–explanation quality" connection; the method is simple but theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers tabular and image data, multiple model architectures, multiple explanation methods, multiple perturbation types, with thorough ablation studies.
- **Practicality**: ⭐⭐⭐⭐⭐ — Near-zero additional inference cost; plug-and-play; valuable for any practitioner using SHAP/LIME.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theory, empirics, and methodology are cohesively integrated; the logical chain is complete; figures and tables are well-designed.
- **Overall**: ⭐⭐⭐⭐ — Makes an important contribution at the intersection of interpretability and calibration, combining theoretical depth with practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LLM Probing with Contrastive Eigenproblems: Improving Understanding and Applicability of CCS](llm_probing_with_contrastive_eigenproblems_improving_understanding_and_applicabi.md)
- [\[NeurIPS 2025\] Minimizing False-Positive Attributions in Explanations of Non-Linear Models](minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[NeurIPS 2025\] URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)

</div>

<!-- RELATED:END -->
