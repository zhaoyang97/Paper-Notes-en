---
title: >-
  [Paper Note] Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders
description: >-
  [ICLR 2026][Causal Inference][Counterfactual explanations] This paper proposes L-GMVAE (Label-Conditional Gaussian Mixture VAE) and the LAPACE algorithm. By learning multiple Gaussian cluster centers per class in the lat…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "Counterfactual explanations"
  - "variational autoencoder"
  - "Gaussian mixture"
  - "robustness"
  - "algorithmic recourse"
date: 2026-05-08
content_hash: e8ac097589365068
---

# Synthesising Counterfactual Explanations via Label-Conditional Gaussian Mixture Variational Autoencoders

**Conference**: ICLR 2026
**arXiv**: [2510.04855](https://arxiv.org/abs/2510.04855)  
**Code**: None (uses CARLA library)  
**Area**: Explainable AI / Causal Inference
**Keywords**: Counterfactual explanations, variational autoencoder, Gaussian mixture, robustness, algorithmic recourse

## TL;DR
This paper proposes L-GMVAE (Label-Conditional Gaussian Mixture VAE) and the LAPACE algorithm. By learning multiple Gaussian cluster centers per class in the latent space and performing linear interpolation from the input's latent representation to the target class center, the method generates path-based counterfactual explanations while guaranteeing validity, plausibility, diversity, and perfect robustness to input perturbations.

## Background & Motivation

**Background**: Counterfactual explanations (CEs) provide algorithmic recourse to individuals affected by automated decisions (e.g., how to change one's profile after a loan rejection). Ideal CEs must satisfy validity, proximity, plausibility (on-manifold), and diversity.

**Limitations of Prior Work**: Most existing methods address these properties in isolation, making it difficult to simultaneously guarantee multiple forms of robustness (input-perturbation robustness, model-change robustness) within a single framework. VAE-based approaches are typically unconditional, ignoring classifier label information and requiring complex latent-space search procedures.

**Key Challenge**: How can one simultaneously satisfy the multi-dimensional requirements of CEs — valid yet plausible, proximate yet robust, diverse yet stable?

**Goal**: To design a unified framework that generates CEs satisfying validity, proximity, plausibility, diversity, input robustness, and model robustness simultaneously.

**Key Insight**: Identify a diverse set of prototypical recourse targets in the target class, then guide all CEs to converge toward these points. These prototypes are learned naturally via a label-conditional GMM in the VAE latent space.

**Core Idea**: Partition GMVAE clusters by class label (K/L clusters per class); the decoded cluster centers serve as valid, plausible, and robust CE targets. Linear interpolation in the latent space from the input representation to the target center yields a sequence of CE candidates along a path.

## Method

### Overall Architecture

**Training phase**: An L-GMVAE is trained using classifier-predicted labels to learn a structured latent space where each class corresponds to a set of Gaussian clusters. **Inference phase**: LAPACE encodes the input into the latent space, performs linear interpolation toward each target-class cluster center, and decodes the interpolated points to produce a CE path.

### Key Designs

1. **L-GMVAE (Label-Conditional Gaussian Mixture VAE)**:

    - **Function**: Learns a Gaussian mixture latent space partitioned by class labels.
    - **Mechanism**: The cluster set $\mathcal{C} = \mathcal{C}_1 \cup \ldots \cup \mathcal{C}_L$ assigns K/L clusters uniformly per class. The generative model is $p(x,c,z|y) = p(c|y)\, p_\theta(z|c)\, p_\theta(x|z)$, with inference model $q(z,c|x,y)$. The ELBO comprises three terms: KL(c) encourages utilization of all clusters, KL(z) encourages cluster separation, and the reconstruction term ensures decoding quality.
    - **Design Motivation**: Cluster centers naturally become valid, plausible, and diverse recourse targets, as the decision boundaries learned by the classifier on training data align with the L-GMVAE clusters.

2. **LAPACE (LAtent PAth Counterfactual Explanations)**:

    - **Function**: Generates CE paths via linear interpolation in the latent space.
    - **Mechanism**: An input $x$ is encoded as $z_x$; for each target-class cluster center $z_{c_j}$, the interpolated point $z_\tau = (1-\tau)z_x + \tau z_{c_j}$ is decoded to yield path points. All paths converge to fixed cluster centers, guaranteeing input robustness.
    - **Design Motivation**: Linear interpolation exploits the smoothness of the VAE latent space; points along the path offer a continuous spectrum from proximity-focused to robustness-focused CEs.

3. **Actionability Constraints**:

    - **Function**: Enforces user-specified feature constraints along the CE path.
    - **Mechanism**: At each step $\tau$, constraints $g(\mathrm{Dec}(z_\tau))$ are checked; when violated, the latent vector is corrected via gradient descent.
    - **Design Motivation**: In practice, certain features may have fixed values or bounded ranges that must be respected.

### Loss & Training

$$\mathrm{ELBO} = \mathrm{KL}(c) + \mathrm{KL}(z) + \text{reconstruction loss}$$

Binary cross-entropy is used for categorical features and MSE for continuous features. One L-GMVAE is trained per dataset–classifier pair, with 5 clusters per class.

## Key Experimental Results

### Main Results

| Method | Validity | Proximity | Plausibility (LOF) | Diversity | Model Robust. | Input Robust. |
|---|---|---|---|---|---|---|
| LAPACE-Last | 100% | Moderate | **Best** | High | **100%** | **Perfect** |
| LAPACE-First | 100% | **Competitive** | Best | High | Moderate | Perfect |
| NNCE | 100% | Best | Good | N/A | — | Good |
| DiCE | <100% | Good | Poor | Good | — | — |
| DRCE | 100% | Good | Good | Good | — | Good |

### Ablation Study

| Dataset | Train on Real vs. Synthetic | Gap | Center Accuracy |
|---|---|---|---|
| heloc-RF | 73.97% vs 71.07% | 2.9% | 100% |
| wine-RF | 89.70% vs 87.42% | 2.3% | 100% |
| adult-RF | 93.82% vs 81.13% | 12.7% | 100% |
| compas-RF | 90.79% vs 85.03% | 5.8% | 100% |

### Key Findings
- **100% cluster center accuracy**: Decoded cluster centers are correctly classified by the original classifier across all datasets.
- **LAPACE achieves the best plausibility**: LOF scores are lowest (closest to 1.0) across all datasets.
- **Perfect input robustness**: Since all paths converge to fixed centers, the generated CEs are completely invariant to input perturbations.
- **100% actionability constraint satisfaction**: LAPACE-constrained finds valid CEs satisfying all constraints in every test case.
- Classifier probability along path points increases monotonically with $\tau$, confirming alignment between the latent space and the classifier.

## Highlights & Insights
- **Practical value of path-based CEs**: Users can trade off between "proximate but less robust" and "robust but requiring larger changes" — a strictly more informative alternative to single-point CEs.
- **Simplicity and effectiveness of label-conditional clustering**: Partitioning GMM clusters by class label straightforwardly yields diverse prototypical recourse targets.
- **Privacy preservation**: The method generates synthetic CEs rather than exposing training data points.

## Limitations & Future Work
- CE validity depends on the quality of L-GMVAE training; cluster centers must be verified to be correctly classified.
- On datasets with a large proportion of categorical features, the quality of synthetic data degrades noticeably (e.g., a 12.7% gap on adult).
- Linear interpolation assumes local smoothness of the latent space, which may be insufficient for complex decision boundaries.
- Causal constraints among features are not considered.

## Related Work & Insights
- **vs. DRCE**: DRCE uses nearest neighbors to ensure input robustness, but its heuristic distance threshold cannot guarantee perfect robustness. LAPACE achieves perfect robustness through convergence to fixed centers.
- **vs. DiCE**: DiCE produces diverse CEs via multi-objective optimization but at the cost of poor plausibility. LAPACE naturally ensures plausibility through the VAE manifold.
- **vs. RobXCE**: RobXCE enhances model robustness by pushing the decision boundary further away but does not guarantee diversity.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of label-conditional GMVAE and path-based CEs is novel and principled.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Eight metrics, five baselines, four datasets, actionability tests, and path analysis constitute a very comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear and well-organized with intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Provides a unified framework addressing the multi-attribute requirements of counterfactual explanations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[ICLR 2026\] Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ICML 2026\] Density-Guided Robust Counterfactual Explanations on Tabular Data under Model Multiplicity](../../ICML2026/causal_inference/density-guided_robust_counterfactual_explanations_on_tabular_data_under_model_mu.md)
- [\[AAAI 2026\] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](../../AAAI2026/causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)

</div>

<!-- RELATED:END -->
