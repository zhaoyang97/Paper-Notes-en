---
title: >-
  [Paper Note] Variational Adapter Cross-modal Similarity Representation
description: >-
  [ICML 2026][Optimization][Cross-modal retrieval] Learn a continuous cross-modal similarity distribution through a variational inference framework—using adaptive uncertainty weights to mitigate the false negative problem…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Cross-modal retrieval"
  - "Variational Autoencoder"
  - "Binary labeling problem"
  - "False negatives"
  - "CLIP fine-tuning"
date: 2026-05-08
content_hash: 85f12fe47fbd43c7
---

# Variational Adapter Cross-modal Similarity Representation

**Conference**: ICML 2026  
**arXiv**: [2605.30968](https://arxiv.org/abs/2605.30968)  
**Code**: To be confirmed  
**Area**: Multi-modal VLM  
**Keywords**: Cross-modal retrieval, Variational Autoencoder, Binary labeling problem, False negatives, CLIP fine-tuning

## TL;DR
Learn a continuous cross-modal similarity distribution through a variational inference framework—using adaptive uncertainty weights to mitigate the false negative problem caused by binary labeling, significantly improving VLM performance in cross-modal retrieval and domain generalization tasks.

## Background & Motivation

**Background**: VLMs such as CLIP align image and text in a unified representation space and have been widely applied in zero-shot classification, cross-modal retrieval, and open-vocabulary detection. However, existing methods often face data labeling limitations during the fine-tuning stage.

**Limitations of Prior Work**: Multi-modal datasets like MS-COCO typically employ binary sparse labeling ("match" or "mismatch"), forcibly partitioning the continuous similarity space into two classes. This prevents the model from capturing fine-grained semantic relationships between samples, severely damaging generalization performance, especially in fine-tuning scenarios with limited samples.

**Key Challenge**: The matching relationship between image-text pairs is inherently continuous and complex (e.g., the match between the Mona Lisa and "mysterious smile" involves both object-level and subjective perception). The coarseness of binary labels leads to a large number of false negatives (semantically related but labeled as mismatched), which undermines the semantic consistency of the representation space.

**Goal**: While keeping the CLIP base model frozen, explicitly model the continuous distribution of cross-modal similarity in the latent space through a fine-tuned adapter, enabling the model to assign higher uncertainty to false negatives.

**Core Idea**: Transform the binary supervised learning problem into a latent variable generative model using a VAE framework, naturally introducing uncertainty-based adaptive sample weights to achieve "adjusting learning intensity according to labeling confidence."

## Method

### Overall Architecture
VACSR consists of three key modules: (1) **Feature Interaction Layer**: Fuses encoder output image features $\bm{v}_i$ and text features $\bm{t}_j$ into a similarity vector $\bm{s}_{i,j} = \bm{v}_i \odot \bm{t}_j$ using the Hadamard product; (2) **Variational Adapter**: Maps the similarity vector to a latent space $\mathbf{z}_{i,j}$ of a two-component Gaussian Mixture Model (GMM) via an encoder network; (3) **Decoder Network**: Reconstructs similarity scores from the latent variables and outputs uncertainty $\sigma^2(\mathbf{z}_{i,j})$.

### Key Designs

1.  **Two-component Gaussian Mixture Posterior**:
    - **Function**: Breaks through the expressiveness limits of unimodal Gaussian distributions, allowing the model to learn more complex semantic representations.
    - **Mechanism**: Approximates the posterior as $p_\phi(\mathbf{z}_{i,j}|\bm{s}_{i,j})=\sum_{k=1}^{2}\alpha_k\mathcal{N}(\mathbf{z}_{i,j}|\mu_k,\sigma_k^2)$, where $\alpha_1,\alpha_2$ are learnable mixing weights. Using Jensen's inequality, the KL divergence provides a computable upper bound $\text{KL}[\sum_k\alpha_k p_k \| q] \leq \sum_k\alpha_k \text{KL}[p_k \| q]$.
    - **Design Motivation**: A unimodal Gaussian struggles to process "matched" and "mismatched" semantic distributions simultaneously; the mixture model allows the encoder to automatically select the appropriate Gaussian component based on the input.

2.  **Uncertainty-based Adaptive Weighting**:
    - **Function**: Assigns different learning intensities to samples of varying labeling quality—false negatives receive high uncertainty (low learning weight), while certain positives and hard samples receive low uncertainty (high learning weight).
    - **Mechanism**: Derived from the limiting behavior of reconstruction loss—when $\sigma^2 \to 0$, the model strictly follows the binary labels; when $\sigma^2 \to \infty$, the labeling signal is submerged by noise. Both the mean $\mu(\mathbf{z}_{i,j})$ and variance $\sigma^2(\mathbf{z}_{i,j})$ are learned concurrently via $\mathcal{L}_{\text{recon}} = \frac{1}{2\sigma^2}\|\hat{y}-\mu\|^2 + \log\sigma + \frac{1}{2}\log 2\pi$.
    - **Design Motivation**: Traditional contrastive losses require a delicate balance between temperature $\tau$ and scaling parameters; this method allows the model to self-learn uncertainty and dynamically adapt to binary labeling noise, avoiding manual parameter tuning.

3.  **ELBO Optimization Objective**:
    - **Function**: Simultaneously maximizes data fitting and constrains the KL divergence of the latent space.
    - **Mechanism**: Based on the standard VAE framework $\text{ELBO} = \mathbb{E}_{p_\phi}[\log q_\theta(\hat{y}|\mathbf{z})] - \text{KL}[p_\phi \| q]$, where the reconstruction term assumes Gaussian likelihood (equivalent to MSE), and the KL term forces the latent representation to follow a standard normal prior.
    - **Design Motivation**: The reconstruction term naturally provides uncertainty weighting without additional design; the KL regularization prevents the model from "cheating" by over-exploiting latent space variance.

## Key Experimental Results

### Main Results (COCO Dataset, 1K and 5K Test Sets)

| Model | 1K R@1(I→T) | 1K R@1(T→I) | 5K R@1(I→T) | 5K R@1(T→I) | Gain |
|------|-------------|-------------|-------------|-------------|------|
| PCME++ (ViT-B/32) | 81.6 | 69.2 | 62.1 | 48.1 | baseline |
| **VACSR (ViT-B/32)** | **84.2** | **70.3** | **66.5** | **49.8** | +3.2%, +1.6% |
| PCME++ (ViT-B/16) | 85.3 | 73.4 | 68.7 | 53.4 | baseline |
| **VACSR (ViT-B/16)** | **87.4** | **74.3** | **71.6** | **54.5** | +2.5%, +1.6% |

### Noise Robustness (COCO with 20% Noisy Labels)

| Method | 1K R@1 | 5K R@1 | RSUM | Gain vs. PCME++ |
|------|---------|---------|-------|----------------|
| PCME++ | 71.6 | 50.4 | 524.6 | baseline |
| **VACSR** | **76.4** | **57.1** | **539.0** | +4.8% (R@1), +13.2% (RSUM) |

### Key Findings
- Under clean labels, VACSR achieves an average improvement of 2-3% over PCME++.
- Advantages are more pronounced in scenarios with 20% noise injection (up to 5%+ improvement), indicating that adaptive uncertainty effectively mitigates labeling noise.
- Cross-dataset (EC/CxC) tests verify generalization performance.

## Highlights & Insights
- **Theoretical Depth**: Rigorously proves the specific harm of binary labeling to contrastive and sigmoid losses through gradient analysis, quantifying the "relative gradient penalty" $r_i$.
- **Elegant Uncertainty Design**: Interprets uncertainty as a "measure of labeling quality" rather than "semantic ambiguity"; this perspective shift allows the model to handle false negatives more rationally.
- **Lightweight Adapter**: Only adds two MLPs on top of frozen CLIP features, resulting in extremely low parameter count and computational overhead.

## Limitations & Future Work
- The choice of Hadamard product was not systematically compared with other feature interaction methods (e.g., bilinear pooling, outer product).
- The fixed number of mixture components (two) may limit modeling of highly complex labeling patterns.
- Label correction limitations—disproportionate concentration of false negatives may still lead to learning bias.
- Improvements: Dynamic number of components; other flexible posterior forms; combining with active learning or manual data cleaning.

## Related Work & Insights
- **vs. Probabilistic Embedding Methods (PCME/PCME++)**: PCME attributes uncertainty to sample semantic ambiguity; VACSR attributes it to labeling noise, which is more consistent with actual data labeling scenarios.
- **vs. Contrastive Learning Temperature Tuning**: Traditional methods require meticulous adjustment of temperature coefficients; VACSR achieves adaptation through self-learning variance parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ Re-modeling the binary labeling problem as variational inference is novel; VAE applications in representation learning have precedents (innovation is moderate).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ COCO/EC/CxC + 1K/5K + noise robustness + domain generalization.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with rigorous theoretical derivation.
- Value: ⭐⭐⭐⭐⭐ Addresses practical problems in CLIP fine-tuning; the method is lightweight and integrable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear Programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)
- [\[NeurIPS 2025\] Least Squares Variational Inference](../../NeurIPS2025/optimization/least_squares_variational_inference.md)
- [\[NeurIPS 2025\] Brain-like Variational Inference](../../NeurIPS2025/optimization/brain-like_variational_inference.md)
- [\[CVPR 2026\] UniFusion: A Unified Image Fusion Framework with Robust Representation and Source-Aware Preservation](../../CVPR2026/optimization/unifusion_a_unified_image_fusion_framework_with_robust_representation_and_source.md)

</div>

<!-- RELATED:END -->
