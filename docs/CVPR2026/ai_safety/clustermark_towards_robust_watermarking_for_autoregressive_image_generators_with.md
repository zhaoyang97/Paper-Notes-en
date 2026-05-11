---
title: >-
  [Paper Note] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering
description: >-
  [CVPR 2026][AI Safety][Autoregressive image generation] This paper proposes ClusterMark, a watermarking scheme based on visual token clustering that adapts KGW-style LLM watermarking to autoregressive image generators. B…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "Autoregressive image generation"
  - "watermark detection"
  - "visual token clustering"
  - "robust watermarking"
  - "VQ-VAE"
date: 2026-05-08
content_hash: 9b05c98c8cb0dd0f
---

# ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering

**Conference**: CVPR 2026
**arXiv**: [2508.06656](https://arxiv.org/abs/2508.06656)
**Code**: [https://github.com/lukovnikov/ClusterMark](https://github.com/lukovnikov/ClusterMark)
**Area**: AI Security / Image Watermarking
**Keywords**: Autoregressive image generation, watermark detection, visual token clustering, robust watermarking, VQ-VAE

## TL;DR

This paper proposes ClusterMark, a watermarking scheme based on visual token clustering that adapts KGW-style LLM watermarking to autoregressive image generators. By assigning visually similar tokens to the same green/red partition, it significantly improves watermark robustness under image perturbations while preserving image quality.

## Background & Motivation

Watermarking AI-generated images is essential for content provenance, misuse prevention, and training data quality control. Current research focuses predominantly on watermark embedding in diffusion models, leaving autoregressive (AR) image generation models comparatively understudied. As AR image models such as LlamaGen and RAR advance rapidly, this gap has become increasingly pressing.

AR image models generate images by autoregressively predicting sequences of visual tokens defined by a VQ-VAE codebook. Drawing inspiration from KGW watermarking in LLMs, a natural approach is to directly transfer token-based watermarking to AR image generation—partitioning the vocabulary into "green" and "red" sets at each sampling step conditioned on the previous token, then biasing sampling toward green tokens.

Direct transfer, however, suffers from severe robustness issues. Watermark verification requires re-encoding the image into a token sequence, yet even minor image perturbations (e.g., JPEG compression, Gaussian noise) can cause the VQ-VAE encoder to produce entirely different tokens—because small shifts in latent space during quantization may cause the representation to jump to a different codebook entry. Since green/red partitions in KGW are assigned randomly, visually similar tokens may fall into different sets, causing the watermark signal to collapse sharply after perturbation.

**Core Insight**: If visually similar tokens are clustered together such that all tokens within a cluster belong to the same set (all green or all red), then even if a perturbation causes a token to change, the watermark signal is preserved as long as the new token remains within the same cluster.

## Method

### Overall Architecture

ClusterMark embeds watermarks during the sampling phase of AR image generation. Detection at verification time requires only the VQ-VAE encoder and a secret key, with no access to the original generative model. The pipeline consists of three stages: (1) a preprocessing stage that applies k-means clustering to the VQ-VAE codebook vectors; (2) a generation stage where green/red set partitions are computed based on the cluster label of the previous token, biasing sampling toward tokens in green clusters; and (3) a verification stage where the candidate image is encoded into a token sequence, the proportion of green tokens is computed, and a binomial hypothesis test is applied.

### Key Designs

1. **Codebook-Clustering-Based Green/Red Set Partitioning**:

    - **Function**: Improve watermark robustness under image perturbations.
    - **Mechanism**: K-means clustering is applied to the $|\mathbb{V}|$ codebook vectors of the VQ-VAE to obtain $k$ clusters (experiments show $k=64$ is optimal). At each generation step, a hash $o_i = \text{hash}(\kappa, c(q_{i-1}))$ is computed from the cluster label of the previous token (rather than the token itself), and the green/red partition is defined at the cluster level rather than the token level. A fraction $\gamma$ of clusters are designated as green clusters, and the green set is the union of all tokens in those clusters. A logit bias $\delta$ is added to green tokens to encourage their sampling.
    - **Design Motivation**: After image perturbation, re-encoded tokens will in most cases still fall within the same cluster as the original token, since codebook vectors close in Euclidean space are grouped together. This preserves the green token count. Experiments confirm that this training-free approach raises TPR under JPEG compression from 69% to 96%.

2. **Token/Cluster Classifier (Adversarial Fine-tuning)**:

    - **Function**: Further improve token/cluster reconstruction accuracy under strong perturbations.
    - **Mechanism**: A copy of the VQ-VAE encoder is created with the pre-quantization layer removed and a classification head added. The token classifier predicts the original token index via cross-entropy loss $\mathcal{L}_{TC}$; the cluster classifier predicts the cluster index directly via $\mathcal{L}_{CC}$. Crucially, random perturbations $\phi(\cdot)$ (JPEG, blur, noise, etc.) are applied to input images during training, teaching the model to recover original tokens/clusters from perturbed images. Training uses 100k watermark-free generated images for 30 epochs.
    - **Design Motivation**: The training-free scheme still struggles under salt-and-pepper noise and strong blur. Adversarial training enables the classifier to effectively "undo" perturbation effects, boosting TPR under salt-and-pepper noise from 40% to near 100%.

3. **Prefix Tuning**:

    - **Function**: Eliminate false positives caused by specific hash prefixes.
    - **Mechanism**: The choice of secret key $\kappa$ has a significant impact on detection performance. Certain values of $\kappa$ cause large uniform regions (e.g., white backgrounds) to produce repeated token bigrams, leading to high green-token proportions even in non-watermarked images. This is addressed by evaluating multiple values of $\kappa$ on a validation set and selecting the best-performing one.
    - **Design Motivation**: This issue is more pronounced with fewer clusters (variance is extremely high at $k=8$), since a small number of clusters makes transition patterns more susceptible to triggering by uniform regions.

### Loss & Training

Token classifier loss: $\mathcal{L}_{TC} = \mathbb{E}[\sum_i \text{CE}(\mathcal{M}_T(\phi(x))_i, q_i)]$; cluster classifier loss: $\mathcal{L}_{CC} = \mathbb{E}[\sum_i \text{CE}(\mathcal{M}_C(\phi(x))_i, c(q_i))]$. Training employs a linearly increasing perturbation strength schedule and completes in approximately 12 hours on a single A40 GPU. Watermark detection uses a right-tailed binomial test; a p-value below threshold $\rho$ indicates a watermarked image.

## Key Experimental Results

### Main Results

**LlamaGen GPT-B (256×256), clustering $k=64$ + Cluster Classifier**

| Perturbation | AUC / TPR@1%FPR | Prev. SOTA (IndexMark) | Gain |
|---|---|---|---|
| Clean | 1.000 / 1.000 | 1.000 / 1.000 | On par |
| JPEG 20 | 0.982 / 0.893 | 0.969 / 0.821 | +7.2% TPR |
| Gaussian Blur R3 | 0.992 / 0.925 | 0.761 / 0.171 | +75.4% TPR |
| Gaussian Noise σ=0.2 | 0.982 / 0.895 | 0.631 / 0.055 | +84.0% TPR |
| Salt-and-Pepper 0.1 | 1.000 / 0.999 | 0.635 / 0.071 | +92.8% TPR |
| Regeneration Attack | 0.993 / 0.935 | 0.951 / 0.761 | +17.4% TPR |

**Image Quality (FID)**

| Method | FID (↓) | Notes |
|---|---|---|
| No watermark baseline | 6.01 | LlamaGen GPT-B |
| ClusterMark ($k=64$) | 6.12 | +0.11 only |
| IndexMark | 5.84 | — |
| SSL | 6.19 | Post-processing watermark |

### Ablation Study

| Configuration | JPEG TPR | Gaussian Noise TPR | Salt-and-Pepper TPR | Notes |
|---|---|---|---|---|
| No Clustering | 0.692 | 0.075 | 0.069 | Direct KGW transfer |
| No Clustering + Token Clf | 0.564 | 0.651 | 0.998 | Classifier helps but inconsistently |
| Clustering $k=64$ | 0.956 | 0.369 | 0.402 | Clustering yields large gains |
| Clustering + Token Clf | 0.875 | 0.900 | 1.000 | Clustering + classifier strongest |
| Clustering + Cluster Clf | 0.893 | 0.895 | 0.999 | Direct cluster prediction also effective |

### Key Findings

- Cluster count $k=64$ offers the best trade-off between quality and robustness; $k<64$ improves robustness but noticeably degrades FID.
- Green fraction $\gamma=0.25$ yields substantially higher robustness than $\gamma=0.5$, at a slight cost to FID.
- Verification is extremely fast (10–25 ms/image), comparable to lightweight post-processing watermarks and far faster than diffusion model watermarking (which requires a full reverse diffusion pass).
- The watermark remains vulnerable to geometric transformations (rotation, cropping), though this can be mitigated with image synchronization layers such as SyncSeal.

## Highlights & Insights

- **Elegant simplicity of the clustering idea**: A training-free codebook clustering alone raises TPR under JPEG from 69% to 96%, demonstrating that the core bottleneck of watermark robustness lies not in the model but in the structure of the token space.
- **Verification requires no generative model**: Only the VQ-VAE encoder and the secret key are needed, resulting in extremely low computational overhead—a crucial advantage for practical deployment, and a sharp contrast with diffusion model watermarking that requires a complete reverse diffusion process.
- **Adversarial training as a perturbation-undoing mechanism**: The classifier improves from near-unusable to near-perfect performance under salt-and-pepper noise, demonstrating a general methodology for compensating the quantization fragility of VQ-VAE through adversarial training.

## Limitations & Future Work

- The approach remains vulnerable to geometric transformations (rotation, cropping) and requires an additional image synchronization layer.
- Secret key selection relies on empirical search; theoretically principled strategies for green/red partitioning warrant further investigation.
- Clustering reduces the effective resolution of the codebook; image quality degrades noticeably when $k$ is too small.
- Validation is conducted only on class-conditional generation; further testing on text-to-image AR models (e.g., Emu-3) is needed.

## Related Work & Insights

- **vs. IndexMark**: IndexMark pairs similar tokens but assigns them to opposite sets (one green, one red); this work takes the opposite approach—assigning similar tokens to the same set. ClusterMark shows a decisive advantage under strong perturbations.
- **vs. WMAR**: WMAR also trains a token reconstructor but additionally fine-tunes the VAE decoder, making training more complex and affecting image appearance. The proposed method is comparatively simpler.
- **vs. LLM KGW Watermarking**: Text tokens are discrete and semantically well-defined; image tokens require clustering to compensate for VQ-VAE quantization fragility.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The clustering idea is intuitively clear and the training-free solution is elegant, though it builds upon the existing KGW framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive ablations across 3 models × 7 perturbation types × multiple configurations with thorough comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Algorithmic descriptions are clear and figures are informative.
- **Value**: ⭐⭐⭐⭐ — A significant advance in watermarking for AR image models with strong practical utility (fast verification + high robustness).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)
- [\[CVPR 2026\] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] Computation and Communication Efficient Federated Unlearning via On-server Gradient Conflict Mitigation and Expression](computation_and_communication_efficient_federated_unlearning_via_on-server_gradi.md)

</div>

<!-- RELATED:END -->
