---
title: >-
  [Paper Note] Dataset Distillation as Pushforward Optimal Quantization
description: >-
  [ICLR2026][Model Compression][Dataset Distillation] This work reformulates decoupled dataset distillation as an optimal quantization problem…
tags:
  - "ICLR2026"
  - "Model Compression"
  - "Dataset Distillation"
  - "Optimal Quantization"
  - "Wasserstein Distance"
  - "Diffusion Models"
  - "Latent Space Clustering"
date: 2026-05-08
content_hash: c9b8c5150a3fead1
---

# Dataset Distillation as Pushforward Optimal Quantization

**Conference**: ICLR2026
**arXiv**: [2501.07681](https://arxiv.org/abs/2501.07681)
**Code**: None
**Area**: Model Compression
**Keywords**: Dataset Distillation, Optimal Quantization, Wasserstein Distance, Diffusion Models, Latent Space Clustering

## TL;DR
This work reformulates decoupled dataset distillation as an optimal quantization problem, proves that latent-space clustering with learned weights via a diffusion prior can converge to approximate the true data distribution, and proposes the DDOQ algorithm, which surpasses baselines such as D4M on ImageNet-1K with minimal additional computation.

## Background & Motivation

### Limitations of Prior Work

**Background**: Dataset distillation (DD) aims to find a small synthetic training set such that models trained on it achieve performance close to those trained on the full dataset. Early bi-level optimization methods incur high computational cost and depend on model architecture. **Decoupled methods** (e.g., SRe2L, D4M) bypass pixel-space optimization by matching data distributions with generative techniques, yet lack theoretical guarantees — no prior work has theoretically established whether a distilled dataset can reasonably approximate the original data distribution.

**Key Challenge**: Methods such as D4M perform $k$-means clustering in latent space followed by decoding, which is essentially solving a **Wasserstein barycenter** problem (with uniform weights). Classical **optimal quantization** theory shows that incorporating automatically learned weights can substantially reduce the Wasserstein distance.

## Method

### Overall Architecture
DDOQ (Dataset Distillation by Optimal Quantization) follows a four-step pipeline:
1. **Encoding**: Map training samples to latent space via an LDM encoder: $Z = \mathcal{E}(\mathcal{T})$
2. **Clustering**: Apply mini-batch $k$-means (i.e., the CLVQ algorithm) per class to obtain $K$ centroids $z_k^{(L)}$ and corresponding weights $w_k^{(L)}$
3. **Decoding**: Generate distilled images via the LDM decoder and diffusion model: $x_k^{(L)} = \mathcal{D} \circ \mathcal{U}_t(z_k^{(L)}, \text{emb})$
4. **Weighted Training**: Train new models using a weighted loss: $\min_\theta \sum_{(x,y,w)} w \cdot \ell(x,y,\theta)$

### Key Designs
**Theorem 1 (Consistency)**: For VESDE or VPSDE diffusion processes, if the Wasserstein-2 distance between latent distributions $\mu_T$ and $\nu_T$ is $\mathcal{W}_2(\mu_T, \nu_T)$, then after reverse diffusion to image space:
$$\|\mathbb{E}_{\mu_\delta}[f] - \mathbb{E}_{\nu_\delta}[f]\| \leq C \cdot L \cdot \mathcal{W}_2(\mu_T, \nu_T)$$
That is, diffusion generation **preserves distributional proximity** — a good approximation in latent space remains a good approximation in image space.

**Corollary 1 (Convergence Rate)**: As the number of quantization points $K$ increases, the approximation error converges at rate $\mathcal{O}(K^{-1/d})$ (where $d$ is the latent space dimensionality), theoretically establishing the consistency of decoupled distillation methods.

**Core Improvement**: Compared to D4M, the only addition is **automatically determined weights** (naturally produced during CLVQ clustering), yielding an average Wasserstein-2 distance reduction of **15.7%** (IPC=10) and **16.1%** (IPC=50).

## Key Experimental Results

**ImageNet-1K (UNet backbone, evaluated with ResNet-18)**:

### Main Results

| IPC | SRe2L | D4M | RDED | **DDOQ** |
|-----|-------|-----|------|----------|
| 10 | 21.3% | 27.9% | 42.0% | **33.1%** |
| 50 | 46.8% | 55.2% | 56.5% | **56.2%** |
| 100 | 52.8% | 59.3% | — | **60.1%** |
| 200 | 57.0% | 62.6% | — | **63.4%** |

- IPC 200 + ResNet-101: DDOQ 68.6% vs. D4M 68.1%, achieving a **30% error reduction** relative to the full-data accuracy of 69.8%
- Cross-architecture generalization (IPC=50): DDOQ consistently outperforms D4M on CNN student models (e.g., MobileNet-V2: 52.1% vs. 47.9%)

**DiT backbone (DDOQ-DiT)**:

### Ablation Study

| Dataset | IPC | Minimax-IGD | **DDOQ-DiT** |
|---------|-----|-------------|-------------|
| ImageNet-1K | 10 | 46.2% | **53.0%** |
| ImageWoof | 10 | 43.3% | **48.8%** |
| ImageNette | 10 | 65.3% | **68.2%** |

- A stronger DiT backbone improves ImageNet-1K IPC=10 accuracy from 33.1% to **53.0%** (+19.9 points)

**Cross-Architecture Generalization Details (IPC=50, ResNet-18 teacher)**:
- ResNet-18 student: DDOQ 56.2% vs. D4M 55.2%
- MobileNet-V2 student: DDOQ 52.1% vs. D4M 47.9% (+4.2 points)
- EfficientNet-B0 student: DDOQ 58.0% vs. D4M 55.4% (+2.6 points)
- Swin-T student: DDOQ 57.4% vs. D4M 58.1% (−0.7 points)

**Wasserstein Distance Analysis**: Incorporating weights reduces the $\mathcal{W}_2$ distance between distilled latent points and encoded training data by an average of 15.7% at IPC=10 and 16.1% at IPC=50, confirming that optimal quantization outperforms the Wasserstein barycenter formulation.

## Highlights & Insights
1. **Solid Theoretical Contributions**: This work is the first to prove consistency and convergence rates for decoupled distillation methods under a diffusion prior, filling a notable theoretical gap in the field.
2. **Minimal Modification**: Compared to D4M, the only change is the introduction of automatically learned weights, incurring virtually no additional computation since the weights arise naturally from the $k$-means process.
3. **Optimal Quantization Perspective**: The work reveals that clustering methods such as $k$-means fundamentally solve an optimal quantization problem, where weights correspond to the measure of Voronoi cells.
4. **Theoretical Guarantees for Diffusion Models**: Theorem 1 proves that diffusion generation preserves distributional proximity, providing theoretical justification for operating in latent space rather than pixel space.

## Limitations & Future Work
- At low IPC settings (e.g., IPC=10), DDOQ still lags behind RDED's patch-based approach (RDED 42.0% vs. DDOQ 33.1% with UNet backbone).
- DDOQ slightly underperforms D4M on Transformer student architectures such as Swin-T (57.4% vs. 58.1%), potentially requiring more careful hyperparameter tuning.
- The convergence rate $\mathcal{O}(K^{-1/d})$ slows as latent space dimensionality $d$ increases, potentially weakening performance in high-dimensional latent settings.
- The method relies on the quality of pre-trained LDM/DiT backbones, with generated image fidelity bounded by the capacity of the underlying model.
- Soft labels require an additional pre-trained classifier (e.g., ResNet-18), and peak performance is thus capped by that classifier's accuracy (69.8%).
- The potential combination with diffusion-guided methods (e.g., IGD) remains unexplored; the two approaches may be complementary.

## Related Work & Insights
- **Comparison with D4M**: Adding weights alone yields consistent improvements, demonstrating that upgrading from Wasserstein barycenter to optimal quantization is the key factor.
- **Comparison with RDED**: RDED is stronger at low IPC but does not scale well; DDOQ performs better at high IPC and maintains constant memory usage.
- The optimal quantization framework is extensible to other scenarios requiring data approximation, such as data summarization in federated learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Optimal quantization perspective + consistency proof; outstanding theoretical contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-IPC, multi-architecture evaluation on ImageNet-1K; limited dataset variety)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous theoretical derivations and clear algorithmic descriptions)
- Value: ⭐⭐⭐⭐⭐ (Provides theoretical foundations for dataset distillation with a concise and effective method)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](../../NeurIPS2025/model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)
- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](compute-optimal_quantization-aware_training.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](understanding_dataset_distillation_via_spectral_filtering.md)
- [\[ICLR 2026\] Grounding and Enhancing Informativeness and Utility in Dataset Distillation](grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)

</div>

<!-- RELATED:END -->
