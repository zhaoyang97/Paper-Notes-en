---
title: >-
  [Paper Note] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation
description: >-
  [NeurIPS 2025][Model Compression][Dataset Distillation] This paper reformulates dataset distillation as an optimal transport (OT) distance minimization problem and achieves fine-grained distributional geometry alignment…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Dataset Distillation"
  - "Optimal Transport"
  - "Distribution Alignment"
  - "Diffusion Models"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: fb08b203b7cb2d16
---

# Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2512.00308](https://arxiv.org/abs/2512.00308)  
**Code**: None  
**Area**: Model Compression / Dataset Distillation
**Keywords**: Dataset Distillation, Optimal Transport, Distribution Alignment, Diffusion Models, Knowledge Distillation

## TL;DR

This paper reformulates dataset distillation as an optimal transport (OT) distance minimization problem and achieves fine-grained distributional geometry alignment through a three-stage pipeline (OT-guided diffusion sampling, label-image alignment soft re-labeling, and OT logit matching), yielding at least 4% improvement over the previous state of the art on ImageNet-1K at IPC=10.

## Background & Motivation

Dataset distillation aims to synthesize a compact dataset such that models trained on it achieve performance comparable to training on the full dataset. Large-scale distillation methods can be broadly categorized into two families: **model inversion methods** (e.g., SRe2L, RDED, EDC), which rely on global batch normalization statistics of pretrained models but are inherently unable to recover instance-level local distributional structure; and **generative model methods** (e.g., IGD, D4M), which incorporate real images into the sampling process but still focus on matching global gradient statistics. Their diversity guidance based on cosine similarity fails to capture fine-grained distributional structure, leading to local mode collapse and distribution mismatch.

The root cause is that existing methods only match global statistics (e.g., mean and variance) while neglecting critical instance-level features and intra-class variation. Distributions sharing the same mean or variance can be geometrically entirely different.

The paper's starting point is that each real data point encodes rich intra-class semantic variation (e.g., sub-class features within a category), and **optimal transport (OT)** naturally provides a geometrically faithful and perceptually aligned measure of distributional discrepancy, making it particularly well-suited for preserving and transferring such fine-grained semantic structures.

## Method

### Overall Architecture

The dataset distillation objective is formalized as minimizing the Wasserstein distance $W(\mu_{\text{true}}, \nu_{\text{new}})$ between the real distribution $\mu_{\text{true}}$ and the student-model-induced distribution $\nu_{\text{new}}$. Exploiting the triangle inequality of the Wasserstein distance, the total OT cost is decomposed into three optimizable terms:

$$W(\mu_{\text{true}}, \nu_{\text{new}}) \leq \underbrace{E_y[W(\mu_{\text{true}}(\mathbf{x}|y), \nu_{\text{distill}}^{(\text{hard})}(\mathbf{x}|y))]}_{\text{OT-guided sampling}} \cdot \underbrace{\alpha(\nu_{\text{distill}}^{(\text{soft})})}_{\text{label-image alignment}} + \underbrace{W(\nu_{\text{distill}}^{(\text{soft})}, \nu_{\text{new}})}_{\text{OT logit matching}}$$

The three stages correspond respectively to image generation, label assignment, and student model training.

### Key Designs

1. **OT-Guided Diffusion Sampling (OTG)**: During the reverse diffusion sampling process, the OT distance between the accumulated synthetic images and a batch of real images in latent space is computed as the guidance function. For each class $c$, when generating the $n$-th latent variable, the Sinkhorn algorithm efficiently computes the OT transport matrix $\mathbf{P}^{\lambda_1}$ to derive the guidance gradient. The sampling update rule is:
    $\mathbf{z}_{t-1}^c = s(\mathbf{z}_t^c, t, \epsilon_\phi) - \rho_t \nabla \mathcal{G}_I - \gamma_t \nabla \mathcal{G}_D - \beta_1 \nabla \mathcal{G}_W(\mathbf{z}_t^c)$
   where $\mathcal{G}_W$ is the OT guidance term, simultaneously accounting for global and local structural information to promote fine-grained geometric alignment.

2. **Label-Image Alignment Soft Re-labeling (LIA)**: The teacher model ensemble is adaptively selected according to IPC. At low IPC, where the image distribution has limited expressiveness, fewer teachers are used to generate low-entropy simplified soft labels to avoid overfitting; at high IPC, more teachers are employed to produce fine-grained soft labels that capture the true label space structure. The formula is:
    $\mathbf{t}(\mathbf{x}_i) = \frac{1}{|\mathbb{T}(\text{IPC})|} \sum_{t \in \mathbb{T}(\text{IPC})} F_t(\mathbf{x}_i)$
   This ensures that the complexity of the soft label distribution matches the capacity of the distilled images, thereby reducing the contraction factor $\alpha$.

3. **OT Logit Matching (OTM)**: During student model training, batch-level OT distance is used to align the student's logit outputs with the soft label distribution. Unlike conventional sample-wise KL divergence or MSE, OT matching captures inter-sample relationships. The Sinkhorn method is used to compute the batch-level cost matrix and transport matrix, and the total loss is:
    $\mathcal{L} = \kappa_1 \mathcal{L}_{\text{CE}} + \kappa_2 \mathcal{L}_{\text{MSE}} + \beta_2 \mathcal{L}_{\text{SD}}$

### Loss & Training

The three stages are optimized sequentially: OTG first generates the distilled image set, LIA then performs IPC-adaptive soft re-labeling, and OTM finally trains the student model. OT computation is solved via Sinkhorn normalized iterations using the $\ell_1$ norm as the cost matrix.

## Key Experimental Results

### Main Results

| Dataset | Architecture | Metric | Ours (300ep) | Ours (1000ep) | DiT-IGD | EDC | Gain |
|--------|------|------|------------|-------------|---------|-----|------|
| ImageNet-1K IPC=10 | ResNet-18 | Top-1 Acc | 52.9 | **58.6** | 45.5 | 48.6 | +10.0 |
| ImageNet-1K IPC=50 | ResNet-18 | Top-1 Acc | 61.9 | **64.2** | 59.8 | 58.0 | +4.4 |
| ImageNet-1K IPC=10 | MobileNet-V2 | Top-1 Acc | 51.0 | **57.6** | 39.2 | 45.0 | +12.6 |
| ImageNet-1K IPC=10 | Swin-T | Top-1 Acc | 50.2 | **63.7** | 44.1 | 46.0 | +17.7 |
| ImageNet-1K IPC=10 | ConvNeXt | Top-1 Acc | 61.2 | **67.0** | 51.9 | 54.4 | +12.6 |
| ImageNette IPC=10 | ResNet-18 | Top-1 Acc | **79.0** | - | 74.8 | - | +4.2 |
| CIFAR-100 IPC=10 | ConvNet-3 | Top-1 Acc | **50.7** | - | 45.8 | - | +4.9 |

### Ablation Study

| Configuration | ConvNet-6 | ResNetAP-10 | ResNet-18 | Note |
|------|-----------|-------------|-----------|------|
| w/o OTG (hard label) | 61.9 | 66.5 | 67.7 | Baseline (IGD sampling) |
| w OTG (hard label) | 67.0 | 68.0 | 69.1 | +OT sampling guidance |
| w/o OTG (soft label) | 72.5 | 74.2 | 77.2 | Soft label baseline |
| w/o LIA | 74.3 | 76.4 | 77.8 | Unaligned soft labels |
| w/o OTM | 73.2 | 75.9 | 77.5 | No OT logit matching |
| Full | **74.5** | **77.8** | **79.0** | Complete method |

**Runtime overhead**: The additional time cost introduced by OT constraints remains consistently below 1%, and the total distillation set generation time (3.7h) is substantially faster than EDC (11.4h).

### Key Findings

- Performance gains are more pronounced at lower IPC, indicating that the OT framework is more effective at preserving fine-grained distributional details when samples are extremely scarce.
- The method consistently outperforms all baselines across CNN, Transformer, and hybrid architectures, demonstrating strong cross-architecture generalizability.
- The distilled images contain sufficient information to benefit from longer training schedules (continuous improvement from 300 to 1000 epochs).

## Highlights & Insights

- Formalizing dataset distillation as OT distance minimization constitutes an elegant theoretical framework; the triangle inequality decomposition provides a clear optimization objective for each stage.
- The OT-guided sampling is inherently cumulative: when generating the $n$-th sample, the OT distance between the preceding $n{-}1$ generated samples and the real data is taken into account, achieving a genuinely globally optimal distributional match.
- The IPC-adaptive soft labeling strategy, grounded in the insight of distribution capacity matching, is simple yet effective.

## Limitations & Future Work

- The current approach adopts the $\ell_1$ norm as the OT cost matrix; semantically aware cost functions are worth exploring.
- The entropy regularization parameter of the Sinkhorn algorithm requires tuning and may need different settings across datasets.
- Performance trends under larger IPC settings (e.g., 200, 500) remain unexplored.

## Related Work & Insights

- **vs IGD**: OT guidance is added on top of IGD's trajectory and diversity guidance, replacing cosine-similarity-based diversity metrics with geometrically faithful distributional matching.
- **vs EDC**: Represents an entirely different paradigm—generative vs. model inversion—with a 3× speedup in generation time.
- **vs RDED**: Addresses the fundamental limitation of model inversion methods—the lack of fine-grained alignment—by introducing instance-level OT matching.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of OT and distillation has theoretical depth, and the three-stage decomposition is elegant, though the individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six architectures, four datasets, multiple IPC settings, with complete ablations and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear; the logical flow from problem formulation to method design is rigorous.
- Value: ⭐⭐⭐⭐ Achieves significant progress in large-scale dataset distillation with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](hyperbolic_dataset_distillation.md)
- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)
- [\[NeurIPS 2025\] Beyond Random: Automatic Inner-Loop Optimization in Dataset Distillation](beyond_random_automatic_inner-loop_optimization_in_dataset_distillation.md)
- [\[AAAI 2026\] Lightweight Optimal-Transport Harmonization on Edge Devices](../../AAAI2026/model_compression/lightweight_optimal-transport_harmonization_on_edge_devices.md)

</div>

<!-- RELATED:END -->
