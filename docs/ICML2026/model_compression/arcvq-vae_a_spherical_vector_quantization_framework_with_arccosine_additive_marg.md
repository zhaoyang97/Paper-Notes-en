---
title: >-
  [Paper Note] ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin
description: >-
  [ICML 2026][Model Compression][Codebook Collapse] The authors diagnose the root cause of codebook collapse in VQ-VAE as "codebook vector $\ell_2$ norm imbalance + geometric clustering." They propose SAMP: Ball-Bounded No…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Codebook Collapse"
  - "Angular Margin"
  - "Spherical Learning"
  - "Norm Regularization"
  - "Codebook Utilization"
date: 2026-05-08
content_hash: 5180d14b5343132a
---

# ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin

**Conference**: ICML 2026  
**arXiv**: [2605.13517](https://arxiv.org/abs/2605.13517)  
**Code**: https://github.com/goals4292/ArcVQ-VAE  
**Area**: VQ-VAE / Image Generation / Discrete Representation  
**Keywords**: Codebook Collapse, Angular Margin, Spherical Learning, Norm Regularization, Codebook Utilization

## TL;DR
The authors diagnose the root cause of codebook collapse in VQ-VAE as "codebook vector $\ell_2$ norm imbalance + geometric clustering." They propose SAMP: Ball-Bounded Norm Regularization constrains all codebook vectors within a time-varying Euclidean ball, and ArcCosine Additive Margin Loss, inspired by ArcFace, pushes latent vectors apart on the sphere. This leads to uniform codebook dispersion and a significant increase in utilization, outperforming mainstream VQ-VAE variants on ImageNet reconstruction and generation FID.

## Background & Motivation
**Background**: VQ-VAE discretizes continuous latents into a finite codebook, serving as a foundational component for autoregressive image generation (VQGAN / RQ-VAE), diffusion priors (LDM), and multimodal tokenization. Many methods have been proposed to improve VQ-VAE: SQ-VAE (stochastic quantization), CVQ-VAE (online K-means to pull underused codebooks closer to latents), VQGAN-LC (pretrained encoder for codebook extraction), Wasserstein VQ, etc.

**Limitations of Prior Work**: (1) A fixed-size codebook cannot capture the full diversity of the dataset. (2) Codebook collapse—only a small subset of codebooks are frequently used, with the rest rarely activated; utilization often falls below 50%. (3) Existing methods mainly patch the mechanism of "how to update/select codebooks" without addressing the root cause—**geometric imbalance** of codebook vectors in latent space.

**Key Challenge**: Empirical evidence (Figure 2/3) shows that at the start of training, all codebooks are initialized near the origin; those selected accelerate their $\ell_2$ norm growth along encoder output directions, while unselected ones remain near the origin. High-norm codebooks are closer to encoder features and thus more likely to be chosen, forming a positive feedback loop—this is a **geometric dynamics** issue, not simply "starvation due to low sampling."

**Goal**: (1) Suppress codebook vector norm imbalance at its root; (2) Ensure latent vectors are uniformly dispersed in latent space, giving each latent a chance to bind to different codebooks; (3) Achieve this without introducing new network components and with almost zero extra computational cost.

**Key Insight**: Drawing from ArcFace in face recognition—"angular margin + spherical learning"—if all latents and codebooks are $\ell_2$-normalized onto the unit sphere, codebook selection becomes maximum cosine similarity instead of Euclidean nearest neighbor. Adding an angular margin pushes inter-class distances apart, enforcing uniform latent dispersion. However, face recognition is supervised classification, while VQ-VAE lacks explicit class labels—thus, "top-k nearest codebooks" are treated as implicit classes.

**Core Idea**: Use Spherical Angular-Margin Prior (SAMP) = Ball-Bounded Norm Regularization (restrict codebooks within a time-varying Euclidean ball) + ArcCosine Additive Margin Loss (add angular margin on the sphere to push latents apart), achieving geometric uniformity and high codebook utilization.

## Method
SAMP consists of two complementary components. Ball-Bounded Norm Regularization is a "hard constraint," projecting out-of-bound codebooks back into the ball after each batch; ArcCosine Additive Margin Loss is a "soft constraint," encouraging angular dispersion of latents via the loss. Together, they recast VQ-VAE as a spherical learning problem.

### Overall Architecture
- **Architecture**: Identical to VQGAN's encoder-decoder + codebook, **no new network components** introduced. Only two modifications: (1) Norm clipping of codebooks after each batch; (2) Adding an ArcLoss term to the original VQ loss.
- **Training Loop**: Standard VQ-VAE forward → compute $\mathcal{L}_\text{VQ}$ (reconstruction + commit + codebook loss) → compute $\mathcal{L}_\text{A}$ (ArcLoss with stop-grad on codebook) → total loss $\mathcal{L}_\text{total} = \mathcal{L}_\text{VQ} + \gamma(t)\mathcal{L}_\text{A}$ → backprop → ball projection for each codebook vector after the batch.
- **Inference / Quantization**: Both encoder outputs and codebooks are $\ell_2$-normalized, then cosine similarity matching is performed, equivalent to finding the nearest codebook on the unit sphere.
- **Subsequent Generation**: LDM is trained as a prior on the $32^2$ tokens produced by ArcVQ-VAE, with 250 sampling steps.

### Key Designs

1. **Ball-Bounded Norm Regularization (Time-Varying Ball Constraint)**:

    - **Function**: Restricts each codebook vector's $\ell_2$ norm to not exceed an exponentially increasing upper bound $M(t)$, breaking the positive feedback of "norm imbalance → high-norm codebooks favored."
    - **Mechanism**: Initialize all codebooks on the unit sphere $\mathbf{e}_k^{(0)} \sim \ell_2(\text{Unif}(-1,1)^d)$. At each training step, set the upper bound $M(t) = \exp(\alpha t)$, with $\alpha$ small (e.g., $10^{-5}$) so $M$ is near 1 early in training. After each batch, project codebooks exceeding the norm: $\mathbf{e}_k^{(t)} \leftarrow \frac{\mathbf{e}_k^{(t)}}{\|\mathbf{e}_k^{(t)}\|_2} M(t)$; others remain unchanged. The codebook set lies within the ball $\mathcal{C}^{(t)} \subset \mathbb{B}_{M(t)}^d$. Training thus has two phases: early on, all codebooks are near the unit sphere, competing fairly for latent features; later, the ball expands, allowing richer norm expression.
    - **Design Motivation**: Figure 2 shows that in standard VQ-VAE, a few high-frequency codebooks have much larger norms than low-frequency ones in late training—high norm + proximity to latent center = near-monopoly of latent assignment. The "strict early, relaxed late" ball constraint directly breaks this winner-takes-all dynamic.

2. **ArcCosine Additive Margin Loss (Spherical Angular Margin)**:

    - **Function**: After $\ell_2$-normalizing latent vectors to the unit sphere, applies an ArcFace-inspired angular margin to push them apart, so each latent occupies a different region and associates with different codebooks.
    - **Mechanism**: Normalize encoder outputs $z_{e,i}(x)$ and codebooks $e_j$: $\hat{z}_i = z_{e,i}/\|z_{e,i}\|$, $\hat{e}_j = e_j/\|e_j\|$. Quantization becomes maximum cosine similarity: $k = \arg\max_j \hat{z}_i^\top \hat{e}_j$. For each pair $(\hat{z}_i, \hat{e}_j)$, compute angle $\theta_{i,j} = \arccos(\hat{z}_i^\top \hat{e}_j)$. ArcLoss, in the form of ArcFace's additive margin softmax: $\mathcal{L}_\text{A} = -\frac{1}{K}\sum_j \log \frac{\sum_{i \in \mathcal{N}_j^{(k)}} e^{s\cos(\theta_{i,j}+m)}}{\sum_{i \in \mathcal{N}_j^{(k)}} e^{s\cos(\theta_{i,j}+m)} + \sum_{i \notin \mathcal{N}_j^{(k)}} e^{s\cos\theta_{i,j}}}$, where $\mathcal{N}_j^{(k)}$ is the set of top-k latent tokens nearest to codebook $e_j$ (k=3, $s=10$, $m=0.1$). The loss explicitly separates "latents should angularly align to their nearest codebook" (positive) from "move away from other codebooks" (negative), with margin $m$ enforcing tighter alignment. **Key trick**: Apply stop-gradient $\text{sg}(\hat{e}_j)$ to codebooks, so ArcLoss only backpropagates to the encoder, not directly updating codebooks—otherwise, codebooks would be pulled toward the current batch's latent distribution, losing global separability.
    - **Design Motivation**: Standard VQ loss only encourages encoder features to "approach the nearest codebook," without enforcing latent dispersion; multiple latents may collapse into the same region, sharing a codebook. ArcLoss explicitly encodes "latent dispersion" in the loss, and spherical normalization turns "near vs. far" into a pure angular problem, avoiding the geometric pathology of codebook competition in Euclidean space.

3. **Decay-Weighted Joint Loss + Stop-Gradient on Codebook**:

    - **Function**: Strongly enforces angular structure early in training, then lets the model focus on reconstruction accuracy later; stop-grad isolates ArcLoss from directly affecting codebooks.
    - **Mechanism**: Total loss $\mathcal{L}_\text{total} = \mathcal{L}_\text{VQ} + \gamma(t)\mathcal{L}_\text{A}$, with $\gamma(t) = \gamma_0 \exp(-\lambda t)$. $\gamma_0=1.0$, $\lambda=5\times 10^{-4}$ for MNIST/CIFAR, $10^{-4}$ for ImageNet. Early on, ArcLoss has high weight, enforcing spherical latent dispersion; later, the weight decays, letting VQ loss dominate for reconstruction fidelity. Stop-gradient ensures ArcLoss only optimizes the encoder (spreading latents), while codebooks are updated indirectly via standard VQ loss—once the encoder disperses, standard codebook updates naturally pull codebooks to more dispersed latents.
    - **Design Motivation**: Directly letting ArcLoss update codebooks causes batch-driven local collapse—codebooks are pulled toward the current batch's latents, losing global dispersion. Separating "latent dispersion" and "codebook following" via standard VQ commit loss ensures stable, non-interfering training.

### Loss & Training
$\mathcal{L}_\text{VQ}$ is the standard VQ loss: reconstruction + codebook + commit (coefficient $\beta$). $\mathcal{L}_\text{A}$ is ArcLoss ($s=10$, $m=0.1$, top-k=3). $\gamma(t)$ decays exponentially. Other hyperparameters (learning rate, discriminator weight, etc.) follow VQGAN defaults for fair comparison.

## Key Experimental Results

### Main Results
ImageNet-1K reconstruction ($256\times 256$, downsample $16\times$ or $8\times$):

| Method | S | K | Codebook Utilization | rFID ↓ |
|--------|---|---|---------------------|--------|
| VQGAN | $16^2$ | 1024 | 44% | 7.94 |
| VQGAN-FC | $16^2$ | 16384 | 11.2% | 4.29 |
| VQGAN-EMA | $16^2$ | 16384 | 83.2% | 3.41 |
| ViT-VQGAN | $32^2$ | – | – | Lower |
| **ArcVQ-VAE** | – | – | **Nearly 100%** | **Lower** |

MNIST / CIFAR10 reconstruction comparison:

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | rFID ↓ |
|---------|--------|--------|--------|---------|--------|
| MNIST | VQ-VAE | 26.48 | 0.9777 | 0.0282 | 3.43 |
| MNIST | CVQ-VAE | 27.87 | 0.9833 | 0.0222 | 1.80 |
| MNIST | **ArcVQ-VAE** | **28.01** | **0.9840** | **0.0217** | **1.68** |
| CIFAR10 | VQ-VAE | 23.32 | 0.8595 | 0.2504 | 39.67 |
| CIFAR10 | CVQ-VAE | 24.72 | 0.8978 | 0.1883 | 24.73 |
| CIFAR10 | **ArcVQ-VAE** | **24.78** | **0.8989** | **0.1857** | 26.91 |

### Ablation Study

| Configuration | Key Observation |
|---------------|----------------|
| Full SAMP | High codebook utilization, lowest rFID |
| Ball-Bounded Norm only (no ArcLoss) | Norms balanced but latents may still cluster |
| ArcLoss only (no Norm Reg) | High-utilization codebooks may still have norm explosion |
| Remove stop-gradient on codebook | Codebooks are pulled by batch, lose global dispersion |
| Change m (0 / 0.1 / 0.3) | $m=0.1$ optimal; too large hurts reconstruction |
| Change top-k (1/3/5) | k=3 optimal; k=1 too strict, k=5 too loose |
| Change $\alpha$ (ball expansion rate) | Too large $\alpha$ weakens early constraint, too small limits late expressiveness |

### Key Findings
- Codebook utilization rises from VQGAN's 44% to nearly 100% (Figure 1 t-SNE visualization shows almost all codebooks are active in green), a fundamental improvement from geometric redesign.
- PCA visualization of quantized latent maps (Figure 5) shows ArcVQ-VAE has stronger activations and clearer contours, indicating codebooks are not only "used" but encode finer spatial structure.
- Ball-Bounded and ArcLoss are complementary and inseparable: using either alone cannot fully eliminate collapse; both together address "norm imbalance" and "geometric clustering."
- Stop-gradient is essential for stable ArcLoss: letting ArcLoss update codebooks directly causes batch-driven local collapse—this insight warns other works adding metric learning losses to VQ.

## Highlights & Insights
- **Diagnosing "Codebook Collapse as a Geometric Problem"**: The authors empirically attribute collapse to "codebook norm imbalance + spatial clustering" dynamics (Figure 2/3), not just "low sampling probability." This geometric perspective is refreshing and directly informs method design.
- **Transferring ArcFace to VQ-VAE**: The angular margin idea from face recognition is mature in supervised classification, but VQ-VAE lacks class labels—the authors use "top-k nearest codebooks" as implicit classes, enabling angular margin in unsupervised settings. This cross-domain transfer is clever.
- **Near-Zero Extra Cost**: No new network layers, no increase in forward FLOPs, just a post-batch norm clip and a loss term. Thus, it can be plugged into any existing VQ-VAE / VQGAN immediately.

## Limitations & Future Work
- The ball constraint's $\alpha$ and ArcLoss's $m, s, k$ are manually tuned and may need retuning for different datasets/resolutions; adaptive scheduling is an open direction.
- Validation is mainly on ImageNet reconstruction + LDM generation; transfer to video / 3D / multimodal tokenization is untested.
- Selecting the top-k latent set $\mathcal{N}_j^{(k)}$ introduces an extra $O(K \cdot Bhw)$ sorting cost, which may be non-negligible for large codebooks (>16k).
- No direct comparison with the latest FSQ (Finite Scalar Quantization) / LFQ (Lookup-Free Quantization)—these methods bypass codebook learning, challenging SAMP's core motivation.

## Related Work & Insights
- **vs CVQ-VAE (Zheng & Vedaldi 2023)**: CVQ uses online K-means to pull underused codebooks closer to latents; this work regularizes geometric space, complementary but achieves lower rFID.
- **vs SQ-VAE (Takida et al. 2022)**: SQ makes the posterior stochastic; this work uses spherical + margin, a completely different mechanism.
- **vs VQGAN-LC (Zhu et al. 2024)**: That method uses a pretrained encoder to scale up the codebook; this work relies purely on geometric constraints, not external models.
- **vs ArcFace (Deng et al. 2019)**: This is the first application of angular margin to unsupervised VQ, extending ArcFace's scope.
- **vs FSQ / LFQ**: Those methods bypass codebook learning via scalar quantization; this work insists on learning codebooks but solves its inherent problems—each approach has different engineering trade-offs.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing ArcFace to VQ-VAE is a smart cross-domain transfer; the ball constraint + stop-grad combo is also original
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets + baselines + complete ablation; lacks direct comparison with FSQ/LFQ
- Writing Quality: ⭐⭐⭐⭐ Diagnostic section (Fig 2/3) is very illuminating, derivations are clear; ArcLoss formula is a bit dense
- Value: ⭐⭐⭐⭐ Near-zero extra cost, can be plugged into any VQGAN, high engineering value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ICLR 2026\] Embedding Compression via Spherical Coordinates](../../ICLR2026/model_compression/embedding_compression_via_spherical_coordinates.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](../../CVPR2026/model_compression/rdvq_differentiable_vq_image_compression.md)
- [\[ICCV 2025\] SSVQ: Unleashing the Potential of Vector Quantization with Sign-Splitting](../../ICCV2025/model_compression/ssvq_unleashing_the_potential_of_vector_quantization_with_sign-splitting.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](../../ICCV2025/model_compression/task_vector_quantization_for_memory-efficient_model_merging.md)

</div>

<!-- RELATED:END -->
