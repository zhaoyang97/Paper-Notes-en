---
title: >-
  [Paper Note] ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin
description: >-
  [ICML 2026][Model Compression][Codebook Collapse] The authors diagnose the root cause of VQ-VAE codebook collapse as "imbalance of codebook vector $\ell_2$ norms + geometric clustering." They propose SAMP: Ball-Bounded N…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Codebook Collapse"
  - "Angular Margin"
  - "Spherical Learning"
  - "Norm Regularization"
  - "Codebook Utilization"
date: 2026-05-08
content_hash: 6b30d0e8c7336c84
---

# ArcVQ-VAE: A Spherical Vector Quantization Framework with ArcCosine Additive Margin

**Conference**: ICML 2026  
**arXiv**: [2605.13517](https://arxiv.org/abs/2605.13517)  
**Code**: https://github.com/goals4292/ArcVQ-VAE  
**Area**: VQ-VAE / Image Generation / Discrete Representations  
**Keywords**: Codebook Collapse, Angular Margin, Spherical Learning, Norm Regularization, Codebook Utilization

## TL;DR
The authors diagnose the root cause of VQ-VAE codebook collapse as "imbalance of codebook vector $\ell_2$ norms + geometric clustering." They propose SAMP: Ball-Bounded Norm Regularization to constrain all codebook vectors within a time-varying Euclidean ball, and ArcCosine Additive Margin Loss to push latent vectors apart on a sphere by drawing inspiration from ArcFace. This ensures uniform codebook distribution and significantly increased utilization, outperforming mainstream VQ-VAE variants in ImageNet reconstruction and FID.

## Background & Motivation
**Background**: VQ-VAE discretizes continuous latents into a finite codebook and serves as a fundamental component for autoregressive image generation (VQGAN / RQ-VAE), diffusion priors (LDM), and multimodal tokenization. Numerous methods exist to improve VQ-VAE, including SQ-VAE (stochastic quantization), CVQ-VAE (using online K-means to pull unused codebooks closer to latents), VQGAN-LC (extracting codebooks via pretrained encoders), and Wasserstein VQ.

**Limitations of Prior Work**: (1) Fixed-size codebooks cannot represent the full richness of datasets. (2) Codebook collapse—only a small portion of the codebook is frequently used while the rest remains inactive, with utilization often below 50%. (3) Existing methods primarily patch mechanics like "how to update/select codebooks" without addressing the fundamental cause—**geometric imbalance** of codebook vectors in the latent space.

**Key Challenge**: Through empirical findings in Figure 2/3, the authors observe that at the start of training, all codebooks are initialized near the origin. Selected codebooks accelerate their $\ell_2$ norm growth along the encoder output direction, while unselected ones remain near the origin. High-norm codebooks stay closer to encoder features and are more likely to be selected, creating a positive feedback loop—this is a **geometric dynamics** issue rather than a simple "starvation due to lack of sampling."

**Goal**: (1) Suppress the norm imbalance of codebook vectors at its root; (2) encourage latent vectors to disperse uniformly in the latent space so each latent has an opportunity to bind to different codebooks; (3) achieve this without introducing new network components and with near-zero extra computational cost.

**Key Insight**: The authors adopt the "angular margin + spherical learning" concept from ArcFace in face recognition. If all latents and codebooks are $\ell_2$-normalized to a unit sphere, codebook selection transitions from Euclidean nearest neighbor to maximum angular cosine matching. Adding an angular margin to push inter-class distances forces latents to disperse uniformly. While face recognition is supervised, the "top-k nearest codebooks" are treated as implicit classes for VQ-VAE.

**Core Idea**: Use Spherical Angular-Margin Prior (SAMP) = Ball-Bounded Norm Regularization (restricting codebooks within a time-varying Euclidean ball) + ArcCosine Additive Margin Loss (pushing latents apart via angular margin on the sphere) to achieve uniform geometric distribution and high codebook utilization.

## Method
SAMP consists of two complementary components. Ball-Bounded Norm Regularization serves as a "hard constraint" that forcibly projects out-of-bounds codebooks back into the ball after each batch. ArcCosine Additive Margin Loss serves as a "soft constraint" to encourage angular dispersion via the loss function. Together, they reshape VQ-VAE as a spherical learning problem.

### Overall Architecture
- **Architecture**: Identical encoder-decoder + codebook architecture as VQGAN, **introducing no new network components**. Changes occur in only two places: (1) norm clipping performed on the codebook after each batch; (2) adding an ArcLoss term to the original VQ loss.
- **Training Cycle**: Standard VQ-VAE forward pass $\rightarrow$ calculate $\mathcal{L}_\text{VQ}$ (reconstruction + commit + codebook loss) $\rightarrow$ calculate $\mathcal{L}_\text{A}$ (ArcLoss with stop-grad on codebook) $\rightarrow$ total loss $\mathcal{L}_\text{total} = \mathcal{L}_\text{VQ} + \gamma(t)\mathcal{L}_\text{A}$ $\rightarrow$ backpropagation $\rightarrow$ ball projection for each codebook vector after the batch.
- **Inference / Quantization**: Encoder outputs and codebooks are $\ell_2$-normalized for angular cosine matching, equivalent to finding the nearest neighbor on a unit sphere.
- **Downstream Generation**: Training an LDM as a prior on the $32^2$ tokens produced by ArcVQ-VAE, with 250 sampling steps.

### Key Designs

1. **Ball-Bounded Norm Regularization (Time-varying Ball Constraint)**:
    - **Function**: Limits the $\ell_2$ norm of each codebook vector to not exceed an exponentially growing upper bound $M(t)$, cutting off the positive feedback loop where "norm imbalance leads to preference for high-norm codebooks."
    - **Mechanism**: Initialize all codebooks on the unit sphere $\mathbf{e}_k^{(0)} \sim \ell_2(\text{Unif}(-1,1)^d)$. Set the upper bound $M(t) = \exp(\alpha t)$ for each training step, where $\alpha$ is small (e.g., $10^{-5}$) to keep $M$ near 1 early in training. After each batch, project codebooks exceeding the norm: $\mathbf{e}_k^{(t)} \leftarrow \frac{\mathbf{e}_k^{(t)}}{\|\mathbf{e}_k^{(t)}\|_2} M(t)$. The set of codebooks remains within the ball $\mathcal{C}^{(t)} \subset \mathbb{B}_{M(t)}^d$. This divides training into two stages: early on, codebooks compete fairly near the unit sphere; later, the ball expands to allow richer norm expression.
    - **Design Motivation**: Figure 2 empirically shows that in traditional VQ-VAE, the norms of high-frequency codebooks far exceed those of low-frequency ones. High norm plus proximity to the latent center area results in a monopoly on latent assignments. This ball constraint breaks winner-takes-all dynamics.

2. **ArcCosine Additive Margin Loss (Spherical Angular Margin)**:
    - **Function**: After $\ell_2$-normalizing latent vectors to a unit sphere, an angular margin derived from ArcFace pushes them apart, ensuring each latent occupies a different spherical region and associates with different codebooks.
    - **Mechanism**: Normalize encoder outputs $z_{e,i}(x)$ and codebooks $e_j$: $\hat{z}_i = z_{e,i}/\|z_{e,i}\|$, $\hat{e}_j = e_j/\|e_j\|$. The quantization rule becomes maximum angular cosine: $k = \arg\max_j \hat{z}_i^\top \hat{e}_j$. Calculate the angle $\theta_{i,j} = \arccos(\hat{z}_i^\top \hat{e}_j)$ for each pair. ArcLoss follows the additive margin softmax form: $\mathcal{L}_\text{A} = -\frac{1}{K}\sum_j \log \frac{\sum_{i \in \mathcal{N}_j^{(k)}} e^{s\cos(\theta_{i,j}+m)}}{\sum_{i \in \mathcal{N}_j^{(k)}} e^{s\cos(\theta_{i,j}+m)} + \sum_{i \notin \mathcal{N}_j^{(k)}} e^{s\cos\theta_{i,j}}}$, where $\mathcal{N}_j^{(k)}$ is the set of top-k latents closest to codebook $e_j$ (parameters used: $k=3, s=10, m=0.1$). This explicitly separates "latents aligning to the nearest codebook" (positive pairs) from "distancing from others" (negative pairs). **Key trick**: apply stop-gradient $\text{sg}(\hat{e}_j)$ to codebooks so ArcLoss only optimizes the encoder without directly modifying codebooks—otherwise, codebooks might collapse toward the current batch distribution.
    - **Design Motivation**: Standard VQ loss only pulls encoder features toward codebooks but does not encourage dispersion. ArcLoss explicitly enforces "latent dispersion." Spherical normalization transforms "proximity" into a pure angular problem, avoiding geometric conflicts where close codebooks compete in Euclidean space.

3. **Decay-Weighted Joint Loss + Stop-Gradient on Codebook**:
    - **Function**: Strongly constrains the angular structure early in training and focuses on reconstruction fidelity later, while isolating ArcLoss from interfering with codebooks via stop-grad.
    - **Mechanism**: Total loss $\mathcal{L}_\text{total} = \mathcal{L}_\text{VQ} + \gamma(t)\mathcal{L}_\text{A}$ where $\gamma(t) = \gamma_0 \exp(-\lambda t)$. With $\gamma_0=1.0$, $\lambda$ is $5\times 10^{-4}$ for MNIST/CIFAR and $10^{-4}$ for ImageNet. Early high ArcLoss weight forces spherical dispersion; later weight decay allows VQ loss to dominate for reconstruction fidelity. Stop-gradient ensures ArcLoss only optimizes the encoder to disperse latents, while codebook updates remain indirect.
    - **Design Motivation**: Allowing ArcLoss to push codebooks directly causes batch-driven local collapse. Separating "latent dispersion" and "codebook following" responsibilities makes training stable.

### Loss & Training
$\mathcal{L}_\text{VQ}$ is the standard VQ loss: reconstruction + codebook + commit (coefficient $\beta$). $\mathcal{L}_\text{A}$ is the ArcLoss ($s=10$, $m=0.1$, top-k=3). $\gamma(t)$ decays exponentially. Other hyperparameters (learning rate, discriminator weight, etc.) follow VQGAN defaults for fair comparison.

## Key Experimental Results

### Main Results
ImageNet-1K Reconstruction ($256\times 256$, downsample $16\times$ or $8\times$):

| Method | S | K | Codebook Utilization | rFID ↓ |
|--------|---|---|-----------|--------|
| VQGAN | $16^2$ | 1024 | 44% | 7.94 |
| VQGAN-FC | $16^2$ | 16384 | 11.2% | 4.29 |
| VQGAN-EMA | $16^2$ | 16384 | 83.2% | 3.41 |
| ViT-VQGAN | $32^2$ | – | – | Lower |
| **ArcVQ-VAE** | – | – | **Near 100%** | **Lower** |

MNIST / CIFAR10 Reconstruction Comparison:

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
|------|----------|
| Full SAMP | High utilization, lowest rFID |
| Ball-Bounded Norm only (No ArcLoss) | Norms balanced but latents may still cluster |
| ArcLoss only (No Norm Reg) | High utilization but codebook norms may explode |
| No stop-gradient on codebook | Codebook pulled by batches, loss of global dispersion |
| Changing m (0 / 0.1 / 0.3) | $m=0.1$ is optimal; too large hurts reconstruction |
| Changing top-k (1/3/5) | k=3 is optimal; k=1 is too strict, k=5 too loose |
| Changing $\alpha$ (expansion rate) | Large $\alpha$ voids early constraints; small $\alpha$ limits expression |

### Key Findings
- Codebook utilization improved from VQGAN's 44% to nearly 100%. Figure 1 t-SNE shows nearly all codebooks as active (green), representing fundamental improvement through geometric redesign.
- PCA visualization of quantized latent maps (Figure 5) shows ArcVQ-VAE has higher activation intensity and clearer contours, indicating codebooks encode finer spatial structures.
- Ball-Bounded regularization and ArcLoss are complementary: neither alone eliminates collapse; together they address "norm imbalance" and "geometric clustering."
- Stop-gradient is essential for stabilizing ArcLoss: directly optimizing codebooks via ArcLoss leads to local collapse.

## Highlights & Insights
- **Diagnosis of collapse as a geometric problem**: The authors empirically attribute collapse to a "norm imbalance + spatial clustering" cycle. This geometric perspective is refreshing and directly informs the method.
- **Cross-domain application of ArcFace**: While angular margins are mature in supervised facial recognition, this work treats "top-k nearest codebooks" as implicit classes to enable as unsupervised usage.
- **Near-zero overhead**: The method requires no new network layers and does not increase forward FLOPs, as it only involves norm clipping and an additional loss term. It can be plugged into any existing VQ-VAE.

## Limitations & Future Work
- Hyperparameters like $\alpha, m, s, k$ require manual sweeping; adaptive scheduling is a potential future direction.
- Evaluation is primarily on ImageNet reconstruction and LDM generation; effects on video, 3D, or multimodal tokenization are untested.
- Selecting the latent set $\mathcal{N}_j^{(k)}$ for top-k introduces an $O(K \cdot Bhw)$ sorting cost, which may become significant for very large codebooks (>16k).
- Comparison with recent FSQ (Finite Scalar Quantization) or LFQ (Lookup-Free Quantization) is missing—those methods bypass codebook learning entirely.

## Related Work & Insights
- **vs CVQ-VAE (Zheng & Vedaldi 2023)**: CVQ pulls inactive codebooks via online K-means; this work uses geometric regularization for lower rFID.
- **vs SQ-VAE (Takida et al. 2022)**: SQ uses stochastic posterior distributions; this work uses spherical margins.
- **vs VQGAN-LC (Zhu et al. 2024)**: That method uses a pretrained encoder to scale codebooks; this work relies on pure geometric constraints.
- **vs ArcFace (Deng et al. 2019)**: This is the first application of angular margins to unsupervised VQ.
- **vs FSQ / LFQ**: Those methods use scalar quantization to avoid learning; this work improves codebook learning directly.

## Rating
- Innovation/Novelty: ⭐⭐⭐⭐ Applying ArcFace to VQ-VAE is a clever transfer; the ball constraint + stop-gradient combo is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete across multiple datasets and baselines, though lacks direct comparison with FSQ/LFQ.
- Writing Quality: ⭐⭐⭐⭐ Diagnosis section (Fig 2/3) is very illuminating; derivations are clear.
- Value: ⭐⭐⭐⭐ High engineering value due to low cost and plug-and-play nature.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ICML 2026\] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](mind_your_margin_and_boundary_are_your_distilled_datasets_truly_robust.md)
- [\[ICLR 2026\] Embedding Compression via Spherical Coordinates](../../ICLR2026/model_compression/embedding_compression_via_spherical_coordinates.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)
- [\[ICML 2026\] Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space](event2vec_processing_neuromorphic_events_directly_by_representations_in_vector_s.md)

</div>

<!-- RELATED:END -->
