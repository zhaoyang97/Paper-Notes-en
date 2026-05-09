---
title: >-
  [Paper Note] Diffusion Mental Averages
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] Proposed Diffusion Mental Averages (DMA), which extracts "mental average" prototype images of concepts from pretrained diffusion models by aligning multiple denoising trajectories in semantic space—achieving consistent and realistic concept averaging visualization for the first time.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Models
  - Conceptual Prototypes
  - Trajectory Alignment
  - Semantic Averaging
  - Model Bias Analysis
date: 2026-05-08
content_hash: 78dffabff0464dd0
---

# Diffusion Mental Averages

**Conference**: CVPR 2026  
**arXiv**: [2603.29239](https://arxiv.org/abs/2603.29239)  
**Code**: [Project Page](https://diffusion-mental-averages.github.io)  
**Area**: Image Generation / Diffusion Models  
**Keywords**: Diffusion Models, Conceptual Prototypes, Trajectory Alignment, Semantic Averaging, Model Bias Analysis

## TL;DR

Proposed Diffusion Mental Averages (DMA), which extracts "mental average" prototype images of concepts from pretrained diffusion models by aligning multiple denoising trajectories in semantic space—achieving consistent and realistic concept averaging visualization for the first time.

## Background & Motivation

When humans imagine a "bird," a typical small bird (like a sparrow) comes to mind rather than an ostrich or a rare species. This "mental prototype" reflects our experiential preferences. Similarly, diffusion models should implicitly encode a "typical representation" for each concept after training—but how can it be visualized?

**Limitations of Prior Work**:

- **Pixel-space Averaging**: Requires spatial alignment; even after alignment, details are averaged out, producing blurry and unrealistic results. Spatial alignment is impossible for abstract concepts (e.g., "freedom").
- **Semantic-space Averaging**: Possible in AE/VAE/GAN via encoding $\rightarrow$ averaging $\rightarrow$ decoding, but diffusion models have **no explicit semantic bottleneck layer**. Semantic information is scattered across different timesteps with no direct decoder.
- **Dataset Distillation Methods** (D4M, MGD3): Optimized for downstream tasks; the resulting prototypes are unnatural or inconsistent.
- **Selective Methods**: Select representative samples from a set, but the total number of samples is finite and cannot transcend existing generations.

**Root Cause**: The semantic information of diffusion models is **distributed across the entire denoising trajectory**—from early coarse-grained layouts to late fine-grained details—making simple averaging at a single level impossible.

## Method

### Overall Architecture

DMA redefines "averaging" as "trajectory alignment": optimizing multiple noise latents simultaneously so their denoising trajectories gradually converge toward a shared "coarse-to-fine" semantic consensus. It primarily utilizes the U-Net bottleneck layer (h-space) as the semantic representation, which has been proven to possess linear interpretability.

### Key Designs

1.  **Progressive Trajectory Alignment**: Gradually unifies semantics during the denoising process $\rightarrow$ aligns global structures before local details $\rightarrow$ conforms to the coarse-to-fine generation logic of diffusion models.
    -   Initialize $K=1000$ noise latents $\{\mathbf{z}_k^{(0)}\}_{k=1}^K$.
    -   At each timestep $t$:
        -   Calculate the mean h-space activation of all latents: $\bar{A}^{(t)} = \frac{1}{K}\sum_{k} H(\mathbf{z}_k^{(t)})$
        -   Optimize each latent: $\min_{\mathbf{z}_i^{(t)}} \|H(\mathbf{z}_i^{(t)}) - \bar{A}^{(t)}\|_2^2$ (300 Adam iterations).
        -   Perform DDIM sampling to the next step after optimization.
    -   After reaching the cutoff $t_{stop}=10$, complete the remaining denoising using standard DDIM.
    -   The prototype image is obtained by decoding any final latent.
    -   **Design Motivation**: Leverages the semantic linearity of h-space to gradually establish consensus across the natural hierarchy of the diffusion process, achieving global consistency before refining local parts.

2.  **Mode Discovery for Multi-modal Concepts**: Handles cases like "dog" having multiple breeds $\rightarrow$ clusters in CLIP space and then averages separately $\rightarrow$ learns lightweight conditional adapters for each cluster.
    -   Generate samples and extract CLIP embeddings, then cluster using GMM.
    -   For each cluster, learn Textual Inversion embeddings or LoRA to guide the diffusion model's conditioning toward the corresponding semantic sub-region.
    -   **Design Motivation**: CLIP space is more stable than h-space and suitable for mode separation. However, since CLIP and h-space can exhibit semantic inconsistency, TI/LoRA acts as a bridge across spaces.

3.  **Grounded Clustering**: Supports user-defined segmentation dimensions.
    -   Uses BLIP-VQA to obtain attribute-focused embeddings (e.g., "doctor" by skin tone, "dog" by breed).
    -   Clusters with GMM after PCA dimensionality reduction.

### Loss & Training

-   Extremely concise core optimization objective: $\mathcal{L} = \|H(\mathbf{z}_k^{(t)}) - \bar{A}_t\|_2^2$
-   Uses Adam optimizer, learning rate $2 \times 10^{-2}$, 300 iterations per latent per timestep.
-   CFG scale = 7.0, 20 DDIM steps.
-   LoRA scheme: rank-1 LoRA, 2000 training steps, learning rate $10^{-4}$, CFG reduced to 3.0.
-   Textual Inversion: 3000 training steps, learning rate $10^{-2}$.
-   Full optimization takes approximately 10 hours (RTX 4080).

## Key Experimental Results

### Main Results

12 concepts (animals, people, objects, abstracts), 1000 samples per group, 10 repetitions:

| Method | Consistency (CLIP) ↓ | Consistency (DreamSim) ↓ | Representativeness (CLIP) ↓ | ImageReward ↑ |
| :--- | :--- | :--- | :--- | :--- |
| GANgealing | 0 | 0 | 0.386 | -0.684 |
| Avg VAE | 0 | 0 | 0.473 | -2.262 |
| D4M | 0.168 | 0.274 | 0.197 | 0.823 |
| MGD3 | 0.180 | 0.319 | 0.195 | 0.755 |
| **DMA (Ours)** | **0.031** | **0.032** | **0.179** | **1.002** |

-   Consistency for GANgealing and Avg VAE is 0 by design, but results are blurry and unrealistic.
-   DMA is superior across consistency, representativeness, and image quality.

### Ablation Study

| Configuration | Description |
| :--- | :--- |
| Different cutoff $t_{stop}$ | Larger values provide more thorough alignment but are computationally expensive; 10 steps are sufficient. |
| LoRA vs Textual Inversion | LoRA preserves color and shape better; TI has limited capacity. |
| Different SD variants | Variants produce prototypes with different styles/biases, validating the generalizability of the method. |
| DiT Architecture | Using the final transformer block instead of h-space is equally effective. |

### Key Findings

-   **Extremely High Consistency**: DMA converges to nearly identical prototypes from different random seeds, with consistency 5-10x better than D4M/MGD3.
-   **Reveals Model Bias**: E.g., "soldier" is always male in SD1.5/Realistic Vision, more neutral in PixelArt, and generates female cartoons in Animerge.
-   **Feasible for Abstract Concepts**: Consistently generates the Statue of Liberty for "freedom" and Venetian canals for "Italy," which baseline methods cannot handle.
-   **Effective Mode Discovery**: Can separate semantic modes like "bird" and "construction crane" for the word "crane."

## Highlights & Insights

1.  **New Research Problem**: First to propose extracting the "mental average" of a concept from diffusion models—focusing on finding the "most typical" representation rather than diverse samples.
2.  **Elegant Concept**: Transforms "averaging" into "trajectory alignment," perfectly fitting the coarse-to-fine generation paradigm of diffusion models.
3.  **Model Probing Tool**: DMA serves as a new tool for analyzing internal concept representations and biases in diffusion models, offering broad analytical value.
4.  **Generality**: Works from SD1.5 to DiT architectures, suggesting that the semantic hierarchical structure is a common property of diffusion models rather than specific to one architecture.
5.  **Combination of Mode Discovery + Conditional Adaptation**: An elegant solution for handling polysemous concepts.

## Limitations & Future Work

1.  **High Computational Cost**: 1000 latents $\times$ 20 timesteps $\times$ 300 optimizations $\approx$ 10 hours (RTX 4080), limiting practical application.
2.  **Dependence on h-space**: Requires manual identification of similar semantic layers for non-U-Net architectures (e.g., DiT); lacks an automated method.
3.  **Impact of CFG and Sample Size**: High-variance concepts require more samples; high CFG increases consistency but reduces diversity.
4.  **Clustering Depends on External Encoders**: Inherits the biases of CLIP/BLIP.
5.  **Subjectivity in Evaluation**: Representativeness metrics depend on the choice of specific embedding spaces.

## Related Work & Insights

-   **Kwon et al.**: Discovered that the U-Net bottleneck layer (h-space) has linear semantic properties, providing a key foundation for this work.
-   **GANgealing**: Performs pixel averaging via spatial alignment in GANs, but relies on pretrained GANs and cannot handle abstract concepts.
-   **D4M / MGD3**: Recent work using diffusion models for dataset distillation, but targeted at classification tasks rather than concept summarization.
-   **Textual Inversion / LoRA**: Innovative use of lightweight model adaptation methods for cross-space semantic alignment.
-   Insight: The timestep dimension in diffusion models contains a semantic hierarchy from abstract to concrete; this structure can be leveraged for more tasks.

## Rating

-   Novelty: ⭐⭐⭐⭐⭐ — Entirely new problem definition; the "mental average" concept is highly inspiring.
-   Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive experiments across concepts, variants, and architectures, though computational costs were not deeply analyzed.
-   Writing Quality: ⭐⭐⭐⭐⭐ — Smooth storyline, introduced via cognitive science analogies, rich visualizations.
-   Value: ⭐⭐⭐⭐ — Highly meaningful as an analytical tool and concept visualization method, despite limited practical application scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] GenIR: Generative Visual Feedback for Mental Image Retrieval](../../NeurIPS2025/image_generation/genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)
- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)

<!-- RELATED:END -->
