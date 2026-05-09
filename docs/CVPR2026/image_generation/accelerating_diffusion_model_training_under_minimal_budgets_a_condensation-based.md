---
title: >-
  [Paper Note] Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective
description: >-
  [CVPR 2026][Image Generation][Dataset Condensation] This paper proposes D2C (Diffusion Dataset Condensation)—the first dataset condensation framework for diffusion models—which achieves 100–233× training speedup while maintaining high-quality image generation by using only 0.8–8% of ImageNet data through a two-stage "Select + Attach" pipeline.
tags:
  - CVPR 2026
  - Image Generation
  - Dataset Condensation
  - Diffusion Model
  - Training Acceleration
  - Data-Centric
  - Efficient Training
date: 2025-07-08
content_hash: 137f2b6dc3a944cd
---

# Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective

**Conference**: CVPR 2026  
**arXiv**: [2507.05914](https://arxiv.org/abs/2507.05914)  
**Code**: TBD  
**Area**: Image Generation  
**Keywords**: Dataset Condensation, Diffusion Model, Training Acceleration, Data-Centric, Efficient Training

## TL;DR

This paper proposes D2C (Diffusion Dataset Condensation)—the first dataset condensation framework for diffusion models—which achieves 100–233× training speedup while maintaining high-quality image generation by using only 0.8–8% of ImageNet data through a two-stage "Select + Attach" pipeline.

## Background & Motivation

**State of the Field**: Current diffusion models (DiT, SiT, etc.) typically require millions of images and millions of training iterations; for instance, SiT-XL/2 on ImageNet needs 7M steps, and REPA still requires 4M steps, consuming hundreds of GPU·hours.

**Limitations of Prior Work**: Existing dataset distillation/condensation methods (e.g., SRe2L, RDED, Herding, K-Center) are almost exclusively designed for discriminative tasks such as classification, and perform extremely poorly when directly transferred to diffusion model training (RDED yields FID of 166.2 on DiT-L/2).

**Root Cause**: Pixel-level distillation methods synthesize images biased toward category-discriminative features while lacking preservation of distributional diversity and semantic structure, leading to generation quality collapse and unstable convergence. Discriminative features ≠ generative features.

**Paper Goals**: Systematically construct a compact, information-rich data subset tailored for diffusion model training from the data perspective, a direction that has been largely unexplored.

**Starting Point**: Simple pruning strategies (random sampling, geometric methods like K-Center/Herding) cannot perform difficulty-aware selection suited to diffusion denoising characteristics.

**Core Idea**: A two-stage pipeline—**Select** (difficulty-aware interval sampling) + **Attach** (dual semantic and visual information injection)—to build a compact yet maximally informative training subset for diffusion models.

## Method

### Overall Architecture

D2C adopts a two-stage pipeline: the **Select** stage filters a compact, diverse, and learnable subset from the full training set; the **Attach** stage enriches each selected image with semantic and visual representations. The diffusion model is then trained from scratch on this augmented condensed dataset with a joint denoising and representation alignment objective.

### Key Designs

1. **Diffusion Difficulty Score + Interval Sampling (Select Stage)**:

    - Function: Computes a "diffusion difficulty score" for each training image, then performs uniform interval sampling over the sorted list
    - Mechanism: A pre-trained diffusion model's class-conditional posterior probability $p_\theta(\mathbf{c}|\mathbf{x})$ is used to rank sample difficulty. Via Bayes' rule, the difficulty score reduces to the negative conditional likelihood (i.e., the denoising loss): $s_{\text{diff}}(\mathbf{x}) = -\mathbb{E}_{\epsilon,t}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})\|_2^2]$. Higher scores indicate more difficult samples. Within each class, samples are sorted by difficulty in ascending order and sampled at fixed interval $k$, balancing learnability of easy samples and diversity of hard samples
    - Design Motivation: Selecting only the easiest samples (Min) yields fast convergence but insufficient diversity; selecting only the hardest (Max) introduces too much noise. Interval sampling achieves uniform coverage across the difficulty distribution—$k=96$ at 0.8% budget and $k=16$ at 4% budget yield optimal results

2. **Dual Conditional Embedding (DC-Embedding, Attach Stage—Semantic Information)**:

    - Function: Fuses pre-trained text encoder (T5-encoder) class description embeddings with learnable class embeddings as the diffusion model's conditional input
    - Mechanism: For each class, a descriptive prompt (e.g., "a photo of a cat") is generated and encoded by the text encoder to produce text embedding $t_c$ and text mask $t_{\text{mask}}$, which are then fused with learnable class embedding $e_c$ via 1D convolution + residual MLP: $y_{\text{text}} = \text{MLP}(\tilde{t}_c) + \tilde{t}_c + e_c$. Text embeddings are pre-computed and stored on disk
    - Design Motivation: Learnable class embeddings trained from scratch lack semantic information under data-limited settings; incorporating rich pre-trained text encoder semantics (especially inter-class discriminability) significantly improves conditional generation quality while retaining the flexibility of learnable embeddings

3. **Visual Information Injection (Attach Stage—Visual Information)**:

    - Function: Extracts instance-level visual representations for each selected image using a pre-trained visual encoder (DINOv2), stored on disk and injected via a representation alignment loss during training
    - Mechanism: DINOv2 extracts patch-level semantic features $y_{\text{vis}} \in \mathbb{R}^{N \times d}$ for each image, truncated to the first $h$ tokens as a compact representation. During training, token features $\{h_i\}$ from an intermediate diffusion model layer are projected and aligned with visual representations via cosine alignment loss: $\mathcal{L}_{\text{proj}} = -\frac{1}{h}\sum_i \langle \frac{\phi(h_i)}{\|\phi(h_i)\|}, \frac{v_i}{\|v_i\|} \rangle$
    - Design Motivation: Semantic embeddings primarily provide inter-class structural discrimination, but intra-class diversity (texture, pose, etc.) requires instance-level visual information. Inspired by REPA's representation alignment strategy, this injects spatial consistency priors into the diffusion model, which is especially critical under extremely small datasets

## Loss & Training

The total training objective consists of two components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \cdot \mathcal{L}_{\text{proj}}$$

- **Denoising loss** $\mathcal{L}_{\text{diff}}$: Standard diffusion model noise prediction MSE, conditioned on class label $y$ and text information $y_{\text{text}}$
- **Representation alignment loss** $\mathcal{L}_{\text{proj}}$: Cosine similarity alignment between intermediate diffusion model tokens and DINOv2 visual representations
- Balancing weight $\lambda = 0.5$

## Key Experimental Results

### Main Results (ImageNet 256×256, different data budgets, gFID-50K with CFG=1.5)

| Data Budget | Steps | DiT-L/2 Random | DiT-L/2 D2C | SiT-L/2 Random | SiT-L/2 D2C |
|------------|-------|----------------|-------------|----------------|-------------|
| 0.8% (10K) | 100k | 35.86 | **4.20** | 4.35 | **3.98** |
| 0.8% (10K) | 300k | 4.19 | **4.13** | 4.33 | **3.98** |
| 4.0% (50K) | 100k | 36.78 | **14.81** | 31.13 | **11.21** |
| 4.0% (50K) | 300k | 11.55 | **5.99** | 14.18 | **5.66** |
| 8.0% (100K) | 100k | 41.02 | **22.55** | 36.64 | **15.01** |
| 8.0% (100K) | 300k | 11.49 | **6.49** | 12.56 | **5.65** |

D2C substantially outperforms Random, K-Center, Herding, and other baselines across all budget and architecture settings. Notably, at 0.8% budget, D2C at 100k steps matches Random at 300k steps.

### Comparison with SRe2L / RDED (0.8% data, DiT-L/2)

| Method | gFID↓ | sFID↓ | IS↑ | Precision↑ |
|--------|-------|-------|-----|-----------|
| RDED | 166.2 | 60.1 | 10.8 | 0.09 |
| SRe2L | 104.2 | 20.2 | 14.1 | 0.20 |
| **D2C** | **4.2** | **11.0** | **283.6** | **0.72** |

Condensation methods designed for discriminative tasks completely fail in diffusion training, with FID 1–2 orders of magnitude worse than D2C.

### Acceleration Results

Using SiT-XL/2, D2C achieves FID 4.3 with only 0.8% data (10K images) at 40k steps, representing **100×** speedup over REPA (4M steps) and **233×** over vanilla SiT (7M steps). At 4% data (50K) + CFG=1.5, it achieves FID 2.78 at 180k steps.

### Ablation Study

- **Select stage alone**: Reduces gFID from 37.07 to 14.96
- **Attach stage**: DC-Embedding alone reduces to 9.01; visual representation alone to 10.37; both combined achieve **7.62**
- **Interval $k$ selection**: Optimal $k$ is roughly inversely proportional to data budget (10K→$k$=96, 50K→$k$=16)
- **Wall-clock**: Attach-only mode requires only 7.4h (0.99% of REPA); the full pipeline takes 9.5h (1.27% of REPA)

## Highlights & Insights

1. **First dataset condensation framework for diffusion models**: Fills the gap in generative task dataset condensation and reveals the critical finding that discriminative condensation methods cannot be directly transferred
2. **Strong performance under extreme compression**: Achieving FID 3.98 with SiT-L/2 using only 0.8% data demonstrates massive data redundancy in diffusion model training
3. **Clean modular design**: Select and Attach stages can be used independently; even Attach-only surpasses REPA
4. **Cross-architecture and cross-resolution generalization**: Comprehensively validated across DiT/SiT × L/XL × 256/512
5. **Significant practical speedup**: End-to-end wall-clock time is ~1% of REPA, demonstrating real-world deployment viability

## Limitations & Future Work

1. **Dependence on pre-trained models**: Requires a pre-trained diffusion model for difficulty scoring + T5 encoder + DINOv2 encoder; the method's independence is limited and cold-start costs are hidden
2. **Only C2I verified**: All main experiments use class-conditional ImageNet generation; T2I (text-to-image) is only briefly mentioned in the appendix, leaving large-scale T2I effectiveness unverified
3. **Interval hyperparameter tuning**: The optimal interval $k$ depends on the data budget in a non-trivial way, requiring additional experiments
4. **Resolution ceiling**: Experiments cap at 512×512; mainstream 1024+ resolution generation remains unexplored
5. **Diminishing returns at scale**: Performance gaps narrow from 0.8% to 8%; whether marginal benefits diminish at larger data scales is unclear

## Related Work & Insights

- **REPA** (Yu et al.): Accelerates training by aligning diffusion model intermediate representations with a pre-trained visual encoder; D2C's visual injection module draws inspiration but further combines it with data selection
- **SRe2L / RDED**: Representative pixel-level/image-level dataset distillation methods for classification; this paper experimentally demonstrates their unsuitability for diffusion training
- **InfoBatch / Patch-based methods**: An alternative data-side efficient training approach via re-sampling/patching, but without constructing condensed subsets
- **Li et al. (2025)**: Studies diffusion training data pruning from a coreset selection perspective, but without attaching additional information and only validated at smaller scales
- **DiT / SiT**: Primary experimental backbone architectures; D2C serves as an orthogonal data-side strategy that can be freely combined

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic study of dataset condensation for diffusion models; the Select+Attach framework is well-designed
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-architecture, multi-resolution, multi-budget comprehensive comparisons + detailed ablations + transparent wall-clock analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, rigorous formulations, information-dense figures and tables, easy to follow
- Value: ⭐⭐⭐⭐ — 100×+ practical speedup with significant engineering impact; opens a new direction for data-model co-optimization

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](d2c_diffusion_dataset_condensation.md)
- [\[CVPR 2026\] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](one_model_many_budgets_elastic_latent_interfaces_f.md)
- [\[CVPR 2026\] Pose-dIVE: Pose-Diversified Augmentation with Diffusion Model for Person Re-Identification](pose-dive_pose-diversified_augmentation_with_diffusion_model_for_person_re-ident.md)
- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)

<!-- RELATED:END -->
