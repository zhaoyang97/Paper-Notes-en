---
title: >-
  [Paper Note] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis
description: >-
  [CVPR 2026][Image Generation][Diffusion Distillation] By decomposing weight changes during distillation into norm and direction components, this work finds that directional change is the primary driver of distillation (with a magnitude 22× larger than norm change). It proposes LoRaD (Low-Rank Weight Direction Rotation) adapters, integrated into the VSD framework to form WaDi, achieving state-of-the-art one-step FID on COCO with only ~10% trainable parameters.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Distillation
  - Weight Direction
  - Low-Rank Rotation
  - One-Step Generation
  - Parameter Efficiency
date: 2026-05-08
content_hash: 52a856e18beec2fb
---

# WaDi: Weight Direction-aware Distillation for One-step Image Synthesis

**Conference**: CVPR 2026
**arXiv**: [2603.08258](https://arxiv.org/abs/2603.08258)
**Code**: [https://github.com/gudaochangsheng/WaDi](https://github.com/gudaochangsheng/WaDi)
**Area**: Image Generation
**Keywords**: Diffusion Distillation, Weight Direction, Low-Rank Rotation, One-Step Generation, Parameter Efficiency

## TL;DR
By decomposing weight changes during distillation into norm and direction components, this work finds that directional change is the primary driver of distillation (with a magnitude 22× larger than norm change). It proposes LoRaD (Low-Rank Weight Direction Rotation) adapters, integrated into the VSD framework to form WaDi, achieving state-of-the-art one-step FID on COCO with only ~10% trainable parameters.

## Background & Motivation
**Background**: Diffusion distillation methods compress multi-step diffusion into one-step generators. Mainstream approaches are divided into full fine-tuning (FT) and LoRA-based fine-tuning, both built upon the VSD (Variational Score Distillation) framework.

**Limitations of Prior Work**: Both FT and LoRA directly update parameters, jointly optimizing weight norm and direction — yet these two quantities change at vastly different scales: the mean and standard deviation of directional change are 22× and 10× larger than those of norm change, respectively. This coupling increases optimization difficulty.

**Key Challenge**: Distillation signals are primarily conveyed through directional adjustments, yet existing adapters (LoRA/DoRA) are not specifically designed for direction optimization, leading to slow convergence, instability, and susceptibility to overfitting.

**Key Validation**: Replacing the one-step model's directions with teacher directions degrades FID by 241; replacing the norms changes FID by only 0.7. The direction residual matrix recovers 93% of the information with 30% of its rank — indicating a low-rank structure.

**Core Idea**: Since the essence of distillation is weight direction rotation, directly learning a low-rank rotation matrix to adjust directions is more principled than indirect influence via LoRA.

## Method

### Overall Architecture
WaDi is built upon the VSD framework: a frozen teacher $\epsilon_\psi$ (multi-step diffusion model) + a student generator $G_{\lambda}$ (one-step) + a fake model $\epsilon_\phi$ (tracking the student distribution). The key innovation is replacing LoRA/FT with LoRaD as the adapter for both the student and the fake model.

### Key Designs

1. **LoRaD (Low-Rank Weight Direction Rotation)**:

    - **Function**: Adjusts only the direction of pretrained weights via learnable rotation matrices, leaving their norms unchanged.
    - **Mechanism**: Inspired by RoPE, each column of weights is partitioned into $d/2$ odd-even paired subspaces, with an independent rotation applied in each 2D subspace:
    $W_{ro} = R_{AB}W = \begin{bmatrix} \cos AB & -\sin AB \\ \sin AB & \cos AB \end{bmatrix} \begin{bmatrix} W_{\text{odd}} \\ W_{\text{even}} \end{bmatrix}$
      The rotation angle matrix $\Theta = AB$, where $A \in \mathbb{R}^{d/2 \times r}$ and $B \in \mathbb{R}^{r \times k}$, enabling low-rank parameterization.
    - **Design Motivation**: Rotation matrices are orthogonal transformations that naturally preserve norms — perfectly aligned with the finding that direction is critical while norm is negligible. Low-rank decomposition exploits the low-rank structure of direction residuals, substantially reducing parameter count.
    - **Implementation Efficiency**: By leveraging the sparse block-diagonal structure of rotation matrices, computation requires only element-wise multiplication with no additional matrix multiplication overhead.

2. **WaDi Training Framework**:

    - **Function**: Integrates LoRaD into the VSD distillation framework.
    - **Mechanism**: The student $G_{\lambda_{\Theta^l}}$ uses high-rank LoRaD (rank=256); the fake model $\epsilon_{\phi_{\Theta^s}}$ uses low-rank LoRaD (rank=32). Both are optimized alternately.
    - Student loss: $\nabla_{\lambda_{\Theta^l}} \mathcal{L}_{\text{wadi}} = \mathbb{E}[\omega(t)(\epsilon_\psi - \epsilon_{\phi_{\Theta^s}}) \frac{\partial G_{\lambda_{\Theta^l}}}{\partial \lambda_{\Theta^l}}]$
    - **Design Motivation**: The student requires higher capacity (rank=256) to fully fit the teacher distribution; the fake model only needs to track the student's evolution (rank=32).

### Loss & Training
- Image-free training: no real images required; only 1.4M JourneyDB text prompts are used.
- Student LR=1e-4, fake model LR=1e-2, AdamW optimizer, batch=128, CFG=1.5.
- Trained for 2 epochs; supports SD1.5, SD2.1, and PixArt-α backbones.

## Key Experimental Results

### Main Results — COCO 2014 Zero-Shot FID

| Method | Backbone | NFE | Trainable Params | FID↓ | CLIP↑ |
|--------|----------|-----|-----------------|------|-------|
| SD 1.5 | U-Net | 25 | 860M | 8.78 | 0.30 |
| DMD2 | U-Net | 1 | 860M | 12.96 | 0.30 |
| SiD-LSG | U-Net | 1 | 860M | 14.27 | 0.30 |
| **WaDi** | **U-Net** | **1** | **83.8M (9.7%)** | **10.79** | **0.31** |
| PixArt-α | DiT | 20 | 610M | 8.75 | 0.32 |
| **WaDi** | **DiT** | **1** | **81.2M (13.3%)** | **18.99** | **0.30** |

### Ablation Study — Adapter Type Comparison

| Adapter | Params | FID↓ | Direction Mean Change |
|---------|--------|------|-----------------------|
| LoRA | 120.9M | 25.27 | 0.83% |
| DoRA | 121.2M | 26.56 | 0.55% |
| DoRA (frozen norm) | 120.9M | 24.52 | 0.92% |
| FT (DMD2) | 860.0M | 23.30 | 2.21% |
| **LoRaD** | **83.8M** | **20.86** | **2.89%** |

### Ablation Study — Effect of Rank Configuration (COCO 2014)

| Setting | Student Rank | Student Params | Fake Model Rank | FID↓ | CLIP↑ |
|---------|-------------|----------------|----------------|------|-------|
| A | 64 | 20.95M | 32 | 13.64 | 0.30 |
| B | 128 | 41.90M | 32 | 13.16 | 0.29 |
| **C** | **256** | **83.80M** | **32** | **10.79** | **0.31** |
| D | 512 | 167.59M | 32 | 12.75 | 0.30 |

### Key Findings
- LoRaD achieves the largest directional change (2.89%) and best FID (20.86) with the fewest parameters (83.8M vs. 860M), perfectly validating the hypothesis that direction is the key variable in distillation.
- Rank=256 is the optimal student configuration; rank=512 leads to overfitting (FID degrades from 10.79 to 12.75).
- Fake model rank primarily affects fidelity (FID) with minimal impact on semantic alignment (CLIP).
- WaDi can be directly applied to downstream tasks including ControlNet (86% inference speedup), ReVersion (89% speedup), and DreamBooth.
- In a user study with 57 participants, WaDi was consistently rated superior to existing baselines in both image quality and text-image alignment.

## Highlights & Insights
- **Weight Norm–Direction Decomposition Analysis**: This work is the first to systematically study the structure of weight changes during distillation — directional change far exceeds norm change, and direction residuals exhibit a low-rank structure. This offers a novel theoretical perspective on distillation.
- **Rotation Instead of Addition**: LoRA updates weights via addition $W + \Delta W$ (jointly modifying both norm and direction), whereas LoRaD applies rotation $R_{\Theta}W$ to modify direction exclusively — more targeted and more efficient.
- **Parameter Efficiency**: Approximately 10% of trainable parameters suffice to surpass full fine-tuning, offering significant value in resource-constrained settings.

## Limitations & Future Work
- The 2D subspace pairing in LoRaD follows a fixed odd-even scheme, which may not be the optimal grouping strategy.
- Although FID is better than DMD2, the difference in CLIP scores is marginal, suggesting that directional rotation primarily improves image fidelity rather than semantic alignment.
- The FID gap on PixArt-α (DiT architecture) remains relatively large (18.99), potentially requiring architecture-specific designs for DiT.
- Ablation studies are conducted only on COCO 2017; validation on additional datasets is lacking.

## Related Work & Insights
- **vs. DMD2**: DMD2 employs full fine-tuning for distillation; WaDi achieves better FID with only 10% of the parameters by precisely targeting the key variable in distillation (direction).
- **vs. LoRA/DoRA**: LoRA's additive updates modify both norm and direction but yield insufficient directional change (0.83%); DoRA decouples the norm but still updates direction via LoRA; LoRaD directly rotates directions, achieving the largest directional change (2.89%).
- **vs. SwiftBrush**: SwiftBrush also builds on VSD but uses full fine-tuning; WaDi combines VSD with LoRaD for far superior parameter efficiency.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The weight norm–direction analysis perspective is original, and the LoRaD design is elegant with well-grounded theoretical motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage across three backbones, downstream tasks, and a user study is comprehensive; ablation is detailed, though broader dataset coverage would strengthen the work.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The motivation analysis is highly convincing (replacement experiments + SVD analysis), with rigorous logical argumentation throughout.
- **Value**: ⭐⭐⭐⭐⭐ Sets a new standard for parameter-efficient distillation; LoRaD is transferable to other fine-tuning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] ChordEdit: One-Step Low-Energy Transport for Image Editing](chordedit_one-step_low-energy_transport_for_image_editing.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](../../NeurIPS2025/image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](../../ICCV2025/image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)

</div>

<!-- RELATED:END -->
