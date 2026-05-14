---
title: >-
  [Paper Note] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing
description: >-
  [ICLR 2026][Image Generation][Drag Editing] The first framework to incorporate the strong generative priors of FLUX (DiT) into drag editing. By replacing conventional point-level supervision with region-level affine supe…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Drag Editing"
  - "DiT"
  - "Region-Based Supervision"
  - "FLUX"
  - "Affine Transformation"
date: 2026-05-08
content_hash: 26b572ee98e0c1c8
---

# DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing

**Conference**: ICLR 2026
**arXiv**: [2510.02253](https://arxiv.org/abs/2510.02253)
**Code**: [GitHub](https://github.com/Edennnnnnnnnn/DragFlow)
**Area**: Diffusion Models / Image Editing
**Keywords**: Drag Editing, DiT, Region-Based Supervision, FLUX, Affine Transformation

## TL;DR
The first framework to incorporate the strong generative priors of FLUX (DiT) into drag editing. By replacing conventional point-level supervision with region-level affine supervision, combined with gradient-mask hard constraints and adapter-enhanced inversion, the method substantially improves drag editing quality.

## Background & Motivation
**Background**: Drag editing allows users to specify spatially local motions via interactive drag instructions, enabling fine-grained controllable image editing. However, existing methods are predominantly based on Stable Diffusion (UNet-based DDPM), and editing results frequently exhibit unnatural deformations and distortions.

**Limitations of Prior Work**: The generative prior of SD is insufficiently strong to pull the optimized latent back onto the natural image manifold. Although DiT + Flow Matching models (e.g., SD3.5, FLUX) possess significantly stronger priors, drag editing has yet to benefit from them.

**Key Challenge**: Directly porting point-level drag frameworks to DiT yields poor results. The UNet bottleneck produces spatially compact, highly compressed features with wide receptive fields, whereas DiT features are finer-grained and spatially precise with narrow receptive fields, making single-point supervision semantically insufficient. Furthermore, FLUX is a CFG-distilled model prone to severe inversion drift, and conventional KV injection is inadequate for maintaining subject consistency.

**Goal**: How to effectively leverage DiT's strong priors for high-quality drag editing?

**Key Insight**: Shift from point-level to region-level supervision, leveraging affine transformations to provide richer and more consistent feature supervision; simultaneously redesign background preservation and inversion enhancement strategies.

**Core Idea**: Replace point-level motion supervision with region-level affine transformation supervision to unleash DiT's potential in drag editing.

## Method

### Overall Architecture
Given an input image, it is first VAE-encoded and inverted to a noisy latent $\bm{z}_t$. Region-level affine supervision then guides iterative latent updates during optimization. Subject consistency is ensured by KV injection combined with adapter-enhanced inversion, while background fidelity is enforced via gradient-mask hard constraints. An MLLM (GPT-5) is employed to infer user intent (editing type and text description).

### Key Designs
1. **Region-Level Affine Supervision**: The user specifies source region masks $\{\bm{M}_i\}$ and target points $\bm{t}_i$. Affine transformations progressively propagate the source masks toward the target positions. The core loss function is:
$$\mathcal{L}_{\text{Drag}} = \sum_{i=1}^{N} \gamma_i \cdot \| \bm{M}_i^{(k)} \odot F(\bm{z}_t^{(k)}) - \text{sg}[\bm{M}_i^{(0)} \odot F(\bm{z}_t^{(0)})] \|_1$$
where $F(\cdot)$ extracts DiT features and $\bm{M}_i^{(k)}$ is obtained via affine transformation. For repositioning/deformation, parameters are determined by the vector from the target point to the centroid; for rotation, by the angle and anchor point. A linear schedule $k/K$ smoothly shifts the mask from source to target.
    - **Why Region-Level is Superior**: Region-level matching provides richer semantic context and alleviates shortsighted gradients. Moreover, point tracking is unnecessary since region features rather than individual points are compared, eliminating tracking error accumulation.

2. **Background Hard Constraint**: Conventional methods employ an auxiliary consistency loss $\mathcal{L}_{\text{BG}}$ for background preservation, but this loss competes with the editing objective, and large inversion drift in CFG-distilled models renders the reference unreliable. This work instead applies a hard constraint:
$$\bm{z}_t^{(k+1)} = \bm{B} \odot (\bm{z}_t^{(k)} - \alpha \cdot \frac{\partial \mathcal{L}_{\text{Drag}}}{\partial \bm{z}_t^{(k)}}) + (1 - \bm{B}) \odot \bm{z}_t^{\text{orig}}$$
An additional pure reconstruction branch provides $\bm{z}_t^{\text{orig}}$, incurring moderate overhead but yielding significant improvements.

3. **Adapter-Enhanced Inversion**: Pretrained open-domain personalization adapters (e.g., IP-Adapter, InstantCharacter) serve as auxiliary subject representation extractors, injecting subject representations into the model prior. This substantially improves inversion quality and subject consistency without additional fine-tuning. Experiments show that FLUX inversion LPIPS decreases from 0.283 to 0.173.

### Loss & Training
- Only $\mathcal{L}_{\text{Drag}}$ is used; background consistency loss is replaced by the hard constraint.
- FireFlow inversion algorithm with 25 diffusion steps; the first 6 steps are skipped, and editing begins at step 19.
- 70 optimization iterations at denoising step 7; learning rate is 1000 for the first 50 iterations and 1200 for the last 20.
- Adaptive weights $\gamma_i$ are determined by the relative size of the operation region.

## Key Experimental Results

### Main Results

| Method | Category | IF_bg↑ | IF_s2t↑ | IF_s2s↓ | MD1↓ | MD2↓ |
|--------|----------|--------|---------|---------|------|------|
| RegionDrag | NFT | 1.000 | 0.957 | 0.957 | 33.69 | 6.38 |
| GoodDrag | OPT | 0.935 | 0.956 | 0.942 | 20.38 | 4.50 |
| InstantDrag | FT | 0.930 | 0.949 | 0.946 | 24.38 | 4.54 |
| **DragFlow (Ours)** | OPT | **0.992** | **0.958** | **0.934** | **19.46** | **4.48** |

*Results on ReD Bench. DragFlow achieves the best performance on all Mean Distance metrics. IF_bg is second only to RegionDrag, which directly copies and pastes without altering the background.*

A similar trend is observed on DragBench-DR: DragFlow MD1=31.59 (second best 35.96), IF_bg=0.969 (second best 0.962). Consistent superiority across both benchmarks demonstrates strong robustness.

### Adapter Inversion Quality Comparison (3,000 images)

| Method | LPIPS↓ | SSIM↑ | PSNR↑ |
|--------|--------|-------|-------|
| DPM-Solver Inv. (SD) | 0.167 | 0.799 | 26.31 |
| FireFlow Inv. w/o adapter (FLUX) | 0.283 | 0.703 | 20.43 |
| FireFlow Inv. w/ adapter (FLUX) | 0.173 | 0.784 | 25.87 |

### Ablation Study

| Configuration | IF_bg↑ | IF_s2t↑ | IF_s2s↓ | MD1↓ | MD2↓ |
|---------------|--------|---------|---------|------|------|
| Baseline (Point-based FLUX) | 0.765 | 0.932 | 0.962 | 51.21 | 9.38 |
| + Region-Level Affine | 0.757 | 0.946 | 0.936 | 31.26 | 5.88 |
| + Background Preservation | 0.925 | 0.948 | 0.943 | 29.67 | 5.39 |
| + Adapter-Enhanced Inversion | 0.991 | 0.959 | 0.938 | 20.15 | 4.48 |

### Key Findings
- Transitioning from point-level to region-level supervision reduces MD1 by 19.95 and improves IF_s2t by 0.027, validating that the region-based paradigm is more suitable for DiT editing.
- The gradient-mask design dramatically raises IF_bg from 0.757 to 0.925.
- Adapter-enhanced inversion improves IF_s2t from 0.948 to 0.959, confirming enhanced foreground consistency.
- All three modules are individually effective and mutually complementary.

## Highlights & Insights
- **First systematic analysis of why point-level drag fails on DiT**: Feature map visualizations clearly illustrate the difference in feature granularity between UNet and DiT.
- **The region-level paradigm elegantly eliminates the point-tracking problem**: No tracking is required, and no errors accumulate.
- **Hard constraint replaces soft loss**: Avoids the trade-off between background preservation and the editing objective.
- **MLLM-based intent inference**: Automatically determines the drag type (repositioning/deformation/rotation), reducing user burden.
- **Proposes ReD Bench**: The first drag editing benchmark with region-level annotations and task labels.

## Limitations & Future Work
- As a CFG-distilled model, FLUX still suffers from non-trivial inversion drift, leading to detail loss on highly complex images.
- Reliance on an external MLLM (GPT-5) for intent inference increases deployment cost.
- An additional reconstruction branch is required to compute $\bm{z}_t^{\text{orig}}$, increasing inference overhead.
- Non-affine deformation scenarios (e.g., non-rigid free-form deformation) remain unexplored.

## Related Work & Insights
- **RegionDrag**: Also uses region-level inputs, but requires users to manually predefine target region masks and employs hand-crafted mapping functions, whereas DragFlow requires only target points. RegionDrag transfers noisy latent patches via point-wise copy-paste, which tends to disrupt intra-region structure; DragFlow extracts region features holistically for supervision.
- **GoodDrag**: The strongest point-level optimization baseline on ReD Bench (MD1=20.38), yet still constrained by SD priors.
- **DragDiffusion**: A pioneering point-level drag work using LoRA fine-tuning on SD; direct transfer to DiT performs poorly.
- **InstantDrag**: A fine-tuning-based method limited by the mismatch between video data and drag instructions, with 914M parameters.
- **CLIPDrag**: Introduces CLIP semantic guidance into point-level methods, but frequently misinterprets repositioning as deformation, producing artifacts.
- **FastDrag**: An optimization- and fine-tuning-free method that directly maps latent patches; fast but reliant on hand-crafted priors, resulting in unnatural edits.
- **Insight**: When the geometric properties of base model features fundamentally change (UNet→DiT), the editing paradigm must be redesigned accordingly. As the DiT ecosystem matures, the value of the DragFlow framework will only grow.

## Rating
- Novelty: ⭐⭐⭐⭐ — The analysis of point-level supervision failure from the perspective of feature granularity is insightful, and the region-level paradigm is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Dual-benchmark evaluation, comprehensive ablation, and a new benchmark.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated with abundant illustrations.
- Value: ⭐⭐⭐⭐ — Points the way for drag editing in the DiT era.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control](follow-your-shape_shape-aware_image_editing_via_trajectory-guided_region_control.md)
- [\[ICCV 2025\] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](../../ICCV2025/image_generation/superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)
- [\[ICCV 2025\] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping](../../ICCV2025/image_generation/inpaint4drag_repurposing_inpainting_models_for_drag-based_image_editing_via_bidi.md)
- [\[ICLR 2026\] Amortising Inference and Meta-Learning Priors in Neural Networks (BNNP)](amortising_inference_and_meta-learning_priors_in_neural_networks.md)
- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)

</div>

<!-- RELATED:END -->
