---
title: >-
  [Paper Note] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation
description: >-
  [CVPR2025][Segmentation][Referring Image Segmentation] Proposed the SERA framework, which introduces lightweight expression-aware Mixture-of-Experts (MoE) refinement into pre-trained vision-language models. It performs expert routing at both the backbone level (SERA-Adapter) and the fusion level (SERA-Fusion), achieving state-of-the-art (SOTA) performance on referring image segmentation benchmarks while updating less than 1% of the parameters.
tags:
  - "CVPR2025"
  - "Segmentation"
  - "Referring Image Segmentation"
  - "Mixture of Experts"
  - "Expert Routing"
  - "Parameter-Efficient Fine-Tuning"
  - "Vision-Language Models"
  - "DINOv2"
  - "CLIP"
date: 2026-05-08
content_hash: 48f27db9762c9123
---

# Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation

**Conference**: CVPR2025  
**arXiv**: [2603.12538](https://arxiv.org/abs/2603.12538)  
**Code**: To be confirmed  
**Area**: Image Segmentation  
**Keywords**: Referring Image Segmentation, Mixture of Experts, Expert Routing, Parameter-Efficient Fine-Tuning, Vision-Language Models, DINOv2, CLIP

## TL;DR
Proposed the SERA framework, which introduces lightweight expression-aware Mixture-of-Experts (MoE) refinement into pre-trained vision-language models. It performs expert routing at both the backbone level (SERA-Adapter) and the fusion level (SERA-Fusion), achieving state-of-the-art (SOTA) performance on referring image segmentation benchmarks while updating less than 1% of the parameters.

## Background & Motivation
- Referring image segmentation (RIS) requires generating pixel-level masks based on natural language expressions, necessitating both language understanding and visual target localization.
- Pre-trained vision-language models (e.g., CLIP + DINOv2) offer powerful semantic alignment capabilities, but default frozen backbones make it challenging for visual representations to adaptively adjust to diverse referring expressions.
- Different referring expressions require diverse types of reasoning: some rely on spatial relations, some on visual appearance, and others on contextual clues.
- Most existing methods adopt a uniform refinement strategy that applies the same processing path to all samples, failing to accommodate diverse reasoning demands.
- Typical failure modes include fragmented regions, boundary leakage, and incorrect target selection under ambiguous expressions.

## Core Problem
How to adaptively select appropriate visual feature refinement strategies based on the diverse reasoning requirements of referring expressions under a parameter-efficient setting with frozen pre-trained backbones?

## Method
SERA consists of two complementary modules, both utilizing the MoE mechanism:

### 1. SERA-Adapter (Backbone-Level Refinement)
- Inserted into the feed-forward residual paths of selected transformer blocks in DINOv2.
- **Pipeline**: Linear projection $\rightarrow$ spatial grid reshaping $\rightarrow$ multi-scale convolutional enrichment (parallel branches of 1x1, 3x3, and 5x5) $\rightarrow$ expert refinement $\rightarrow$ cross-modal attention $\rightarrow$ residual update.
- **Two Experts**:
    - **Boundary Expert**: Depthwise separable 3x3 convolution + scaled residual ($\beta=0.1$) to enhance boundary-sensitive responses.
    - **Spatial Expert**: Depthwise separable 3x3 convolution + BN + ReLU + scaled residual ($\alpha=0.3$) to enhance local feature consistency.
- **Soft Routing**: Performs global average pooling on spatial tokens $\rightarrow$ linear projection $\rightarrow$ softmax to obtain weights $w_s$ and $w_b$ for the two experts.
- Injected back into the backbone as residuals after being aligned with text embeddings via cross-modal attention.

### 2. SERA-Fusion (Fusion-Level Refinement)
- Refines intermediate spatial feature maps during the vision-language fusion phase.
- **Four Experts**:
    - **Spatial Expert**: Injects a normalized coordinate grid (via 1x1 convolutional projection) to provide explicit position information.
    - **Contextual Expert**: Self-attention-based context aggregation to capture long-range spatial dependencies.
    - **Boundary Expert**: Uses a fixed Sobel operator to extract horizontal/vertical gradients + 1x1 convolution to fuse gradient magnitude.
    - **Shape Expert**: Combines depthwise blurring and Laplacian filtering to integrate low-frequency smoothing and high-frequency structural cues.
- **Sparse Top-K Routing**: Global average pooling $\rightarrow$ two-layer MLP to predict routing logits $\rightarrow$ addition of Gaussian noise during training $\rightarrow$ Top-k selection + softmax normalization.

### 3. Strategies to Prevent Expert Collapse
- SERA-Adapter utilizes soft routing (stabilizing backbone adaptation), while SERA-Fusion employs sparse Top-k routing (encouraging specialization).
- Auxiliary regularization: Logit square penalty + load balancing (squared coefficient of variation) + token assignment regularization.
- Z-loss penalizes the mean-square magnitude of routing logits.

### 4. Parameter-Efficient Training
- DINOv2 and CLIP encoders are fully frozen.
- Only LayerNorm and bias parameters are updated, affecting less than 1% of the backbone parameters.

## Key Experimental Results
On three standard benchmarks: RefCOCO, RefCOCO+, and G-Ref (mIoU):

| Dataset | val | testA | testB |
|--------|-----|-------|-------|
| RefCOCO | 76.5 | 78.2 | 73.7 |
| RefCOCO+ | 70.4 | 74.4 | 62.8 |
| G-Ref(u) | 68.8/68.9 | | |
| G-Ref(g) | 66.6 | | |
| **Average** | **71.1** | | |

- Achieves an average mIoU of 71.1, outperforming methods like DETRIS-B (70.4) and RISCLIP-B (70.6).
- Reaches 70.4 on RefCOCO+ val, showing more pronounced advantages in harder settings where absolute spatial words are excluded.
- Notably, this is achieved under an extremely parameter-efficient setting with a frozen backbone where only bias/LN are updated.

### Ablation Study
- Adding only SERA-Adapter: RefCOCO val +0.45, RefCOCO+ val +0.72, G-Ref(g) +0.64
- Adding both SERA-Adapter & SERA-Fusion (full SERA): RefCOCO val +1.60, RefCOCO+ val +1.70, G-Ref(g) +1.52
- In Top-K routing, K=1 achieves the worst performance, while K=4 is the overall optimal; returns diminish for K>2.
- Training environment: Single NVIDIA A6000 GPU, batch size 16, Adam oscillator, initial learning rate of 1e-4.

## Highlights & Insights
1. **First to Introduce MoE into Referring Image Segmentation**: Complementary expert routing mechanisms are designed for both the backbone and fusion stages.
2. **Extreme Parameter Efficiency**: Updates <1% of the backbone parameters while achieving or even surpassing full fine-tuning methods.
3. **Meticulous Routing Stabilization Strategies**: Distinct designs of soft routing vs. sparse routing at different stages, effectively preventing expert collapse.
4. **Four Complementary Expert Designs**: Spatial, contextual, boundary, and shape experts cover the diverse visual cues required for RIS.
5. **Zero-Shot Cross-Dataset Generalization**: Demonstrates strong transfer capabilities across the RefCOCO suite.

## Limitations & Future Work
- The number and types of experts are manually designed (2+4); automated expert architecture search could be explored.
- The selection of k in Top-k routing requires hyperparameter tuning, and different datasets may have different optimal k values.
- Evaluated only on the RefCOCO suite, lacking more diverse RIS benchmarks (such as PhraseCut).
- The contextual expert in the fusion layer introduces self-attention, which incurs higher computational costs compared to other experts.
- The performance combined with larger backbones (such as ViT-L/ViT-G) remains unexplored.

## Related Work & Insights
- vs. **DETRIS**: Built upon DETRIS, SERA improves the average mIoU from 70.4 to 71.1 by incorporating MoE.
- vs. **LAVT/CRIS**: These traditional full-finetuning methods only reach 72.7/70.5 on RefCOCO, which is significantly lower than SERA's 76.5.
- vs. **VATEX**: VATEX reaches 78.2 on RefCOCO val (with full fine-tuning), whereas SERA achieves 76.5 while updating <1% of the parameters.
- vs. **V-MoE**: V-MoE applies MoE for image classification scalability, while SERA is the first to employ MoE expert routing for dense pixel-level referring segmentation.

## Insights & Connections
- The application of MoE in dense prediction tasks warrants further exploration, extending beyond RIS to semantic segmentation and panoptic segmentation.
- The design concepts of the four experts (boundary, spatial, context, and shape) can be generalized to tasks like semantic segmentation and instance segmentation.
- The differentiated strategies of soft routing vs. sparse routing offer valuable reference for other MoE usage scenarios.
- Performance improvements under extreme parameter efficiency indicate that pre-trained models contain rich information that simply needs to be "activated".
- The pipeline design of SERA-Adapter ("projection $\rightarrow$ spatial grid $\rightarrow$ multi-scale convolution $\rightarrow$ expert refinement $\rightarrow$ cross-modal attention") is clear and can serve as a general adapter paradigm.
- The combination of traditional operators like Sobel/Laplacian with learnable modules (boundary/shape experts) demonstrates the effectiveness of incorporating prior knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐ (First systematic application of MoE in RIS)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three major benchmarks + ablation + zero-shot generalization)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, expert design is theoretically motivated)
- Value: ⭐⭐⭐⭐ (Provides new insights for parameter-efficient dense prediction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting](../../CVPR2026/segmentation/promptmoe_a_segmentation_refinement_framework_leveraging_mixture_of_experts_for_.md)
- [\[AAAI 2026\] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts](../../AAAI2026/segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)
- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](../../ICCV2025/segmentation/latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](../../NeurIPS2025/segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)
- [\[ECCV 2024\] ReMamber: Referring Image Segmentation with Mamba Twister](../../ECCV2024/segmentation/remamber_referring_image_segmentation_with_mamba_twister.md)

</div>

<!-- RELATED:END -->
