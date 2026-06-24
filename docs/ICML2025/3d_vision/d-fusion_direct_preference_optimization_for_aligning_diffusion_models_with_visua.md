---
title: >-
  [Paper Note] D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples
description: >-
  [ICML 2025][3D Vision][Diffusion Models] This paper proposes D-Fusion, a method that constructs visually consistent preference data pairs and preserves denoising trajectories via mask-guided Self-Attention Fusion. It addresses the performance limitations in training diffusion models with DPO caused by visual inconsistency, significantly improving prompt-image alignment quality across various RL algorithms and prompt types.
tags:
  - "ICML 2025"
  - "3D Vision"
  - "Diffusion Models"
  - "DPO"
  - "Visual Consistency"
  - "Self-Attention Fusion"
  - "Text-to-Image Alignment"
date: 2026-05-08
content_hash: 701ff602683a0507
---

# D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples

**Conference**: ICML 2025  
**arXiv**: [2505.22002](https://arxiv.org/abs/2505.22002)  
**Code**: [https://github.com/hu-zijing/D-Fusion](https://github.com/hu-zijing/D-Fusion)  
**Area**: LLM Alignment/RLHF  
**Keywords**: Diffusion Models, DPO, Visual Consistency, Self-Attention Fusion, Text-to-Image Alignment

## TL;DR
This paper proposes D-Fusion, a method that constructs visually consistent preference data pairs and preserves denoising trajectories via mask-guided Self-Attention Fusion. It addresses the performance limitations in training diffusion models with DPO caused by visual inconsistency, significantly improving prompt-image alignment quality across various RL algorithms and prompt types.

## Background & Motivation
**Background**: Diffusion models have achieved remarkable success in text-to-image generation, but the misalignment between generated images and text prompts remains severe, limiting practical applications.

**Limitations of Prior Work**: Recent studies have introduced DPO into diffusion models to enhance alignment, but the efficacy remains limited. The core reason lies in the **visual inconsistency** present in DPO training data: high-preference and low-preference images denoised from different starting noises differ significantly in structure, style, and appearance, making it difficult for the model to identify which factors are positively correlated with alignment.

**Key Challenge**: In RLHF for language models, fine-grained edits can be made at the token level to obtain consistent training pairs. In diffusion models, however, manual editing operations are performed at the pixel level, which **loses the step-by-step denoising trajectory**, rendering the edited images unusable for RL training.

**Goal**: How to generate RL-trainable image pairs that are both **visually consistent** and **preserve denoising trajectories**?

**Key Insight**: Leveraging the self-attention mechanism within the U-Net of diffusion models, attention fusion is performed progressively during the denoising process. This guarantees visual consistency between the generated image and the original low-preference image, while naturally preserving the complete denoising trajectory.

**Core Idea**: Utilizing cross-attention masks to locate alignment-relevant regions, the alignment information of high-preference samples is fused into low-preference samples at the self-attention layer. This generates visually consistent samples that can be directly utilized for DPO training.

## Method

### Overall Architecture
D-Fusion consists of two stages: (1) Sampling stage: a target image that is visually consistent with the base image and equivalently aligned as the reference image is generated via mask-guided self-attention fusion; (2) Training stage: intermediate states during the fusion process are collected to form denoising trajectories for training RL algorithms such as DPO, DDPO, and DPOK.

### Key Designs

1. **Cross-Attention Mask Extraction**:

    - **Function**: Automatically extract masks of alignment-relevant regions from the denoising process of the reference image (high-preference image).
    - **Mechanism**: Leverage the attention distribution of prompt keywords in the cross-attention map to locate target regions in the image associated with alignment.
    - **Design Motivation**: Manual mask annotation is costly and not scalable. The cross-attention map naturally reflects the correspondence between prompt words and image regions, enabling automated extraction.

2. **Self-Attention Fusion**:

    - **Function**: At each timestep of the denoising process, the self-attention features of the reference image are fused into the base image within the masked region.
    - **Mechanism**: Since self-attention controls the structure and style of an image, replacing self-attention features in alignment-relevant regions transfers alignment information while maintaining the base image's original appearance in unmasked regions.
    - **Design Motivation**: Unlike direct pixel editing, self-attention fusion is performed step-by-step during the denoising process, thus naturally preserving the complete denoising trajectory.
    - **Difference from prior methods**: Methods like Prompt-to-Prompt transform images across different prompts, whereas D-Fusion transfers alignment information from one image to another under the same prompt.

3. **Denoising Trajectory Preservation and RL Training**:

    - **Function**: Collect intermediate noise states at each timestep during the fusion process to compose the complete denoising trajectory of the target image.
    - **Mechanism**: Since fusion is conducted step-by-step, the (state, action) pairs at each step naturally form an MDP trajectory.
    - **Design Motivation**: RL algorithms such as DPO and PPO require access to denoising trajectories to compute policy gradients; manually edited images lack this trajectory information.

### Loss & Training
- The standard Diffusion-DPO loss function is adopted, utilizing the base image as the low-preference sample and the target image as the high-preference sample for training.
- As a data construction method, D-Fusion is compatible with multiple RL algorithms such as DPO, DDPO, and DPOK.
- Preference pairs in DPO training consist of (base image, target image), with preference order determined by evaluators like CLIP.
- In DDPO and DPOK, the denoising trajectory of the target image directly serves as the positive sample trajectory for policy optimization.
- Shared random noise is used during training to ensure visual consistency between the base image and the target image.
- Fusion operations are applied only during the sampling phase, while standard RL algorithms are utilized in the training phase, incurring no extra training overhead.

## Key Experimental Results

### Main Results

| Prompt Type | Index | SD + DPO | SD + D-Fusion(DPO) | Gain |
|------------|------|----------|-------------------|------|
| Object Action | CLIP Score | Lower | Significant Improvement | Obvious |
| Object Attribute | CLIP Score | Lower | Significant Improvement | Obvious |
| Spatial Relation | CLIP Score | Lower | Significant Improvement | Obvious |

### Compatibility with Different RL Algorithms

| RL Algorithm | Without D-Fusion | With D-Fusion | Description |
|---------|------------|------------|------|
| DPO | Baseline | Improvement | Effective across all prompt types |
| DDPO | Baseline | Improvement | Compatible with policy gradient methods |
| DPOK | Baseline | Improvement | Compatible with hybrid methods |

### Key Findings
- Compared to traditional random sampling, visually consistent training pairs significantly boost the efficacy of DPO on diffusion models.
- Target images generated by D-Fusion are not only visually consistent with the base images but also exhibit identical alignment quality to the reference images.
- The method is effective across three prompt types (action, attribute, spatial relation), demonstrating its generalizability.
- D-Fusion can be seamlessly integrated with multiple RL algorithms, not limited to DPO.
- Ablation studies show that mask guidance is critical for fusion quality—global fusion without masks destroys visual consistency.
- The choice of fusion timesteps also impacts performance: early-timestep fusion affects global structure more, while late-timestep fusion affects details.

## Highlights & Insights
- **First to explicitly identify** the core challenge of visual inconsistency in DPO training for diffusion models, providing a fresh perspective for research in this field.
- Cleverly utilizes self-attention characteristics to achieve the dual goals of "fusing alignment information while preserving denoising trajectories."
- High generalizability allows it to serve as a data augmentation module integrated into any RL-based diffusion model fine-tuning pipeline.
- Inspired by the sentence-level to token-level fine-grained training shift in language model RLHF, a similar "fine-grained" consistency training paradigm is analogously established for diffusion models.

## Limitations & Future Work
- The evaluation is primarily conducted on Stable Diffusion, and has not yet been extended to more advanced diffusion architectures such as SDXL or DiT.
- Mask extraction relies heavily on the quality of cross-attention maps, which may be imprecise for certain complex prompts.
- Self-attention fusion incurs higher computational overhead than standard sampling, potentially impacting training efficiency.
- The codebase is relatively concise, and specific numerical details from some experiments in the paper are not fully retrieved.
- Future work can explore attention-free fusion within other modules, such as Resnet blocks.
- Adaptive mask strategies can be explored to dynamically adjust the fusion region size based on prompt complexity.

## Related Work & Insights
- Technically related to attention control methods like Prompt-to-Prompt and Plug-and-Play, but with different goals (this work focuses on alignment, whereas they focus on editing).
- Inspired by the transition from sentence-level to token-level RLHF in language models, translating a similar concept to the pixel-to-attention level in images.
- Opens up a new direction of "data consistency" for diffusion model alignment research.
- Differs from image editing methods like Imagic and InstructPix2Pix as D-Fusion preserves the entire denoising trajectory.
- Provides fresh empirical support for the importance of data quality in preference learning.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] ADHMR: Aligning Diffusion-based Human Mesh Recovery via Direct Preference Optimization](adhmr_aligning_diffusion-based_human_mesh_recovery_via_direct_preference_optimiz.md)
- [\[CVPR 2026\] Circular-DPO: Aligning Multi-Stage 3D Generative Models via Preference Feedback Loop](../../CVPR2026/3d_vision/circular-dpo_aligning_multi-stage_3d_generative_models_via_preference_feedback_l.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](../../ICCV2025/3d_vision/spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [\[NeurIPS 2025\] Linearly Constrained Diffusion Implicit Models](../../NeurIPS2025/3d_vision/linearly_constrained_diffusion_implicit_models.md)

</div>

<!-- RELATED:END -->
