---
title: >-
  [Paper Note] InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions
description: >-
  [ICLR 2026][Image Restoration][Multi-person video generation] InterActHuman is proposed to enable audio-driven video generation for multi-person/human interaction scenarios via an automatic spatiotemporal layout mask pre…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "Multi-person video generation"
  - "audio-driven animation"
  - "mask prediction"
  - "layout control"
  - "DiT"
date: 2026-05-08
content_hash: 9a69b2aa172947e0
---

# InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions

**Conference**: ICLR 2026
**arXiv**: [2506.09984](https://arxiv.org/abs/2506.09984)
**Code**: Partially open-sourced (reproduction based on Wan2.1)
**Area**: Image Restoration
**Keywords**: Multi-person video generation, audio-driven animation, mask prediction, layout control, DiT

## TL;DR
InterActHuman is proposed to enable audio-driven video generation for multi-person/human interaction scenarios via an automatic spatiotemporal layout mask predictor and an iterative mask guidance strategy, supporting independent speech-driven lip synchronization and body motion for each character.

## Background & Motivation

**Background**: Audio-driven human animation has achieved significant progress (CyberHost, OmniHuman), but primarily focuses on single-person scenarios. Multi-person interactive video generation requires precise matching of different audio streams to their respective characters.

**Limitations of Prior Work**: The core challenge in multi-person scenarios is a chicken-and-egg problem — injecting audio into the correct spatial locations requires knowledge of each character's layout (mask), but the mask depends on the final generated video, which has not yet been produced. Using fixed masks leads to motion artifacts and unnatural rigid effects.

**Key Challenge**: Global audio conditioning cannot distinguish which character is speaking in a multi-person scene; fixed masks cannot adapt to character motion; dynamic masks are needed at inference time but the video is not yet determined.

**Goal**: (a) Audio-to-character matching in multi-person scenes, (b) dynamic layout prediction at inference time, and (c) identity-preserving multi-concept video generation.

**Key Insight**: During the denoising process of a DiT (Diffusion Transformer), intermediate features are used to predict the mask at the current step, which then guides audio injection at the next step — iterative refinement.

**Core Idea**: A lightweight mask prediction head infers character layouts from DiT intermediate features, and the iterative denoising process progressively refines both the masks and the video generation.

## Method

### Overall Architecture
Built upon a MMDiT + Flow Matching architecture (7B-parameter DiT), the model takes reference character images and corresponding audio as inputs. Three core modules — multi-concept reference injection, mask predictor, and local audio condition injection — work collaboratively to generate multi-person interactive videos.

### Key Designs

1. **Multi-Concept Reference Image Injection**:

    - Function: Inject identity information from multiple reference character images into the generation process.
    - Mechanism: Each reference image $X_i$ is encoded into a latent representation $x_i$ via VAE, then concatenated with the noisy video latent $v$ along the channel dimension. The DiT's self-attention layers are reused directly for feature interaction, requiring no additional parameters.
    - Design Motivation: Avoids introducing extra networks (e.g., IP-Adapter), reducing parameter count and training complexity, while implicitly injecting identity information through the DiT's own attention mechanism.

2. **Mask Predictor**:

    - Function: Predict the spatiotemporal mask of each reference character in the video from DiT intermediate features.
    - Mechanism: A lightweight head (linear projection + 3D RoPE + cross-attention + 2-layer MLP + sigmoid) is appended to each DiT layer. Cross-attention is computed between video hidden features $h_v$ and reference features $h_i^r$, producing soft masks in $[0, 1]$. The final mask is averaged over the last several layers.
    - Design Motivation: Adds only 56M parameters (vs. 7B DiT) and 0.013s inference overhead per DiT block. Mask training uses focal loss to handle foreground-background imbalance.

3. **Iterative Mask Prediction Strategy (Denoising-time Mask Guidance)**:

    - Function: Resolves the chicken-and-egg problem at inference time — the video is unknown at the start of inference, making mask prediction infeasible.
    - Mechanism: Two-stage inference — Stage 1 (first 10 steps) proceeds without masks, allowing the DiT to form a coarse layout; Stage 2 (subsequent steps) caches the mask from step $k$ and uses it to guide audio injection at step $k+1$. Masks are progressively refined throughout the denoising process.
    - Design Motivation: Analogous to iterative refinement, leveraging the progressive nature of the diffusion process. Early steps establish the coarse layout; later steps refine the details.

4. **Local Audio Condition Injection**:

    - Function: Inject each character's audio only into the spatial tokens corresponding to that character's location.
    - Mechanism: Audio features are extracted via wav2vec and injected into the DiT via cross-attention. Crucially, the mask predicted at the previous step is used for soft weighting — audio influences only the tokens of the corresponding character, with soft transitions at mask boundaries.
    - Design Motivation: Global audio injection cannot distinguish who is speaking (experiment shows Sync-D of 9.482); local injection reduces Sync-D to 6.670.

### Loss & Training
- Flow matching objective: velocity prediction loss
- Mask loss: focal loss (to handle foreground-background imbalance)
- Two-stage training: single-person audio pre-training followed by multi-concept fine-tuning
- Data: 2.6 million video-mask-caption triplets, annotated with dense captions from Qwen2-VL and masks from Grounding-SAM2

## Key Experimental Results

### Main Results (Single-Person Audio-Driven)

| Method | Sync-C (↑) | HKV (↑) | Sync-D (↓) | FVD (↓) |
|--------|-----------|---------|-----------|--------|
| CyberHost | 6.627 | 24.733 | 8.974 | 54.797 |
| OmniHuman (no mask) | 7.443 | 47.561 | 9.482 | 33.895 |
| OmniHuman (fixed mask) | - | - | 7.068 | 40.239 |
| **InterActHuman** | **7.272** | **59.635** | **6.670** | **22.881** |

### User Study (Multi-Person Audio-Driven)

| Method | Avg. Score | Top-1 Selection Rate |
|--------|-----------|---------------------|
| Kling | 1.70 | 14.5% |
| OmniHuman | 1.82 | 25.6% |
| **InterActHuman** | **2.48** | **59.9%** |

### Ablation Study

| Audio Injection Strategy | Sync-D (↓) | FVD (↓) |
|--------------------------|-----------|--------|
| Global audio conditioning | 9.482 | 33.895 |
| ID Embedding | 8.627 | 35.665 |
| Fixed mask | 7.068 | 40.239 |
| **Predicted mask** | **6.670** | **22.881** |

### Key Findings
- Predicted masks outperform fixed masks across all metrics: Sync-D reduced by 5.6%, FVD reduced by 43.1% (40.239 → 22.881).
- HKV (hand keypoint variance) is the highest among all methods (59.635), indicating the richest body motion.
- Multi-concept identity preservation (CLIP-I = 0.744, DINO-I = 0.533) significantly surpasses commercial products such as Pika and Vidu.
- The mask predictor introduces minimal overhead: each additional reference adds only 0.4s (vs. 6.5s for the base DiT).

## Highlights & Insights
- **Iterative Mask Strategy**: Elegantly exploits the multi-step nature of the diffusion process to progressively refine layout masks during denoising. This is a clean bootstrapping solution that requires no external detector.
- **Zero-parameter Reference Injection**: Directly reuses DiT self-attention by stacking VAE latents of reference images, maintaining architectural simplicity.
- **Industrial-Scale Data Pipeline**: The annotation pipeline for 2.6 million videos (Qwen2-VL captioning + Gemini structured parsing + SAM2 masking) constitutes a valuable engineering contribution in its own right.

## Limitations & Future Work
- Inference time scales quadratically with the number of reference characters (attention complexity).
- Mask prediction quality depends on DiT intermediate feature quality; masks may be inaccurate at very early denoising steps.
- Currently supports at most 3 interacting characters; scenarios with more participants remain unvalidated.
- Audio conditioning is limited to speech; driving via music or ambient sound is unexplored.
- The core model is based on ByteDance's internal 7B DiT, posing barriers to full reproduction.

## Related Work & Insights
- **vs. OmniHuman**: The direct competitor; however, OmniHuman does not support multi-person audio matching.
- **vs. CyberHost**: An earlier audio-driven method with a substantially larger performance gap.
- **vs. Phantom (multi-concept customization)**: Phantom excels at multi-concept generation but does not support audio-driven animation; InterActHuman addresses both.

## Rating
- Novelty: ⭐⭐⭐⭐ The iterative mask prediction strategy is novel, though the overall framework integrates existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across single-person, multi-person, and multi-concept settings, with user studies and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Architecture descriptions are clear, though the dense mathematical notation requires careful reading.
- Value: ⭐⭐⭐⭐⭐ A practical system for multi-person interactive video generation with high industrial applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](../../CVPR2026/image_restoration/phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](../../NeurIPS2025/image_restoration/audio_super-resolution_with_latent_bridge_models.md)
- [\[CVPR 2026\] UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation](../../CVPR2026/image_restoration/udapose_unsupervised_domain_adaptation_for_low_light_human_pose_estimation.md)
- [\[AAAI 2026\] ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration](../../AAAI2026/image_restoration/clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](../../AAAI2026/image_restoration/large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)

</div>

<!-- RELATED:END -->
