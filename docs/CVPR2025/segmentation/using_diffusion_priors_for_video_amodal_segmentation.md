---
title: >-
  [Paper Note] Using Diffusion Priors for Video Amodal Segmentation
description: >-
  [CVPR 2025][Segmentation][Video amodal segmentation] This paper reformulates video amodal segmentation as a conditional generation task, leveraging the shape priors of a pretrained video diffusion model (Stable Video Diffusion). Conditioning on modal masks and pseudo-depth maps, it achieves completion in occluded areas with an improvement of up to 13% mIoU, and realizes video-level amodal content completion for the first time.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Video amodal segmentation"
  - "video diffusion models"
  - "occlusion completion"
  - "temporal consistency"
  - "depth conditioning"
date: 2026-05-08
content_hash: 7b73f124f1e1c0f9
---

# Using Diffusion Priors for Video Amodal Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2412.04623](https://arxiv.org/abs/2412.04623)  
**Code**: None  
**Area**: Segmentation  
**Keywords**: Video amodal segmentation, video diffusion models, occlusion completion, temporal consistency, depth conditioning

## TL;DR

This paper reformulates video amodal segmentation as a conditional generation task, leveraging the shape priors of a pretrained video diffusion model (Stable Video Diffusion). Conditioning on modal masks and pseudo-depth maps, it achieves completion in occluded areas with an improvement of up to 13% mIoU, and realizes video-level amodal content completion for the first time.

## Background & Motivation

**Background**: Current mainstream segmentation methods (such as the SAM series) only handle the visible parts of objects (modal segmentation) without considering their complete shapes when occluded. Amodal segmentation aims to predict the complete boundary of objects, including occluded regions, which is crucial for robotic manipulation, autonomous driving, and video editing.

**Limitations of Prior Work**: (1) Single-frame amodal methods cannot handle severe or full occlusion—when an object is heavily occluded, single-frame information is insufficient to infer its complete shape; (2) Existing video amodal methods (such as SaVos, EoRaS) are limited to rigid objects and rely on additional inputs (camera poses, optical flow), leading to poor generalization; (3) The lack of real-world amodal annotation data constrains the training and evaluation of these methods.

**Key Challenge**: Amodal perception is inherently an ill-posed problem, as there are multiple plausible ways to complete occluded regions. Single-frame methods lack sufficient information, while multi-frame methods are constrained by data availability and representation capacity.

**Goal**: (1) Infer the complete shapes of objects under severe or full occlusion by leveraging temporal information; (2) Avoid relying on extra inputs like camera poses or optical flow; (3) Concurrently achieve both amodal mask prediction and RGB content completion.

**Key Insight**: Large-scale pretrained video diffusion models internalize rich priors of object shapes—these models "learn" how object boundaries should extend while generating pixels. Leveraging this prior for conditional generation can naturally address occlusion completion.

**Core Idea**: Reformulate Stable Video Diffusion as a conditional generative model. It is conditioned on modal mask sequences and pseudo-depth maps to generate amodal mask sequences (Stage 1), and then conditioned on modal RGB content and amodal masks to complete the appearance of occluded regions (Stage 2).

## Method

### Overall Architecture

The proposed method consists of two stages: The first stage takes modal mask sequences and pseudo-depth maps as inputs to generate amodal mask sequences using a modified SVD model; the second stage uses the predicted amodal masks from the first stage and the modal RGB content of objects as conditions to complete the RGB content of occluded areas through another SVD model. Both stages share the same 3D UNet architecture but employ different conditions.

### Key Designs

1. **Stage 1: Modal-to-Amodal**:

    - **Function**: Predict complete amodal mask sequences from modal mask sequences.
    - **Mechanism**: Replace the input condition of SVD from RGB images with binary modal mask sequences. Since the VAE requires a 3-channel input, the single-channel mask is replicated three times and encoded into latents. The encoded latents are concatenated with noise latents as inputs to the 3D UNet. Meanwhile, the CLIP embeddings of the modal masks are injected via cross-attention, providing temporal information regarding the visibility of objects across frames. Unlike the original SVD which duplicates a single frame $T$ times, the input here consists of $T$ independent modal frames.
    - **Design Motivation**: Harness the powerful shape priors SVD acquired through pretraining on 152 million samples, transferring these priors to the amodal segmentation task via conditional generation.

2. **Pseudo-Depth Conditioning**:

    - **Function**: Provide implicit cues regarding occlusion relationships within the scene.
    - **Mechanism**: Convert RGB frames into pseudo-depth maps using Depth Anything V2, which are then concatenated as extra channels to the 3D UNet input. This updates the input latent shape to $\mathcal{R}^{T \times 3C_1 \times \frac{H}{F} \times \frac{W}{F}}$. Training employs a two-stage fine-tuning strategy: first training the model with only mask conditions, and then initializing the mask-and-depth-conditioned model based on it. The newly added depth channels are initialized via zero convolutions, preserving the learned capability from the mask conditions.
    - **Design Motivation**: Occluding objects are typically closer to the camera, so depth maps directly reveal the occluder-occludee relationship. Experiments demonstrate that depth is more effective than RGB frames: reliance on textures and appearances from RGB frames can degrade generalization performance.

3. **Stage 2: Amodal Content Completion**:

    - **Function**: Complete the RGB appearance of objects in occluded regions.
    - **Mechanism**: Employ a second SVD model sharing the same 3D UNet architecture but with different conditions: the object's modal RGB content (appearance of visible regions) and the amodal masks predicted in Stage 1. Since synthetic datasets also lack ground-truth RGB annotations for occluded regions, self-supervised training pairs are constructed: high-visibility (>95%) objects are selected, and amodal masks of other objects are randomly overlaid to simulate occlusions.
    - **Design Motivation**: This is the first work attempting video-level amodal content completion. The self-supervised training pair construction effectively overcomes the lack of training data.

### Loss & Training

The model is trained using the EDM framework with a weighted L2 denoising objective. A two-stage fine-tuning strategy (first masks, then adding depth) combined with zero-convolution initialization ensures stable training. Trained on the SAIL-VOS synthetic dataset (128x256 resolution, batch size 8), it takes approximately 30 hours using 8x RTX 3090. During inference, a higher resolution (256x512) is used to obtain more precise pixel-level predictions.

## Key Experimental Results

### Main Results

| Method | SAIL-VOS mIoU | SAIL-VOS mIoU_occ | TAO-Amodal AP50 | Type |
|------|-------------|------------------|----------------|------|
| PCNet-M | 74.20 | 42.52 | 85.11 | Single-frame |
| AISFormer | 73.51 | 39.16 | 81.93 | Single-frame |
| pix2gestalt (Top-1) | 54.83 | 26.59 | 57.50 | Single-frame diffusion |
| EoRaS | 81.76 (MOVi-B) | 49.39 | - | Multi-frame |
| 3D-UNet baseline | 72.79 | 39.54 | 83.83 | Multi-frame |
| **Ours (Top-1)** | **77.07** | **55.12** | **89.25** | Multi-frame diffusion |
| **Ours (Top-3)** | **79.23** | **59.69** | **92.46** | Multi-frame diffusion |

### Ablation Study

| Condition Configuration | SAIL-VOS mIoU | mIoU_occ | TAO AP50 | Explanation |
|---------|-------------|----------|---------|------|
| Mask only | 75.17 | 51.28 | 85.03 | Baseline condition |
| Mask + RGB | 76.59 | 53.30 | 86.59 | RGB is helpful but limited |
| Mask + Depth | 77.07 | 55.12 | 89.25 | Depth performs better |
| Mask + RGB + Depth | 77.19 | 54.59 | 87.16 | Adding RGB hurts generalization |

### Key Findings

- Improvement is most significant in occluded regions (measured by mIoU_occ, outperforming the runner-up PCNet-M by nearly 13%), demonstrating that the paper's core strength lies in handling severe occlusions.
- Trained strictly on the synthetic SAIL-VOS dataset, the model still leads by a large margin on the real-world TAO-Amodal dataset in a zero-shot setting, demonstrating the strong generalization capabilities of SVD priors.
- Pseudo-depth conditioning is more effective and generalizes better than RGB conditioning—the texture details in RGB can behave as noise when generalizing across datasets.
- In the user study, 85.6% of users preferred the content completion results of the proposed method (compared to pix2gestalt).

## Highlights & Insights

- **Reformulating the segmentation task as conditional video generation** is a highly clever shift of perspective. It naturally introduces temporal consistency while leveraging rich priors from large-scale pretrained models. This paradigm of "borrowing generative models for discriminative tasks" can be transferred to many other video understanding tasks.
- The finding that **pseudo-depth encodes occlusion information better than RGB** is highly insightful. Depth maps inherently represent near-far spatial relationships, while RGB texture details can be detrimental when generalizing across domains. This discovery provides guidance for selecting conditioning signals.
- **The first video-level amodal content completion** is a significant contribution. The self-supervised training pair construction is simple yet effective and can scale to other video understanding tasks lacking annotations.

## Limitations & Future Work

- Training and evaluation are mostly conducted on synthetic data. Real-world amodal annotations remain scarce, which limits more comprehensive evaluations.
- Inference efficiency is not discussed in detail—utilizing 25-step EDM denoising may still be too slow for real-time applications.
- The model has limited capability in handling out-of-frame occlusion.
- Quality evaluation for the content completion stage still relies on user studies, lacking quantitative metrics.

## Related Work & Insights

- **vs PCNet-M**: PCNet-M is a classic single-frame modal-to-amodal prediction method. While it works well in simple occlusion scenarios, it lacks sufficient information under severe occlusions. The proposed method significantly outperforms it by leveraging multi-frame context and diffusion priors.
- **vs pix2gestalt**: Also a diffusion-based method but operates solely on single frames, failing to guarantee temporal consistency and performing poorly under high occlusion.
- **vs EoRaS**: EoRaS relies on extra inputs like optical flow and is limited to rigid objects, whereas the proposed method requires no extra inputs and handles non-rigid objects.

## Rating

- Novelty: ⭐⭐⭐⭐ Creatively applies video diffusion models to amodal segmentation, a clever shift of perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on four datasets with multiple baselines, detailed ablation studies, and a user study.
- Writing Quality: ⭐⭐⭐⭐ Clear and systematic, with a solid logical motivation.
- Value: ⭐⭐⭐⭐ Achieves significant progress in the critical task of occlusion understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Generative Video Propagation](generative_video_propagation.md)
- [\[CVPR 2026\] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](../../CVPR2026/segmentation/from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)
- [\[CVPR 2025\] LiVOS: Light Video Object Segmentation with Gated Linear Matching](livos_light_video_object_segmentation_with_gated_linear_matching.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)

</div>

<!-- RELATED:END -->
