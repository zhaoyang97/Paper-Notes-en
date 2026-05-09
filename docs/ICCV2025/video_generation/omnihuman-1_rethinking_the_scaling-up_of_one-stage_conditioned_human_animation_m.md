---
title: >-
  [Paper Note] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models
description: >-
  [ICCV 2025][Video Generation][human animation] This paper proposes OmniHuman, a multi-condition human animation generation framework based on Diffusion Transformer. Through an omni-conditions training strategy that mixes motion-related conditions including text, audio, and pose, the framework enables effective data scaling. It is the first single model to support audio-driven human video generation with arbitrary body proportions and aspect ratios, achieving state-of-the-art performance on both portrait and half-body animation tasks.
tags:
  - ICCV 2025
  - Video Generation
  - human animation
  - diffusion transformer
  - audio-driven
  - omni-conditions training
  - data scaling
date: 2026-05-08
content_hash: d06261000fa3875a
---

# OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models

**Conference**: ICCV 2025
**arXiv**: [2502.01061](https://arxiv.org/abs/2502.01061)
**Code**: Unavailable (Project page: [https://omnihuman-lab.github.io/](https://omnihuman-lab.github.io/))
**Area**: Video Generation
**Keywords**: human animation, diffusion transformer, audio-driven, omni-conditions training, data scaling

## TL;DR

This paper proposes OmniHuman, a multi-condition human animation generation framework based on Diffusion Transformer. Through an omni-conditions training strategy that mixes motion-related conditions including text, audio, and pose, the framework enables effective data scaling. It is the first single model to support audio-driven human video generation with arbitrary body proportions and aspect ratios, achieving state-of-the-art performance on both portrait and half-body animation tasks.

## Background & Motivation

DiT-based general video generation has achieved remarkable progress through large-scale data (on the order of O(100M) clips), yet the scaling effect in human animation remains largely unexplored. Existing audio-driven methods suffer from severe data waste: audio signals are primarily correlated with facial expressions and unrelated to body pose or background motion, necessitating strict filtering and cropping (lip-sync quality, pose visibility, etc.) that retains fewer than 10% of raw data.

**Core insight**: Mixing multiple condition signals (text, audio, pose) during training significantly reduces data waste. Data discarded by single-condition models can still be utilized for weaker-condition tasks (e.g., text-conditioned generation); moreover, different condition signals are complementary—for instance, pose conditions can compensate for audio's weak control over body motion.

## Method

### Overall Architecture

OmniHuman builds upon a text-to-video pretrained model with an MMDiT backbone and introduces multi-condition support via three-stage progressive training. The model employs a Causal 3D VAE to project videos into a compact latent space. Inputs consist of a reference image combined with one or more driving conditions (text/audio/pose), and the output is the generated human animation video.

### Key Designs

1. **Driving Condition Injection**:

   - **Audio condition**: Multi-scale acoustic features extracted by wav2vec → compressed and aligned to the DiT hidden dimension and 25 fps frame rate via MLP → concatenated with adjacent timestamp audio → injected into each MMDiT block via per-frame cross-attention.
   - **Pose condition**: Skeleton sequence encoded by a Pose Guider → concatenated with adjacent frames → stacked channel-wise with the noisy latent before being fed into the model.
   - **Text condition**: Processed through the original MMDiT text branch without modification.

2. **Appearance Condition Injection (Minimalist Design)**: Rather than introducing an additional reference network (which would double parameters), the original DiT backbone is reused to encode the reference image. The reference image and noisy video latents are flattened into token sequences, packed together, and fed into the DiT for interaction via self-attention. Reference and video tokens are distinguished by setting the temporal component of the 3D RoPE to zero for reference image tokens, incurring zero additional parameter overhead.

3. **Omni-Conditions Training Strategy (Core Innovation)**: The strategy follows two principles:

   - **Principle 1**: Stronger-condition tasks can leverage weaker-condition tasks and their data to expand training data. For example, data filtered out due to poor lip-sync quality can still be used for text+image→video tasks. Stage 1 trains only the text+image→video task to maximize data utilization.
   - **Principle 2**: Stronger conditions should occupy a lower training proportion. Strong conditions (e.g., pose) provide precise control but may suppress the learning of weaker conditions (e.g., audio). Stage 2 introduces audio without pose; Stage 3 introduces all conditions, with training proportions progressively halved for text/audio/pose (T=90%, A=50%, P=25%), assigning higher gradient weight to more challenging tasks.

### Loss & Training

- Diffusion training based on Rectified Flow
- Learning rate $5 \times 10^{-5}$, AdamW optimizer, gradient clipping 1.0, batch size 256, weight decay 0.01
- Trained on 400 A100 GPUs, approximately 10 days per stage
- 18.7K hours of human-related data, with 13% meeting audio and pose quality criteria
- CFG applied to audio and text at inference (scale=6.5); CFG not applied to pose
- Long video generation: the last 5 frames of the previous segment serve as motion frames for the next

## Key Experimental Results

### Main Results (Audio-Driven Portrait Animation)

| Method | CelebV-HQ IQA↑ | CelebV-HQ Sync-C↑ | CelebV-HQ FID↓ | CelebV-HQ FVD↓ | RAVDESS Sync-C↑ | RAVDESS FVD↓ |
|------|----------------|-------------------|----------------|----------------|----------------|-------------|
| SadTalker | 2.953 | 3.843 | 36.65 | 171.85 | 4.304 | 22.52 |
| Hallo | 3.505 | 4.130 | 35.96 | 53.99 | 4.062 | 38.47 |
| Loopy | 3.780 | 4.849 | 33.20 | 49.15 | 4.814 | 16.13 |
| Hallo-3 | 3.451 | 3.933 | 38.48 | 42.13 | 4.448 | 26.03 |
| **OmniHuman** | **3.875** | **5.199** | **31.44** | **46.39** | **5.255** | **15.91** |

**Audio-Driven Half-Body Animation (CyberHost Test Set)**:

| Method | IQA↑ | Sync-C↑ | FID↓ | FVD↓ | HKV | HKC↑ |
|------|------|---------|------|------|-----|------|
| DiffTED | 2.701 | 0.926 | 95.46 | 58.87 | - | 0.769 |
| CyberHost | 3.990 | 6.627 | 32.97 | 28.00 | 24.73 | 0.884 |
| **OmniHuman** | **4.142** | **7.443** | **31.64** | **27.03** | **47.56** | **0.898** |

Hand Keypoint Variance (HKV) increases from 24.73 to 47.56, nearly doubling the richness of hand motion.

### Ablation Study (Omni-Conditions Training Strategy)

**Principle 1 Validation (Effect of Text Data Proportion)**:

| T-Data Ratio | CelebV-HQ FVD↓ | CelebV-HQ Sync-C↑ | CyberHost HKV | CyberHost HKC↑ |
|---|---|---|---|---|
| 0% | 47.86 | 4.299 | 35.82 | 0.871 |
| 25% | 47.04 | 3.311 | 40.39 | 0.877 |
| 50% | 46.22 | 3.696 | 40.69 | 0.872 |
| 100% | 43.74 | 4.987 | 43.54 | 0.882 |

Increasing the proportion of text data consistently improves FVD, FID, lip-sync accuracy, and gesture generation quality.

**Principle 2 Validation (Condition Introduction Order and Proportion)**:

| Strategy | CelebV-HQ Sync-C↑ | CelebV-HQ FVD↓ | RAVDESS FVD↓ |
|------|---|---|---|
| IA (audio only) | 4.987 | 43.74 | 15.13 |
| IPA (pose first) | 2.788 | 44.70 | 25.05 |
| IAP, A<P | 4.201 | 44.63 | 17.18 |
| **IAP, A>P** | **4.934** | **43.36** | **15.66** |

Introducing pose before audio (IPA) significantly degrades final performance; maintaining A>P training proportion yields the best results.

### Key Findings

- When text-conditioned data is insufficient, lip-sync metrics can degrade; sufficient data volume is required for positive cross-modal synergy.
- Mixed-condition training (IAP) decouples hand motion from audio correlation, mitigating over-exaggerated gesture generation.
- The omni-conditions training strategy enables the model to progressively follow the quality distribution of input images rather than that of training data.
- This work is the first to achieve audio-driven generation with a single model supporting arbitrary body proportions and image styles.

## Highlights & Insights

- The design principle of "weaker conditions receive higher training proportion" offers deep insight and effectively resolves the seesaw effect in multi-condition training.
- The minimalist design for appearance condition injection—reusing the backbone with zero additional parameters—is key to scalability.
- The data scaling strategy of reducing data waste through mixed conditions proves more effective than simply enlarging the dataset.
- Supporting arbitrary aspect ratios and body proportions represents an important milestone in the field of human animation.

## Limitations & Future Work

- Weak correlation between audio and motion still leads to incoherent or over-exaggerated movements in some cases.
- Object interactions are sometimes unrealistic due to insufficient representation of such samples in the training data.
- A relatively high CFG scale (6.5) is required to maintain synthesis stability, potentially introducing a degree of overfitting.
- Future work may incorporate richer motion conditions (style, intensity, intent) to improve naturalness.

## Related Work & Insights

- The omni-conditions training strategy is generalizable to other multimodal generation tasks (e.g., music-driven dance generation).
- The principle of "strong condition, low proportion" provides a reference for other multi-task learning scenarios.
- The elegant design of distinguishing reference image tokens from video tokens via the temporal component of 3D RoPE can be applied to other generative models that require conditioned images.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The two principles of the omni-conditions training strategy are systematic and insightful; this is the first work to achieve human animation with arbitrary body proportions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-task comparisons; complete ablation studies validating both principles; rich visual demonstrations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is well-articulated and the framework is described clearly.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a paradigmatic approach to scaling human animation; of very high industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICCV 2025\] VACE: All-in-One Video Creation and Editing](vace_all-in-one_video_creation_and_editing.md)
- [\[CVPR 2026\] Vanast: Virtual Try-On with Human Image Animation via Synthetic Triplet Supervision](../../CVPR2026/video_generation/vanast_virtual_try-on_with_human_image_animation_via_synthetic_triplet_supervisi.md)
- [\[NeurIPS 2025\] Scaling RL to Long Videos](../../NeurIPS2025/video_generation/scaling_rl_to_long_videos.md)

</div>

<!-- RELATED:END -->
