---
title: >-
  [Paper Note] Multi-subject Open-set Personalization in Video Generation
description: >-
  [CVPR 2025][Video Generation][Personalized video generation] Video Alchemist is proposed, which integrates multi-subject, open-set video personalization capabilities directly into the Diffusion Transformer architecture, supporting foreground object and background customization without requiring test-time optimization.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Personalized video generation"
  - "Multi-Subject Customization"
  - "Open-Set Entities"
  - "Diffusion Transformer"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 16eb9fdeaf0eaec0
---

# Multi-subject Open-set Personalization in Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2501.06187](https://arxiv.org/abs/2501.06187)  
**Code**: [https://github.com/snap-research/MSRVTT-Personalization](https://github.com/snap-research/MSRVTT-Personalization) (Yes, benchmark code)  
**Area**: Video Generation  
**Keywords**: Personalized video generation, Multi-Subject Customization, Open-Set Entities, Diffusion Transformer, Data Augmentation

## TL;DR

Video Alchemist is proposed, which integrates multi-subject, open-set video personalization capabilities directly into the Diffusion Transformer architecture, supporting foreground object and background customization without requiring test-time optimization.

## Background & Motivation

Personalized video generation aims to synthesize videos containing specific people, pets, or scenes, but existing methods have significant limitations:

1. **Limited domains**: Many methods only support faces (e.g., Magic-Me) or a single subject (e.g., DreamVideo, VideoBooth), failing to handle multi-subject and open-set categories.
2. **High test-time optimization costs**: Methods like DreamVideo require fine-tuning for each new concept, which is time-consuming and prone to overfitting.
3. **Inability to customize foreground and background simultaneously**: Most methods focus solely on foreground objects and cannot customize the video background.
4. **Copy-and-paste issue**: When extracting reference and target frames from the same video for training, models tend to directly duplicate irrelevant information like illumination, pose, and occlusion from the reference image, rather than learning the identity features.

**Key Challenge**: How to construct training data and design a model architecture that can support multi-subject, open-set, background-inclusive video personalization without requiring fine-tuning?

## Method

### Overall Architecture

Video Alchemist is constructed based on a latent Diffusion Transformer (DiT), with inputs consisting of text prompts and multiple reference images (one or more per entity). The core innovation lies in adding a dedicated cross-attention layer within the DiT block to process personalization embeddings, enabling the binding and fusion of image and text concepts.

### Key Designs

1. **Binding of Image and Word Concepts**: For each reference entity, a frozen DINOv2 encoder is used to extract image tokens $x_n \in \mathbb{R}^{l \times d}$, and the corresponding entity word token $c_n$ is retrieved from the text embeddings. The entity word token is flattened, replicated $l$ times, and concatenated with the image token along the channel axis. This is then projected linearly with residual connections to obtain the personalization embedding $f_n$. The embeddings of all entities are concatenated as $f = \text{Concat}(f_1, ..., f_N)$, which interacts with the video tokens through an independent cross-attention layer. **Design Motivation**: Without a binding mechanism, the model tends to apply reference images to the wrong subjects (e.g., pasting a human face onto a dog).

2. **Automated Data Construction Pipeline**: A three-step pipeline: (a) LLMs extract entity words (subject/object/background) from captions; (b) GroundingDINO + SAM segment targets in the first, middle, and last frames of the video; (c) Erosion, dilation, and inpainting are applied to generate a clean background image. Frames from different timestamps are selected to capture variation in pose and illumination.

3. **Anti-overfitting Data Augmentation**: To address the copy-and-paste issue, various augmentations are applied to the reference images: downsampling & Gaussian blur (to prevent resolution overfitting), color jitter & brightness adjustment (to prevent illumination overfitting), and horizontal flip/shear/rotation (to prevent pose overfitting). This guides the model to focus on the subject's identity features rather than irrelevant attributes of the reference image.

### Loss & Training

- Use Rectified Flow formulation for denoising training
- **Two-stage training**: Stage one only trains the text cross-attention; stage two introduces personalized cross-attention and performs full-model fine-tuning (with warmup)
- DINOv2 (frozen) is used as the image encoder, which outperforms CLIP in subject similarity
- RoPE positional encoding, Flash Attention, and Fused LayerNorm are used for acceleration
- Self-conditioning technique is applied to enhance visual quality

## Key Experimental Results

### Main Results — MSRVTT-Personalization（Subject Mode, Single Reference Image）

| Method | Test-Time Opt. | Text-S ↑ | Vid-S ↑ | Subj-S ↑ | Dync-D ↑ |
|------|---------|----------|---------|----------|----------|
| ELITE | No | 0.245 | 0.620 | 0.359 | - |
| VideoBooth | No | 0.222 | 0.612 | 0.395 | 0.448 |
| DreamVideo | Yes | 0.261 | 0.611 | 0.310 | 0.311 |
| **Video Alchemist** | **No** | **0.269** | **0.732** | **0.617** | **0.466** |

### User Preference Study

| Method | Quality Pref. ↑ | Fidelity Pref. ↑ |
|------|-----------|-----------|
| ELITE | 2.7% | 0.6% |
| VideoBooth | 0.3% | 0.8% |
| DreamVideo | 0.5% | 0.5% |
| **Video Alchemist** | **96.5%** | **98.1%** |

### Ablation Study

| Configuration | Text-S ↑ | Vid-S ↑ | Subj-S ↑ | Dync-D ↑ | Description |
|------|----------|---------|----------|----------|------|
| CLIP encoder | 0.269 | 0.768 | 0.569 | 0.552 | Good text alignment |
| DINOv2 w/o word token | 0.256 | 0.790 | 0.566 | 0.569 | Missing concept binding |
| DINOv2 w/o augmentation | 0.251 | 0.781 | 0.609 | 0.506 | Severe copy-paste |
| **DINOv2 + word token + aug.** | **0.257** | **0.790** | **0.600** | **0.570** | Best balance |

### Key Findings

- Video Alchemist outperforms VideoBooth by **22.2%** in Subject similarity (0.395 $\rightarrow$ 0.617).
- Even as an open-set model, its Face similarity surpasses face-specific models like IP-Adapter (0.382 vs 0.269).
- In the user preference study, it achieves **96.5%** quality preference and **98.1%** fidelity preference.
- Multi-reference image input further improves fidelity (single vs multi Subj-S: 0.617 $\rightarrow$ 0.626).
- Background reference images make the video more similar to the ground truth (Vid-S: 0.743 $\rightarrow$ 0.780), though slightly reducing text alignment.
- DINOv2 is better suited than CLIP for capturing unique object features (reflecting the difference between self-supervised learning and text-image alignment objectives).

## Highlights & Insights

- **Elegant Architecture Design**: Personalization capabilities are integrated directly into the DiT block rather than relying on external adapters, enabling end-to-end training.
- **Concept Binding Mechanism is Key**: Without this mechanism, identity confusion occurs in multi-subject scenarios, a finding that offers valuable reference for future work.
- **Exquisite Data Augmentation**: Each augmentation directly addresses a specific overfitting pattern (resolution $\rightarrow$ object size, occlusion $\rightarrow$ generating occluders, etc.).
- **MSRVTT-Personalization Benchmark**: Comprises 2130 samples and supports multiple evaluation modes including face, single subject, multi-subject, and foreground+background.
- The open-set + tuning-free paradigm significantly outperforms methods requiring test-time optimization.

## Limitations & Future Work

- The current resolution is $512 \times 288$, which limits generation quality.
- Multiple reference images sometimes reduce text alignment, representing a trade-off between flexibility and fidelity.
- Background customization relies only on a single frame generated by inpainting, which may introduce artifacts.
- Extreme scenarios, such as heavily occluded or very low-resolution reference images, are not handled.
- Training requires large-scale captioned video datasets and a multi-step preprocessing pipeline.

## Related Work & Insights

- IP-Adapter: A pioneer in decoupling cross-attention, but mixing text and image tokens in a single attention layer yields suboptimal results.
- DreamBooth / Textual Inversion: Representatives of the optimization paradigm, suffering from heavy test-time overhead.
- SnapVideo / Sora: DiT-architecture large-scale video generation foundations.
- The concept binding and data augmentation strategies proposed in this paper can be extended to tasks like 3D generation and long video synthesis.
- The relatively annotated benchmark design (segment-level instead of image-level similarity) is a valuable reference.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First multi-subject + open-set + foreground/background + tuning-free video personalization model.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Proposes a brand new benchmark, and includes comprehensive quantitative, qualitative, user study, and ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, and the data construction pipeline diagram is highly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Drives significant progress in the field of personalized video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)
- [\[AAAI 2026\] MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation](../../AAAI2026/video_generation/mofu_scale-aware_modulation_and_fourier_fusion_for_multi-subject_video_generatio.md)
- [\[CVPR 2026\] ID-Crafter: VLM-Grounded Online RL for Compositional Multi-Subject Video Generation](../../CVPR2026/video_generation/id-crafter_vlm-grounded_online_rl_for_compositional_multi-subject_video_generati.md)
- [\[CVPR 2026\] Open-world Hand-Object Interaction Video Generation Based on Structure and Contact-aware Representation](../../CVPR2026/video_generation/open-world_hand-object_interaction_video_generation_based_on_structure_and_conta.md)

</div>

<!-- RELATED:END -->
