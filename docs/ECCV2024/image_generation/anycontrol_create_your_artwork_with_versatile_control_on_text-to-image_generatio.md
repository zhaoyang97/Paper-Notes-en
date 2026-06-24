---
title: >-
  [Paper Note] AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation
description: >-
  [ECCV 2024][Image Generation][Multi-Control] Proposes AnyControl, which supports arbitrary combinations of multiple spatial control signals (depth, edge, segmentation, pose) via a Multi-Control Encoder featuring an alternating fusion and alignment block structure, outperforming existing methods on the COCO multi-control benchmark with an FID of 44.28.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Multi-Control"
  - "text-to-image"
  - "Controllable Generation"
  - "Spatial Conditions"
  - "ControlNet"
date: 2026-05-08
content_hash: 0b493051d43f5fad
---

# AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation

**Conference**: ECCV 2024  
**arXiv**: [2406.18958](https://arxiv.org/abs/2406.18958)  
**Code**: [https://any-control.github.io](https://any-control.github.io)  
**Area**: Image Generation / Controllable Generation  
**Keywords**: Multi-Control, text-to-image, Controllable Generation, Spatial Conditions, ControlNet

## TL;DR
Proposes AnyControl, which supports arbitrary combinations of multiple spatial control signals (depth, edge, segmentation, pose) via a Multi-Control Encoder featuring an alternating fusion and alignment block structure, outperforming existing methods on the COCO multi-control benchmark with an FID of 44.28.

## Background & Motivation
**Background**: Methods like ControlNet have achieved single-condition controllable text-to-image (T2I) generation. However, practical applications often require multiple control signals simultaneously (e.g., depth map + segmentation map + pose), where existing multi-control methods (Multi-ControlNet, Uni-ControlNet) exhibit significant limitations.

**Limitations of Prior Work**:
   - **Insufficient input flexibility**: Most methods cannot freely combine arbitrary types or quantities of control signals.
   - **Poor spatial compatibility**: Simple weighted summation of multiple conditions leads to blending artifacts in occluded regions.
   - **Poor text compatibility**: Spatial conditions can be overly dominant, suppressing text semantics (e.g., specifying "Transformers Robot" but the pose condition forces the generation of a standard human figure).

**Key Challenge**: The need to handle multiple heterogeneous spatial conditions simultaneously while aligning them with text, whereas simple concatenation or weighted summation cannot model the complex relationships between conditions (such as occlusion or conflicts).

**Goal**: (a) Support arbitrary combinations of multi-control inputs; (b) model the compatibility among spatial conditions; (c) preserve the fidelity of textual semantics.

**Key Insight**: Using learnable query tokens to extract compatible features from multiple spatial conditions via cross-attention, and subsequently aligning them with text tokens through self-attention.

**Core Idea**: The Multi-Control Encoder leverages query tokens as an information hub—first fusing multiple spatial conditions via cross-attention (fusion), and then aligning them with text tokens via self-attention (alignment), stacked alternately.

## Method

### Overall Architecture
Based on a frozen SD1.5, a trainable duplicate of the UNet encoder and a Multi-Control Encoder are added. The Multi-Control Encoder processes three types of tokens: text tokens $\mathcal{T}$ (CLIP text), visual tokens $\mathcal{V}$ (CLIP image, multi-level), and query tokens $\mathcal{Q}$ (256 learnable parameters).

### Key Designs

1. **Multi-Control Fusion Block**:

    - **Function**: Extract compatible information from all spatial conditions into query tokens.
    - **Mechanism**: Use query tokens as Q, and the concatenated visual tokens of all conditions as KV: $\mathcal{Q}_j = CrossAttn(\mathcal{Q}_j, [\mathcal{V}_{1,j}+P, \mathcal{V}_{2,j}+P, ..., \mathcal{V}_{n,j}+P])$.
    - Positional embedding P is shared across conditions, allowing the model to learn spatial alignment.
    - **Design Motivation**: Cross-attention naturally accommodates a variable number of KV inputs (eliminating the need for fixed channel counts) and automatically handles occlusions/conflicts through attention weights.

2. **Multi-Control Alignment Block**:

    - **Function**: Align the spatial information in query tokens with the semantic information of text tokens.
    - **Mechanism**: $[\mathcal{Q}_{j+1}, \mathcal{T}_{j+1}] = SelfAttn([\mathcal{Q}_j, \mathcal{T}_j])$.
    - An additional textual task prompt (e.g., "depth + segmentation") is appended to the user prompt to bridge modality gaps.
    - **Design Motivation**: Prevent spatial conditions from suppressing text semantics—allowing both to negotiate via self-attention to ensure text descriptions are respected.

3. **Multi-Level Visual Tokens**:

    - **Function**: Extract visual features from multiple layers of CLIP to provide multi-granularity control.
    - Ablation studies indicate that 4 layers are optimal (FID of 43.67); using too many layers slightly degrades performance.

4. **Unaligned Training Data**:

    - **Function**: Synthesize training data where the foreground and background are unaligned, bridging the gap between training (perfectly aligned) and testing (arbitrarily combined).
    - Reduces FID from 52.10 to 44.28, representing a substantial gain.

### Loss & Training
- MultiGen dataset: 2.8M aligned image pairs + 0.44M synthesized unaligned images.
- Training setup: 8×A100, batch size of 8, 90K iterations, learning rate of 1e-5.
- During training, either 2 conditions are randomly sampled (from aligned data) or 1 foreground + 1 background is sampled (from unaligned data).
- Inference uses DDIM with 50 steps and a CFG scale of 7.5.

## Key Experimental Results

### Multi-Control Evaluation (COCO-UM Benchmark)

| Method | FID↓ | CLIP↑ | Depth RMSE↓ | Seg mPA↑ | Pose mAP↑ |
|------|------|-------|-------------|----------|-----------|
| Multi-ControlNet | 55.95 | 24.80 | 17.81 | 42.78 | 15.69 |
| Uni-ControlNet | 55.28 | 24.48 | 20.57 | 41.10 | 18.40 |
| Cocktail | 47.39 | 25.33 | - | 31.74 | 12.16 |
| **AnyControl** | **44.28** | **26.41** | 18.00 | **43.34** | **18.81** |

### Single-Control Evaluation (COCO-5K)

| Method | Depth FID | Seg FID | Edge FID | Pose FID |
|------|-----------|---------|----------|----------|
| ControlNet | 19.80 | 20.39 | 16.16 | 26.15 |
| **AnyControl** | **18.04** | **18.89** | 18.89 | **24.12** |

→ Even under single control, it outperforms specialized ControlNet.

### Ablation Study

| Configuration | FID↓ | CLIP↑ | Description |
|------|------|-------|------|
| Without unaligned data | 52.10 | 25.62 | Large train-test gap |
| **With unaligned data** | **44.28** | **26.40** | -7.8 FID |
| 1 level tokens | 45.64 | 26.35 | Single-layer features |
| **4 level tokens** | **43.67** | **26.39** | Optimal number of layers |

### Key Findings
- **Highest CLIP score (26.41)**: Indicates that the alignment block effectively preserves textual semantics, which is a major weakness in other methods.
- **Significant lead in FID (44.28 vs 47.39 Cocktail)**: Delivers the best overall image quality.
- **Superior single control as well**: Instead of compromising single-control performance for multi-control capability, the unified framework actually performs better.
- **Unaligned data is key**: Without this design, FID degrades by 7.8—since multiple control inputs in real-world scenarios are almost never perfectly aligned.

## Highlights & Insights
- **Query tokens as information hubs**: Unlike straightforward concatenation or summation, utilizing learnable queries in cross-attention naturally supports a variable number of inputs. While inspired by Q-Former/Perceiver, applying this concept to control signal fusion is highly elegant.
- **Alternating Fusion + Alignment blocks**: Separating the spatial condition fusion from textual alignment ensures neither objective is compromised.
- **Training strategy with unaligned data**: Synthetically bridging the discrepancy between training (perfectly aligned) and inference (arbitrarily combined) scenarios proves highly practical and effective.

## Limitations & Future Work
- **Quality degradation under excessive conditions**: Semantic confusion arises when handling 8+ conditions, primarily because the CLIP text encoder has limited understanding of highly complex prompts, and softmax precision degrades when there are too many KV tokens.
- **Reliance on SD1.5**: Validation has not been conducted on newer base models such as SDXL or SD3.
- **Limitations of the CLIP visual encoder**: Leveraging CLIP to extract visual features of control signals may lead to the loss of fine-grained spatial information.
- **Training data coverage**: Synthesized unaligned data might not fully reflect the complex inconsistencies encountered in real-world condition combinations.

## Related Work & Insights
- **vs. Multi-ControlNet**: Multi-ControlNet employs multiple independent ControlNets and applies a weighted sum, failing to model the relationships between conditions. AnyControl enables conditions to "negotiate" via attention mechanisms.
- **vs. Uni-ControlNet**: Uni-ControlNet utilizes MoE to process multiple conditions, which is constrained by a fixed number of channels. AnyControl naturally supports variable inputs via attention.
- **vs. Cocktail**: Cocktail relies on optimization (test-time fine-tuning), making it slow and unstable. AnyControl is a feed-forward model with inference speeds comparable to standard Stable Diffusion.

## Rating
- Novelty: ⭐⭐⭐⭐ The query-based multi-control fusion design is pioneering, and the unaligned training data strategy is highly practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering multi-control, single-control, ablations, user studies, and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear mapping from three challenges to three solutions, accompanied by well-structured method figures.
- Value: ⭐⭐⭐⭐ Directly valuable for controllable image generation, offering an exceptionally practical unified framework design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] Collaborative Control for Geometry-Conditioned PBR Image Generation](collaborative_control_for_geometry-conditioned_pbr_image_generation.md)
- [\[ECCV 2024\] Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](be_yourself_bounded_attention_for_multisubject_texttoimage_g.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](../../CVPR2025/image_generation/precisecam_precise_camera_control_for_text-to-image_generation.md)
- [\[ECCV 2024\] Generating Human Interaction Motions in Scenes with Text Control](generating_human_interaction_motions_in_scenes_with_text_control.md)

</div>

<!-- RELATED:END -->
