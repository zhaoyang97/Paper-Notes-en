---
title: >-
  [Paper Note] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking
description: >-
  [CVPR 2026][Image Generation][document watermarking] This paper proposes TRACE, a document watermarking framework based on character structure encoding. It leverages a diffusion model (DragDiffusion) to precisely displac…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "document watermarking"
  - "data hiding"
  - "diffusion model"
  - "character structure"
  - "cross-media robustness"
date: 2026-05-08
content_hash: af5c8bc9ae19650d
---

# TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking

**Conference**: CVPR 2026
**arXiv**: [2603.12873](https://arxiv.org/abs/2603.12873)  
**Code**: To be confirmed  
**Area**: Image Generation
**Keywords**: document watermarking, data hiding, diffusion model, character structure, cross-media robustness

## TL;DR

This paper proposes TRACE, a document watermarking framework based on character structure encoding. It leverages a diffusion model (DragDiffusion) to precisely displace skeleton keypoints of characters for information embedding. Through three core components—Adaptive Diffusion Initialization (ADI), Guided Diffusion Encoding (GDE), and Masked Region Replacement (MRR)—TRACE simultaneously achieves cross-media robustness, multi-language/multi-font generalizability, and high visual imperceptibility.

## Background & Motivation

**The trilemma of document watermarking**: Existing document steganography methods struggle to simultaneously satisfy robustness, generalizability, and imperceptibility:
   - **Image-based methods** (pixel flipping): embed data by adjusting black-white pixel ratios, but cross-media transmission (print–scan–photograph) introduces noise that severely corrupts pixel-level features.
   - **Font-based methods** (predefined codebooks): replace original characters with codebook-designed variants, offering good robustness but limited generalizability—codebooks cannot cover all possible characters and fail on handwritten or artistic fonts.
   - **Format-based methods** (line/word spacing): suffer from low embedding capacity and poor robustness.

**Advantages of character structure**: Character structure (skeleton + keypoints + strokes) offers three inherent advantages: (a) stability under noise—structural features largely survive cross-media transmission; (b) a unified representation across languages and fonts—skeletons can be extracted from any character; (c) modifying a small number of pixels near the structure does not alter visual appearance, ensuring imperceptibility.

**New capabilities from diffusion-based image editing**: Point-to-point image editing methods such as DragDiffusion provide precise local pixel manipulation, establishing a technical foundation for structure-based character watermarking.

## Method

### Overall Architecture

TRACE comprises two stages: **data embedding** and **data extraction**. The embedding stage consists of three steps: Adaptive Diffusion Initialization (ADI) → Guided Diffusion Encoding (GDE) → Masked Region Replacement (MRR).

### Key Designs 1: Adaptive Diffusion Initialization (ADI)

Given a text image $I_{\text{cover}}$, ADI determines three key elements that guide the diffusion process:

**Keypoint detection**: A lightweight OpenPose architecture extracts the endpoint set $E$ and crosspoint set $C$, producing a three-channel heatmap (endpoints / crosspoints / background).

**Movement Probability Evaluator (MPE)**: Automatically selects the optimal handle point $P_h$ and reference point $P_r$.
- Only endpoints are considered as candidates for $P_h$ (crosspoints connect multiple strokes and moving them would destroy structure).
- For each endpoint $p_i^e$, reference point candidates $R_i$ are sought within a $\tau$-neighborhood.
- Scoring rule: initial score of 1; +1 if $p_i^e$ and $p_{i,j}^r$ are not on the same stroke; when multiple candidates tie for top score, +1 for the smallest $y$-coordinate.
- The highest-scoring endpoint becomes $P_h$, and its corresponding reference point becomes $P_r$.

**Target Point Estimation (TPE)**: Determines the target point $P_t$ based on the bit value to be embedded.

The directional axis $\lambda$ is defined as:

$$\lambda\text{-axis} = \begin{cases} X\text{-axis}, & d_x \leq d_y \\ Y\text{-axis}, & d_x > d_y \end{cases}$$

where $d_x = |x_h - x_r|,\ d_y = |y_h - y_r|$, and $\Delta(P_h, P_r) = \min\{d_x, d_y\}$.

Embedding rule:
- Bit 0: if $\Delta(P_h, P_r) > T_{\text{embed}}$, move $P_h$ such that $\Delta(P_t, P_r) \leq T_{\text{embed}}$.
- Bit 1: if $\Delta(P_h, P_r) \leq T_{\text{embed}}$, move $P_h$ such that $\Delta(P_t, P_r) > T_{\text{embed}}$.

The displacement direction is jointly determined by the stroke direction vector $\vec{\mathcal{V}}$ and the direction vector from $P_h$ to $P_r$, denoted $\vec{\mathcal{H}}$:

$$x_t = x_h + \mathcal{D} \times \frac{\mathcal{V}_x}{\|\vec{\mathcal{V}}\|} \times \text{sgn}(\mathcal{H}_x)$$

**Mask Drawing Module (MDM)**: Constructs a minimal rectangular editing mask $\mathcal{M}$ based on $P_h$ and $P_t$, with boundary expansion $\sigma$ to ensure diffusion quality.

### Key Designs 2: Guided Diffusion Encoding (GDE)

DragDiffusion is used to displace $P_h$ to $P_t$ via the following steps:
1. LoRA fine-tuning of the UNet to capture features of the original image.
2. DDIM inversion to generate the initial diffusion latent.
3. Iterative optimization via motion supervision and point tracking until the handle point aligns with the target point.
4. Reference latent control: keys and values from the initial latent replace those of the editing latent in self-attention, preserving consistency.

A **local consistency loss** $L_{lc}$ is introduced to ensure feature coherence within the masked region before and after editing:

$$L_{lc}(\hat{z}_t^k) = \sum_{q \in \Omega} \|G_{q+d}(\hat{z}_{t-1}^k) - \text{sg}(G_q(\hat{z}_{t-1}^0))\|_1$$

Total loss: $L(\hat{z}_t^k) = L_{ms}(\hat{z}_t^k) + \eta L_{lc}(\hat{z}_t^k)$, where $\eta = 0.003$.

### Key Designs 3: Masked Region Replacement (MRR)

The masked region content from the diffusion-edited image is composited back into the corresponding area of the original image, so that data is embedded only within the target region, minimizing impact on the rest of the image.

### Data Extraction

1. Individual characters are segmented using the CRAFT algorithm.
2. MPE is applied to each character to identify $P_h$ and $P_r$.
3. $\Delta(P_h, P_r)'$ is computed: if $> T_{\text{embed}}$, bit 1 is extracted; otherwise, bit 0.

## Key Experimental Results

### Screenshot Robustness (ACC, %)

| Font | Method | 12pt | 16pt | 20pt | 24pt | 28pt | 36pt |
|------|--------|------|------|------|------|------|------|
| Arial | ASF | 85.83 | 91.67 | 90.00 | 87.50 | 88.33 | 82.50 |
| Arial | **TRACE** | **96.67** | **97.50** | **99.17** | **100** | **100** | **100** |
| Calibri | ASF | 95.00 | 95.83 | 96.67 | 98.33 | 97.50 | 100 |
| Calibri | **TRACE** | **97.50** | **99.17** | **99.17** | **100** | **100** | **100** |

### Print–Scan Robustness

| Font | Method | 12pt | 16pt | 20pt | 24pt | 28pt | 36pt |
|------|--------|------|------|------|------|------|------|
| Arial | ASF | 80.83 | 68.33 | 70.83 | 72.50 | 70.00 | 76.67 |
| Arial | **TRACE** | **95.83** | **97.50** | **99.17** | **99.17** | **100** | **100** |
| TNR | ASF | 64.17 | 72.50 | 79.17 | 68.33 | 79.17 | 72.50 |
| TNR | **TRACE** | **92.50** | **94.17** | **95.83** | **97.50** | **99.17** | **99.17** |

### Imperceptibility Comparison

| Metric | StegaStamp | IHA | **TRACE** |
|--------|-----------|-----|-----------|
| Screenshot ACC | 100 | 84.58 | **100** |
| Print–Scan ACC | 98.54 | 84.29 | **99.05** |
| Photo ACC | 98.12 | 83.94 | **98.75** |
| PSNR↑ | 27.19 | 29.60 | **33.34** |
| SSIM↑ | 0.8986 | 0.9910 | **0.9962** |

TRACE achieves state-of-the-art performance in both robustness and imperceptibility, with PSNR exceeding StegaStamp by over 6 dB.

### Generalizability

- Handwritten fonts: screenshot 94.43%, print–scan 93.17%, photo 91.67%, PSNR 38.20.
- Artistic fonts: screenshot 97.27%, print–scan 94.77%, photo 92.93%, PSNR 41.37.
- Successfully extended to multiple languages including Chinese and Japanese, as well as mathematical expressions.

### Ablation Study

| Setting | MPE | TPE | ACC |
|---------|-----|-----|-----|
| Setting 1 | ✗ | ✗ | 49.95% |
| Setting 2 | ✓ | ✗ | 68.75% |
| Setting 3 | ✗ | ✓ | 53.50% |
| **Setting 4 (Ours)** | **✓** | **✓** | **100%** |

- MPE and TPE must work jointly to achieve error-free extraction.
- The $L_{lc}$ loss significantly improves shape preservation within the masked region.
- MRR consistently improves PSNR/SSIM across different fonts.

## Highlights & Insights

**Strengths**:
- The first work to introduce character structure encoding into document watermarking, establishing a fundamentally new paradigm.
- Simultaneously addresses the robustness–generalizability–imperceptibility trilemma, surpassing prior state-of-the-art on all dimensions.
- The ADI design (MPE + TPE + MDM) is elegantly automated, ensuring synchronized encoding and decoding.
- Supports a hybrid mode combining precomputed codebooks (for common characters) and dynamic generation (for unseen characters).
- Maintains over 96% extraction accuracy under structural distortion attacks.

**Limitations**:
- Only 1 bit per character is embedded, resulting in relatively low embedding capacity.
- The DragDiffusion-based encoding pipeline requires LoRA fine-tuning and DDIM inversion, incurring considerable computational cost.
- Characters with very few strokes (e.g., "一", "I") offer limited available keypoints.

## Rating

⭐⭐⭐⭐

This is a conceptually elegant piece of work—using the inherent stability of character skeleton structure as the watermark carrier, which aligns perfectly with the core requirements of document watermarking. The automated scoring mechanism in MPE is carefully designed, and the stroke-direction-based displacement strategy in TPE ensures synchronized encoding and decoding. Experiments cover English and Chinese × multiple fonts × multiple sizes × multiple transmission channels (screenshot / print–scan / photograph), providing thorough validation. TRACE achieves comprehensive superiority in the three-dimensional space of robustness, generalizability, and imperceptibility where prior methods have struggled to excel simultaneously. The 1-bit-per-character capacity constraint and the computational overhead of diffusion-based encoding are the primary bottlenecks; however, as a paradigm-establishing first work, the contribution is well deserved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation](../../ICML2026/image_generation/semantic-aware_motion_encoding_for_topology-agnostic_character_animation.md)
- [\[CVPR 2026\] Towards Robust Content Watermarking Against Removal and Forgery Attacks](towards_robust_content_watermarking_against_removal_and_forgery_attacks.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[CVPR 2026\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)

</div>

<!-- RELATED:END -->
