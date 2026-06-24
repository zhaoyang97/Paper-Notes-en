---
title: >-
  [Paper Note] Rotary Position Embedding for Vision Transformer
description: >-
  [ECCV 2024][Segmentation][RoPE] This work systematically investigates the application of RoPE (Rotary Position Embedding) from 1D language models to 2D vision tasks. It proposes RoPE-Mixed (mixed learnable frequencies) to replace the conventional Axial frequency allocation. This approach achieves significant resolution extrapolation performance gains on ViT and Swin Transformer, yielding consistent improvements in ImageNet classification, COCO detection…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "RoPE"
  - "Position Encoding"
  - "Vision Transformer"
  - "Resolution Extrapolation"
  - "2D Rotary Position Embedding"
date: 2026-05-08
content_hash: fba6c7a407095e8e
---

# Rotary Position Embedding for Vision Transformer

**Conference**: ECCV 2024  
**arXiv**: [2403.13298](https://arxiv.org/abs/2403.13298)  
**Authors**: Byeongho Heo, Song Park, Dongyoon Han, Sangdoo Yun (NAVER AI Lab)  
**Code**: [https://github.com/naver-ai/rope-vit](https://github.com/naver-ai/rope-vit)  
**Area**: Image Segmentation  
**Keywords**: RoPE, Position Encoding, Vision Transformer, Resolution Extrapolation, 2D Rotary Position Embedding  

## TL;DR
This work systematically investigates the application of RoPE (Rotary Position Embedding) from 1D language models to 2D vision tasks. It proposes RoPE-Mixed (mixed learnable frequencies) to replace the conventional Axial frequency allocation. This approach achieves significant resolution extrapolation performance gains on ViT and Swin Transformer, yielding consistent improvements in ImageNet classification, COCO detection, and ADE20k segmentation.

## Background & Motivation

**Background**: There are two main positional encodings for ViTs: APE (Absolute Positional Embedding, added to the stem layer) used in standard ViTs, and RPB (Relative Position Bias, added to the attention matrix) used in hierarchical ViTs like Swin Transformers. In the NLP domain, RoPE has become the standard positional encoding for LLMs (such as LLaMA and Mistral), showing outstanding performance particularly in long-sequence extrapolation.

**Limitations of Prior Work**: While APE and RPB perform well under fixed training resolutions, they adapt poorly to resolution changes. APE relies on learnable parameters of a fixed length, which require bilinear interpolation when changing resolutions, resulting in sub-optimal performance. RPB uses relative position tables with fixed dimensions that can only fall back on zero-padding when exceeding boundaries. However, vision tasks frequently require resolution changes (e.g., 224x224 for classification, 800+ for detection, and 512x512 for segmentation). Therefore, the extrapolation ability of positional encodings directly impacts downstream performance.

**Key Challenge**: Although RoPE has demonstrated excellent length extrapolation capabilities in LLMs, extending it from 1D text to 2D images is not a trivial task. Previous studies (EVA-02, Unified-IO 2, FiT) utilize 2D Axial RoPE, which applies independent frequencies only along the x/y axes respectively, failing to handle diagonal spatial relationships—whereas convolutional networks naturally handle diagonal relationships using square kernels.

**Goal**: How can RoPE be better designed for 2D images to enable both resolution extrapolation and full utilization of 2D spatial structures?

**Key Insight**: From a Fourier analysis perspective, Axial frequencies in 2D space can only represent frequency components along the axes, introducing axial artifacts during image reconstruction. This work proposes Mixed frequencies that allow each frequency channel to simultaneously use frequency parameters of both x and y axes, optimized end-to-end as learnable parameters.

**Core Idea**: To extend RoPE to 2D vision tasks using mixed-axis learnable frequencies (RoPE-Mixed), enabling each frequency channel to represent spatial relationships in any direction (including diagonals), substantially boosting resolution extrapolation and downstream task performance.

## Method

### Overall Architecture
RoPE is applied to the query and key vectors of self-attention, injecting positional information through the Hadamard product. Unlike APE (which is added to input tokens) and RPB (added to the attention matrix), RoPE modulates query-key similarity calculations directly through complex rotation, incorporating relative positions into attention weights as rotation angles:

$$\mathbf{A}'_{(n,m)} = \mathrm{Re}[\mathbf{q}_n \mathbf{k}_m^* e^{i(n-m)\theta}]$$

This enables a multiplicative interaction between relative positional information and content features (query-key), rather than the additive bias used in RPB, which is theoretically more expressive.

### Key Designs

1. **2D Axial RoPE (Baseline)**:

    - Function: Replaces the 1D RoPE token index with 2D coordinates $(p_n^x, p_n^y)$, splitting the head dimension in half to encode the x-axis in even channels and the y-axis in odd channels.
    - Core Equation: $\mathbf{R}(n, 2t) = e^{i\theta_t p_n^x}$, $\mathbf{R}(n, 2t+1) = e^{i\theta_t p_n^y}$
    - The frequency base is reduced from $10000$ to $100$ ($\sqrt{10000}$) because the index range of 2D images is shorter than that of 1D sequences.
    - Limitations: Each frequency channel only considers a single axis, failing to represent diagonal spatial relationships.

2. **RoPE-Mixed (Ours / Core Contribution)**:

    - Function: Allows each frequency channel to simultaneously use frequency parameters from both x and y axes, representing any 2D direction.
    - Core Equation: $\mathbf{R}(n, t) = e^{i(\theta_t^x p_n^x + \theta_t^y p_n^y)}$
    - The attention matrix becomes: $\mathbf{A}'_{(n,m)} = \mathrm{Re}[\mathbf{q}_n \mathbf{k}_m^* e^{i(\theta_t^x(p_n^x - p_m^x) + \theta_t^y(p_n^y - p_m^y))}]$
    - $(\theta_t^x, \theta_t^y)$ are optimized end-to-end as **learnable parameters**, with independent sets of frequency parameters for each head and layer.
    - RoPE-Axial is a special case of RoPE-Mixed where $\theta_t^y = 0$ or $\theta_t^x = 0$.
    - Design Motivation: 2D Fourier analysis reveals that Axial frequencies can only cover axis-aligned frequency components, causing cross-shaped artifacts during image reconstruction. Mixed frequencies can cover diverse directions in the 2D frequency domain, producing sharper reconstructions. Furthermore, learnable frequencies allow the network to autonomously decide the optimal directional allocation.
    - Additional Parameters: $d$ parameters per layer (accounting for approximately 0.01% of the total parameters in ViT-B), which is virtually negligible.

3. **Combination with Traditional Positional Encodings**:

    - RoPE can be used jointly with APE or RPB (RoPE+APE / RoPE+RPB).
    - Empirical Finding: RoPE+APE exhibits advantages in the interpolation range (resolution < training resolution) but reduces extrapolation gains. RoPE+RPB yields almost no additional gains over Mixed.
    - Conclusion: For tasks requiring extrapolation, using RoPE-Mixed alone is sufficient; for fixed-resolution or interpolation-heavy tasks, RoPE-Mixed+APE performs better.

### Loss & Training
- No special training strategy is required; the standard training recipes are used directly (DeiT-III for 400 epochs on ViT, Swin for 300 epochs).
- RoPE can simply replace or be appended to existing positional encodings, without requiring multi-resolution training, self-distillation, or other additional techniques.
- This is the core advantage over other multi-resolution methods like ResFormer and CAPE—highly versatile and plug-and-play.

### Analysis & Insights
- **Attention Distance Analysis**: RoPE increases attention distance and entropy in middle layers, allowing attention to interact with more distant and diverse tokens, which is beneficial for capturing global context.
- **Phase Shift Requires No Explicit Modeling**: The query/key projection matrices $\mathbf{W}_q$ and $\mathbf{W}_k$ of RoPE can already implicitly learn phase shifts $\phi$.
- **Computational Overhead**: The rotation matrices are precomputed, requiring only Hadamard products during inference, which adds a mere 0.01% FLOPs for ViT-B.

## Key Experimental Results

### Main Results

**ImageNet-1k Multi-resolution Classification (ViT-B, DeiT-III 400ep)**:

| Resolution | APE | RoPE-Axial | RoPE-Mixed | Change |
|-----------|-----|-----------|-----------|------|
| 224 (Training) | 83.5 (Baseline) | ≈83.5 | ≈83.5 | Comparable |
| 384 (Extrap.) | Significant drop | Significantly better than APE | **Best** | Significant gain |
| 512 (Large Extrap.) | Severe drop | Better than APE | **Best** | Larger gain |

**COCO Detection (DINO-ViTDet)**:

| Backbone| APE | RoPE-Axial | RoPE-Mixed | Gain |
|----------|-----|-----------|-----------|------|
| ViT-B | 49.4 | 50.8(+1.4) | **51.2(+1.8)** | +1.8 AP |
| ViT-L | 51.1 | 52.2(+1.1) | **52.9(+1.8)** | +1.8 AP |

**ADE20k Semantic Segmentation (UperNet-ViT)**:

| Backbone | APE | RoPE-Mixed | Mixed+APE | Gain |
|----------|-----|-----------|-----------|------|
| ViT-B (single) | 47.7 | 49.6(+1.9) | **50.0(+2.3)** | +2.3 mIoU |
| ViT-B (multi) | 48.4 | 50.7(+2.3) | **50.9(+2.5)** | +2.5 mIoU |
| ViT-L (single) | 50.8 | 51.5(+0.7) | **52.0(+1.2)** | +1.2 mIoU |

**Performance on Swin Transformer**:
- COCO Detection: Swin-B +0.3 AP (RoPE-Mixed vs RPB)
- ADE20k Segmentation: Swin-S +0.9 mIoU (RoPE-Mixed vs RPB)
- RoPE-Mixed effectively replaces RPB on Swin, with substantially better extrapolation performance.

### Ablation Study

| Positional Encoding Config | 224 Classification | 384 Classification| Detection AP| Segmentation mIoU | Notes |
|-------------|---------|---------|--------|----------|------|
| APE (baseline) | Baseline | Significant drop | 49.4 | 47.7 | Poor extrapolation |
| RoPE-Axial | ≈Baseline | Improved | 50.8 | 49.0 | Missing diagonal direction |
| **RoPE-Mixed** | ≈Baseline | Better | **51.2** | **49.6** | Best |
| RoPE-Mixed+APE | Slight improvement | Slightly lower | 51.1 | **50.0** | Best for segmentation |

### Key Findings
- **RoPE-Mixed outperforms RoPE-Axial comprehensively**: The diagonal handling capability enabled by mixed frequencies yields consistent improvements across all tasks.
- **Largest gains in extrapolation scenarios**: The advantage of RoPE is most pronounced in resolution extrapolation from 224 to 384/512, which aligns with RoPE's length extrapolation behavior in LLMs.
- **Most significant gain in detection (+1.8 AP)**: Because object detection processes inputs with much larger resolutions (800+) than the training phase, the demand for extrapolation is the strongest.
- **Adding APE helps in segmentation**: For UperNet with 512x512 input, RoPE-Mixed+APE yields the best segmentation performance.
- **RoPE-Mixed can completely replace RPB**: Replacing RPB with RoPE-Mixed on Swin Transformer delivers superior performance, with virtually no extra benefit from combining the two (+RPB).
- **vs ResFormer**: RoPE-Mixed+APE outclasses the specially designed ResFormer (which requires multi-resolution training and self-distillation) across multi-resolution inference, while RoPE requires no special training strategies.

## Highlights & Insights
- **Successful transfer from language to vision**: The core edge of RoPE—extrapolation backed by periodic functions—translates seamlessly from 1D text to 2D images, demonstrating the cross-modal versatility of this design philosophy.
- **Highly intuitive mixed-frequency design**: Utilizing 2D Fourier analysis to contrast Axial vs Mixed frequency coverage provides a very intuitive and compelling motivation, explaining why CNNs process diagonal directions via square kernels.
- **Extreme plug-and-play capability**: Requiring no changes to training recipes, multi-resolution training, or distillation, it delivers a massive +1.8 AP on detection and +2.3 mIoU on segmentation simply by swapping the positional encodings, highlighting its immense practical value.
- **Flexibility of learnable frequencies**: Treating frequencies as learnable parameters rather than fixed constants allows the network to learn diverse, adaptive spatial attention patterns across different heads and layers.

## Limitations & Future Work
- **Comparable performance at training resolution**: RoPE's advantage shines primarily in extrapolation; its improvement over APE at the 224x224 training resolution is minimal.
- **Inferior interpolation performance compared to APE**: RoPE falls behind APE when the test resolution is smaller than the training resolution (requiring +APE for compensation), revealing limitations of RoPE in downscaling scenarios.
- **Lack of visual interpretation of RoPE frequencies**: Although 2D Fourier analysis is provided, there is no in-depth analysis of what spatial patterns are represented by the learned frequencies.
- **Evaluation limited to classification, detection, and segmentation**: The performance on other vision tasks, such as generative tasks (diffusion models) or video understanding, remains unexplored.
- **Future directions**: Incorporating physical/frequency scaling techniques (such as YaRN, NTK-aware scaling) to tackle more extreme resolution extrapolation.

## Related Work & Insights
- **vs APE (ViT)**: APE relies on absolute locations and depends on interpolation for extrapolation, suffering severe performance drops during resolution shifts. In contrast, RoPE is based on periodic functions, intrinsically supporting extrapolation.
- **vs RPB (Swin)**: RPB uses a relative position look-up table of fixed size that cannot extrapolate (falling back on zero-padding). RoPE's rotation angles scale naturally to arbitrary relative distances.
- **vs ResFormer**: ResFormer demands expensive and less versatile multi-resolution training paired with self-distillation. RoPE achieves superior multi-resolution performance without modifying the standard training recipe.
- **vs CPE (Conditional PE)**: CPE utilizes depth-wise convolutions to inject positional information, supporting resolution variations. RoPE is orthogonal to CPE and can be combined with it.
- Offers systematic guidance for subsequent works like InternViT and EVA-02 adopting RoPE in large-scale VLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ The mixed learnable frequency design of RoPE-Mixed is creative, backed by clear motivation from 2D Fourier analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, spanning both ViT and Swin architectures across classification, detection, and segmentation, alongside diverse combinations and configurations.
- Writing Quality: ⭐⭐⭐⭐ Clear and logically sound narrative flow, leading seamlessly from positional encoding fundamentals to RoPE's extension and experimental validation.
- Value: ⭐⭐⭐⭐⭐ An exceptional plug-and-play positional encoding improvement that yields solid practical gains, highly beneficial for all future works building upon ViT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GiT: Towards Generalist Vision Transformer through Universal Language Interface](git_towards_generalist_vision_transformer_through_universal_language_interface.md)
- [\[ICLR 2026\] Locality-Attending Vision Transformer](../../ICLR2026/segmentation/locality-attending_vision_transformer.md)
- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](../../CVPR2025/segmentation/mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](../../CVPR2025/segmentation/revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[ECCV 2024\] SiLC: Improving Vision Language Pretraining with Self-Distillation](silc_improving_vision_language_pretraining_with_self-distillation.md)

</div>

<!-- RELATED:END -->
