---
title: >-
  [Paper Note] VP-SAM: Taming Segment Anything Model for Video Polyp Segmentation via Disentanglement and Spatio-Temporal Side Network
description: >-
  [ECCV 2024][Segmentation][Video Polyp Segmentation] This paper proposes VP-SAM, which leverages the amplitude information of the Fourier spectrum via a Semantic Disentanglement Adapter (SDA) to help SAM distinguish low-contrast polyps from the background, while designing a Spatio-Temporal Side Network (STSN) to inject inter-frame temporal information into SAM. It achieves SOTA performance on datasets such as SUN-SEG, CVC-612, and CVC-300.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Video Polyp Segmentation"
  - "SAM Adaptation"
  - "Semantic Disentanglement"
  - "Spatio-Temporal Modeling"
  - "Frequency Domain Analysis"
date: 2026-05-08
content_hash: 52f244102fd5ee18
---

# VP-SAM: Taming Segment Anything Model for Video Polyp Segmentation via Disentanglement and Spatio-Temporal Side Network

**Conference**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03403.pdf)  
**Code**: [https://github.com/zhixue-fang/VPSAM](https://github.com/zhixue-fang/VPSAM)  
**Area**: Segmentation  
**Keywords**: Video Polyp Segmentation, SAM Adaptation, Semantic Disentanglement, Spatio-Temporal Modeling, Frequency Domain Analysis

## TL;DR

This paper proposes VP-SAM, which leverages the amplitude information of the Fourier spectrum via a Semantic Disentanglement Adapter (SDA) to help SAM distinguish low-contrast polyps from the background, while designing a Spatio-Temporal Side Network (STSN) to inject inter-frame temporal information into SAM. It achieves SOTA performance on datasets such as SUN-SEG, CVC-612, and CVC-300.

## Background & Motivation

**Background**: Video Polyp Segmentation (VPS) is a key task in colonoscopy-assisted diagnosis, aiming to segment polyp regions frame-by-frame from colonoscopy videos. In recent years, the Segment Anything Model (SAM) as a general segmentation foundation model has demonstrated strong generalization capabilities, becoming a base for adaptation in various downstream segmentation tasks.

**Limitations of Prior Work**: (1) Polyps and surrounding mucosal tissues are highly similar in color and texture with extremely low contrast, making features trained on natural images by SAM difficult to effectively distinguish between the two; (2) SAM is an image-level model and lacks the capability to handle temporal information of video sequences—where polyp size, position, and shape vary drastically between continuous frames (due to camera motion and bowel peristalsis), and single-frame processing cannot exploit temporal consistency; (3) Existing SAM adaptation methods (such as SAM-Adapter, Medical SAM Adapter) are primarily designed for static images and do not consider spatio-temporal dynamics in videos.

**Key Challenge**: There exists a gap between the powerful general representation ability of SAM and the specific requirements of polyp segmentation—SAM excels at segmenting high-contrast objects but struggles with low-contrast medical targets, and SAM processes single frames while VPS is a chronological task. How can these two gaps be bridged without compromising the pre-trained knowledge of SAM?

**Goal**: (1) How to assist SAM in disentangling the foreground (polyp) and background (mucosa) in low-contrast scenarios? (2) How to inject video temporal information into SAM to achieve inter-frame tracking?

**Key Insight**: The authors observe that although polyps and the background appear similar in the spatial domain, they exhibit different frequency characteristics in the frequency domain—polyp regions typically present distinct texture frequency distributions compared to the background. Utilizing the amplitude spectrum of the Fourier transform can amplify this discrepancy. Meanwhile, by designing a lightweight side network parallel to SAM to handle temporal information, spatio-temporal awareness can be injected without modifying the SAM backbone.

**Core Idea**: Utilizing frequency domain amplitude information to disentangle low-contrast foreground/background combined with a spatio-temporal side network to inject inter-frame information, taking a two-pronged approach to adapt SAM for video polyp segmentation tasks.

## Method

### Overall Architecture

VP-SAM employs a frozen SAM encoder (ViT-B/H) as the backbone, onto which two trainable modules are added: (1) a Semantic Disentanglement Adapter (SDA), inserted into each Transformer layer of the SAM encoder, leveraging frequency domain information to enhance front-background discriminative capability; (2) a Spatio-Temporal Side Network (STSN), running parallel to the SAM encoder, which receives multi-frame inputs to extract temporal features and merges them with SAM features across multiple scales. The input consists of continuous endoscopic video frames, and the output is the polyp segmentation mask for each frame.

### Key Designs

1. **Semantic Disentanglement Adapter (SDA)**:

    - **Function**: Leveraging amplitude information from the Fourier spectrum to help SAM distinguish visually similar polyps and background
    - **Mechanism**: A 2D FFT is performed on the intermediate feature maps of the SAM encoder to extract the amplitude spectrum $|F(u,v)| = \sqrt{Re^2 + Im^2}$. The amplitude spectrum encodes the energy distribution of different frequency components—where texture patterns specific to the polyp region correspond to particular frequency components. This amplitude spectrum is mapped back to the feature space via a lightweight MLP to obtain the frequency-domain enhanced feature $f_{freq}$, which is then fused with the original spatial-domain feature through element-wise addition: $f_{out} = f_{spatial} + \gamma \cdot f_{freq}$, where $\gamma$ is a learnable scaling parameter
    - **Design Motivation**: In the spatial domain, the color/brightness of polyps and the background are highly similar (low contrast), but their texture patterns differ—the polyp surface is usually smoother while the surrounding mucosa has folded textures. The Fourier amplitude spectrum can explicitly capture these texture differences, providing auxiliary distinguishing clues for SAM

2. **Spatio-Temporal Side Network (STSN)**:

    - **Function**: Injecting inter-frame temporal information of the video into the segmentation model without modifying the SAM backbone
    - **Mechanism**: STSN is an independent lightweight encoder that accepts the current frame and its neighboring frames (e.g., 2 frames before and after) as input. First, a shared CNN extracts spatial features for each frame, and then a temporal attention module models inter-frame relationships: $A_{t} = \text{softmax}(Q_t K_{1:T}^T / \sqrt{d}) V_{1:T}$ to capture the motion trajectories and deformation patterns of polyps across continuous frames. The output of STSN is fused with the intermediate features of the SAM encoder across multiple resolutions via skip connections
    - **Design Motivation**: SAM is a single-frame model, unable to perceive the motion states (appearance, disappearance, deformation) of polyps in videos. By injecting temporal information through a parallel side network, the integrity of SAM's pre-trained knowledge is preserved while equipping the model with temporal awareness. The "side network" design is safer than fine-tuning the SAM backbone, successfully avoiding catastrophic forgetting

3. **Multi-Scale Fusion Decoder**:

    - **Function**: Integrating SAM features and STSN temporal features to generate refined segmentation masks
    - **Mechanism**: Across four resolution scales, the intermediate layer features of the SAM encoder are fused with the temporal features of the corresponding scale from STSN using an attention-gated mechanism. The gating weights are determined by the correlation of both feature pathways: $g = \sigma(W_s f_{SAM} + W_t f_{STSN})$, $f_{fused} = g \odot f_{SAM} + (1-g) \odot f_{STSN}$. The fused multi-scale features are decoded into final segmentation masks via progressive upsampling
    - **Design Motivation**: Different scales capture distinct information—low-resolution features provide global context and polyp location information, whereas high-resolution features offer boundary details. Gated fusion allows the model to adaptively decide whether to rely more on SAM features or temporal features at each position

### Loss & Training

A weighted combination of BCE loss and Dice loss is used: $L = L_{BCE} + L_{Dice}$. During training, the SAM encoder backbone (ViT weights) is frozen, and only SDA, STSN, and the decoder are trained. The AdamW optimizer is employed with a learning rate of 1e-4 and a cosine annealing scheduler. Training data consists of consecutive video frame segments (grouped in sets of 5 frames).

## Key Experimental Results

### Main Results

| Dataset | Metric | VP-SAM | PNS+ | SANet | Polyp-PVT | Gain |
|--------|------|--------|------|-------|-----------|------|
| SUN-SEG (Easy) | Dice ↑ | **88.3** | 84.7 | 83.2 | 85.1 | +3.2 |
| SUN-SEG (Hard) | Dice ↑ | **80.1** | 74.6 | 72.8 | 75.9 | +4.2 |
| CVC-612 | Dice ↑ | **92.7** | 89.3 | 88.1 | 90.5 | +2.2 |
| CVC-300 | Dice ↑ | **91.5**| 88.0 | 87.2 | 89.1 | +2.4 |
| SUN-SEG (Hard) | IoU ↑ | **72.8** | 66.3 | 64.1 | 68.2 | +4.6 |

### Ablation Study

| Configuration | SUN-SEG Easy Dice | SUN-SEG Hard Dice | Description |
|------|-------------------|-------------------|------|
| VP-SAM (Full) | **88.3** | **80.1** | Full model |
| w/o SDA | 85.6 | 76.5 | SDA contributes significantly in hard scenarios |
| w/o STSN (Single Frame) | 86.1 | 75.8 | Temporal information is crucial for hard scenarios |
| w/o Frequency Domain (Spatial-only adapter) | 86.8 | 77.3 | Frequency domain information adds about 2.8 gain |
| SDA + STSN w/o Gated Fusion | 87.2 | 78.4 | Gated fusion contributes about 1.7 gain |
| SAM-ViT-B Backbone | 88.3 | 80.1 | Standard configuration |
| SAM-ViT-H Backbone | 89.1 | 81.5 | Larger model further boosts performance |

### Key Findings
- The improvement on the "hard" subset (+4.2 Dice) is significantly greater than that on the "easy" subset (+3.2), suggesting that SDA and STSN are highly effective in low-contrast and large-deformation scenarios.
- The contribution of STSN to hard scenarios (-4.3 Dice when removed) is greater than that of SDA (-3.6), indicating that temporal information is more critical when handling polyp motion and occlusion.
- Frequency-domain analysis (amplitude spectrum) contributes approximately 2.8 more Dice compared to the spatial-only adapter, validating the effectiveness of frequency-domain information for distinguishing low-contrast targets.

## Highlights & Insights
- The concept of **disentangling foreground and background in the frequency domain** is highly ingenious: when foreground and background are indistinguishable in the spatial domain, transforming into the frequency domain may reveal new distinguishing cues. This concept can be transferred to other low-contrast segmentation tasks, such as skin lesion segmentation or retinal vessel segmentation.
- The **side network architecture** strikes a balance between "preserving pre-trained knowledge" and "injecting new capabilities" when adapting foundation models. It serves as a more flexible adaptation paradigm than LoRA/Adapter, especially suited for scenarios that require injecting entirely new modal information (such as temporal sequences).
- The differentiated analysis across easy and hard subsets provides deep insights into the method's effectiveness.

## Limitations & Future Work
- STSN utilizes a fixed frame count (5 frames), which may be insufficient for capturing long-term temporal dependencies in long sequences (such as a polyp reappearing after disappearing).
- The validation is confined to polyp segmentation, without assessing generalization across other medical video segmentation tasks (such as surgical instrument segmentation or ultrasound video segmentation).
- The side network increases parameter count and inference time; although the paper claims it is "lightweight", specific FPS data is not reported.
- Advancements could introduce SAM 2 (which supports video understanding) as the base foundation instead of SAM, which might further improve temporal processing capacities.

## Related Work & Insights
- **vs PNS+ (Polyp-NeoNet-Seg+)**: PNS+ is a video segmentation method tailored for polyps, using optical flow for inter-frame alignment. VP-SAM replaces explicit optical flow computation with the temporal attention mechanism of STSN, rendering it more robust and eliminating the need for an extra optical flow network.
- **vs SAM-Adapter**: SAM-Adapter only inserts adapters in the spatial domain, whereas VP-SAM incorporates frequency-domain disentanglement and temporal modeling, comprehensively addressing the specific requirements of the VPS task.
- **vs Medical SAM Adapter**: Medical SAM Adapter is designed for static medical images, while VP-SAM introduces STSN specifically for video scenarios, expanding the adaptation scope.

## Rating
- Novelty: ⭐⭐⭐⭐ The joint design of frequency-domain disentanglement and a spatio-temporal side network is novel, with the frequency-domain disentanglement concept being particularly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple datasets, detailed ablation studies, and insightful easy/hard subset analysis.
- Writing Quality: ⭐⭐⭐⭐ The motivations are clearly stated, and the method diagrams are intuitive and easy to understand.
- Value: ⭐⭐⭐⭐ Provides an effective solution for adapting SAM to medical video segmentation, and the concept of frequency-domain disentanglement is widely transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SAM 3: Segment Anything with Concepts](../../ICLR2026/segmentation/sam_3_segment_anything_with_concepts.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](../../AAAI2026/segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](../../AAAI2026/segmentation/sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [\[ECCV 2024\] Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)
- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](../../AAAI2026/segmentation/segment_and_matte_anything_in_a_unified_model.md)

</div>

<!-- RELATED:END -->
