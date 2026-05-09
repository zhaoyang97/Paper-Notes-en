---
title: >-
  [Paper Note] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection
description: >-
  [CVPR 2026][Segmentation][RGB-T salient object detection] RSONet is a two-stage RGB-T salient object detection framework that first generates region guidance maps via three parallel encoder-decoder branches and selects the dominant modality based on similarity, then fuses dual-modality features through a selective optimization module. It achieves MAE of 0.020/0.014/0.021 on VT5000/VT1000/VT821, outperforming 27 state-of-the-art methods.
tags:
  - CVPR 2026
  - Segmentation
  - RGB-T salient object detection
  - modality selection
  - region guidance
  - selective optimization
  - Swin Transformer
date: 2026-05-08
content_hash: f63f57c05a24a374
---

# RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.12685](https://arxiv.org/abs/2603.12685)
**Code**: N/A
**Area**: Image Segmentation
**Keywords**: RGB-T salient object detection, modality selection, region guidance, selective optimization, Swin Transformer

## TL;DR

RSONet is a two-stage RGB-T salient object detection framework that first generates region guidance maps via three parallel encoder-decoder branches and selects the dominant modality based on similarity, then fuses dual-modality features through a selective optimization module. It achieves MAE of 0.020/0.014/0.021 on VT5000/VT1000/VT821, outperforming 27 state-of-the-art methods.

## Background & Motivation

**Background**: Salient object detection (SOD) aims to identify the most visually prominent objects in a scene at the pixel level. With the advancement of deep learning, RGB-T SOD leverages thermal infrared images to compensate for the limitations of RGB in challenging scenes, making it an active direction in multimodal salient object detection.

**Limitations of Prior Work**: (1) RGB images suffer from detection difficulties under complex backgrounds, low contrast, or low-light conditions; (2) thermal infrared images, while immune to illumination, may fail to distinguish targets from backgrounds due to environmental temperature and material properties; (3) existing RGB-T fusion strategies (addition/multiplication/concatenation/attention) implicitly treat both modalities as equally informative, introducing substantial noise when the quality gap between modalities is large.

**Key Challenge**: The distribution of salient regions is inconsistent across modalities—one modality may contain accurate target information while the other is dominated by noise, and equal-weight fusion degrades the quality of both.

**Goal**: Adaptively determine which modality is more reliable and allow the reliable modality to dominate the fusion process, thereby avoiding noise interference from the lower-quality modality.

**Key Insight**: A "region guidance stage" independently predicts guidance maps for RGB, Thermal, and RGB+T streams, then compares their similarities to select the dominant modality, which subsequently guides the "saliency generation stage."

**Core Idea**: Assess modality quality prior to fusion, enabling the higher-quality modality to guide the lower-quality one rather than blending them indiscriminately.

## Method

### Overall Architecture

The framework consists of two stages. **Region guidance stage**: Three parallel encoder-decoder branches for RGB, Thermal, and RGB+T (sharing a Swin Transformer backbone) generate guidance maps $G^R$, $G^T$, and $G^{RT}$, respectively. The similarity of $G^R$ and $G^T$ to $G^{RT}$ is computed to select the dominant modality. **Saliency generation stage**: A selective optimization (SO) module fuses dual-modality features based on the similarity result; low-level features are enhanced by the DDE module to preserve edge details, while high-level features are processed by the MIS module to mine positional cues. Cross-level connections yield the final saliency map.

### Key Designs

1. **Context Interaction (CI) Module + Spatial-aware Fusion (SF) Module + Similarity Computation**
   - The CI module adopts a layer-adaptive convolution kernel strategy: low-level features are processed by four parallel branches with kernels of 1×1/3×3/5×5/7×7 to capture multi-scale context; the 7×7 branch is removed for mid-level features; only 1×1/3×3 branches are retained for high-level features—since high-level feature maps have low resolution, large kernels tend to introduce background noise.
   - Residual connections between branches (adding the output of the previous branch to the input of the current branch) bridge features across different scales.
   - The SF module generates spatial weights via global max pooling + 1×1 conv + sigmoid, applies multiplicative and additive refinement to CI outputs, and fuses features hierarchically in a top-down manner.
   - Similarity computation: pixel-wise means $M^R$, $M^T$, $M^{RT}$ are computed from the three guidance maps; the modality with the smaller difference $|M^R - M^{RT}|$ vs. $|M^T - M^{RT}|$ is selected as the dominant modality.

2. **Selective Optimization (SO) Module**
   - Dual-modality features are first enhanced via element-wise multiplication and addition with guidance map $G^{RT}$ to suppress background regions.
   - Channel attention (1×1 conv → GAP → sigmoid) is applied to each modality to further refine channel responses.
   - The spatial attention of the dominant modality is applied to the non-dominant modality's features for cross-modal optimization; the two streams are then summed to yield the fused output.
   - Two symmetric fusion paths (R→T or T→R) are defined depending on the selected dominant modality.

3. **DDE (Dense Detail Enhancement) + MIS (Mutual Interactive Semantic)**
   - DDE employs a four-branch dilated convolution structure (d=1,3,5,7) with dense connections (each branch output is added to all subsequent branch inputs), followed by four VSS (Visual State Space) blocks to capture spatial relationships. It processes low-level features to preserve edge details.
   - MIS adopts a 3-main-branch × 3-sub-branch (d=1,2,3) mutual interaction structure for high-level features: the output of the first sub-branch is added to the inputs of the remaining two, enabling multi-scale receptive field interaction. Final channel attention aggregates the outputs.

### Loss & Training

A joint supervision combining BCE loss, boundary IoU loss, and F-measure loss is applied to five saliency maps (deep supervision). The Swin Transformer backbone is initialized with ImageNet pre-trained weights. Training uses RMSprop ($lr=1 \times 10^{-4}$), input resolution 384×384, on a single RTX 4080 GPU.

## Key Experimental Results

### Main Results

| Dataset | MAE↓ | $F_\beta$↑ | $E_\xi$↑ | $S_\alpha$↑ |
|---------|------|-----------|---------|------------|
| VT5000 | **0.020** | **0.910** | **0.926** | **0.963** |
| VT1000 | **0.014** | 0.923 | 0.946 | 0.972 |
| VT821 | 0.021 | 0.883 | 0.921 | 0.946 |

vs. PATNet (KBS24): VT5000 $F_\beta$ +3.4%, $E_\xi$ +1.2%
vs. ContriNet (TPAMI25): VT5000 $F_\beta$ +3.6%, $S_\alpha$ +2.4%
Speed: ~8.8 FPS (101.3M parameters), significantly slower than CGFNet at 52.3 FPS.

### Ablation Study

| Variant | VT5000 MAE↓ | VT5000 $F_\beta$↑ |
|---------|-------------|------------------|
| Full RSONet | 0.0197 | 0.9071 |
| SO module → simple addition | 0.0208 | 0.8952 |
| SO module → concatenation | 0.0217 | 0.8857 |
| w/o similarity-guided selection (fixed fusion direction) | 0.0215 | 0.8896 |
| w/o DDE + MIS | 0.0217 | 0.8834 |
| Swin Transformer → ResNet50 | — | 0.8146 |

### Key Findings

- Similarity-guided modality selection contributes significantly—removing it increases MAE by 9.1%.
- The SO module outperforms all simple fusion strategies (addition/multiplication/concatenation/channel attention).
- DDE and MIS are complementary—removing both raises MAE by 10.2%, and removing either individually also degrades performance.
- Swin Transformer substantially outperforms ResNet variants, with a $F_\beta$ gap of up to 9 percentage points.

## Highlights & Insights

- The adaptive modality selection strategy is novel—it selects the dominant modality based on per-image conditions rather than applying equal-weight fusion, offering general insights for multimodal fusion tasks.
- The layer-adaptive convolution kernel design is well-motivated—large receptive fields for low-level features and small receptive fields for high-level features align with the resolution characteristics of each stage.
- The comprehensive evaluation against 27 competing methods covers RGB-T SOD work from 2021 to 2025.

## Limitations & Future Work

- At 8.8 FPS, inference speed is insufficient for real-time applications due to the computational overhead of the three-branch parallel encoder and dense dilated convolutions.
- The similarity computation is overly simplistic (scalar comparison based on global pixel mean), making it incapable of capturing spatial distribution differences—scenarios where one modality is locally superior cannot be handled.
- The method may fail on extremely small or elongated targets, or when both modalities degrade simultaneously.
- The quality of guidance maps depends on the encoder-decoder's predictive capability, which may yield erroneous guidance on difficult samples.

## Related Work & Insights

- **SAMSOD (Liu et al.)**: SAM-based RGB-T SOD that handles modality imbalance via gradient conflict resolution; VT5000 MAE 0.021 vs. Ours 0.020.
- **Samba (CVPR25)**: A pure Mamba framework for salient object detection; VT5000 $F_\beta$ 0.894 vs. Ours 0.910.
- **ContriNet (TPAMI25)**: A three-stream divide-and-merge design; VT5000 $F_\beta$ 0.878 vs. Ours 0.910.
- The modality selection strategy is generalizable to any multimodal fusion task (RGB-D/RGB-Event/multispectral, etc.), with the core philosophy of "assess before fuse."
- VSS blocks demonstrate strong performance in low-level feature detail enhancement and are worth exploring in other dense prediction tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The region-guided modality selection is original, though the overall paradigm remains encoder-decoder with attention mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 27 competing methods, 3 datasets, 4 metrics, and multi-faceted ablation studies.
- Writing Quality: ⭐⭐⭐ Method descriptions are detailed but involve numerous modules and symbols, raising the reading barrier.
- Value: ⭐⭐⭐⭐ Practically valuable within the RGB-T SOD subfield; the modality selection idea is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportionaware_dynamic_adaptive_sali.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)

</div>

<!-- RELATED:END -->
