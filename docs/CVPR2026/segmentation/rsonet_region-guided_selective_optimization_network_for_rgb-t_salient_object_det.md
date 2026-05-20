---
title: >-
  [Paper Note] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection
description: >-
  [CVPR 2026][Segmentation][RGB-T salient object detection] This paper proposes RSONet, a two-stage RGB-T salient object detection network. In the region guidance stage…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "RGB-T salient object detection"
  - "region guidance"
  - "selective optimization"
  - "multimodal fusion"
  - "Swin Transformer"
  - "visual state space model"
date: 2026-05-08
content_hash: 1f83404cf123302a
---

# RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.12685](https://arxiv.org/abs/2603.12685)  
**Code**: To be confirmed  
**Area**: Semantic Segmentation / Salient Object Detection
**Keywords**: RGB-T salient object detection, region guidance, selective optimization, multimodal fusion, Swin Transformer, visual state space model

## TL;DR

This paper proposes RSONet, a two-stage RGB-T salient object detection network. In the region guidance stage, similarity scores between RGB/thermal guidance maps and a joint guidance map are computed to select the more reliable modality. In the saliency generation stage, a selective optimization (SO) module fuses dual-modality features based on the selection result, while Dense Detail Enhancement (DDE) and Mutual Interaction Semantic (MIS) modules extract detail and positional information, respectively, to produce high-quality saliency maps. RSONet achieves state-of-the-art performance on three RGB-T benchmarks.

## Background & Motivation

**Limitations of single-modal RGB**: Pure RGB methods suffer significant performance degradation in complex backgrounds, low contrast, and ambiguous boundary scenarios, necessitating auxiliary modality information.

**Insufficiency of depth information**: In RGB-D methods, depth maps struggle to distinguish objects from spatially adjacent backgrounds, and depth quality is highly sensitive to acquisition hardware and distance.

**Introduction of the thermal modality**: Thermal images are invariant to illumination changes and effectively complement RGB in nighttime and low-light conditions, though they are themselves affected by ambient temperature and material properties.

**Bimodal inconsistency** (core motivation): Salient region distributions in RGB and thermal images are frequently inconsistent — some samples have clear RGB but blurry thermal maps, and vice versa. Naive concatenation, addition, or attention-based fusion introduces substantial noise under such conditions.

**Limitations of existing fusion strategies**: Additive, multiplicative, concatenation, and attention-based mechanisms implicitly assume equal modality importance and cannot adapt to large disparities in information quality.

**Lack of modality selection mechanisms**: Most methods do not explicitly assess which modality is more reliable. This paper addresses this gap by introducing region guidance combined with similarity computation for adaptive modality dominance selection.

## Method

### Overall Architecture

RSONet consists of two stages:

- **Region Guidance Stage**: Three parallel encoder-decoder branches generate guidance maps $\mathbf{G}^R$, $\mathbf{G}^T$, and $\mathbf{G}^{RT}$; similarity computation then selects the dominant modality.
- **Saliency Generation Stage**: Based on the selection result, an SO module fuses dual-modality features; DDE (lower layers) and MIS (higher layers) modules then extract detail and positional information, respectively, with cross-layer connections producing the final saliency map.

All backbone networks are based on Swin Transformer, extracting five-level multi-scale features.

### Key Designs

#### 1. Context Interaction Module (CI)

To accommodate resolution differences across feature levels, three variants are designed:

| Variant | Applied Level | Kernel Sizes |
|---------|--------------|--------------|
| Variant 1 | Low-level $\mathbf{F}_1$ | 1×1, 3×3, 5×5, 7×7 |
| Variant 2 | Mid-level $\mathbf{F}_{2/3}$ | 1×1, 3×3, 5×5 |
| Variant 3 | High-level $\mathbf{F}_{4/5}$ | 1×1, 3×3 |

Each branch employs a cascaded accumulation strategy, adding the previous branch output to the current input to break inter-scale barriers, with final concatenation along the channel dimension.

#### 2. Spatial-aware Fusion Module (SF)

Two layers of 3×3 convolutions are applied to CI output features, followed by global max pooling → 1×1 convolution → Sigmoid to generate spatial weights, which are multiplied and added back to the original features for spatial-dimension optimization. Top-down layer-by-layer fusion ultimately generates the guidance map for each branch.

#### 3. Similarity Computation and Modality Selection

The mean values $M^R$, $M^T$, and $M^{RT}$ are computed from the three guidance maps. By comparing $|M^R - M^{RT}|$ and $|M^T - M^{RT}|$, the modality with the smaller difference is deemed more informative and assigned as the dominant modality for subsequent fusion.

#### 4. Selective Optimization Module (SO)

- The joint guidance map $\mathbf{G}^{RT}$ is used to multiplicatively enhance dual-modality features via residual addition.
- Channel attention suppresses interference introduced by the guidance map.
- Spatial attention from the more reliable modality guides the optimization of the other modality's features.
- The two optimized feature streams are summed as output.

#### 5. Dense Detail Enhancement Module (DDE)

Applied to low-level features (layers 1–3), DDE employs four parallel dilated convolutions (dilation rates 1/3/5/7) with dense connections so that each branch shares multi-scale receptive field information. Each branch is followed by a Visual State Space (VSS) block for further spatial relationship modeling, with final concatenation along the channel dimension.

#### 6. Mutual Interaction Semantic Module (MIS)

Applied to high-level features (layers 4–5), MIS uses 3×3 convolutions with dilation rates 1/2/3, organized into three main branches. Within each main branch, three sub-branches mutually interact through their outputs; the three main branches are then concatenated and processed with channel attention to suppress noise.

### Loss & Training

The joint loss combines BCE, BIoU, and F-measure losses, applied with deep supervision across 5 scales:

$$L_{total} = \frac{1}{N}\sum_{i=1}^{N}(L_{bce} + L_{iou} + L_{fm})$$

## Key Experimental Results

### Datasets and Settings

- **Training set**: VT5000 training split (2,500 images)
- **Test sets**: VT5000 test split (2,500 images), VT1000 (1,000 images), VT821 (821 images)
- Input resolution: 384×384; RMSprop optimizer; learning rate: 1e-4; single RTX 4080 GPU

### Main Results

| Dataset | $\mathcal{M}$↓ | $F_\beta$↑ | $S_\alpha$↑ | $E_\xi$↑ |
|---------|------|------|------|------|
| VT5000 | **0.020** | **0.910** | **0.926** | **0.963** |
| VT1000 | **0.014** | 0.923 | 0.946 | 0.972 |
| VT821 | 0.021 | 0.883 | 0.921 | 0.946 |

Compared against 27 state-of-the-art methods, RSONet improves $F_\beta$ by 3.4%, $E_\xi$ by 1.2%, and $S_\alpha$ by 1.1% over PATNet on VT5000.

### Model Efficiency

| Metric | Value |
|--------|-------|
| Parameters | 88M |
| FLOPs | 143.8G |
| Inference Speed | 9.4 FPS |

Parameter count is moderate (owing to weight sharing across the three branches), but the two-stage design results in relatively low inference speed.

### Ablation Study

| Setting | $\mathcal{M}$↓ | $F_\beta$↑ | $S_\alpha$↑ | $E_\xi$↑ |
|---------|------|------|------|------|
| w/o SO (addition) | 0.0217 | 0.8883 | 0.9213 | 0.9523 |
| w/o SO (multiplication) | 0.0208 | 0.8948 | 0.9231 | 0.9587 |
| w/o SO (concatenation) | 0.0215 | 0.8896 | 0.9224 | 0.9558 |
| w/o SO (pixel-wise soft gating) | 0.0203 | 0.8951 | 0.9239 | 0.9605 |
| R→T (no region guidance) | 0.0215 | 0.8898 | 0.9230 | 0.9561 |
| T→R (no region guidance) | 0.0216 | 0.8896 | 0.9233 | 0.9554 |
| w/o DDE | 0.0203 | 0.9082 | 0.9213 | 0.9631 |
| w/o MIS | 0.0203 | 0.8997 | 0.9241 | 0.9593 |
| w/o DDE & MIS | 0.0217 | 0.9053 | 0.8995 | 0.9556 |
| **Full RSONet** | **0.0197** | **0.9071** | **0.9261** | **0.9632** |

### Backbone Ablation

Swin Transformer substantially outperforms ResNet-18/34/50 and also surpasses frozen SAM/DINO backbones, as large pre-trained models unadapted to the RGB-T domain without task-specific adaptors yield degraded performance.

### Key Findings

- Replacing the SO module with naive fusion alternatives leads to significant performance drops, confirming that region-guided selective fusion is the core contribution.
- Fixing the fusion direction to either R→T or T→R underperforms adaptive selection, validating the necessity of modality selection.
- DDE and MIS each contribute complementary detail and localization information; removing both simultaneously causes $S_\alpha$ to drop to 0.8995.

## Highlights & Insights

- The paper explicitly models the bimodal salient region inconsistency problem and proposes a region-guidance plus similarity-based modality dominance selection strategy with clear motivation.
- The CI module employs different kernel sizes adapted to feature resolutions at different levels, reflecting thoughtful design.
- The combination of dense connections and VSS blocks in DDE effectively captures low-level spatial structural information.
- Ablation experiments are thorough, with detailed comparisons and visualizations for SO, DDE, and MIS modules.

## Limitations & Future Work

- Inference speed of only 9.4 FPS makes real-time deployment impractical due to the computational overhead of the two-stage, three-branch architecture.
- The similarity computation relies solely on a global comparison of guidance map means, making it insensitive to local regional differences.
- Evaluation is limited to three RGB-T datasets (VT5000/VT1000/VT821), with no generalization experiments on RGB-D or video SOD benchmarks.
- Detection quality degrades for extremely small targets or when both modalities are simultaneously of low quality (acknowledged as failure cases in the paper).
- Frozen large-scale models (SAM/DINO) as backbones yield inferior results, yet the paper does not explore fine-tuning or adaptor-based adaptation.

## Related Work & Insights

- **RGB-D SOD**: Methods leveraging depth information for enhanced detection, including representative works EMTrans and the Group Transformer by Fang et al.
- **RGB-T SOD**: MCFNet (modality complementary fusion), HRTransNet (high-resolution Transformer), WaveNet (frequency-domain perspective), Samba (pure Mamba framework), and SAMSOD (SAM-based approach).
- **Single-modal SOD**: Classic methods including AttFeedback, DenseAttFluid, and BilateralExtreme.
- The most recent competing methods include ContriNet (TPAMI 2025), Samba (CVPR 2025), and SAMSOD (TMM 2026).

## Rating

- **Novelty**: ⭐⭐⭐ — The region-guidance and similarity-based modality selection idea is reasonably novel, though the individual sub-module designs (CI/SF/DDE/MIS) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparisons against 27 methods, extensive ablations, backbone analysis, visualizations, and failure case analysis.
- **Writing Quality**: ⭐⭐⭐ — Formulations and module descriptions are detailed, but the writing is verbose and notation is dense.
- **Value**: ⭐⭐⭐ — Achieves state-of-the-art performance in the RGB-T SOD sub-field, though limited inference speed constrains practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportionaware_dynamic_adaptive_sali.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](../../AAAI2026/segmentation/sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)

</div>

<!-- RELATED:END -->
