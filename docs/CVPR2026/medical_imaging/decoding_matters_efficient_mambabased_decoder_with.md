---
title: >-
  [Paper Note] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Medical Image Segmentation] This paper proposes Deco-Mamba, a decoder-centric segmentation network that employs a Co-Attention Gate (CAG) for bidirectional encoder–decoder feature fusion, a Visual State Space Module (VSSM) for long-range dependency modeling, and deformable convolutions for detail recovery. A windowed distribution-aware KL-divergence deep supervision scheme is further introduced. The method achieves state-of-the-art performance on 7 medical segmentation benchmarks at moderate computational cost.
tags:
  - CVPR 2026
  - Medical Imaging
  - Medical Image Segmentation
  - Mamba
  - Decoder-Centric
  - Deep Supervision
  - Co-Attention Gate
date: 2026-05-08
content_hash: 71c8552dbbb2184b
---

# Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.12547](https://arxiv.org/abs/2603.12547)
**Code**: To be released (upon acceptance)
**Area**: Medical Image Segmentation / State Space Models / Decoder Design
**Keywords**: Medical Image Segmentation, Mamba, Decoder-Centric, Deep Supervision, Co-Attention Gate

## TL;DR
This paper proposes Deco-Mamba, a decoder-centric segmentation network that employs a Co-Attention Gate (CAG) for bidirectional encoder–decoder feature fusion, a Visual State Space Module (VSSM) for long-range dependency modeling, and deformable convolutions for detail recovery. A windowed distribution-aware KL-divergence deep supervision scheme is further introduced. The method achieves state-of-the-art performance on 7 medical segmentation benchmarks at moderate computational cost.

## Background & Motivation

### State of the Field

**Background**: Two unresolved bottlenecks persist in medical image segmentation: (1) most methods are optimized for a single dataset or modality, resulting in poor cross-modal generalization; (2) research attention has been disproportionately focused on the encoder (leveraging large pretrained backbones), leaving decoder design largely neglected. Existing Mamba-based methods (Mamba-UNet, U-Mamba, Swin-UMamba, etc.) primarily apply Mamba to enhance the encoder, without fully exploiting its long-range modeling capability in the decoding stage. Conventional deep supervision resizes intermediate outputs to full resolution, incurring information loss.

### Approach

**Goal**: How to design a computationally efficient and cross-modal generalizable decoder that achieves fine-grained multi-scale feature reconstruction and boundary recovery with low parameter count?

## Method

### Overall Architecture
The network follows a U-Net-like structure with a dual-branch encoder (7×7 CNN for high-resolution detail retention + PVT-V2 Transformer for global context capture). The six-stage decoder cascades Co-Attention Gate → VSSM Block → Deformable Residual Block (DRB), coupled with multi-scale distribution-aware supervision. Two model variants are provided: V0 (PVT-B0, 9.67M) and V1 (PVT-B2, 46.93M).

### Key Designs
1. **Co-Attention Gate (CAG)**: Extends the conventional unidirectional Attention Gate to a bidirectional mechanism — encoder features and decoder features serve as each other's input and gating signal. The two resulting attention outputs are concatenated and refined via channel attention (adaptive max + average pooling → dual 1×1 convolutions → sigmoid) to select the most informative channels. Formally: $D_i' = CA[AG(x=X_i, g=D_{i+1}), AG(x=D_{i+1}, g=X_i)]$
2. **Visual State Space Mamba Block (VSSMB)**: Employs a continuous-time SSM with selective scanning along horizontal, vertical, and their reverse directions to model global context at linear complexity. Two VSSMBs are used in the bottleneck; stages 2–5 each use one; the final stage omits it.
3. **Deformable Residual Block (DRB)**: Combines standard 3×3 convolutions with deformable convolutions that predict pixel-level offsets and modulation masks (sigmoid-constrained to $[0, 2]$), recovering local details and boundaries that SSM processing may smooth.
4. **Multi-Scale Distribution-Aware Deep Supervision (MSDA)**: Instead of resizing intermediate outputs to full resolution, the method computes the intra-window class frequency distribution $\tilde{P}^{(s)}$ at each decoder's native resolution and aligns it with the predicted softmax distribution via KL divergence. Boundary weighting is defined as $W_{h,w}^{(s)} = (1 - \max_n \tilde{P}_{h,w,n}^{(s)})^\alpha$, assigning higher weights to mixed-class regions near boundaries.

### Loss & Training
$\mathcal{L}_{total} = \mathcal{L}_{dice} + \sum_s \lambda_s \mathcal{L}_{dist}^{(s)}$, with monotonically increasing stage weights $\lambda_1 < \lambda_2 < ... < \lambda_S$. AdamW optimizer with cosine learning rate scheduling (warm restart $T=2$); input resolution 224×224; learning rate 1e-4, batch size 16 (primary datasets); training performed on an A5000 24 GB GPU.

## Key Experimental Results

| Dataset | Metric | Deco-Mamba-V1 | Prev. SOTA | Gain |
|---|---|---|---|---|
| Synapse (8-class) | DSC/HD95 | 85.07/14.72 | 83.59/15.99 (Cascaded-MERIT) | +1.48/+1.27 |
| BTCV (13-class) | DSC/HD95 | 78.45/11.77 | 75.87/17.02 (PAG-TransYnet) | +2.58/+5.25 |
| ACDC (cardiac) | DSC | 92.35 | 92.12 (PVT-EMCAD-B2) | +0.23 |
| MoNuSeg | DSC | 85.14 | 81.45 (Swin-UMamba) | +3.69 |
| GlaS | DSC | 96.91 | 96.91 (Cascaded-MERIT) | Tied |

Deco-Mamba-V0 (only 9.67M parameters) approaches the performance of Transformer-based methods with ~150M parameters.

### Ablation Study
- Removing CNN branch: DSC 84.07 (−1.00); removing VSSMB: DSC 83.51 (−1.56)
- CAG vs. conventional AG: 82.98 → 85.07; vs. LGAG: 82.69 → 85.07; vs. CBAM: 84.01 → 85.07
- Deformable conv vs. standard conv: 84.53 vs. 85.07; vs. dynamic conv: 83.77 vs. 85.07
- MSDA vs. conventional deep supervision: the latter improves DSC but degrades HD95 (15.89 vs. 14.72); MSDA improves both metrics
- vs. boundary-aware / distance-based boundary loss: HD95 of 21.43 / 20.64 vs. MSDA's 14.72
- Backbone comparison: PVT-B0 (9.67M) DSC 83.16; Swin-T (70.12M) DSC 83.76; PVT-B2 DSC 85.07

## Highlights & Insights
- Impressive efficiency–accuracy trade-off: V0 with only 9.67M parameters outperforms Mamba-based methods such as SliceMamba and VM-UNet, approaching the 148M-parameter Cascaded-MERIT.
- MSDA loss avoids the information loss caused by resizing in conventional deep supervision by operating directly at native decoder resolutions.
- Comprehensive validation across 7 cross-modal benchmarks (CT / MRI / ultrasound / dermoscopy / pathology) demonstrates strong generalizability.

## Limitations & Future Work
- Validation is limited to 2D slices; extension to 3D volumetric segmentation remains unexplored.
- The method relies on PVT pretrained weights; alternative pretraining strategies (e.g., self-supervised learning) have not been investigated.
- The multi-directional scanning strategy in VSSM lacks systematic ablation.

## Related Work & Insights
- vs. EMCAD (CVPR 2024): Both target decoder enhancement, but EMCAD does not model long-range dependencies. Deco-Mamba-V0 with PVT-B0 already surpasses EMCAD with PVT-B2.
- vs. Cascaded-MERIT (147.86M): Deco-Mamba-V1 achieves +1.48% DSC using approximately one-third of the parameters.
- vs. Swin-UMamba: +3.69% DSC on MoNuSeg with fewer parameters.

## Related Work & Insights
- The decoder-centric design philosophy merits attention — a lightweight encoder paired with a strong decoder may be more efficient than the converse.
- Distribution-aware deep supervision is generalizable to other dense prediction tasks.

## Rating
- Novelty: ⭐⭐⭐ CAG, VSSM, and MSDA each offer incremental contributions; the combination is effective but individual innovations are limited in scope.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 datasets, 20+ comparison methods, comprehensive ablation study.
- Writing Quality: ⭐⭐⭐⭐ Clear figures and tables; detailed module descriptions.
- Value: ⭐⭐⭐ Strong practical utility; valuable to the medical segmentation community; design is transferable to other tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_generalpurpose_vision_models_all_we_need_for_2.md)
- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)

<!-- RELATED:END -->
