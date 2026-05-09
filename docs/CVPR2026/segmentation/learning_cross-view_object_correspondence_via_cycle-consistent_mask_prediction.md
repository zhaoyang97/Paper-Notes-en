---
title: >-
  [Paper Note] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction
description: >-
  [CVPR 2026][Segmentation][cross-view correspondence] This paper proposes CCMP, a cross-view object correspondence framework based on conditional binary segmentation. It leverages cycle-consistency constraints as a self-supervised signal and supports test-time training (TTT), achieving state-of-the-art performance of 44.57% mIoU on Ego-Exo4D.
tags:
  - CVPR 2026
  - Segmentation
  - cross-view correspondence
  - cycle consistency
  - conditional segmentation
  - test-time training
  - egocentric view
date: 2026-05-08
content_hash: 561390d9a89facca
---

# Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction

**Conference**: CVPR 2026
**arXiv**: [2602.18996](https://arxiv.org/abs/2602.18996)
**Code**: [GitHub](https://github.com/shannany0606/CCMP)
**Area**: Segmentation / Cross-View Correspondence
**Keywords**: cross-view correspondence, cycle consistency, conditional segmentation, test-time training, egocentric view

## TL;DR

This paper proposes CCMP, a cross-view object correspondence framework based on conditional binary segmentation. It leverages cycle-consistency constraints as a self-supervised signal and supports test-time training (TTT), achieving state-of-the-art performance of 44.57% mIoU on Ego-Exo4D.

## Background & Motivation

Cross-view visual correspondence — particularly between egocentric and exocentric perspectives — is a core capability for embodied intelligence. For example, a service robot must localize objects in a third-person view based on instructions from a first-person perspective. This task poses three major challenges:

**Severe appearance variation**: Egocentric views suffer from shake, clutter, and motion blur, while exocentric views are stable but may lack fine-grained detail.

**Large spatial context discrepancy**: The surrounding environment of an object differs drastically across views, making background-based matching infeasible.

**Distinct temporal dynamics**: Object motion and deformation vary significantly between camera perspectives.

Existing methods either rely on auxiliary modules (ObjectRelator) or require pre-generated candidate masks (O-MaMa), resulting in complex architectures with limited generalizability.

## Method

### Overall Architecture

An end-to-end conditional binary segmentation framework: given a source image $I_s$, a target image $I_t$, and a source mask $M_s$, the model predicts the corresponding object mask $M_t$ in the target view. The framework consists of three components: a Source Feature Extractor (ConvNeXt-based DINOv3-L), a Transformer Encoder (ViT-based DINOv3-L), and a Multi-task Decoder.

### Key Designs

1. **Conditioning Token (CDT) Injection**: Source image features are extracted via the backbone, and a compact object representation is obtained through mask-weighted average pooling: $z_s = \sum_{i,j} \tilde{M}_s[i,j] \cdot F_s[:,i,j]$, where $\tilde{M}_s$ is the normalized source mask. This representation is linearly projected into a CDT token and injected into the Transformer encoder, propagating object-aware information across visual tokens of the target image via cross-token attention. Design advantage: only a single additional token is introduced, remaining fully compatible with the pretrained backbone with minimal architectural modification.

2. **Cycle-Consistency Loss**: The core self-supervised signal. The source mask $M_s$ is used to predict the target mask $\hat{M}_t$, which is then inversely mapped back to the source view to reconstruct $\hat{M}_s$, with the constraint $\mathcal{L}_{cycle} = \mathcal{L}_{bce}(M_s, \hat{M}_s)$. A key property is that **no ground-truth target mask is required**, enabling direct use at inference time and thus TTT. This forms an elegant self-supervised closed loop — the model must learn view-invariant representations to successfully complete the round-trip mapping.

3. **Test-Time Training (TTT)**: At inference time, the cycle-consistency loss is used to fine-tune the model on each test sample pair for a few steps — updating only the last $K$ layers of the Transformer encoder for $T$ gradient steps at lr $= 5 \times 10^{-6}$. For Ego2Exo: $K=4, T=2$; for the harder Exo2Ego direction: $K=11, T=6$. TTT enables the model to adapt to distribution shifts specific to each test pair.

### Loss & Training

- **Mask loss** $\mathcal{L}_{mask} = \mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice}$, $\lambda_{dice}=5$
- **Auxiliary loss** $\mathcal{L}_{aux}$: the same mask loss applied to the second-to-last Transformer layer output (deep supervision)
- **Total loss** $\mathcal{L}_{total} = \mathcal{L}_{mask} + \lambda_{aux}\mathcal{L}_{aux} + \lambda_{cycle}\mathcal{L}_{cycle}$, $\lambda_{aux}=1, \lambda_{cycle}=10$
- Two-stage training: Stage 1 freezes the DINOv3 backbone for 64K iterations; Stage 2 fine-tunes all parameters for 640K iterations
- Gradient accumulation over 16 steps; trained on 8×A800 GPUs for approximately 72 hours
- Data augmentation: unified bidirectional Ego2Exo/Exo2Ego training, synthesis of same-view pairs (Ego2Ego, Exo2Exo), and relaxed temporal alignment

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Ego-Exo4D (Exo Query) | IoU | **47.18** | 44.08 (O-MaMa) | +3.10 |
| Ego-Exo4D (Ego Query) | IoU | 41.95 | 42.57 (O-MaMa) | -0.62 |
| Ego-Exo4D | mIoU | **44.57** | 43.32 (O-MaMa) | +1.25 |
| Ego-Exo4D | CA (Ego) | **0.669** | 0.590 (O-MaMa) | +13.4% |
| HANDAL-X (zero-shot) | IoU | **78.8** | 42.8 (ObjectRelator) | +36.0 |
| HANDAL-X (fine-tuned) | IoU | **85.0** | 84.7 (ObjectRelator) | +0.3 |

### Ablation Study

| Configuration | Ego-IoU | Exo-IoU | mIoU | Note |
|------|---------|---------|------|------|
| Full model | 41.95 | 47.18 | **44.57** | All components |
| w/o $\mathcal{L}_{cycle}$ | 40.28 | 45.82 | 43.05 | Cycle consistency is critical |
| w/o $\mathcal{L}_{aux}$ | 40.64 | 43.81 | 42.90 | Deep supervision matters |
| w/o TTT | 41.79 | 44.18 | 42.99 | TTT contributes +1.58 |
| w/o same-view augmentation | 40.88 | 45.50 | 43.19 | Data diversity is important |

### Key Findings

- Ego Query is consistently harder than Exo Query, as exocentric target objects tend to be smaller and surrounded by more cluttered environments.
- General segmentation models such as SEEM and PSALM perform poorly on this task (IoU < 10%), demonstrating the necessity of cross-view training.
- Zero-shot IoU of 78.8% on HANDAL-X substantially outperforms all baselines, indicating strong cross-domain generalization after training on Ego-Exo4D.
- Even when DINOv3 features are replaced with the weaker DINOv2, the proposed method still outperforms "baseline + DINOv3," confirming that performance gains primarily stem from the method design rather than stronger features.

## Highlights & Insights

- **Minimal design, strong performance**: The introduction of a single CDT token and a cycle-consistency loss results in negligible architectural overhead yet achieves excellent performance.
- Cycle consistency unifies training and inference — the same loss serves as supervision during training and enables TTT at inference time.
- TTT is applied to cross-view correspondence for the **first time** and consistently yields performance improvements.
- The data augmentation strategy (same-view pairing, relaxed temporal alignment) is simple yet effective and warrants adoption in related settings.

## Limitations & Future Work

- Ego Query performance remains slightly below O-MaMa (41.95 vs. 42.57); segmentation of small objects in exocentric views still has room for improvement.
- TTT requires additional gradient update steps at inference time, increasing latency.
- The visibility prediction (CLS Head) is trained in a post-hoc manner, separate from the main model, which may limit joint optimization.
- Extreme cases where the object is invisible during the cycle are not explicitly handled, though such cases are rare in Ego-Exo4D.

## Related Work & Insights

- **O-MaMa**: A mask-matching method that relies on FastSAM to pre-generate candidates; the proposed end-to-end approach is more straightforward.
- **ObjectRelator**: Fuses visual and textual cues with a complex architecture; the purely visual approach proposed here generalizes better (HANDAL-X +36%).
- **TTT family** (Sun et al. 2020, 2024): Test-time training has been extended from classification to video and language tasks; this work is the first to apply it to cross-view correspondence.
- Insight: The combination of cycle consistency and TTT can be generalized to other correspondence tasks requiring self-supervision, such as 3D point cloud registration.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of cycle consistency and TTT is novel; CDT injection is elegant and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-benchmark evaluation on Ego-Exo4D and HANDAL-X with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; motivation and design rationale are coherent.
- Value: ⭐⭐⭐⭐ Establishes a clean and efficient new baseline for cross-view correspondence; the TTT strategy has broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FoV-Net: Rotation-Invariant CAD B-rep Learning via Field-of-View Ray Casting](fov-net_rotation-invariant_cad_b-rep_learning_via_field-of-view_ray_casting.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)
- [\[ICCV 2025\] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views](../../ICCV2025/segmentation/o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)
- [\[CVPR 2026\] GenMask: Adapting DiT for Segmentation via Direct Mask Generation](genmask_adapting_dit_for_segmentation_via_direct_mask_generation.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)

<!-- RELATED:END -->
