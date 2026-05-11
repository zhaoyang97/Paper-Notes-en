---
title: >-
  [Paper Note] Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation
description: >-
  [NeurIPS 2025][3D Vision][3D Reconstruction] Fin3R is proposed to improve the geometric accuracy and robustness of feed-forward 3D reconstruction models (DUSt3R/MASt3R/CUT3R/VGGT) in a unified and lightweight manner…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D Reconstruction"
  - "Feed-forward Reconstruction"
  - "Knowledge Distillation"
  - "LoRA Fine-tuning"
  - "Monocular Depth Estimation"
  - "DUSt3R"
  - "MASt3R"
  - "CUT3R"
  - "VGGT"
date: 2026-05-08
content_hash: c360e6541abb3c96
---

# Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2511.22429](https://arxiv.org/abs/2511.22429)
**Code**: [visual-ai/fin3r](https://github.com/visual-ai/fin3r)
**Area**: 3D Vision
**Keywords**: 3D Reconstruction, Feed-forward Reconstruction, Knowledge Distillation, LoRA Fine-tuning, Monocular Depth Estimation, DUSt3R, MASt3R, CUT3R, VGGT

## TL;DR

Fin3R is proposed to improve the geometric accuracy and robustness of feed-forward 3D reconstruction models (DUSt3R/MASt3R/CUT3R/VGGT) in a unified and lightweight manner, by freezing the decoder and fine-tuning the encoder via monocular knowledge distillation with re-normalization LoRA adapters.

## Background & Motivation

**Rise of feed-forward 3D reconstruction**: Models such as DUSt3R, MASt3R, CUT3R, and VGGT regress pointmaps from multi-view images in a single forward pass, bypassing the iterative optimization of traditional SfM pipelines, yet their geometric detail remains coarse.

**Gap with monocular methods**: Despite their efficiency advantages, the depth predictions of feed-forward models still lag behind state-of-the-art monocular geometry estimation methods such as Depth Anything V2 and MoGe, exhibiting blurry boundaries and inaccurate reconstruction of transparent or reflective surfaces.

**Scarcity of high-quality training data**: Existing real-world datasets suffer from noisy depth labels, imprecise pose annotations, and a bias toward indoor scenes, limiting generalization.

**Pointmap degradation in long sequences**: Multi-view pointmap regression inherently couples pose estimation with depth estimation; views distant from the reference frame exhibit drift and scale ambiguity, causing loss of geometric detail in non-reference views.

**Limitations of existing fine-tuning approaches**: Methods such as LoRA-3D and Test3R rely on per-scene test-time optimization and cannot generalize zero-shot to new scenes. Approaches like Align3R and Pow3R that inject external depth priors introduce additional inference modules and runtime overhead.

**Encoder as the bottleneck**: Analysis reveals that insufficient reconstruction detail originates primarily from the encoder's feature extraction capability, while the decoder's multi-view matching ability is inherently strong; targeted reinforcement of the encoder alone is therefore sufficient.

## Method

### Overall Architecture

Fin3R builds upon pretrained feed-forward 3D reconstruction models, freezing the decoder (responsible for cross-view matching) and applying lightweight fine-tuning exclusively to the shared encoder (responsible for feature extraction). Knowledge is distilled from the strong monocular teacher model MoGe on the large-scale unannotated dataset SA-1B into the encoder via customized re-normalization LoRA adapters. The same implementation directly accommodates four distinct architectures: DUSt3R, MASt3R, CUT3R, and VGGT.

### Key Design 1: Encoder-only Distillation

- **Function**: The decoder is frozen; only the encoder is fine-tuned with LoRA adapters to distill fine-grained geometric knowledge from the teacher model MoGe.
- **Mechanism**: The encoder handles single-image feature extraction while the decoder handles cross-view association; insufficient local geometric detail originates from the encoder, so reinforcing the encoder alone addresses this shortcoming.
- **Design Motivation**: Freezing the decoder preserves existing multi-view matching capability; using LoRA rather than full-parameter fine-tuning keeps the number of additional parameters minimal, with negligible impact on inference memory and latency.

### Key Design 2: Re-normalization LoRA

- **Function**: A customized re-normalization layer is embedded in each LoRA block to constrain the L2 norm of the merged weights back to the level of the original weights after the weight update: $W' = (W + \Delta W) \cdot \|W\|_2 / \|W + \Delta W\|_2$.
- **Mechanism**: Monocular distillation causes the encoder feature norms to grow continuously, deviating from the feature distribution expected by the frozen decoder and thereby degrading multi-view matching. Re-normalization explicitly prevents this feature norm drift.
- **Design Motivation**: Experiments show that naive LoRA with multi-view data replay fails to resolve feature drift (the mean feature norm rises from 9.61 to 10.53/10.34); only with re-normalization does it recover to 9.73, preserving multi-view performance.

### Key Design 3: Multi-view Data Replay

- **Function**: A small amount of multi-view data (Hypersim + TartanAir) is mixed into the distillation training alongside the monocular distillation data.
- **Mechanism**: Multi-view pointmap regression samples are interleaved with the large volume of monocular distillation samples, ensuring the encoder learns fine-grained geometry without forgetting the requirements of multi-view tasks.
- **Design Motivation**: Pure monocular distillation degrades multi-view performance even with a frozen decoder; data replay provides an anchor for the feature distribution and complements the re-normalization strategy.

## Loss & Training

- **Distillation loss**: $\mathcal{L}_{\text{distill}} = \beta^D \|D - \hat{D}\|_2^2 - \lambda \log \beta^D$, aligning the predicted depth $D$ with the pseudo-labels $\hat{D}$ from the teacher MoGe, where $\beta^D$ is an uncertainty weight.
- **Pointmap regression loss**: $\mathcal{L}_{\text{pointmap}} = \mathbf{1}_{\text{mv}} (\beta^P \|P - P^{GT}\|_2^2 - \lambda \log \beta^P)$, applied only to multi-view samples.
- **Training configuration**: Each epoch samples 20,000 images from SA-1B, 1,000 from Hypersim, and 1,000 from TartanAir; training runs for 10 epochs on 4×NVIDIA L20 GPUs, completing in approximately one day.

## Key Experimental Results

### Monocular Depth Estimation (Table 1)

Scale-invariant relative depth is evaluated on 7 standard benchmarks; all models improve consistently with Fin3R:

| Method | NYUv2 Rel↓ | KITTI Rel↓ | ETH3D Rel↓ | Avg. Rel↓ | Avg. δ₁↑ |
|--------|-----------|-----------|-----------|----------|----------|
| DUSt3R | 3.83 | 7.64 | 5.35 | 7.03 | 92.3 |
| DUSt3R+Fin3R | **3.68** | **6.02** | **4.41** | **5.58** | **94.8** |
| VGGT | 3.14 | 5.83 | 3.64 | 5.77 | 94.0 |
| VGGT+Fin3R | **3.10** | **4.59** | **3.07** | **4.29** | **96.7** |
| MoGe (Teacher) | 3.02 | 4.39 | 2.96 | 4.14 | 96.9 |

VGGT+Fin3R approaches the performance of the teacher model MoGe very closely. MASt3R's metric depth average Rel drops substantially from 49.62 to 27.60.

### Relative Pose Estimation (Table 2 - ScanNet1500)

| Method | AUC@5 | AUC@10 | AUC@20 |
|--------|-------|--------|--------|
| DUSt3R | 31.61 | 53.77 | 70.99 |
| DUSt3R+Fin3R | **33.73** | **55.67** | **72.66** |
| MASt3R | 37.60 | 59.96 | 76.24 |
| MASt3R+Fin3R | **37.93** | **60.21** | **76.68** |
| VGGT | 28.40 | 47.36 | 61.51 |
| VGGT+Fin3R | **35.21** | **56.70** | **72.80** |

The improvement on VGGT is particularly pronounced (AUC@5: 28.40→35.21); after fine-tuning it surpasses the dedicated pose regression model Reloc3R (34.79) at the 5° threshold.

## Highlights & Insights

- **Minimal yet universal**: The same LoRA fine-tuning scheme directly accommodates four architecturally distinct feed-forward reconstruction models without any structural modification.
- **Negligible inference overhead**: Only lightweight LoRA weights are added; inference-time memory and latency are virtually unchanged.
- **Insight of re-normalization LoRA**: The paper identifies and quantifies the feature norm drift induced by monocular distillation and proposes a concise, effective remedy.
- **Consistent across tasks**: Monocular depth, relative pose, multi-view depth, and pointmap regression all improve consistently without degrading multi-view performance.

## Limitations & Future Work

- Errors in the teacher model MoGe propagate to the student, capping distillation quality at the teacher's performance ceiling.
- Although re-normalization is effective in most settings, the authors acknowledge it may not resolve all types of feature drift.
- Fine-tuning only the encoder leaves the decoder untouched, providing no benefit in scenarios where the decoder's matching capability is itself insufficient.
- Gains on dynamic scenes (e.g., Sintel) are relatively limited, likely because the baseline models were not trained on dynamic data.

## Related Work & Insights

- **Feed-forward 3D reconstruction**: DUSt3R pioneered direct pointmap regression from uncalibrated images; MASt3R adds a matching feature head; CUT3R adopts a recurrent architecture for long sequences; VGGT employs a fully parallel Transformer. Fin3R serves as a general-purpose fine-tuning framework for this family of models.
- **Monocular prior injection**: Align3R, Pow3R, and Mono3R inject external depth predictions into DUSt3R-style models but require additional runtime modules. Fin3R avoids inference overhead by performing distillation at training time.
- **Test-time optimization**: LoRA-3D and Test3R perform per-scene test-time fine-tuning and cannot generalize zero-shot. Fin3R requires only a single training run to achieve universal applicability.

## Rating

- Novelty: ⭐⭐⭐⭐ — The re-normalization LoRA insight is novel and well-supported by theory and experiments, though the overall framework remains a combination of knowledge distillation and LoRA.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers four baseline models, four evaluation tasks, and multiple datasets, with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is clear (the visualization of feature norm drift is particularly effective) and the writing is fluent.
- Value: ⭐⭐⭐⭐ — The method is simple, general, and highly practical, offering direct value to the feed-forward 3D reconstruction community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](../../CVPR2026/3d_vision/speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [\[NeurIPS 2025\] Learning Efficient Fuse-and-Refine for Feed-Forward 3D Gaussian Splatting](learning_efficient_fuse-and-refine_for_feed-forward_3d_gaussian_splatting.md)
- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](../../ICLR2026/3d_vision/splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)

</div>

<!-- RELATED:END -->
