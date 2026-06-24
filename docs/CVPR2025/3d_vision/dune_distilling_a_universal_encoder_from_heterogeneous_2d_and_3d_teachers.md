---
title: >-
  [Paper Note] DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers
description: >-
  [3D Vision] DUNE proposes a co-distillation framework for heterogeneous teachers, unifying 2D (DINOv2) and 3D (MASt3R, Multi-HMR) teacher models from different tasks and data domains into a single ViT-Base universal encoder. It matches or exceeds the performance of their respective ViT-Large teachers across multiple tasks such as semantic segmentation, depth estimation, 3D reconstruction, and human pose recovery.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: bc1e1c4296d634b2
---

# DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers

## TL;DR

DUNE proposes a co-distillation framework for heterogeneous teachers, unifying 2D (DINOv2) and 3D (MASt3R, Multi-HMR) teacher models from different tasks and data domains into a single ViT-Base universal encoder. It matches or exceeds the performance of their respective ViT-Large teachers across multiple tasks such as semantic segmentation, depth estimation, 3D reconstruction, and human pose recovery.

## Background & Motivation

Existing multi-teacher distillation methods (e.g., AM-RADIO, UNIC, Theia) have successfully unified multiple vision foundation models into a single encoder. However, these methods only distill **homogeneous teachers**—namely, self-supervised models trained on web-scraped general data (DINOv2, CLIP, SAM). In this scenario, even using ImageNet-1K is sufficient to match the teachers' performance.

However, when the teacher pool contains **highly specialized** models (such as MASt3R specialized in 3D scene reconstruction and Multi-HMR specialized in human mesh recovery), the problem becomes entirely different:

1. **Task Heterogeneity**: The training objectives of the teachers vary significantly—DINOv2 learns general representations, MASt3R learns dense matching, and Multi-HMR learns SMPL parameters.
2. **Data Domain Heterogeneity**: The training data distributions are completely different, ranging from web-crawled natural images to synthetic 3D data, CAD models, and human scans.
3. **Discrepancy in Coding Patterns**: Different teachers encode information patterns differently within patch features (e.g., Multi-HMR encodes the entire human pose within head patches).

The core question is: can a **universal vision encoder** that excels in both 2D and 3D tasks be distilled from such a heterogeneous set of teachers?

## Method

### Overall Architecture

DUNE is based on a standard multi-teacher distillation framework: the output of the student ViT-Base encoder $f$ is mapped via teacher-specific projectors $h_i$, and then aligned with the outputs of each teacher encoder using cosine similarity loss and smooth-$\ell_1$ loss. After distillation, the projectors are discarded, keeping only the encoder, and the decoding heads for each task are then fine-tuned.

### Key Designs

#### 1. Transformer Projector (TP)

- **Function**: Captures teacher-specific inter-patch interaction patterns to replace traditional per-patch MLP projectors.
- **Mechanism**: The attention patterns of different teachers vary significantly (MASt3R is highly localized, DINOv2 has a wide attention span, and Multi-HMR focuses on the human head), requiring the projector to model cross-patch interactions. The TP consists of a single Transformer block containing a self-attention layer and an MLP, followed by a linear projection via a residual connection.
- **Design Motivation**: Standard MLP projectors can only operate patch-by-patch and cannot explicitly model inter-patch interactions, forcing all teacher-specific spatial interaction patterns to be fully borne by the shared encoder. TP distributes this burden to the projectors, allowing the encoder to focus more on learning universal features. Experiments demonstrate that TP outperforms LP and SP on all tasks.

#### 2. Heterogeneous Data Sharing Strategy

- **Function**: Decides which data is fed into which teacher's projector during distillation.
- **Mechanism**: Explores three strategies—no sharing (each projector only uses data corresponding to its teacher), full sharing (all data is fed into all projectors), and general data sharing (each projector uses corresponding data plus ImageNet). Experiments show that **full sharing** achieves the best performance.
- **Design Motivation**: Training data domains of heterogeneous teachers differ significantly, and intuitively, out-of-domain data might be harmful. However, experiments demonstrate that teachers can still produce useful signals for out-of-domain images, and full sharing provides the encoder with more learning signals. Interestingly, sharing only general data is optimal for semantic segmentation, suggesting that semantic information is better preserved when ImageNet is processed by 3D teachers.

#### 3. Discarding Projectors + Fine-tuning Decoding Heads at Inference

- **Function**: Achieves efficient inference, preventing the parameter size from growing linearly with the number of teachers during inference.
- **Mechanism**: After distillation, all projectors are discarded, and each teacher's decoder module is attached to the frozen encoder for individual fine-tuning. Consequently, inference only requires a single ViT-Base encoder and task-specific decoders.
- **Design Motivation**: Existing methods (such as AM-RADIO, Theia) need to retain projectors during inference to reuse teacher decoders, which increases parameter size and memory footprint with the number of teachers. Although fine-tuning decoding heads incurs a one-time cost, it introduces no extra modules during inference, keeping the encoder size and memory consumption constant.

### Loss & Training

The distillation loss is the sum of the cosine similarity loss and the smooth-$\ell_1$ loss across all teachers:

$$\mathcal{L}_{\text{distil}} = \sum_{i=1}^{N} \mathcal{L}_{cos}(f_i(x), t_i(x)) + \mathcal{L}_{s\ell_1}(f_i(x), t_i(x))$$

where $f_i = h_i(f(x))$, and teacher dropping regularization from UNIC is also used to prevent overfitting to a single teacher.

## Key Experimental Results

### Main Results (Tab. 3)

| Model | Encoder | ADE20K (mIoU↑) | NYUd (RMSE↓) | BEDLAM PA-PVE↓ | MapFree AUC↑ |
|------|--------|---------------|-------------|----------------|-------------|
| DINOv2 Teacher | ViT-L | 47.7 | 0.384 | - | - |
| Multi-HMR Teacher | ViT-L | - | - | 36.9 | - |
| MASt3R Teacher | ViT-L | - | - | - | 91.2 |
| DINOv2 | ViT-B | 47.3 | 0.399 | 76.5 | 89.6 |
| AM-RADIO-v2.5 | ViT-B | **50.0** | 0.718 | 83.2 | 93.1 |
| **DUNE (336)** | ViT-B | 44.9 | 0.377 | 68.3 | 93.7 |
| **DUNE (448)** | ViT-B | 45.6 | **0.358** | **56.0** | **94.7** |

### Ablation Study (Tab. 1 & 2)

**Projector design ablation** (using all data):

| Projector | ADE20K | NYUd RMSE | MapFree AUC | BEDLAM PA-PVE |
|--------|--------|-----------|-------------|---------------|
| SP | 42.3 | 0.413 | 92.2 | 73.1 |
| LP | 44.7 | 0.384 | 91.5 | 78.2 |
| **TP** | **44.9** | **0.377** | **93.7** | **68.3** |

**Data sharing strategy ablation**:

| Strategy | ADE20K | NYUd RMSE | MapFree AUC | BEDLAM PA-PVE |
|------|--------|-----------|-------------|---------------|
| No sharing | 41.6 | 0.426 | 93.2 | 68.7 |
| General data sharing | 40.1 | 0.416 | 92.7 | 71.7 |
| **Full sharing** | **44.9** | **0.377** | **93.7** | **68.3** |

### Key Findings

1. **ViT-Base Surpasses ViT-Large**: DUNE (448) achieves an AUC of 94.7% on Map-Free visual relocalization, surpassing MASt3R ViT-Large's 91.2%. On human mesh recovery, its PA-PVE of 56.0 is also significantly better than the baseline (note that PA-PVE is an error metric; here DUNE is still higher than the teacher, but with a much smaller parameter size).
2. Distillation using only ImageNet is insufficient—using all 19 heterogeneous datasets significantly improves performance on all tasks.
3. The TP projector consistently outperforms LP and SP on all tasks.

## Highlights & Insights

1. **First to define the heterogeneous teacher distillation problem**: Scaling multi-teacher distillation from "homogeneous foundation model fusion" to "heterogeneous model unification across tasks and data domains" represents a significant concept upgrade.
2. **Surprise of smaller models outperforming larger ones**: The ViT-Base encoder outperforms the ViT-Large teacher on Map-Free relocalization, indicating that the complementarity of multi-teacher signals can compensate for the model capacity gap.
3. **Full data sharing outperforms isolation**: Counter-intuitively, out-of-domain data is not harmful but beneficial, implying that heterogeneous teachers can still provide effective supervision signals for out-of-domain images.
4. **Transformer projector design is simple and effective**: A single Transformer block is sufficient to capture teacher-specific patch interaction patterns, being more efficient than multi-level LPs.

## Limitations & Future Work

1. **Insufficient semantic segmentation performance**: DUNE (44.9) on ADE20K is significantly lower than AM-RADIO-v2.5 (50.0), as the latter distills CLIP and SAM, which are highly semantic teachers.
2. **Lack of systematic guidance on teacher selection**: Currently, only 3 teachers have been experimented with; how to choose the optimal combination of teachers to maximize universality remains unexplored.
3. **Computational overhead**: The distillation stage requires running forward passes for all teachers, and training on 20.7 million images across 19 datasets incurs non-negligible costs.
4. Fine-tuning different decoding heads is required for each task at inference, failing to achieve true multi-task output with a single forward pass.

## Related Work & Insights

- **AM-RADIO / UNIC / Theia**: Precursor works to homogeneous teacher distillation, upon which DUNE extends to heterogeneous scenarios.
- **MASt3R**: Foundation model for 3D scene reconstruction, serving as the 3D teacher for DUNE.
- **Multi-HMR**: Human mesh recovery model, serving as DUNE's human understanding teacher.
- Insight: Multi-teacher distillation might be an effective path toward building "all-round vision encoders." Future work could introduce more professional teachers (e.g., medical imaging, remote sensing).

## Rating: ⭐⭐⭐⭐

The problem definition is clear and important, and the experimental design is systematic and comprehensive (covering projectors, data sharing, and multi-task evaluation). The results of ViT-Base outperforming ViT-Large are impressive. One star is deducted due to the gap in semantic segmentation performance compared to SOTA, and the lack of scaling experiments with more teachers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)
- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)
- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)
- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
