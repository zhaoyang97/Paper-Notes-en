---
title: >-
  [Paper Note] Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels
description: >-
  [CVPR2026][Segmentation][Topology-preserving segmentation] Ours proposes Same Class Neighbor Penalization (SCNP), which significantly improves the topological accuracy of segmentation at an extremely low cost (only 3 lines of code, a few milliseconds/iteration). By replacing each pixel's logit with its worst neighbor prediction within the same class during training, the model is forced to prioritize fixing weak pixels in the neighborhood.
tags:
  - "CVPR2026"
  - "Segmentation"
  - "Topology-preserving segmentation"
  - "Neighborhood Penalization"
  - "SCNP"
  - "Loss Function"
  - "Connected Components"
date: 2026-05-08
content_hash: cfc5ada707d043fb
---

# Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels

**Conference**: CVPR2026  
**arXiv**: [2603.18671](https://arxiv.org/abs/2603.18671)  
**Code**: [SCNP](https://jmlipman.github.io/SCNP-SameClassNeighborPenalization)  
**Area**: Semantic Segmentation / Topology Accuracy  
**Keywords**: Topology-preserving segmentation, Neighborhood Penalization, SCNP, Loss Function, Connected Components

## TL;DR

Ours proposes Same Class Neighbor Penalization (SCNP), which significantly improves the topological accuracy of segmentation at an extremely low cost (only 3 lines of code, a few milliseconds/iteration). By replacing each pixel's logit with its worst neighbor prediction within the same class during training, the model is forced to prioritize fixing weak pixels in the neighborhood.

## Background & Motivation

**Topology errors are ubiquitous**: Standard deep learning segmentation models perform independent pixel-wise inference, failing to guarantee topological correctness. This leads to broken tubular structures and isolated false positive regions, affecting downstream quantitative analysis (e.g., cell counting, road connectivity).

**High cost of Persistent Homology methods**: Topology losses based on Persistence Homology (TopoLoss, Betti Matching, etc.) require computing PH during training, causing training time to inflate from hours to days.

**Skeletonization methods are limited to tubular structures**: Skeleton-based losses such as clDice and SkelRecall are only applicable to tubular morphologies and do not work for non-tubular structures like cells, organs, or brain lesions.

**High memory overhead and tuning requirements for clDice**: The differentiable soft skeletonization technique in clDice consumes significant GPU memory and is sensitive to hyperparameters.

**Lack of a universal plug-and-play solution**: Existing methods either require special architectures/post-processing or are limited to specific morphologies. No CPU/GPU-efficient, general topology improvement method exists for various structural forms.

**Underutilized neighborhood info in small/thin structures**: Disconnected segments or false positive pixels are inevitably the worst-predicted pixels in their neighborhood; this prior is not explicitly utilized by existing losses.

## Method

### Overall Architecture

The goal is to solve the persistence of topological errors in segmentation models due to pixel-wise independent inference. SCNP is an extremely lightweight module placed between the logit output and the loss function: the model outputs logits $\mathbf{Z}$, SCNP rewrites them into penalized $\tilde{\mathbf{Z}}$, which is then fed into a standard loss $\mathcal{L}(\sigma(\tilde{\mathbf{Z}}), \mathbf{Y})$. It requires only 3 lines of code during training and remains unchanged during inference.

### Key Designs

**1. Same-Class Neighbor Penalization: Replacing each pixel with its worst same-class neighborhood prediction**

A common feature of breaks and false positives is that they are inevitably the worst-predicted pixels in the same-class neighborhood. SCNP explicitly utilizes this prior by modifying the logit $z_{ki}$ of each pixel $i$ for class $k$: if the pixel is foreground ($y_{ki}=1$), it is replaced by the minimum logit of its neighbors $\Omega(i)$ that are also foreground $\tilde{z}_{ki} = \min_{j \in \Omega(i), y_{kj}=1} z_{kj}$; if it is background ($y_{ki}=0$), it is replaced by the maximum logit of its neighbors that are also background $\tilde{z}_{ki} = \max_{j \in \Omega(i), y_{kj}=0} z_{kj}$. This replacement has three effects: since the logit is worsened, the loss increases; the worst pixel is penalized as many times as it is propagated to a neighborhood, forcing the model to fix it; and gradients become coupled across neighborhood pixels and classes, turning "fixing one weakness" into collaborative optimization.

**2. Implementation via MaxPool/MinPool: Neighborhood propagation in 3 lines of code**

While the min/max-over-neighbors might seem to require loops, it can be implemented with pooling operations. SCNP multiplies background logits by a large positive number $\kappa$ before MinPool (to prevent background from polluting foreground propagation) and multiplies foreground logits by a large negative number $-\kappa$ before MaxPool. This simultaneously obtains the neighborhood minimum for foreground and the neighborhood maximum for background. The only hyperparameter is the window size $w$ (default $w=3$, stride=1, padding to maintain size), adding only a few milliseconds per iteration and a few MiB of VRAM—whereas TopoLoss based on Persistence Homology slows down iterations from milliseconds to seconds.

### Loss & Training

SCNP is orthogonal to any loss function. The paper primarily uses $\mathcal{L}_{CEDice+\overline{CEDice}}$—calculating CE+Dice on both original logits and SCNP-penalized logits. Ablations show that integrating SCNP into 8 types of losses (CE, Dice, Tversky, clDice, SkelRecall, TopoLoss, Focal, RWLoss) is effective.

## Experiments

### Experimental Setup

- **Datasets**: 13 datasets covering 4 scenarios—① Medical Tubular (FIVES, Axons, PulmonaryVA), ② Non-medical Tubular (TopoMortar, DeepRoads, Crack500), ③ Medical Non-tubular (ATLAS2, ISLES24, CirrMRI600, MSLesSeg), ④ Medical Round Cells (IHC_TMA, LyNSeC, NuInsSeg).
- **Frameworks**: nnUNetv2 (medical semantic segmentation), Detectron2/DeepLabv3+ (non-medical semantic segmentation), InstanSeg (cell instance segmentation).
- **Metrics**: Dice, $\beta_{0e}$ (Betti error, difference in connected components), clDice (tubular), Roundness (cells).

### Main Results

| Dataset Group | SCNP Performance | Key Findings |
|---|---|---|
| ① Medical Tubular (3) | 3/3 Lowest $\beta_{0e}$ | Dice/clDice maintained; outperforms all topology losses. |
| ② Non-medical Tubular (3) | 2/3 Lowest $\beta_{0e}$ | Comprehensive lead in TopoMortar and Crack500; DeepRoads shows better topology but slight Dice drop. |
| ③ Medical Non-tubular (4) | 1/4 Significantly effective | $\beta_{0e}$ halved on CirrMRI600; harmful on MSLesSeg (extremely small structures). |
| ④ Medical Cells (3) | 2/3 Lowest $\beta_{0e}$ | Roundness improved across all datasets. |

### Ablation Study

Integrating SCNP into 8 loss functions on the FIVES dataset: **$\beta_{0e}$ decreased for all losses**, while Dice and clDice either improved or remained stable. Typical improvements:

| Loss Function | $\beta_{0e}$ (Original) | $\beta_{0e}$ (+SCNP) |
|---|---|---|
| CE | 11.93 | **7.53** |
| Dice | 12.03 | **7.88** |
| clDice | 36.55 | **5.44** |
| SkelRecall | 12.45 | **5.07** |
| Focal | 16.08 | **7.75** |

### Key Findings

- **Hyperparameter Sensitivity**: The optimal window $w$ correlates with the thickness of tubular structures (when median vessel thickness is ~9.7 pixels, $w=9$ is optimal), but the default $w=3$ is effective enough for most scenarios.
- **Computational Efficiency**: SCNP adds only a few milliseconds/iteration and a few MiB of VRAM, whereas TopoLoss increases iteration time by orders of magnitude.
- **Failure Scenarios**: SCNP is harmful on extremely small structures (MSLesSeg, average only 447 voxels). It is hypothesized that tiny structures with low contrast are unsuitable for the neighborhood smoothing effect.

## Highlights & Insights

- **Minimalist Design**: Only 3 lines of code and 1 intuitive hyperparameter; plug-and-play for any segmentation framework and loss function.
- **High Universality**: Validated across 13 datasets, 3 frameworks, and 8 loss functions, covering tubular, non-tubular, and cell morphologies.
- **Clear Theoretical Explanation**: Strictly analyzes from a gradient perspective how SCNP couples neighborhood gradients and why it focuses on the worst predictions.
- **Efficiency**: Training efficiency is improved by orders of magnitude compared to PH-based methods; no morphological limitations compared to skeleton-based methods.

## Limitations & Future Work

- Performance is unstable or even harmful in extremely small structures and low-contrast scenarios (e.g., MSLesSeg).
- While the default $w=3$ is general, there is room for tuning in tubular structures; optimal $w$ requires prior knowledge.
- Focuses only on $\beta_0$ (connected components) topological errors; topological preservation for $\beta_1$ (holes) and $\beta_2$ (cavities) has not been deeply verified.
- Cannot fully replace post-processing: while it reduces topological errors, post-processing is still needed for perfect topology.
- Relies on ground truth foreground/background masks for mask pooling during training; not suitable for unlabeled or extremely noisy labels.

## Related Work & Insights

- **PH-based Topology Losses**: TopoLoss [Hu+ NeurIPS'19], Betti Matching [Stucki+ ECCV'22]—Accurate but extremely slow.
- **Skeleton-based Topology Losses**: clDice [Shit+ CVPR'21], SkelRecall [Kirchhoff+ ECCV'24]—Efficient but limited to tubular structures.
- **Neighborhood-aware Methods**: Max Pooling Loss [Rota Bulo+ CVPR'17] (amplifies worst misclassifications), NeighborLoss [Yuan & Xu] (penalizes based on number of different-class neighbors but does not consider GT).
- **Boundary/Distance Weighted Losses**: Boundary Loss [Kervadec+ MIDL'19], RWLoss—None directly optimize topology.
- The core advantage of SCNP lies in its orthogonality to loss functions, lack of morphological constraints, and negligible computational cost.

## Rating

- Novelty: ⭐⭐⭐⭐ — Improves topology from a simple "worst neighborhood propagation" perspective; elegant and novel principle.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 13 Datasets × 3 Frameworks × 8 Losses; detailed ablation and sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete theoretical derivation, and concise algorithm pseudocode.
- Value: ⭐⭐⭐⭐ — A highly practical 3-line plug-and-play solution for topology improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hilbert Curve-Based Attention Enabling Topology-Preserving Image Tensor Representation for Semantic Segmentation Network](hilbert_curve-based_attention_enabling_topology-preserving_image_tensor_represen.md)
- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](../../CVPR2025/segmentation/the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[CVPR 2026\] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)
- [\[CVPR 2026\] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)
- [\[CVPR 2026\] High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy](high-precision_dichotomous_image_segmentation_via_depth_integrity-prior_and_fine.md)

</div>

<!-- RELATED:END -->
