---
title: >-
  [Paper Note] Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels
description: >-
  [CVPR2026][Segmentation][Topology-preserving segmentation] This paper proposes Same Class Neighbor Penalization (SCNP), which replaces each pixel's logit with the worst prediction among its same-class neighbors during tr…
tags:
  - "CVPR2026"
  - "Segmentation"
  - "Topology-preserving segmentation"
  - "neighbor penalization"
  - "SCNP"
  - "loss function"
  - "connected components"
date: 2026-05-08
content_hash: f1ce0e86c48ee775
---

# Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels

**Conference**: CVPR2026
**arXiv**: [2603.18671](https://arxiv.org/abs/2603.18671)  
**Code**: [SCNP](https://jmlipman.github.io/SCNP-SameClassNeighborPenalization)  
**Area**: Semantic Segmentation / Topology Accuracy
**Keywords**: Topology-preserving segmentation, neighbor penalization, SCNP, loss function, connected components

## TL;DR

This paper proposes Same Class Neighbor Penalization (SCNP), which replaces each pixel's logit with the worst prediction among its same-class neighbors during training, thereby forcing the model to prioritize correcting weakly classified pixels within local neighborhoods. This approach achieves significant improvements in topological accuracy at negligible cost (only 3 lines of code and a few milliseconds per iteration).

## Background & Motivation

**Topological errors are pervasive**: Standard deep learning segmentation models perform per-pixel independent inference, offering no guarantee of topological correctness. This leads to broken tubular structures and isolated false-positive regions, which adversely affect downstream quantitative analysis (e.g., cell counting, road connectivity).

**Persistent homology methods are computationally expensive**: Topology-aware losses based on Persistence Homology (PH), such as TopoLoss and Betti Matching, require PH computation during training, inflating training time from hours to days.

**Skeletonization methods are limited to tubular structures**: Skeleton-based losses such as clDice and SkelRecall are applicable only to tubular morphologies and are unsuitable for non-tubular structures like cells, organs, or brain lesions.

**clDice incurs high memory overhead and requires hyperparameter tuning**: The differentiable soft-skeletonization technique in clDice consumes substantial GPU memory and its performance is sensitive to hyperparameter settings.

**Lack of a universal plug-and-play solution**: Existing methods either require specialized architectures or post-processing, or are restricted to specific morphologies. No efficient, morphology-agnostic approach for improving topological accuracy currently exists.

**Neighborhood information around small structures and thin boundaries is underutilized**: Segmentation breaks and false-positive pixels are necessarily the worst-predicted pixels within their local neighborhoods—a prior that no existing loss explicitly exploits.

## Method

### Overall Architecture

SCNP is inserted as a lightweight post-processing module between the logit output and the loss function: the model outputs logits $\mathbf{Z}$ → SCNP generates penalized logits $\tilde{\mathbf{Z}}$ → standard loss $\mathcal{L}(\sigma(\tilde{\mathbf{Z}}), \mathbf{Y})$. Only 3 additional lines of code are required during training, with no modification at inference time.

### Key Designs: Same-Class Neighbor Penalization

For the logit $z_{ki}$ of pixel $i$ under class $k$, SCNP is defined as:

- **Foreground class** ($y_{ki}=1$): take the minimum logit among foreground neighbors in $\Omega(i)$ → $\tilde{z}_{ki} = \min_{j \in \Omega(i), y_{kj}=1} z_{kj}$
- **Background class** ($y_{ki}=0$): take the maximum logit among background neighbors → $\tilde{z}_{ki} = \max_{j \in \Omega(i), y_{kj}=0} z_{kj}$

This yields three effects: (1) the loss increases because logits are degraded; (2) the worst-predicted pixel is penalized multiple times—once for each neighborhood it propagates into; (3) gradients become coupled across neighboring pixels and across classes.

### Efficient Implementation

SCNP is implemented via MaxPool and MinPool operations: background logits are multiplied by a large positive scalar $\kappa$ before MinPool (to exclude them from foreground propagation), and foreground logits are multiplied by $-\kappa$ before MaxPool. The window size $w$ is the sole hyperparameter, defaulting to $w=3$, with stride=1 and padding to preserve spatial dimensions.

### Loss & Training

SCNP is compatible with any loss function. The paper primarily employs $\mathcal{L}_{CEDice+\overline{CEDice}}$, which simultaneously optimizes CE+Dice losses on both the standard logits and the SCNP-penalized logits. Ablation studies confirm that SCNP can be integrated effectively into 8 loss functions: CE, Dice, Tversky, clDice, SkelRecall, TopoLoss, Focal, and RWLoss.

## Key Experimental Results

### Experimental Setup

- **Datasets**: 13 datasets spanning 4 categories — ① medical tubular (FIVES, Axons, PulmonaryVA), ② non-medical tubular (TopoMortar, DeepRoads, Crack500), ③ medical non-tubular (ATLAS2, ISLES24, CirrMRI600, MSLesSeg), ④ medical circular cells (IHC_TMA, LyNSeC, NuInsSeg)
- **Frameworks**: nnUNetv2 (medical semantic segmentation), Detectron2/DeepLabv3+ (non-medical semantic segmentation), InstanSeg (cell instance segmentation)
- **Metrics**: Dice, $\beta_{0e}$ (Betti error, difference in connected components), clDice (tubular), Roundness (cells)

### Main Results

| Dataset Group | SCNP Effect | Key Findings |
|---|---|---|
| ① Medical tubular (3) | Lowest $\beta_{0e}$ on 3/3 | No degradation in Dice/clDice; outperforms all topology losses |
| ② Non-medical tubular (3) | Lowest $\beta_{0e}$ on 2/3 | Leads on TopoMortar and Crack500; DeepRoads shows topological gains with slight Dice drop |
| ③ Medical non-tubular (4) | Significantly effective on 1/4 | $\beta_{0e}$ halved on CirrMRI600; harmful on MSLesSeg (very small structures) |
| ④ Medical cells (3) | Lowest $\beta_{0e}$ on 2/3 | Roundness improves across all datasets |

### Ablation Study

SCNP is integrated into 8 loss functions on the FIVES dataset: **$\beta_{0e}$ decreases for all losses**, with no degradation in Dice or clDice. Representative improvements:

| Loss Function | $\beta_{0e}$ (baseline) | $\beta_{0e}$ (+SCNP) |
|---|---|---|
| CE | 11.93 | **7.53** |
| Dice | 12.03 | **7.88** |
| clDice | 36.55 | **5.44** |
| SkelRecall | 12.45 | **5.07** |
| Focal | 16.08 | **7.75** |

### Key Findings

- **Hyperparameter sensitivity**: The optimal window size $w$ correlates with the thickness of tubular structures (median vessel thickness ~9.7 pixels yields $w=9$ as optimal), but the default $w=3$ is sufficient in the vast majority of scenarios.
- **Computational efficiency**: SCNP adds only a few milliseconds per iteration and a few MiB of GPU memory, whereas TopoLoss extends iteration time from milliseconds to several seconds.
- **Failure cases**: SCNP is harmful for extremely small structures (MSLesSeg, mean 447 voxels), presumably because the neighborhood smoothing effect is inappropriate for tiny, low-contrast structures.

## Highlights & Insights

- **Minimal design**: Only 3 lines of code and 1 intuitive hyperparameter, plug-and-play into any segmentation framework and loss function.
- **Strong generalizability**: Validated across 13 datasets, 3 frameworks, and 8 loss functions, covering tubular, non-tubular, and cellular morphologies.
- **Clear theoretical justification**: A rigorous gradient-based analysis explains how SCNP couples neighborhood gradients and focuses optimization on the worst-predicted pixels.
- **High efficiency**: Orders-of-magnitude faster than PH-based methods; no morphological constraints compared to skeleton-based approaches.

## Limitations & Future Work

- Performance is unstable or even detrimental for extremely small and low-contrast structures (e.g., MSLesSeg, mean 447 voxels).
- Although the default $w$ is broadly applicable, further tuning on tubular structures requires prior knowledge of structure thickness.
- The method addresses only $\beta_0$ (connected component) topological errors; preservation of $\beta_1$ (loops) and $\beta_2$ (voids) topology has not been thoroughly validated.
- SCNP cannot fully replace post-processing: it reduces topological errors but perfect topological correctness still requires post-processing.
- During training, the method relies on ground-truth foreground/background masks for masked pooling, making it unsuitable for unannotated or highly noisy label settings.

## Related Work & Insights

- **PH-based topology losses**: TopoLoss [Hu+ NeurIPS'19], Betti Matching [Stucki+ ECCV'22] — precise but extremely slow.
- **Skeletonization-based topology losses**: clDice [Shit+ CVPR'21], SkelRecall [Kirchhoff+ ECCV'24] — efficient but restricted to tubular structures.
- **Neighborhood-aware methods**: Max Pooling Loss [Rota Bulo+ CVPR'17] (amplifies the worst misclassifications), NeighborLoss [Yuan & Xu] (penalizes based on number of differently-labeled neighbors, but ignores ground truth).
- **Boundary/distance-weighted losses**: Boundary Loss [Kervadec+ MIDL'19], RWLoss — neither directly optimizes topology.
- SCNP's core advantages lie in being orthogonal to the choice of loss function, morphology-agnostic, and computationally negligible.

## Rating

- Novelty: ⭐⭐⭐⭐ — Improving topology through the elegant lens of "worst-neighbor propagation"; the idea is original and conceptually clean.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 13 datasets × 3 frameworks × 8 loss functions, with comprehensive ablation and sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete theoretical derivations, and concise algorithmic pseudocode.
- Value: ⭐⭐⭐⭐ — A plug-and-play, 3-line topology improvement solution with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)
- [\[CVPR 2026\] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)
- [\[ICML 2026\] Segment Anything with Robust Uncertainty-Accuracy Correlation](../../ICML2026/segmentation/segment_anything_with_robust_uncertainty-accuracy_correlation.md)
- [\[ICCV 2025\] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](../../ICCV2025/segmentation/topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)
- [\[NeurIPS 2025\] SRSR: Enhancing Semantic Accuracy in Real-World Image Super-Resolution with Spatially Re-Focused Text-Conditioning](../../NeurIPS2025/segmentation/srsr_enhancing_semantic_accuracy_in_real-world_image_super-resolution_with_spati.md)

</div>

<!-- RELATED:END -->
