---
title: >-
  [Paper Note] SPROUT: Supervise Less, See More — Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting
description: >-
  [ICML 2026][Segmentation][Nuclear segmentation] SPROUT is the first completely training-free and zero-annotation framework for pathology nuclear segmentation. It leverages H&E staining priors to self-construct high-confi…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "Nuclear segmentation"
  - "SAM prompting"
  - "H&E staining prior"
  - "Partial Optimal Transport"
  - "Training-free"
date: 2026-05-08
content_hash: 7cc7ac3acdb45f6f
---

# SPROUT: Supervise Less, See More — Training-free Nuclear Instance Segmentation with Prototype-Guided Prompting

**Conference**: ICML 2026  
**arXiv**: [2511.19953](https://arxiv.org/abs/2511.19953)  
**Code**: https://github.com/Y-Research-SBU/SPROUT  
**Area**: Medical Imaging / Pathology / SAM Prompt Engineering  
**Keywords**: Nuclear segmentation, SAM prompting, H&E staining prior, Partial Optimal Transport, Training-free

## TL;DR
SPROUT is the first completely training-free and zero-annotation framework for pathology nuclear segmentation. It leverages H&E staining priors to self-construct high-confidence foreground/background regions on each slide → extracts prototypes → performs feature-prototype soft alignment via Partial Optimal Transport (POT) → outputs positive/negative point prompts for SAM. It achieves an AJI 8.2% higher than training-based methods on benchmarks like MoNuSeg.

## Background & Motivation

**Background**: Nuclear instance segmentation in pathology H&E slides is fundamental for cancer prognosis and diagnosis. Existing methods fall into four categories based on supervision levels: fully supervised (e.g., HoVer-Net, requiring dense annotation), semi-supervised, weakly supervised (e.g., points/voronoi), and self-supervised. Following the emergence of SAM, SAM-based approaches (MedSAM, PromptNucSeg, UN-SAM) have risen but mostly require fine-tuning or training a prompter.

**Limitations of Prior Work**: (1) Pathology images feature a narrow color spectrum, inconsistent staining, thousands of dense nuclei per patch, weak boundaries, and extremely expensive pixel-level annotations. (2) Zero-shot performance of SAM is poor due to the large distribution shift between the pathology domain and SA-1B. (3) Existing SAM-adapter methods still require medical annotations and training. (4) Reference-based training-free methods (Matcher, Bridge, SAT) rely on external reference images, which fail for dense small targets (thousands of nuclei per patch) because few-shot settings cannot find suitable references given the high variation in staining, density, and morphology.

**Key Challenge**: To perform nuclear segmentation without supervision or training, high-quality SAM prompts are required. High-quality prompts require image-reference semantic correspondence; however, stable external references are difficult to find in pathology, and external backbone (DINOv2, H-optimus-1) features are insufficiently precise. Traditional reference-based approaches fail to close the loop in pathology.

**Goal**: To develop a completely training-free and zero-external-reference framework that constructs reliable prompts from the image itself, allowing SAM to perform precise nuclear segmentation without any annotations or parameter updates.

**Key Insight**: Break away from the "external reference" framework. Use the biochemical priors of H&E staining (Hematoxylin stains nuclei dark blue/purple; Eosin stains cytoplasm pink) to perform color deconvolution and self-construct high-confidence foreground/background regions as "self-references." This self-reference strategy utilizes the physical properties of pathology staining to bypass the instability of external references.

**Core Idea**: Stain prior → self-reference mask → clustering prototypes → Partial Optimal Transport (POT) for feature-prototype alignment → SAM point prompts. The entire pipeline requires no training and no annotation.

## Method

### Overall Architecture

Three steps:
1. **Feature-Prototype Similarity Mapping**: Image patch encoding (DINOv2 or H-optimus-1) → stain decomposition (OD space + Otsu) → high-confidence foreground/background masks → K-means clustering to obtain prototypes $\mathcal{P}_{fg}, \mathcal{P}_{bg}$.
2. **POT-Scan + Active Prompting**: Progressive partial OT (starting with a small $\rho_0$ and gradually increasing) performs feature-prototype soft alignment, avoiding the instability of point-level hard matching and filtering ambiguous features.
3. **Instance Mask Prediction + Optimization**: Prototype re-weighted activation → binarization → watershed for positive points + dilated background for negative points → SAM inference → containment-aware NMS.

### Key Designs

1. **Self-reference via H&E Stain Prior**:

    - **Function**: Constructs high-confidence foreground/background regions from the image itself to replace external references.
    - **Mechanism**: OD space transformation $OD = -\log(x/x_0)$ → solve for concentration map $S = Q^+ \cdot OD$ using a normalized staining matrix $Q = [Q_H, Q_E]$ → Otsu thresholding for coarse foreground/background → select top-$t$ intensity pixels per region as high-confidence masks $\bm M_{fg}, \bm M_{bg}$ → cluster features within masked regions to obtain prototypes.
    - **Design Motivation**: The physical properties of pathology staining naturally separate nuclei (darkly stained) from extracellular space (lightly stained), making this more reliable and adaptive than external references for each slide. This solves the fundamental problem of reference-based methods in pathology.

2. **Partial Optimal Transport Scan (POT-Scan)**:

    - **Function**: Stably propagates prototype semantics to all features and filters noise without forcing all features to align.
    - **Mechanism**: Cost $C_{ij} = 1 - \tilde F P^\top / (\|\tilde F\|\|P\|)$; Partial OT allows a $1-\rho$ portion of features to remain unmatched: $\min_T \langle T, C\rangle_F + \lambda KL(T^\top \bm 1_N \| \tfrac{\rho}{M} \bm 1_M)$, s.t. $T \bm 1_M \leq \tfrac{1}{N}\bm 1_N$. A progressive approach increases $\rho$ from small to large—matching easy features first before incorporating difficult ones. An additional slack column converts the partial problem into a standard Sinkhorn solution.
    - **Design Motivation**: Standard OT forces total mass transport, causing noisy features to be incorrectly aligned with prototypes. Partial OT naturally filters ambiguous regions. Progressive scanning acts as "soft curriculum learning" to avoid noise amplification from early-stage ambiguous features.

3. **Active Prompting + Containment-aware NMS**:

    - **Function**: Converts alignment results into positive/negative point prompts for SAM and optimizes the output.
    - **Mechanism**: Prototype re-weighted activation $F^\star = \tilde F \odot T^\star$ → DenseCRF smoothing → threshold binarization → combined with initial high-confidence masks for watershed sampling of positive points (one per connected component); negative points are sampled uniformly from the dilated background mask. A stopping rule prevents the merging of distinct nuclei. Containment-aware NMS: uses stricter non-maximum suppression for SAM candidates exhibiting containment relationships.
    - **Design Motivation**: SAM is sensitive to the number and location of point prompts; watershed naturally provides "one point per nucleus." Containment-aware NMS addresses the issue of standard NMS erroneously deleting nested small nuclei in dense scenarios.

## Key Experimental Results

### Main Results: MoNuSeg and CPM17 (Comparison across supervision levels)

| Method | SAM | Supervision | MoNuSeg AJI↑ | MoNuSeg PQ↑ | CPM17 AJI↑ | CPM17 PQ↑ |
|------|----|----|------|------|------|------|
| U-Net | ✗ | Full | 0.421 | 0.403 | 0.477 | 0.435 |
| HoVer-Net | ✗ | Full | 0.589 | 0.510 | 0.617 | 0.547 |
| TopoSeg | ✗ | Full | 0.604 | 0.522 | 0.625 | 0.561 |
| Voronoi (Weakly) | ✗ | Weak | 0.501 | 0.443 | 0.531 | 0.475 |
| Self-supervised Baseline | ✗ | Self | 0.452 | 0.385 | 0.495 | 0.432 |
| MedSAM (fine-tuned) | ✓ | Full | 0.595 | 0.517 | 0.618 | 0.554 |
| PromptNucSeg | ✓ | Trained prompter | 0.610 | 0.531 | 0.627 | 0.563 |
| Matcher (Ref-based, Training-free) | ✓ | None | 0.523 | 0.456 | 0.548 | 0.482 |
| **SPROUT** | ✓ | **None** | **0.692** | **0.601** | **0.687** | **0.617** |

Ours (SPROUT) exceeds all training-based methods (including fully supervised TopoSeg) without any supervision or training, with an AJI 8.2% higher than PromptNucSeg.

### Robustness of POT-Scan Hyperparameters

| Configuration | AJI |
|------|------|
| $\rho_0 = 0.1, K = 8$ | 0.687 |
| $\rho_0 = 0.2, K = 8$ | **0.692** |
| $\rho_0 = 0.3, K = 8$ | 0.689 |
| $K = 4$ | 0.673 |
| $K = 16$ | 0.685 |

The AJI remains stable between 0.67-0.69 under perturbations of the main hyperparameters (initial transport ratio $\rho_0$, number of prototypes $K$).

### Ablation Study

| Configuration | AJI | Δ |
|------|------|---|
| Full SPROUT | 0.692 | – |
| Replace self-reference with external reference | 0.548 | −0.144 |
| Replace Partial OT with Standard OT | 0.621 | −0.071 |
| Replace progressive scan with single OT | 0.658 | −0.034 |
| Remove containment-aware NMS | 0.661 | −0.031 |

The self-reference strategy contributes the most (+0.144 AJI), proving that "internal staining priors are more reliable than external references" is the core innovation.

### Key Findings
- **Self-reference > External reference**: Self-constructed masks via staining priors are more accurate than any external reference because they adapt to the staining variations of each slide.
- **Partial OT is the key technology**: Standard OT amplifies noise by forcing full matching; Partial OT allows ambiguous regions to be excluded.
- **Training-free + Zero-label + SOTA**: Challenges the traditional assumption that models "must be trained" or "must be annotated."
- **Robust across datasets**: Consistent lead across MoNuSeg, CPM17, TNBC, and PanNuke datasets.

## Highlights & Insights
- **The "Image itself is the best reference" insight**: Solves the fundamental dilemma of reference-based pathology methods—pathology images vary too much for external references, but each image possesses internal physical consistency in staining. This logic can be extended to other medical imaging with strong physical priors (e.g., specific markers in fluorescence microscopy, tracer distribution in PET).
- **Correct implementation of "soft alignment" via Partial OT**: While previous OT-based feature alignment usually defaulted to total transport, this work treats "ignoring uncertain features" as a first-class citizen via partial and progressive scanning, which is applicable to many noise-sensitive tasks.
- **Completely training-free SOTA**: In a field where medical imaging heavily relies on annotations, this work proves that zero-annotation and zero-training methods can achieve SOTA, providing huge practical value for low-resource scenarios (underserved regions, rare diseases, new staining protocols).
- **A paradigm for SAM Prompt Engineering**: Using SAM as a general segmentor while injecting domain knowledge into prompt generation. This decoupled design allows foundation models and domain expertise each to do what they do best.

## Limitations & Future Work
- Dependence on the physical properties of H&E staining—other stains (IHC, Masson's Trichrome, etc.) would require rewritten stain decomposition; not directly applicable to non-H&E pathology (e.g., Electron Microscopy).
- SAM inference itself still incurs computational overhead; thousands of SAM calls in dense nuclear scenarios can be slow.
- Containment-aware NMS is a heuristic that might erroneously suppress nested nuclear structures (e.g., nucleoli within a nucleus).
- The self-reference strategy may fail on low-quality slides with extremely poor staining (over-exposed/under-stained); failure cases were not quantified.
- No head-to-head comparison directly against foundation models like H-optimus-1 (though it was used as a backbone).

## Related Work & Insights
- **vs. Full/Weak/Self-supervised Nuclear Segmentation (HoVer-Net, Voronoi, etc.)**: Those require training and annotation; SPROUT outperforms them at zero cost.
- **vs. SAM Pathology Fine-tuning (MedSAM, PromptNucSeg)**: Those require medical annotation and training; SPROUT uses the general SAM directly.
- **vs. Reference-based Training-free (Matcher, Bridge, SAT)**: Those require external references, which are unstable in pathology; SPROUT's self-reference is a breakthrough.
- **Insight**: Treat "domain physical prior → self-reference → foundation model prompting" as a general paradigm for zero-shot medical imaging. OT + Partial soft alignment is suitable for all "feature-prototype alignment + noise filtering" tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Stain prior self-reference + Partial OT" is a truly new training-free paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 datasets × multi-supervision baselines × detailed ablations × hyperparameter robustness; comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear framework; solid mathematical derivation for POT-Scan; provides theoretical guarantees (POT convergence proof in appendix).
- Value: ⭐⭐⭐⭐⭐ Pathology annotation is extremely expensive and variable; zero-annotation SOTA directly lowers the barrier for medical AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](../../CVPR2026/segmentation/insid3_training-free_in-context_segmentation_with_dinov3.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](../../ICCV2025/segmentation/training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](../../CVPR2026/segmentation/looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)

</div>

<!-- RELATED:END -->
