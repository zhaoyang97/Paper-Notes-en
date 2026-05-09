---
title: >-
  [Paper Note] Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation
description: >-
  [AAAI 2026 (Student Abstract)][Medical Imaging][Semi-supervised segmentation] This paper proposes TGC (Topology Graph Consistency), a framework that introduces graph-theoretic topological constraints by aligning the Laplacian spectra, connected component counts, and adjacency statistics between prediction graphs and reference graphs. TGC achieves near-fully-supervised histopathology segmentation performance using only 5–10% of labeled data.
tags:
  - AAAI 2026 (Student Abstract)
  - Medical Imaging
  - Semi-supervised segmentation
  - histopathology images
  - topological consistency
  - graph-theoretic constraints
  - pseudo-labels
date: 2026-05-08
content_hash: 6dc31fb76a9b799c
---

# Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation

**Conference**: AAAI 2026 (Student Abstract)
**arXiv**: [2509.22689](https://arxiv.org/abs/2509.22689)
**Code**: None
**Area**: Medical Imaging
**Keywords**: Semi-supervised segmentation, histopathology images, topological consistency, graph-theoretic constraints, pseudo-labels

## TL;DR

This paper proposes TGC (Topology Graph Consistency), a framework that introduces graph-theoretic topological constraints by aligning the Laplacian spectra, connected component counts, and adjacency statistics between prediction graphs and reference graphs. TGC achieves near-fully-supervised histopathology segmentation performance using only 5–10% of labeled data.

## Background & Motivation

**Background**: Semantic segmentation is critical in computational pathology for accurately delineating glands, tubular structures, and other tissue regions from whole slide images (WSIs). However, dense pixel-level annotation in pathology is extremely costly, requiring expert pathologists to label images pixel by pixel—far more expensive than annotating natural images.

**Limitations of Prior Work**: Semi-supervised semantic segmentation (SSSS) methods typically rely on pixel-level consistency (e.g., mean teacher, consistency regularization) to exploit unlabeled data. In histopathology images, however, pixel-level consistency suffers from two critical problems: (1) noisy pseudo-labels are propagated and amplified during consistency training; and (2) generated segmentation masks are often fragmented or topologically invalid—e.g., glands are split into multiple disconnected fragments, or adjacent glands are erroneously merged.

**Key Challenge**: Pixel-level constraints focus on local accuracy while neglecting global topology. A segmentation mask may appear acceptable at the pixel level yet be entirely incorrect in terms of topological structure—such as producing disconnected glandular structures or spurious ring-shaped regions—whereas topological correctness is essential for downstream pathological analysis.

**Goal**: To introduce global topological constraints into semi-supervised histopathology segmentation, ensuring that segmentation results are not only pixel-accurate but also topologically correct.

**Key Insight**: The authors model segmentation predictions and reference annotations as graph structures, and then enforce topological consistency using graph-theoretic tools—specifically Laplacian spectra, connected components, and adjacency relations.

**Core Idea**: Replace (or supplement) pixel-level consistency with graph-theoretic constraints—learning topologically correct segmentation by aligning topological statistics between the prediction graph and the reference graph.

## Method

### Overall Architecture

TGC extends a standard semi-supervised segmentation framework (e.g., mean teacher). Labeled data is trained with standard supervised losses; for unlabeled data, graph-theoretic topological consistency constraints are introduced in addition to conventional pixel-level consistency. Each segmentation prediction is converted into a graph representation and then topologically aligned with the graph representation of the reference (teacher model prediction or ground truth label).

### Key Designs

1. **Prediction-to-Graph Conversion**:

    - **Function**: Converts segmentation masks into graph structures amenable to topological analysis.
    - **Mechanism**: Connected components extracted from segmentation masks serve as graph nodes, with edges established between adjacent components. Node attributes include component size and position; edge attributes include adjacency length and distance. Label masks and prediction masks are each converted into graphs, yielding a "reference graph" and a "prediction graph," respectively.
    - **Design Motivation**: Pixel-level representations cannot directly measure topological differences. Graph representations naturally encode topological information such as connectivity and adjacency relations.

2. **Laplacian Spectral Alignment**:

    - **Function**: Enforces similar global topological structure between the prediction graph and the reference graph.
    - **Mechanism**: The graph Laplacian $L = D - A$ (where $D$ is the degree matrix and $A$ is the adjacency matrix) is computed, and its eigenvalue spectrum is extracted. The Laplacian spectrum encodes global structural information—eigenvalues reflect connectivity and cluster structure. Topological alignment is achieved by minimizing the distance between the Laplacian spectra of the prediction graph and the reference graph.
    - **Design Motivation**: The Laplacian spectrum is a strong invariant for graph isomorphism; graphs with similar spectra share similar topological structures, providing richer topological information than simply comparing connected component counts.

3. **Connected Component and Adjacency Statistics Constraints**:

    - **Function**: Provides more intuitive topological constraints as a complement to spectral alignment.
    - **Mechanism**: In addition to spectral alignment, the framework constrains the number of connected components in the prediction graph to match the reference (preventing fragmentation or over-merging), and aligns adjacency statistics (e.g., average degree, maximum degree) with those of the reference. These simpler constraints provide interpretable topological regularization.
    - **Design Motivation**: While the Laplacian spectrum is comprehensive, its gradient signal may be weak. Simpler statistics such as connected component counts provide stronger and more direct constraints.

### Loss & Training

Total loss = standard supervised loss + pixel-level consistency loss + graph-theoretic topological consistency loss (Laplacian spectral distance + connected component count discrepancy + adjacency statistics discrepancy). Training follows the mean teacher framework, with the teacher model updated via EMA.

## Key Experimental Results

### Main Results

Evaluated on the GlaS and CRAG histopathology segmentation datasets.

| Dataset | Label Ratio | Metric | TGC | Prev. SOTA | Full Supervision | Notes |
|---------|-------------|--------|-----|------------|------------------|-------|
| GlaS | 5% | Dice/IoU | SOTA | -- | Upper bound | Small gap from full supervision |
| GlaS | 10% | Dice/IoU | SOTA | -- | Upper bound | Gap significantly reduced |
| CRAG | 5% | Dice/IoU | SOTA | -- | Upper bound | Consistently effective across datasets |
| CRAG | 10% | Dice/IoU | SOTA | -- | Upper bound | Near fully-supervised performance |

### Ablation Study

| Configuration | Dice | Notes |
|---------------|------|-------|
| TGC (Full) | Best | All topological constraints |
| Pixel consistency only | Degraded | Conventional semi-supervised baseline |
| + Connected component constraint | Improved | Reduces fragmentation |
| + Laplacian spectrum | Further improved | More accurate global topology |
| + Adjacency statistics | Full performance | Complementary multi-dimensional topological constraints |

### Key Findings

- TGC substantially narrows the performance gap with full supervision using only 5–10% labeled data, demonstrating that graph-theoretic topological constraints provide effective supervisory signals for unlabeled data.
- Laplacian spectral alignment contributes the most among all components, providing richer topological information than simple statistics.
- Topological correctness and pixel accuracy are not necessarily correlated in histopathology images—TGC produces topologically more plausible segmentations at equivalent Dice scores.
- The method is consistently effective across two distinct pathology datasets, indicating good generalizability.

## Highlights & Insights

- The **intersection of graph theory and semi-supervised learning** is elegantly formulated—introducing classical graph-theoretic tools (Laplacian spectrum) into semi-supervised segmentation as regularization represents a methodological innovation.
- **The practical significance of topological constraints** is especially pronounced in pathological diagnosis—fragmented gland segmentation leads to erroneous gland counts and morphological analyses.
- The method is **plug-and-play** and can be incorporated into any existing semi-supervised segmentation framework.

## Limitations & Future Work

- As a Student Abstract, experimental details and scale may be limited.
- Graph construction relies on binarization of intermediate segmentation results; the choice of binarization threshold may affect graph quality.
- Computing Laplacian eigenvalues may incur non-trivial computational costs for large graphs.
- The approach could be extended to more complex scenarios such as 3D segmentation and instance segmentation.

## Related Work & Insights

- **vs. Mean Teacher**: Mean Teacher relies solely on pixel-level consistency; TGC additionally introduces topological constraints.
- **vs. Topology-based loss methods (e.g., clDice)**: clDice focuses on centerline connectivity, whereas TGC employs a more comprehensive graph-theoretic topological description.
- **vs. Persistent homology methods**: Persistent homology can detect topological features but incurs high computational cost; the graph-theoretic approach is more lightweight and gradient-friendly.

## Rating

- Novelty: ⭐⭐⭐⭐ Graph-theoretic topological constraints for semi-supervised segmentation is a novel combination
- Experimental Thoroughness: ⭐⭐⭐ Limited by Student Abstract format, but core results are clear
- Writing Quality: ⭐⭐⭐⭐ Clearly presented within the constrained format
- Value: ⭐⭐⭐⭐ Practically valuable for histopathology segmentation with good generalizability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](../../NeurIPS2025/medical_imaging/match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](../../CVPR2026/medical_imaging/uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2026/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[AAAI 2026\] Bidirectional Channel-selective Semantic Interaction for Semi-Supervised Medical Segmentation](bidirectional_channel-selective_semantic_interaction_for_semi-supervised_medical.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing.md)

</div>

<!-- RELATED:END -->
