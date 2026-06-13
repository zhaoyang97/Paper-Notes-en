---
title: >-
  [Paper Note] PartCo: Part-Level Correspondence Priors Enhance Category Discovery
description: >-
  [ICML 2026][Self-Supervised Learning][Category Discovery] PartCo introduces a **plug-and-play** framework to enhance Generalized Category Discovery by explicitly leveraging **part-level feature correspondences** inherent…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Category Discovery"
  - "Part-Level Correspondence"
  - "ViT Features"
  - "Part-Level Contrastive Learning"
date: 2026-05-08
content_hash: 1d930662cc0a78b9
---

# PartCo: Part-Level Correspondence Priors Enhance Category Discovery

**Conference**: ICML 2026  
**arXiv**: [2509.22769](https://arxiv.org/abs/2509.22769)  
**Code**: To be confirmed  
**Area**: Self-Supervised Learning / Open-World Vision  
**Keywords**: Category Discovery, Part-Level Correspondence, ViT Features, Part-Level Contrastive Learning

## TL;DR
PartCo introduces a **plug-and-play** framework to enhance Generalized Category Discovery by explicitly leveraging **part-level feature correspondences** inherent in Vision Transformer patch tokens—improving baselines such as SimGCD, SPTNet, and FlipClass by 2-10% across multiple benchmarks including CUB, Stanford-Cars, and ImageNet-100.

## Background & Motivation

**Background**: Generalized Category Discovery (GCD) aims to identify both known and novel categories in unlabeled data by utilizing a small set of labeled samples from known categories.

**Limitations of Prior Work**: Existing GCD methods primarily rely on global image representations (such as the [CLS] token of Transformers), which capture global semantic information but abstract away fine-grained part-level details—leading to poor performance when distinguishing highly similar categories.

**Key Challenge**: Patch tokens in ViT models contain rich part-level semantic information, but direct utilization faces three major challenges: (1) lack of explicit part-level semantic labels; (2) confusion from foreground-background noise; (3) variations in scale and orientation of objects across samples.

**Goal**: Automatically extract part-level correspondence labels from ViT patch tokens and use them as supervisory signals to guide feature learning.

**Key Insight**: Patch token features from self-supervised foundation models (especially DINOv2) naturally contain part-level correspondence information—rather than letting the model learn parts from scratch, it is more effective to explicitly construct part-level labels to guide feature alignment.

**Core Idea**: Extract object regions and fine-grained features from frozen DINO model patch tokens via two-stage PCA projection, generate part-level labels using k-means clustering, and subsequently design a corresponding contrastive loss function.

## Method

### Overall Architecture
Two phases—**Offline Phase**: Use a frozen pre-trained DINO model to automatically generate part-level correspondence labels on a subset of labeled data via two-step PCA projection; **Training Phase**: Aggregate ViT patch features based on part-level labels, and introduce a part-level correspondence loss for joint optimization with GCD baseline losses.

### Key Designs

1.  **Two-stage PCA Projection + Part Label Construction**:
    - **Function**: Automatically discover object parts from ViT patch tokens and generate correspondence labels.
    - **Mechanism**: The first-stage PCA applies PCA to all patch features $\mathbf{F} \in \mathbb{R}^{M \times N \times d}$ to obtain the maximum variance direction $\mathbf{w}_{\text{obj}}$, computes the objectness score $\mathbf{F}_{\text{obj}} = \mathbf{F} \cdot \mathbf{w}_{\text{obj}}$, and generates a foreground mask $\mathbf{M}$ using a threshold $\tau_{\text{obj}} = 0.6$. The second-stage PCA applies PCA to masked features $\mathbf{F} \odot \mathbf{M}$ to extract the first three principal components $\mathbf{F}_{\text{fg}}$. Adaptive k-means automatically selects the optimal number of clusters by maximizing the product of cluster center distances and cluster size balance.
    - **Design Motivation**: Avoid manual part annotation by automatically obtaining supervisory signals through the inherent structure of frozen foundation models; use first-stage for fine-grained datasets (sufficient) and second-stage for general datasets (requiring more detail).

2.  **Part-Level Feature Aggregation + Correspondence Loss**:
    - **Function**: Aggregate patch features by part groups and design a part-level contrastive loss to promote intra-class compactness of the same parts and inter-class separation of different parts.
    - **Mechanism**: Compute the average feature $\mathbf{f}_c$ for each part category $c \in \mathcal{C}$; project to the contrastive learning space $\mathbf{h}_c = \psi_p(\mathbf{f}_c)$ via a part projection head $\psi_p$. Supervised part contrastive loss: $\mathcal{L}_{\text{pc}}^{\text{sup}} = \frac{1}{|B_l|} \sum_i \frac{1}{|\mathcal{C}|} \sum_c \frac{1}{|\mathbb{N}_i^c|} \sum_q -\log \frac{\exp(\mathbf{h}_c \cdot \mathbf{h}_q / \tau_r)}{\sum_{j \notin \mathbb{N}_i^c} \exp(\mathbf{h}_c \cdot \mathbf{h}_j / \tau_r)}$. The unsupervised version replaces ground truth labels with pseudo-labels.
    - **Design Motivation**: Force the model to learn part correspondences across samples through explicit part-level constraints; capture subtle visual structural differences compared to relying solely on global features.

3.  **Plug-and-Play Integration Strategy**:
    - **Function**: Seamlessly integrate the PartCo framework into any existing GCD method by only adding the part-level loss.
    - **Mechanism**: Total loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{gcd}} + \mathcal{L}_{\text{pc}}$; the balancing factor $\lambda_b = 0.35$ is validated as optimal through experiments.
    - **Design Motivation**: No modification to the original method's pipeline, only stacking new constraints at the loss function level; any GCD baseline can benefit.

## Key Experimental Results

### Main Results

| Method | Dataset | All ACC | Old ACC | New ACC | Gain Over Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SimGCD | CUB | 71.5% | 78.1% | 68.3% | Baseline |
| **PartCo-SimGCD** | CUB | **81.1%** | 82.4% | 80.5% | **+9.6%** |
| SPTNet | CUB | 76.3% | 79.5% | 74.6% | Baseline |
| **PartCo-SPTNet** | CUB | **82.6%** | 82.3% | 81.8% | **+6.3%** |
| FlipClass | CUB | 79.3% | 80.7% | 78.5% | Baseline |
| **PartCo-FlipClass** | CUB | **85.2%** | 86.3% | 84.7% | **+5.9%** |
| SelEx | CUB | 87.4% | 85.1% | 88.5% | Baseline |
| **PartCo-SelEx** | CUB | **90.6%** | 84.5% | 93.2% | **+3.2%** |

**Average Results on SSB Fine-Grained Benchmarks** (DINOv2): SimGCD 69.0% → 78.8% (+9.8%); SPTNet 72.2% → 80.4% (+8.2%); FlipClass 76.1% → 80.9% (+4.8%); SelEx 83.1% → 85.5% (+2.4%).

### Ablation Study

| Configuration | CUB All | Stanford-Cars All | Note |
| :--- | :--- | :--- | :--- |
| No Part Constraint (Baseline) | 71.5% | 71.5% | Original Baseline |
| First-stage Labels Only | 79.3% | 76.9% | Suitable for fine-grained |
| Second-stage Labels Only | 73.1% | 71.8% | Over-segmentation on fine-grained |
| Stage 1 + Stage 2 Combined | 77.2% | 75.6% | Hybrid scheme |
| **Full PartCo** | **81.1%** | **78.9%** | Adaptive optimal |

### Key Findings
- First-stage labels perform best on fine-grained datasets; second-stage labels outperform first-stage on general datasets (ImageNet-100).
- Projection dimension $d' = 128$ is optimal; higher dimensions lead to overfitting.
- The addition of unsupervised part loss significantly improves performance (+2.1%)—constraints on unlabeled data are critical.
- The balancing factor $\lambda_b = 0.35$ shows the strongest robustness.

## Highlights & Insights
- **Clever Construction of Self-Supervised Signals**: Part-level correspondence labels are obtained directly by freezing the foundation model's innate structure (patch tokens), avoiding extra annotation; compared to SPTNet's pixel-wise prompt learning, the PCA + clustering approach is more stable and efficient.
- **Universal and Lightweight Enhancement Mechanism**: PartCo acts as an independent module that can be combined with any GCD method; gains are observed across different paradigms like SimGCD, SPTNet, FlipClass, and SelEx.
- **Dataset-Adaptive Granularity Selection**: Automatically switching between first- and second-stage labels elegantly handles the heterogeneity of fine-grained and general datasets.

## Limitations & Future Work
- Offline label construction cost: Part labels must be pre-constructed (5-180 minutes depending on dataset size).
- Heuristics in threshold and cluster selection: Still relies on preset thresholds $\tau_{\text{obj}} = 0.6$ and k-means initialization.
- Limitations of part semantics: The proposed part labels are purely results of visual clustering and may not correspond to true semantic parts.
- Improvements: Introduce weak semantic priors to guide PCA projection; design online dynamic update mechanisms; extend to 3D object discovery, video sequences, etc.

## Related Work & Insights
- **vs SPTNet**: Both focus on part-level information, but SPTNet requires supervised backpropagation to learn pixel-level prompt masks; PartCo is more stable by directly extracting correspondence labels from frozen models.
- **vs HypCD**: HypCD changes geometric metrics via hyperbolic space; PartCo injects visual inductive bias through explicit part structures.
- **vs Self-Supervised Part Discovery Methods**: PartCo leverages implicit part knowledge from frozen DINO models—large-scale self-supervised pre-training has already sufficiently learned visual structures.

## Rating
- Novelty: ⭐⭐⭐⭐ The automatic construction of part-level correspondence labels is clever, and the integration with existing GCD methods is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers fine-grained and general datasets + multiple baselines + multiple backbones (DINOv2/v3/CLIP) + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and smooth logical progression.
- Value: ⭐⭐⭐⭐⭐ Provides a simple yet effective enhancement mechanism for the important open-world vision problem of GCD, with the potential to become a standard component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism](../../CVPR2026/self_supervised/omnigcd_abstracting_generalized_category_discovery_for_modality_agnosticism.md)
- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](../../AAAI2026/self_supervised/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)
- [\[CVPR 2026\] Learning Like Humans: Analogical Concept Learning for Generalized Category Discovery](../../CVPR2026/self_supervised/learning_like_humans_analogical_concept_learning_for_generalized_category_discov.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)

</div>

<!-- RELATED:END -->
