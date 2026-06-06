---
title: >-
  [Paper Note] Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild
description: >-
  [CVPR2026][Self-Supervised Learning][Semantic Correspondence] This paper reformulates semantic correspondence as a Fused Gromov-Wasserstein (FGW) optimal transport problem…
tags:
  - "CVPR2026"
  - "Self-Supervised Learning"
  - "Semantic Correspondence"
  - "Optimal Transport"
  - "Gromov-Wasserstein"
  - "3D Geometric Constraints"
  - "Pseudo Labels"
  - "Foundation Models"
date: 2026-05-08
content_hash: aa609240a17a63a5
---

# Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild

**Conference**: CVPR2026  
**arXiv**: [2603.11618](https://arxiv.org/abs/2603.11618)  
**Code**: [Shape-of-You](https://github.com/im-jiin/Shape-of-You)  
**Area**: Self-Supervised  
**Keywords**: Semantic Correspondence, Optimal Transport, Gromov-Wasserstein, 3D Geometric Constraints, Pseudo Labels, Foundation Models

## TL;DR

This paper reformulates semantic correspondence as a Fused Gromov-Wasserstein (FGW) optimal transport problem, leveraging geometric structural constraints from 3D foundation models to generate globally consistent pseudo labels, thereby addressing the geometric inconsistency caused by the locality and 2D appearance ambiguity inherent in conventional nearest-neighbor matching.

## Background & Motivation

**Core challenge in semantic correspondence**: Establishing pixel-level semantic correspondence in the wild is extremely difficult due to extreme viewpoint changes, lighting variations, and intra-class shape differences. Fully supervised methods rely on expensive per-pixel annotations and exhibit poor scalability.

**Inherent limitations of pseudo-label methods**: Current methods without explicit geometric annotations (e.g., ASIC) rely on nearest-neighbor (NN) matching using features from foundation models such as DINO and SD to generate pseudo labels. However, NN matching is a local operation that ignores global structural relationships.

**Geometric ambiguity of 2D features**: Models trained purely on 2D appearance cannot reflect the true 3D geometric structure of objects, leading to correspondences that are semantically plausible but geometrically incorrect (e.g., mismatches on symmetric parts or repeated texture regions).

**Lack of structural consistency in local matching**: NN matching makes independent per-point decisions and does not guarantee global geometric coherence of the matched set, introducing training noise that degrades model performance.

**Computational bottleneck of GW**: Gromov-Wasserstein is a non-convex quadratic programming problem; direct solving is computationally infeasible, necessitating efficient approximation schemes.

**Partial visibility and occlusion**: In real scenes, objects are frequently occluded or truncated, making the hard marginal constraints of balanced optimal transport (requiring all mass to be transported) unsuitable for such partial matching situations.

## Method

### Overall Architecture

Shape-of-You (SoY) adopts a two-stage pipeline: (1) **Pseudo-label generation**: high-quality pseudo labels are generated via FGW optimal transport that fuses semantic feature similarity with 3D geometric structural consistency; (2) **Adapter training**: a lightweight adapter network is trained using soft-target loss; at inference, only a single forward pass is required with no iterative optimization.

**Input representation**: Given source/target image pairs, instance regions are segmented with SAM and divided into patch grids. Each patch obtains two representations — semantic features $\mathbf{f}_i \in \mathbb{R}^d$ extracted by DINOv2/SD, and 3D coordinates $\mathbf{v}_i \in \mathbb{R}^3$ lifted by VGGT (a 3D foundation model).

### Key Designs

**Stage 1 — Semantic UOT Initialization ($t=0$)**:

- Compute the semantic cost matrix $C^{\text{sem}}_{ij} = 1 - \cos(\mathbf{f}_i^A, \mathbf{f}_j^B)$
- Adopt unbalanced optimal transport (UOT) rather than classical OT, relaxing the marginal constraints via KL divergence penalties to handle occlusion and non-overlapping regions
- Solve with the Sinkhorn algorithm to obtain the initial transport plan $\pi^{(0)}$

**Stage 2 — Anchor-Based FGW Iterative Refinement ($t>0$)**:

- **Anchor selection**: Select $K=64$ high-confidence anchor pairs from $\pi^{(t-1)}$, requiring cycle consistency (3D cycle error of forward-then-backward matching below threshold $\delta$)
- **GW linearization**: Replace one $\pi$ in the quadratic term with the sparse anchor transport plan $\hat{\pi}$, reducing the $O(N^2M^2)$ quadratic problem to an $O(NMK)$ linear problem: $C^{\text{geo}}_{ij} = \frac{1}{K}\sum_{(a_A,a_B)\in\mathcal{A}} |D^A_{i,a_A} - D^B_{j,a_B}|$
- **Fused cost**: After normalization, costs are fused as $C^{\text{total}} = (1-\alpha)\tilde{C}^{\text{sem}} + \alpha\tilde{C}^{\text{geo}}$ ($\alpha=0.3$), followed by UOT solving to obtain $\pi^{(t)}$
- Iterated for $T$ rounds with progressively improved anchors; the final $\pi^{(T)}$ serves as the pseudo label

**Soft Target Loss**:

- Extract top-$k$ candidates from $\pi^{(T)}$ to construct a multi-hot binary target $\pi^{\text{hard}}$
- Compute semantic soft targets $\pi^{\text{curr}}$ via OT using the network's current features (stop gradient)
- Mix: $\pi^{\text{soft}} = (1-\beta)\pi^{\text{hard}} + \beta\pi^{\text{curr}}$ ($\beta=0.5$), preventing excessive penalization of semantically similar but unselected candidates

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{soft}} + \mathcal{L}_{\text{dense}}$$

- $\mathcal{L}_{\text{soft}}$: Symmetric cross-entropy loss computed between the predicted similarity distribution and $\pi^{\text{soft}}$, with a learnable temperature parameter $\tau$
- $\mathcal{L}_{\text{dense}}$: Standard dense correspondence loss that propagates gradients to all feature map locations via soft-argmax, regularized with Gaussian noise $\epsilon$

## Key Experimental Results

### Main Results

| Method | SPair-71k PCK@0.1 | SPair-71k PCK@0.05 | AP-10k I.S. | AP-10k C.S. |
|------|-------------------|---------------------|-------------|-------------|
| ASIC | 36.9 | - | - | - |
| DINOv2 | 55.7 | - | - | - |
| DistillDIFT | 59.8 | 42.7 | 65.5 | 62.8 |
| DINOv2 + SD | 63.5 | 48.3 | 65.5 | 63.3 |
| **SoY (Ours)** | **67.9** | **50.8** | **68.0** | **65.8** |

On SPair-71k, the proposed method achieves 67.9% PCK@0.1, surpassing the zero-shot baseline DINOv2+SD by 4.4 percentage points, and attaining the best or second-best performance on 17 of 18 categories. Under zero-shot evaluation on AP-10k, the method achieves top performance across all three settings.

### Ablation Study

| Pseudo-Label Strategy | Geometry-aware PCK_label@0.1 |
|-----------|----------------------------|
| Nearest Neighbor | 53.8% |
| Semantic OT | 54.5% |
| Fused OT (balanced) | 55.7% |
| **Fused UOT (Ours)** | **56.1%** |

| Component Ablation | PCK@0.1 |
|---------|---------|
| Backbone (DINOv2+SD) zero-shot | 63.5 |
| Adapter + NN labels | 64.6 |
| Adapter + FGW labels | 66.8 |
| + relaxed c.c. + soft target loss | **67.9** |

### Key Findings

- **3D geometric distance is critical**: In intra-structure comparison experiments, both 2D distance (65.7%) and semantic distance (66.2%) are inferior to 3D geometric distance (67.6%); 2D distance even underperforms the baseline without GW (66.5%)
- **Geometric constraints are especially important for ambiguous scenarios**: The GW term yields a +2.3%p gain on the Geometry-aware subset, while the overall gain is comparatively modest, confirming that 3D constraints primarily resolve geometric ambiguity
- **Synergistic effect of each component**: FGW pseudo labels (+2.0%p) > cycle consistency (+0.2%p), demonstrating that global structural matching is far more effective than local post-processing; the soft target loss provides additional gains on top of FGW labels

## Highlights & Insights

- **Novel problem formulation**: This work is the first to model semantic correspondence as an FGW optimal transport problem, naturally integrating cross-domain semantic similarity (Wasserstein) with intra-domain geometric consistency (Gromov-Wasserstein)
- **Efficient and practical anchor linearization**: By using 64 anchor points, the NP-hard quadratic GW problem is approximated as a linear OT solvable by Sinkhorn, making the approach engineering-feasible
- **Judicious integration of 3D foundation models**: VGGT is employed to lift 2D images into 3D, providing meaningful geometric metrics for the GW term rather than computing distances in raw 2D coordinates
- **Inference efficiency**: After training, inference requires only a lightweight adapter forward pass followed by nearest-neighbor matching, with no iterative OT solving

## Limitations & Future Work

- **Dependence on 3D foundation model quality**: Failures of VGGT on transparent surfaces or planar reconstructions can yield erroneous anchors
- **Difficulty with symmetric parts**: Symmetric object parts (e.g., car components at intermediate viewpoints) may lead to incorrect anchor matches
- **Risk of over-smoothing with soft targets**: The mixing ratio $\beta=0.5$ may blur geometric signals in certain scenarios
- **Dependence on SAM segmentation**: Pseudo-label generation relies on the instance segmentation quality of SAM; inaccurate segmentation degrades downstream matching

## Related Work & Insights

- **Semantic correspondence without geometric annotations**: ASIC (NN pseudo labels), SHIC (image-to-shape), DistillDIFT (feature distillation) — all rely on local matching or 2D distillation, lacking global structural awareness
- **Optimal transport for structural matching**: Classical Wasserstein OT for correspondence learning; GECO introduces unbalanced OT for geometry-consistent feature learning — but neither combines the structural comparison capability of Gromov-Wasserstein
- **3D geometric foundation models**: DUSt3R and MASt3R require multi-stage pipelines; VGGT predicts multiple 3D properties in a single forward pass — this work leverages VGGT to provide geometric priors for GW
- **GW approximation methods**: GW approximations in graph matching, video action segmentation, and related fields — the anchor linearization proposed here is consistent with the anchor-based GW surrogate paradigm

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of FGW formulation and 3D foundation model geometric priors is pioneering)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two datasets + detailed ablations + PCK_label metric validating pseudo-label quality)
- Writing Quality: ⭐⭐⭐⭐ (OT background is clearly presented, method derivation is complete, figures and tables are highly informative)
- Value: ⭐⭐⭐⭐ (Establishes a new paradigm for annotation-free semantic correspondence; the 3D+OT framework is transferable to other visual matching tasks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[ICCV 2025\] MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning](../../ICCV2025/self_supervised/mosic_optimal-transport_motion_trajectory_for_dense_self-supervised_learning.md)
- [\[ICML 2026\] PartCo: Part-Level Correspondence Priors Enhance Category Discovery](../../ICML2026/self_supervised/partco_part-level_correspondence_priors_enhance_category_discovery.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [\[ICLR 2026\] Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models](../../ICLR2026/self_supervised/gradient-sign_masking_for_task_vector_transport_across_pre-trained_models.md)

</div>

<!-- RELATED:END -->
