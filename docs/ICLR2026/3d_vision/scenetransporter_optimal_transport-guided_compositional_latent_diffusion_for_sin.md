---
title: >-
  [Paper Note] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation
description: >-
  [ICLR 2026][3D Vision][Structured 3D Scene] SceneTransporter reformulates open-world structured 3D scene generation as a global correspondence assignment problem by introducing an entropic optimal transport (OT) framewor…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Structured 3D Scene"
  - "Optimal Transport"
  - "Compositional Diffusion"
  - "Instance Disentanglement"
  - "Cross-Attention Gating"
date: 2026-05-08
content_hash: f1850f48b5ec93f2
---

# SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation

**Conference**: ICLR 2026
**arXiv**: [2602.22785](https://arxiv.org/abs/2602.22785)  
**Code**: [Project Page](https://2019epwl.github.io/SceneTransporter/)  
**Area**: 3D Vision / Structured Scene Generation
**Keywords**: Structured 3D Scene, Optimal Transport, Compositional Diffusion, Instance Disentanglement, Cross-Attention Gating

## TL;DR

SceneTransporter reformulates open-world structured 3D scene generation as a global correspondence assignment problem by introducing an entropic optimal transport (OT) framework into the denoising loop of a compositional 3D latent diffusion model. The OT plan gates cross-attention to enforce exclusive patch-to-part routing (preventing feature entanglement), while edge-regularized assignment costs encourage clean instance separation at image boundaries. The approach achieves state-of-the-art instance-level consistency and geometric fidelity on 74 diverse open-world scene images.

## Background & Motivation

**Background**: High-quality 3D scene generation is a cornerstone of immersive technology and embodied AI. However, most existing scene generators produce monolithic meshes that cannot be directly used in downstream tasks—material assignment, physical simulation, asset retrieval, and fine-grained editing all require explicit instance-level object-context decomposition.

**Limitations of Prior Work**:

1. **Fragile divide-and-conquer pipelines**: Segmenting the input image → generating 3D objects separately → assembling the scene. This pipeline is heavily dependent on 2D segmentation quality, handles occlusions poorly, and allows minor segmentation errors to propagate into severe 3D geometric artifacts.
2. **End-to-end compositional generation fails in open-world settings**: Methods such as PartPacker and PartCrafter perform well at object-level part generation but exhibit two pathological failure modes when generalized to complex open-world scenes:
    - **Structural Mispartition**: Semantic instances fail to form disjoint parts, with a single object's geometry scattered across multiple part-tokens.
    - **Geometric Redundancy**: Multiple latents compete to describe the same spatial region, causing overlaps.
3. **Key Challenge**: Unconstrained soft attention cannot establish globally consistent patch-to-part assignments.

**Core Problem**: Part-level generators implicitly encode correct instance grouping information in their features (recoverable via debiased clustering), yet the models themselves lack structural constraints to make these associations explicit.

**Goal**: Introduce an optimal transport framework to impose explicit global assignment constraints—the OT marginal constraints prevent feature entanglement, coverage budget constraints prevent part-token information starvation, and edge regularization prevents cross-boundary leakage.

## Method

### Overall Architecture

SceneTransporter is built upon an existing compositional 3D generator (PartPacker's rectified-flow DiT). At each denoising step $t$, it: (1) computes an edge-regularized cost matrix between image patches and part-level tokens; (2) solves entropic OT to obtain the optimal transport plan $\mathbf{A}_t$; (3) uses the transport plan to gate the Keys and Values in cross-attention; and (4) updates the latent for the next denoising step. The entire process is training-free and functions as a plug-and-play inference-time mechanism.

### Key Design 1: Debiased Clustering Probe (Diagnostic Tool)

Prior to designing a solution, the authors first quantitatively diagnose the problem via debiased clustering:

1. **Identifying shared subspaces**: Canonical Correlation Analysis (CCA) is applied to identify shared components across sets of part-level latents.
2. **Suppressing shared components**: Tokens are projected onto the orthogonal complement of the shared subspace to isolate object-specific variation.
3. **Re-clustering**: Clustering is performed on the residual tokens.

Experiments show that clustering raw part-tokens directly fails to produce stable instance groupings, whereas post-CCA debiased clustering succeeds reliably. This confirms that the features **implicitly** contain correct grouping information, but the model **fails to establish** these associations explicitly—necessitating external structural constraints.

### Key Design 2: OT Plan-Gated Cross-Attention

At denoising step $t$, the entropic OT problem between $N$ 3D parts and $L$ image patches is solved:

$$\mathbf{A}_t = \arg\min_{\mathbf{A} \ge 0} \langle \mathbf{C}_t, \mathbf{A} \rangle + \varepsilon_t \mathcal{H}(\mathbf{A}) \quad \text{s.t.} \quad \mathbf{A}\mathbf{1} = \boldsymbol{\mu}, \; \mathbf{A}^\top\mathbf{1} = \boldsymbol{\nu}$$

where $\boldsymbol{\mu}$ is the per-part capacity budget (preventing part "starvation") and $\boldsymbol{\nu} = \frac{1}{L}\mathbf{1}_L$ (each patch contributes equal information). The problem is solved via stabilized log-domain Sinkhorn iteration for 40 steps.

The transport plan is converted into a gating signal that modulates Keys and Values through a bounded identity-preserving function:

$$\psi_{\lambda_t, \varepsilon_g}(w) = \varepsilon_g + (1-\varepsilon_g) w^{\lambda_t}$$

where $\lambda_t$ controls gating strength ($\lambda_t = 0$ degenerates to standard attention) and $\varepsilon_g$ is the minimum transmission rate (preventing complete blockage). After gating, each part attends exclusively to its own view of image memory, ensuring routing exclusivity.

### Key Design 3: Edge-Regularized Assignment Cost

In cluttered scenes, patch features near contact boundaries may be compatible with multiple parts, causing information leakage across objects. An image edge prior is introduced to constrain assignments:

1. An edge map $\mathbf{E}$ is extracted and downsampled to the patch grid.
2. A 4-neighborhood graph is constructed, with edge-aware coupling weights: $w_{j\ell} = \exp(-\gamma_{\text{edge}} \max\{\mathbf{E}_\downarrow(j), \mathbf{E}_\downarrow(\ell)\})$.
3. Part-patch cosine similarities are smoothed in an edge-aware manner (propagating across low-edge regions, blocking across high-edge regions).
4. Contrast normalization yields the final OT cost: $\mathbf{C}_t(i,j) = \frac{1}{2}(1 - \widetilde{S}_{i,j})$.

This produces clean instance separation at object contact regions without any instance mask supervision, relying solely on image edges.

## Key Experimental Results

### Main Results: Quantitative Evaluation on 74 Open-World Scenes

| Method | Requires Mask | ULIP↑ | ULIP-2↑ | Uni3D↑ | IoU_max↓ | IoU_mean↓ | Inference Time (s) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MIDI | ✓ | 0.1397 | 0.2763 | 0.2518 | 0.0458 | 0.1642 | 149.68 |
| PartCrafter | ✗ | 0.1177 | 0.3096 | 0.2635 | **0.0042** | **0.0539** | 157.97 |
| PartPacker | ✗ | 0.1417 | 0.3083 | 0.2887 | 0.0319 | 0.2142 | 47.41 |
| **Ours** | ✗ | **0.1466** | **0.3220** | **0.3021** | 0.0101 | 0.0926 | 54.99 |

SceneTransporter achieves the best performance on all three geometric fidelity metrics (ULIP=0.1466, ULIP-2=0.3220, Uni3D=0.3021) and ranks second on part disentanglement metrics (PartCrafter achieves lower IoU by discarding background/floor, at the cost of scene completeness). Inference time is only 7.6 seconds slower than PartPacker (54.99 vs. 47.41 s), and substantially faster than MIDI (149.68 s) and PartCrafter (157.97 s).

### User Study: Subjective Evaluation by 30 Participants

| Method | Geometry Quality↑ | Layout Consistency↑ | Segmentation Plausibility↑ |
|------|:---:|:---:|:---:|
| MIDI | 2.61 | 1.82 | 2.29 |
| PartCrafter | 2.44 | 1.63 | 2.17 |
| PartPacker | 2.81 | 2.95 | 1.97 |
| **Ours** | **3.09** | **3.34** | **3.22** |

Using a forced-ranking scheme (1–4, higher is better), SceneTransporter receives the highest preference scores across all three dimensions, with a particularly large margin in segmentation plausibility (3.22 vs. PartPacker's 1.97).

### Ablation Study

**OT Plan Gating vs. Standard Attention**: Standard cross-attention produces noisy and chaotic attention maps with disordered patch-to-part mappings, leading to corrupted geometry. With OT gating, attention maps for distinct parts (e.g., ground vs. building) separate cleanly, and hard affinity maps show non-overlapping spatial assignments, yielding clean part geometry.

**Evolution of OT Plan During Denoising**: The transport plan stabilizes rapidly after approximately $t \approx 540/600$ steps—coarse-grained semantic routing is established early and maintained, with only local detail refinement occurring in later steps. This explains the high degree of instance-level consistency observed in the final parts.

**Effect of Edge Regularization**: At object contact regions (e.g., sofa and corner side table, wooden stakes and fence), edge regularization produces clean separation between adjacent but semantically distinct objects, whereas the variant without edge regularization exhibits mixed parts and ambiguous boundaries in these areas.

## Highlights & Insights

### Strengths

1. **Diagnosis-driven methodology**: The debiased clustering probe quantitatively exposes the root cause of failure before a solution is designed—methodologically rigorous.
2. **Mathematical elegance**: Structured 3D generation is reformulated as an optimal transport problem, with constraints carrying clear semantics (exclusivity, coverage, edge-awareness); all operations are differentiable and require no training.
3. **Plug-and-play**: Applied as an inference-time mechanism on top of a pretrained generator, adding only ~7.6 seconds of inference overhead, making it highly practical.
4. **Comprehensive evaluation**: Quantitative metrics + 30-participant user study + extensive ablation analysis + denoising process visualization.

### Limitations & Future Work

1. Evaluation is conducted on only 74 images, limiting statistical reliability.
2. PartCrafter's lower IoU scores stem from discarding background/floor rather than superior disentanglement, making the comparison partially unfair; a controlled comparison under identical scene completeness requirements is absent.
3. Edge detection relies on low-level features (e.g., Canny/Sobel), which may produce excessive spurious edges in texture-rich complex scenes, potentially degrading OT assignment quality.

### Rating

⭐⭐⭐⭐⭐ — A work of both theoretical depth and practical efficacy. The complete pipeline from diagnosis to solution, the elegant integration of optimal transport with diffusion models, and the training-free plug-and-play design establish SceneTransporter as a landmark method in structured 3D scene generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image](one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](../../CVPR2026/3d_vision/pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[ICML 2026\] AvAtar: Learning to Align via Active Optimal Transport](../../ICML2026/3d_vision/avatar_learning_to_align_via_active_optimal_transport.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](../../ICCV2025/3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICML 2026\] Streaming Sliced Optimal Transport](../../ICML2026/3d_vision/streaming_sliced_optimal_transport.md)

</div>

<!-- RELATED:END -->
