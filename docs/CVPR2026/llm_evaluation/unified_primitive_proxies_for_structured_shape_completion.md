---
title: >-
  [Paper Note] Unified Primitive Proxies for Structured Shape Completion
description: >-
  [CVPR 2026][LLM Evaluation][shape completion] This paper proposes UniCo, which learns unified primitive representations over shared shape features via primitive proxies…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "shape completion"
  - "primitive assembly"
  - "3D reconstruction"
  - "Transformer"
  - "structured understanding"
date: 2026-05-08
content_hash: e8200cd67a574345
---

# Unified Primitive Proxies for Structured Shape Completion

**Conference**: CVPR 2026
**arXiv**: [2601.00759](https://arxiv.org/abs/2601.00759)  
**Code**: [https://unico-completion.github.io](https://unico-completion.github.io)  
**Area**: LLM Evaluation
**Keywords**: shape completion, primitive assembly, 3D reconstruction, Transformer, structured understanding

## TL;DR
This paper proposes UniCo, which learns unified primitive representations over shared shape features via primitive proxies, jointly predicting complete point clouds and assembly-ready quadric primitives (with geometry, semantics, and membership) in a single forward pass. UniCo reduces Chamfer distance by up to 50% and improves normal consistency by up to 7% on synthetic and real-world point cloud benchmarks.

## Background & Motivation

1. **Background**: 3D shape completion aims to recover missing geometry from incomplete scans. Dominant methods (PoinTr, AdaPoinTr, ODGNet, etc.) optimize point-wise discrepancies, recovering local geometry but lacking structured understanding. Primitive assembly models surfaces as compact sets of parametric primitives, providing structured and interpretable geometric representations suitable for downstream editing and topology control tasks.

2. **Limitations of Prior Work**: The prevailing paradigm follows a cascade of "complete then assemble," which suffers from fundamental issues: (a) assembly solvers (e.g., PrimFit, PolyFit) expect structured input, whereas point-wise completion outputs are unstructured; (b) cascaded pipelines are prone to error propagation—mistakes in primitive counts or parameters affect subsequent association steps; (c) two-stage methods such as PaCo, which first regress primitive parameters and then enforce membership, tend to overfit in regions with sparse evidence and support only planar primitives.

3. **Key Challenge**: Point completion and primitive inference are driven by different supervision signals—the former requires point-wise guidance, while the latter relies on discrete and relational cues. The core challenge is enabling the two to be jointly optimized rather than cascaded.

4. **Goal**: To directly predict assembly-ready structured primitives (including geometry, semantic type, and inlier membership) from incomplete point clouds in a single forward pass.

5. **Key Insight**: Three design principles — (a) coordinated pathways: point completion and primitive inference decode in parallel over shared features; (b) unified representation: learnable queries (primitive proxies) aggregate distributed structural information from shape features; (c) consistent optimization: online updating of primitive targets paired with permutation-invariant matching.

6. **Core Idea**: Learnable primitive proxies query shared shape features, enabling a single network to jointly predict point completion and assembly-ready primitives.

## Method

### Overall Architecture
Given an incomplete point cloud, an encoder extracts shared shape features $\mathcal{T} = \{\mathbf{t}^u\}_{u=1}^U$ ($U=512$). Two parallel pathways follow: (1) a point pathway that decodes dense complete point clouds from features based on AdaPoinTr; and (2) a primitive pathway in which $K=40$ learnable primitive proxies query the shared features, are contextualized through a Transformer decoder, and are passed to dedicated prediction heads that output semantic type, geometric parameters, and inlier membership for each primitive. During training, online target updating and Hungarian matching maintain optimization consistency. At inference, a confidence score selects the valid subset of primitives to pass to downstream assembly solvers.

### Key Designs

1. **Primitive Proxies**:

    - **Function**: Aggregate structural information distributed across shared features into a unified primitive-level representation.
    - **Mechanism**: $K=40$ learnable queries $\mathcal{R}^{(0)}$ are initialized and contextualized through 4 Transformer decoder layers. Each layer first performs cross-attention (queries → shared shape features $\mathcal{T}$), then self-attention (inter-query interaction): $\mathcal{R}^{(l)} = \text{self-att}(\text{cross-att}(\mathcal{R}^{(l-1)}, \text{MLP}(\mathcal{T})))$. The resulting contextualized proxies are shared by three prediction heads: a **semantic head** (MLP + softmax predicting primitive type: plane/cylinder/sphere/cone/$\emptyset$); a **membership head** (computing dot-product similarity between proxy embeddings and shape features in a shared latent space, $m_k^u = \text{sigmoid}(\langle \text{MLP}(\mathbf{r}_k), \text{MLP}(\mathbf{t}^u)\rangle)$, with threshold 0.5 to determine inliers); and a **geometry head** (MLP predicting homogeneous quadric parameters $\mathbf{A}_k \in \mathbb{R}^{4 \times 4}$ to uniformly represent all primitive types).
    - **Design Motivation**: Analogous to instance segmentation frameworks such as Mask2Former, queries eliminate hand-crafted clustering steps. A key distinction is that the input here is incomplete and geometric parameters must be predicted simultaneously, necessitating feature sharing with the point completion pathway.

2. **Online Target Update**:

    - **Function**: Resolves the instability in membership supervision caused by continuously changing point predictions during training.
    - **Mechanism**: Primitive targets are dynamically updated at each iteration. (a) Each predicted point $\hat{\mathbf{y}}_j^u$ retrieves the primitive label $p_{i^*}$ of its nearest GT point; (b) majority voting over each patch yields a patch-level primitive label $\hat{\mathcal{P}}^u$; (c) patches belonging to the same primitive are collected to form the online target $\mathcal{I}_g$. These targets are recomputed at every iteration, enabling joint optimization of assignments and network parameters.
    - **Design Motivation**: Conventional approaches provide membership supervision on a fixed point set, but the predicted point distribution of the completion network shifts continuously during training, causing unstable optimization under fixed membership correspondences. Ablation experiments show that removing online target updating causes CD to catastrophically increase from 2.44 to 12.22 (5×), establishing it as the most critical design choice.

3. **Matching and Joint Loss**:

    - **Function**: Align unordered predicted primitives with GT primitives and compute a comprehensive loss.
    - **Mechanism**: A pairwise cost matrix is constructed comprising three terms: semantic cost (correct classification), membership cost (CE + Dice loss), and geometric cost (Chamfer distance over inliers + parameter L1 distance). The Hungarian algorithm finds the optimal bipartite matching. The total loss equals the sum of matched primitive costs plus a global object-level Chamfer distance. Unmatched predictions are down-weighted through the semantic term to handle class imbalance.
    - **Design Motivation**: Since the primitive set is unordered, permutation-invariant matching is required, analogous to the DETR object detection paradigm, but extended to multi-task matching that simultaneously handles semantics, geometry, and membership.

### Inference
At inference, valid primitives are selected via a confidence score: $s_k = \pi_k[\hat{c}_k] \cdot \frac{1}{|\hat{\mathcal{I}}_k|} \sum_{u \in \hat{\mathcal{I}}_k} m_k^u$. Primitives with $s_k > 0.5$ are passed to downstream assembly solvers.

## Key Experimental Results

### Main Results (ABC-multi + PrimFit Assembly)

| Method | Primitive Extractor | CD ↓ | HD ↓ | NC ↑ | FR ↓ |
|--------|-------------------|------|------|------|------|
| AdaPoinTr | HPNet | 4.41 | 13.36 | 0.872 | 8.97% |
| ODGNet | HPNet | 4.33 | 13.63 | 0.873 | 7.41% |
| ODGNet | RANSAC | 4.80 | 22.15 | 0.868 | 0.39% |
| SymmComplete | HPNet | 4.57 | 13.58 | 0.865 | 9.84% |
| **UniCo (Ours)** | **Built-in** | **2.18** | **7.53** | **0.935** | **1.49%** |

### Ablation Study (ABC-multi, 200 epochs)

| Configuration | CD ↓ | NC ↑ |
|--------------|------|------|
| Full model (UniCo) | 2.44 | 0.924 |
| no param. head | 2.52 (-0.08) | 0.921 |
| no prim. Chamfer | 2.53 (-0.09) | 0.920 |
| CE-only membership | 2.53 (-0.09) | 0.923 |
| Dice-only membership | 2.66 (-0.22) | 0.914 |
| **no online target** | **12.22 (-9.78)** | **0.631** |
| two-stage training | 2.55 (-0.11) | 0.919 |

### Real-World Data (Building-PCC + PolyFit)

| Method | CD ↓ | HD ↓ | NC ↑ | FR ↓ |
|--------|------|------|------|------|
| AdaPoinTr | 4.87 | 10.61 | 0.934 | 0.85% |
| ODGNet | 3.97 | 9.09 | 0.947 | 0.87% |
| PaCo | 4.89 | 10.74 | 0.932 | 0.54% |
| **UniCo (Ours)** | **3.84** | **9.18** | **0.949** | **0.39%** |

### Key Findings
- Online target updating is the most critical design: removing it causes CD to catastrophically worsen by 5× (2.44 → 12.22) and NC to collapse from 0.924 to 0.631, demonstrating that dynamically synchronizing primitive supervision with continuously changing point predictions is essential in completion tasks.
- Better point-wise metrics do not imply better reconstruction: SymmComplete achieves the lowest point-wise CD but yields among the highest post-assembly CD, indicating that structured output matters more than raw point accuracy.
- UniCo consistently outperforms baselines across 4 different assembly solvers (PrimFit, PolyFit, KSR, COMPOD), demonstrating the generality of its primitive quality.
- Robustness analysis: as incompleteness increases from 25% to 75%, UniCo's CD rises only from 1.8 to 2.7, while baselines roughly double to ~6.0.
- Noteworthy observation: primitive proxies spontaneously develop consistent proxy-level semantics—specific proxies consistently represent the same object parts across different inputs.

## Highlights & Insights
- Transferring DETR-style query mechanisms to 3D shape completion is an elegant adaptation: primitive proxies are analogous to object queries but extended to simultaneously handle geometry, semantics, and membership in a completion setting.
- Online target updating addresses a fundamental problem—how to provide stable structured supervision when predictions continuously evolve—a principle that generalizes to other learning tasks with dynamically changing prediction targets.
- Homogeneous quadric parameterization provides a unified representation for planes, cylinders, spheres, cones, and other primitive types, simplifying network design and facilitating easy extension to new primitive types.

## Limitations & Future Work
- The method prioritizes assembly-ready structure over point-wise accuracy; for highly unstructured geometry, the benefit of primitive abstraction is limited.
- Final reconstruction quality depends on downstream assembly solvers.
- The number of proxies $K=40$ is fixed; it may be insufficient for more complex models.
- Future directions include leveraging the emergent correspondences of primitive proxies for part-aware assembly, and scaling to large-scale scenes.

## Related Work & Insights
- **vs. PaCo**: PaCo follows a cascade (first predicting primitive parameters, then associating inliers) and supports only planar primitives. UniCo jointly optimizes both pathways and supports mixed primitive types, reducing CD from 1.87 to 1.69 on ABC-plane and from 4.89 to 3.84 on Building-PCC.
- **vs. AdaPoinTr/ODGNet**: These methods achieve good point-wise metrics but perform poorly after assembly because their outputs lack primitive-aware structural information. UniCo's structured completion directly yields assembly-ready primitives.
- **vs. Point2CAD/BSP-Net**: These reconstruction methods perform poorly on partial input. Even when provided with the best available point completion (ODGNet), Point2CAD's CD remains 55% higher than UniCo's.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The primitive proxy concept is novel, transferring query mechanisms to structured completion; online target updating resolves a critical training challenge.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets (synthetic + real), four assembly solvers, detailed ablation studies, and robustness analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Design principles are clearly articulated with rigorous logical derivation from problem formulation to solution.
- **Value**: ⭐⭐⭐⭐ Provides an effective recipe for 3D structured understanding, though the application scope is specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Generalizable Shape Completion with SIM(3) Equivariance](../../NeurIPS2025/llm_evaluation/learning_generalizable_shape_completion_with_sim3_equivariance.md)
- [\[CVPR 2026\] VGA-Bench: A Unified Benchmark for Video Aesthetics and Generation Quality Evaluation](vga_bench_unified_benchmark_for_video_aesthetics_and_generation_quality.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[CVPR 2026\] CryoHype: Reconstructing a Thousand Cryo-EM Structures with Transformer-Based Hypernetworks](cryohype_reconstructing_a_thousand_cryo-em_structures_with_transformer-based_hyp.md)
- [\[CVPR 2026\] Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)

</div>

<!-- RELATED:END -->
