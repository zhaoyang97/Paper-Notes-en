---
title: >-
  [Paper Note] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding
description: >-
  [CVPR 2026][3D Vision][Point Cloud Understanding] This paper proposes SADG, the first framework to introduce Mamba into in-context learning for multi-task point cloud domain generalization. Through three modules — structure-aware serialization (Centroid Distance Spectrum + Geodesic Curvature Spectrum), Hierarchical Domain-Aware Modeling, and Spectral Graph Alignment — SADG comprehensively surpasses the state of the art on reconstruction, denoising, and registration tasks.
tags:
  - CVPR 2026
  - 3D Vision
  - Point Cloud Understanding
  - Domain Generalization
  - Mamba
  - In-Context Learning
  - Structure-Aware Serialization
date: 2026-05-08
content_hash: 3f3ce503635435c0
---

# Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.20739](https://arxiv.org/abs/2603.20739)
**Code**: [https://github.com/Jinec98/SADG](https://github.com/Jinec98/SADG)
**Area**: 3D Vision
**Keywords**: Point Cloud Understanding, Domain Generalization, Mamba, In-Context Learning, Structure-Aware Serialization

## TL;DR

This paper proposes SADG, the first framework to introduce Mamba into in-context learning for multi-task point cloud domain generalization. Through three modules — structure-aware serialization (Centroid Distance Spectrum + Geodesic Curvature Spectrum), Hierarchical Domain-Aware Modeling, and Spectral Graph Alignment — SADG comprehensively surpasses the state of the art on reconstruction, denoising, and registration tasks.

## Background & Motivation

1. **Background**: Transformer and Mamba architectures have achieved notable progress in point cloud representation learning, yet are typically designed for single-task or single-domain settings. DG-PIC is the first work to explore multi-task domain generalization, employing Transformer-based in-context learning (ICL), but suffers from quadratic complexity and the lack of sequential ordering.
2. **Limitations of Prior Work**: Directly applying Mamba to multi-task domain generalization poses serious challenges. Existing Mamba methods rely on coordinate-driven serialization (e.g., axis scanning, Hilbert curves), which is highly sensitive to viewpoint changes and missing regions, disrupts the hierarchical structure of point clouds, and leads to unstable state propagation and "structural drift."
3. **Key Challenge**: Reconstruction, denoising, and registration all depend on preserving the structural hierarchy of point clouds (global topology and local geometric continuity). However, under domain shifts (noise, occlusion, pose variation), coordinate-based serialization distorts neighborhood relationships and intrinsic topology, rendering Mamba's recurrent modeling fragile.
4. **Goal**: (1) Design transformation-invariant, structure-aware serialization; (2) stabilize Mamba's sequential modeling under cross-domain scenarios; (3) achieve test-time domain adaptation without parameter updates.
5. **Key Insight**: The core observation is that reconstruction, denoising, and registration share the requirement of preserving structural hierarchy. Designing serialization based on intrinsic geometry can therefore serve all three tasks simultaneously.
6. **Core Idea**: Serializing unordered point cloud tokens into structurally consistent sequences via intrinsic geometric spectra (topology + curvature), endowing Mamba with structure-aware domain generalization capability.

## Method

### Overall Architecture

The SADG framework comprises three stages. During training: (1) local patch tokens are extracted from multi-source domain point clouds and arranged into ordered sequences via two structure-aware serializations, CDS and GCS; (2) the serialized tokens are processed by Hierarchical Domain-Aware Modeling (HDM), which first performs intra-domain structural modeling followed by inter-domain relational fusion. During testing: (3) Spectral Graph Alignment (SGA) aligns target features toward source domain prototypes in the spectral domain, enabling structure-preserving feature transfer without any parameter updates.

### Key Designs

1. **Centroid Distance Spectrum (CDS)**:

    - **Function**: Transformation-invariant serialization that preserves global topological layout.
    - **Mechanism**: The global centroid of the point cloud $c = \frac{1}{N}\sum u_i$ serves as the starting point. An affinity graph among tokens is constructed as $w_{CDS}(i,j) = \exp(-\|u_i - u_j\|^2 / \sigma^2)$. BFS traversal begins from the token nearest to the centroid, expanding to the highest-affinity neighbor at each step. This is superior to naive distance sorting, which causes abrupt jumps between spatially distant tokens; BFS traversal guarantees local spatial continuity.
    - **Design Motivation**: Naive coordinate sorting (e.g., axis sorting) is not invariant to rotation and viewpoint changes. CDS is based on relative distance relationships, remaining consistent under translation and rotation, while BFS ensures coarse-to-fine encoding of topological information.

2. **Geodesic Curvature Spectrum (GCS)**:

    - **Function**: Encodes curvature continuity of intrinsic surface geometry.
    - **Mechanism**: Geodesic distances between tokens (shortest paths on a KNN adjacency graph) are first computed to characterize manifold connectivity. A heat diffusion process on the Laplace-Beltrami operator then implicitly encodes curvature — heat dissipates rapidly in high-curvature regions and persists in flat regions. A multi-scale heat kernel $K_\tau(i,i)$ yields the curvature descriptor $h_i = [K_{\tau_1}(i,i), ..., K_{\tau_S}(i,i)]$, upon which an affinity graph is built for serialization.
    - **Design Motivation**: Explicit curvature estimation (relying on normals or dense sampling) is extremely fragile under noise, incompleteness, and domain shift. Heat diffusion implicitly encodes curvature with far greater stability than explicit methods, and remains robust to synthetic-to-real domain transfer.

3. **Hierarchical Domain-Aware Modeling (HDM)**:

    - **Function**: Stabilizes cross-domain inference and prevents state propagation disruption caused by naively concatenating sequences from different domains.
    - **Mechanism**: A two-level cascaded design. **Intra-domain Structural Modeling (ISM)**: two independent Mamba branches process the serialized features of the prompt and query domains respectively, $Z^p = \text{Mamba}^p(X_{seq}^p)$, $Z^q = \text{Mamba}^q(X_{seq}^q)$. **Inter-domain Relational Fusion (IRF)**: features from both domains are interleaved in structural order as $Z^{pq} = [z_{\pi(1)}^p, z_{\pi(1)}^q, z_{\pi(2)}^p, z_{\pi(2)}^q, ...]$ and fed into a shared Mamba for joint modeling.
    - **Design Motivation**: Transformer-based ICL directly concatenates tokens from different domains and relies on attention for interaction. Mamba, however, is sequence-sensitive — naively concatenating tokens from different domains disrupts state propagation at domain boundaries. The hierarchical intra-then-inter design ensures that each domain's structural patterns are first stably aggregated within the domain, before implicit feature exchange is achieved through recurrent propagation over the interleaved sequence.

4. **Spectral Graph Alignment (SGA)**:

    - **Function**: Structure-preserving domain adaptation at test time without parameter updates.
    - **Mechanism**: The serialized features of the target domain are treated as graph signals on the CDS/GCS graph. After projection to the spectral domain via Graph Fourier Transform (GFT), they are adaptively aligned toward source domain prototypes as $\hat{X}_{*,i}^t \leftarrow \alpha_i \hat{X}_{*,i}^t + (1-\alpha_i)(\hat{P}_*^s - \hat{X}_{*,i}^t)$, where the alignment strength is adaptively regulated by cosine similarity.
    - **Design Motivation**: With parameters frozen at test time, domain discrepancy must be bridged while maintaining structural consistency. Spectral-domain alignment leverages the intrinsic frequency basis of the structural graph, ensuring that the alignment process preserves topological and geometric consistency.

### Loss & Training

Following the DG-PIC framework, the AdamW optimizer is used with a learning rate of $1 \times 10^{-4}$, cosine decay, batch size of 96, and 300 training epochs. Chamfer Distance serves as the unified loss for all three tasks (reconstruction, denoising, registration). Bidirectional sequences (forward + backward) × two spectra (CDS + GCS) = 4-way sequence concatenation, expanding Mamba's receptive field.

## Key Experimental Results

### Main Results

| Method | Setting | ModelNet Rec. | ShapeNet Den. | ScanNet Reg. | ScanObjectNN Rec. | MP3DObject Rec. |
|--------|---------|---------------|---------------|--------------|-------------------|-----------------|
| DG-PIC | ICL+DG | 6.84 | 9.81 | 5.10 | 4.52 | 5.91 |
| Vanilla Mamba ICL | ICL+DG | 7.69 | 10.19 | 5.56 | 6.93 | 8.28 |
| **SADG (Ours)** | ICL+DG | **5.99** | **9.34** | **3.63** | **4.29** | **3.55** |

*Chamfer Distance ×10⁻³, lower is better. SADG outperforms DG-PIC across all 15 task configurations spanning 5 domains.*

### Ablation Study

| Configuration | Key Impact | Notes |
|---------------|-----------|-------|
| w/o CDS (GCS only) | CD increases | Loss of global topological information |
| w/o GCS (CDS only) | CD increases | Loss of local curvature continuity |
| Naive coordinate sorting replacing CDS/GCS | CD increases significantly | Sensitive to rotation/viewpoint, structural drift |
| w/o HDM (direct concatenation) | CD increases | State propagation disrupted at domain boundaries |
| w/o SGA | CD increases | Reduced test-time domain transfer capability |
| Vanilla Mamba ICL | CD 8.28 vs 3.55 | No structure awareness; severe performance degradation |

### Key Findings

- **Structural drift is the core bottleneck in multi-task domain generalization**: Vanilla Mamba ICL performs substantially worse than DG-PIC (Transformer) (8.28 vs. 5.91 on MP3DObject), demonstrating the fragility of coordinate-based serialization in domain generalization. SADG's structure-aware serialization fundamentally resolves this issue.
- **CDS and GCS are complementary**: CDS primarily improves global reconstruction quality (topological hierarchy), while GCS contributes more to local denoising performance (geometric continuity); their combination yields the best results.
- **The most pronounced advantage is on MP3DObject**: CD drops from 5.91 (DG-PIC) to 3.55 (40% improvement), indicating the greater value of structure-aware modeling in real-scan scenarios characterized by heavier noise and more severe occlusion.
- **SGA's test-time alignment is effective yet conservative**: Adaptive regulation of alignment strength prevents over-correction in irregular regions.

## Highlights & Insights

- The idea of **implicitly encoding curvature via heat diffusion** is particularly elegant: it circumvents the dependence of traditional explicit curvature estimation on normals and sampling density, instead capturing curvature information implicitly through eigendecomposition of the Laplace-Beltrami operator and multi-scale heat kernels. This approach is inherently robust to noise and incompleteness.
- **The inter-domain interleaved sequence design** exploits Mamba's recurrent propagation: features at adjacent positions interact naturally through state propagation, and the interleaved arrangement places structurally corresponding prompt-query token pairs in close proximity within the sequence, achieving implicit structural matching.
- The **spectral-domain alignment** paradigm is transferable to domain adaptation in other structured data settings (e.g., molecular graphs, social networks), requiring only the definition of an appropriate graph structure.

## Limitations & Future Work

- Spectral decomposition (eigenvalue computation) may become a bottleneck for large-scale point clouds; the paper does not discuss computational efficiency when the number of tokens $N$ is large.
- The MP3DObject dataset contains only 7 categories, limiting class diversity.
- The framework is validated only on reconstruction, denoising, and registration; generalization to other point cloud tasks such as classification and segmentation remains unverified.
- **Future directions**: (1) Explore approximate spectral methods (e.g., Chebyshev polynomial approximation) for acceleration; (2) extend structure-aware serialization to large outdoor scene point clouds (e.g., autonomous driving LiDAR); (3) integrate structure-aware serialization into the pre-training stage of point cloud foundation models (e.g., Point-MAE).

## Related Work & Insights

- **vs. DG-PIC**: DG-PIC employs Transformer-based ICL with quadratic complexity and no sequential ordering. SADG replaces the Transformer with Mamba, achieving linear complexity and structure-aware serialization, with superior performance and efficiency.
- **vs. PointMamba**: PointMamba is a single-task Mamba-based point cloud model relying on coordinate serialization. SADG introduces intrinsic geometric spectrum serialization, resolving the structural drift issue in domain generalization.
- **vs. PointDGMamba**: Focuses on domain generalization but targets only classification. SADG is the first to combine Mamba with ICL for multi-task domain generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce Mamba into ICL-based multi-task point cloud domain generalization; all three technical modules (SAS/HDM/SGA) present clear innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 5 domains × 3 tasks; introduces the new MP3DObject dataset; ablation study is thorough.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous, though notation density is high and some derivations could be made more intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides an important methodological contribution to the application of Mamba on structured 3D data.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](../../AAAI2026/3d_vision/dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[CVPR 2026\] QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment](qd-pcqa_quality-aware_domain_adaptation_for_point_cloud_quality_assessment.md)
- [\[CVPR 2026\] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation](context-nav_context-driven_exploration_and_viewpoint-aware_3d_spatial_reasoning_.md)
- [\[CVPR 2026\] STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding](sts_mixer_4d_point_cloud.md)

<!-- RELATED:END -->
