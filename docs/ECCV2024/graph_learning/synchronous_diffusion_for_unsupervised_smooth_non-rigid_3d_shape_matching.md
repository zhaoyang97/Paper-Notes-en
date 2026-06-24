---
title: >-
  [Paper Note] Synchronous Diffusion for Unsupervised Smooth Non-Rigid 3D Shape Matching
description: >-
  [ECCV 2024][Graph Learning][Non-rigid 3D shape matching] A synchronous diffusion regularization method is proposed for unsupervised non-rigid 3D shape matching. The core idea is that "synchronously diffusing the same function on two shapes should yield consistent outputs." Through this simple yet efficient regularization, the matching smoothness of existing deep functional map methods is significantly improved, achieving SOTA performance on several datasets including FAUST…
tags:
  - "ECCV 2024"
  - "Graph Learning"
  - "Non-rigid 3D shape matching"
  - "synchronous diffusion"
  - "functional maps"
  - "smooth regularization"
  - "unsupervised learning"
date: 2026-05-08
content_hash: 61c0bf36a8aaa80b
---

# Synchronous Diffusion for Unsupervised Smooth Non-Rigid 3D Shape Matching

**Conference**: ECCV 2024  
**arXiv**: [2407.08244](https://arxiv.org/abs/2407.08244)  
**Code**: None  
**Area**: Graph Learning  
**Keywords**: Non-rigid 3D shape matching, synchronous diffusion, functional maps, smooth regularization, unsupervised learning

## TL;DR
A synchronous diffusion regularization method is proposed for unsupervised non-rigid 3D shape matching. The core idea is that "synchronously diffusing the same function on two shapes should yield consistent outputs." Through this simple yet efficient regularization, the matching smoothness of existing deep functional map methods is significantly improved, achieving SOTA performance on several datasets including FAUST, SCAPE, and TOPKIDS.

## Background & Motivation

**Background**: Non-rigid 3D shape matching (i.e., finding point-to-point correspondences between two non-rigidly deformed 3D shapes) is a fundamental problem in computer vision and computer graphics, playing a key role in various tasks such as texture transfer, pose transfer, and statistical shape analysis. Recently, deep learning methods based on the Functional Map framework have made significant progress, efficiently establishing correspondences by learning mapping relationships in the low-frequency spectral domain.

**Limitations of Prior Work**: Deep functional map methods perform matching in the spectral domain (low frequencies) and then convert it to point-to-point correspondences in the spatial domain. A core defect in this process is that low-frequency mapping in the spectral domain cannot constrain high-frequency correspondences, leading to spatial unsmoothness in the resulting pointwise correspondences—adjacent vertices may be mapped to distant locations on the counterpart shape. Many subsequent works attempt to compensate for this issue using spatial-domain shape registration, but these methods typically require pre-aligned shapes and are sensitive to topological noise.

**Key Challenge**: The efficiency of the functional map framework stems from low-frequency approximation, yet smooth point-to-point correspondence requires high-frequency information. Existing regularization methods (e.g., Dirichlet energy) either incur high computational costs (requiring iterative optimization) or lead to degenerate solutions (mapping all points to the same location). A smoothing regularization that is both efficient and effective is highly demanded.

**Goal**: To design a general regularization method that can be seamlessly integrated into existing deep functional map frameworks, significantly enhancing the spatial smoothness of point-to-point correspondences without imposing heavy computational overhead.

**Key Insight**: Inspired by graph message passing and the Weisfeiler-Leman graph isomorphism test, the authors propose the concept of "synchronous diffusion"—if the correspondence between two shapes is correct and smooth, the function values at corresponding positions should remain consistent after performing heat diffusion synchronously on both shapes.

**Core Idea**: Leveraging the heat diffusion process as a regularization signal to encourage spatially smooth matching by penalizing correspondences that are "inconsistent after synchronous diffusion."

## Method

### Overall Architecture
The proposed method is built upon the deep functional map framework. Given two triangular meshes $\mathcal{M}$ and $\mathcal{N}$ as input, vertex features are first extracted via a shared DiffusionNet feature extractor. Then, a soft pointwise correspondence $\Pi$ is established through feature matching, and the functional map matrix $C$ is computed in the spectral domain. Synchronous diffusion regularization is integrated into the training as an additional loss term, optimized jointly with existing spectral regularizations (bijectivity, orthogonality, coupling). During inference, the final pointwise correspondence is obtained via spectral-domain nearest neighbor search.

### Key Designs

1. **Synchronous Diffusion Regularization**:

    - **Function**: Detecting and penalizing spatially unsmooth pointwise correspondences via the heat diffusion process.
    - **Mechanism**: Given a random initial function $F_\mathcal{M}$ on shape $\mathcal{M}$, it is first transferred to shape $\mathcal{N}$ using the soft correspondence matrix $\Pi_{\mathcal{NM}}$, yielding $F_\mathcal{N} = \Pi_{\mathcal{NM}} F_\mathcal{M}$. Then, heat diffusion of time $t$ is performed on both shapes independently: $F_\mathcal{M}(t) = h_t^\mathcal{M}(F_\mathcal{M})$ and $F_\mathcal{N}(t) = h_t^\mathcal{N}(F_\mathcal{N})$. Finally, $F_\mathcal{N}(t)$ is transferred back to $\mathcal{M}$, and its difference from $F_\mathcal{M}(t)$ is calculated: $L_{diff} = \|F_\mathcal{M}(t) - \Pi_{\mathcal{MN}} F_\mathcal{N}(t)\|_F^2$. If the correspondence is correct and smooth, the two diffused functions should be consistent; if local errors exist (such as adjacent points mapped to distant locations), the difference after diffusion will expose these errors.
    - **Design Motivation**: Heat diffusion naturally exchanges neighborhood information over shapes without relying on precise mesh connectivity. Compared to message passing (which relies on discrete neighborhoods), diffusion is more robust as it does not depend on a specific mesh triangulation. This makes the regularization applicable to shapes with different resolutions and topologies.

2. **Multi-Scale Random Diffusion Time Sampling**:

    - **Function**: Realizing multi-scale smoothness regularization by randomly sampling different diffusion times $t$.
    - **Mechanism**: For each column (each scalar function) of the initial function $F_\mathcal{M}$, a diffusion time is independently sampled as $t_i \sim \text{Uniform}(0, T)$. A small $t$ covers only a local neighborhood (analogous to one-step message passing), while a large $t$ covers larger regions (analogous to multi-step message passing). The final multi-scale loss is formulated as $L_{diff} = \sum_{i=1}^h \|h_{t_i}^\mathcal{M}(F_\mathcal{M}^i) - \Pi_{\mathcal{MN}} h_{t_i}^\mathcal{N}(\Pi_{\mathcal{NM}} F_\mathcal{M}^i)\|_F^2$. Diffusion is accelerated using spectral computation: $h_t(u) = \Phi \exp(-t\Lambda) \Phi^\top u$, where $\Phi$ and $\Lambda$ are Laplace-Beltrami eigenfunctions and eigenvalues.
    - **Design Motivation**: A fixed diffusion time can only detect inconsistencies at a specific scale. The random multi-scale strategy allows the regularization to cover different scales from local to global simultaneously and is more robust than manual tuning. Experiments also verify that random sampling outperforms a fixed time by 2.4 points on TOPKIDS.

3. **Random Initial Functions**:

    - **Function**: Using randomly sampled unit-norm functions as diffusion inputs instead of predefined ones.
    - **Mechanism**: Each column of the initial function $F_\mathcal{M} \in \mathbb{R}^{n_\mathcal{M} \times h}$ is a randomly sampled unit-norm vector, which is resampled at each training iteration. This acts as a matrix sketching approximation of the full Quadratic Assignment Problem (QAP)—replacing $n$ Dirac delta functions with $h$ random functions ($h \ll n$), significantly reducing computational costs while maintaining sufficient detection capability.
    - **Design Motivation**: Using Dirac functions (i.e., heat kernel matching) requires $O(n^2)$ computation, while using Laplace-Beltrami eigenfunctions, though deterministic, lacks the diversity introduced by randomness. Random functions provide different "probe signals" at each iteration, statistically covering the entire shape surface. Experiments show they yield better performance than deterministic initial functions (such as Laplace-Beltrami eigenfunctions).

### Loss & Training
The total loss constitutes the standard loss of the functional map framework augmented by the synchronous diffusion regularization: $L_{total} = L_{diff} + \lambda_{couple} L_{couple} + \lambda_{struct} L_{struct}$. Here, $L_{couple}$ constrains the consistency between pointwise maps and functional maps, and $L_{struct}$ contains bijectivity and orthogonality constraints of functional maps. Hyperparameters are set to $\lambda_{couple} = \lambda_{struct} = 1$. The maximum diffusion time is set to $T = 10^{-2}$ (for near-isometric/topological noise datasets) or $T = 10^{-4}$ (for non-isometric datasets, as large diffusion times are inconsistent across non-isometric shapes). The dimension of the random functions is $h = 128$.

## Key Experimental Results

### Main Results

| Dataset | Metric (×100) | Ours | URSSM | AttnFMaps | Gain |
|--------|-----------|------|-------|-----------|------|
| FAUST | Geodesic Error ↓ | 1.5 | 1.6 | 1.9 | -6.3% |
| SCAPE | Geodesic Error ↓ | 1.8 | 1.9 | 2.2 | -5.3% |
| SHREC'19 (Cross-dataset) | Geodesic Error ↓ | 3.4 | 4.6 | 5.8 | -26.1% |
| TOPKIDS (Topological Noise) | Geodesic Error ↓ | 5.4 | 9.2 | 11.1 | -41.3% |
| SMAL (Non-isometric) | Geodesic Error ↓ | 6.7 | 7.5 | 8.8 | -10.7% |
| DT4D-H inter (Non-isometric) | Geodesic Error ↓ | 4.8 | 5.3 | 7.3 | -9.4% |
| SHREC'16 CUTS | Geodesic Error ↓ | 6.1 | 7.0 | - | -12.9% |
| SHREC'16 HOLES | Geodesic Error ↓ | 8.7 | 10.5 | - | -17.1% |

### Ablation Study (TOPKIDS Dataset)

| Configuration | Geodesic Error (×100) | Description |
|------|---------------|------|
| Full model (Ours) | 5.4 | Random initialization + Random diffusion time |
| (a) w/o Regularization | 9.2 | Baseline URSSM, no diffusion regularization |
| (b) Fixed single diffusion time | 7.8 | Loses multi-scale capability |
| (c) Using Laplace eigenfunctions | 10.7 | Deterministic initial functions perform poorly |
| (d) Replacing with heat kernel matching | 8.9 | Traditional QAP paradigm, incurs higher computational overhead |
| (e) Replacing with Dirichlet energy | 9.5 | Subject to the risk of degenerate solutions |
| (f) Replacing with Cycle-consistency | 9.8 | Equivalent to diffusion at t=0, lacking smoothness constraint |

### Key Findings
- **The largest improvement is observed in topological noise scenarios** (41.3% error reduction on TOPKIDS), as the diffusion process is robust to minor topological noise. By restricting the maximum diffusion time $T$ to a smaller value, the model can focus on local neighborhoods and reduce the global influence of topological noise.
- **Cross-dataset generalization is significantly superior to existing methods** (26.1% reduction on SHREC'19), indicating that synchronous diffusion regularization effectively enhances the generalization capability of the model.
- Random initial functions outperform deterministic ones (Laplace-Beltrami eigenfunctions) by 5.3 points, because random functions provide diverse probe signals across different iterations.
- When $t=0$, the regularization degenerates to cycle-consistency, verifying that the diffusion process itself is the source of smoothness.
- The diffusion time $T$ is highly sensitive on topological noise datasets (with $T=10^{-2}$ being optimal) but less sensitive on clean shapes.

## Highlights & Insights
- **The concept of synchronous diffusion is elegant and general**: Smoothness constraints are established via a simple definition (synchronous diffusion consistency) without requiring complex spatial registration. The core formulation is only a single line, yet the effect is remarkable.
- **Solid theoretical foundation**: The paper provides theoretical insights from two perspectives—it can be viewed as a randomized approximation (matrix sketching) of the QAP, as well as a continuous-domain WL test. This renders the method not only effective but also interpretable.
- **Plug-and-play regularization**: It only requires adding a single term to the loss function of existing deep functional map methods, without modifying the network architecture or inference pipeline. This non-invasive design significantly lowers the barrier to adoption, enabling easy integration into future methods.
- **Randomized multi-scale strategy outperforms fixed-time**: This finding serves as a valuable reference for many methods that employ diffusion processes.

## Limitations & Future Work
- The authors admit that the method has only been validated in functional map-based non-rigid 3D shape matching. Although conceptually generalizable to scenarios such as image keypoint matching, graph matching, and point cloud matching, it has not yet been experimentally verified.
- It cannot handle partial-to-partial shape matching, which is an inherent limitation of the functional map framework itself.
- For severely non-isometric shape pairs, the diffusion process is intrinsically inconsistent across the two shapes (due to differing heat kernel structures), making the theoretical assumption of synchronous diffusion not strictly hold. Although this issue is mitigated by a small $T$ in experiments, it requires manual parameter tuning for different datasets.
- The choice of the hyperparameter $T$ is relatively sensitive on topological noise datasets, and currently, there is a lack of an automatic selection mechanism for $T$.

## Related Work & Insights
- **vs DGMC (Deep Graph Matching Consensus)**: DGMC utilizes message passing to achieve neighborhood consistency verification, but requires an additional GNN as a verifier, which incurs high computational overhead. In contrast, this work replaces discrete message passing with continuous heat diffusion, which is more efficient and more robust to mesh triangulation.
- **vs Heat Kernel Matching**: Heat kernel matching measures distances in the kernel space with a calculation matrix size of $n \times n$ (vertex pairs). On the contrary, synchronous diffusion operates in the feature space with a matrix size of $n \times h$ ($h \ll n$), yielding higher computational efficiency. Meanwhile, heat kernel matching can be viewed as a special case of synchronous diffusion.
- **vs Dirichlet Energy Regularization**: The optimal solution of Dirichlet energy is degenerate (mapping all points to the same position), requiring additional constraints to prevent it. Synchronous diffusion naturally encourages bijective/high-coverage correspondences, avoiding the degeneracy issue entirely.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of synchronous diffusion is novel, with a perfect combination of theory and experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation conducted across six scenarios, alongside a detailed and systematic ablation study.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear concepts, rigorous theoretical derivations, and an excellent structure.
- Value: ⭐⭐⭐⭐ A plug-and-play regularizer that makes a significant contribution to the field of 3D shape matching.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior](../../ICCV2025/graph_learning/pasta_part-aware_sketch-to-3d_shape_generation_with_text-aligned_prior.md)
- [\[ICLR 2026\] Topological Flow Matching](../../ICLR2026/graph_learning/topological_flow_matching.md)
- [\[ICLR 2026\] Bures-Wasserstein Flow Matching for Graph Generation](../../ICLR2026/graph_learning/bures-wasserstein_flow_matching_for_graph_generation.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)
- [\[ICML 2026\] Deep Neural Sheaf Diffusion](../../ICML2026/graph_learning/deep_neural_sheaf_diffusion.md)

</div>

<!-- RELATED:END -->
