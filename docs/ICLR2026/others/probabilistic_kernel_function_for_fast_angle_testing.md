---
title: >-
  [Paper Note] Probabilistic Kernel Function for Fast Angle Testing
description: >-
  [ICLR 2026][approximate nearest neighbor search] This paper studies the angle testing problem in high-dimensional Euclidean space and proposes two deterministic probabilistic kernel functions, $K_S^1$ and $K_S^2$…
tags:
  - "ICLR 2026"
  - "approximate nearest neighbor search"
  - "probabilistic kernel function"
  - "angle testing"
  - "random projection"
  - "similarity search"
date: 2026-05-08
content_hash: 4a5d56be0b3c47b3
---

# Probabilistic Kernel Function for Fast Angle Testing

**Conference**: ICLR 2026
**arXiv**: [2505.20274](https://arxiv.org/abs/2505.20274)  
**Code**: [GitHub](https://github.com/KejingLu-810/KS)  
**Area**: Other
**Keywords**: approximate nearest neighbor search, probabilistic kernel function, angle testing, random projection, similarity search

## TL;DR

This paper studies the angle testing problem in high-dimensional Euclidean space and proposes two deterministic probabilistic kernel functions, $K_S^1$ and $K_S^2$, based on reference angles for angle comparison and angle threshold judgment, respectively. Theoretical guarantees are obtained without relying on asymptotic assumptions of Gaussian distributions. Applied to approximate nearest neighbor search (ANNS), the method achieves 2.5×–3× QPS speedup on HNSW graphs.

## Background & Motivation

Vector similarity search is a fundamental problem in machine learning, data mining, and information retrieval. In high-dimensional spaces, the $\ell_2$ norm, cosine similarity, and inner product are the most commonly used similarity measures, which can often be reduced to computing angles between normalized vectors. In many practical scenarios, the exact angle value is less important than the relative ordering of angles (angle comparison) or whether an angle exceeds a threshold (angle threshold judgment)—collectively referred to as **angle testing**.

Existing works such as CEOs and PEOs generate projection vectors from Gaussian distributions and rely on Lemma 1.3 (Pham, 2021) to establish the relationship between angles and projected values. However, this lemma has a significant theoretical limitation: the relationship assumes $m \to \infty$ projection vectors asymptotically and cannot provide precise guarantees for finite $m$. The paper's starting point is to overcome this limitation.

The authors make two key observations: (1) Gaussian distributions are not fundamentally necessary; the critical factor governing estimation accuracy is the **reference angle**, i.e., the angle between the query vector and its nearest projection vector; (2) by introducing random rotation matrices, the reference angle becomes predictable.

## Method

### Overall Architecture

The paper proposes two probabilistic kernel functions corresponding to two angle testing problems:
- **$K_S^1(\mathbf{q}, \mathbf{v})$**: for angle comparison (Problem 1.1), determining which data vector is more similar to the query
- **$K_S^2(\mathbf{q}, \mathbf{v})$**: for angle threshold judgment (Problem 1.2), determining whether an angle falls below a given threshold

The core idea is to exploit deterministic projection vector structures and reference angle information in place of Gaussian random projections.

### Key Designs

1. **Kernel Function $K_S^1$ (Angle Comparison)**: Defined as $K_S^1(\mathbf{q}, \mathbf{v}) = \langle \mathbf{v}, Z_{HS}(\mathbf{q}) \rangle$, where $H$ is a random rotation matrix and $Z_{HS}(\mathbf{q})$ is the reference vector of $\mathbf{q}$ in the rotated projection set $HS$ (the vector with the largest inner product). Lemma 4.2 proves that its conditional CDF can be expressed exactly using the regularized incomplete Beta function: $F_{K_S^1}(x | A_S(\mathbf{q}) = \cos\psi) = I_t\left(\frac{d-2}{2}, \frac{d-2}{2}\right)$, and the probabilistic guarantee strengthens as the reference angle $\psi$ decreases. **Design Motivation**: to replace asymptotic assumptions with deterministic probabilistic relationships.

2. **Kernel Function $K_S^2$ (Angle Threshold)**: Defined as $K_S^2(\mathbf{q}, \mathbf{v}) = \langle H\mathbf{q}, Z_S(H\mathbf{v}) \rangle / A_S(H\mathbf{v})$, normalized by the cosine of the reference angle $A_S(H\mathbf{v})$. Lemma 4.3 proves that $K_S^2$ is angle-sensitive and satisfies the probabilistic guarantees of Problem 1.2. **Design Motivation**: to leverage reference angle information for more accurate threshold judgment, which existing methods (PEOs) do not utilize.

3. **Projection Vector Configuration**: Two deterministic structures are proposed to replace Gaussian random projections:
    - **Antipodal Pair Structure (Algorithm 1)**: generates $m/2$ random points and their antipodal counterparts in each subspace; simple and efficient
    - **Multi-Cross-Polytope Structure (Algorithm 2)**: exploits the optimal covering property of cross-polytopes, generating multiple cross-polytopes via random rotations and selecting the configuration with the best coverage
    - A multi-level ($L$ subspaces) product quantization technique is introduced to virtually generate $m^L$ projection vectors, significantly reducing the reference angle

4. **KS2 Routing Test**: Applies $K_S^2$ to the routing mechanism of similarity graphs, yielding a concise test inequality: $\sum_{i=1}^L \mathbf{q}_i^\top \mathbf{u}_{e[i]}^i \geq A_S(\mathbf{e}) \cdot \frac{\|\mathbf{w}\|^2/2 - \tau - \mathbf{v}^\top\mathbf{q}}{\|\mathbf{e}\|}$. Compared to PEOs tests, the inequality is simpler (faster to evaluate) and requires fewer stored constants (smaller index).

### Loss & Training

This paper does not involve neural network training. The core optimization objective is to maximize the coverage quality $J(S) = \mathbb{E}_{\mathbf{v} \in U(\mathbb{S}^{d-1})}[A_S(\mathbf{v})]$, i.e., the average coverage of the unit sphere by the projection vector configuration $S$. This is approximated by evaluating $\tilde{J}(S,N)$ on $N$ uniformly sampled points.

## Key Experimental Results

### Main Results (ANNS Performance Comparison)

| Dataset | Metric (QPS@Recall=90%) | HNSW+KS2 | HNSW+PEOs | HNSW | Speedup |
|--------|----------------------|----------|-----------|------|-----------|
| SIFT | QPS | Best | 2nd | Baseline | **2.5×–3× vs HNSW** |
| GloVe1M | QPS | Best | 2nd | Baseline | **10%–30% vs PEOs** |
| GloVe2M | QPS | Best | 2nd | Baseline | 1.1–1.3× vs PEOs |
| Tiny | QPS | Best | 2nd | Baseline | Significant gain over HNSW and ScaNN |
| GIST | QPS | Best | 2nd | Baseline | Significant gain over HNSW |

### Ablation Study (KS1 vs CEOs, $k$-MIPS, $k=10$, $m=2048$)

| Configuration | Word Probe@1K | GloVe1M Probe@1K | Notes |
|------|-------------|-----------------|------|
| CEOs(2048) | 90.203% | 24.166% | Gaussian random projection baseline |
| KS1($S_{sym}$) | 90.265% | 24.456% | Antipodal pair structure, marginal gain |
| KS1($S_{pol}$) | 90.678% | 24.556% | Multi-cross-polytope, best performance |

### Key Findings

- Smaller reference angles yield higher kernel function accuracy—this is the central insight throughout the paper
- Gaussian distributions are not the optimal choice for projection vectors; deterministic structures (especially cross-polytopes) provide smaller reference angles
- HNSW+KS2 achieves optimal search performance when the subspace dimension is $d' = d/L \approx 16$
- The KS2 index is approximately 5% smaller than that of PEOs, and the test inequality is simpler and faster to evaluate
- KS1 yields only modest improvements over CEOs (approximately 0.8%), as the difference lies solely in the projection vector distribution

## Highlights & Insights

- **Theoretical Contribution**: Establishes deterministic probabilistic relationships (Eq. 4) without asymptotic assumptions, representing a fundamental improvement over existing theory (Lemma 1.3)
- **Practical Impact**: HNSW+KS2 achieves 2.5×–3× speedup on ANNS, offering direct practical value
- **Suboptimality of Gaussian Projections**: Both theory and experiments demonstrate that Gaussian projections are suboptimal for angle testing, challenging common assumptions in the random projection literature
- **SIMD-Friendly**: The KS2 test can process 16 edges in parallel via SIMD, enabling efficient hardware-level optimization

## Limitations & Future Work

- KS1 provides limited improvement over CEOs; practical gains are primarily attributable to KS2's graph search acceleration
- The best covering problem remains open in general; no known optimal solution exists for $m > d+3$
- Experiments are conducted on CPU only; performance in GPU-accelerated settings remains to be verified
- Theoretical analysis focuses on Euclidean space and cosine similarity; generalization to other metric spaces has not been discussed

## Related Work & Insights

- **CEOs (Pham, 2021)**: $K_S^1$ can be viewed as a generalization of CEOs, replacing the Gaussian asymptotic relationship with a deterministic reference angle
- **PEOs (Lu et al., 2024)**: $K_S^2$ extends PEOs by adding reference angle normalization, simplifying the test inequality
- **Falconn (Andoni et al., 2015)**: Uses cross-polytopes for LSH; this paper borrows the structural idea for a different purpose
- **Insight**: The reference angle as a key factor controlling estimation accuracy may have analogous roles in other random projection applications, such as dimensionality reduction and sketching algorithms

## Rating

- Novelty: ⭐⭐⭐⭐ The reference angle perspective is novel; replacing asymptotic relationships with deterministic ones represents a significant theoretical advance
- Experimental Thoroughness: ⭐⭐⭐⭐ Six datasets provide comprehensive coverage; comparisons with SOTA are thorough; ablation studies are complete
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, though the notation is dense and some definitions are scattered across the main text and appendix
- Value: ⭐⭐⭐⭐ Substantial contribution to the ANNS field; 2.5×–3× speedup offers direct practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation](learning_adaptive_distribution_alignment_with_neural_characteristic_function_for.md)
- [\[ICLR 2026\] An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)
- [\[ICLR 2026\] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)
- [\[ICLR 2026\] Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)
- [\[ICLR 2026\] Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)

</div>

<!-- RELATED:END -->
