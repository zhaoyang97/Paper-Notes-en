---
title: >-
  [Paper Note] Dual Quaternion SE(3) Synchronization with Recovery Guarantees
description: >-
  [ICML 2026][Robotics][SE(3) Synchronization] This work replaces $4\times4$ matrices with Unit Dual Quaternions (UDQ) to parameterize the SE(3) synchronization problem. It utilizes power iteration on Hermitian dual quater…
tags:
  - "ICML 2026"
  - "Robotics"
  - "SE(3) Synchronization"
  - "Dual Quaternions"
  - "Spectral Methods"
  - "Generalized Power Method"
  - "Multi-scan Point Cloud Registration"
date: 2026-05-08
content_hash: 44230a8a9f18c798
---

# Dual Quaternion SE(3) Synchronization with Recovery Guarantees

**Conference**: ICML 2026  
**arXiv**: [2602.00324](https://arxiv.org/abs/2602.00324)  
**Code**: https://github.com/jnzhao333/dq_sync  
**Area**: 3D Vision / Optimization / Pose Synchronization  
**Keywords**: SE(3) Synchronization, Dual Quaternions, Spectral Methods, Generalized Power Method, Multi-scan Point Cloud Registration  

## TL;DR
This work replaces $4\times4$ matrices with Unit Dual Quaternions (UDQ) to parameterize the SE(3) synchronization problem. It utilizes power iteration on Hermitian dual quaternion matrices for spectral initialization, followed by a Dual Quaternion Generalized Power Method (DQGPM) with element-wise projection onto $\mathrm{UDQ}^n$ for iterative refinement. It provides the first finite-step linear convergence analysis and explicit error bounds for SE(3) synchronization, achieving lower rotation/translation errors and faster execution times than matrix-based methods in multi-scan point cloud registration.

## Background & Motivation

**Background**: SE(3) synchronization—estimating absolute poses of all nodes from a set of noisy relative poses $T_{ij}$—is a fundamental primitive in SLAM, multi-point cloud registration, and SfM. Prevalent approaches represent poses using $4\times4$ matrices and apply spectral relaxation (EIG), semidefinite relaxation (SDR), or Lie algebraic averaging/Riemannian optimization, followed by a "rounding" step back to SE(3).

**Limitations of Prior Work**: Matrix representations suffer from a structural **representation gap**. When representing $\mathrm{SO}(2)$ using complex numbers, the isometry group of the eigenspace exactly matches $\mathrm{SO}(2)$, making rounding a simple normalization. For $\mathrm{SO}(d)$ matrix representations, the isometry group is $\mathrm{O}(d)$, introducing a mild reflection ambiguity solvable via SVD. However, for SE(3), since $\mathrm{SE}(3)=\mathrm{SO}(3)\ltimes\mathbb{R}^3$ is **non-compact** while the relaxed eigenspace geometry is compact and orthogonal, relaxed solutions drift far from the manifold. Rounding becomes an "imposition of structure" rather than error correction, requiring multi-step heuristics (splitting rotation/translation and projecting separately) that are unstable and difficult to analyze.

**Key Challenge**: To select a parameterization such that the **ambiguity group of the spectral relaxation aligns exactly with the global gauge symmetry of SE(3)**, ensuring that rounding acts as a benign projection for which theoretical analysis is tractable. Dual quaternions provide this alignment—Unit Dual Quaternions ($\mathrm{UDQ}$) form a compact 7D manifold encoding both rotation and translation, and the eigenstructure of Hermitian dual quaternion matrices matches the gauge symmetry of SE(3) synchronization.

**Goal**: (1) Formulate SE(3) synchronization as a QPQC over $\mathrm{UDQ}^n$; (2) Design a two-stage algorithm with theoretical guarantees—spectral initialization + iterative refinement; (3) Provide explicit error bounds and finite-step convergence guarantees.

**Key Insight**: The authors observe that while dual quaternions $\mathbb{DH}$ form a ring with zero divisors (not a field), where the norm is a dual-valued pseudo-norm and principal eigenpairs are defined lexicographically, classical synchronization theory can be adapted if: (a) the normalization mapping $\mathcal{N}(\cdot)$ on $\mathrm{UDQ}$ is expressed in closed form and proven to be Lipschitz; (b) errors are controlled separately using the "standard part + dual part" of dual numbers, allowing the spectral+GPM analysis framework from the matrix case to be translated.

**Core Idea**: Each pose is represented as a unit dual quaternion $x_i\in\mathrm{UDQ}$. The optimization $\min_{\bm{x}\in\mathrm{UDQ}^n}\|\bm{C}-\bm{x}\bm{x}^*\|_F^2$ is solved by first obtaining an initialization via dual quaternion power iteration of the Hermitian DQ matrix $\bm{C}$, followed by refinement using the Generalized Power Method (DQGPM) with per-step projection onto $\mathrm{UDQ}^n$. This ensures every iterate is feasible and facilitates proof of linear convergence to an $O(\|\bm{\Delta}\hat{\bm{x}}\|_2/n)$ error floor.

## Method

### Overall Architecture
The input is a Hermitian dual quaternion measurement matrix $\bm{C}\in\mathbb{DH}^{n\times n}$, where $C_{ij}=\hat{x}_i\hat{x}_j^*+\Delta_{ij}$ and $\Delta_{ij}$ represents observation noise. The output is $\bm{x}\in\mathrm{UDQ}^n$, providing dual quaternion estimates for $n$ absolute poses. The pipeline consists of two stages:

1. **Spectral Initialization** (Algorithm 1): Perform power iteration on $\bm{C}$ to obtain the principal eigenvector $\bm{u}_1\in\mathbb{DH}^n$ (subject to $\|\bm{u}_1\|_2^2=n$), followed by element-wise projection $\bm{x}^0=\Pi(\bm{u}_1)\in\mathrm{UDQ}^n$.
2. **DQGPM Refinement** (Algorithm 2): Iteratively perform matrix-vector multiplication $\bm{y}^k=\bm{C}\bm{x}^{k-1}$ followed by projection $\bm{x}^k=\Pi(\bm{y}^k)$ until convergence.

The problem is formulated as a QPQC: $\arg\max_{\bm{x}\in\mathrm{UDQ}^n} \bm{x}^*\bm{C}\bm{x}$ (Proposition 2.1 proves equivalence to the original least squares problem up to a constant factor of 2). Relaxing $\mathrm{UDQ}^n$ to a dual quaternion sphere $\|\bm{x}\|_2^2=n$ yields the principal right eigenvector $\bm{u}_1$ of $\bm{C}$ as the optimal solution, which serves as the spectral estimate.

### Key Designs

1. **Closed-form Projection $\Pi:\mathbb{DH}^n\to\mathrm{UDQ}^n$ and Lipschitz Property**:
    - **Function**: Maps any (potentially infeasible) dual quaternion vector element-wise back to the $\mathrm{UDQ}^n$ set, ensuring the distance to any feasible point is expanded by at most a factor of 2.
    - **Mechanism**: Single-point normalization $\mathcal{N}(x)$ is defined: when the standard part $x_{\mathrm{st}}\neq 0$, $u_{\mathrm{st}}=x_{\mathrm{st}}/|x_{\mathrm{st}}|$ and $u_{\mathcal{I}}=x_{\mathcal{I}}/|x_{\mathrm{st}}| - (x_{\mathrm{st}}/|x_{\mathrm{st}}|)\cdot\mathrm{sc}(x_{\mathrm{st}}^*/|x_{\mathrm{st}}|\cdot x_{\mathcal{I}}/|x_{\mathrm{st}}|)$. This explicitly subtracts the component of the dual part "parallel" to the standard part, corresponding to the projection of translation onto the rotation in SE(3). $\Pi$ uses a fallback $\mathcal{N}(\bm{e}^*\bm{y})$ at $y_i=0$ for well-definedness. Lemma 2.5 proves $|\mathcal{N}(y)-z|\le 2|y-z|$, a critical tool for error propagation.
    - **Design Motivation**: Rounding in matrix methods uses multi-step heuristics (SVD for rotation, centroid calculation, translation correction) which lack analytical forms. The closed-form $\Pi$ provides a controllable operator, allowing the spectral error bound $4\|\bm{\Delta}\|_{\mathrm{op}}/\sqrt{n}$ (Proposition 2.4) to translate into a post-projection bound of $8\|\bm{\Delta}\|_{\mathrm{op}}/\sqrt{n}$ (Theorem 2.8).

2. **Dual Quaternion Power Iteration Spectral Initialization**:
    - **Function**: Executes power iteration on $\bm{C}$ to find the principal eigenvector $\bm{u}_1$ such that $\bm{u}_1^*\bm{C}\bm{u}_1\ge\hat{\bm{x}}^*\bm{C}\hat{\bm{x}}$, providing the feasible initial value $\bm{x}^0$ via projection.
    - **Mechanism**: Iterates $\bm{y}^k=\bm{C}\bm{w}^{k-1},\ \bm{w}^k=\bm{y}^k\cdot(\|\bm{y}^k\|_2)^{-1}$. This is well-defined if $\lambda_{1,\mathrm{st}}\neq 0$ and the initialization's standard part is not orthogonal to the principal eigenvector's standard part. Convergence speed is governed by $r=|\lambda_{1,\mathrm{st}}/\lambda_{2,\mathrm{st}}|>1$.
    - **Design Motivation**: Since the dual quaternion ring has zero divisors, vector division is not always well-defined. The authors decouple the analysis into a "standard part dominant convergence" followed by the dual part, bypassing zero-divisor issues. Proposition 2.4 provides a nonasymptotic bound $\mathrm{d}(\bm{x},\hat{\bm{x}})\le 4\|\bm{\Delta}\|_{\mathrm{op}}/\sqrt{n}$ under operator norm noise.

3. **DQGPM: Per-step Feasible Generalized Power Method with Convergence Guarantees**:
    - **Function**: Starting from $\bm{x}^0$, it alternates $\bm{y}^k=\bm{C}\bm{x}^{k-1}$ and $\bm{x}^k=\Pi(\bm{y}^k)$. Every $\bm{x}^k$ is "stop-anytime feasible" on $\mathrm{UDQ}^n$.
    - **Mechanism**: Extends GPM (Journée et al. 2010) to dual quaternions. Theorem 3.2 proves that when $\|\bm{\Delta}\|_{\mathrm{op},\mathrm{st}}\le n/350$, standard part error contracts linearly: $\mathrm{d}_{\mathrm{st}}(\bm{x}^k,\hat{\bm{x}})\le (1/10)^k\cdot\sqrt{n}/25 + (700/53n)\|(\bm{\Delta}\hat{\bm{x}})_{\mathrm{st}}\|_2$. Dual part error also contracts linearly using auxiliary variables for coupling, reaching an $O(\|\bm{\Delta}\hat{\bm{x}}\|_2/n)$ floor.
    - **Design Motivation**: GPM for SE(3) lacked convergence proofs due to non-compactness and pseudo-norms. The authors bypass this using Euclidean metrics for standard parts and coupling inequalities for dual parts. The resulting $O(\|\bm{\Delta}\hat{\bm{x}}\|_2/n)$ error is significantly tighter than spectral estimates for i.i.d. Gaussian noise.

### Loss & Training
Ours is an iterative algorithm, not a learning-based method. It uses a threshold on $\|\bm{x}^k-\bm{x}^{k-1}\|_2$ as a stopping criterion. The iteration count for initialization $K_{\mathrm{init}}$ is estimated via the explicit lower bound in Corollary 3.4.

## Key Experimental Results

### Main Results
Synthetic Data: $n$ nodes, relative poses observed via an ER graph (edge probability $p$), with i.i.d. Hermitian dual quaternion noise (translation $\sigma_t$, rotation $\sigma_r$). Baselines: EIG, SPEC, SDR.

| Configuration (Noise / rate) | Error_r (Ours) | Error_r (SPEC) | Error_r (EIG) | Error_t (Ours) | Error_t (SPEC) | Error_t (EIG) |
|---|---|---|---|---|---|---|
| (0.05, 5°), p=0.05 | **0.132 ± 0.042** | 1.639 ± 1.971 | 0.174 ± 0.156 | **0.102 ± 0.032** | 0.480 ± 0.530 | 0.551 ± 1.109 |
| (0.20, 20°), p=0.05 | **0.424 ± 0.060** | 2.035 ± 1.823 | 0.585 ± 0.572 | **0.369 ± 0.078** | 0.660 ± 0.455 | 1.043 ± 1.359 |
| (0.05, 5°), p=0.30 | **0.027 ± 0.001** | 0.032 ± 0.013 | 0.098 ± 0.618 | **0.021 ± 0.001** | 0.023 ± 0.001 | 0.137 ± 0.441 |
| (0.20, 20°), p=0.30 | **0.111 ± 0.005** | 0.141 ± 1.126 | 0.219 ± 0.613 | **0.085 ± 0.005** | 0.090 ± 0.005 | 0.194 ± 0.257 |

Observations: In sparse scenarios ($p=0.05$), DQGPM's rotation error is roughly 1/10 of SPEC and 1/3 of EIG, with order-of-magnitude advantages in translation. It remains dominant and stable in dense scenarios ($p=0.30$). SDR is omitted due to poor scalability.

### Multi-scan Point Cloud Registration (Real Data)

| Dataset (sparse) | Missing | DQGPM Time (s) | DQGPM Err | Best Baseline |
|---|---|---|---|---|
| Bunny | 48.00% | 0.0010 | **Best** | EIG/SDR 1-2 orders slower |
| Buddha | 66.67% | 0.0021 | **Best** | Same as above |
| Dragon | 60.44% | 0.0014 | **Best** | Same as above |
| Armadillo | 58.33% | — | **Best** | Same as above |

On the Stanford datasets, DQGPM achieves the lowest rotation/translation error in both sparse and dense settings. Iteration time is in milliseconds, 1-2 orders of magnitude faster than SE-Sync (SDR-based).

### Key Findings
- **Sparse Connectivity is the Differentiator**: While methods converge at $p=0.30$, SPEC collapses and EIG variance explodes at $p=0.05$. DQGPM is nearly unaffected, validating that the "rounding gap" of dual quaternions is crucial when data is scarce.
- **Theoretical Error Floor Validation**: DQGPM's asymptotic error $O(\|\bm{\Delta}\hat{\bm{x}}\|_2/n)$ is tighter than the spectral $O(\|\bm{\Delta}\|_{\mathrm{op}}/\sqrt{n})$ by a factor of $\sqrt{n}$, especially evident in dense, large-$n$ settings.
- **Efficiency via SDP Avoidance**: DQGPM uses $O(n^2)$ matrix-vector multiplications and element-wise projections, avoiding the $O(n^3)$ complexity or PSD cone constraints of SDR, making it highly scalable.

## Highlights & Insights
- **Representation Alignment**: By comparing $\mathrm{SO}(2)$/$\mathrm{SO}(d)$/$\mathrm{SE}(3)$, it becomes clear why matrix methods require "patches" for SE(3) and why dual quaternions are the natural choice.
- **Closed-form Projection as a Pivot**: Lemma 2.5 elevates projection from a heuristic to a controllable operator. This approach is transferable to other manifold optimization problems.
- **Coupling Technique for Dual Numbers**: The "standard part dominant, dual part follows" approach avoids pseudo-norm issues and non-compactness, providing a template for other dual-parameterized optimization algorithms.
- **First Finite-step Guarantee**: Unlike previous asymptotic conclusions, this work provides explicit bounds on error after $k$ steps, enabling the design of adaptive stopping criteria.

## Limitations & Future Work
- **Strict Noise Constants**: Theorem 3.2 requires $\|\bm{\Delta}\|_{\mathrm{op},\mathrm{st}}\le n/350$, which is stricter than typical $\mathrm{SO}(d)$ bounds ($n/100$). Empirical results suggest the method works under higher noise.
- **Non-uniform Noise**: Experiments assume i.i.d. noise, but SLAM noise is often heteroscedastic and trajectory-dependent.
- **GPU Implementation**: While $O(n^2)$ operations are GPU-friendly, lack of standard CUDA kernels for dual quaternions meant only CPU testing was performed.
- **Outlier Robustness**: Current analysis focuses on Gaussian-type noise; future work should integrate truncated least squares or IRLS to handle gross outliers.

## Related Work & Insights
- **vs SE-Sync (Rosen 2019)**: DQGPM provides convergence guarantees via first-order iterations without requiring $O(n^3)$ SDP solvers, showing better accuracy in sparse connectivity.
- **vs SPEC (Doherty 2022)**: SPEC collapses under $p=0.05$, corroborating the argument that matrix representation gaps are fatal in sparse regimes.
- **vs Previous DQ Works**: Unlike prior heuristic dual quaternion synchronization, this is the first to provide comprehensive theoretical analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First complete theory + algorithm for DQ SE(3) synchronization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong comparison, though lacks diverse noise distributions.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the comparative Table 1 is very effective.
- Value: ⭐⭐⭐⭐⭐ High engineering value; can replace backbones in SLAM/SfM without SDP solvers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Statistical Guarantees for Offline Domain Randomization](../../ICLR2026/robotics/statistical_guarantees_for_offline_domain_randomization.md)
- [\[ICML 2026\] Dual Advantage Fields](dual_advantage_fields.md)
- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)
- [\[CVPR 2026\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](../../CVPR2026/robotics/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)
- [\[ICLR 2026\] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](../../ICLR2026/robotics/janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)

</div>

<!-- RELATED:END -->
