---
title: >-
  [Paper Note] Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project
description: >-
  [NeurIPS 2025][LLM Evaluation][Gaussian Process] This paper proposes the ADASAP algorithm, which extends the sketch-and-project framework to large-scale GP inference via approximate subspace preconditioning…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Gaussian Process"
  - "Sketch-and-Project"
  - "Large-Scale Inference"
  - "Distributed Computing"
  - "Nyström Approximation"
date: 2026-05-08
content_hash: bb953d5e0abd0568
---

# Turbocharging Gaussian Process Inference with Approximate Sketch-and-Project

**Conference**: NeurIPS 2025
**arXiv**: [2505.13723](https://arxiv.org/abs/2505.13723)
**Code**: [GitHub](https://github.com/pratikrathore8/scalable_gp_inference)
**Area**: Gaussian Process / Large-Scale Inference
**Keywords**: Gaussian Process, Sketch-and-Project, Large-Scale Inference, Distributed Computing, Nyström Approximation

## TL;DR

This paper proposes the ADASAP algorithm, which extends the sketch-and-project framework to large-scale GP inference via approximate subspace preconditioning, distributed computation, and Nesterov acceleration. It is the first method to scale exact GP inference beyond $>3\times10^8$ samples, while theoretically establishing condition number-free convergence guarantees for the SAP family.

## Background & Motivation

Gaussian Processes (GPs) are a cornerstone tool in Bayesian optimization and scientific computing, offering probabilistic predictions and uncertainty quantification. However, GP inference requires solving a dense $n \times n$ linear system $(K + \lambda I)w = y$, and direct methods such as Cholesky decomposition incur $\mathcal{O}(n^3)$ cost, limiting applicability to $n \sim 10^4$.

Existing large-scale approaches each suffer from distinct drawbacks:
- **PCG (Preconditioned Conjugate Gradient)**: Robust to ill-conditioning, but slow at $n \sim 10^6$ due to $\mathcal{O}(n^2)$ per-step cost and linear memory requirements for the preconditioner.
- **SDD (Stochastic Dual Descent)**: Scalable to $n \gg 10^6$, but lacks theoretical guarantees, is sensitive to step-size (SDD-100 diverges on all datasets), and converges slowly on ill-conditioned matrices.

The root cause is that no existing method simultaneously achieves inference quality, robustness to ill-conditioning, ease of hyperparameter tuning, and scalability. ADASAP addresses this via the sketch-and-project framework, injecting second-order information through subspace preconditioning while maintaining per-step costs comparable to SDD.

## Method

### Overall Architecture

ADASAP builds on the sketch-and-project framework of Gower & Richtárik (2015), incorporating three key designs: approximate preconditioning, distributed matrix multiplication, and Nesterov acceleration. The overall objective is to solve $K_\lambda W = Y$, where $K_\lambda = K + \lambda I$.

### Key Designs

1. **SAP Base Algorithm**: At each step, a row index set $\mathcal{B}$ of size $b$ is sampled, the subspace gradient $K_{\mathcal{B}n}W_t + \lambda I_\mathcal{B}W_t - Y_\mathcal{B}$ is computed, and preconditioning is applied via the subspace Hessian $(K_{\mathcal{B}\mathcal{B}} + \lambda I)^{-1}$. Unlike SDD, SAP incorporates second-order information. The per-step cost is $\mathcal{O}(bn + b^3)$, substantially lower than PCG's $\mathcal{O}(n^2)$. The design motivation is to strike a balance between SDD's scalability and PCG's robustness to ill-conditioning.

2. **Approximate Subspace Preconditioning**: The bottleneck of SAP is the $\mathcal{O}(b^3)$ exact factorization cost, which limits the feasible block size. ADASAP replaces the exact $K_{\mathcal{B}\mathcal{B}}$ with a rank-$r$ Nyström approximation $\hat{K}_{\mathcal{B}\mathcal{B}}$, reducing the preconditioner application cost from $\mathcal{O}(b^3)$ to $\mathcal{O}(br)$. This exploits the approximate low-rank structure of kernel matrices. On the taxi dataset, this enables block sizes of $b = 1.65 \times 10^5$.

3. **Distributed Matrix Multiplication**: The two bottleneck operations $K_{\mathcal{B}n}W_t$ ($\mathcal{O}(bn)$) and $K_{\mathcal{B}\mathcal{B}}\Omega$ ($\mathcal{O}(b^2r)$) are distributed across multiple GPUs via ColDistMatMat and RowDistMatMat. On the taxi dataset with 4 GPUs, a $3.4\times$ speedup is achieved, approaching perfect parallelism.

4. **Nesterov Acceleration**: Momentum parameters $\mu, \nu$ (defaulting to $\lambda$ and $n/b$, respectively) are introduced, and convergence is accelerated via a three-sequence update $(W, V, Z)$. Step sizes are computed automatically at each iteration, eliminating the need for manual tuning as required by SDD.

### Theoretical Contributions

Using the theory of determinantal point processes (DPP), the paper proves a two-phase convergence result for SAP along the top-$\ell$ spectral basis functions:

$$\mathbb{E}\|\text{proj}_\ell(\hat{m}_t) - \text{proj}_\ell(m_n)\|^2_\mathcal{H} \leq \min\left\{\frac{8\phi(b,\ell)}{t}, \left(1-\frac{1}{2\phi(b,n)}\right)^{t/2}\right\}\|y\|^2_{K_\lambda^{-1}}$$

where $\phi(b,\ell)$ is the smooth condition number. The first term is a condition number-free sublinear rate (analogous to but stronger than SGD), and the second term is an asymptotic linear rate. When the kernel matrix exhibits polynomial spectral decay, $\phi(b,\ell)=\mathcal{O}(1)$, meaning early-phase convergence is entirely independent of the condition number.

## Key Experimental Results

### Main Results: Large-Scale GP Inference

| Dataset | n | d | ADASAP RMSE | PCG RMSE | SDD-10 RMSE | Best |
|--------|---|---|------------|----------|-------------|--------|
| yolanda | 3.6×10⁵ | 100 | **0.795** | **0.795** | 0.801 | Tie |
| song | 4.6×10⁵ | 90 | **0.752** | **0.752** | 0.767 | Tie |
| benzene | 5.7×10⁵ | 66 | **0.012** | 0.141 | 0.112 | ADASAP |
| malonaldehyde | 8.9×10⁵ | 36 | **0.015** | 0.273 | Diverges | ADASAP |
| acsincome | 1.5×10⁶ | 9 | **0.789** | 0.875 | **0.792** | ADASAP |
| houseelec | 1.8×10⁶ | 9 | **0.027** | 1.278 | 0.119 | ADASAP |

### Ultra-Large-Scale Experiments

| Dataset | n | ADASAP RMSE | SDD-10 RMSE | PCG |
|--------|---|-------------|-------------|-----|
| taxi | 3.31×10⁸ | **0.50** | 0.52 | OOM |

ADASAP is 45% faster than SDD-10 (saving 14 hours); PCG cannot run due to out-of-memory errors.

### Ablation Study (Bayesian Optimization)

| Method | Improvement at length-scale=2.0 (%) | Improvement at length-scale=3.0 (%) |
|------|---------------------|---------------------|
| ADASAP | **10.42** | **13.86** |
| ADASAP-I (no preconditioning) | 7.04 | 11.27 |
| SDD-1 | 6.50 | 11.17 |
| PCG | 0.13 | 5.54 |

### Key Findings

- SDD is extremely sensitive to step size: SDD-100 diverges on all datasets, and performance varies substantially between SDD-1 and SDD-10.
- ADASAP works out of the box with default hyperparameters, requiring no manual tuning.
- Approximate preconditioning (ADASAP vs. ADASAP-I) yields significant gains on ill-conditioned problems (benzene: 0.012 vs. 0.168).
- This represents the first demonstration of exact GP inference at $>3\times10^8$ samples.

## Highlights & Insights

- Combining DPP theory with sketch-and-project yields the first condition number-free algorithm for posterior mean estimation.
- The definition of the "smooth condition number" $\phi(b, \ell)$ is elegant: block size $b$ and subspace dimension $\ell$ jointly govern convergence, enabling low-cost injection of second-order information.
- The default hyperparameter design is principled: acceleration parameters $\mu=\lambda$ and $\nu=n/b$ are derived directly from the problem structure.

## Limitations & Future Work

- Theoretical analysis does not yet cover ADASAP's approximate preconditioning, acceleration, or uniform sampling (currently only SAP is analyzed).
- GPU hardware is required, imposing hardware constraints.
- The potential of half-precision (float16) computation remains unexplored.
- Tail averaging underperforms in practice relative to theoretical expectations, indicating a gap between theory and empirics.

## Related Work & Insights

- Nyström approximation is widely used in kernel methods; its integration with sketch-and-project is a novel contribution of this work.
- SDD's step-size sensitivity is a genuine practical bottleneck; ADASAP's automatic step-size selection is a meaningful advantage.
- The approach can inspire other settings requiring large-scale kernel matrix solves, such as kernel ridge regression and kernel SVMs.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of DPP and SAP and the resulting condition number-free guarantee constitute a genuine theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage spans from small to ultra-large scale; the taxi dataset result is particularly impressive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, though the gap between theory and the actual algorithm could be discussed more explicitly.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the scalability bottleneck of GP inference with clear practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version](../../AAAI2026/llm_evaluation/streaming_generated_gaussian_process_experts_for_online_learning_and_control_ext.md)
- [\[ICCV 2025\] SketchSplat: 3D Edge Reconstruction via Differentiable Multi-view Sketch Splatting](../../ICCV2025/llm_evaluation/sketchsplat_3d_edge_reconstruction_via_differentiable_multi-view_sketch_splattin.md)
- [\[AAAI 2026\] OptScale: Probabilistic Optimality for Inference-time Scaling](../../AAAI2026/llm_evaluation/optscale_probabilistic_optimality_for_inference-time_scaling.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)

</div>

<!-- RELATED:END -->
