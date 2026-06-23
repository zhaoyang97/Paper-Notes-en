---
title: >-
  [Paper Note] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks
description: >-
  [ICML 2026][Others][Muon] This paper systematically compares the training performance of Muon and Adam on equivariant/geometric networks (EGNN, DGCNN, PointNet, GotenNet, GINE). It finds that Muon consistently outperforms Adam on 3D point cloud tasks and that the solutions converged upon exhibit significant structural differences across three d
tags:
  - ICML 2026
  - Others
  - Muon
  - Adam
  - Hessian
  - ModelNet40
date: 2026-05-08
content_hash: dd4589a9c2b76f15
---
# How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks

**Conference**: ICML2026 (Workshop on Weight-Space Symmetries)  
**arXiv**: [2605.27662](https://arxiv.org/abs/2605.27662)  
**Code**: None  
**Area**: Equivariant Neural Networks / Optimizers / Loss Landscape Analysis  
**Keywords**: Muon, Adam, Equivariance, Hessian, Spectral Rank, ModelNet40

## TL;DR
This paper systematically compares the training performance of Muon and Adam on equivariant/geometric networks (EGNN, DGCNN, PointNet, GotenNet, GINE). It finds that Muon consistently outperforms Adam on 3D point cloud tasks and that the solutions converged upon exhibit significant structural differences across three dimensions: Hessian curvature, local smoothness of the loss landscape, and the spectral rank of weights/representations. This work repositions "optimizer choice" as a severely neglected inductive bias in the training of equivariant networks.

## Background & Motivation

**Background**: Equivariant Neural Networks (ENNs) obtain inductive bias by baking geometric symmetries directly into the architecture (e.g., $E(n)$ equivariance in EGNN, dynamic k-NN graphs in DGCNN, and symmetric pooling in PointNet). However, it is increasingly recognized that hard equivariance constraints make optimization difficult, leading to many critical points or even spurious minima in the loss landscape (Xie & Smidt 2025). Consequently, equivariant networks often struggle to scale compared to more relaxed counterparts (Xie et al. 2025; Brehmer et al. 2025).

**Limitations of Prior Work**: To address this, the community has largely focused on "relaxing architectural constraints"—using methods like approximate equivariance (Wang et al. 2022) or relaxed equivariance (Pertigkiozoglou et al. 2024; Manolache et al. 2025; Elhag et al. 2025). All these works treat the **architecture** as the center of the optimization problem while treating the **optimizer** as a black box.

**Key Challenge**: Recent research, such as Pascanu et al. (2025), emphasizes that different optimizers do not just differ in convergence speed; they guide networks to *qualitatively different* solutions. If the difficulty in training ENNs is an optimization problem, why has switching the optimizer not been explored?

**Goal**: Without modifying the architecture, this study replaces Adam with Muon (a new optimizer proposed by Jordan et al. 2024 that orthogonalizes momentum using Newton-Schulz iterations) to measure the impact of the optimizer alone and analyze where it pushes solutions within the loss landscape.

**Key Insight**: The core mechanism of Muon is to perform orthogonalized updates on the momentum buffers of 2D parameters. This design "promotes small but important directions," contrasting with the "automatic equalization" of Adam's adaptive learning rate. The authors hypothesize that this spectral-level difference interacts interestingly with equivariant constraints.

**Core Idea**: Simply changing the optimizer (Adam → Muon) leads to stable performance gains in ENNs. Furthermore, Muon's solutions are more "spread out" in terms of the Hessian spectrum, local geometry, and weight/representation spectral rank. This suggests that "preventing spectral concentration" should be a design principle for equivariant optimizers.

## Method

This paper is an empirical study and loss landscape analysis rather than a new algorithm paper.

### Overall Architecture

The research process follows three steps: (1) Conduct rigorous grid searches and 4-seed repetitions for Adam and Muon on ModelNet40/ModelNet40-C (3D point cloud classification) and QM9/Peptides-func/ZINC (molecular learning) to obtain baseline accuracy comparisons; (2) Characterize local geometry using Hessian estimation (power iteration for the top eigenvalue, Hutchinson estimation for the trace) and 2D loss slices (style of Li et al. 2018); (3) Describe the spectral structure of trained weight matrices and intermediate representations using stable rank and effective rank.

### Key Designs

**1. Architectural Coverage: Spanning three levels of equivariance strength to ensure the effect is not accidental.**

To ensure the benefits are not architecture-specific, the authors include three levels of equivariance: EGNN (explicit $E(n)$ equivariance) as the "hardest," DGCNN (dynamic k-NN graphs, permutation equivariance + local geometry) as the "medium," and PointNet (symmetric pooling, global permutation invariance) as the "weakest." GotenNet ($E(3)$ equivariant Transformer) is used for molecular tasks, and GINE (permutation equivariant message passing) is used for graph tasks. For fairness, Muon uses default settings (Newton-Schulz iterations, spectral scaling), and Adam uses the same (lr × wd) search range. This design is motivated by the fact that equivariance strength determines the amount of hidden symmetry in the parameter space—which Xie & Smidt (2025) noted can "tear" the loss landscape. Results shows that the stronger the equivariance, the more the model benefits from Muon.

**2. Local Geometry Characterization: Combining Hessian summaries and 2D slices to avoid single-dimension bias.**

To understand where Muon pushes solutions, the authors examine the 4-seed checkpoint closest to average accuracy. They calculate the maximum eigenvalue $\lambda_{\max}$ via power iteration and estimate the trace via Hutchinson (Rademacher probes). Simultaneously, they plot 2D loss contours along two filter-normalized directions. Combining these is crucial because loss slices are low-dimensional projections, while Hessian summaries can be affected by the fact that sharpness is not a functional invariant under parameter symmetries (Dinh et al. 2017). This dual approach reveals the paradox that Muon solutions have higher local curvature yet appear smoother on slices.

**3. Spectral Structure Analysis: Quantifying spectral concentration using stable and effective rank.**

To test if the "optimizer is a spectral inductive bias," the authors quantify the concentration of singular value distributions. For each weight matrix $W$, they calculate stable rank $\|W\|_F^2/\|W\|_2^2=\sum_i\sigma_i^2/\sigma_1^2$ and effective rank $\exp(H(p))$ (where $p_i=\sigma_i/\sum_j\sigma_j$ and $H$ is Shannon entropy). Both fall in the range $[1,\mathrm{rank}(W)]$, where higher values indicate more uniform spectra. Representations are mean-pooled at the point level for intermediate layers. This directly addresses the "low-rank implicit bias" commonly reported for gradient descent (Arora et al. 2019), as Muon’s orthogonalized momentum rescales small singular directions.

### Loss & Training
Grid searches were performed for (learning rate, weight decay) for every dataset-optimizer pair. Best configurations were repeated with 4 seeds, reporting mean ± std based on the best-checkpoint. This ensures Muon's inherent strength is evaluated rather than just its ease of tuning.

## Key Experimental Results

### Main Results: Classification Accuracy on ModelNet40 and ModelNet40-C

| Setup | Architecture | Adam | Muon | $\Delta$ |
|------|------|------|------|----------|
| Clean | EGNN | 76.91 ± 0.94 | **82.08 ± 0.36** | **+5.17** |
| Clean | PointNet | 84.53 ± 0.70 | **87.21 ± 0.39** | **+2.67** |
| Clean | DGCNN | 87.10 ± 0.69 | **89.06 ± 0.17** | **+1.96** |
| Corrupted | EGNN | 65.76 ± 0.95 | **70.12 ± 0.10** | **+4.36** |
| Corrupted | PointNet | 72.85 ± 1.05 | **75.87 ± 0.28** | **+3.02** |
| Corrupted | DGCNN | 75.26 ± 1.63 | **77.84 ± 0.27** | **+2.58** |

Muon consistently improves performance across architectures with different geometric biases. The strongest equivariant model, EGNN, saw the largest gain (+5.17%), while the weakest, DGCNN, saw the smallest (+1.96%). On QM9 (GotenNet), Muon was superior in 11/12 targets. However, on Peptides-func/ZINC using GINE, Muon's advantage disappeared, suggesting the benefit is centered on 3D $SE(3)$-style equivariant tasks.

### Hessian Estimation on ModelNet40 Checkpoints

| Metric | Architecture | Adam | Muon | Ratio |
|------|------|------|------|------|
| Top eigenvalue | EGNN | 27.14 | 128.83 | **4.75×** |
| Top eigenvalue | PointNet | 32.75 | 714.49 | **21.82×** |
| Top eigenvalue | DGCNN | 12.14 | 136.23 | **11.22×** |
| Trace | EGNN | 402.37 | 1472.78 | **3.66×** |
| Trace | PointNet | 482.61 | 7362.05 | **15.25×** |
| Trace | DGCNN | 184.47 | 1218.74 | **6.61×** |

The Hessian curvature ($\lambda_{\max}$ and trace) of Muon solutions is 4–22 times larger than Adam's, contradicting the intuition that Muon converges to "flatter" solutions.

### Key Findings
- **The "Smooth Slice vs. High Curvature" Paradox**: While 2D loss slices around Muon solutions look smoother than Adam's, the Hessian values indicate higher curvature. Per Dinh et al. (2017), sharpness is not invariant under parameter symmetry; the optimizer likely changes the checkpoint's *position* in the landscape rather than the "intrinsic curvature" of that location.
- **Counter-Low-Rank Bias**: Adam leaves a concentrated spectrum (implicit low-rank bias), whereas Muon's weights show higher stable and effective rank across most layers. In EGNN, every layer showed higher spectral rank. The effective rank of EGNN's final layer representation was ~2× higher with Muon.
- **3D vs. Graph Disparity**: Significant gains occurred in 3D point cloud/molecular tasks ($SE(3)$ equivariance), but not in graph tasks with pure permutation equivariance, suggesting the optimizer-bias interaction is task/symmetry dependent.
- **EGNN Benefit**: The highest gains in the "hardest" equivariant model (EGNN) echo Xie & Smidt's (2025) findings on how hidden symmetries fragment the landscape; Muon appears better at navigating to superior regions.

## Highlights & Insights
- **Repositioning the Optimizer**: While most literature focuses on altering architecture (relaxed equivariance), this work shows +2-5% gains simply by changing the optimizer, which should not be ignored.
- **Visual vs. Numerical Sharpness**: This study provides an empirical case where slice smoothness and Hessian curvature move in opposite directions, highlighting the need for "symmetry-robust" sharpness measures.
- **Spectral Rank as a Metric**: Muon promotes spectral expansion in both weights and representations, correlating with accuracy gains. This serves as an actionable target: "preventing spectral concentration."
- **Symmetry-Optimizer Sensitivity**: The success in 3D versus the failure in graphs suggest that optimizer choice should be tailored to the specific symmetry group.

## Limitations & Future Work
- **Scope**: As a workshop paper, it focuses on ModelNet40 and limited molecular data; it lacks the scale to address whether equivariance still matters at massive scales (Brehmer et al. 2025).
- **Post-hoc Analysis**: Analysis was done on checkpoints rather than tracking training dynamics; process-oriented evidence of "when and how" Muon pushes solutions is missing.
- **Hyperparameter Fairness**: Despite grid searches for (lr, wd), other Muon hyperparameters (Newton-Schulz iterations) were not searched.
- **Graph Failure**: The failure on Peptides-func/ZINC is noted but not mechanistically explained.
- **Causality**: Observations of spectral rank, Hessian, and accuracy are correlational; intervention experiments (e.g., constraining rank) are needed to prove causality.

## Related Work & Insights
- **vs. Relaxed Equivariance (Pertigkiozoglou et al. 2024, etc.)**: These works modify the objective; this work modifies the optimizer. They are orthogonal and could potentially be combined.
- **vs. Xie & Smidt (2025)**: Their theory on fragmented landscapes finds an empirical corollary here, where Muon seems to find better "regions."
- **vs. Jordan et al. (2024)**: This study extends the original general-purpose Muon paper to the specific regime of ENNs.
- **vs. Arora et al. (2019)**: Provides a counterexample to the "low-rank implicit bias" of standard gradient methods.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic comparison of Muon in ENNs combining spectral and Hessian analysis.
- Experimental Thoroughness: ⭐⭐⭐ Standard grid search and multi-seed runs, but scale and graph-task explanations are limited.
- Writing Quality: ⭐⭐⭐⭐ Concise, with honest assessment of conclusions and uncertainties.
- Value: ⭐⭐⭐⭐ Re-integrates "optimizers" into the ENN research agenda and provides clear future directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)

</div>

<!-- RELATED:END -->
