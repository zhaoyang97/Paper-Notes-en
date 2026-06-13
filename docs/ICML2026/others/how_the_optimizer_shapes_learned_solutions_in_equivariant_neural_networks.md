---
title: >-
  [Paper Note] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks
description: >-
  [ICML2026 (Workshop on Weight-Space Symmetries)][Muon] This paper systematically compares the training effects of Muon and Adam on equivariant/geometric networks (EGNN, DGCNN, PointNet, GotenNet…
tags:
  - "ICML2026 (Workshop on Weight-Space Symmetries)"
  - "Muon"
  - "Adam"
  - "Equivariance"
  - "Hessian"
  - "Spectral Rank"
  - "ModelNet40"
date: 2026-05-08
content_hash: 5e86a935581469a7
---

# How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks

**Conference**: ICML2026 (Workshop on Weight-Space Symmetries)  
**arXiv**: [2605.27662](https://arxiv.org/abs/2605.27662)  
**Code**: None  
**Area**: Equivariant Neural Networks / Optimizers / Loss Landscape Analysis  
**Keywords**: Muon, Adam, Equivariance, Hessian, Spectral Rank, ModelNet40

## TL;DR
This paper systematically compares the training effects of Muon and Adam on equivariant/geometric networks (EGNN, DGCNN, PointNet, GotenNet, GINE). It finds that Muon consistently outperforms Adam on 3D point cloud tasks. Furthermore, the solutions reached by Muon exhibit significant structural differences across three dimensions: Hessian curvature, local smoothness of the loss landscape, and weight/representation spectral rank—repositioning "optimizer selection" as a severely overlooked inductive bias in equivariant network training.

## Background & Motivation

**Background**: Equivariant neural networks gain inductive bias by encoding geometric symmetries directly into the architecture (e.g., $E(n)$ equivariance in EGNN, dynamic k-NN graphs in DGCNN, symmetric pooling in PointNet), which is a mainstream approach in geometric deep learning. However, it is increasingly acknowledged in experiments that hard equivariance constraints make optimization difficult. The loss landscape contains numerous critical points and even spurious minima (Xie & Smidt 2025), and equivariant networks often underperform their "relaxed" counterparts at scale (Xie et al. 2025; Brehmer et al. 2025).

**Limitations of Prior Work**: In response to this issue, the community's answer has almost exclusively been to "relax architectural constraints"—such as approximate equivariance (Wang et al. 2022) or relaxed equivariance (Pertigkiozoglou et al. 2024; Manolache et al. 2025; Elhag et al. 2025). All these works treat the **architecture** as the center of the optimization problem while treating the **optimizer** as a black box.

**Key Challenge**: However, recent research (e.g., Pascanu et al. 2025) emphasizes that different optimizers not only differ in convergence speed but also guide the network to converge to *qualitatively different* solutions. If training equivariant networks is an optimization problem, why hasn't anyone tried changing the optimizer?

**Goal**: Without modifying the architecture, this study replaces Adam with Muon (a new optimizer proposed by Jordan et al. 2024 that uses Newton-Schulz iteration to orthogonalize momentum). The goal is to measure the isolated effect of the optimizer and analyze where it pushes the solution within the loss landscape.

**Key Insight**: The core mechanism of Muon is to perform orthogonalized updates on 2D parameter momentum buffers. By design, this "promotes small but important directions," contrasting with the "automatic equalization" of Adam’s adaptive learning rate. The authors hypothesize that this spectral-level difference interacts interestingly with equivariant constraints.

**Core Idea**: Switching the optimizer (Adam → Muon) allows for consistent performance gains in equivariant networks. Moreover, Muon's solutions are more "spread out" in terms of the Hessian spectrum, local geometry of the loss, and weight/representation spectral rank, suggesting "prevention of spectral concentration" as a design principle for equivariant optimizers.

## Method

This paper is not a new algorithm paper but rather an empirical study + loss landscape analysis.

### Overall Architecture

The research pipeline consists of three steps: (1) Rigorous grid search + 4-seed replication for Adam and Muon on ModelNet40 / ModelNet40-C (3D point cloud classification) and QM9 / Peptides-func / ZINC (molecular learning) to obtain standardized accuracy comparisons; (2) Use of Hessian estimation (power iteration for the top eigenvalue, Hutchinson estimation for the trace) and 2D loss slices (Li et al. 2018 style) to characterize the local geometry of the two types of solutions; (3) Application of stable rank and effective rank statistics to describe the spectral structure of the trained weight matrices and internal representations.

### Key Designs

1.  **Architecture Coverage: Three Levels of Equivariance Intensity**:
    *   **Function**: Incorporates three levels of architecture—"Hard Equivariant $\to$ Weak Geometric $\to$ Pure Permutation Invariant"—to ensure the observed effects are not specific to a single architecture.
    *   **Mechanism**: EGNN (explicit $E(n)$ equivariance) serves as the hardest equivariant case; DGCNN (dynamic k-NN graphs, only permutation equivariance + local geometry) as the medium case; PointNet (symmetric pooling + complete permutation invariance) as the weakest case. Additionally, GotenNet ($E(3)$ equivariant Transformer) is used for molecular tasks and GINE (permutation equivariant message passing) for graph tasks. Muon parameters (Newton-Schulz iterations, spectral scaling) remain at defaults, while Adam uses the same grid search range (lr × wd) for fairness.
    *   **Design Motivation**: The strength of equivariance directly determines the amount of hidden parameter symmetry (Xie & Smidt 2025 indicate that hidden parameter symmetry shatters the loss landscape). If the optimizer effect is coupled with equivariance, cross-level comparisons should show varying degrees of improvement.

2.  **Local Geometry Characterization: Hessian Summary + 2D Slicing**:
    *   **Function**: Uses two complementary perspectives to depict "where Muon pushes the solution."
    *   **Mechanism**: For each set of 4-seed trained checkpoints, the one closest to the average accuracy is selected. Power iteration with autograd Hessian-vector products is used to calculate the maximum eigenvalue $\lambda_{\max}$, and Hutchinson estimation (with Rademacher probes) calculates the trace. Simultaneously, 2D loss contours are plotted in two filter-normalized directions according to Li et al. (2018) for an intuitive view of local shape.
    *   **Design Motivation**: Loss slices are low-dimensional projections and can be misleading; Hessian summaries can be contaminated by the "sharpness invariance under parameter symmetry" issue pointed out by Dinh et al. (2017). Viewing both simultaneously allows the paper to present the seemingly contradictory facts that "Muon solutions actually have higher local curvature" yet "look smoother on slices," avoiding biased conclusions from a single dimension.

3.  **Spectral Structure Analysis: Stable Rank and Effective Rank**:
    *   **Function**: Quantifies "how concentrated the singular value distribution is" at the weight and internal representation scales.
    *   **Mechanism**: For each weight matrix $W$ (and representation matrix for each layer), two values are computed: stable rank $\|W\|_F^2/\|W\|_2^2 = \sum_i \sigma_i^2/\sigma_1^2$; and effective rank $\exp(H(p))$, where $p_i = \sigma_i/\sum_j \sigma_j$ and $H$ is Shannon entropy. Both fall in $[1, \mathrm{rank}(W)]$, where higher values indicate a more uniform spectrum. Representations are mean-pooled at the point level for intermediate layers and pooled natively for the final layer.
    *   **Design Motivation**: Gradient descent is widely reported to have an "implicit low-rank bias" (Arora et al. 2019), whereas Muon's orthogonalized momentum rescales small singular directions. Observing that Muon's weights and representations are indeed more spectrally spread provides direct evidence of the "optimizer as a spectral inductive bias" and echoes failure modes involving rank collapse (e.g., Dong et al. 2021).

### Training Protocol
For each dataset × optimizer combination, a grid search for (learning rate, weight decay) was performed. After selecting the optimal configuration, it was repeated across 4 seeds, and the mean ± std was reported based on the best-checkpoint. This avoids misinterpreting "Muon helps you tune fewer parameters" as "Muon is inherently stronger."

## Key Experimental Results

### Main Results: Classification Accuracy on ModelNet40 and ModelNet40-C

| Setup | Architecture | Adam | Muon | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- |
| Clean | EGNN | 76.91 ± 0.94 | **82.08 ± 0.36** | **+5.17** |
| Clean | PointNet | 84.53 ± 0.70 | **87.21 ± 0.39** | **+2.67** |
| Clean | DGCNN | 87.10 ± 0.69 | **89.06 ± 0.17** | **+1.96** |
| Corrupted | EGNN | 65.76 ± 0.95 | **70.12 ± 0.10** | **+4.36** |
| Corrupted | PointNet | 72.85 ± 1.05 | **75.87 ± 0.28** | **+3.02** |
| Corrupted | DGCNN | 75.26 ± 1.63 | **77.84 ± 0.27** | **+2.58** |

Muon consistently improves performance across three architectures with different geometric inductive biases, and the std is generally smaller (indicating Muon solutions are more stable). The "hardest" equivariant network, EGNN, sees the largest gain (+5.17%), while the "weakest," DGCNN, sees the smallest (+1.96%), showing a trend where "stronger equivariance benefits more from Muon." On QM9 (GotenNet), Muon is superior on 11/12 targets. However, for GINE on Peptides-func / ZINC, the advantage disappears, indicating that the benefits are concentrated in 3D $SE(3)$-style equivariant tasks.

### Secondary Experiments: Hessian Estimation on ModelNet40 Checkpoints

| Metric | Architecture | Adam | Muon | Ratio |
| :--- | :--- | :--- | :--- | :--- |
| Top eigenvalue | EGNN | 27.14 | 128.83 | **4.75×** |
| Top eigenvalue | PointNet | 32.75 | 714.49 | **21.82×** |
| Top eigenvalue | DGCNN | 12.14 | 136.23 | **11.22×** |
| Trace | EGNN | 402.37 | 1472.78 | **3.66×** |
| Trace | PointNet | 482.61 | 7362.05 | **15.25×** |
| Trace | DGCNN | 184.47 | 1218.74 | **6.61×** |

The Hessian curvature ($\lambda_{\max}$ and trace) of Muon solutions is 4–22 times larger than that of Adam—directly contradicting the intuitive explanation that "Muon converges to flatter solutions."

### Key Findings
- **The "Smooth Slice vs. High Curvature" Paradox**: On 2D loss slices, the region around Muon is significantly smoother than Adam (most notable in PointNet), yet Hessian values show this is a low-dimensional projection illusion. Following the warning of Dinh et al. (2017), sharpness is not a functional invariant under parameter symmetry reparameterization; thus, the optimizer changes the "position of the checkpoint in the landscape" rather than the "inherent curvature of that location."
- **Spectral Anti-Low-Rank Bias**: Adam leaves concentrated spectra (implicit low-rank bias), whereas Muon's weight stable rank and effective rank are higher across most layers of all ModelNet40 architectures (especially every layer of EGNN). The representation spectrum follows the same trend, with the final layer effective rank ratio for EGNN being approximately 2×. This aligns with Muon's design goal of rescaling "rare directions."
- **3D vs. Graph Disparity**: Significant gains occur in 3D point cloud/molecular tasks ($SE(3)$ equivariance), while Muon lacks advantage in pure permutation equivariant graph tasks. This suggests the interaction between the optimizer and geometric inductive bias is task/symmetry dependent rather than a universal "free lunch."
- **EGNN Gains Most**: The strongest equivariant network, EGNN, shows the largest improvement (+5.17% Clean), echoing findings from Xie & Smidt (2025) about hidden symmetries partitioning the landscape. Muon seems more capable of reliably navigating to superior regions.

## Highlights & Insights
- **Returning the Optimizer to the Center of Equivariant Research**: While existing literature focuses on modifying architectures for relaxed equivariance, this work shows that merely swapping the optimizer leads to +2–5% gains, suggesting the community should not ignore optimizer choice as an independent dimension.
- **"High Curvature + Smooth Slice" is a Phenomenon Worth Investigating**: Sharpness metrics have been critiqued (Dinh et al. 2017). This paper reveals that visual smoothness and numerical curvature can move in opposite directions, providing an empirical case for developing sharpness/flatness metrics robust to parameter symmetry.
- **Spectral Rank as a Metric for Equivariant Optimization**: Muon expands the spectrum at both weight and representation levels, correlating with accuracy gains. This transforms "optimizer design principles" from "adaptive LR/momentum" to the actionable goal of "preventing spectral concentration."
- **Task/Symmetry Sensitivity**: 3D equivariance wins while graph equivariance does not. This suggests that the "equivariance + optimizer" combination needs to be discussed by symmetry group type, a clear direction for further research.

## Limitations & Future Work
- **Workshop Short Paper Scope**: Experiments are limited to ModelNet40 and small molecular datasets. The scale is far below where "whether equivariance still matters at scale" (Brehmer et al. 2025) is typically discussed.
- **Post-hoc Mechanism Analysis**: Spectral and Hessian analyses are performed on final checkpoints rather than tracking training dynamics; process evidence of "when and where Muon pushes the solution away" is missing.
- **Hyperparameter Fairness**: Although grid searches for (lr, wd) were conducted, Muon has other hyperparameters (Newton-Schulz iterations, spectral scale) not included in the search, potentially leaving tuning sensitivities.
- **Unexplained Muon Failure on Graph Tasks**: No mechanism is proposed for why Muon lacks advantage on Peptides-func / ZINC. Whether this relates to the specific geometry of GIN optimization or data scale remains to be explored.
- **No Causal Chain**: Observations of spectral rank, Hessian, and accuracy are correlational. No intervention experiments (e.g., manually constraining spectral rank to see the effect on accuracy) were performed to prove causality.

## Related Work & Insights
- **vs. Pertigkiozoglou et al. (2024) / Manolache et al. (2025)**: These works modify the architecture to help SGD find better minima; this paper changes the optimizer without modifying the objective. These are orthogonal and could potentially be combined.
- **vs. Xie & Smidt (2025)**: They identified landscape partitioning due to symmetry. This paper empirically shows Muon may favor "better" partitions, serving as an empirical corollary.
- **vs. Jordan et al. (2024)**: While Muon is a general optimizer, this work provides specific evidence that its "orthogonalized momentum preventing low-rankness" mechanism also holds in equivariant settings.
- **vs. Dinh et al. (2017)**: The "high Hessian + smooth slice" finding is a new instance of sharpness being unreliable under parameter symmetry, serving as a control sample for future sharpness-aware metric research.
- **vs. Arora et al. (2019)**: While they argue for the low-rank preference of gradient methods, this work shows Muon breaks this preference, acting as an optimizer-side counterexample.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically comparing Muon in equivariant networks and combining spectral rank + Hessian analysis is a first.
- Experimental Thoroughness: ⭐⭐⭐ 4 seeds + grid search is standard, but scale is small and graph conclusions are weak.
- Writing Quality: ⭐⭐⭐⭐ Compact for a short paper, with honest discussion of conclusions and uncertainties.
- Value: ⭐⭐⭐⭐ Repositions the "optimizer" in the equivariant research agenda and points toward clear directions for optimizer design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[ICLR 2026\] From Movement to Cognitive Maps: RNNs Reveal How Locomotor Development Shapes Hippocampal Spatial Coding](../../ICLR2026/others/from_movement_to_cognitive_maps.md)

</div>

<!-- RELATED:END -->
