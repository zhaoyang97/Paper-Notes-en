---
title: >-
  [Paper Note] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks
description: >-
  [ICML 2026][Others][Muon] This paper systematically compares the training effects of Muon versus Adam on equivariant/geometric networks (EGNN, DGCNN, PointNet, GotenNet, GINE). It finds that Muon consistently outperforms Adam on 3D point cloud tasks and that the converged solutions exhibit significant structural differences across three dimensi
tags:
  - ICML 2026
  - Others
  - Muon
  - Adam
  - Hessian
  - ModelNet40
date: 2026-05-08
content_hash: ad4e799d35b635b7
---
# How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks

**Conference**: ICML2026 (Workshop on Weight-Space Symmetries)  
**arXiv**: [2605.27662](https://arxiv.org/abs/2605.27662)  
**Code**: None  
**Area**: Equivariant Neural Networks / Optimizers / Loss Landscape Analysis  
**Keywords**: Muon, Adam, Equivariance, Hessian, Spectral Rank, ModelNet40

## TL;DR
This paper systematically compares the training effects of Muon versus Adam on equivariant/geometric networks (EGNN, DGCNN, PointNet, GotenNet, GINE). It finds that Muon consistently outperforms Adam on 3D point cloud tasks and that the converged solutions exhibit significant structural differences across three dimensions: Hessian curvature, local smoothness of the loss landscape, and the spectral rank of weights/representations. This work repositions "optimizer choice" as a severely neglected inductive bias in the training of equivariant networks.

## Background & Motivation

**Background**: Equivariant Neural Networks (ENNs) obtain inductive bias by baking geometric symmetries directly into the architecture (e.g., $E(n)$ equivariance in EGNN, dynamic k-NN geometric graphs in DGCNN, symmetric pooling in PointNet), representing a mainstream path in geometric deep learning. However, it is increasingly recognized in experiments that hard equivariance constraints make optimization difficult; the loss landscape contains numerous critical points and even spurious minima (Xie & Smidt 2025), and equivariant networks often struggle to compete with more relaxed counterparts at scale (Xie et al. 2025; Brehmer et al. 2025).

**Limitations of Prior Work**: In response to this problem, the community has almost exclusively focused on "relaxing architectural constraints"—approximate equivariance (Wang et al. 2022), relaxed equivariance (Pertigkiozoglou et al. 2024; Manolache et al. 2025; Elhag et al. 2025), etc. All these works treat the **architecture** as the center of the optimization problem while treating the **optimizer** as a black box.

**Key Challenge**: Recent research, such as Pascanu et al. (2025), emphasizes that different optimizers not only differ in convergence speed but also guide networks to converge to *qualitatively different* solutions. If the difficulty of training equivariant networks is an optimization problem, why has no one tried changing the optimizer?

**Goal**: Without modifying the architecture, this study replaces Adam with Muon (a new optimizer proposed by Jordan et al. (2024) that orthogonalizes momentum using Newton-Schulz iterations) to measure the impact of the optimizer alone and analyze where it pushes solutions within the loss landscape.

**Key Insight**: The core mechanism of Muon is to perform orthogonalized updates on the momentum buffer for 2D parameters. By design, this "promotes small but important directions," forming a contrast with Adam’s "automatic equalization" via adaptive learning rates. The authors hypothesize that this spectral-level difference interacts interestingly with equivariant constraints.

**Core Idea**: Simply switching the optimizer (Adam $\rightarrow$ Muon) can consistently improve performance in equivariant networks. Furthermore, Muon's solutions are more "spread out" in terms of the Hessian spectrum, local loss geometry, and weight/representation spectral rank, suggesting that "preventing spectral concentration" should be used as a design principle for equivariant optimizers.

## Method

This paper is not a new algorithm paper but rather an empirical study + loss landscape analysis.

### Overall Architecture

The research process is divided into three steps: (1) Rigorous grid search + 4-seed replication for Adam and Muon on ModelNet40 / ModelNet40-C (3D point cloud classification) and QM9 / Peptides-func / ZINC (molecular learning) to obtain unified accuracy comparisons; (2) Characterization of the local geometry of both types of solutions using Hessian estimation (power iteration for top eigenvalue, Hutchinson estimation for trace) and 2D loss slices (Li et al. 2018 style); (3) Description of the spectral structure of trained weight matrices and intermediate representations using two statistics: stable rank and effective rank.

### Key Designs

**1. Architecture Coverage: Spanning three levels of equivariance strength to ensure effects are not accidental to a single architecture**

If improvements were seen by changing the optimizer on only one architecture, it would be difficult to rule out the possibility that it was "just a property of that network." The authors intentionally included three levels of equivariance strength: EGNN (explicit $E(n)$ equivariance) as the hardest end, DGCNN (dynamic k-NN graphs, permutation equivariance + local geometry only) as the medium level, and PointNet (symmetric pooling, fully permutation invariant) as the weakest end. For molecular tasks, GotenNet ($E(3)$ equivariant Transformer) is used, and for graph tasks, GINE (permutation equivariant message passing) is utilized. To ensure fairness, Muon uses the default settings (Newton-Schulz iterations, spectral scaling constant), and Adam uses the same (lr × wd) grid search range. This design is chosen because the strength of equivariance directly determines the number of hidden symmetries in the parameter space—Xie & Smidt (2025) noted that hidden parameter symmetries can tear the loss landscape—if the optimizer effect is indeed coupled with equivariance, cross-level comparisons should show varying degrees of improvement. The results confirmed this: "the stronger the equivariance, the more Muon benefits."

**2. Local Loss Geometry Characterization: Combining Hessian summary + 2D slices to avoid single-dimension misinterpretation**

To answer "where Muon pushes solutions in the landscape," looking at a single perspective can be misleading. For each checkpoint trained across 4 seeds that is closest to median accuracy, the authors simultaneously perform two actions: run power iteration using autograd Hessian-vector products to calculate the maximum eigenvalue $\lambda_{\max}$ and estimate the trace using Hutchinson (Rademacher probes); then plot 2D loss contours along two filter-normalized directions as in Li et al. (2018). Looking at both is rational—loss slices are low-dimensional projections and can be misleading, while Hessian summaries can be contaminated by "sharpness not being a function-level invariant under parameter symmetry," as noted by Dinh et al. (2017). Juxtaposing the two reveals the seemingly contradictory facts that "Muon's solutions actually have higher local curvature" while "appearing smoother on slices," thereby avoiding biased conclusions from a single dimension.

**3. Spectral Structure Analysis: Quantifying spectral concentration via stable rank and effective rank**

Finally, to verify the hypothesis that "the optimizer is a spectral inductive bias," the concentration of the singular value distribution must be quantified. For each weight matrix $W$ (and each layer's intermediate activation feature matrix), the authors calculate two values: stable rank $\|W\|_F^2/\|W\|_2^2=\sum_i\sigma_i^2/\sigma_1^2$ and effective rank $\exp(H(p))$ (where $p_i=\sigma_i/\sum_j\sigma_j$ and $H$ is Shannon entropy). Both fall in the range $[1,\mathrm{rank}(W)]$, with larger values indicating a more uniform spectrum. Representations are mean-pooled at the point level for intermediate layers and pooled using the architecture's native method for the final layer. This metric strikes the heart of the issue: gradient descent is widely reported to have an "implicit low-rank bias" (Arora et al. 2019), whereas Muon's orthogonalized momentum specifically rescales small singular directions. Observing that Muon-trained weights and representation spectra are indeed more "spread out" provides direct evidence for "optimizers as spectral inductive bias" and echoes the degradation failure modes of "pure attention dropping exponentially to rank-1" discussed by Dong et al. (2021).

### Training Protocol
For each dataset × optimizer combination, a (learning rate, weight decay) grid search is performed. After selecting the optimal configuration, it is repeated with 4 seeds, and the mean ± std is reported according to the best-checkpoint. This avoids the confusion of misinterpreting "Muon helped you tune fewer parameters" as "Muon is inherently stronger."

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

Muon consistently improves performance across three architectures with different geometric inductive biases, and the std is generally smaller (indicating Muon's solutions are more stable). The "hardest" equivariant model, EGNN, shows the largest improvement (+5.17%), while the "weakest," DGCNN, shows the smallest (+1.96%), demonstrating a trend where "the stronger the equivariance, the more Muon benefits." On QM9 (GotenNet), Muon outperformed Adam on 11/12 targets (e.g., r2 dropped from 0.4320 to 0.2310), but on Peptides-func / ZINC with GINE, Muon's advantage disappeared or worsened, indicating that the benefit is primarily concentrated in 3D $SE(3)$-style equivariant tasks.

### Secondary Results: Hessian Estimation on ModelNet40 Checkpoints

| Metric | Architecture | Adam | Muon | Ratio |
|------|------|------|------|------|
| Top eigenvalue | EGNN | 27.14 | 128.83 | **4.75×** |
| Top eigenvalue | PointNet | 32.75 | 714.49 | **21.82×** |
| Top eigenvalue | DGCNN | 12.14 | 136.23 | **11.22×** |
| Trace | EGNN | 402.37 | 1472.78 | **3.66×** |
| Trace | PointNet | 482.61 | 7362.05 | **15.25×** |
| Trace | DGCNN | 184.47 | 1218.74 | **6.61×** |

The Hessian curvature ($\lambda_{\max}$ and trace) of Muon's solutions is actually 4–22 times larger than Adam's—this directly refutes the intuitive explanation that "Muon converges to flatter solutions."

### Key Findings
- **The Paradox of "Smooth Slices vs. High Curvature"**: Muon's surroundings are significantly smoother than Adam's on 2D loss slices (most notable in PointNet), but Hessian values indicate this is merely a low-dimensional projection illusion. Consistent with the warning by Dinh et al. (2017), sharpness is not a function-level invariant under parameter symmetry reparameterization; thus, the optimizer changes the "location of the checkpoint in the landscape" rather than the "intrinsic curvature of that location."
- **Spectral Structure Against Low-Rank Bias**: Adam leaves a concentrated spectrum (implicit low-rank bias), whereas Muon's weight stable rank and effective rank are higher across most layers for all ModelNet40 architectures (and higher in every layer for EGNN). Representation spectra follow the same trend, with the effective rank of EGNN’s final layer being approximately 2× higher. This aligns with the original intent of Muon (Jordan et al. 2024) to rescale "rare directions."
- **3D vs. Graph Differences**: All significant improvements occurred in 3D point cloud / molecular tasks (involving $SE(3)$ equivariance), while Muon had no advantage in pure permutation equivariance graph tasks. This suggests that the interaction between optimizers and geometric inductive bias is task/symmetry-type dependent rather than a universal free lunch.
- **EGNN Gains the Most**: The sharpest improvement in performance (+5.17% Clean, +4.36% Corrupted) was observed in the most strictly equivariant EGNN. This echoes the finding by Xie & Smidt (2025) that "hidden parameter symmetry cuts the loss landscape into multiple regions"—Muon seems more reliably able to reach the superior regions.

## Highlights & Insights
- **Repositioning "Optimizer" at the Center of ENN Research**: While the relaxed equivariance literature focuses on modifying architectures, this paper demonstrates that simply switching the optimizer (leaving the architecture unchanged) can yield stable gains of +2~5%, suggesting that the community should no longer ignore optimizer choice as an independent dimension.
- **"High Curvature + Smooth Slices" is a Phenomenon Worth Pursuing**: Since single-dimension sharpness metrics have been debunked by Dinh et al. (2017), this paper reveals that "slice smoothness" and "Hessian curvature" can move in opposite directions, providing a concrete empirical case for designing sharpness/flatness metrics that are "robust to parameter symmetry."
- **Spectral Rank May Be a Key Metric for Equivariant Optimization**: Muon causes the spectrum to unfold at both the weight and representation levels, which correlates positively with accuracy gains. This shifts "optimizer design principles" from "adaptive lr/momentum" to the concrete, actionable goal of "preventing spectral concentration."
- **Task/Symmetry-Type Sensitivity**: The success in 3D equivariance versus the lack thereof in graph tasks indicates that "equivariance + optimizer" combinations need to be discussed according to the type of symmetry group, providing a clear path for future research (different symmetry groups likely require different spectral behaviors from the optimizer).

## Limitations & Future Work
- **Workshop Short Paper status**: The experimental scope is focused on ModelNet40 + limited molecular data; the scale is an order of magnitude smaller than the discussions in Brehmer et al. (2025) regarding whether equivariance still matters at scale.
- **Post-hoc Mechanism Analysis**: All spectral and Hessian analyses are performed on checkpoints without tracking training dynamics; process-oriented evidence of "when and where Muon pushes the solution away" is missing.
- **Optimizer Hyperparameter Fairness**: Although (lr, wd) grid search was conducted, Muon has independent hyperparameters like Newton-Schulz iteration counts and spectral scaling constants that were not searched, potentially leaving tuning sensitivity issues.
- **Unexplained Muon Failure on Graph Tasks**: Muon showed no advantage on Peptides-func / ZINC. The authors acknowledge this observation but provide no mechanistic hypothesis; it remains to be seen if this relates to the specific optimization geometry of permutation-equivariant GIN or data scale issues.
- **Lack of Causal Links**: Observations of spectral rank, Hessian, and accuracy are correlational only; no intervention experiments (e.g., manually constraining spectral rank to see the effect on accuracy) were performed to prove the direction of causality.

## Related Work & Insights
- **vs. Pertigkiozoglou et al. (2024) / Manolache et al. (2025) / Elhag et al. (2025)**: These works help SGD find better minima by relaxing equivariance to change the training problem. This paper, conversely, changes only the optimizer without altering the training objective; the two paths are orthogonal and complementary and could, in principle, be stacked.
- **vs. Xie & Smidt (2025)**: They noted that the equivariant loss landscape is segmented due to hidden parameter symmetries. This paper empirically shows that Muon seems more inclined to reach the "good" regions, providing an empirical corollary to their theoretical framework.
- **vs. Jordan et al. (2024)**: While the Muon paper introduced a general-purpose optimizer, this work specifically applies it to equivariant networks and provides evidence that the "orthogonalized momentum preventing low-rankness" mechanism holds true for equivariant models.
- **vs. Dinh et al. (2017)**: The "high Hessian + smooth slice" observation in this paper serves as a new concrete case of Dinh et al.’s argument that sharpness is unreliable under parameter symmetry, making it a useful baseline for future research on improved sharpness-aware metrics.
- **vs. Arora et al. (2019)**: While they argue for the implicit low-rank bias of gradient methods, this paper empirically demonstrates that Muon’s orthogonalized momentum breaks this bias, serving as an optimizer-side counterexample to that line of research.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically comparing Muon in the context of equivariant networks and analyzing spectral rank alongside Hessian is a first.
- Experimental Thoroughness: ⭐⭐⭐ Standard 4-seed + grid search is present, but the scale is small and conclusions on graph tasks are weak.
- Writing Quality: ⭐⭐⭐⭐ The short paper is compact, and both conclusions and uncertainties are honestly addressed.
- Value: ⭐⭐⭐⭐ Refocuses equivariant network research on "optimizers" and points toward a clear direction for subsequent optimizer design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)

</div>

<!-- RELATED:END -->
