---
title: >-
  [Paper Note] Quantile-Free Uncertainty Quantification in Graph Neural Networks
description: >-
  [ICML 2026][Graph Learning][GNN] QpiGNN proposes a "quantile-free, post-hoc-free" GNN node-level prediction interval framework, using a dual-head GNN (one head predicts the mean…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "GNN"
  - "Prediction Interval"
  - "Quantile Regression"
  - "Dual-head Architecture"
  - "Label-only Loss"
date: 2026-05-08
content_hash: af5dffa662204e24
---

# Quantile-Free Uncertainty Quantification in Graph Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2605.04847](https://arxiv.org/abs/2605.04847)  
**Code**: Available (paper marks anonymous.4open.science/r/QpiGNN-30808)  
**Area**: Graph Neural Networks / Uncertainty Quantification / Node Regression  
**Keywords**: GNN, Prediction Interval, Quantile Regression, Dual-head Architecture, Label-only Loss

## TL;DR
QpiGNN proposes a "quantile-free, post-hoc-free" GNN node-level prediction interval framework, using a dual-head GNN (one head predicts the mean, the other predicts the half-width) combined with a label-level joint loss that directly optimizes "coverage + interval width." Across 19 synthetic/real datasets, it achieves an average 22% improvement in coverage and a 50% reduction in interval width.

## Background & Motivation

**Background**: Node regression GNNs are widely used in high-risk domains such as healthcare and criminal justice, but most GNNs only provide point estimates without uncertainty quantification. Existing UQ methods fall into two categories: Bayesian (VI, posterior approximation, which scale poorly and are sensitive to priors) and frequentist (resampling like ensembles, post-hoc calibration like Conformal Prediction). Frequentist methods are computationally expensive and often rely on the **exchangeability** assumption—which almost never holds for graph data with structural dependencies.

**Limitations of Prior Work**: Quantile Regression (QR) appears to be a good choice to bypass distributional assumptions, but standard QR requires the quantile level $\tau$ as input or a separate model for each $\tau$, leading to issues like "quantile crossing" (lower quantile predictions exceeding higher ones). SQR learns multiple quantiles in one model, RQR uses a width-regularized loss for MLPs to estimate center+spread, but **these approaches collapse when directly applied to GNNs**: message passing causes oversmoothing of node representations, SQR is unstable and poorly calibrated on graphs, and RQR’s single-head design causes gradient interference between center and spread.

**Key Challenge**: The bottleneck of QR methods is "quantile input + single-head representation," which structurally conflicts with GNNs’ "neighborhood aggregation induces global smoothing." To leverage GNNs’ relational modeling while achieving node-level adaptive and compact intervals, "prediction" and "uncertainty" must be decoupled both architecturally and in supervision.

**Goal**: (i) Design a GNN UQ framework that does not rely on quantile input or post-hoc calibration; (ii) Provide theoretical guarantees for coverage and width under graph dependencies; (iii) Achieve both calibration and compactness.

**Key Insight**: The authors observe that RQR can use a "label-only" loss to directly learn input-dependent bounds on MLPs, and that QR’s "quantile input" can actually be bypassed; on GNNs, the root cause of oversmoothing is single-head sharing, so **dual-head decoupling + direct label supervision** can simultaneously resolve both issues.

**Core Idea**: Use a **dual-head GNN (one head predicts $\hat y$, one predicts half-width $\hat d$)** + **quantile-free joint loss** (directly penalizing "$\hat c$ deviating from $1-\alpha$" and "average interval width"), requiring neither quantile input nor post-processing.

## Method

### Overall Architecture
Given a graph $G=(\mathcal V,\mathcal E)$ and node features $\mathbf X$, QpiGNN uses a shared GNN encoder to compute node embeddings $\mathbf H=\text{GNN}(\mathbf X,\mathcal E)$, followed by two linear heads: prediction head $\hat{\mathbf y}=\mathbf W_{\text{pred}}\mathbf H+\mathbf b_{\text{pred}}$, and half-width head $\hat{\mathbf d}=\text{Softplus}(\mathbf W_{\text{diff}}\mathbf H+\mathbf b_{\text{diff}})$. The final prediction interval is $[\hat y_v-\hat d_v,\ \hat y_v+\hat d_v]$. Training uses a three-part joint loss: coverage squared error + violation penalty + width penalty, directly supervised by label $y_v$. At inference, a single forward pass yields calibrated node-level intervals, with no need for a calibration set or conformal post-processing.

### Key Designs

1. **Dual-head GNN decouples prediction and uncertainty**:

    - **Function**: Allows $\hat y$ and $\hat d$ to learn targeted representations (one for accuracy, one for coverage), avoiding oversmoothing and gradient conflict from shared representations.
    - **Mechanism**: Shared GNN encoder $\mathbf H$, with two separate linear heads. The half-width head uses Softplus to ensure $\hat d>0$, making intervals naturally well-ordered (no more quantile crossing). This design echoes the successful "separate heads for different signals" approach in heteroscedastic/Bayesian regression (Kendall & Gal, Lakshminarayanan et al.), but is more lightweight.
    - **Design Motivation**: On graphs, node representations are repeatedly averaged via message passing; single-head models inevitably push both center and spread toward local means, undermining node-level adaptivity. Structural decoupling allows the spread head to learn a function class entirely different from the center—e.g., naturally giving wider intervals at hub nodes.

2. **Quantile-free joint loss directly supervises coverage and width**:

    - **Function**: Removes "quantile input" and "post-hoc calibration," using label $y_v$ to simultaneously calibrate coverage and compress width in one step.
    - **Mechanism**: $\mathcal L_{\text{total}}=\underbrace{(\hat c-(1-\alpha))^2 + \hat\ell_{\text{viol}}}_{\mathcal L_{\text{coverage}}} + \underbrace{\lambda_{\text{width}}\cdot\mathbb E_v[\hat y_v^{\text{up}}-\hat y_v^{\text{low}}]}_{\mathcal L_{\text{width}}}$. Here, $\hat c=\mathbb P(\hat y_v^{\text{low}}\le y_v\le \hat y_v^{\text{up}})$ is empirical coverage; $\hat\ell_{\text{viol}}=\mathbb E[|y_v-\hat y_v^{\text{low}}|\cdot\mathds 1[y_v<\hat y_v^{\text{low}}]+|y_v-\hat y_v^{\text{up}}|\cdot\mathds 1[y_v>\hat y_v^{\text{up}}]]$ provides fine-grained gradients for violating nodes; width penalty uses L1 form to avoid L2 instability under outliers. $\lambda_{\text{width}}\in [0.2,0.5]$ is selected via Bayesian optimization.
    - **Design Motivation**: RQR-W entangles coverage and width into a single conditional loss, which on GNNs is pushed by oversmoothing into globally wide intervals. QpiGNN decouples them into additive terms: first pulls $\hat c$ to the target $1-\alpha$, then compresses width while maintaining coverage. This "Lagrangian relaxation" perspective gives $\lambda_{\text{width}}$ a clear interpretation.

3. **Asymptotic + finite-sample coverage guarantees**:

    - **Function**: Provides provable coverage convergence even on graph data violating i.i.d./exchangeability.
    - **Mechanism**: Proposition 4.1 assumes noise $\varepsilon_v$ is bounded and weakly dependent, $\hat y_v$ and $\hat d_v$ converge in probability to targets, and node embeddings are sufficiently diverse, then $\hat c\xrightarrow{P}1-\alpha$ (WLLN). For finite samples, McDiarmid/Hoeffding inequalities apply: single-node perturbation affects coverage estimate by at most $1/N+\delta_G$, so $|\hat c-(1-\alpha)|=\mathcal O(1/\sqrt N)$. Under symmetric $P(y\mid x_v)$, the minimal width satisfies $d_v^*=F_v^{-1}(1-\alpha/2)$, and the loss is interpreted as a Lagrangian relaxation of this constraint optimization.
    - **Design Motivation**: CP’s coverage guarantee relies on exchangeability; QpiGNN builds its guarantee on "approximate bounded-difference under neighborhood smoothing," better suited for graph data.

### Loss & Training
End-to-end SGD training, with the loss as the weighted sum of the three terms above; diminishing learning rate ensures convergence to a stationary point under non-convexity. $\alpha$ is typically set to 0.1 (90% target coverage), $\lambda_{\text{width}}\in[0.2,0.5]$ is selected by BO; for comparison, a GNN variant of RQR is also implemented, with an ordering penalty $\gamma_{\text{order}}\cdot\text{ReLU}(\hat y^{\text{low}}-\hat y^{\text{up}})$ to mitigate quantile crossing.

## Key Experimental Results

### Main Results
On 19 datasets (9 synthetic structures such as BA/ER/Grid/Tree + real datasets), using PICP (empirical coverage) and MPIW (mean prediction interval width) as metrics, with a target coverage of 90%:

| Dataset (Synthetic) | Model | PICP | MPIW |
|---|---|---|---|
| Basic | SQR-GNN | 0.85 | 0.33 |
| Basic | RQR^adj-GNN | 0.90 | 0.82 |
| Basic | CF-GNN | 0.92 | 1.90 |
| Basic | BayesianNN | 1.00 | 3.01 |
| Basic | **QpiGNN** | **≥0.90** | **Smallest and meets target** |
| Gaussian | RQR^adj-GNN | 0.88 | 0.53 |
| Gaussian | CF-GNN | 0.91 | 2.90 |
| Gaussian | **QpiGNN** | **≥0.90** | Significantly smallest |
| Grid | RQR^adj-GNN | 0.72 | 0.48 |
| Grid | **QpiGNN** | **≥0.90** | Smallest and meets target |

On average, QpiGNN achieves 22% higher coverage and 50% narrower intervals than all baselines. SQR-GNN often undercovers (0.75–0.85), BayesianNN achieves full coverage but with constant width ≈3, which is impractical; CF-GNN (conformal) meets coverage but interval width is inflated by structural heterogeneity (MPIW 6.89 on BA, 11.92 on Grid).

### Ablation Study

| Configuration | Explanation | Effect |
|---|---|---|
| Full QpiGNN | dual-head + joint loss | Optimal |
| Single-head + joint loss | Shared representation for center+spread | Coverage meets target but width increases |
| Dual-head + fixed-margin | Half-width set as constant | No node-level adaptivity |
| Dual-head + RQR-W loss | Uses entangled loss | Oversmoothing recurs |
| Only $\mathcal L_{\text{coverage}}$ | No width compression | Coverage meets target but intervals are huge |
| Only $\mathcal L_{\text{width}}$ | No coverage constraint | Intervals collapse to 0 |

### Key Findings
- **Both dual-head and joint loss are indispensable**: Removing either causes collapse in coverage or explosion in width.
- **CP does not adapt well to graphs**: CF-GNN’s MPIW explodes on structurally heterogeneous (hub/heterophily) graphs, confirming the failure of the exchangeability assumption.
- **Training trajectory matches Lagrangian intuition**: Loss first rapidly reduces coverage violation, then steadily compresses interval width (Figure 2).

## Highlights & Insights
- **Completely removes QR’s "quantile input"**: Previously, QR was thought to require conditioning on $\tau$; this work shows that with "dual-head + label-only loss," quantile input is redundant—an eye-opening paradigm shift for all "quantile regression" fields.
- **Dual-head is not new but cleverly applied**: Dual-heads in heteroscedastic regression (Kendall & Gal) are for learning prediction and variance simultaneously; here, the structure is borrowed but the purpose is different—to **block GNN message passing from oversmoothing the spread head**. This "repurposing old architectures for new problems" is worth emulating.
- **Finite-sample coverage bounds for graph-dependent data**: Instead of relying on exchangeability, McDiarmid’s bounded-difference is adapted for graph data, yielding a practical $\mathcal O(1/\sqrt N)$ bound—offering a feasible path for transferring CP-style frequentist guarantees to graph-dependent data.

## Limitations & Future Work
- Theoretical symmetry assumption ($P(y\mid x_v)$ symmetric) does not strictly hold for skewed distributions; the authors acknowledge this is only a sketch.
- $\lambda_{\text{width}}$ still requires BO selection; adaptive weight annealing strategies may further reduce tuning costs.
- Experiments focus on node regression; extension to node classification (discrete outputs), link prediction, and graph regression remains to be validated.
- Comparison with modern conformal variants (local CP, weighted CP) could be more comprehensive; current focus is mainly on CF-GNN.

## Related Work & Insights
- **vs SQR-GNN**: Uses a single model + continuous quantile sampling, but calibration is unstable under GNN smoothing; QpiGNN removes quantile input entirely.
- **vs RQR-GNN**: Width-regularized loss effective on MLPs collapses in single-head GNNs; QpiGNN overcomes this with dual-head + decoupled loss.
- **vs CF-GNN (Conformal)**: CP’s MPIW explodes on heterogeneous graphs (hub/heterophily); QpiGNN is stable as it does not rely on exchangeability.
- **vs Bayesian/MC-Dropout/Ensembles**: Bayesian methods scale poorly or have excessive width, ensembles are computationally expensive; QpiGNN provides node-level intervals with a single model and forward pass.

## Rating
- Novelty: ⭐⭐⭐⭐ High originality in simultaneously removing "quantile input" and "post-hoc calibration."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 19 datasets + 7+ baselines with PICP/MPIW comparisons, covering synthetic/real/structurally heterogeneous scenarios.
- Writing Quality: ⭐⭐⭐⭐ Motivation is well developed, theory and experiments mutually reinforce; theorem statements are somewhat sketchy.
- Value: ⭐⭐⭐⭐ Provides a practical, post-processing-free route for GNN node-level UQ, with direct value for graph regression in healthcare/finance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](../../ICLR2026/graph_learning/are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](../../NeurIPS2025/graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](../../NeurIPS2025/graph_learning/graph_neural_networks_for_interferometer_simulations.md)

</div>

<!-- RELATED:END -->
