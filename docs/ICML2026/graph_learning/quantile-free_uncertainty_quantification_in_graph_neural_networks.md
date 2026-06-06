---
title: >-
  [Paper Note] Quantile-Free Uncertainty Quantification in Graph Neural Networks
description: >-
  [ICML 2026][Graph Learning][GNN] QpiGNN proposes a "no-quantile-input, no-post-processing" GNN node-level prediction interval framework. By utilizing a dual-head GNN (one head for mean…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "GNN"
  - "Prediction Interval"
  - "Quantile Regression"
  - "Dual-head Architecture"
  - "Label-only Loss"
date: 2026-05-08
content_hash: 06729c70ebefed0d
---

# Quantile-Free Uncertainty Quantification in Graph Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2605.04847](https://arxiv.org/abs/2605.04847)  
**Code**: Available (labeled anonymous.4open.science/r/QpiGNN-30808 in the paper)  
**Area**: Graph Neural Networks / Uncertainty Quantification / Node Regression  
**Keywords**: GNN, Prediction Interval, Quantile Regression, Dual-head Architecture, Label-only Loss

## TL;DR
QpiGNN proposes a "no-quantile-input, no-post-processing" GNN node-level prediction interval framework. By utilizing a dual-head GNN (one head for mean, one for half-width) combined with a joint label-level loss that directly optimizes "coverage + interval width," the method achieves a 22% increase in average coverage and a 50% reduction in interval width across 19 synthetic/real datasets.

## Background & Motivation

**Background**: Node regression GNNs are widely used in high-risk fields such as healthcare and criminal justice, yet most GNNs only output point estimates without uncertainty. Available uncertainty quantification (UQ) methods are mainly divided into two categories: Bayesian (VI, posterior approximation; poor scalability and sensitive to priors) and Frequentist (resampling like ensembles, post-hoc calibration like Conformal Prediction). Frequentist methods are computationally expensive and often rely on the **exchangeability** assumption—which naturally fails on graph data with structural dependencies.

**Limitations of Prior Work**: Quantile Regression (QR) seems a good choice to bypass distribution assumptions, but standard QR requires the quantile level $\tau$ as input or needs an independent model for each $\tau$, leading to issues like "quantile crossing." SQR learns multiple quantiles in one model, and RQR uses a width-regularized loss for MLP center+spread estimation. However, **direct migration to GNNs leads to failure**: message passing causes node representations to suffer from oversmoothing; SQR exhibits poor stability and calibration failure on graphs; and the single-head design of RQR causes gradient interference as the center and spread share representations.

**Key Challenge**: The bottleneck for the QR series is "quantile input + single-head representation," which is structurally incompatible with the "neighborhood aggregation producing globally smooth representations" of GNNs. To utilize GNN relationship modeling while achieving node-level adaptive tight intervals, one must decouple "prediction" and "uncertainty" in both architecture and supervision.

**Goal**: (i) Design a GNN UQ framework independent of quantile input and post-hoc calibration; (ii) Provide theoretical guarantees for coverage and width under graph dependency; (iii) Balance calibration and compactness.

**Key Insight**: The authors found that RQR on MLPs can learn input-dependent upper/lower bounds using a "label-only" loss, meaning the "quantile input" of QR can be bypassed. Furthermore, since the root cause of oversmoothing in GNNs is single-head sharing, **dual-head decoupling + direct label supervision** can simultaneously release these two constraints.

**Core Idea**: Use a **dual-head GNN (one head for $\hat y$, one for half-width $\hat d$)** + a **quantile-free joint loss** (directly penalizing "$\hat c$ deviation from $1-\alpha$" and "average interval width"), requiring neither quantile input nor post-processing.

## Method

### Overall Architecture
Given a graph $G=(\mathcal V,\mathcal E)$ and node features $\mathbf X$, QpiGNN uses a shared GNN encoder to compute node embeddings $\mathbf H=\text{GNN}(\mathbf X,\mathcal E)$, followed by two linear heads: the prediction head $\hat{\mathbf y}=\mathbf W_{\text{pred}}\mathbf H+\mathbf b_{\text{pred}}$ and the half-width head $\hat{\mathbf d}=\text{Softplus}(\mathbf W_{\text{diff}}\mathbf H+\mathbf b_{\text{diff}})$. The final prediction interval is $[\hat y_v-\hat d_v,\ \hat y_v+\hat d_v]$. Training employs a triple joint loss: coverage squared error + violation penalty + width penalty, supervised directly by labels $y_v$. At inference, a single end-to-end forward pass provides calibrated node-level intervals without needing a calibration set or conformal post-processing.

### Key Designs

1. **Dual-head GNN for Decoupling Prediction and Uncertainty**:
    - **Function**: Allows $\hat y$ and $\hat d$ to learn specialized representations (one for accuracy, one for coverage), avoiding oversmoothing and gradient conflict under shared representations.
    - **Mechanism**: Encoding $\mathbf H$ via a shared GNN, while two linear heads compute their respective outputs. The half-width head uses Softplus to ensure $\hat d>0$, making the interval naturally well-ordered (avoiding quantile crossing). This echoes successful experiences in heteroscedastic/Bayesian regression (e.g., Kendall & Gal, Lakshminarayanan) but is more lightweight.
    - **Design Motivation**: On graphs, node representations are repeatedly averaged through message passing. A single-head model inevitably pushes both center and spread toward the local mean, destroying node-level adaptivity. Structural decoupling allows the spread head to learn a function class entirely different from the center—for instance, naturally providing wider intervals on hub nodes.

2. **Quantile-free joint loss for Direct Coverage and Width Supervision**:
    - **Function**: Replaces "quantile input" and "post-hoc calibration" with a one-step training process supervised by labels $y_v$ to simultaneously calibrate coverage and compress width.
    - **Mechanism**: $\mathcal L_{\text{total}}=\underbrace{(\hat c-(1-\alpha))^2 + \hat\ell_{\text{viol}}}_{\mathcal L_{\text{coverage}}} + \underbrace{\lambda_{\text{width}}\cdot\mathbb E_v[\hat y_v^{\text{up}}-\hat y_v^{\text{low}}]}_{\mathcal L_{\text{width}}}$. Here $\hat c=\mathbb P(\hat y_v^{\text{low}}\le y_v\le \hat y_v^{\text{up}})$ is the empirical coverage; $\hat\ell_{\text{viol}}=\mathbb E[|y_v-\hat y_v^{\text{low}}|\cdot\mathds 1[y_v<\hat y_v^{\text{low}}]+|y_v-\hat y_v^{\text{up}}|\cdot\mathds 1[y_v>\hat y_v^{\text{up}}]]$ provides fine-grained gradients for violating nodes; the width penalty uses an L1 form to avoid instability under outliers. $\lambda_{\text{width}}\in [0.2,0.5]$ is chosen via Bayesian Optimization.
    - **Design Motivation**: RQR-W merges coverage and width into a single conditional loss, which oversmoothing pushes toward globally over-wide intervals on GNNs. QpiGNN decouples them into additive terms: first pulling $\hat c$ to target $1-\alpha$, then compressing the width while maintaining coverage. This "Lagrangian relaxation" perspective gives $\lambda_{\text{width}}$ a clear meaning.

3. **Asymptotic + Finite-sample Coverage Guarantees**:
    - **Function**: Provides provable coverage convergence on graph data where i.i.d. / exchangeability is violated.
    - **Mechanism**: Proposition 4.1 assumes the noise $\varepsilon_v$ is bounded and weakly correlated, $\hat y_v$ and $\hat d_v$ converge in probability to the target, and node embeddings are sufficiently diverse, leading to $\hat c\xrightarrow{P}1-\alpha$ (WLLN). Under finite samples, using McDiarmid/Hoeffding inequalities: a single node perturbation affects the coverage estimate by $\le 1/N+\delta_G$, hence $|\hat c-(1-\alpha)|=\mathcal O(1/\sqrt N)$. Also, under the assumption that $P(y\mid x_v)$ is symmetric, the minimum width satisfies $d_v^*=F_v^{-1}(1-\alpha/2)$, and the loss here is interpreted as the Lagrangian relaxation of this constrained optimization.
    - **Design Motivation**: CP coverage guarantees rely on exchangeability. QpiGNN builds guarantees on "approximate bounded-difference under neighborhood smoothing," which is more suitable for graph data.

### Loss & Training
End-to-end SGD training is used with the triple-term weighted sum loss. A diminishing learning rate ensures convergence to a stationary point under non-convexity. $\alpha$ is typically $0.1$ (90% target coverage), and $\lambda_{\text{width}}\in[0.2,0.5]$ is selected via BO. For comparison, a GNN variant of RQR was implemented with an ordering penalty $\gamma_{\text{order}}\cdot\text{ReLU}(\hat y^{\text{low}}-\hat y^{\text{up}})$ to mitigate quantile crossing.

## Key Experimental Results

### Main Results
Evaluated on 19 datasets (9 synthetic structures like BA/ER/Grid/Tree + real datasets) using PICP (Empirical Coverage) and MPIW (Mean Prediction Interval Width) as metrics, with a target coverage of 90%.

| Dataset (Synthetic) | Model | PICP | MPIW |
|---|---|---|---|
| Basic | SQR-GNN | 0.85 | 0.33 |
| Basic | RQR^adj-GNN | 0.90 | 0.82 |
| Basic | CF-GNN | 0.92 | 1.90 |
| Basic | BayesianNN | 1.00 | 3.01 |
| Basic | **QpiGNN** | **≥0.90** | **Smallest reached target** |
| Gaussian | RQR^adj-GNN | 0.88 | 0.53 |
| Gaussian | CF-GNN | 0.91 | 2.90 |
| Gaussian | **QpiGNN** | **≥0.90** | Significantly smallest |
| Grid | RQR^adj-GNN | 0.72 | 0.48 |
| Grid | **QpiGNN** | **≥0.90** | Smallest reached target |

On average, QpiGNN yields 22% higher coverage and 50% narrower width than all baselines. SQR-GNN often under-covers (0.75–0.85), while BayesianNN provides full coverage but constant width ≈3, which is impractical. CF-GNN (Conformal) barely hits the coverage target but the width is amplified by structural heterogeneity (e.g., MPIW 6.89 on BA, 11.92 on Grid).

### Ablation Study

| Configuration | Explanation | Effect |
|---|---|---|
| Full QpiGNN | dual-head + joint loss | Optimal |
| Single-head + joint loss | Learn center+spread via shared rep | Coverage reached but width increases |
| Dual-head + fixed-margin | Half-width set as constant | No node-level adaptivity |
| Dual-head + RQR-W loss | Using entangled loss | Oversmoothing recurs |
| Only $\mathcal L_{\text{coverage}}$ | No width compression | Coverage reached but intervals are huge |
| Only $\mathcal L_{\text{width}}$ | No coverage constraint | Intervals collapse to 0 |

### Key Findings
- **dual-head + joint loss are both indispensable**: Removing either leads to collapsed coverage or exploded width.
- **CP is ill-suited for graphs**: CF-GNN's MPIW grows explosively on structurally heterogeneous graphs (hubs/heterophily), validating the failure of the exchangeability assumption.
- **Training trajectory matches Lagrangian intuition**: The loss quickly reduces coverage violation first, then continuously compresses interval width (Figure 2).

## Highlights & Insights
- **Completely removing the "quantile input" of QR**: It was previously assumed that QR must be conditioned on $\tau$. This paper proves that with the correct architecture and supervision, quantile input is redundant—a paradigm shift for the "Quantile Regression" field.
- **dual-head is not new but used cleverly**: While dual-heads in heteroscedastic regression (Kendall & Gal) aim to learn prediction and variance simultaneously, here the goal is to **block the oversmoothing of the spread head caused by GNN message passing**. This "new solution for old architectures" perspective is noteworthy.
- **Graph-dependent finite-sample bounds for coverage**: By bypassing exchangeability and using McDiarmid's bounded-difference principle, the authors provide a practical bound of $\mathcal O(1/\sqrt N)$ for graph data—a viable path for migrating Frequentist guarantees to graph-dependent data.

## Limitations & Future Work
- The theoretical symmetry assumption ($P(y\mid x_v)$ is symmetric) does not strictly hold for skewed distributions; the authors acknowledge this as a sketch.
- $\lambda_{\text{width}}$ still requires BO selection; an adaptive weight annealing strategy might further reduce tuning costs.
- Experiments focus on node regression; extension to node classification (discrete output), link prediction, and graph regression remains to be verified.
- Comparisons could be more comprehensive against modern conformal variants (Local CP, Weighted CP), as it currently focuses on CF-GNN.

## Related Work & Insights
- **vs SQR-GNN**: Uses a single model + continuous quantile sampling, but calibration is unstable under GNN smoothing; QpiGNN removes quantile input.
- **vs RQR-GNN**: Width-regularized loss effective on MLPs fails on single-head GNNs; QpiGNN breaks through via dual-head + decoupled loss.
- **vs CF-GNN (Conformal)**: CP MPIW explodes on heterogeneous graphs (hubs/heterophily); QpiGNN is robust as it does not rely on exchangeability.
- **vs Bayesian/MC-Dropout/Ensembles**: Bayesian series lack scalability or have exploded widths; ensembles are computationally expensive. QpiGNN provides node-level intervals with a single model and forward pass.

## Rating
- Novelty: ⭐⭐⭐⭐ High originality in simultaneously removing "quantile input" and "post-hoc calibration."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ PICP/MPIW comparison across 19 datasets + 7+ baselines, covering synthetic, real, and structural heterogeneity.
- Writing Quality: ⭐⭐⭐⭐ The motivation is well-developed, and theory/experiments corroborate each other, though the theorem statements are somewhat sketchy.
- Value: ⭐⭐⭐⭐ Provides a practical, post-processing-free route for node-level UQ in GNNs, with direct value for medical/financial graph regression tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](../../NeurIPS2025/graph_learning/graph_neural_networks_for_interferometer_simulations.md)

</div>

<!-- RELATED:END -->
