---
title: >-
  [Paper Note] Quantile-Free Uncertainty Quantification in Graph Neural Networks
description: >-
  [ICML 2026][Graph Learning][GNN] QpiGNN proposes a GNN node-level prediction interval framework that requires "no quantile input and no post-processing." By employing a dual-head GNN (one head for mean prediction and one for half-width) paired with a label-level joint loss that directly optimizes "coverage + interval width," it achieves a 22% increase in average coverage and a 50% reduction in interval width across 19 synthetic and real-world datasets.
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "GNN"
  - "Prediction Interval"
  - "Quantile Regression"
  - "Dual-head Architecture"
  - "Label-only Loss"
date: 2026-05-08
content_hash: ecd5468f8b66e8c0
---

# Quantile-Free Uncertainty Quantification in Graph Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2605.04847](https://arxiv.org/abs/2605.04847)  
**Code**: Yes (Paper marked anonymous.4open.science/r/QpiGNN-30808)  
**Area**: Graph Neural Networks / Uncertainty Quantification / Node Regression  
**Keywords**: GNN, Prediction Interval, Quantile Regression, Dual-head Architecture, Label-only Loss

## TL;DR
QpiGNN proposes a GNN node-level prediction interval framework that requires "no quantile input and no post-processing." By employing a dual-head GNN (one head for mean prediction and one for half-width) paired with a label-level joint loss that directly optimizes "coverage + interval width," it achieves a 22% increase in average coverage and a 50% reduction in interval width across 19 synthetic and real-world datasets.

## Background & Motivation

**Background**: Node regression GNNs are widely utilized in high-risk domains such as healthcare and criminal justice; however, most GNNs only output point estimates without providing uncertainty measures. Existing uncertainty quantification (UQ) methods are primarily categorized into two types: Bayesian (e.g., VI, posterior approximation—which suffer from poor scalability and sensitivity to priors) and Frequentist (e.g., resampling like ensembles, or post-hoc calibration like Conformal Prediction). Frequentist methods are computationally expensive and often rely on the **exchangeability** assumption, which is almost inherently violated in graph data with structural dependencies.

**Limitations of Prior Work**: Quantile Regression (QR) appears to be a viable choice for bypassing distributional assumptions, but standard QR must take the quantile level $\tau$ as an input or train an independent model for each $\tau$, leading to issues like "quantile crossing" (where lower quantile predictions exceed higher ones). SQR learns multiple quantiles within a single model, and RQR uses a width-regularized loss to estimate center+spread for MLPs. However, **these methods collapse when directly applied to GNNs**: message passing over-smoothes node representations. Specifically, SQR exhibits poor stability and fails calibration on graphs, while RQR’s single-head design causes representation sharing between the center and spread, leading to gradient interference.

**Key Challenge**: The bottleneck of the QR series is the combination of "quantile input + single-head representation," which is structurally incompatible with the GNN mechanism where "neighborhood aggregation produces globally smoothed representations." To leverage GNN’s relational modeling while obtaining node-level adaptive tight intervals, "prediction" and "uncertainty" must be decoupled at both the architecture and supervision levels.

**Goal**: (i) Design a GNN UQ framework that does not depend on quantile inputs or post-hoc calibration; (ii) Provide theoretical guarantees for coverage and width under graph dependency; (iii) Balance calibration and compactness.

**Key Insight**: The authors discovered that RQR can learn input-dependent upper and lower bounds on MLPs using a "label-only" loss, suggesting that the "quantile input" of QR can be bypassed. Furthermore, the root cause of over-smoothing in GNNs is single-head sharing. Thus, by using **dual-head decoupling + direct label supervision**, both constraints can be resolved simultaneously.

**Core Idea**: Utilizing a **dual-head GNN (one head for $\hat y$ and one for half-width $\hat d$)** + a **quantile-free joint loss** (directly penalizing "deviation of $\hat c$ from $1-\alpha$" and "average interval width") removes the need for both quantile inputs and post-processing.

## Method

### Overall Architecture
QpiGNN aims to provide both point estimates and compact, calibrated prediction intervals for GNN node regression without relying on quantile inputs or post-hoc conformal calibration. The approach decouples "prediction" and "uncertainty" in both architecture and supervision: a shared GNN encoder first computes node embeddings $\mathbf H=\text{GNN}(\mathbf X,\mathcal E)$, which then connect to two linear heads—a prediction head for the interval center $\hat y$ and a half-width head for $\hat d$. The resulting interval is $[\hat y_v-\hat d_v,\ \hat y_v+\hat d_v]$. During training, a joint loss directly optimizes the labels to ensure "coverage is close to the target + the interval is as narrow as possible." Calibrated intervals are obtained via a single forward pass during inference, requiring no calibration set or post-processing.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Node Features X + Edge Set E"] --> B
    subgraph DH["Dual-head GNN"]
        direction TB
        B["Shared GNN Encoder<br/>Node Embeddings H"] --> C["Prediction Head<br/>Interval Center ŷ"]
        B --> D["Half-width Head Softplus<br/>Half-width d̂ > 0"]
    end
    C --> E["Prediction Interval [ŷ − d̂, ŷ + d̂]"]
    D --> E
    E --> F["Quantile-free Joint Loss<br/>Coverage Term + Width Term"]
```

> The coverage guarantee under graph dependency (Design 3) serves as the theoretical support for the joint loss above rather than a separate data processing stage, hence it is not listed as a separate node in the flowchart.

### Key Designs

**1. Dual-head GNN: Decoupling Prediction and Uncertainty at the Representation Layer**

The root cause of QR methods collapsing on GNNs is the shared representation in single-head designs. Since message passing repeatedly averages over neighborhoods, a single-head model pushes both the interval center and half-width toward local means, erasing node-level adaptivity. QpiGNN branches into two independent linear heads on top of the shared encoding $\mathbf H$: the prediction head $\hat{\mathbf y}=\mathbf W_{\text{pred}}\mathbf H+\mathbf b_{\text{pred}}$ focuses on accuracy, while the half-width head $\hat{\mathbf d}=\text{Softplus}(\mathbf W_{\text{diff}}\mathbf H+\mathbf b_{\text{diff}})$ focuses on coverage. The Softplus function ensures $\hat d>0$, making the interval naturally well-ordered and preventing quantile crossing. This structural decoupling allows the half-width head to learn a function class entirely different from the center—for instance, providing wider intervals for hub nodes without being influenced by the smoothing trend of the center head. This echoes the heteroscedastic/Bayesian regression insights of Kendall & Gal and Lakshminarayanan regarding "learning different signals with different heads," but here it specifically blocks over-smoothing from contaminating the half-width head using a lightweight additional linear head.

**2. Quantile-free Joint Loss: Simultaneous Coverage Calibration and Interval Compression via Labels**

Standard QR requires the quantile level $\tau$ as an input or separate models for each $\tau$. RQR-W merges coverage and width into a single conditional loss, which is pushed toward globally over-wide intervals by over-smoothing in GNNs. QpiGNN eliminates the quantile input entirely, using an additive three-term label-level loss for direction supervision:

$$\mathcal L_{\text{total}}=\underbrace{(\hat c-(1-\alpha))^2 + \hat\ell_{\text{viol}}}_{\mathcal L_{\text{coverage}}} + \underbrace{\lambda_{\text{width}}\cdot\mathbb E_v[\hat y_v^{\text{up}}-\hat y_v^{\text{low}}]}_{\mathcal L_{\text{width}}}$$

Where $\hat c=\mathbb P(\hat y_v^{\text{low}}\le y_v\le \hat y_v^{\text{up}})$ is the empirical coverage; the squared term pulls it toward the target $1-\alpha$. The violation penalty $\hat\ell_{\text{viol}}=\mathbb E[|y_v-\hat y_v^{\text{low}}|\cdot\mathds 1[y_v<\hat y_v^{\text{low}}]+|y_v-\hat y_v^{\text{up}}|\cdot\mathds 1[y_v>\hat y_v^{\text{up}}]]$ provides fine-grained gradients based on the distance only for nodes falling outside the interval. The width penalty uses $L_1$ instead of $L_2$ to prevent outliers from inflating the width term. Decoupling coverage and width into additive terms allows for a clear training sequence—attaining the target $\hat c$ first, then compressing the width while maintaining coverage. This corresponds to a Lagrangian relaxation of a constrained optimization where the hyperparameter $\lambda_{\text{width}}\in[0.2,0.5]$ (selected via Bayesian Optimization) acts as the multiplier. The entire process is supervised solely by labels $y_v$, requiring no quantile inputs or post-hoc calibration.

**3. Coverage Guarantees under Graph Dependency: Bypassing Exchangeability**

Since the coverage guarantees of CP rely on exchangeability—which graph data naturally violates due to structural dependencies—QpiGNN rebuilds guarantees on "approximate bounded-difference under neighborhood smoothing." Proposition 4.1 uses the Weak Law of Large Numbers to show asymptotic convergence $\hat c\xrightarrow{P}1-\alpha$ under assumptions that noise $\varepsilon_v$ is bounded and weakly correlated, $\hat y_v$ and $\hat d_v$ converge in probability to their targets, and node embeddings are sufficiently diverse. For the finite sample case, the McDiarmid/Hoeffding inequality is used: the impact of a single node perturbation on the coverage estimate is bounded by $1/N+\delta_G$, resulting in $|\hat c-(1-\alpha)|=\mathcal O(1/\sqrt N)$, which provides a practical frequentist bound. Additionally, under the assumption that $P(y\mid x_v)$ is symmetric, the minimum width satisfies $d_v^*=F_v^{-1}(1-\alpha/2)$, confirming that the joint loss is indeed the Lagrangian relaxation of this constrained optimization.

### Loss & Training
The model is trained end-to-end using SGD on the three-term weighted sum above, with a diminishing learning rate to ensure convergence to a stationary point under non-convex conditions. $\alpha$ is typically set to $0.1$ (i.e., 90% target coverage), and $\lambda_{\text{width}}\in[0.2,0.5]$ is selected via BO. For comparison, the authors also implemented a GNN variant of RQR, adding an ordering penalty $\gamma_{\text{order}}\cdot\text{ReLU}(\hat y^{\text{low}}-\hat y^{\text{up}})$ to mitigate its quantile crossing.

## Key Experimental Results

### Main Results
Evaluations were conducted on 19 datasets (9 synthetic structures like BA/ER/Grid/Tree + real-world datasets) using PICP (Empirical Coverage) and MPIW (Mean Prediction Interval Width) as metrics, with a target coverage of 90%.

| Dataset (Synthetic) | Model | PICP | MPIW |
|---|---|---|---|
| Basic | SQR-GNN | 0.85 | 0.33 |
| Basic | RQR^adj-GNN | 0.90 | 0.82 |
| Basic | CF-GNN | 0.92 | 1.90 |
| Basic | BayesianNN | 1.00 | 3.01 |
| Basic | **QpiGNN** | **≥0.90** | **Smallest and met target** |
| Gaussian | RQR^adj-GNN | 0.88 | 0.53 |
| Gaussian | CF-GNN | 0.91 | 2.90 |
| Gaussian | **QpiGNN** | **≥0.90** | Significantly smallest |
| Grid | RQR^adj-GNN | 0.72 | 0.48 |
| Grid | **QpiGNN** | **≥0.90** | Smallest met target |

On average, QpiGNN achieved 22% higher coverage and 50% narrower widths than all baselines. SQR-GNN frequently suffered from under-coverage (0.75–0.85). BayesianNN maximized coverage but produced constant widths $\approx3$, which is impractical. CF-GNN (conformal) met the target coverage but its width was inflated by structural heterogeneity (e.g., MPIW of 6.89 on BA graphs and 11.92 on Grid).

### Ablation Study

| Configuration | Explanation | Effect |
|---|---|---|
| Full QpiGNN | dual-head + joint loss | Optimal |
| Single-head + joint loss | Shared representation learning center + spread | Met target coverage but width increased |
| Dual-head + fixed-margin | Half-width set as a constant | Not node-level adaptive |
| Dual-head + RQR-W loss | Used entangled loss | Over-smoothing recurred |
| Only $\mathcal L_{\text{coverage}}$ | No width compression | Met target coverage but intervals extremely wide |
| Only $\mathcal L_{\text{width}}$ | No coverage constraint | Intervals collapsed to 0 |

### Key Findings
- **Both dual-head + joint loss are indispensable**: Removing either leads to collapsed coverage or explosive width.
- **CP is ill-suited for graphs**: CF-GNN's MPIW grows explosively on structurally heterogeneous graphs (hub / heterophily), validating the failure of the exchangeability assumption.
- **Training trajectory aligns with Lagrangian intuition**: The loss first rapidly reduces coverage violation, then continuously compresses the interval width (Figure 2).

## Highlights & Insights
- **Completely dismantling the "quantile input" of QR**: Previously, it was believed that QR must be conditioned on $\tau$. This paper proves through "dual-head + label-only loss" that if the architecture and supervision format are correct, the quantile input is redundant—this represents a paradigm shift for all quantile regression fields.
- **Clever application of dual-head**: While dual-head is not a new concept in heteroscedastic regression (Kendall & Gal), the motivation here is different—it is used to **block over-smoothing of GNN message passing on the spread head**. This perspective of "using old architectures to solve new problems" is noteworthy.
- **Graph-dependent finite sample bounds for coverage**: By not relying on exchangeability and instead using McDiarmid's bounded-difference for graph data, the paper provides a practical bound of $\mathcal O(1/\sqrt N)$. This offers a viable path for migrating CP-style frequentist guarantees to graph-dependent data.

## Limitations & Future Work
- The theoretical symmetry assumption ($P(y\mid x_v)$ is symmetric) does not strictly hold for skewed distributions; the authors acknowledge this is only a sketch.
- $\lambda_{\text{width}}$ still requires selection via BO; an adaptive weight annealing strategy might further reduce tuning costs.
- Experiments focused on node regression; extension to node classification (discrete output), link prediction, and graph regression remains to be verified.
- Comparisons with modern conformal variants (local CP, weighted CP) could be more comprehensive, as the current focus is primarily on CF-GNN.

## Related Work & Insights
- **vs SQR-GNN**: Uses a single model + continuous quantile sampling, but calibration is unstable under GNN smoothing; QpiGNN removes quantile input.
- **vs RQR-GNN**: Width-regularized loss effective on MLPs collapses on GNNs with a single head; QpiGNN breaks through using dual-head + decoupled loss.
- **vs CF-GNN (Conformal)**: MPIW of CP explodes on heterogeneous graphs (hub/heterophily); QpiGNN is stable as it does not rely on exchangeability.
- **vs Bayesian/MC-Dropout/Ensembles**: Bayesian methods have poor scalability or width inflation; ensembles are computationally expensive. QpiGNN obtains node-level intervals with a single model and a single forward pass.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of simultaneously removing "quantile input" and "post-hoc calibration" is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ PICP/MPIW comparisons across 19 datasets and 7+ baselines, covering synthetic, real, and structurally heterogeneous scenarios.
- Writing Quality: ⭐⭐⭐⭐ The motivation is progressively developed with mutual evidence from theory and experiments, though the theorem descriptions are somewhat sketchy.
- Value: ⭐⭐⭐⭐ Provides a practical route for GNN node-level UQ without post-processing, offering direct value for the deployment of graph regression in healthcare and finance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning](gilt_an_llm-free_tuning-free_graph_foundational_model_for_in-context_learning.md)
- [\[ICML 2026\] Understanding Truncated Positional Encodings for Graph Neural Networks](understanding_truncated_positional_encodings_for_graph_neural_networks.md)
- [\[ICLR 2026\] Learning from Historical Activations in Graph Neural Networks](../../ICLR2026/graph_learning/learning_from_historical_activations_in_graph_neural_networks.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[CVPR 2026\] Adaptive Learned Image Compression with Graph Neural Networks](../../CVPR2026/graph_learning/adaptive_learned_image_compression_with_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
