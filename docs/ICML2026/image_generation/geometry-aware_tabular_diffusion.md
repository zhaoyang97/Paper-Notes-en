---
title: >-
  [Paper Note] Geometry-Aware Tabular Diffusion
description: >-
  [ICML 2026][Image Generation][Tabular Diffusion] The authors propose GATD (Geometry-Aware Tabular Diffusion), which explicitly incorporates geometric features—specifically "angles and lengths between column pairs"—into t…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Tabular Diffusion"
  - "Inter-column Geometric Features"
  - "Auxiliary Supervision"
  - "Portable Inductive Bias"
  - "TabDiff"
date: 2026-05-08
content_hash: 25b52d2e03940461
---

# Geometry-Aware Tabular Diffusion

**Conference**: ICML 2026  
**arXiv**: [2606.02607](https://arxiv.org/abs/2606.02607)  
**Code**: TBD  
**Area**: Tabular Data Generation / Diffusion Models / Geometric Deep Learning  
**Keywords**: Tabular Diffusion, Inter-column Geometric Features, Auxiliary Supervision, Portable Inductive Bias, TabDiff

## TL;DR
The authors propose GATD (Geometry-Aware Tabular Diffusion), which explicitly incorporates geometric features—specifically "angles and lengths between column pairs"—into the tabular diffusion denoiser input and loss as auxiliary supervision signals. Using a small MLP with parameters only 1/3.5 (or even 1/25 for classification tasks) the size of TabDiff, GATD achieves 8/10 Shape, 7/10 Trend, and 9/10 downstream utility wins across 10 datasets. The default hyperparameters can be directly transferred to GNN and Transformer denoisers, yielding improvements in 27/30 Shape and 25/30 Trend metrics.

## Background & Motivation

**Background**: Tabular data is the most common data format in enterprise, medical, and scientific research. Synthetic tables are widely used for privacy-preserving data sharing and data augmentation. In recent years, diffusion models have become the mainstream for tabular synthesis, with TabDDPM, STaSy, TabSyn, and TabDiff successively improving performance. TabDiff, the current SOTA, utilizes Transformer self-attention to model inter-column relationships.

**Limitations of Prior Work**: Existing methods rely entirely on the denoising loss to implicitly learn the relationship of "how columns should co-vary." While Transformer attention is flexible, it must discover inter-column structures through remote, weakly supervised objectives like denoising MSE. Consequently, models require larger sizes and longer training to learn effectively, and they lack a shared inductive bias for cross-architecture transfer.

**Key Challenge**: Objective geometric relationships (such as the direction and magnitude of differences between two numerical columns) exist between tabular columns. However, current architectures neither feed these to the model nor require the model to explicitly predict them, essentially discarding a free, supervisable structural signal.

**Goal**: (1) Formulate pairwise column geometric relationships into explicit, differentiable features and auxiliary prediction targets; (2) Verify if this supervision signal transfers across MLP, GNN, and Transformer denoisers; (3) Challenge the Transformer SOTA using a minimal MLP.

**Key Insight**: Drawing from geometric deep learning, as the success of GNNs, point clouds, and Transformer positional encodings stems from "explicitly providing geometric structure," the difference between tabular column pairs can be parameterized as geometric quantities similar to graph edges. Crucially, ablation studies with architecture-matched models prove that **simply feeding geometric features as input yields no benefit (Cohen's $d=-0.08$), whereas using them as auxiliary prediction targets for supervision has a large effect ($d=0.81$)**. That is, the effectiveness comes from being "forced to learn geometry" rather than just "seeing it."

**Core Idea**: Construct explicit geometric targets using $\arctan$ (angle) and $\frac{1}{2}\log(1+\Delta^2)$ (length) for every pair of column values. The weights for these targets significantly exceed the diffusion loss itself (accounting for approx. 95% of the total loss), forcing the denoiser to internalize inter-column relationships into its representations. This serves as a relational inductive bias portable to any denoising architecture.

## Method

### Overall Architecture
GATD retains the diffusion backbone of TabDiff (EDM for continuous columns, masked diffusion for categorical columns, and a learnable per-column $\rho$ noise scheduler) and wraps the denoiser with a "geometric input + geometric prediction head + geometric supervision" suite.

Mechanism: (1) Map all columns (continuous or categorical) to a unified scalar space $v \in [-1, 1]$; (2) Calculate the ground truth angle $\theta_{ij}$ and length $\ell_{ij}$ for all $\binom{d}{2}$ column pairs as supervision targets, while using geometric quantities under noise as model inputs; (3) The denoiser receives `[time embedding; noisy continuous; one-hot categorical; input angle; input length]` and generates a hidden representation $\mathbf{h}$; (4) The geometric head predicts $\hat{\boldsymbol{\theta}}$ (constrained by $\frac{\pi}{2}\tanh$) and $\hat{\boldsymbol{\ell}}$ from $\mathbf{h}$; (5) The augmented representation $\mathbf{h}_{\text{aug}}=[\mathbf{h};\hat{\boldsymbol{\theta}}]$ enters the continuous/categorical denoising heads; (6) Total Loss = Diffusion Loss + Angle/Length MSE + Consistency Loss, with geometric terms weighted to dominate.

During sampling, geometric inputs are calculated but supervision is not applied, and the **length head is detached during generation** (used only for regularization during training as sign information is lost via squaring). Post-generation, boundary values are folded back into $[0, 1]$ via reflection (rather than hard clipping) to avoid quality accumulation at the boundaries.

### Key Designs

1.  **Pairwise Angle + Length Geometric Characterization**:
    - **Function**: Losslessly encodes the difference between any two columns into a pair of explicit geometric quantities for input and supervision.
    - **Mechanism**: First normalize all columns to $v\in[-1,1]^d$—continuous columns via quantile transformation $v=2\cdot\text{QT}(x)-1$, and categorical columns via a fixed deterministic mapping $v=2\cdot\text{idx}/\max(\text{card}-1,1)-1$. For each $i<j$, calculate $\theta_{ij}=\arctan(v_j-v_i)$ and $\ell_{ij}=\frac{1}{2}\log(1+(v_j-v_i)^2)$. Angles are naturally bounded and anti-symmetric; lengths utilize log-compression for large differences. Since $v_j-v_i=\tan(\theta_{ij})$, the angle is theoretically sufficient to recover the difference, thus only the predicted angle is concatenated into $\mathbf{h}_{\text{aug}}$.
    - **Design Motivation**: A unified geometric signal for mixed-type columns requires a shared scalar space. $\arctan$ ensures stable, bounded supervision targets (ablation shows raw differences are slightly worse). Fixed mapping instead of learned embeddings allows geometric features to be computed on-the-fly without extra trainable parameters. An incidental benefit is that ordinal categories (e.g., education level, Likert scales) automatically receive ordered reinforcement.

2.  **Architecture-Matched "Input vs. Supervision" Controlled Experiment**:
    - **Function**: Isolates whether geometry works by being *seen* or by being *forced to be predicted*.
    - **Mechanism**: Compare three configurations: NoGeom, InputsOnly (feed geometry + add head but $\lambda_\theta=\lambda_\ell=\lambda_c=0$), and +Geom (feed geometry + head + active weights). Architecture, parameter count, and gradient topology remain identical; only the geometric loss weight is toggled.
    - **Design Motivation**: Traditional ablations often remove inputs, heads, and losses simultaneously, confounding "extra capacity" with "explicit supervision." The authors found Cohen's $d=-0.08$ (negligible) for InputsOnly vs. NoGeom, while $d=0.81$ (large effect) for +Geom vs. NoGeom. This cleanly proves that **auxiliary supervision is the key, not geometric inputs or capacity**. This methodology explains why Transformer self-attention fails to learn such structures—no objective forces it.

3.  **Inverted Loss Hierarchy (Geometry 95% / Diffusion 5%)**:
    - **Function**: Ensures the geometric auxiliary task dominates optimization, forcing the backbone to encode inter-column relations before "incidentally" completing denoising.
    - **Mechanism**: Default weights $(\lambda_\epsilon,\lambda_{\text{cat}},\lambda_\theta,\lambda_\ell,\lambda_c)=(1.0,0.05,15,15,8)$ make the geometric terms account for ~95% of the total loss upon convergence. The consistency loss $\mathcal{L}_c=\mathbb{E}[(1-t)^2]\cdot(\|\hat{\boldsymbol{\theta}}-\text{sg}(\boldsymbol{\theta}_{\text{pred}})\|^2+\|\hat{\boldsymbol{\ell}}-\text{sg}(\boldsymbol{\ell}_{\text{pred}})\|^2)$ uses $(1-t)^2$ in low-noise regimes to force geometric head predictions to align with geometry recalculated from the denoised output.
    - **Design Motivation**: Weight ablation revealed that reducing geometric weights to the same level as diffusion leads to performance drops—contrary to the multitask learning intuition that auxiliary weights should be small. The explanation is that denoising MSE is a local, element-wise loss that does not push gradients toward "understanding column relations." A heavy auxiliary loss is required to guide the model. The same weights work across MLP/GNN/Transformer backbones without retuning, proving they are structural rather than overfitted.

### Loss & Training
Total loss $\mathcal{L}=\lambda_\epsilon\mathcal{L}_{\text{cont}}+\lambda_{\text{cat}}\mathcal{L}_{\text{cat}}+\lambda_\theta\mathcal{L}_{\text{angle}}+\lambda_\ell\mathcal{L}_{\text{length}}+\lambda_c\mathcal{L}_{\text{consistency}}$. $\mathcal{L}_{\text{cont}}$ is EDM-weighted denoising MSE; $\mathcal{L}_{\text{cat}}$ is weighted cross-entropy on masked tokens; $\mathcal{L}_{\text{angle}}/\mathcal{L}_{\text{length}}$ are L2 relative to ground truth $\theta/\ell$; $\mathcal{L}_{\text{consistency}}$ aligns predicted geometry with recalculated geometry from denoised outputs (with stop-gradient). Optimizer: AdamW + EMA, 20,000 epochs (vs. 8,000 for TabDiff). Despite 2.5× more epochs, end-to-end wall-clock time is 1.7× faster than TabDiff. Sampling uses EDM Euler 1000 steps + categorical iterative unmasking + reflection boundary handling.

## Key Experimental Results

### Main Results
On 10 TabDiff-style benchmarks (5 classification + 5 regression), 3 training seeds × 20 generation seeds.

| Evaluation Metric | Ours (GATD-MLP) | TabDiff (Transformer SOTA) | Key Gain |
|---|---|---|---|
| Parameters | ~400K–6M | ~10M | 3.5× smaller avg. (25× categorization) |
| Shape Wins | **8/10** | 2/10 | 27% Error reduction |
| Trend Wins | **7/10** | 3/10 | 20% Error reduction |
| Downstream Utility | **9/10** | 1/10 | XGBoost performance on real test set |
| Training Time | **1.7× Faster** | Baseline | Despite 2.5× more epochs |

Cross-architecture portability (using the same default $(\lambda_\theta,\lambda_\ell,\lambda_c)=(15,15,8)$):

| Denoising Backbone | Shape Wins (+Geom vs. Baseline) | Trend Wins |
|---|---|---|
| Residual MLP | 9/10 | 8/10 |
| GNN + Laplacian eigenmap | 8/10 | 9/10 |
| Column-wise Transformer | 10/10 | 8/10 |
| **Total** | **27/30** | **25/30** |

Treating each "Architecture × Dataset × Metric" as a Bernoulli trial, 52 wins out of 60 yields a two-sided sign-test $p=5.21\times 10^{-9}$.

### Ablation Study
The critical "Input vs. Supervision" architecture-matched ablation:

| Configuration | Geometric Input | Prediction Head | Geometric Loss | vs. NoGeom Effect Size |
|---|---|---|---|---|
| NoGeom | No | No | No | baseline |
| InputsOnly | **Yes** | **Yes** | **No** | Cohen's $d=-0.08$ (negligible) |
| +Geom (GATD) | **Yes** | **Yes** | **Yes** | Cohen's $d=0.81$ (large effect) |

Other findings: (1) Replacing $\arctan/\log$ with raw differences leads to slightly worse performance; (2) Reducing geometric weights to the same scale as diffusion significantly degrades results; (3) Optimal $n_{\text{blocks}}$ for MLP is 0 for classification and 8 for regression.

### Key Findings
- **Supervision is the unique variable**: "Looking" at geometric features is useless; the model must be forced to "predict" them. This solidifies the argument that "explicit inductive bias = explicit supervision."
- **Geometric signals are structural**: The same default weights yield gains across three distinct denoiser architectures, indicating this is a universal auxiliary task for tabular diffusion rather than an MLP-specific fix.
- **Small models can challenge large ones**: An MLP with at most 6M parameters can outperform a 10M Transformer SOTA via auxiliary supervision, suggesting a significant tradeoff between compute and explicit structure in tabular data.
- **Weights must be inverted**: Contrary to multitask learning conventions, auxiliary weights must be much larger than the main task to effectively pull the local gradients of denoising MSE toward structural understanding.

## Highlights & Insights
- **Methodological Contribution > Method Contribution**: The InputsOnly vs. +Geom ablation is exemplary—it independently rules out "extra capacity," "extra feature channels," and "extra heads," attributing gain solely to "auxiliary supervision."
- **Portable Inductive Bias**: Packaging "geometric input + head + loss + $\mathbf{h}_{\text{aug}}$" as a drop-in module that works across backbones without retuning is a powerful approach that could extend to time series or graph diffusion.
- **Unified Scalar Space Technique**: Converting categorical columns to $[-1, 1]$ allows them to share geometric calculations with continuous columns, a practical trick to avoid complex differentiable distances for categories.
- **Reflection Boundary Handling**: Using reflection ($s\mapsto 2-s$ or $s\mapsto -s$) rather than hard clipping for values outside $[0, 1]$ effectively prevents synthetic data from accumulating at quantile boundaries.

## Limitations & Future Work
- **$O(d^2)$ Expansion**: The number of geometric features grows quadratically with columns, which may tax memory and compute for wide tables (hundreds of features); experiments were conducted on tables with up to ~48 columns.
- **Categorical Sorting Bias**: Deterministic index mapping for non-ordinal categories introduces a bias where "adjacent" categories are more likely to be confused, potentially affecting the symmetry of synthetic categorical structures.
- **Backbone-Specific Mechanisms**: Earlier explanations for MLP performance (e.g., "categorical anchor" mechanism) do not generalize cross-architecture, leaving the exact reason for GNN/Transformer gains as an open question.
- **Diffusion-Specific Validation**: Portability was only verified within the diffusion framework; effectiveness for GANs, VAEs, or autoregressive models remains unknown.

## Related Work & Insights
- **vs. TabDiff (Shi et al., 2025)**: Shares the EDM + masked diffusion base, but TabDiff relies on Transformer self-attention to **implicitly** learn column relations. GATD uses auxiliary loss to **explicitly** force learning, allowing a much smaller MLP to outperform it. Feeding GATD's signals back into a Transformer yields even further gains, showing the two are **complementary**.
- **vs. Geometric Deep Learning**: While GNNs apply geometry to data with *pre-existing* graph structures (molecules, point clouds), GATD "fabricates" a geometric relationship graph for seemingly unstructured tabular data and supervises it.
- **Insight**: Auxiliary task supervision is an undervalued design dimension—when the gradient of the main task (denoising MSE) does not point toward desired representations (column relations), designing a high-weight auxiliary task to supervise those representations is more effective than increasing architecture capacity.

## Rating
- Novelty: ⭐⭐⭐⭐ (Inverse application of geometric DL to unstructured tables with rigorous attribution to supervision).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Broad cross-architecture evaluation across 10 datasets and 60 test cases with sign-test verification).
- Writing Quality: ⭐⭐⭐⭐ (Clear logical progression and design motivation; minor organizational roughness).
- Value: ⭐⭐⭐⭐ (Provides a drop-in module for tabular diffusion and a methodological shift toward explicit auxiliary supervision).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GASS: Geometry-Aware Spherical Sampling for Disentangled Diversity Enhancement in Text-to-Image Generation](gass_geometry-aware_spherical_sampling_for_disentangled_diversity_enhancement_in.md)
- [\[ICML 2026\] Geometry-based Schrödinger Bridges for Trustworthy Multimodal Fusion](geometry-based_schrödinger_bridges_for_trustworthy_multimodal_fusion.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [\[ICML 2026\] Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation](semantic-aware_motion_encoding_for_topology-agnostic_character_animation.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](../../ICLR2026/image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)

</div>

<!-- RELATED:END -->
