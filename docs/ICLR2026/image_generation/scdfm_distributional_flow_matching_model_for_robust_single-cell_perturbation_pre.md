---
title: >-
  [Paper Note] scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction
description: >-
  [ICLR 2026][Image Generation][single-cell perturbation prediction] This paper proposes scDFM, a generative framework based on conditional flow matching (CFM) that enforces distribution-level fidelity via MMD regularization and introduces the PAD-Transformer backbone to handle noisy and sparse single-cell data. On combinatorial perturbation prediction, scDFM reduces MSE by 19.6% over the strongest baseline CellFlow.
tags:
  - ICLR 2026
  - Image Generation
  - single-cell perturbation prediction
  - conditional flow matching
  - MMD regularization
  - differential attention
  - gene co-expression graph
date: 2026-05-08
content_hash: f7799de4f35c6c05
---

# scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction

**Conference**: ICLR 2026
**arXiv**: [2602.07103](https://arxiv.org/abs/2602.07103)
**Code**: [GitHub](https://github.com/AI4Science-WestlakeU/scDFM)
**Area**: Image Generation
**Keywords**: single-cell perturbation prediction, conditional flow matching, MMD regularization, differential attention, gene co-expression graph

## TL;DR
This paper proposes scDFM, a generative framework based on conditional flow matching (CFM) that enforces distribution-level fidelity via MMD regularization and introduces the PAD-Transformer backbone to handle noisy and sparse single-cell data. On combinatorial perturbation prediction, scDFM reduces MSE by 19.6% over the strongest baseline CellFlow.

## Background & Motivation
- Predicting transcriptomic responses of cells to genetic or pharmacological perturbations is a central challenge in systems biology and drug discovery.
- Due to the destructive nature of RNA sequencing, pre- and post-perturbation states of the same cell cannot be jointly observed, yielding unpaired data.
- Existing methods (CPA, GEARS, etc.) primarily focus on mean expression profiles, neglecting higher-order distributional statistics such as variance, skewness, and shifts in subpopulation proportions.
- Single-cell data suffer from sparsity, zero-inflation, and severe noise; complex gene regulatory networks exist yet most models treat genes as independent features.
- **Core Motivation**: A generative framework is needed that models complete distributional changes while remaining robust to noise and sparsity.

## Method

### Overall Architecture
scDFM is built on conditional flow matching (CFM), learning a time-dependent velocity field $v_\theta(x_t | t, c_x, c_p)$ that transforms a noisy source distribution into the post-perturbation gene expression distribution. Training combines a CFM loss with a multi-kernel MMD regularizer, and the backbone network is the PAD-Transformer.

### Key Designs

1. **Conditional Flow Matching (CFM)**:

    - Directly applies the FM framework in the high-dimensional gene expression space (a first attempt in this domain).
    - Source distribution $x_0$ is noisy gene expression; target distribution $x_1$ is post-perturbation expression.
    - Linear interpolation path: $\pi_t(x_0, x_1) = (1-t)x_0 + tx_1$
    - Training objective: $\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}[\|v_\theta(x_t | t, c_x, c_p) - v(x_t | x_0, x_1, t, c_x, c_p)\|_2^2]$
    - **Design Motivation**: FM directly learns conditional transformations, making it well-suited for mapping noisy intermediate states to true perturbation states.

2. **Multi-Kernel MMD Regularization**:

    - CFM guarantees only local dynamics consistency, not terminal distribution alignment.
    - MMD is introduced to directly compare the generated distribution $\hat{X}_1$ against the true perturbation distribution $X_1$.
    - Mixed Gaussian RBF kernel: $k_{\text{mix}}(x, x') = \frac{1}{L}\sum_{\ell=1}^L \exp(-\frac{\|x-x'\|^2}{2\sigma_\ell^2})$
    - One-step endpoint prediction: $\hat{x}_1 = x_t + (1-t) \cdot v_\theta(x_t | t, c_x, c_p)$
    - Final objective: $\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda \mathcal{L}_{\text{MMD}}$
    - **Design Motivation**: Compensates for CFM's weakness in global distribution alignment, ensuring population-level fidelity.

3. **PAD-Transformer (Perturbation-Aware Differential Transformer)**:

    - **Gene co-expression graph attention mask**: Constructs a KNN graph based on Pearson correlation $w_{ij} = |\text{Cov}(x_i, x_j) / (\sigma(x_i)\sigma(x_j))|$, restricting attention computation to biologically relevant gene pairs.
    - **Differential attention module**: $\alpha_{\text{diff}} = A_1 - \lambda A_2$, suppressing irrelevant attention from noisy genes.
    - **Per-layer perturbation injection**: Perturbation embedding $e_p$ is injected at every layer via an MLP adapter.
    - Three-step refinement: perturbation injection → self differential attention → cross differential attention (using control representation $h_c$ to guide perturbation-state refinement).
    - **Design Motivation**: Standard Transformers tend to over-attend to noisy tokens; differential attention distinguishes control-state signals from perturbation-state signals.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda \mathcal{L}_{\text{MMD}}$, where $\lambda > 0$ balances trajectory consistency with terminal distribution fidelity.
- MMD bandwidth is adaptively selected via the median heuristic.
- Timestep $t$ is encoded with sinusoidal embeddings followed by an MLP, providing adaLN-Zero modulation.

## Key Experimental Results

### Main Results (Norman Additive Split)

| Model | MSE ↓ | MAE ↓ | DE-Spearman ↑ | DS ↑ | Pearson $\hat{\Delta}_{20}$ ↑ |
|------|-------|-------|---------------|------|------|
| scDFM (Ours) | **0.00315** | **0.02155** | **0.5705** | **0.9737** | **0.9260** |
| CellFlow | 0.00392 | 0.02207 | 0.5503 | 0.9321 | 0.8988 |
| GEARS | 0.01387 | 0.06624 | 0.5624 | 0.8601 | 0.2032 |
| scGPT | 0.01349 | 0.03796 | 1.07e-5 | 0.5404 | 0.2414 |
| CPA | 0.03435 | 0.07894 | 0.0713 | 0.6021 | 0.2254 |

### Ablation Study

| Configuration | Key Metric Change | Note |
|------|---------|------|
| w/o MMD | MSE increases, DS decreases | MMD is critical for distribution-level fidelity |
| w/o gene co-expression graph | DE-Spearman decreases | Biologically-informed attention masking is effective |
| w/o differential attention | Increased noise sensitivity | Differential attention suppresses noise |
| Standard Transformer replacing PAD | Overall degradation | PAD-Transformer components are complementary |

### Key Findings
- scDFM reduces MSE by 19.6% over CellFlow (0.00315 vs. 0.00392), while achieving a discriminator score (DS) of 0.9737.
- Strong performance is also maintained in the holdout setting (unseen perturbations), validating generalization capability.
- Pretrained models such as scGPT yield near-zero DE-Spearman scores, indicating that foundation models struggle to capture perturbation-specific effects.
- The additive baseline is itself competitive (consistent with Ahlmann-Eltze et al.), suggesting that combinatorial perturbations often exhibit approximately additive effects.

## Highlights & Insights
- scDFM is the first to directly apply conditional flow matching in high-dimensional gene expression space, operating more directly than CellFlow, which works in PCA space.
- MMD regularization elegantly compensates for CFM's local-only consistency guarantee, achieving dual fidelity at both the local (trajectory) and global (distribution) levels.
- The gene co-expression graph serves as a biological prior injected into the attention mask, effectively filtering noise while preserving regulatory structure.
- Differential attention is particularly suited for noisy biological data—only a subset of genes responds to a perturbation, and the rest should be suppressed.

## Limitations & Future Work
- Validation is limited to two datasets: Norman (genetic perturbation) and ComboSciPlex (drug perturbation).
- Pre-computation of the gene co-expression graph introduces additional computational overhead in data preparation.
- The distribution-level evaluation metric (DS), while useful, does not directly reflect biological interpretability.
- No detailed comparison is made against recent diffusion-based methods (e.g., scDiffusion).

## Related Work & Insights
- CellFlow (Klein et al. 2025): Performs flow matching in PCA space; scDFM operates in the original expression space.
- GEARS (Roohani et al. 2024): Incorporates biological priors such as gene ontology; scDFM uses co-expression graphs instead.
- Diff Transformer (Ye et al. 2025): Original proposal of differential attention; scDFM adapts it to perturbation prediction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of CFM + MMD + PAD-Transformer is innovative and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-setting evaluation, comprehensive metrics, and ablation studies are included.
- Writing Quality: ⭐⭐⭐⭐ Technical descriptions are clear and motivations are well-articulated.
- Value: ⭐⭐⭐⭐⭐ Significant value for computational biology; code is publicly available.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](../../CVPR2026/image_generation/renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[ICLR 2026\] FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](flowcast_advancing_precipitation_nowcasting_with_conditional_flow_matching.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)

<!-- RELATED:END -->
