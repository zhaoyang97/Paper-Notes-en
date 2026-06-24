---
title: >-
  [Paper Note] RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation
description: >-
  [AAAI 2026][Image Restoration][missing data imputation] RefiDiff proposes a four-stage framework (pre-processing → warm-up → diffusion → polishing) that progressively unifies the predictive and generative imputation paradigms for the first time. Combined with a Mamba-based denoising network, it achieves state-of-the-art performance across 9 datasets while running 4× faster than DIFFPUTER.
tags:
  - "AAAI 2026"
  - "Image Restoration"
  - "missing data imputation"
  - "diffusion model"
  - "Mamba"
  - "tabular data"
  - "MNAR"
date: 2026-05-08
content_hash: 81c54f15b2e914f9
---

# RefiDiff: Progressive Refinement Diffusion for Efficient Missing Data Imputation

**Conference**: AAAI 2026
**arXiv**: [2505.14451](https://arxiv.org/abs/2505.14451)  
**Code**: [GitHub](https://github.com/Atik-Ahamed/RefiDiff)  
**Area**: Data Imputation / Tabular Data
**Keywords**: missing data imputation, diffusion model, Mamba, tabular data, MNAR

## TL;DR
RefiDiff proposes a four-stage framework (pre-processing → warm-up → diffusion → polishing) that progressively unifies the predictive and generative imputation paradigms for the first time. Combined with a Mamba-based denoising network, it achieves state-of-the-art performance across 9 datasets while running 4× faster than DIFFPUTER.

## Background & Motivation

### State of the Field

**Background**: Missing values are ubiquitous in high-dimensional mixed-type (numerical + categorical) datasets and arise under three mechanisms: MCAR (Missing Completely At Random), MAR (Missing At Random), and MNAR (Missing Not At Random). Existing imputation methods fall into two paradigms: predictive methods (e.g., XGBoost regression) are efficient and deterministic but lack uncertainty modeling and tend toward a local view; generative methods (e.g., diffusion models) can model global distributions but are computationally intensive.

### Limitations of Prior Work

**Limitations of Prior Work**: (1) The two paradigms each have distinct advantages but are rarely unified effectively — MICE-style methods require slow iterative convergence, while DIFFPUTER relies on EM iterations with poor efficiency; (2) existing diffusion methods (TabDDPM, DIFFPUTER) use Transformers as denoisers, incurring substantial computational overhead on high-dimensional tabular data; (3) the MNAR setting is the most challenging, since the missingness mechanism is correlated with the missing values themselves, causing most methods to degrade significantly.

### Root Cause

**Key Challenge**: Predictive methods excel at locally precise imputation but cannot model global distributional uncertainty, while generative methods excel at global distribution modeling but introduce excessive noise and computational overhead. The central challenge is how to obtain the advantages of both paradigms simultaneously without sacrificing efficiency.

### Resolution

**Goal**: Design a progressive framework in which a predictive method provides a high-quality initialization and a generative method subsequently refines the global distribution. **Key Insight**: Decompose the imputation process into three stages — warm-up (predictive), diffusion (generative), and polishing (predictive) — each progressively improving imputation quality. **Core Idea**: Apply XGBoost in a single-pass warm-up to produce an initial imputation, use a Mamba-based diffusion model to perform global distributional correction, and finally apply regression polishing to remove residual noise.

## Method

### Overall Architecture
RefiDiff comprises a four-stage pipeline: Pre-processing → Warm-up Refinement → Diffusion Imputation → Post-processing & Polishing. The input is mixed-type tabular data with missing values $\mathbf{Z} \in \mathbb{R}^{N \times D}$ and a missingness mask $\mathbf{M}$; the output is the fully imputed dataset.

### Key Designs

1. **Pre-processing + Warm-up Refinement**:

    - **Function**: Normalize raw data and provide a one-pass initial imputation.
    - **Mechanism**: Categorical features are binary-encoded; numerical features are standardized using the mean $\mu$ and standard deviation $\sigma$ computed from observed values, with missing positions filled with zero. A lightweight XGBoost model $\theta_1^{(j)}$ is then trained for each feature column $f_j$, and all missing values are imputed in a single pass. Three key properties hold: Non-overwriting (observed values are not modified), Well-defined mapping (each column has a dedicated predictor), and One-pass completion (no iteration required).
    - **Design Motivation**: Unlike the multi-round iterations of MICE, the single-pass warm-up dramatically reduces computational cost while providing a substantially better starting point for diffusion than random noise.

2. **Mamba-based Diffusion Module**:

    - **Function**: Global distribution modeling and conditional sampling.
    - **Mechanism**: A VE SDE continuous-time diffusion framework is adopted. A Mamba-based denoising network $\theta_2$ is trained using a diamond architecture (2 up-sampling + 2 down-sampling residual blocks), each containing a Mamba layer, FC layer, positional encoding, LayerNorm, and Dropout. The training objective is the EDM loss: $\mathcal{L}_{SM} = \mathbb{E}_{X_0,\varepsilon,t}[\|\theta_2(X_t,t,M) - \nabla_{X_t}\log p(X_t|X_0)\|_2^2]$. At inference, $N$ reverse diffusion runs are averaged, and observed positions are always clamped unchanged.
    - **Design Motivation**: Replacing the Transformer with Mamba achieves linear complexity, capturing long-range dependencies while delivering a 4× speedup. The diamond architecture enhances expressiveness through multi-scale feature fusion.

3. **Post-refinement (Polishing)**:

    - **Function**: Remove residual noise from the diffusion output.
    - **Mechanism**: A column-wise regression correction is applied to the diffusion output in a second pass, exploiting the local precision of predictive methods to compensate for noise potentially introduced by the generative stage.
    - **Design Motivation**: The stochasticity of diffusion sampling may introduce small perturbations in certain columns; the polishing stage uses deterministic prediction to correct these deviations.

### Theoretical Guarantee
The paper establishes a KL divergence upper bound: $\text{KL} \leq C_1 T \varepsilon_\theta^2 + C_2 \delta t + C_3 / N$, proving that the conditional sampling result converges to the true distribution as denoiser accuracy, time-step density, and number of samples increase.

## Key Experimental Results

### Main Results

Evaluation is conducted on 9 real-world datasets under all three missingness mechanisms: MCAR, MAR, and MNAR.

| Method | MNAR MAE/RMSE | MCAR MAE/RMSE | MAR MAE/RMSE | Avg Rank |
|------|:---:|:---:|:---:|:---:|
| DIFFPUTER | 37.27/86.86 | 31.72/63.49 | 39.15/90.95 | 2.67 |
| ReMasker | 39.66/80.23 | 35.84/65.19 | 38.39/78.82 | 3.00 |
| **RefiDiff** | **34.49/78.83** | **31.41/63.16** | **34.52/78.22** | **1.17** |

RefiDiff runs **4× faster** than DIFFPUTER with fewer parameters.

### Ablation Study

| Configuration | OOS RMSE (MAR) | OOS RMSE (MNAR) | Note |
|------|:---:|:---:|------|
| Full RefiDiff | 73.82 | 70.12 | Complete model |
| w/o Diffusion | 91.80 | 81.07 | −diffusion: +24.3%/+15.6% |
| w/o Warm-up | 82.45 | 76.89 | −warm-up: +11.7%/+9.6% |
| w/o Polishing | 78.93 | 73.41 | −polishing: +6.9%/+4.7% |

### Key Findings
- The diffusion module contributes most: removing it raises MAR RMSE from 73.82 to 91.80 (+24.3%), demonstrating the indispensability of global distribution modeling.
- Warm-up provides effective initialization: removing it causes an 11.7% performance drop, indicating that predictive initialization substantially benefits diffusion quality.
- The largest gains appear in the MNAR setting: RMSE is 9.3% lower than DIFFPUTER (78.83 vs. 86.86), as the warm-up provides more accurate initial estimates for MNAR-missing values.
- Classification accuracy also ranks first (average rank 1.17).

## Highlights & Insights
- **Unified paradigm design**: The first work to seamlessly integrate predictive and generative imputation through a progressive warm-up → diffusion → polishing refinement pipeline, yielding an elegant and practical solution.
- **Mamba replacing Transformer**: Validates Mamba's ability to capture long-range dependencies on non-sequential (tabular) data, achieving a 4× speedup with superior performance.
- **Plug-and-play**: No architecture or hyperparameter adjustments are needed for different datasets, ensuring strong generality.
- The three-stage progressive design of "Warm-up + Generation + Polishing" is generalizable to other generative imputation and restoration tasks.

## Limitations & Future Work
- Binary encoding of categorical features followed by continuous diffusion may compromise discrete semantic fidelity.
- Validation is limited to medium-scale datasets (e.g., UCI); applicability to million-scale datasets remains unexplored.
- The single-pass warm-up may be insufficient under extremely high missingness rates (>70%).
- Incremental imputation for streaming or online data scenarios has not been explored.

## Related Work & Insights
- **vs. DIFFPUTER**: Eliminates iterative EM; a single warm-up pass combined with one diffusion run achieves better performance. Mamba replaces TabDDPM's Transformer for a 4× speedup.
- **vs. ReMasker**: ReMasker uses masked autoencoding, whereas RefiDiff employs diffusion with progressive refinement, yielding better out-of-sample (OOS) generalization.
- **vs. MICE**: MICE requires multiple iterations until convergence; the predictive stage in RefiDiff is one-pass, offering an order-of-magnitude improvement in efficiency.
- The successful application of Mamba to non-image, non-text tasks warrants attention and could be extended to other long-sequence modalities such as audio and time-series signals.

## Rating
- Novelty: ⭐⭐⭐⭐ The progressive unification of two paradigms is elegant; the application of Mamba to tabular data is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 datasets, 3 missingness mechanisms, complete ablation study.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are well integrated; the four-stage pipeline is clearly presented.
- Value: ⭐⭐⭐⭐ Practically valuable to the missing data imputation community, with notable contributions in the MNAR setting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](../../CVPR2025/image_restoration/progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[ICLR 2026\] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](../../ICLR2026/image_restoration/horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)
- [\[NeurIPS 2025\] Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](../../NeurIPS2025/image_restoration/enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)
- [\[ICLR 2026\] FideDiff: Efficient Diffusion Model for High-Fidelity Image Motion Deblurring](../../ICLR2026/image_restoration/fidediff_efficient_diffusion_model_for_high-fidelity_image_motion_deblurring.md)
- [\[CVPR 2026\] 2-Shots in the Dark: Low-Light Denoising with Minimal Data Acquisition](../../CVPR2026/image_restoration/2-shots_in_the_dark_low-light_denoising_with_minimal_data_acquisition.md)

</div>

<!-- RELATED:END -->
