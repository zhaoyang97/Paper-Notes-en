---
title: >-
  [Paper Note] Towards Precise Scaling Laws for Video Diffusion Transformers
description: >-
  [CVPR 2025][Video Generation][Scaling Laws] This paper systematically verifies the existence of scaling laws in Video Diffusion Transformers (Video DiT) for the first time. It is discovered that video models are more sensitive to learning rate and batch size than language models. Subsequently, the paper proposes a precise scaling law formula that simultaneously predicts optimal hyperparameters, optimal model size, and validation loss, reducing inference cost by 40.1% or model…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Scaling Laws"
  - "Video Diffusion"
  - "Hyperparameter Optimization"
  - "DiT"
  - "Compute Budget Allocation"
date: 2026-05-08
content_hash: 09df366e7eee1d22
---

# Towards Precise Scaling Laws for Video Diffusion Transformers

**Conference**: CVPR 2025  
**arXiv**: [2411.17470](https://arxiv.org/abs/2411.17470)  
**Code**: None  
**Area**: Diffusion Models/Video Generation  
**Keywords**: Scaling Laws, Video Diffusion, Hyperparameter Optimization, DiT, Compute Budget Allocation

## TL;DR

This paper systematically verifies the existence of scaling laws in Video Diffusion Transformers (Video DiT) for the first time. It is discovered that video models are more sensitive to learning rate and batch size than language models. Subsequently, the paper proposes a precise scaling law formula that simultaneously predicts optimal hyperparameters, optimal model size, and validation loss, reducing inference cost by 40.1% or model size by 39.9% under the same compute budget.

## Background & Motivation

**Background**: Video Diffusion Transformers (such as Sora/Movie Gen) have significantly improved video generation quality by scaling up model sizes, with the largest models currently reaching 30 billion parameters. Scaling laws in large language models (such as OpenAI Scaling Law, Chinchilla) have been widely used to predict the optimal model size and budget allocation, but scaling laws in visual generation models, especially video models, remain largely unexplored.

**Limitations of Prior Work**: (1) Directly applying existing LLM scaling laws to Video DiT yields inaccurate predictions because video models are more sensitive to batch size and learning rate; (2) The OpenAI scaling law assumes smaller batch sizes are more efficient but ignores the impact of hyperparameters on fitting precision; (3) The Chinchilla law derives the optimal model size through the relationship between loss function and compute budget, but its loss fitting itself is not precise enough; (4) The DeepSeek scaling law introduces optimal hyperparameters but ignores the effect of model size.

**Key Challenge**: The sensitivity of Video DiT to hyperparameters (learning rate, batch size) causes systematic bias in scaling laws fitted with fixed, non-optimal hyperparameters—which tends to recommend overly large model sizes and yield higher validation losses.

**Goal**: (1) Confirm the existence of scaling laws in Video DiT; (2) Establish optimal hyperparameter prediction formulas that incorporate model size and data volume; (3) Derive precise performance predictions for any model size and compute budget.

**Key Insight**: Starting from the convergence theory of mini-batch SGD, the power-law relationships of the optimal batch size and learning rate with respect to model size $N$ and the number of training tokens $T$ are derived. Small-scale model experiments are used to fit the parameters, which are then extrapolated to large-scale models.

**Core Idea**: Two power-law formulas, $B_{\text{opt}} = \alpha_B T^{\beta_B} N^{\gamma_B}$ and $\eta_{\text{opt}} = \alpha_\eta T^{\beta_\eta} N^{\gamma_\eta}$, are proposed to explicitly incorporate model size into hyperparameter prediction. This enables fitting a more precise model-performance-budget scaling law under optimal hyperparameters.

## Method

### Overall Architecture

A three-tiered progressive scaling law system is established: (1) **Hyperparameter Scaling**—establishing the power-law formulas for the optimal batch size $B_{\text{opt}}(N, T)$ and learning rate $\eta_{\text{opt}}(N, T)$; (2) **Performance Scaling**—fitting the relationship between validation loss $L$, model size $N$, and training tokens $T$ under optimal hyperparameters; (3) **Budget Allocation**—deriving the optimal model size $N_{\text{opt}}(C)$ given a compute budget $C$. Experiments are fitted on small models ranging from 0.02B to 0.26B and extrapolated to 1.07B for validation.

### Key Designs

1. **Scaling Laws for Hyperparameters**:

    - Function: Given the model size and training data volume, precisely predict the optimal batch size and learning rate.
    - Mechanism: Theoretical derivation: Starting from the Lipschitz smoothness assumption and mini-batch SGD convergence analysis, the step loss change is $\Delta L_k \approx -\eta \|G(\theta_k)\|^2 + \frac{1}{2}\eta^2(G^T H G + \frac{\text{tr}(H\Sigma)}{B})$. The optimal learning rate is $\eta_{\text{opt}}(B) = \frac{\|G\|^2}{G^T H G + \text{tr}(H\Sigma)/B}$. As the model size scales up, the Lipschitz constant $L$ increases, requiring a smaller learning rate and a larger batch size. Empirical fitting yields $B_{\text{opt}} = 2.18 \times 10^4 \cdot T^{0.81} \cdot N^{0.19}$ and $\eta_{\text{opt}} = 0.0002 \cdot T^{-0.045} \cdot N^{-0.162}$. Validation on a 1.07B model confirms that the predicted values indeed correspond to the minimum validation loss.
    - Design Motivation: Prior scaling law studies either ignore hyperparameters (OpenAI) or do not consider model size (DeepSeek). Experiments show that using fixed hyperparameters in Video DiT leads to systematically higher loss points on the loss curve (Fig.1 grey dots vs red dots), making the fitting imprecise. Explicitly modeling model size as an independent variable is key.

2. **Performance Scaling with Optimal Hyperparameters**:

    - Function: Precisely predict the achievable validation loss for any model size and number of training tokens.
    - Mechanism: Fit the parametric form of $L(N, T)$ under optimal hyperparameters. Unlike Chinchilla, which uses IsoFLOP profiles, this paper directly fits the loss surface over experimental data points. This yields more accurate predictions when extrapolating to larger models—with a prediction error of only about 0.5% for 1.07B/10B tokens. It is found that under a fixed compute budget, the validation loss changes very little (flat region) when the model size is adjusted around its optimal value, providing theoretical support for using smaller models (to reduce inference costs) in practice.
    - Design Motivation: Loss curves fitted with non-optimal hyperparameters are systematically shifted upward, overestimating the optimal model size. Fitting under optimal hyperparameters eliminates this bias, leading to higher extrapolation accuracy.

3. **Relationship between Optimal Model Size and Compute Budget**:

    - Function: Directly predict the optimal model parameter count given a compute budget.
    - Mechanism: Fitting yields $N_{\text{opt}} = 1.5787 \cdot C^{0.4146}$, where the exponent 0.4146 lies between OpenAI (0.73) and Chinchilla (0.50), indicating that Video DiT requires a more balanced allocation between model size and data volume than LLMs. Taking the compute budget of Movie Gen (6144 H100 GPUs) as an example, this method recommends an 18.05B model (vs 30.05B using the fixed hyperparameter method), which reduces parameters by 39.9% while maintaining comparable performance.
    - Design Motivation: In actual deployment, inference cost is directly related to model size. Achieving similar performance with a smaller model can significantly reduce deployment costs.

### Loss & Training

Standard DDPM denoising target. Constant learning rates (without cosine annealing) are used to simplify the problem. Experiments are conducted on four models (0.017B/0.057B/0.13B/0.26B), with each model trained under various configurations of 2B to 12B tokens. The compute budget is defined as $C_{\text{token}} = \frac{3}{4}N(7 + n_{\text{ctx}}/d)$.

## Key Experimental Results

### Main Results

| Model Size | Method | Recommended Model | Validation Loss |
|---------|------|---------|---------|
| Movie Gen Budget | Fixed Hyperparameters | 30.05B | Baseline |
| Movie Gen Budget | **Optimal Hyperparameters** | **18.05B (-39.9%)** | **Comparable** |
| 1e10 TFlops | Fixed Hyperparameters | Baseline | Baseline |
| 1e10 TFlops | **Optimal Hyperparameters** | **-40.1% Inference Cost** | **Comparable** |

### Ablation Study (Hyperparameter Prediction Validation - 1.07B Model)

| Configuration | 4B tokens Pred | 4B tokens Actual | 10B tokens Pred | 10B tokens Actual |
|-----|-------------|-------------|-------------|-------------|
| Optimal BS | Predicted | ✓ Minimizes Loss | Predicted | ✓ Minimizes Loss |
| Optimal LR | Predicted | ✓ Minimizes Loss | Predicted | ✓ Minimizes Loss |

### Key Findings

- **Video DiT is more sensitive to hyperparameters than LLMs**: Scaling laws fitted using fixed, non-optimal hyperparameters overestimate the optimal model size by 39.9%. While this bias is less noticeable in LLMs, it has a significant impact on Video DiT.
- **Model size influences optimal hyperparameters**: Larger models require a larger batch size ($\gamma_B = 0.19 > 0$) and a smaller learning rate ($\gamma_\eta = -0.16 < 0$), which aligns with SGD performance theory (where larger models have larger Lipschitz constants).
- **Flat loss near the optimal region**: Under a fixed compute budget, deviating from the optimal model size by $\pm 30\%$ results in a loss increase of $< 1\%$, allowing for a flexible trade-off between inference efficiency and performance.
- The scaling laws successfully extrapolate from 0.26B to 1.07B ($4 \times$ scale-up), validating the reliability of extrapolation.
- The appendix validates that the same formula holds for Image DiT (1-frame video).

## Highlights & Insights

- **Jointly modeling hyperparameters and model size as part of the scaling law** is the key methodological contribution: Prior works either ignored hyperparameters (assuming fixed configurations are good enough) or assumed hyperparameters depend only on data volume. This work theoretically and experimentally verifies that model size is an independent variable in hyperparameter selection.
- **High practical value**: Training Video DiT is extremely expensive (Movie Gen uses 6144 H100 GPUs). Fitting scaling laws with small-model experiments allows determining the optimal configuration before starting large-scale training, saving massive compute resources.
- The discovery of a flat optimal region provides deployment flexibility—one does not have to strictly use the theoretically optimal model size, but can opt for a slightly smaller model depending on hardware constraints.

## Limitations & Future Work

- The scaling law is only established on verification loss and is not directly correlated with downstream video quality metrics (such as FVD, human preference).
- Constant learning rates are used to simplify the problem; cosine annealing, which is commonly used in realistic training, might alter the optimal configuration.
- Extrapolation is only validated up to 1.07B (about $4\times$ scale-up), and the accuracy at the $10\text{B}+$ scale remains unknown.
- The impacts of factors such as data quality and data diversity on scaling laws are not analyzed.

## Related Work & Insights

- **vs Chinchilla (Hoffmann et al.)**: Chinchilla establishes an equal-scaling law of model and data for LLMs (exponent of 0.50), whereas the exponent for Video DiT is 0.4146, indicating that video models require a larger allocation proportion for data. Moreover, Chinchilla uses the IsoFLOP profile method, which this study finds to be less precise for loss fitting.
- **vs DeepSeek Scaling Law**: DeepSeek was the first to model optimal hyperparameters, but it did not include model size as an independent variable. This paper extends this idea and provides theoretical backing.
- **vs Movie Gen**: Movie Gen uses 30B parameters, whereas the scaling laws in this paper suggest that an 18B model can achieve the same performance under the same compute budget, reducing inference costs by 40%.

## Rating

- Novelty: ⭐⭐⭐⭐ First to establish precise scaling laws incorporating hyperparameters for Video DiT
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid pipeline of theoretical derivation, small-scale fitting, and large-scale validation
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical analysis with clear formula derivations
- Value: ⭐⭐⭐⭐⭐ Highly direct practical guidance for training video generation models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](video_motion_transfer_with_diffusion_transformers.md)
- [\[CVPR 2025\] VideoDirector: Precise Video Editing via Text-to-Video Models](videodirector_precise_video_editing_via_text-to-video_models.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[NeurIPS 2025\] Scaling RL to Long Videos](../../NeurIPS2025/video_generation/scaling_rl_to_long_videos.md)

</div>

<!-- RELATED:END -->
