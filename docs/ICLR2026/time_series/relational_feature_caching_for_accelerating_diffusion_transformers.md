---
title: >-
  [Paper Note] Relational Feature Caching for Accelerating Diffusion Transformers
description: >-
  [ICLR 2026][Time Series][Feature Caching] This paper proposes Relational Feature Caching (RFC), a framework that enhances the accuracy of cached feature prediction by exploiting the strong correlation between input and output features of DiT modules. RFC comprises two components: RFE, which estimates output change magnitude from input variations, and RCS, which uses input prediction error as a proxy to determine when full computation is required. RFC significantly outperforms existing temporal extrapolation-based caching methods on both image and video generation tasks.
tags:
  - ICLR 2026
  - Time Series
  - Feature Caching
  - DiT Acceleration
  - Input-Output Relationship
  - Dynamic Scheduling
  - Prediction Accuracy
date: 2026-05-08
content_hash: 3481ece605c742a2
---

# Relational Feature Caching for Accelerating Diffusion Transformers

**Conference**: ICLR 2026  
**arXiv**: [2602.19506](https://arxiv.org/abs/2602.19506)  
**Code**: [Project Page](https://cvlab.yonsei.ac.kr/projects/RFC)  
**Area**: Diffusion Models / Inference Acceleration  
**Keywords**: Feature Caching, DiT Acceleration, Input-Output Relationship, Dynamic Scheduling, Prediction Accuracy

## TL;DR
This paper proposes Relational Feature Caching (RFC), a framework that enhances the accuracy of cached feature prediction by exploiting the strong correlation between input and output features of DiT modules. RFC comprises two components: RFE, which estimates output change magnitude from input variations, and RCS, which uses input prediction error as a proxy to determine when full computation is required. RFC significantly outperforms existing temporal extrapolation-based caching methods on both image and video generation tasks.

## Background & Motivation
- **State of the Field**: Diffusion Transformers (DiT) achieve strong performance on text-to-image and video generation tasks, but incur prohibitively high inference costs due to repeated expensive forward passes across many denoising timesteps. Feature caching methods exploit the high similarity between features at adjacent timesteps: full computation is performed and outputs are cached at selected timesteps, while subsequent timesteps reuse or predict cached features to skip redundant computation.
- **Limitations of Prior Work**: (1) Early caching methods (FORA, DeepCache) directly reuse cached features without adjustment, causing accumulated errors and severe quality degradation at large cache intervals. (2) Recent prediction-based methods (FasterCache with linear extrapolation, TaylorSeer with Taylor expansion) assume smooth temporal evolution of features, but in practice **the magnitude of output feature changes is highly irregular across timesteps**, leading to significant prediction errors from purely temporal extrapolation. (3) Fixed uniform cache interval schedules ignore error variability across timesteps, resulting in suboptimal efficiency.
- **Root Cause**: Temporal extrapolation-based caching methods fail to capture the irregularity of output feature change magnitudes, yet directly measuring output errors requires expensive full computation.
- **Paper Goals**: To more accurately predict cached features and dynamically determine when full computation is necessary, without introducing significant computational overhead.
- **Starting Point**: Through detailed feature analysis, the authors identify a key observation: **the change in input features and the change in output features of the same module are highly correlated**, and obtaining input features requires only lightweight operations (LayerNorm, scaling, shifting) that are essentially free.
- **Core Idea**: Leverage the input-output relationship to enhance feature prediction: (1) use the magnitude of input feature changes to estimate the magnitude of output feature changes (RFE); (2) use input prediction error as a proxy to estimate output prediction error and dynamically trigger full computation (RCS).

## Method

### Overall Architecture
RFC builds upon existing Taylor expansion-based caching methods and enhances feature prediction accuracy through two complementary components: RFE, which more accurately estimates output features at cache steps, and RCS, which dynamically determines when full computation is needed. Both components exploit input-output feature correlations and require only lightweight additional input feature operations. The overall pipeline is: cache output and input features at full-computation steps → at cache steps, use RFE to correct output predictions via input changes → simultaneously, RCS monitors accumulated input prediction error → trigger full computation when the accumulated error exceeds a threshold.

### Key Designs

1. **RFE (Relational Feature Estimation)**:

    - **Function**: Estimates the magnitude of output feature changes to enhance the prediction accuracy of Taylor expansion.
    - **Mechanism**: The ratio of output change to input change is defined as $s_k(t-k) = \frac{\|\Delta_k O(t-k)\|_2}{\|\Delta_k I(t-k)\|_2}$. Empirical analysis shows this ratio is remarkably stable across timesteps (relative standard deviation ~2%). Theoretically, when the input-to-output mapping is locally linear and the direction of input change remains consistent, this ratio is invariant with respect to $k$. Based on this, the ratio $s_N(t)$ estimated from the two most recent full-computation steps is used to approximate the output change magnitude: $\|\Delta_k O(t-k)\| \approx s_N(t) \|\Delta_k I(t-k)\|_2$. The final prediction formula is: $O_{\text{RFE}}(t-k) = O(t) + (s_N(t)\|\Delta_k I(t-k)\|_2) \cdot g\left(\sum_{i=1}^{m}\frac{k^i}{i!}\frac{\Delta_N^i O(t)}{N^i}\right)$, where $g(\cdot)$ denotes L2 normalization. That is, the Taylor expansion provides the **direction** of change, while the **magnitude** is estimated from input feature variations.
    - **Design Motivation**: Pure temporal extrapolation cannot adapt to the irregularity of output feature change magnitudes. Since input feature changes are highly correlated with output feature changes, and obtaining input features incurs negligible computational cost (requiring only lightweight operations such as LayerNorm), input information can be leveraged essentially for free to correct the predicted magnitude.

2. **RCS (Relational Cache Scheduling)**:

    - **Function**: Dynamically determines when to execute full computation, replacing fixed-interval cache schedules.
    - **Mechanism**: The input prediction error is defined as $\mathcal{E}_I(t-k) = \frac{\|E_I(t-k)\|_1}{\|I(t-k)\|_1}$, where $E_I(t-k) = I(t-k) - I_{\text{Taylor}}(t-k)$ is the difference between the actual and Taylor-predicted input features. Full computation is triggered when the accumulated input error exceeds a threshold: $\sum_{j=1}^{k} \mathcal{E}_I(t-j) > \tau$. The quality-efficiency trade-off is controlled by adjusting the threshold $\tau$.
    - **Design Motivation**: Directly measuring output prediction error requires full computation (a catch-22 problem), but output and input error trends are highly consistent (Fig. 2b), making input error a suitable proxy. Monitoring the input error of only the first module is sufficient, without the need to monitor all modules (validated in Table 6), keeping overhead minimal.

3. **Theoretical Guarantee (Proposition 1)**:

    - **Function**: Provides a theoretical basis for the approximate invariance of the ratio $s_k(t-k)$ with respect to $k$.
    - **Mechanism**: Assuming the input-to-output mapping is locally linear $O(t) = AI(t) + b$ and that the direction of input change $u_k(t-k)$ remains constant for $1 \leq k \leq N$, then $s_k(t-k) = \|A u_k(t-k)\|_2$ is a constant. Both assumptions are reasonable in diffusion models: small feature changes between adjacent timesteps support local linearity, and prior work has shown that the direction of feature change remains consistent across timesteps.
    - **Design Motivation**: Provides theoretical support for the core operation of RFE, justifying the use of $s_N(t)$ to approximate $s_k(t-k)$.

### Loss & Training
RFC is a **training-free** inference acceleration framework that requires no additional training or fine-tuning. It is applied directly to the inference phase of pretrained DiT models. The only hyperparameter to tune is the threshold $\tau$ in RCS; in experiments, $\tau$ is adjusted to match the number of full computations (NFC) with baseline methods for fair comparison. The Taylor expansion order $m$ is typically set to 1 or 2.

## Key Experimental Results

### Main Results

**Class-Conditional Image Generation (DiT-XL/2, ImageNet)**:

| Method | NFC | FLOPs(T) | FID↓ | sFID↓ | FID2FC↓ | sFID2FC↓ |
|--------|-----|----------|------|-------|---------|----------|
| Full-Compute | 50 | 23.74 | 2.32 | 4.32 | - | - |
| TaylorSeer (N=4) | 14 | 6.66 | 2.55 | 5.30 | 0.44 | 2.17 |
| **RFC (m=2)** | 14.01 | 6.67 | **2.52** | **4.60** | **0.30** | **1.33** |
| TaylorSeer (N=7) | 8 | 3.82 | 3.46 | 6.97 | 1.30 | 5.61 |
| **RFC (m=2)** | 8.02 | 3.83 | **3.12** | **5.07** | **0.81** | **3.10** |
| TaylorSeer (N=9) | 7 | 3.35 | 4.90 | 7.92 | 2.33 | 7.35 |
| **RFC (m=2)** | 7.04 | 3.37 | **3.40** | **5.21** | **1.03** | **3.66** |

**Text-to-Image Generation (FLUX.1 dev, DrawBench)**:

| Method | NFC | FLOPs(T) | PSNR↑ | SSIM↑ | LPIPS↓ | IR↑ |
|--------|-----|----------|-------|-------|--------|-----|
| Full-Compute | 50 | 2813.50 | - | - | - | 0.9655 |
| TaylorSeer (N=4,m=2) | 14 | 788.59 | 19.77 | 0.771 | 0.318 | 0.941 |
| **RFC (m=2)** | 13.80 | 777.44 | **20.35** | **0.793** | **0.295** | **0.950** |
| TaylorSeer (N=9,m=2) | 8 | 451.10 | 16.55 | 0.656 | 0.533 | 0.800 |
| **RFC (m=2)** | 8.03 | 452.91 | **16.92** | **0.694** | **0.471** | **0.919** |

**Text-to-Video Generation (HunyuanVideo, VBench)**:

| Method | NFC | FLOPs(T) | PSNR↑ | SSIM↑ | LPIPS↓ | VBench↑ |
|--------|-----|----------|-------|-------|--------|---------|
| Full-Compute | 50 | 7520.00 | - | - | - | 81.40 |
| TaylorSeer (N=6,m=1) | 9 | 1359.19 | 15.53 | 0.461 | 0.245 | 79.52 |
| **RFC (m=1)** | 8.96 | 1354.65 | **18.54** | **0.635** | **0.133** | **80.83** |
| TaylorSeer (N=8,m=1) | 7 | 1058.45 | 15.20 | 0.441 | 0.262 | 79.59 |
| **RFC (m=1)** | 7.09 | 1072.65 | **18.25** | **0.616** | **0.144** | **80.49** |

### Ablation Study

**Component Ablation (DiT-XL/2, m=1)**:

| Method | NFC | FID↓ | sFID↓ | FID2FC↓ | sFID2FC↓ |
|--------|-----|------|-------|---------|----------|
| TaylorSeer | 14 | 2.65 | 5.60 | 0.57 | 2.77 |
| +RFE | 14 | 2.52 | 5.18 | 0.43 | 2.02 |
| +RCS | 14 | 2.52 | 4.76 | 0.36 | 1.88 |
| **RFC (RFE+RCS)** | 14 | **2.51** | **4.66** | **0.31** | **1.41** |
| TaylorSeer | 11 | 2.87 | 5.85 | 0.73 | 3.53 |
| +RFE | 11 | 2.69 | 5.22 | 0.55 | 2.55 |
| +RCS | 11 | 2.77 | 5.21 | 0.62 | 3.09 |
| **RFC (RFE+RCS)** | 11 | **2.71** | **4.88** | **0.51** | **2.30** |

**RFE vs. Alternative Magnitude Estimation Strategies (NFC=14)**:

| Method | FID2FC↓ | sFID2FC↓ |
|--------|---------|----------|
| Linear (FasterCache) | 0.73 | 3.40 |
| w(t)=0.8 | 0.73 | 3.36 |
| w(t)=1.0 (TaylorSeer) | 0.57 | 2.77 |
| w(t)=1.2 | 0.52 | 2.51 |
| **RFE** | **0.43** | **2.02** |

### Key Findings
- RFC outperforms existing methods across all computational budgets, with **larger advantages under tighter budgets**: for example, at N=9, RFC achieves better sFID with 3.37 TFLOPs than TaylorSeer at N=6 with 4.76 TFLOPs.
- Improvements are particularly pronounced on video generation: RFC improves PSNR from 15.53 to 18.54 (+3 dB), reduces LPIPS from 0.245 to 0.133 (nearly halved), and raises VBench score from 79.52 to 80.83, approaching the full-computation score of 81.40.
- RFE and RCS are complementary: each individually outperforms TaylorSeer, and their combination yields further improvement.
- The relative standard deviation of the ratio $s_k(t-k)$ is only ~2%, validating the stability of the input-output relationship.
- Monitoring the input error of only the first module is sufficient for RCS scheduling, without the need to monitor all modules.

## Highlights & Insights
- **A concise yet profound core insight**: output changes are irregular but highly correlated with input changes → use inputs to estimate outputs. This observation is supported by both experiments and theory. Decomposing Taylor-based prediction into a **direction** component (handled by Taylor expansion) and a **magnitude** component (handled by the input-output relationship) constitutes an elegant and principled decoupling design.
- **Plug-and-play with zero training overhead**: RFC requires no training or fine-tuning whatsoever and can be directly applied to any pretrained DiT model. Obtaining input features requires only lightweight operations such as LayerNorm, making the additional computational cost negligible. This renders RFC highly practical for real-world deployment.
- **Clever proxy design in RCS dynamic scheduling**: using input prediction error as a proxy for output prediction error to trigger full computation elegantly sidesteps the circular dependency of "measuring output error requires full computation." Furthermore, monitoring only the first module suffices, incurring minimal overhead.

## Limitations & Future Work
- **Limited expressiveness of a scalar ratio**: $s_k(t-k)$ is a global scalar applied uniformly across all tokens and positions. In practice, the input-output relationship may vary across spatial locations, and a finer-grained (e.g., per-token or per-channel) ratio could further improve accuracy.
- **Manual tuning of threshold $\tau$**: The RCS threshold is a hyperparameter that must be adjusted per model and task. While the paper sets it by matching NFC to baselines, automatically selecting the optimal threshold in practice remains an open problem.
- **Generalization to non-Transformer architectures**: The theoretical analysis and experiments focus exclusively on DiT architectures; applicability to other diffusion model architectures such as U-Net has not been verified.
- **Limited gains from higher-order Taylor expansion**: Experimental results show only marginal improvement from $m=1$ to $m=2$, suggesting the bottleneck lies not in the Taylor expansion order but in the accuracy of magnitude estimation — an aspect where RFC already achieves substantial improvement.

## Related Work & Insights
- **vs. TaylorSeer**: TaylorSeer predicts feature changes via high-order Taylor expansion but relies entirely on temporal extrapolation, failing to adapt to the irregularity of output change magnitudes. RFC retains the Taylor-predicted **direction** while correcting the **magnitude** via the input-output relationship, comprehensively outperforming TaylorSeer under identical computational budgets.
- **vs. FORA**: FORA directly reuses cached features without any adjustment, leading to severe performance degradation at large cache intervals (FID rising from 2.32 to 12.63 at N=7). Through accurate prediction and dynamic scheduling, RFC achieves an FID of only 3.40 even at NFC=7.
- **vs. FasterCache/GOC**: Linear extrapolation methods use fixed or linearly varying scaling coefficients $w(t)$, which cannot adapt to the actual change magnitude at each timestep. RFC's $s_N(t)$ is dynamically computed from actual input variations and is therefore more expressive.
- **vs. TeaCache**: TeaCache also uses input features to trigger full computation but compares the current input against the cached input via distance, requiring an additional calibration step. RFC's RCS uses input **prediction error** rather than simple distance, making it better suited for forecasting-based methods, and requires no calibration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of leveraging input-output relationships to correct cached feature predictions is concise and effective; the direction-magnitude decoupling design is elegant, with solid theoretical and empirical support.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers three tasks — class-conditional generation (DiT-XL/2), text-to-image (FLUX.1 dev), and text-to-video (HunyuanVideo) — with multiple metrics, comprehensive ablation studies, and thorough comparison against prior methods.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is clearly structured and analytically rigorous, with a logical progression from observations to theory to method to experiments. Figures and tables effectively illustrate the core findings.
- **Value**: ⭐⭐⭐⭐ As a training-free plug-and-play acceleration method, RFC achieves approximately 5–6× computational savings while preserving generation quality, offering significant practical value for DiT deployment. Improvements are especially notable in video generation scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)
- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](../../NeurIPS2025/time_series/diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [\[NeurIPS 2025\] Diffusion Transformers as Open-World Spatiotemporal Foundation Models](../../NeurIPS2025/time_series/diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)
- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[ICLR 2026\] Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models](dissecting_chronos_sparse_autoencoders_reveal_causal_feature_hierarchies_in_time.md)

<!-- RELATED:END -->
