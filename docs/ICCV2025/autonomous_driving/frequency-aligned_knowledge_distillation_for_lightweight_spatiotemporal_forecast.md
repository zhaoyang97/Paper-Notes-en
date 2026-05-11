---
title: >-
  [Paper Note] SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting
description: >-
  [ICCV 2025][Autonomous Driving][Spatiotemporal Forecasting] This paper proposes SDKD (Spectral Decoupled Knowledge Distillation), a framework that leverages a frequency-aware teacher model and a frequency-aligned distill…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Spatiotemporal Forecasting"
  - "Knowledge Distillation"
  - "Frequency Decoupling"
  - "Lightweight Models"
  - "CNN-Transformer Hybrid Architecture"
date: 2026-05-08
content_hash: 1fd1fd4fd9049b4f
---

# SDKD: Frequency-Aligned Knowledge Distillation for Lightweight Spatiotemporal Forecasting

**Conference**: ICCV 2025
**arXiv**: [2507.02939](https://arxiv.org/abs/2507.02939)
**Code**: [https://github.com/itsnotacie/SDKD](https://github.com/itsnotacie/SDKD)
**Area**: Autonomous Driving
**Keywords**: Spatiotemporal Forecasting, Knowledge Distillation, Frequency Decoupling, Lightweight Models, CNN-Transformer Hybrid Architecture

## TL;DR

This paper proposes SDKD (Spectral Decoupled Knowledge Distillation), a framework that leverages a frequency-aware teacher model and a frequency-aligned distillation strategy to transfer multi-scale spectral knowledge from complex spatiotemporal forecasting models to lightweight student networks, achieving up to 81.3% MSE reduction on the Navier-Stokes dataset.

## Background & Motivation

- **Background**: Spatiotemporal forecasting (e.g., traffic flow, weather evolution, fluid dynamics) is a core task in smart cities and climate science. These tasks require simultaneously modeling the complex coupling between **high-frequency local patterns** (e.g., sudden traffic congestion, turbulent vortices) and **low-frequency global dynamics** (e.g., slowly varying atmospheric pressure, daily traffic cycles). CNN-Transformer hybrid architectures have achieved state-of-the-art accuracy by combining local convolution with global attention mechanisms.
- **Limitations of Prior Work**: These models face severe deployment challenges: **the self-attention mechanism in Transformers has complexity $O(N^2d)$**, and combined with deep CNN stacking, the computational cost and memory consumption are prohibitively high for edge deployment. While knowledge distillation (KD) is a natural solution, existing KD frameworks are primarily designed for classification tasks and perform poorly when directly applied to spatiotemporal forecasting.
- **Key Challenge**: Spatiotemporal signals exhibit **multi-band spectral characteristics**. Simple feature mimicry fails to maintain the critical balance between high-frequency details and low-frequency trends, leading to a **spectral entanglement dilemma**: without an explicit mechanism to decouple and transfer knowledge at different frequency bands, methods either over-smooth high-frequency variations (e.g., traffic surge peaks are erased) or fail to capture slowly evolving trends (e.g., seasonal climate changes are ignored). Moreover, the optimal frequency bands of spatiotemporal signals are non-stationary in both space and time.
- **Goal**: The paper introduces SDKD, exploiting the natural properties of CNN as a high-pass filter and Transformer as a low-pass filter to achieve frequency-domain decoupling in the teacher's latent space, and using these decoupled spectral representations as distillation supervision signals.
- **Core Idea**: Spatiotemporal signals exhibit inherent **spectral duality**—high-frequency components (rapid local variations) are governed by spatial locality, while low-frequency components (slow global dynamics) are dominated by temporal continuity.

## Method

### Overall Architecture

SDKD consists of four stages: (1) teacher model pre-training with a frequency-decoupled architecture (CNN for high frequencies + Transformer for low frequencies); (2) lightweight student architecture design (ResNet/U-Net/MLP-Mixer); (3) offline frequency-aligned distillation training; (4) online inference deployment.

### Key Designs

#### 1. Frequency-Decoupled Teacher Model (ST-AlterNet)

- **Function**: Explicitly separates high-frequency and low-frequency spatiotemporal patterns in the latent space, providing structured spectral priors for distillation.
- **Mechanism**: Adopts an encoder–latent evolution–decoder architecture. The encoder maps input $X$ to latent space $Z = \mathcal{E}(X)$ via multi-layer convolutions. The latent evolution module contains two parallel frequency-sensitive branches:
    - **High-Frequency Extractor (CNN branch)**: $Z^h = \text{ConvBlock}(Z) = \sigma(\text{Conv2D}(Z; \mathbf{W}_h) + b_h)$. CNN acts as an implicit high-pass filter via local gradient operators $\nabla_{x,y}$, capturing abrupt changes and fine-grained spatial patterns.
    - **Low-Frequency Modeler (Transformer branch)**: $Z^l = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$. The self-attention mechanism acts as a low-pass filter: $\mathcal{F}_{\text{low}}(Z)(\omega) = \frac{1}{1+\|\omega\|^2/\lambda} \cdot \hat{Z}(\omega)$, capturing global long-range dependencies and trend dynamics.
    - High-low frequency fusion: $Z_{\text{evolved}} = \text{LayerNorm}(Z^h + Z^l)$.
- **Design Motivation**: Based on the theoretical analysis of Park et al., convolutional layers are naturally biased toward high frequencies while Transformers are biased toward low frequencies. The alternating design aligns the spectral energy distribution of the latent space with physical principles (experiments verify that the latent space conforms to the Kolmogorov energy spectrum $E(\omega) \propto \omega^{-5/3}$), providing interpretable spectral priors for distillation.

#### 2. Frequency-Aligned Distillation Mechanism

- **Function**: Extracts multi-scale spectral features from the teacher's latent space to guide the student model in simultaneously learning high-frequency details and low-frequency trends.
- **Mechanism**: Defines a spectral transport loss:
  $\mathcal{L}_{\text{KD}} = \|\Psi_h(\mathcal{G}(X)) - \Psi_h(\mathcal{F}(X))\|_2^2 + \alpha \|\Psi_l(\mathcal{G}(X)) - \Psi_l(\mathcal{F}(X))\|_2^2$
  
  where $\Psi_h$ and $\Psi_l$ extract high- and low-frequency features respectively, and $\alpha$ balances the two. The overall training objective is:
  $\min_{\theta_\mathcal{G}} \|\mathcal{G}(X) - Y\|_2^2 + \lambda \mathcal{L}_{\text{KD}}$
- **Design Motivation**: This is an **architecture-agnostic** frequency-domain alignment—independent of the student's specific structure, directly aligning output features in the spectral domain. Spectrally sensitive adaptive weighting is also employed to mitigate the student model's inherent bias toward high-frequency overfitting and low-frequency underfitting.

#### 3. Multi-Teacher Distillation (CAMKD)

- **Function**: Leverages complementary knowledge from multiple teacher models, adaptively weighting contributions via multi-objective optimization in gradient space.
- **Mechanism**: For each teacher $\mathcal{F}_m$, a distillation loss $\ell_m$ is defined. In each mini-batch, the "Agree to Disagree" (A2D) method solves for optimal weights $\{\alpha_m^*\}$:
  $\min_{\{\alpha_m\}} \frac{1}{2}\|\sum_{m=1}^M \alpha_m \nabla_\theta \ell_m\|^2, \quad \text{s.t.} \sum \alpha_m = 1, 0 \leq \alpha_m \leq C$
  
  Student update direction: $d = -\sum_m \alpha_m^* \nabla_\theta \ell_m$.
- **Design Motivation**: Different teachers excel at different frequency bands and scenarios. Simple averaging may discard information when teacher gradients conflict. A2D automatically down-weights conflicting teachers via gradient-space optimization, fully exploiting complementarity.

### Loss & Training

- Prediction task loss + frequency-aligned distillation loss, balanced by $\lambda$.
- Adam optimizer with a fixed learning rate of $1 \times 10^{-4}$, trained for 300 epochs with early stopping based on the validation set.
- The student's latent dimension is only 20%–30% of the teacher's, with up to 5× total parameter reduction.

## Key Experimental Results

### Main Results — Weatherbench and TaxiBJ+

| Method | Weatherbench MAE↓ | Weatherbench PSNR↑ | Weatherbench SSIM↑ | TaxiBJ+ MSE↓ |
|------|-------------------|---------------------|---------------------|---------------|
| T1: ST-AlterNet (Teacher) | 0.7287 | 33.32 | 0.9493 | 0.0683 |
| T2: SimVP (Teacher) | 0.7475 | 32.11 | 0.9331 | 0.0697 |
| U-Net (Student baseline) | 0.9822 | 29.37 | 0.8635 | 0.0831 |
| U-Net + AEKD (MSE Loss) | 0.8457 | 31.69 | 0.8849 | 0.0715 |
| U-Net + AVER-MKD (AB Loss) | **0.8541** | **31.45** | **0.8812** | 0.0728 |
| ResNet + AEKD (MSE Loss) | 0.9102 | 30.82 | 0.8801 | 0.0784 |
| MLP-Mixer + AEKD (MSE Loss) | 1.1428 | 25.32 | 0.7893 | 0.0901 |

After distillation, U-Net's MAE decreases from 0.9822 to 0.8457 (↓13.9%) and PSNR improves from 29.37 to 31.69 (↑7.9%), significantly closing the gap with the teacher.

### Ablation Study — Distillation Method Comparison (NS Dataset + U-Net Student)

| Method | MSE↓ | MAE↓ | SSIM↑ | Overall |
|------|------|------|-------|------|
| Baseline (S) | 0.141 | 0.239 | 0.658 | — |
| AVER-MKD | 0.136 | 0.233 | 0.670 | ✓ |
| AEKD | 0.135 | 0.235 | 0.663 | ✓ |
| **CAMKD (Multi-teacher frequency-aligned)** | **0.136** | **0.234** | **0.667** | **✓** |

On the RBC dataset, CAMKD achieves the best results with MSE=0.0261, MAE=0.112, and SSIM=0.728, outperforming all single-teacher and naive multi-teacher methods.

### Key Findings

- **Spectral analysis confirms theory**: The teacher model ST-AlterNet exhibits low prediction error in both high- and low-frequency regions, validating the frequency decoupling design. The undistilled U-Net shows particularly large errors in high-frequency regions (over-smoothing caused by the convolutional structure); after distillation, errors in both frequency regions decrease significantly.
- **Inference acceleration**: On the NS dataset, the distilled U-Net achieves 2.28× faster inference than the teacher (0.0050s vs. 0.0115s).
- **Physical interpretability**: The spectral distribution of the teacher's latent space is consistent with the Kolmogorov energy spectrum $E(\omega) \propto \omega^{-5/3}$ in fluid mechanics, where high frequencies correspond to the viscous dissipation term and low frequencies correspond to the advection term of the Navier-Stokes equations.

## Highlights & Insights

- **Reframing spatiotemporal distillation from a spectral perspective**—moving beyond conventional "simple feature mimicry" and explicitly identifying that high-frequency and low-frequency knowledge must be transferred separately.
- **Strong theoretical grounding**: Neural Tangent Kernel theory and Plancherel's theorem are employed to prove the high-pass/low-pass properties of CNN/Transformer and the validity of frequency-domain error decomposition.
- **Architecture-agnostic**: The student can be any lightweight architecture (U-Net, ResNet, MLP-Mixer) without requiring structural alignment with the teacher.

## Limitations & Future Work

- Frequency decoupling relies on prior assumptions about CNN and Transformer architectures; applicability is unclear when the teacher uses other architectures (e.g., FNO or LSTM).
- The four benchmarks are relatively small-scale (at most 35K training samples); performance on larger-scale real-world data remains to be verified.
- Distillation hyperparameters $\alpha$ and $\lambda$ require manual tuning, lacking an adaptive mechanism.
- The optimal frequency partition for spatiotemporally non-stationary signals should vary with spatiotemporal position, whereas the current method uses a fixed cutoff frequency.

## Related Work & Insights

- Methodologically related to FitNet-R (distillation for regression) and FOLK (frequency-masked distillation for classification), but SDKD is the first frequency-decoupled distillation framework specifically designed for spatiotemporal signals.
- Complementary to FNO (Fourier Neural Operator): FNO explicitly models PDEs in the frequency domain, while SDKD achieves more flexible decoupling via the implicit spectral properties of CNN/Transformer.
- Open question: Can frequency-decoupled distillation be extended to other spatiotemporal tasks such as video prediction and point cloud sequence forecasting?

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ACAM-KD: Adaptive and Cooperative Attention Masking for Knowledge Distillation](acam-kd_adaptive_and_cooperative_attention_masking_for_knowledge_distillation.md)
- [\[ICCV 2025\] Occupancy Learning with Spatiotemporal Memory](occupancy_learning_with_spatiotemporal_memory.md)
- [\[ICCV 2025\] Passing the Driving Knowledge Test](passing_the_driving_knowledge_test.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](future-aware_interaction_network_for_motion_forecasting.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)

</div>

<!-- RELATED:END -->
