---
title: >-
  [Paper Note] Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting
description: >-
  [ICML 2026][Physics & Scientific Computing][Precipitation Nowcasting] SDIR reformulates 0–2 hour radar precipitation nowcasting as a "frequency-decoupled iterative refinement" process. It first extracts a stable low-freq…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Precipitation Nowcasting"
  - "Spectral Decoupling"
  - "Iterative Refinement"
  - "Fourier Neural Operator"
  - "Power Spectral Density Loss"
date: 2026-05-08
content_hash: 7d0ad5bfb7c4fe2f
---

# Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting

**Conference**: ICML 2026  
**arXiv**: [2606.02661](https://arxiv.org/abs/2606.02661)  
**Code**: https://github.com/RuntimeWarning/SDIR  
**Area**: Scientific Computing / Weather Forecasting / Spatiotemporal Prediction  
**Keywords**: Precipitation Nowcasting, Spectral Decoupling, Iterative Refinement, Fourier Neural Operator, Power Spectral Density Loss

## TL;DR
SDIR reformulates 0–2 hour radar precipitation nowcasting as a "frequency-decoupled iterative refinement" process. It first extracts a stable low-frequency synoptic skeleton using SFG-Former, then progressively synthesizes high-frequency convective details via FR-Refiner (incorporating Fourier Neural Operators). A Phase-Consistent Power Spectral Density (PCPSD) loss, aligned with Kolmogorov's turbulence power law, replaces pure MSE to prevent over-smoothing. The model significantly outperforms both regression-based and diffusion-based SOTA on CIKM, Shanghai, and SEVIR benchmarks.

## Background & Motivation
**Background**: Precipitation nowcasting is critical for urban flood control, aviation, and disaster prevention. Early methods relied on optical flow for radar echo extrapolation, followed by spatiotemporal models like ConvLSTM, PredRNN, PhyDNet, Earthformer, and SimVP, which improved accuracy through supervised regression. Recent trends utilize GANs (DGMR, NowcastNet) and diffusion models (PreDiff, CasCast, DiffCast) to pursue visual realism in high-frequency details through generative modeling.

**Limitations of Prior Work**: Both existing approaches face structural issues:
- Regression models, driven by pixel-wise losses like MSE, "average out" spatial uncertainty. This leads to rapid power spectrum decay at high frequencies, blurred convective cores, and suppressed peak intensities, violating the Kolmogorov power law of atmospheric turbulence.
- Diffusion models recover high frequencies by sampling from Gaussian noise. While visually sharp, the generated convective cells often appear at incorrect locations or with exaggerated intensities, described by the authors as "unanchored hallucinations"—visually plausible but physically groundless.

**Key Challenge**: Real-world precipitation evolution is inherently multi-scale and progressive—small-scale convection grows upon a large-scale synoptic skeleton acting as a boundary condition. Single-step or single-branch models fail to balance "global stability (no drift)" with "local sharpness (no blur)," resulting in either over-smoothing or hallucinations.

**Goal**: Construct a **deterministic** framework that decomposes forecasting into frequency bands. It aims to stabilize the low-frequency skeleton first and then synthesize high-frequency details through iterative steps constrained by physical spectra, ensuring the entire power spectrum conforms to turbulence statistics.

**Key Insight**: Embed the concept of "gradually revealing higher frequency bands" into the model architecture. During training, DCT truncation provides low-frequency conditions $C_s$ of varying bandwidths, teaching the model to synthesize the next frequency layer given a skeleton. During inference, this is combined into an iterative schedule from $s=0 \to s=W-1$, serving as a deterministic, frequency-conditional "diffusion process" targeting spectral depth rather than noise levels.

**Core Idea**: Use frequency bands instead of noise levels as iterative variables, combined with global operators in the Fourier domain (SFNO) and explicit turbulence constraints via PCPSD loss to achieve "neither blur nor hallucination."

## Method
SDIR is an end-to-end dual-branch network: SFG-Former outputs the baseline skeleton $\hat Y_{base}$, while FR-Refiner outputs the high-frequency residual $\hat Y_{res}$. The final prediction is $\hat Y = \hat Y_{base} + \hat Y_{res}$. All branches are modulated by a frequency scale signal $s \in \{0, 1, \dots, W-1\}$, which is randomly sampled during training and follows an increasing schedule during inference.

### Overall Architecture
**Input**: Historical radar echo sequences $X \in \mathbb{R}^{B \times T_{in} \times C \times H \times W}$. During training, a low-frequency condition $C_s = \operatorname{IDCT}(\operatorname{Trunc}_{s \times s}(\operatorname{DCT}(Y)))$ is derived from the ground truth $Y$ via DCT truncation.
**Output**: Precipitation fields $\hat Y$ for future $T_{out}$ frames.
**Mechanism**: (i) A frequency scale signal $s \sim \operatorname{Beta}(1,3)$ determines the spectral depth for the current training step; (ii) SFG-Former uses a Scale-Adaptive Transformer (SAT) with 3D RoPE, fusing $X$ and $C_s$ to provide the skeleton $\hat Y_{base}$; (iii) FR-Refiner is a U-Net-shaped Fourier residual generator with SFNO blocks in the bottleneck, modulated by $s$ via Adaptive Normalization to output $\hat Y_{res}$; (iv) The system is optimized by reconstruction L1 loss and a dynamically weighted PCPSD spectral loss; (v) Inference follows a schedule $\mathcal{S} = \{s_1=0, s_2, \dots, s_K\}$, where each step uses the DCT-truncated output of the previous step as $C_s$.

### Key Designs

1.  **Spectral-Decoupled Training Curriculum: DCT Truncation $C_s$ + Beta(1,3) Sampling**:
    - **Function**: Injects "spectral depth" as a controllable variable, covering both "synthesis from scratch" and "high-frequency completion," biased towards learning low frequencies first.
    - **Mechanism**: For each batch, $\sigma \sim \operatorname{Beta}(1,3)$ and $s = \lfloor W\sigma \rfloor$ are sampled. A 2D DCT is performed on the ground truth, coefficients outside the top-left $s \times s$ block are zeroed, and an IDCT produces the ideal low-pass condition $C_s$. When $s=0$, $C_s=0$ (cold start); when $s=W-1$, $C_s \approx Y$ (final detail enhancement). The Beta(1,3) distribution prioritizes large-scale skeletons.
    - **Design Motivation**: This is a deterministic frequency-domain counterpart to diffusion models. While diffusion slices by noise levels, this method slices by spatial frequency, which has explicit physical meaning (synoptic vs. convective scales) and avoids stochastic sampling.

2.  **SFG-Former + 3D RoPE: Frequency-Adaptive Global Skeleton**:
    - **Function**: Processes historical sequences and low-frequency conditions to output spatio-temporally consistent base predictions $\hat Y_{base}$.
    - **Mechanism**: $X$ and $C_s$ are concatenated along the time dimension, patchified, and projected to $z \in \mathbb{R}^{B \times L \times D}$. Each SAT block contains a Frequency Scale Embedder (FSE) that maps $s$ to modulation parameters $(\gamma, \beta, \alpha)$. Features are modulated as $z_{mod} = (1 + \gamma) \odot \operatorname{LN}(z) + \beta$, with a gated residual $z_{out} = z + \alpha \odot \operatorname{Transformer}(z_{mod})$. 3D RoPE maintains translation invariance across space and time.
    - **Design Motivation**: Transformers naturally bias toward medium-resolution features. FSE injects the "spectral depth for reconstruction" into every layer, allowing the skeleton branch to output stable low-$s$ predictions or sharper high-$s$ versions without competing with the refiner. 3D RoPE enhances robustness against spatial translation and temporal drift.

3.  **FR-Refiner + SFNO + PCPSD Loss: High-Frequency Synthesis and Turbulence Constraints**:
    - **Function**: Synthesizes high-frequency residuals $\hat Y_{res}$ using Fourier-domain global operators and enforces Kolmogorov's power law via explicit PCPSD supervision.
    - **Mechanism**: The refiner uses a U-Net topology with PixelUnshuffle/PixelShuffle for detail-preserving resolution changes. The bottleneck stacks SFNO blocks—FFT transforms features to the frequency domain for linear transformation and SoftShrink sparsification before IFFT. PCPSD loss is calculated by applying a 2D Hann window to suppress edge artifacts, performing rFFT to get the 2D power spectrum $P(k_y, k_x)$, and computing a radial bin average for the 1D isotropic power spectrum $S(k)$. The loss compares log-spectra: $\mathcal{L}_{pcpsd} = \frac{\sum_k \Omega(k,s)(\log S_{pred}(k) - \log S_{gt}(k))^2}{\sum_k \Omega(k,s)}$. Dynamic weights $\Omega(k,s) = (k + \epsilon)^\gamma \cdot \{0.2 \text{ if } k \le k_s(s); 1.0 \text{ otherwise}\}$ emphasize unlocked high frequencies.
    - **Design Motivation**: Spatial convolutions are local and inevitably "wash out" high frequencies. SFNO captures cross-scale coupling in the Fourier domain. PCPSD addresses the root cause of over-smoothing: since MSE gradients are dominated by low frequencies, the model favors smooth solutions. Log-spectral distance forces energy distribution to match the GT spectrum.

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_{base} + \mathcal{L}_{res} + \phi(s)\mathcal{L}_{pcpsd}$, where the first two terms are L1 losses for the base and residual, and $\phi(s) = \eta(s/W)^2$ with $\eta=0.01$. Optimized with AdamW, initial $lr = 3 \times 10^{-4}$ on 4×RTX 4090D. SFG-Former uses 8 SAT blocks; FR-Refiner uses an SFNO bottleneck with 8 blocks.

## Key Experimental Results

### Main Results
Comparison staged on CIKM, Shanghai, and SEVIR datasets against ConvLSTM, PredRNN, PhyDNet, SimVP, Earthformer, MIMO, DiffCast, and AlphaPre.

**Table 1 — CIKM Dataset (AVG of HSS / CSI / SSIM ↑, MAE ↓)**

| Model | HSS AVG | CSI AVG | SSIM | MAE |
|---|---|---|---|---|
| ConvLSTM | 0.3142 | 0.2615 | 0.4860 | 738.05 |
| PredRNN | 0.3737 | 0.3359 | 0.5157 | 784.84 |
| PhyDNet | 0.4128 | 0.3563 | 0.4306 | 694.99 |
| Earthformer | 0.4159 | 0.3544 | 0.4903 | 674.99 |
| DiffCast | 0.4071 | 0.3477 | 0.4710 | 669.01 |
| AlphaPre | 0.3633 | 0.3092 | 0.4775 | 661.40 |
| **SDIR (Ours)** | **0.4724** | **0.4043** | **0.5574** | **600.37** |

**Table 2 — Shanghai and SEVIR Datasets (AVG, SEVIR in parentheses)**

| Model | Shanghai HSS / CSI / SSIM / MAE | SEVIR HSS / CSI / SSIM / MAE |
|---|---|---|
| ConvLSTM | 0.3602 / 0.2611 / 0.7438 / 1846.2 | 0.3512 / 0.2715 / 0.6062 / 2896.9 |
| PhyDNet | 0.5203 / 0.3892 / 0.8133 / 1386.0 | 0.4172 / 0.3311 / 0.7063 / 2103.3 |
| Earthformer | 0.5015 / 0.3711 / 0.7643 / 1395.8 | 0.4066 / 0.3230 / 0.6706 / 2241.8 |
| DiffCast | 0.4920 / 0.3628 / 0.8080 / 1450.1 | 0.3972 / 0.3057 / 0.6690 / 2595.5 |
| AlphaPre | 0.4276 / 0.3145 / 0.7534 / 1445.3 | 0.4052 / 0.3193 / 0.6100 / 2463.0 |
| **SDIR (Ours)** | **0.5882 / 0.4497 / 0.8548 / 1129.1** | **0.4401 / 0.3499 / 0.7544 / 1897.9** |

SDIR achieves a +13.6% HSS AVG and +14.1% CSI AVG gain on CIKM, consistently ranking first across all metrics and thresholds.

### Ablation Study
**Table 4 — Module Ablation (Shanghai)**: S-I = SFG-Former, S-II = FR-Refiner.

| Exp | S-I | S-II | PCPSD | HSS | CSI | SSIM | MAE |
|---|---|---|---|---|---|---|---|
| (a) | ✓ | | | 0.3529 | 0.2559 | 0.8478 | 1248.8 |
| (b) | | ✓ | | 0.4614 | 0.3266 | 0.8125 | 1586.1 |
| (c) | ✓ | ✓ | | 0.5367 | 0.4057 | 0.8512 | 1138.3 |
| Ours | ✓ | ✓ | ✓ | **0.5882** | **0.4497** | **0.8548** | **1129.1** |

**Table 5–7 — Training and Inference Configurations (Shanghai)**

| Config | HSS | CSI | SSIM | MAE | Remarks |
|---|---|---|---|---|---|
| L1 / MAE Only | 0.4509 | 0.3420 | 0.8555 | 1106.4 | Lowest MAE but HSS/CSI drop |
| Uniform $s$ Sampling | 0.2842 | 0.2097 | 0.8458 | 1284.1 | Curriculum collapse |
| w/o Adaptive Norm | 0.5073 | 0.3725 | 0.8102 | 1476.4 | Scale signal failure |
| Inference 1 Step | 0.5584 | 0.4243 | 0.8522 | 1111.0 | 0.30s, under-refined |
| Inference 8 Steps | 0.5882 | 0.4497 | 0.8548 | 1129.1 | 1.17s, optimal balance |
| Beta(1.0, 3.0) | **0.5882** | **0.4497** | **0.8548** | **1129.1** | Best distribution |

### Key Findings
- PCPSD is the most critical component: removing it drops HSS from 0.5882 to 0.5367 and CSI from 0.4497 to 0.4057, proving that pure spatial losses still result in smooth predictions.
- The dual-branch structure is indispensable: SFG-Former alone lacks high frequencies (CSI 0.2559), while FR-Refiner alone lacks a stable skeleton (MAE 1586.1).
- Long lead-time advantage: SDIR's performance gap over baselines widens significantly in the 60–120 min range, demonstrating better error accumulation control through spectral decoupling.
- Inference steps exhibit a "sweet spot": 8 steps offer the best balance, whereas 32 steps introduce noise and degrade SSIM/MAE.

## Highlights & Insights
- Progressive learning via frequency bands represents a "clean" paradigm shift: noise scales are replaced by physically meaningful spatial frequencies, retaining iterative refinement without hallucinations.
- PCPSD transforms the Kolmogorov turbulence law into a differentiable, dynamically weighted signal. This provides a template for tasks suffering from high-frequency collapse (SR, fluid simulation).
- SFNO in the bottleneck allows the U-Net to capture global cross-scale coupling with low computational overhead, effectively adapting FourCastNet principles to nowcasting.
- Frequency scale modulation enables an interpretable speed-quality curve, allowing deployment to switch between fast/stable skeletons and high-quality detailed forecasts based on operational needs.

## Limitations & Future Work
- Inference latency (approx. 1.17s for 8 steps) remains higher than single-step models (0.3s), suggesting a need for distillation or one-step refinement techniques.
- Schedule $\mathcal{S}$ and Beta parameters are manually selected; they do not yet adapt to specific regions, seasons, or weather modes (e.g., typhoons vs. convective storms).
- PCPSD assumes isotropic turbulence. For anisotropic scenarios like frontal rain, radial averaging might lose directional information.
- The framework is currently tested on 2D radar fields; extension to 3D vertical profiles or multi-modal data (satellite + NWP) remains unexplored.

## Related Work & Insights
- **vs. DiffCast (CVPR'24)**: DiffCast uses a deterministic backbone plus a diffusion residual, which still suffers from hallucinations. SDIR replaces the residual path with a deterministic, frequency-conditional Fourier refiner.
- **vs. PreDiff / CasCast**: Pure latent diffusion sampling is slow and stochastic; SDIR requires only 8 deterministic steps.
- **vs. Earthformer**: While Earthformer uses cuboid attention for global structure, it lacks high-frequency refinement. SDIR integrates global structural capacity into SFG-Former while recovering details via PCPSD.
- **vs. NowcastNet**: NowcastNet uses physics-conditional GANs. SDIR achieves competitive visual sharpness and physical consistency without the instability of adversarial training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Frequency decoupling + Fourier refiner + PCPSD is a precise solution to nowcasting pain points.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three public datasets, eight SOTA baselines, and extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear physical motivation (turbulence energy cascade) and well-structured methodology.
- **Value**: ⭐⭐⭐⭐ Provides a deterministic, physically consistent, and high-resolution route for operational nowcasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation](iterative_refinement_neural_operators_are_learned_fixed-point_solvers_a_principl.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/physics/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)
- [\[ICML 2026\] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)
- [\[ICML 2026\] Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs](hermite-ngp_gradient-augmented_hash_encoding_for_learning_pdes.md)

</div>

<!-- RELATED:END -->
