---
title: >-
  [Paper Note] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training
description: >-
  [ICLR 2026][Image Restoration][scale anchoring] This paper defines the novel problem of "Scale Anchoring" (SA)—wherein training on low-resolution data causes inference errors to remain anchored at training-resolution lev…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "scale anchoring"
  - "frequency representation"
  - "zero-shot super-resolution"
  - "spatiotemporal forecasting"
  - "Nyquist frequency"
date: 2026-05-08
content_hash: 1c246b4654d91d66
---

# Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training

**Conference**: ICLR 2026
**arXiv**: [2512.05132](https://arxiv.org/abs/2512.05132)  
**Code**: To be confirmed  
**Area**: Object Detection (Spatiotemporal Forecasting / Zero-Shot Super-Resolution)
**Keywords**: scale anchoring, frequency representation, zero-shot super-resolution, spatiotemporal forecasting, Nyquist frequency

## TL;DR

This paper defines the novel problem of "Scale Anchoring" (SA)—wherein training on low-resolution data causes inference errors to remain anchored at training-resolution levels during high-resolution inference—and proposes an architecture-agnostic Frequency Representation Learning (FRL) method. By introducing Nyquist-normalized frequency encodings, FRL enables errors to decrease as resolution increases, with effectiveness validated across 8 mainstream architectures.

## Background & Motivation

**Zero-Shot Super-Resolution Spatiotemporal Forecasting (ZS-SR STF)**: Training deep learning models on low-resolution data and performing inference at high resolution, motivated by the prohibitive cost of high-resolution DNS simulation data for training.

**Prevailing Misconception**: Existing methods consider a cross-resolution error ratio (RMSE_Ratio) close to 1 as "successful generalization." However, for a $p$-th order numerical solver, an $\alpha$-fold resolution increase should reduce the error by $\alpha^p$.

**Root Cause of Scale Anchoring**: The Nyquist frequency of low-resolution training data imposes an upper bound on the physical frequency content a model can learn. During high-resolution inference, the model cannot process high-frequency components unseen during training, causing errors to remain anchored at the training resolution.

**Distinction from Known Phenomena**:
- **vs. Spectral Bias (SB)**: SB concerns the preference for learning low-to-high frequencies within the training frequency band; SA is an information-theoretic constraint *outside* the training frequency band.
- **vs. Discretization Mismatch Error (DME)**: DME arises from architectural/optimization choices and can be mitigated by design; SA is a hard constraint imposed by the Shannon–Nyquist sampling theorem.

**Key Theoretical Results**:
- **Theorem 1 (Frequency Blind Zone)**: The frequency response of a network trained at resolution $\rho_0$ is undefined or erroneous for $\omega > \rho_0/2$.
- **Theorem 2 (High-Frequency Error Dominance)**: High-resolution inference error is dominated by frequency components in $[\rho_0/2, \rho/2]$.

## Method

### Three-Step FRL (Architecture-Agnostic)

1. **Multi-Resolution Data Construction**: Downsampling the original data to generate multiple resolution versions $\rho_j = \rho_0/2^j$ (standard multi-scale training technique).
2. **Nyquist-Normalized Frequency Representation (Core Contribution)**: $PE_{freq}(x, k, \rho) = \sin(2\pi k \cdot x / k_{Nyq}(\rho))$, which ensures that the same physical frequency produces identical representation values across different resolutions. This decouples frequency from resolution and constitutes the paper's primary methodological novelty.
3. **Frequency-Aware Training Loss**: Standard loss augmented with a frequency-domain consistency loss $\lambda \cdot \|F_\Theta - \hat{u}\|^2_{freq}$, ensuring cross-scale spectral consistency (standard spectral regularization technique).

### Loss & Training

Steps 1 and 3 follow established practices; Step 2—Nyquist frequency alignment—is the sole methodological innovation. Ablation studies confirm that Step 2 is a necessary condition for breaking Scale Anchoring.

## Key Experimental Results

### 3D Fluid Simulation (Trained at 32³, Tested up to 129³)

| Method | RMSE_Ratio: Baseline → +FRL | High-Res RMSE Reduction |
|--------|----------------------------|------------------------|
| GNN | 1.018 → **0.175** | 5.82× |
| CNN | 1.060 → **0.137** | 7.74× |
| NO | 1.017 → **0.135** | 7.53× |
| Transformer | 1.021 → **0.165** | 6.19× |
| Diffusion | 1.041 → **0.181** | 5.75× |

### ERA5 Weather Forecasting (Trained at 180×90, Inferred up to 1440×721)

| Architecture | RMSE_Ratio → +FRL | ACC Gain |
|--------------|-------------------|---------|
| Transformer | 1.053 → 0.708 | 0.44 → 0.65 |
| CNN | 1.066 → 0.662 | 0.42 → 0.68 |

### Ablation Study

| Configuration | RMSE_Ratio | Note |
|---------------|-----------|------|
| Step 1 only (multi-resolution training) | ~1.0 | Does not break SA |
| Step 3 only (frequency loss) | ~1.0 | Does not break SA |
| **Steps 1+2+3 (full FRL)** | **0.135–0.181** | Successfully breaks SA |
| Step 2 only | 0.3–0.4 | Effective but insufficient |

### Key Findings

- Scale Anchoring is present in all 8 mainstream architectures (GNN / Transformer / CNN / Diffusion / NO / Neural ODE / Mamba / NN).
- FRL extends the effective frequency response bandwidth from the vicinity of the training Nyquist frequency to the full frequency range.
- FRL degrades in extreme regimes such as high-Reynolds-number turbulence, where spectral relationships are not smooth.

## Highlights & Insights

- **High Value of Problem Formulation**: SA is rigorously distinguished from SB and DME for the first time, supported by complete theoretical analysis.
- **Architecture Agnosticism**: Demonstrated effectiveness across 8 representative architectures, indicating that SA is a universal limitation of data-driven models.
- **Honest Discussion of Limitations**: The authors explicitly describe failure modes and suggest incorporating Kolmogorov spectral constraints as future work.
- **Cross-Domain Validation**: Verified in two physically distinct domains—fluid simulation and weather forecasting.

## Limitations & Future Work

- Strict convergence order is not guaranteed, as deep learning models are fundamentally not equivalent to numerical solvers.
- In high-Reynolds-number turbulence and discontinuous problems, non-smooth spectral relationships degrade FRL's extrapolation capability.
- Multi-resolution training introduces additional computational and memory overhead (approximately 2–3×).
- Validation is limited to spatiotemporal forecasting tasks; applicability to domains such as image super-resolution remains unexplored.

## Related Work & Insights

- **FNO/PINO**: Improve resolution generalization but do not explicitly address Nyquist frequency constraints.
- **Anti-Spectral-Bias Methods**: Target high-frequency learning within the training band; SA concerns frequencies outside the training band.
- **Multi-Resolution Training / Spectral Regularization**: FRL's Steps 1 and 3 draw on these established techniques.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Novel problem formulation + rigorous theoretical analysis + empirical validation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 architectures × 2 physical domains + comprehensive ablation
- Writing Quality: ⭐⭐⭐⭐ Clear structure with honest discussion of failure modes
- Value: ⭐⭐⭐⭐ Significant implications for the scientific machine learning community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](../../CVPR2026/image_restoration/fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](../../ICCV2025/image_restoration/outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration](../../ICCV2025/image_restoration/foundir_unleashing_million-scale_training_data_to_advance_foundation_models_for_.md)

</div>

<!-- RELATED:END -->
