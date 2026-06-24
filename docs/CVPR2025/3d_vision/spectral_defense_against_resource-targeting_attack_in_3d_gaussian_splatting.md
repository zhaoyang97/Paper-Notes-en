---
title: >-
  [Paper Note] Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3DGS Security] This paper proposes a spectral defense mechanism against resource exhaustion attacks on 3DGS. It jointly defends in the 3D spectral domain (by linking Gaussian covariance with high-frequency response through Fourier analysis to prune anomalous high-frequency splats) and the 2D spectral domain (penalizing anisotropic angular energy distribution using an entropy loss). It suppresses Gaussian overgrowth by 5.92×, reduces memory by 3.66×…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3DGS Security"
  - "Adversarial Attack"
  - "Spectral Defense"
  - "Gaussian Pruning"
  - "Resource Exhaustion Attack"
date: 2026-05-08
content_hash: e86051df8b2b0be8
---

# Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2603.12796](https://arxiv.org/abs/2603.12796)  
**Code**: To be confirmed  
**Area**: 3D Vision / AI Security  
**Keywords**: 3D Gaussian Splatting, Adversarial Defense, Resource Attack, Frequency Domain Analysis, Frequency-Aware Pruning

## TL;DR
Against resource-targeting attacks on 3DGS (which trigger Gaussian overgrowth via poisoned training images to deplete resources), this paper proposes a spectral defense: a 3D frequency filter achieves frequency-aware pruning by relating Gaussian covariance to spectral response, and 2D spectral regularization suppresses attack noise by penalizing the angular energy anisotropy of rendered images with entropy. This achieves a 5.92× compression in Gaussian count, a 3.66× reduction in memory, and a 4.34× speedup.

## Background & Motivation
**Background**: 3DGS is the mainstream method for 3D reconstruction owing to its high-quality rendering via adaptive Gaussian densification. However, this adaptive mechanism exposes a new attack surface—resource-targeting attacks.

**Limitations of Prior Work**: Poison-Splat attacks inject imperceptible perturbations into training images, causing the 3DGS optimizer to misinterpret noise patterns as detailed structures. This triggers excessive densification, exhausting GPU memory and severely slowing down training. Existing defenses (image smoothing, general thresholds) either destroy realistic details or fail to generalize across different scenes.

**Key Challenge**: Poisoned perturbations are invisible in the pixel domain but manifest as anomalous high-frequency amplification and directional anisotropy in the frequency domain. While spatial domain detection is unreliable, frequency domain features are highly distinguishable.

**Goal**: How to simultaneously suppress Gaussian overgrowth and attack artifacts in rendered images from a frequency-domain perspective when training supervision is poisoned?

**Key Insight**: It is discovered that poisoned images exhibit abnormal high-frequency amplification and angular energy anisotropy (whereas high-frequency energy in clean images is approximately isotropic) in the spectrum. Consequently, spectral defenses are designed for the 3D parameter space and 2D rendering space, respectively.

**Core Idea**: The root cause of poisoning attacks is anomalous spectral behavior rather than spatial structure anomalies—thus, spectral defense is effective for both 3D Gaussian pruning and 2D rendering regularization.

## Method

### Overall Architecture
Two complementary components work jointly:
- **3D Frequency Filter**: Periodically prunes splats with anomalous high-frequency responses in the Gaussian parameter space.
- **2D Spectral Regularization**: Constrains the angular energy distribution in the rendered image space via frequency-domain priors.
- Both are integrated into the standard 3DGS training loop (Algorithm 1).

### Key Designs

1. **3D Frequency Filter (Gaussian Frequency Representation $\rightarrow$ Frequency-Aware Pruning)**:

    - Function: Relates Gaussian covariance to its high-frequency response to selectively remove anomalous high-frequency splats.
    - Mechanism: The Fourier transform amplitude of a Gaussian $G(\mathbf{x})$ is determined by $\gamma(t) = (2\pi)^{3/2}|\Sigma|^{1/2}\exp(-2\pi^2 t^\top \Sigma t)$. The minimum eigenvalue of the covariance $\Sigma$, $\sigma_{\min}$, controls the narrowest spatial extent—the smaller it is, the stronger the high-frequency response. Define scoring: $\mathcal{S}(G) = \exp(-2\pi^2 t^2 \sigma_{\min}^2)$, $\mathcal{W}(G) = (1-\mathcal{S}(G))^\alpha$.
    - Combining the ray-hit $\text{hit}(G)$, the overall score is $\text{score}(G) = \mathcal{W}(G) \cdot \text{hit}(G)$. The lowest $\rho$ ratio of splats is pruned every $T_{\text{prune}}=100$ steps.
    - Design Motivation: Attack-induced splats are typically extremely small (exceptionally low $\sigma_{\min}$) and seldom observed (low hit), naturally receiving low scores to be prioritized for pruning.

2. **2D Spectral Regularization (Angular Anisotropy Penalty)**:

    - Function: Constrains the high-frequency energy of rendered images to be uniformly distributed across angular directions.
    - Mechanism: Performs a 2D DFT on the rendered image to extract the high-frequency band $\mathcal{E}(u,v)$, divides the angular domain $[-\pi, \pi)$ into $B$ sectors, and computes the angular energy $\mathcal{E}_b$ and probability distribution $\mathcal{P}_b = \mathcal{E}_b / \sum_j \mathcal{E}_j$.
    - The anisotropy loss is based on normalized entropy: $\mathcal{L}_{\text{ani}} = 1 - \frac{\mathcal{H}}{\log B}$, where $\mathcal{H} = -\sum_b \mathcal{P}_b \log \mathcal{P}_b$.
    - Design Motivation: The high-frequency energy of clean images is approximately isotropic (entropy close to maximum), whereas that of poisoned images concentrates on a few angular directions (low entropy). Penalizing anisotropy suppresses streak/band artifacts while preserving natural textures.

3. **Joint Optimization**:

    - Total loss: $\min_{\mathcal{G}} (\mathcal{L}(\dot{\mathcal{V}}^p, \mathcal{V}^p) + \lambda(\mathcal{L}_{\text{freq}}(\dot{\mathcal{V}}^p) + \mathcal{L}_{\text{tv}}(\dot{\mathcal{V}}^p)))$
    - 3D pruning is executed every 100 steps, while 2D regularization participates in gradient updates at every step.

### Loss & Training
Standard 3DGS reconstruction loss $(1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$ + spectral regularization (anisotropy + TV).

## Key Experimental Results

### Main Results (Tanks&Temples + NeRF-Synthetic + Mip-NeRF360)

| Scene | Metric | Clean | Poison | Defense | Attack/Defense Ratio |
|------|------|-------|--------|---------|-------------|
| TT-Auditorium | Gaussian Count(M) | 0.692 | 2.740 (3.96×↑) | 0.907 (3.02×↓) | — |
| TT-Auditorium | Memory(MB) | 4918 | 15280 (3.11×↑) | 7633 (2.00×↓) | — |
| TT-Courtroom | Gaussian Count(M) | 2.950 | 5.403 (1.83×↑) | 1.903 (2.84×↓) | — |
| TT-Family | Gaussian Count(M) | 2.119 | 3.696 (1.74×↑) | 1.298 (2.85×↓) | — |
| Overall Best | Overgrowth Suppression | — | — | — | **5.92×** |
| Overall Best | Memory Reduction | — | — | — | **3.66×** |
| Overall Best | Speedup | — | — | — | **4.34×** |

### Ablation Study

| Configuration | Gaussian Count↓| PSNR | Description |
|------|--------|------|------|
| No Defense (Poison) | High | Low | Overgrowth + artifacts |
| 3D Frequency Filter Only | Medium | Medium | Effectively suppresses count, but rendering remains noisy |
| 2D Spectral Regularization Only | Medium-High | Medium-High | Improves rendering quality, but count does not drop significantly |
| **Joint Defense** | **Lowest** | **Highest** | Complementary: 3D controls count + 2D controls quality |

### Key Findings
- The 3D filter and 2D regularization are highly complementary—the former suppresses count growth in the parameter space, while the latter constrains rendering quality in the image space.
- Frequency-aware pruning is more effective in poisoned scenarios than general pruning based on opacity/gradients, due to the unique frequency-domain signatures of attack-induced splats.
- Post-defense rendering quality is close to, and in some scenes even surpasses, the clean baseline—because spectral regularization acts as a regularizer improving training stability.
- Angular anisotropy is the key signal distinguishing poisoned from natural high frequencies—high-frequency energy of natural textures is distributed approximately uniformly.

## Highlights & Insights
- **Understanding Attack Mechanisms from the Frequency Domain**: The essence of poisoning is not injecting visible noise, but abnormal spectral structures—this insight raises the defense from the pixel domain to the frequency domain.
- **Elegant Mapping between Gaussian Covariance and Spectral Response**: The eigenvalues of $\Sigma$ directly determine the frequency contribution of the Gaussian—leveraging already computed parameters for security evaluation with zero extra overhead.
- **Angular Anisotropy as a Signal**: Distinguishing natural high frequencies (isotropic) from attack high frequencies (anisotropic) is a valuable signal transferable to other rendering security problems.

## Limitations & Future Work
- Pruning ratio $\rho$ and frequency cutoff $t$ are hyperparameters—different attack strengths may require different settings.
- Only one attack (Poison-Splat) is evaluated—other types of 3DGS attacks (e.g., accuracy-targeting) are not covered.
- Spectral regularization increases FFT computational overhead per step.
- The defense assumes the attack injects anisotropic spectra—if attackers know the defense mechanism, they might design isotropic poisoning.

## Related Work & Insights
- **vs Poison-Splat**: Poison-Splat proposes the attack but only discusses simple defenses (smoothing/thresholds). This work is the first systematic defense scheme for 3DGS.
- **vs Efficiency-Oriented Pruning (LightGaussian, PUP)**: These methods are designed for compression and lack the ability to distinguish attack-induced vs. realistic high frequencies.
- **Significant reference value for 3DGS security research**—the frequency-domain analysis approach can be generalized to defend against accuracy-targeting or backdoor attacks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First defense scheme against 3DGS resource attacks, with a unique and elegant spectral perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive ablations across three datasets and clean/poison/defense settings.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and convincing frequency-domain analysis.
- Value: ⭐⭐⭐⭐ Practically meaningful for 3DGS security and robustness.

**Area**: 3D Vision / AI Security  
**Keywords**: 3DGS Security, Adversarial Attack, Spectral Defense, Gaussian Pruning, Resource Exhaustion Attack

## TL;DR
This paper proposes a spectral defense mechanism against resource exhaustion attacks on 3DGS. It jointly defends in the 3D spectral domain (by linking Gaussian covariance with high-frequency response through Fourier analysis to prune anomalous high-frequency splats) and the 2D spectral domain (penalizing anisotropic angular energy distribution using an entropy loss). It suppresses Gaussian overgrowth by 5.92×, reduces memory by 3.66×, and speeds up rendering by 4.34×, while maintaining rendering quality.

## Background & Motivation
1. **Background**: The adaptive growth mechanism of 3DGS is its core advantage, but also becomes an attack surface.
2. **Limitations of Prior Work**: Maliciously injecting imperceptible high-frequency perturbations can trigger Gaussian over-expansion, leading to memory exhaustion and rendering slowdown—referred to as "resource exhaustion attacks".
3. **Key Challenge**: The growth mechanism cannot distinguish "normal high-frequency details" from "attack-induced high-frequency perturbations".
4. **Goal**: How to defend against resource exhaustion attacks that exploit the 3DGS growth mechanism?
5. **Key Insight**: The essence of the attack is the injection of anomalous high-frequency components—which can be detected and filtered using spectral analysis.
6. **Core Idea**: 3D spectral pruning of anomalous splats + 2D angular entropy regularization to protect rendering quality.

## Method

### Key Designs
1. **3D Frequency Filter**: Relates Gaussian covariance to its high-frequency response using Fourier analysis, pruning splats with abnormally strong high-frequency contributions.
2. **2D Spectral Regularization**: Penalizes anisotropic angular energy distribution, while preserving natural isotropic high-frequency content via an entropy loss.
3. **Joint Defense**: 3D handles Gaussian overgrowth, while 2D handles rendering artifacts.

## Key Experimental Results

| Metric | Gain | Description |
|------|------|------|
| Gaussian Count Suppression | 5.92× | Suppresses over-expansion caused by attacks |
| Peak Memory Reduction | 3.66× | |
| Rendering Speedup | 4.34× | |
| Clean Image Rendering Quality | Maintained | Does not affect quality when there is no attack |

### Key Findings
- 3D and 2D joint defense are both indispensable—3D handles overgrowth, 2D handles rendering artifacts.
- Spectral analysis can effectively distinguish attack perturbations from natural high-frequency details.

## Highlights & Insights
- **Solving 3DGS security from a spectral perspective**: Establishing an analytical link between Gaussian covariance and frequency response provides an interpretable defense mechanism.
- **A new angle on attacks**: Resource exhaustion attacks do not affect visual quality but drain computational resources—posing a significant threat to deployment scenarios.

## Limitations & Future Work
- Frequency thresholds and angular binning need careful tuning.
- The defense targets specific attacks; its impact on clean inputs is uncertain.

## Related Work & Insights
- **First defense method against 3DGS resource exhaustion attacks**

## Rating
- Novelty: ⭐⭐⭐⭐ Novel spectral defense perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough evaluation across multiple dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical analysis.
- Value: ⭐⭐⭐⭐ Highly significant for safe deployments of 3DGS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation](../../AAAI2026/3d_vision/cheating_stereo_matching_in_full-scale_physical_adversarial_attack_against_binoc.md)
- [\[CVPR 2025\] Spectral Informed Mamba for Robust Point Cloud Processing](spectral_informed_mamba_for_robust_point_cloud_processing.md)
- [\[CVPR 2025\] NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary](nopain_no-box_point_cloud_attack_via_optimal_transport_singular_boundary.md)
- [\[CVPR 2025\] 3D-HGS: 3D Half-Gaussian Splatting](3d-hgs_3d_half-gaussian_splatting.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
