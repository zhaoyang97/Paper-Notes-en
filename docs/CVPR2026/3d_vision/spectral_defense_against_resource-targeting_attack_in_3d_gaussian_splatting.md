---
title: >-
  [Paper Note] Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting
description: >-
  [CVPR2026][3D Vision][3D Gaussian Splatting] This paper proposes the first frequency-domain defense framework against resource-targeting attacks on 3DGS. By combining a 3D frequency filter that selectively prunes anomalous high-frequency Gaussians with 2D spectral regularization that constrains anisotropic noise in rendered images, the method suppresses Gaussian over-proliferation by up to 5.92×, reduces peak GPU memory by up to 3.66×, and accelerates rendering by up to 4.34× under attack, while maintaining reconstruction quality.
tags:
  - CVPR2026
  - 3D Vision
  - 3D Gaussian Splatting
  - adversarial defense
  - resource-targeting attack
  - frequency-domain analysis
  - Gaussian pruning
  - spectral regularization
date: 2026-05-08
content_hash: ece613d90dc48865
---

# Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting

**Conference**: CVPR2026
**arXiv**: [2603.12796](https://arxiv.org/abs/2603.12796)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, adversarial defense, resource-targeting attack, frequency-domain analysis, Gaussian pruning, spectral regularization

## TL;DR

This paper proposes the first frequency-domain defense framework against resource-targeting attacks on 3DGS. By combining a 3D frequency filter that selectively prunes anomalous high-frequency Gaussians with 2D spectral regularization that constrains anisotropic noise in rendered images, the method suppresses Gaussian over-proliferation by up to 5.92×, reduces peak GPU memory by up to 3.66×, and accelerates rendering by up to 4.34× under attack, while maintaining reconstruction quality.

## Background & Motivation

- **Background**: 3D Gaussian Splatting employs an adaptive densification mechanism to match scene complexity, but this flexibility introduces a new attack surface — resource-targeting attacks. Adversaries need only poison training images to induce excessive Gaussian proliferation, resulting in GPU memory exhaustion and significant degradation of training and rendering speed.

- **Limitations of Prior Work**: The simple defenses proposed alongside Poison-Splat — image smoothing or uniform Gaussian count thresholds — exhibit clear shortcomings. Smoothing destroys valid fine-grained structures, while a uniform threshold fails to generalize across scenes of varying complexity, being either overly restrictive for some scenes or insufficiently strict for others. Efficiency-oriented pruning methods such as LightGaussian and PUP are designed for clean inputs and cannot reliably distinguish fine details from adversarial noise textures under poisoned inputs, making them unable to identify and remove attack-induced Gaussians.

- **Key Challenge**: Spatial-domain detection is inherently unreliable because poisoning perturbations are imperceptible in pixel space (constrained within an $\varepsilon$-ball), yet manifest in the frequency domain as anomalous high-frequency energy amplification and directional anisotropy. The authors identify that the root cause of over-proliferation lies in spectral behavior rather than spatial structure: poisoned images exhibit abnormal energy concentration and directional skewness in the high-frequency region of the Fourier domain, misleading the optimizer into interpreting noise patterns as structural detail. Direct high-frequency suppression is infeasible because natural scenes contain legitimate high-frequency components (edges, textures), and naive filtering severely degrades reconstruction fidelity.

- **Goal**: To develop a principled frequency-domain defense that distinguishes legitimate from adversarial high-frequency content, suppressing attack-induced Gaussian over-growth without sacrificing scene reconstruction quality.

## Method

### Overall Architecture

Spectral Defense operates jointly in 3D parameter space and 2D image space:

- **3D Frequency Filter**: Defines a frequency-aware importance score for each Gaussian based on the frequency-domain characteristics of its covariance matrix, and periodically prunes Gaussians exhibiting anomalous high-frequency responses.
- **2D Spectral Regularization**: Constrains the angular distribution of high-frequency energy in the Fourier domain of rendered images, penalizing anisotropic noise patterns.
- **Joint Optimization**: Integrates reconstruction loss, spectral regularization loss, and total variation loss into a unified objective.

### Key Designs

**3D Frequency Filter**

The covariance matrix $\Sigma$ of each 3D Gaussian $G$ fully determines its frequency response. After Fourier transformation, the amplitude decay is given by $\gamma(t) = (2\pi)^{3/2}|\Sigma|^{1/2}\exp(-2\pi^2 t^\top \Sigma t)$, where a smaller minimum eigenvalue $\sigma_{\min}$ of $\Sigma$ corresponds to stronger high-frequency response.

- **High-frequency decay score**: $\mathcal{S}(G) = \exp(-2\pi^2 t^2 \sigma_{\min}^2)$, evaluated at a fixed cutoff frequency $t=8$.
- **Frequency-aware weight**: $\mathcal{W}(G) = (1 - \mathcal{S}(G))^\alpha$ ($\alpha=2$), assigning lower weights to anomalously high-frequency Gaussians.
- **Composite score**: $\text{score}(G) = \mathcal{W}(G) \cdot \text{hit}(G)$, incorporating ray hit count to reflect geometric visibility.
- **Periodic pruning**: Every $T_{\text{prune}}=100$ steps, $K^*=48$ viewpoints are randomly sampled to compute scores, and the lowest-scoring $\rho\%$ of Gaussians are removed.

**2D Spectral Regularization**

A 2D DFT is applied to rendered images, and a high-frequency band $\mathcal{E}(u,v)$ is extracted where energy falls within $[\dot{\gamma}_{\min}, \dot{\gamma}_{\max}] = [0.3, 0.9]$. The angular energy distribution is then analyzed:

- The angular domain $[-\pi, \pi)$ is uniformly discretized into $B=36$ sectors, and the high-frequency energy $\mathcal{E}_b$ within each sector is aggregated.
- The distribution is normalized to a probability distribution $\mathcal{P}_b$; clean images approximate a uniform (isotropic) distribution, while poisoned images exhibit sharp concentration in a small number of directions (anisotropy).
- **Anisotropy loss**: $\mathcal{L}_{\text{ani}} = 1 - \mathcal{H}/\log B$, where $\mathcal{H}$ is the information entropy of the angular distribution.

### Loss & Training

$$\min_{\mathcal{G}} \Big(\mathcal{L}(\dot{\mathcal{V}}^p, \mathcal{V}^p) + \lambda\big(\mathcal{L}_{\text{freq}}(\dot{\mathcal{V}}^p) + \mathcal{L}_{\text{tv}}(\dot{\mathcal{V}}^p)\big)\Big)$$

where $\mathcal{L}$ is the standard 3DGS reconstruction loss (L1 + D-SSIM), $\mathcal{L}_{\text{freq}}$ is the mean per-view anisotropy loss, and $\mathcal{L}_{\text{tv}}$ is the total variation loss. $\lambda$ is set to 4–5 depending on scene complexity.

## Key Experimental Results

### Experimental Setup

- **Datasets**: Tanks and Temples (21 scenes), NeRF-Synthetic (8 objects), Mip-NeRF 360 (9 scenes).
- **Baselines**: Universal Threshold (UT▽), LightGaussian (LG▽), PUP 3D-GS (PUP▽), all implemented under the poisoned setting.
- **Metrics**: Gaussian count, peak GPU memory, training time, FPS, PSNR, SSIM.
- **Hardware**: Single NVIDIA RTX A6000.

### Main Results

| Dataset | Metric | Clean | Poison | Defense | Effect |
|---------|--------|-------|--------|---------|--------|
| TT (avg) | Gaussians (M) | 1.751 | 2.889 (1.65×↑) | 1.128 (2.56×↓) | Effective suppression |
| NS (avg) | Gaussians (M) | 0.291 | 0.720 (2.47×↑) | 0.273 (2.64×↓) | Below clean |
| MIP (avg) | Gaussians (M) | 3.191 | 7.045 (2.21×↑) | 1.876 (3.76×↓) | Significant compression |
| MIP-bonsai | Gaussians (M) | 1.273 | 6.139 (4.82×↑) | 1.037 (**5.92×↓**) | Best case |
| TT-Train | Peak memory (MB) | 5674 | 15805 (2.79×↑) | 4324 (**3.66×↓**) | Best case |
| MIP-garden | FPS | — | 48 (poison) | 208 (**4.34×↑**) | Best case |

In terms of rendering quality, the proposed defense outperforms all pruning baselines across all scenes. For example, on MIP-bonsai, PSNR improves from 27.14 (poisoned) to 29.07 (UT▽ achieves only 22.68), and SSIM improves from 0.64 to 0.84.

### Ablation Study

| Ablation Factor | Key Findings |
|----------------|-------------|
| Reference frequency $t$ and exponent $\alpha$ | $t=8, \alpha=2$ yields the best performance; results are stable across different settings. |
| Pruning ratio $\rho$ and sample count $K^*$ | $\rho=3\%, K^*=48$ achieves optimal balance on NS; excessively high $\rho$ degrades PSNR. |
| Frequency thresholds $[\dot{\gamma}_{\min}, \dot{\gamma}_{\max}]$ | [0.3, 0.9] is globally optimal; the method is insensitive to hyperparameter variation. |
| Angular bin count $B$ | $B=36$ is optimal; larger values cause Gaussian count to rebound. |
| Loss weight $\lambda$ | Set to 4 for TT/NS and 5 for MIP; excessively large values over-suppress natural patterns. |
| Attack strength $\epsilon$ | Defense remains effective from $\epsilon=8/255$ to unconstrained attacks; gains are more pronounced under stronger attacks. |

### Key Findings

- Under the defense setting, the Gaussian count can be compressed **below the clean baseline** (e.g., NS average: 0.273M defended vs. 0.291M clean), indicating that the frequency filter also removes redundancy present in unattacked scenes.
- Applying the defense to clean inputs is equally effective (Table 4): on MIP-bicycle, the Gaussian count decreases from 5.782M to 1.339M (4.32×↓), demonstrating dual utility as an efficiency optimization tool.
- Black-box attack experiments (Table 5): when attacks are generated against 3DGS but the victim is Scaffold-GS, the defense remains effective, demonstrating cross-architecture generalization.

## Highlights & Insights

- **Pioneering contribution**: This is the first defense framework against resource-targeting attacks on 3DGS, filling a critical gap in 3DGS security research.
- **Novel frequency-domain perspective**: Analyzing the root cause of attacks through spectral behavior reveals high-frequency anisotropy as the core signal, offering a more principled approach than spatial-domain methods.
- **Complementary dual-layer defense**: The 3D frequency filter addresses parameter-space redundancy while 2D spectral regularization corrects image-domain noise; the two components are synergistic and more effective than either alone.
- **Strong practicality**: The method integrates as a plug-and-play module into the training loop without requiring clean supervision, and is equally applicable as an efficiency optimization tool in non-adversarial settings.
- **Comprehensive evaluation**: Experiments span 38 scenes across 3 datasets, with averaged results over 3 runs, covering clean/poisoned/defense settings with thorough ablation and generalization studies.

## Limitations & Future Work

- The pruning ratio $\rho$ and loss weight $\lambda$ require manual tuning for scenes of different scales (NS: 3%/4, TT: 4.5%/4, MIP: 5%/5), limiting automation.
- Spectral regularization based on global DFT may be insufficiently sensitive to localized attack patterns (i.e., perturbations confined to specific image regions).
- Only Poison-Splat is evaluated as the attack method; the framework has not been assessed against potential adaptive adversarial attacks specifically designed to evade frequency-domain defenses.
- Training time reduction for complex scenes (e.g., MIP-counter) remains marginal (1.12×↓) after defense, indicating efficiency bottlenecks for large-scale scenes.
- The cutoff frequency $t$ is fixed as a global constant rather than adaptively tuned to scene content.

## Related Work & Insights

- **Poison-Splat** [Lu et al., 2024]: The first resource-targeting attack on 3DGS, which establishes the attack setting studied in this paper.
- **LightGaussian** [Fan et al., 2024]: Importance-score-based Gaussian pruning; used as a comparison baseline.
- **PUP 3D-GS** [Hanson et al., 2025]: An alternative pruning strategy; also used as a comparison baseline.
- **Scaffold-GS** [Lu et al., 2024]: An anchor-based Gaussian representation used in the black-box attack generalization experiments.
- **MaskGaussian** [Liu et al., 2025]: A learnable mask-based pruning strategy.
- **3DGS security research**: StealthAttack [Ke et al., 2025] targets rendering accuracy; IPA-NeRF [Jiang et al., 2024] targets NeRF poisoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First defense against 3DGS resource-targeting attacks; frequency-domain perspective is distinctive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 38 scenes across 3 datasets, multiple baselines, comprehensive ablations, and generalization to clean inputs and black-box settings.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rigorous frequency-domain derivations, and information-dense figures and tables.
- **Value**: ⭐⭐⭐⭐ — Fills the security defense gap and offers practical utility as an efficiency optimization tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[CVPR 2026\] ReLaGS: Relational Language Gaussian Splatting](relags_relational_language_gaussian_splatting.md)
- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)

</div>

<!-- RELATED:END -->
