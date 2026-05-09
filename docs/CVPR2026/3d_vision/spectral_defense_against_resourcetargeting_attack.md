---
title: >-
  [Paper Note] Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes the first frequency-domain defense framework against resource-targeting attacks on 3DGS — a 3D frequency filter that selectively prunes high-frequency anomalous Gaussians, combined with a 2D angular anisotropy regularization that penalizes directionally concentrated high-frequency noise. The method suppresses attack-induced Gaussian over-growth by up to 5.92×, reduces peak memory by 3.66×, improves rendering speed by 4.34×, and even raises PSNR by +1.93 dB.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - resource-targeting attack
  - frequency-domain defense
  - frequency-aware pruning
  - anisotropy regularization
date: 2026-05-08
content_hash: a68a0c835ba9ef21
---

# Spectral Defense Against Resource-Targeting Attack in 3D Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2603.12796](https://arxiv.org/abs/2603.12796)
**Code**: None
**Area**: 3D Vision / Neural Rendering Security
**Keywords**: 3D Gaussian Splatting, resource-targeting attack, frequency-domain defense, frequency-aware pruning, anisotropy regularization

## TL;DR

This paper proposes the first frequency-domain defense framework against resource-targeting attacks on 3DGS — a 3D frequency filter that selectively prunes high-frequency anomalous Gaussians, combined with a 2D angular anisotropy regularization that penalizes directionally concentrated high-frequency noise. The method suppresses attack-induced Gaussian over-growth by up to 5.92×, reduces peak memory by 3.66×, improves rendering speed by 4.34×, and even raises PSNR by +1.93 dB.

## Background & Motivation

**State of the Field**: 3DGS adaptively densifies and prunes Gaussian primitives to match scene complexity, and has become the dominant paradigm for real-time 3D reconstruction. The Poison-Splat attack reveals a new threat surface: by injecting imperceptible perturbations into training images (constrained within an $\epsilon$-ball), it constructs a max-min bi-level optimization that induces excessive densification in 3DGS, causing GPU memory explosion and rendering slowdown.

**Limitations of Prior Work**: (1) Image smoothing destroys genuine fine-grained structures; (2) uniform Gaussian thresholding (UT) fails to generalize across scenes — too aggressive for some, too lenient for others; (3) efficiency-oriented pruning methods (LightGaussian/PUP/MaskGaussian) are designed for compression rather than robustness, and cannot distinguish adversarial Gaussians from real details when supervised by poisoned views.

**Root Cause**: Attack perturbations are visually imperceptible in pixel space but exhibit two characteristic fingerprints in the frequency domain — **anomalous high-frequency amplification** and **directional anisotropy**. Spatial-domain defenses cannot capture these covert spectral distortions, causing the optimizer to misinterpret noise patterns as fine structures, triggering unstable Gaussian over-growth.

**Paper Goals**: To design a defense mechanism from a frequency-domain perspective that selectively suppresses attack-induced Gaussian over-growth without compromising genuine scene structures.

**Starting Point**: A mathematical connection is established between the Gaussian covariance matrix and its frequency response — the smaller the minimum eigenvalue of the covariance, the weaker the high-frequency attenuation and the stronger the high-frequency response. This provides a theoretical basis for distinguishing attack noise from genuine scene detail.

**Core Idea**: The root cause of the attack lies in spectral behavior rather than spatial structure — a 3D frequency filter selectively prunes anomalous high-frequency Gaussians, while a 2D anisotropy regularization constrains the directional high-frequency distribution of rendered images.

## Method

### Overall Architecture

The defense operates jointly in two domains within the 3DGS training loop: render image at each iteration → execute 3D frequency-aware pruning if the pruning interval is reached (every $T_{prune}=100$ steps) → compute 2D anisotropy loss and TV loss → jointly optimize to update the Gaussian set. The input is the poisoned image set $\mathcal{V}^p$ and the output is the defended Gaussian set $\mathcal{G}$.

### Key Designs

1. **3D Frequency Filter (Parameter-Space Pruning)**

    - **Function**: Selectively prunes attack-induced redundant Gaussians based on their frequency-domain characteristics.
    - **Mechanism**: The Fourier transform amplitude of a 3D Gaussian $G$ is $\gamma(t) = (2\pi)^{3/2}|\Sigma|^{1/2}\exp(-2\pi^2 t^\top \Sigma t)$, where the covariance $\Sigma$ fully determines the frequency characteristics (position $\mu$ only affects phase, not spectral distribution). A smaller minimum eigenvalue $\sigma_{min}$ implies weaker high-frequency attenuation and stronger high-frequency response. A high-frequency decay score is defined as $\mathcal{S}(G) = \exp(-2\pi^2 t^2 \sigma_{min}^2)$, mapped to an importance weight $\mathcal{W}(G) = (1-\mathcal{S}(G))^\alpha$ — Gaussians with large $\mathcal{S}$ (strong high-frequency response) receive low weights. Visibility is incorporated by randomly sampling $K^*$ views and counting ray hits $\text{hit}(G)$, yielding a final score $\text{score}(G) = \mathcal{W}(G) \cdot \text{hit}(G)$.
    - **Design Motivation**: Pruning high-frequency Gaussians alone is insufficient, as natural textures also contain legitimate high frequencies. By combining the frequency decay score (distinguishing the extreme high frequencies of attack noise) with visibility (low score = "rarely observed yet exhibiting strong high-frequency behavior" = typical attack-induced Gaussian), the method achieves precise discrimination.

2. **2D Spectral Regularization (Image-Space Constraint)**

    - **Function**: Constrains the directional frequency distribution of rendered images to suppress anisotropic high-frequency noise introduced by attacks.
    - **Mechanism**: Apply 2D DFT to the rendered view $\dot{V}$ → extract the high-frequency band $\mathcal{E}(u,v)$ using amplitude thresholds $[\dot{\gamma}_{min}, \dot{\gamma}_{max}]$ → discretize the frequency plane $[-\pi,\pi)$ into $B=36$ angular bins → aggregate high-frequency energy per bin to obtain a probability distribution $\mathcal{P}_b = \mathcal{E}_b / \sum_j \mathcal{E}_j$ → compute normalized entropy $\text{norm}(\mathcal{H}) = -\sum_b \mathcal{P}_b \log \mathcal{P}_b / \log B$ → anisotropy loss $\mathcal{L}_{ani} = 1 - \text{norm}(\mathcal{H})$.
    - **Design Motivation**: High frequencies in clean images are approximately isotropic ($\mathcal{H} \to \log B$, loss $\to 0$), whereas poisoned images concentrate high-frequency energy in a few directions (low $\mathcal{H}$, high loss). Since 3D pruning operates only in parameter space, the model may still converge toward noise artifacts when optimized on poisoned views — 2D regularization provides a complementary constraint from the image space.

### Loss & Training

Total loss: $\mathcal{L}_{total} = (1-\lambda_0)\mathcal{L}_1 + \lambda_0\mathcal{L}_{\text{D-SSIM}} + \lambda(\mathcal{L}_{freq} + \mathcal{L}_{tv})$, where $\mathcal{L}_{freq} = \frac{1}{K}\sum_{k=1}^K \mathcal{L}_{ani}(\dot{V}_k^p)$. Hyperparameters: $t=8$, $\alpha=2$; pruning rates NS $\rho=3\%$, TT $\rho=4.5\%$, MIP $\rho=5\%$; $\lambda=4$ (NS/TT) or $5$ (MIP); $K^*=48$, $B=36$. Default attack strength $\epsilon=16/255$.

## Key Experimental Results

### Main Results — Training Resource Suppression (3 Datasets, 38 Scenes)

| Dataset | Metric | Clean | Poisoned | Defended | Attack Suppression |
|--------|------|------|------|--------|-------------|
| TT (21 scenes) | Max Gaussians (M) | 1.751 | 2.889 (1.65×↑) | 1.128 | **2.56×↓** |
| NS (8 scenes) | Max Gaussians (M) | 0.291 | 0.720 (2.47×↑) | 0.273 | **2.64×↓** |
| MIP (9 scenes) | Max Gaussians (M) | 3.191 | 7.045 (2.21×↑) | 1.876 | **3.76×↓** |
| TT average | Peak Memory (MB) | 7408 | 11276 | 6614 | **1.70×↓** |
| MIP average | Peak Memory (MB) | 12510 | 24445 | 11491 | **2.13×↓** |

### Extreme Scenes and Rendering Quality

| Scene | Metric | Poisoned | Defended | Effect |
|------|------|------|--------|------|
| MIP-bonsai | Gaussian Count | 6.139M | 1.037M | **5.92×↓** |
| MIP-bonsai | PSNR | 27.14 | 29.07 | **+1.93 dB** |
| MIP-garden | FPS | 48 | 208 | **4.34×↑** |
| NS-hotdog | Memory | 28124 MB | 7781 MB | **3.61×↓** |

### Ablation Study

| Ablation Dimension | Key Findings |
|----------|----------|
| Frequency reference $t$ | $t=8$ is optimal; results are stable for $t \in [4,10]$ |
| 2D regularization hyperparameters | Highly robust: $\dot{\gamma} \in [0.28,0.92] \to [0.40,0.85]$, PSNR variation $<0.2$ dB |
| Attack strength robustness | Effective across the full range from $\epsilon=8/255$ to unbounded |
| Black-box generalization | Remains effective when attacking Scaffold-GS (MIP-bonsai: 11.1M→2.0M, 5.52×↓) |
| Clean input compatibility | Also compresses Gaussians without attack (MIP-bicycle: 5.78M→1.34M, 4.32×↓) without quality loss |

### Key Findings

- PSNR improves after defense — frequency-aware pruning removes precisely the noisy Gaussians, and denoising enhances reconstruction quality.
- The method remains effective on clean scenes, effectively serving dual roles as both a security defense and an efficient compression technique.
- Black-box transfer to variants such as Scaffold-GS is effective, demonstrating the generality of spectral fingerprints.

## Highlights & Insights

- **Pioneering frequency-domain defense perspective**: This work is the first to reveal the spectral fingerprints of poisoning attacks (high-frequency amplification + directional anisotropy) and addresses the problem from the frequency domain rather than spatial domain, with a solid theoretical foundation.
- **Elegant mathematical derivation**: The derivation chain from Gaussian covariance → Fourier transform → frequency-aware scoring is complete and rigorous. The conclusion that $\Sigma$ fully determines frequency characteristics while $\mu$ does not is clean and elegant.
- **Defense as compression**: The method also functions as efficient Gaussian compression in clean scenes without degrading rendering quality — a bonus effect combining security and efficiency.
- **Comprehensive experiments**: 3 datasets × 38 scenes × 3 settings, plus black-box transfer, clean compatibility, and detailed hyperparameter ablations, ensuring high reproducibility.

## Limitations & Future Work

- The core assumption is that attacks introduce high-frequency anisotropy — an adaptive attacker generating isotropic noise could potentially bypass the 2D regularization.
- The method targets only resource-targeting attacks and does not address accuracy-targeting attacks (e.g., StealthAttack).
- The pruning rate $\rho$ is manually set per dataset (NS 3% / TT 4.5% / MIP 5%) without adaptive adjustment.
- The frequency thresholds $\dot{\gamma}_{min}/\dot{\gamma}_{max}$ are fixed values and are not adaptively coupled to scene complexity.

## Related Work & Insights

- **vs. Poison-Splat original defenses**: The image smoothing and uniform thresholding proposed in the original paper are naive baselines; this work provides the first theoretically grounded defense.
- **vs. LightGaussian/PUP efficiency methods**: These methods fail under poisoning (unable to distinguish adversarial Gaussians from genuine details), yet the proposed method performs comparably even on clean scenes.
- The frequency-domain analysis paradigm is generalizable to security research on other Gaussian-based 3D representations, such as 4DGS and GS-SLAM.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The frequency-domain perspective on defending 3DGS attacks is entirely novel, with rigorous mathematical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 38 scenes × 3 settings + black-box transfer + clean compatibility + 8 ablation groups.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, and the frequency-domain analysis is insightful.
- Value: ⭐⭐⭐⭐ Significant implications for secure deployment of 3DGS, with additional practical value as a compression technique.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)

<!-- RELATED:END -->
