---
title: >-
  [Paper Note] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation
description: >-
  [CVPR 2026][3D Vision][Monocular Depth Estimation] Iris proposes a deterministic diffusion framework that injects real-world priors into a diffusion model via a two-stage Prior-to-Geometry Decoupled (PGD) schedule: Stage…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Diffusion Model"
  - "Spectral Gated Distillation"
  - "Prior-to-Geometry Framework"
  - "Deterministic Diffusion"
date: 2026-05-08
content_hash: 97a89a1a4a581474
---

# Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation

**Conference**: CVPR 2026
**arXiv**: [2603.16340](https://arxiv.org/abs/2603.16340)
**Code**: [https://github.com/NUST-Machine-Intelligence-Laboratory/Iris](https://github.com/NUST-Machine-Intelligence-Laboratory/Iris)
**Area**: 3D Vision / Monocular Depth Estimation
**Keywords**: Monocular Depth Estimation, Diffusion Model, Spectral Gated Distillation, Prior-to-Geometry Framework, Deterministic Diffusion

## TL;DR

Iris proposes a deterministic diffusion framework that injects real-world priors into a diffusion model via a two-stage Prior-to-Geometry Decoupled (PGD) schedule: Stage 1 extracts low-frequency layout priors from a teacher model using Spectral Gated Distillation (SGD) at high timesteps, while Stage 2 refines high-frequency geometric details using synthetic data at low timesteps. A Spectral Gated Consistency (SGC) loss is further introduced to align high-frequency information across stages. The method achieves state-of-the-art zero-shot depth estimation performance under limited data and computational budget.

## Background & Motivation

1. **Background**: Monocular depth estimation (MDE) is a fundamental task in computer vision. Existing approaches are broadly categorized into feed-forward methods (e.g., Depth Anything V2) and diffusion-based methods (e.g., Marigold, Lotus). Feed-forward methods rely on large-scale training data, while diffusion-based methods leverage pretrained visual priors.
2. **Limitations of Prior Work**: Depth Anything V2, despite strong generalization, depends on a large-scale training pipeline that is difficult to reproduce, and still exhibits deficiencies in fine detail and boundary accuracy. Diffusion-based methods preserve detail but generalize poorly due to synthetic-to-real domain gaps.
3. **Key Challenge**: A *frequency-reliability mismatch* exists — pseudo-labels from teacher models on real images provide reliable low-frequency structure but inaccurate high-frequency details, while synthetic ground truth offers precise high-frequency geometry but lacks real-world distribution. Training both signals jointly in a single step leads to gradient interference.
4. **Goal**: To construct a model that preserves fine-grained details, generalizes robustly across domains, and matches the accuracy of large-scale training methods — all under limited annotation data and computational budget.
5. **Key Insight**: The observation that diffusion models correspond to different signal-to-noise ratios (SNR) at different timesteps — high timesteps are suited for learning global layout, while low timesteps are suited for fine geometry.
6. **Core Idea**: Decouple prior injection and geometric refinement into two distinct diffusion states, and precisely control the frequency range of knowledge transfer via spectral-domain gating mechanisms.

## Method

### Overall Architecture

Iris is built upon the Stable Diffusion architecture and adopts deterministic inference (no multi-step sampling). The input RGB image is encoded via VAE, and a U-Net with shared weights processes two sequential stages: Stage 1 injects real-world priors via teacher distillation at high timestep $t_{\text{high}}$; Stage 2 refines geometry using synthetic ground truth at low timestep $t_{\text{low}}$. Both stages share the same U-Net weights, with timesteps serving only as conditional indices.

### Key Designs

1. **Prior-to-Geometry Decoupled Framework (PGD)**:

    - **Function**: Decouples prior learning and geometric refinement into two diffusion states.
    - **Mechanism**: Stage 1 operates at high timesteps (low SNR), where the predictor focuses on global layout and boundary structure while suppressing fine textures, producing $\hat{z}^y_{\text{prior}} = f_\theta(z^x, t_{\text{high}})$. Stage 2 operates at low timesteps (high SNR), taking Stage 1's output as input and training on synthetic ground truth for precise geometry: $\hat{z}^y_{\text{geo}} = f_\theta(\hat{z}^y_{\text{prior}}, t_{\text{low}})$.
    - **Design Motivation**: Avoids gradient interference between real pseudo-labels and synthetic ground truth in single-step training; different SNR levels naturally emphasize different frequency components.

2. **Spectral Gated Distillation (SGD)**:

    - **Function**: Extracts reliable low-frequency priors from a frozen teacher model during Stage 1.
    - **Mechanism**: A lightweight low-pass gate $\mathcal{G}^{\text{low}}_\phi$ with only three learnable parameters $\phi = \{\kappa, \beta, s\}$ is designed, implementing soft truncation in the Fourier domain via a Sigmoid function. Only the low-frequency components of the teacher and student outputs are aligned; high-frequency components are left unconstrained. The loss is defined as $\mathcal{L}_{\text{sgd}} = \|\mathcal{G}^{\text{low}}_\phi(\hat{z}^y_{\text{prior}}) - \mathcal{G}^{\text{low}}_\phi(z^y_{\text{teach}})\|^2$.
    - **Design Motivation**: Teacher pseudo-labels are reliable in low frequencies (global layout, scale) but unreliable in high frequencies. Direct regression to pseudo-labels suppresses the diffusion model's inherent capacity for high-frequency detail generation.

3. **Spectral Gated Consistency (SGC)**:

    - **Function**: Uses the sharp boundaries unexpectedly produced in Stage 1 as a high-frequency teacher to guide high-frequency learning in Stage 2.
    - **Mechanism**: The low-pass alignment in Stage 1 concentrates supervision on global structure, which incidentally produces sharper boundary transitions. A differentiable high-pass gate aligns Stage 2 with Stage 1 outputs in the high-frequency domain, while an auxiliary constraint suppresses over-activation in Stage 1.
    - **Design Motivation**: A surprising finding — Stage 1 at high timesteps produces sharper boundaries and finer textures, because low-pass alignment concentrates supervision on stable global structures.

### Loss & Training

The overall training objective combines the SGD loss (Stage 1, real images), synthetic supervision loss (Stage 2, synthetic data), and the SGC consistency loss (cross-stage high-frequency alignment). Both stages share weights and are executed sequentially from high to low timesteps. Depth Anything V2 is used as the teacher model, and training is conducted on the Hypersim and Virtual KITTI synthetic datasets.

## Key Experimental Results

### Main Results

| Dataset | Metric (AbsRel↓) | Iris | DAv2-L | Lotus | Gain |
|--------|---------------|------|--------|-------|------|
| NYUv2 | AbsRel | 4.6 | 4.3 | 5.5 | Comparable to DAv2, outperforms Lotus |
| KITTI | AbsRel | 5.4 | 5.5 | 7.3 | Outperforms both DAv2 and Lotus |
| ETH3D | AbsRel | 4.8 | 5.3 | 5.6 | Significantly outperforms both |
| DIODE-Indoor | AbsRel | 14.1 | 16.0 | 18.8 | Large margin over both |

Iris achieves the best overall performance on most real-image benchmarks, with particularly strong cross-domain generalization over other diffusion-based methods.

### Ablation Study

| Configuration | AbsRel (NYU) | Note |
|------|-------------|------|
| Full Iris | 4.6 | Complete model |
| w/o SGD (direct distillation) | 5.2 | Contribution of low-frequency gating |
| w/o SGC | 4.9 | Contribution of high-frequency consistency |
| Single-stage training | 5.4 | Contribution of PGD two-stage decoupling |
| w/o teacher distillation | 5.8 | Importance of real-world priors |

### Key Findings

- SGD is the most critical module; its removal causes the largest performance drop, demonstrating the importance of spectrally-aware prior injection.
- The emergence of sharp boundaries in Stage 1 at high timesteps is a key finding that motivated the design of SGC.
- Iris uses 1–2 orders of magnitude less training data than DAv2 yet achieves comparable or superior performance on most benchmarks.
- Iris significantly outperforms DAv2 in boundary sharpness and fine detail fidelity.

## Highlights & Insights

- **Frequency-Reliability Mismatch Insight**: Different frequency bands in teacher labels carry different levels of reliability — a broadly applicable observation for other knowledge distillation scenarios.
- **3-Parameter Spectral Gate**: Precise frequency-selective knowledge transfer is achieved with an extremely lightweight design, elegant and efficient.
- **Counter-Intuitive Finding of Sharp Boundaries at High Timesteps**: Low-pass constraints paradoxically improve high-frequency performance, revealing an interesting internal property of diffusion models.

## Limitations & Future Work

- The method still depends on the reconstruction quality of the pretrained VAE; extreme fine details may be limited by VAE resolution.
- The teacher model is fixed as DAv2; employing a stronger teacher could yield further improvements.
- Two-stage training increases hyperparameter complexity (i.e., selection of the two timesteps).
- The PGD framework could be extended to other dense prediction tasks such as surface normal estimation and optical flow.

## Related Work & Insights

- **vs. Depth Anything V2**: DAv2 relies on large-scale data distillation; Iris achieves comparable results with far less data through spectral gating.
- **vs. Marigold/Lotus**: These diffusion methods are fine-tuned solely on synthetic data and lack real-world prior injection, resulting in weaker domain transfer.
- **vs. GenPercept**: GenPercept also employs single-step diffusion but does not decouple prior injection from geometric refinement.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of decoupling prior injection and geometric refinement in the spectral domain is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation is comprehensive; ablations are detailed.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; method description is thorough.
- Value: ⭐⭐⭐⭐⭐ Matching large-scale method accuracy under small-data regimes has significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)
- [\[CVPR 2026\] TR2M: Transferring Monocular Relative Depth to Metric Depth with Language Descriptions and Dual-Level Scale-Oriented Contrast](tr2m_transferring_monocular_relative_depth_to_metric_depth_with_language_descrip.md)
- [\[AAAI 2026\] Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization](../../AAAI2026/3d_vision/enhancing_generalization_of_depth_estimation_foundation_model_via_weakly-supervi.md)
- [\[CVPR 2026\] AnthroTAP: Learning Point Tracking with Real-World Motion](anthrotap_learning_point_tracking_with_real-world_motion.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)

</div>

<!-- RELATED:END -->
