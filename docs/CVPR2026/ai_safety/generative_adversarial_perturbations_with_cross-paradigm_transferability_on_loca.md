---
title: >-
  [Paper Note] Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting
description: >-
  [CVPR 2026][AI Safety][adversarial attack] This paper proposes CrowdGen, the first cross-paradigm adversarial attack framework targeting both density-map and point-regression crowd counting models. A lightweight UNet generator combined with a multi-task loss (logit suppression, density suppression, GradCAM guidance, and frequency-domain constraint) achieves high transferability (TR up to 1.69) across seven SOTA crowd counting models while maintaining visual imperceptibility (~19 dB PSNR), increasing attack MAE by an average factor of 7×.
tags:
  - CVPR 2026
  - AI Safety
  - adversarial attack
  - crowd counting
  - cross-paradigm transferability
  - generative adversarial perturbation
  - black-box attack
date: 2026-05-08
content_hash: 573e115a2dc893e2
---

# Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting

**Conference**: CVPR 2026
**arXiv**: [2603.24821](https://arxiv.org/abs/2603.24821)
**Code**: [https://github.com/simurgh7/CrowdGen](https://github.com/simurgh7/CrowdGen)
**Area**: AI Safety / Adversarial Attacks
**Keywords**: adversarial attack, crowd counting, cross-paradigm transferability, generative adversarial perturbation, black-box attack

## TL;DR
This paper proposes CrowdGen, the first cross-paradigm adversarial attack framework targeting both density-map and point-regression crowd counting models. A lightweight UNet generator combined with a multi-task loss (logit suppression, density suppression, GradCAM guidance, and frequency-domain constraint) achieves high transferability (TR up to 1.69) across seven SOTA crowd counting models while maintaining visual imperceptibility (~19 dB PSNR), increasing attack MAE by an average factor of 7×.

## Background & Motivation
Localized crowd counting is widely deployed in public safety, retail analytics, and epidemic monitoring. Current mainstream approaches fall into two paradigms: **density-map methods** (e.g., SASNet, FIDTM), which regress spatial density distributions and extract localization via post-processing, and **point-regression methods** (e.g., P2PNet, PET), which directly predict coordinates and confidence scores end-to-end.

Existing adversarial attacks suffer from the following limitations:

**Attack strength vs. imperceptibility trade-off**: PAP and GE-AdvGAN achieve good visual quality (PSNR ≥ 22 dB) but weak attacks (MAE < 120); DiffAttack yields strong attacks (MAE = 414) but severe visual degradation (PSNR = 11.5 dB).

**Single-paradigm limitation**: Existing transferable attacks (APAM, PAP) transfer only within density-map methods and do not consider cross-paradigm transfer (density-map ↔ point-regression).

**Black-box deployment requirements**: Real-world crowd counting systems are typically black-box, necessitating surrogate-model-based transfer attacks.

**Core Idea**: The shared backbone feature space (e.g., VGG-16, ResNet-50) across both paradigms encodes common inductive biases. By combining paradigm-specific attack losses with paradigm-agnostic perceptual constraints, a unified generative perturbation generator can be learned.

## Method

### Overall Architecture
The framework consists of three core components: a 3-layer UNet generator $G_\theta$ that maps input images to bounded perturbations $\delta$, with the total loss composed of a **paradigm-specific model loss** $\mathcal{L}_{model}$ and a **cross-paradigm perturbation loss** $\mathcal{L}_{pert}$. Training is performed against surrogate models; at inference time, adversarial examples are generated in a single forward pass without per-image optimization.

### Key Designs

1. **Logit Suppression Loss (for point-regression models)**:

    - Attacks are focused on high-confidence detections $\mathcal{P}_{high} = \{i : s_i^{(h)} > \tau\}$
    - Dense scenes ($C_{gt} > C_{sparse}$): directly minimizes logit values in high-confidence regions
    - Sparse scenes ($C_{gt} \le C_{sparse}$): applies weighted penalties to detections near the confidence boundary
    - Adaptive threshold decay: $\tau(t) = \max(\tau_{min}, \tau_{max} - \nu \cdot t / T_{max})$, progressively lowering the threshold during training to expand the attack surface

2. **Density Suppression Loss (for density-map models)**:

    - **Heatmap suppression** $\mathcal{L}_{hmap}$: simultaneously attacks salient peaks and near-threshold regions; local maxima are detected via 3×3 max-pooling, with foreground separated by an adaptive threshold $\phi = \phi' \cdot \max(\mathcal{D})$
    - **Peak suppression** $\mathcal{L}_{peak}$: additionally incorporates peak prominence (difference between peak value and local neighborhood) to emphasize isolated high-density clusters
    - An isolation ratio (proportion of peaks with no neighbors within a 5×5 window > 0.7) automatically selects which loss to apply

3. **Cross-paradigm Perturbation Loss**:

    - **Frequency-domain constraint** $\mathcal{L}_{freq}$: suppresses high-frequency components via FFT, exploiting the low-frequency dominance of crowd scenes to improve transferability
    - **GradCAM guidance** $\mathcal{L}_{cam}$: concentrates perturbations on semantically important regions identified by the shared backbone, minimizing perturbation outside attention regions
    - **Magnitude constraint** $\mathcal{L}_{hinge}$: bounds perturbation energy via L2 norm
    - **Spatial smoothness regularization** $\mathcal{L}_{tv}$: total variation regularization to reduce perturbation artifacts

### Loss & Training
Total loss: $\mathcal{L}_{attack} = \alpha \cdot \mathcal{L}_{model} + \beta \cdot \mathcal{L}_{hinge} + \gamma \cdot \mathcal{L}_{tv} + \zeta \cdot \mathcal{L}_{freq} + \kappa \cdot \mathcal{L}_{cam}$

- Perturbation budget $\epsilon = 8/255$; image resolution 512×512
- Cosine annealing learning rate schedule
- Hyperparameters tuned via grid search on a validation set: $\beta=0.01, \gamma=0.05, \zeta=0.01, \kappa=0.5$

## Key Experimental Results

### Main Results (Cross-model Transferability, SHHA Dataset)

| Surrogate → Target | MAE / TR | Notes |
|---------|---------|------|
| HMoDE → P2PNet | 420.71 / **1.69** | Cross-paradigm super-transfer: stronger than white-box self-attack |
| FIDTM → P2PNet | 426.89 / 1.64 | Density-map → point-regression strong transfer |
| SASNet → APGCC | 397.96 / 1.32 | Density-map → point-regression |
| P2PNet → SASNet | 281.00 / 0.89 | Point-regression → density-map |
| APGCC → HMoDE | 171.53 / **0.55** | Weakest transfer, yet MAE still doubles |
| Clean baseline | 28–75 | Counting error on clean images |

### Ablation Study (Loss Combinations, SHHA Dataset)

| Loss Combination | Miss Rate (%) | PSNR (dB) | Notes |
|---------|-------------|----------|------|
| $\mathcal{L}_{hmap} + \mathcal{L}_{hinge}$ (baseline) | 45.35 | 17.67 | Basic density attack only |
| + $\mathcal{L}_{cam}$ | 59.47 | 17.67 | GradCAM adds +14% MR |
| + $\mathcal{L}_{freq}$ | 60.46 | 17.75 | Frequency constraint also significantly improves |
| Full combination (density-map) | **60.89** | 17.47 | Best attack strength |
| $\mathcal{L}_{logit} + \mathcal{L}_{hinge}$ (baseline) | 45.15 | 19.09 | Basic logit attack only |
| + $\mathcal{L}_{cam}$ | **45.61** | **19.10** | Best trade-off for point-regression |

### Key Findings
- **Cross-paradigm super-transfer**: CNN-based HMoDE attacking Transformer-based PET achieves TR = 1.60 on UCF-QNRF, validating the shared backbone inductive bias hypothesis.
- The contribution of each loss component is **paradigm-dependent**: frequency-domain constraints are critical for density-map models, while GradCAM guidance benefits point-regression models more.
- In dense scenes, the miss rate reaches 58%, concealing the majority of the crowd, whereas prior methods achieve only 15–31%.

## Highlights & Insights
- This work is the first to reveal cross-paradigm adversarial vulnerability in crowd counting models; the super-transfer phenomenon (TR > 1) suggests that black-box attackers may be more effective than white-box ones.
- The scene-density-adaptive logit suppression strategy (dense vs. sparse branches) elegantly handles the distinct characteristics of different scene types.
- The generative single-forward-pass attack is more practical than iterative optimization methods, offering high inference efficiency.

## Limitations & Future Work
- Only digital-domain attacks are evaluated; physical-world scenarios (printing, projection) are not considered.
- The attack focuses primarily on under-counting; the over-counting direction (hallucinating crowds) remains unexplored.
- The perturbation budget $\epsilon = 8/255$ is standard; performance under smaller budgets has not been verified.
- Experiments against adversarial defense methods are absent.

## Related Work & Insights
- This framework can serve as a standardized benchmark for robustness evaluation of crowd counting systems.
- The GradCAM-guided perturbation allocation strategy is generalizable to adversarial attacks on other dense prediction tasks.
- The finding of shared-backbone vulnerability across paradigms has important implications for security strategies in model deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ First cross-paradigm adversarial attack on crowd counting, though the generative adversarial perturbation framework itself is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Transfer matrix across 7 models × 2 datasets, comparison with 9 baselines, and ablation studies — fairly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and notation is complete, though some symbols are heavy.
- Value: ⭐⭐⭐⭐ Provides important insights into the vulnerability of safety-critical crowd analysis systems.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Boosting Adversarial Transferability with Spatial Adversarial Alignment](../../NeurIPS2025/ai_safety/boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[CVPR 2026\] A Unified Perspective on Adversarial Membership Manipulation in Vision Models](a_unified_perspective_on_adversarial_membership_manipulation_in_vision_models.md)
- [\[NeurIPS 2025\] Private Continual Counting of Unbounded Streams](../../NeurIPS2025/ai_safety/private_continual_counting_of_unbounded_streams.md)

<!-- RELATED:END -->
