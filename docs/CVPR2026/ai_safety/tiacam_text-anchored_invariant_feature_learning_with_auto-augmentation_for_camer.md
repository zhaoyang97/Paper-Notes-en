---
title: >-
  [Paper Note] TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking
description: >-
  [CVPR2026][AI Safety][Zero-watermarking] This paper proposes TIACam, a framework that simulates camera distortions via a learnable auto-augmentor…
tags:
  - "CVPR2026"
  - "AI Safety"
  - "Zero-watermarking"
  - "cross-modal alignment"
  - "learnable data augmentation"
  - "camera robustness"
  - "CLIP"
  - "adversarial training"
  - "invariant feature learning"
date: 2026-05-08
content_hash: e375c640f2d72cad
---

# TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking

**Conference**: CVPR2026
**arXiv**: [2602.18863](https://arxiv.org/abs/2602.18863)  
**Code**: To be confirmed  
**Area**: Object Detection (actually Multimedia Security / Watermarking)
**Keywords**: Zero-watermarking, cross-modal alignment, learnable data augmentation, camera robustness, CLIP, adversarial training, invariant feature learning

## TL;DR

This paper proposes TIACam, a framework that simulates camera distortions via a learnable auto-augmentor, learns invariant features through text-anchored cross-modal adversarial training, and binds binary messages to features via a zero-watermarking head—achieving camera-robust zero-watermarking without modifying any image pixels. TIACam attains state-of-the-art bit accuracy across three real-world scenarios: screen recapture, print-and-scan, and screenshot.

## Background & Motivation

1. **Zero-watermarking paradigm**: Conventional watermarking modifies images in the spatial or transform domain; zero-watermarking instead associates the watermark with intrinsic image features without altering pixels, balancing invisibility with verification reliability.
2. **Camera recapture challenge**: Camera recapture introduces compound, spatially coupled degradations—perspective distortion, illumination variation, sensor noise, and Moiré patterns—making it one of the most challenging scenarios for watermark extraction.
3. **Limitations of handcrafted noise layers**: Methods such as StegaStamp and PIMoG manually design camera noise layers, but real optical distortions vary with environment and are nonlinearly coupled, making fixed augmentations insufficient.
4. **Suboptimal pretrained features**: The robustness of features from self-supervised models such as DINO is a byproduct of pretraining, not explicitly optimized for watermarking tasks.
5. **Insufficient single-source invariance**: Invariance learning based solely on text guidance or solely on distortion adversarial training cannot simultaneously guarantee semantic consistency and distortion robustness.
6. **Lack of a unified framework**: Existing methods treat augmentation, feature learning, and watermark binding as separate stages, lacking an end-to-end joint optimization mechanism.

## Method

### Overall Architecture

TIACam consists of three modules trained jointly in a tripartite adversarial loop:

- **Auto-Augmentor**: A differentiable camera distortion simulation pipeline.
- **Text-Anchored Invariant Feature Learner**: Cross-modal adversarial alignment based on CLIP.
- **Zero-Watermarking Head**: Binding binary messages in the invariant feature space.

### Learnable Auto-Augmentor

Six differentiable modules are cascaded:

| Module | Function | Key Parameters |
|--------|----------|----------------|
| Geometric | Perspective/rotation/scaling transforms | Learnable 3×3 perspective matrix $A$ |
| Photometric | Brightness/contrast/gamma | Learnable $\alpha, \gamma, \beta$ |
| Additive Noise | Sensor noise | Reparameterized $\sigma \cdot z,\ z \sim \mathcal{N}(0,1)$ |
| Filtering | Optical blur/lens smear | Learnable convolution kernel $K$ |
| Compression | JPEG quantization & frequency masking | Smooth quantization + trainable mask $M$ |
| Moiré | Sensor–display interference fringes | Learnable frequency $(f_x, f_y)$ and amplitude $\alpha$ |

The composition formula is:
$$\hat{x} = \mathcal{T}_{\text{aug}}(x;\Theta) = \mathcal{T}_{\text{comp}} \circ \mathcal{T}_{\text{filter}} \circ \mathcal{T}_{\text{add}} \circ \mathcal{T}_{\text{photo}} \circ \mathcal{T}_{\text{geo}} \circ \mathcal{T}_{\text{moire}}(x)$$

Each module is pretrained on 10k samples of its corresponding distortion type using MSE+SSIM loss, then fine-tuned during overall adversarial training.

### Text-Anchored Invariant Feature Learner

- **Feature extractor $f_\theta$**: Frozen CLIP image encoder + trainable invariant feature extractor (3 residual blocks + projection head → 1024-dim).
- **Discriminator $D_\psi$**: 4-layer Transformer (8-head attention, hidden dim 512), receiving image–text feature pairs and judging semantic match.
- **Training objective**: Image $x$ and its augmented version $\hat{x}$ are paired with positive/negative text anchors $T^+/T^-$ to form real/fake pairs; the discriminator loss is $\mathcal{L}_{\text{disc}}$ and the generator loss is $\mathcal{L}_{\text{adv}}$.
- **Augmentor–extractor adversarial dynamic**: The augmentor maximizes $\mathcal{L}_{\text{inv}} - \lambda_{\text{sem}}\mathcal{L}_{\text{sem}}$; the extractor minimizes $\mathcal{L}_{\text{inv}}$, where semantic fidelity is measured by cosine similarity from a frozen ViT.

Three-way alternating updates: ① Update $D_\psi$ to improve paired discrimination → ② Update $\Theta$ to generate stronger distortions → ③ Update $f_\theta$ to align with positive text anchors while resisting distortions.

### Zero-Watermarking Head

- Extract invariant features $\tilde{F} = \Psi(f_\theta(x))$, where $\Psi$ denotes global average pooling followed by linear projection.
- Maintain a learnable reference matrix $C \in \mathbb{R}^{k \times d}$, where the $i$-th row is the direction code for bit $i$.
- Prediction: $\hat{W}_i = \sigma(\tilde{F} \cdot C_i)$
- **Registration phase**: For each image–message pair, optimize $C$ and $\Psi$ (BCE + L2 regularization) with $f_\theta$ frozen.
- **Extraction phase**: For a distorted image $x'$, compute $\tilde{F}' = \Psi(f_\theta(x'))$ and recover the binary message by thresholding at 0.5.

## Key Experimental Results

### Feature Invariance (Cosine Similarity, Original vs. Distorted Images)

| Distortion Type | SimCLR | BYOL | Barlow | VICReg | VIbCReg | **TIACam** |
|-----------------|--------|------|--------|--------|---------|------------|
| Additive Noise | 0.82 | 0.88 | 0.79 | 0.83 | 0.89 | **0.97** |
| Photometric Change | 0.84 | 0.84 | 0.81 | 0.76 | 0.88 | **0.93** |
| Perspective Transform | 0.87 | 0.85 | 0.87 | 0.83 | 0.88 | **0.95** |
| JPEG Compression | 0.79 | 0.80 | 0.87 | 0.81 | 0.73 | **0.98** |
| Moiré Pattern | 0.85 | 0.83 | 0.84 | 0.89 | 0.87 | **0.97** |
| Filtering Blur | 0.88 | 0.88 | 0.89 | 0.87 | 0.88 | **0.98** |
| All Combined | 0.74 | 0.71 | 0.74 | 0.77 | 0.77 | **0.94** |

### Watermark Extraction Accuracy in Real-World Scenarios (Bit Accuracy %)

| Method | Screen 30b | Screen 100b | Print 30b | Print 100b | Screenshot 30b | Screenshot 100b |
|--------|:----------:|:-----------:|:---------:|:----------:|:--------------:|:---------------:|
| HiDDeN | 70.6 | 68.8 | 67.1 | 65.7 | 74.5 | 70.6 |
| PIMoG | 82.3 | 80.1 | 75.7 | 72.3 | 79.7 | 78.6 |
| StegaStamp | 93.8 | 91.2 | 92.2 | 91.3 | 93.7 | 93.9 |
| **TIACam** | **99.1** | **98.2** | **96.6** | **95.1** | **97.4** | **95.2** |

### Ablation Study: Contribution of the Invariant Feature Extractor

| Dataset | CLIP Only | CLIP + TIACam |
|---------|:---------:|:-------------:|
| Visual Genome | 0.78 | **0.92** |
| Flickr | 0.84 | **0.93** |
| MSCOCO | 0.76 | **0.89** |
| ImageNet | 0.82 | **0.93** |

The feature extractor improves cosine similarity by approximately 13–15%, demonstrating that the robustness gains stem from the proposed framework rather than CLIP pretraining alone.

### Feature Discriminability Test

Across 200 image pairs generated from identical captions: only the registered image achieves 100% watermark recovery; the cosine similarity of another image's features averages 0.73, with extraction accuracy dropping to ~84%, indicating that the framework preserves inter-instance visual discriminability alongside invariance.

## Highlights & Insights

- **Tripartite adversarial unified framework**: The augmentor, feature extractor, and discriminator are jointly optimized, representing the first unification of distortion simulation and cross-modal alignment into a single training loop.
- **Fully differentiable augmentation pipeline**: Six differentiable modules cover geometric, photometric, noise, filtering, compression, and Moiré distortions, enabling gradient backpropagation to optimize augmentation strategies.
- **No pixel modification required**: The zero-watermarking paradigm leaves images entirely unaltered; message extraction relies solely on dot products and thresholding in the feature space.
- **Thorough real-world validation**: TIACam substantially outperforms prior SOTA under three physically realistic degradation scenarios: screen recapture, print-and-scan, and screenshot.
- **No localization step required**: The strong robustness of the invariant feature space enables direct watermark extraction from the full image without prior detection of watermark regions.

## Limitations & Future Work

- The area label is classified as object detection but the actual domain is multimedia security/watermarking; the classification requires correction.
- Images are uniformly resized to 128×128; the framework's ability to preserve local features in high-resolution images is not thoroughly discussed.
- Zero-watermark registration requires individually optimizing $C$ and $\Psi$ for each image–message pair; batch registration efficiency may be a deployment bottleneck.
- Experiments are conducted solely on an RTX 4090; inference latency and feasibility of deployment on mobile or embedded devices are not discussed.
- Semantically similar but visually distinct images still achieve ~84% accuracy (ideally this should be lower), raising concerns about cross-instance feature leakage in the feature space.
- Acquiring text anchors (captions) in practice requires additional modules or manual provision.

## Related Work & Insights

| Method | Type | Augmentation Strategy | Feature Source | Camera Robustness |
|--------|------|-----------------------|----------------|:-----------------:|
| HiDDeN | Embedding | Fixed noise layer | Self-trained CNN | Low |
| StegaStamp | Embedding | Handcrafted camera noise layer | Self-trained CNN | Medium-High |
| PIMoG | Embedding | Handcrafted projection noise | Self-trained CNN | Medium |
| InvZW | Zero-watermark | Distortion adversarial | Adversarial training | Medium |
| DINO-based | Zero-watermark | None | Pretrained SSL | Medium |
| **TIACam** | **Zero-watermark** | **Learnable auto-augmentation** | **CLIP + adversarial training** | **High** |

Core distinction: TIACam is the first method to unify a learnable augmentor, cross-modal text anchoring, and zero-watermarking within a single adversarial training framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The tripartite adversarial training framework and differentiable augmentation pipeline design are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic and real-world scenarios, ablation studies, and discriminability tests are relatively comprehensive, though runtime efficiency analysis is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, mathematical derivations are complete, and illustrations are intuitive.
- **Value**: ⭐⭐⭐⭐ — Represents significant progress in camera-robust zero-watermarking, though practical deployment feasibility warrants further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)
- [\[CVPR 2026\] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[ICML 2026\] Rotation-Invariant Spherical Watermarking via Third-Order SO(3) Representation Coupling](../../ICML2026/ai_safety/rotation-invariant_spherical_watermarking_via_third-order_so3_representation_cou.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)

</div>

<!-- RELATED:END -->
