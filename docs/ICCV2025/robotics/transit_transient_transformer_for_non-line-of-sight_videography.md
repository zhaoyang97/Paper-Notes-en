---
title: >-
  [Paper Note] TransiT: Transient Transformer for Non-line-of-sight Videography
description: >-
  [ICCV 2025][Robotics][Non-line-of-sight imaging] TransiT is a novel architecture for real-time NLOS video reconstruction that achieves 64×64 resolution at 10 FPS from sparse fast-scan (16×16, 0.4 ms/point) transient measurements. The system integrates transient compression, inter-frame feature fusion, and a spatiotemporal Transformer, and further proposes an MMD-based transfer learning strategy to bridge the distribution gap between synthetic and real data.
tags:
  - ICCV 2025
  - Robotics
  - Non-line-of-sight imaging
  - NLOS video reconstruction
  - Transformer
  - transient signals
  - transfer learning
date: 2026-05-08
content_hash: 1e8d7a4120d0e522
---

# TransiT: Transient Transformer for Non-line-of-sight Videography

**Conference**: ICCV 2025
**arXiv**: [2503.11328](https://arxiv.org/abs/2503.11328)
**Code**: Coming soon
**Area**: Robotics / Computational Imaging
**Keywords**: Non-line-of-sight imaging, NLOS video reconstruction, Transformer, transient signals, transfer learning

## TL;DR

TransiT is a novel architecture for real-time NLOS video reconstruction that achieves 64×64 resolution at 10 FPS from sparse fast-scan (16×16, 0.4 ms/point) transient measurements. The system integrates transient compression, inter-frame feature fusion, and a spatiotemporal Transformer, and further proposes an MMD-based transfer learning strategy to bridge the distribution gap between synthetic and real data.

## Background & Motivation

### State of the Field

Non-line-of-sight (NLOS) imaging reconstructs occluded scenes by emitting laser pulses onto a relay wall and recording transient signals with single-photon detectors. This technology holds significant value in autonomous driving, disaster rescue, and related applications. Although numerous methods for static NLOS reconstruction exist (e.g., f-k migration, LCT, deep learning approaches), dynamic NLOS video reconstruction still faces a fundamental trade-off between frame rate and reconstruction quality.

### Limitations of Prior Work

**Frame rate vs. quality trade-off**: High-resolution scanning (e.g., 64×64) requires long acquisition times, yielding low frame rates (~2 FPS); reducing scan density or dwell time improves frame rates but degrades SNR and spatial detail.

**Fast-scan distortion**: The minimum response time (~0.4 ms) of galvanometric mirrors causes the laser to follow a continuous path between scan points rather than performing ideal point-by-point illumination, introducing path-integral distortion into the measurements.

**Synthetic-to-real domain gap**: Real measurements are affected by hardware characteristics (laser, galvanometric mirrors, SPAD) that are difficult to simulate accurately, resulting in a significant distribution mismatch with synthetic data.

**Existing learning methods are ill-suited for dynamic scenes**: Prior super-resolution networks (e.g., USM) are designed for static NLOS reconstruction and exhibit limited performance on dynamic scenes.

### Starting Point

NLOS video is fundamentally a video problem. TransiT directly leverages the spatiotemporal modeling capabilities of Video Transformers to process sequences of NLOS transient signals, bypassing the conventional paradigm of first upsampling transients and then applying physics-based reconstruction algorithms.

## Method

### Overall Architecture

The TransiT pipeline consists of three core modules: (1) **Transient Compression**: a linear layer compresses the temporal histogram of each scan point into a 32-dimensional feature vector; (2) **Feature Fusion**: inter-frame feature differences between the current and previous frames are concatenated to enhance temporal dynamics; (3) **Spatiotemporal Transformer**: 8 ViT blocks (8 attention heads) with spatiotemporal positional encoding, followed by a linear projection head that outputs a 64×64 reconstructed image.

### Key Designs

#### 1. Transient Compression

- **Function**: Compresses raw transient signals (potentially hundreds of time bins) into compact low-dimensional feature vectors.
- **Mechanism**: A linear layer directly compresses the temporal axis, mapping high-dimensional transient histograms to $\mathcal{F} \in \mathbb{R}^{16 \times 16 \times 32}$.
- **Design Motivation**: Unlike prior methods (e.g., USM) that first upsample transients to 64×64 before reconstruction, direct compression substantially reduces computational cost. Experiments demonstrate that compression to 32 dimensions incurs no performance degradation, since in fast-scan dynamic scenarios the informative content of transient histograms is concentrated at a small number of peak locations and amplitudes.

#### 2. Feature Fusion

- **Function**: Exploits inter-frame differences to enhance the representation of dynamic scenes.
- **Core Formula**: $\mathcal{F}_{fuse} = \text{concat}(\mathcal{F}_t, \mathcal{F}_t - \mathcal{F}_{t-1})$
- **Design Motivation**: Recovering fine-grained details of dynamic scenes from sparse measurements is highly challenging. Inter-frame differences explicitly highlight motion changes in the scene (newly appearing or disappearing regions), providing the Transformer with explicit temporal change cues.

#### 3. Fast-Scan Distortion Model

- **Function**: Models the deviation between actual measurements and ideal measurements caused by the finite mirror response time.
- **Core Formula**: The ideal transient is defined as $\tau(\bar{\mathbf{x}}_n, t) = \frac{1}{r^4} \iiint_\Omega \rho(\mathbf{x}) \cdot \delta(2\|\bar{\mathbf{x}}_n - \mathbf{x}\| - tc) d\mathbf{x}$. Under fast scanning, the actual measurement is a path integral over neighboring scan positions: $\hat{\tau}(\bar{\mathbf{x}}_n, t) = \frac{1}{\|S\|} \int_S \tau(\bar{\mathbf{x}}_n^{\mathbf{s}}, t) d\mathbf{s}$
- **Design Motivation**: During training, this model converts 64×64 ideal transients into distorted 16×16 transients, enabling TransiT to learn to recover high-resolution reconstructions from distorted inputs.

#### 4. MMD-Based Transfer Learning

- **Function**: Bridges the distribution gap between synthetic training data and real measurements.
- **Core Formula**: $\mathcal{L}_{MMD} = \left\| \frac{1}{n}\sum_{i=1}^n \phi(\mathcal{F}_{real}^i) - \frac{1}{m}\sum_{j=1}^m \phi(\mathcal{F}_{syn}^j) \right\|^2$, with total loss $\mathcal{L}_{total} = \mathcal{L}_{imaging} + \lambda \mathcal{L}_{MMD}$
- **Design Motivation**: Real NLOS systems involve material reflectance properties and hardware noise that are difficult to simulate precisely. The MMD objective is self-supervised — it requires only real measurements without corresponding ground truth — and effective domain alignment is achieved with as few as 200 real frames.

### Loss & Training

Two-stage training procedure:
1. **Stage 1**: Train on a synthetic dataset (100K frames) with MSE loss for 1,000 epochs on 24× A800 GPUs (~24 hours).
2. **Stage 2**: Fine-tune for 100 epochs on 8× A800 GPUs using $\mathcal{L}_{total}$ ($\lambda=0.01$) with 200 real frames (~2 hours).

**Inference**: On a single RTX 3090, each frame requires ~0.6 ms, supporting real-time reconstruction at 10 FPS.

## Key Experimental Results

### Main Results (Synthetic Data, Dynamic Scenes)

| Object | Method | ED↓ | CS↑ | SSIM↑ | PSNR↑ |
|--------|--------|------|------|-------|-------|
| Character | f-k | 0.1286 | 0.7876 | 0.6677 | 17.86 |
| Character | PnP | 0.0923 | 0.8575 | 0.6764 | 20.77 |
| Character | **TransiT** | **0.0520** | **0.9418** | **0.9227** | **25.87** |
| Propeller | f-k | 0.3180 | 0.6854 | 0.1902 | 10.01 |
| Propeller | PnP | 0.2707 | 0.7800 | 0.2818 | 11.39 |
| Propeller | **TransiT** | **0.0904** | **0.9781** | **0.8211** | **20.91** |
| Human | f-k | 0.1136 | 0.6265 | 0.5791 | 19.00 |
| Human | PnP | 0.1018 | 0.6939 | 0.6706 | 19.92 |
| Human | **TransiT** | **0.0415** | **0.9272** | **0.8041** | **29.45** |

### Ablation Study (Static Scenes, No Distortion)

| Object | Method | ED↓ | CS↑ | SSIM↑ | PSNR↑ |
|--------|--------|------|------|-------|-------|
| Character | f-k | 0.1352 | 0.6657 | 0.1224 | 17.37 |
| Character | PnP | 0.0763 | 0.8639 | 0.8852 | 22.34 |
| Character | USM | 0.0548 | 0.9317 | 0.9163 | 25.19 |
| Character | **TransiT** | **0.0531** | **0.9567** | **0.9261** | **25.49** |
| Human | f-k | 0.1525 | 0.4828 | 0.0957 | 16.34 |
| Human | PnP | 0.0754 | 0.6925 | 0.8515 | 22.49 |
| Human | USM | 0.0483 | 0.8641 | 0.9227 | 24.24 |
| Human | **TransiT** | **0.0241** | **0.9652** | **0.9361** | **26.39** |

### Key Findings

- **TransiT substantially outperforms baselines on dynamic scenes**: On the Human sequence, PSNR exceeds PnP by 9.5 dB and SSIM by 0.13.
- **The Propeller scene is most challenging**: Rapid blade rotation induces severe motion blur; TransiT nonetheless recovers sharp contours (PSNR 20.91 vs. PnP 11.39).
- **TransiT achieves top performance on distortion-free static scenes as well**: This demonstrates that TransiT's advantages stem not only from distortion modeling but also from the representational capacity of the Transformer.
- **Compressing transients to 32 dimensions incurs no performance loss**: This validates the hypothesis that transient information is highly concentrated in fast-scan dynamic scenarios.
- **MMD transfer learning is effective**: Fine-tuning with only 200 real frames significantly improves reconstruction quality on real measurements.

## Highlights & Insights

1. **Systematic distortion modeling**: Starting from a physical light transport model, the authors derive a path-integral distortion formula for fast scanning and integrate it into training data generation. This "model distortion → synthetic training → transfer learning → real-world deployment" pipeline generalizes naturally to other computational imaging scenarios.
2. **Minimalist yet effective temporal fusion**: The inter-frame difference concatenation strategy is extremely simple yet effectively captures motion information, offering a compelling alternative to more complex approaches such as optical flow estimation or 3D convolution.
3. **Large-scale synthetic dataset construction**: A synthetic NLOS video dataset comprising 100K frames and 2,000+ motion sequences, covering diverse objects and motion types, constitutes a valuable contribution to the research community.
4. **Real-time inference capability**: Per-frame inference at 0.6 ms far exceeds real-time requirements, providing ample computational headroom for practical deployment.

## Limitations & Future Work

1. **Block artifacts in real-data results**: Likely attributable to material mismatch between training (Lambertian reflectance) and real scenes (which may include specular components).
2. **Fixed upsampling ratio (16×16 → 64×64)**: Support for flexible input/output resolutions remains to be explored.
3. **Limitations of the serpentine scan strategy**: Directional distortions introduced by the scan path between adjacent frames suggest that more intelligent scanning strategies could further improve quality.
4. **Single-SPAD hardware bottleneck**: Although TransiT inference achieves 10 FPS, the hardware scan itself (102 ms/frame) is the actual frame rate bottleneck.
5. **Multi-bounce reflections and occlusions not addressed**: The current method assumes a simple confocal setup; complex occlusion scenarios are not discussed.

## Related Work & Insights

- The LCT framework (O'Toole et al.) established the standard 3D deconvolution paradigm for confocal NLOS; TransiT replaces physics-based reconstruction with a learning-based approach on top of this foundation.
- The PnP method employs plug-and-play denoising to reconstruct video from 16×16 scans at 4 FPS; TransiT surpasses it substantially in both speed and quality.
- The USM transient super-resolution method is designed for static scenes; TransiT's inter-frame fusion enables superior performance in dynamic scenarios.
- The spatiotemporal attention mechanisms of Video Transformers (ViViT, TimeSformer) are successfully transferred to NLOS transient signal processing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying Video Transformers to NLOS video reconstruction is a novel cross-domain contribution; the fast-scan distortion modeling is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluations cover synthetic and real data, dynamic and static scenes, and include comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Physical modeling and method descriptions are clear, though the paper spans a broad interdisciplinary scope (optics + deep learning) that requires background knowledge from readers.
- **Value**: ⭐⭐⭐⭐ — Represents a meaningful advance in NLOS video reconstruction; real-time reconstruction at 10 FPS demonstrates practical deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [\[AAAI 2026\] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](../../AAAI2026/robotics/touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)
- [\[AAAI 2026\] Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](../../AAAI2026/robotics/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)
- [\[CVPR 2026\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](../../CVPR2026/robotics/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)
- [\[ICCV 2025\] Certifiably Optimal Anisotropic Rotation Averaging](certifiably_optimal_anisotropic_rotation_averaging.md)

</div>

<!-- RELATED:END -->
