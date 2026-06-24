---
title: >-
  [Paper Note] Quaffure: Real-Time Quasi-Static Neural Hair Simulation
description: >-
  [CVPR 2025][Human Understanding][Hair Simulation] Quaffure proposes the first physical self-supervised real-time quasi-static hair simulation method. By decomposing hair deformation into a rigid pose transformation and a learned correction, it trains a CNN decoder using an improved Cosserat elastic energy as a self-supervised loss, predicting physically plausible hair draping effects for various hairstyles, body shapes, and poses in just a few milliseconds on consumer-grade h…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Hair Simulation"
  - "Quasi-Static"
  - "Self-Supervised Learning"
  - "Cosserat Rod Model"
  - "Real-Time Inference"
date: 2026-05-08
content_hash: 9233686b66800a5d
---

# Quaffure: Real-Time Quasi-Static Neural Hair Simulation

**Conference**: CVPR 2025  
**arXiv**: [2412.10061](https://arxiv.org/abs/2412.10061)  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: Hair Simulation, Quasi-Static, Self-Supervised Learning, Cosserat Rod Model, Real-Time Inference

## TL;DR

Quaffure proposes the first physical self-supervised real-time quasi-static hair simulation method. By decomposing hair deformation into a rigid pose transformation and a learned correction, it trains a CNN decoder using an improved Cosserat elastic energy as a self-supervised loss, predicting physically plausible hair draping effects for various hairstyles, body shapes, and poses in just a few milliseconds on consumer-grade hardware.

## Background & Motivation

**Background**: Realistic hair movement is a key component of high-quality digital humans. Physical simulation produces high-quality results but is computationally expensive (a GPU-accelerated XPBD method still requires 63 seconds for a single quasi-static simulation), while data-driven methods (such as GroomGen) require generating a large amount of simulation training data beforehand.

**Limitations of Prior Work**: (1) Physical simulation is unsuitable for real-time applications; (2) Data-driven methods face data generation bottlenecks due to the lack of public hair simulation datasets; (3) GroomGen can only handle a single hairstyle and simple variations (gravity direction), failing to generalize to complex poses and body shapes; (4) The full Cosserat rod model converges extremely slowly during NN training.

**Key Challenge**: The conflict between real-time performance and physical realism—fast methods lack realism, while realistic methods are too slow.

**Goal**: A neural network approach that requires no simulation training data, predicts quasi-static hair draping within 3ms, and generalizes to different hairstyles, body shapes, and poses.

**Key Insight**: Decomposing hair deformation into an accurately computable rigid pose transformation and a physical correction that needs to be learned. The neural network is trained only for the latter using physical energy as a self-supervised loss.

**Core Idea**: Training a groom deformation decoder with an improved Cosserat elastic potential energy (optimizing only position displacements) as a self-supervised loss, completely avoiding training data generation.

## Method

### Overall Architecture

The input consists of a hairstyle latent space code (16-dimensional), body shape parameters (10-dimensional), and bone poses (81-dimensional), outputting naturally draped hair. Pipeline: (1) A hairstyle autoencoder encodes different hairstyles into latent vectors; (2) A Groom Deformation Decoder predicts the deformation field. Finally: hair = rigid transformation + learned correction. All geometries are represented as 64x64 2D texture maps (mapping hair strand roots to the scalp UV).

### Key Designs

1. **Hair Deformation Decomposition Strategy (Posed Hair + Learned Corrector)**:

    - **Function**: Decomposing complex hair simulation into simple rigid transformations and learnable corrections.
    - **Mechanism**: Each strand is attached to a scalp triangle, undergoing a rigid transformation with head pose changes to yield the posed hair. The deformation decoder predicts the residual, resulting in hair = posed + deformation.
    - **Design Motivation**: Resolving large rigid motions analytically so that the NN only learns smaller physical deformations, thereby reducing the learning difficulty.

2. **Improved Cosserat Elastic Potential Energy**:

    - **Function**: Assisting as a self-supervised loss to guide the network in predicting physically plausible deformations.
    - **Mechanism**: The full Cosserat model requires optimizing both positions and orientation quaternions, causing extremely slow training. The simplified version only optimizes position displacements, replacing quaternions with edge directions after the rigid transformation. A Hookean stretching potential energy is additionally introduced to control stretching independently, since the simplified Cosserat model penalizes both stretching and shearing, leading to excessive rigidity.
    - **Design Motivation**: Training on the full Cosserat model is infeasible; mass-spring models fail to preserve the curliness and volume of curly hair.

3. **SPH-based Self-Collision Handling**:

    - **Function**: Preventing inter-penetration between hair strands.
    - **Mechanism**: Local density anomalies are detected using SPH density estimation. When the density exceeds a reference threshold, a cubic penalty pushes vertices apart. Body collision handling is similar, using a cubic penalty with signed distance to maintain a minimum distance.
    - **Design Motivation**: SPH converts collision detection into field queries for highly efficient GPU parallelization, while the cubic penalty ensures smooth gradients.

### Loss & Training

The total loss consists of 5 components: elastic potential energy (stretching + modified Cosserat), gravitational potential energy, body collision, self-collision, and pose regularization. Pose regularization encourages smooth deformation across continuous frames. The entire process is self-supervised, requiring no simulation data. A 2D convolutional network architecture is adopted, implemented in PyTorch, and trained on an RTX A6000.

## Key Experimental Results

### Main Results

| Method | Inference Time | Body Penetration Rate | Length Preservation | Orientation Preservation |
|------|---------|-----------|----------|----------|
| Adam (Optimization) | 179.38s | 0.22% | 103.53 | 76.15 |
| L-BFGS | 281.18s | 0.22% | 89.53 | 70.22 |
| XPBD (GPU) | 63.26s | 0.01% | 57.96 | 18.10 |
| GroomGen | 0.00249s | 0.39% | 1319.74 | 1281.96 |
| **Quaffure** | 0.00286s | **0.26%** | **175.42** | **286.13** |

### Ablation Study

| Configuration | Effect |
|------|------|
| Improved Cosserat | Curly hair preserves curliness, complete model |
| Mass-spring Replacement | Curly hair straightens and volume collapses, failing to maintain complex hairstyles |
| High Stiffness | Better shape preservation, artistic control |
| Low Stiffness | More gravity influence, artistic control |

### Key Findings

- Achieves 7.5x better length preservation and 4.5x better orientation preservation than GroomGen, with fewer penetrations.
- Constant inference time of ~3ms, independent of complexity.
- Capable of predicting 1000 hairstyles simultaneously in just 0.3 seconds, demonstrating linear scaling.
- The improved Cosserat model yields a qualitative improvement over mass-spring models in preserving the shape of curly hair.
- Excellent temporal stability; hair slides smoothly over shoulders without jittering.

## Highlights & Insights

- **Fully Self-Supervised Physical Simulation Learning**: The first to apply physical self-supervision to hair, bypassing the need for simulation data, tools, or expert knowledge. This approach can be extended to other flexible objects like cloth and ropes.
- **Practical Value of Improved Cosserat**: Formulates an approximation that requires only position offsets without orientation optimization, accelerating training by orders of magnitude. This simplification holds broad engineering value.
- **Design Philosophy of Deformation Decomposition**: Divides the problem into "parts that can be analytically computed" and "parts that need to be learned," allowing the NN to focus solely on learning residuals.

## Limitations & Future Work

- Only handles quasi-statics, unable to simulate dynamic motion (e.g., fluttering during running).
- Uses the same set of physical parameters for all hairstyles, making it impossible to differentiate between different hair textures.
- Evaluated only on an internal dataset, lacking a public standardized benchmark.
- Limited generalization to extreme body shape variations.

## Related Work & Insights

- **vs GroomGen**: Supervised training requires simulation data and only supports a single hairstyle. Quaffure is self-supervised, supports multiple hairstyles, and significantly leads in quantitative metrics.
- **vs XPBD**: XPBD provides the highest quality but requires 63 seconds. Quaffure achieves a 22,000x speedup at a minor cost in quality.
- **vs DrapNet**: Extension of clothing self-supervision to the hair domain, where hair possesses higher degrees of freedom and more complex self-collisions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to apply physical self-supervision to hair; the improved Cosserat model is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough ablation studies, comprehensive comparisons with multiple baselines, but lacks public benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear physical formulas and abundant diagrams.
- Value: ⭐⭐⭐⭐⭐ 3ms inference allows direct deployment in games and VR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VI3NR: Variance Informed Initialization for Implicit Neural Representations](vi3nr_variance_informed_initialization_for_implicit_neural_representations.md)
- [\[CVPR 2025\] Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios](remote_photoplethysmography_in_real-world_and_extreme_lighting_scenarios.md)
- [\[CVPR 2026\] Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation](../../CVPR2026/human_understanding/avatar_forcing_real-time_interactive_head_avatar_generation_for_natural_conversa.md)
- [\[CVPR 2026\] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](../../CVPR2026/human_understanding/remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)
- [\[ICML 2026\] DiscoForcing: A Unified Framework for Real-Time Audio-Driven Character Control with Diffusion Forcing](../../ICML2026/human_understanding/discoforcing_a_unified_framework_for_real-time_audio-driven_character_control_wi.md)

</div>

<!-- RELATED:END -->
