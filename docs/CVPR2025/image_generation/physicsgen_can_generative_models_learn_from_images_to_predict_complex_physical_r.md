---
title: >-
  [Paper Note] PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?
description: >-
  [CVPR 2025][Image Generation][Physical Simulation] This work introduces the PhysicsGen benchmark, which contains 300,000 image pairs covering three physical simulation tasks (acoustic wave propagation, lens distortion, and rolling/bouncing dynamics). It systematically evaluates the ability of generative models to learn physical relations, revealing that physical relations described by high-order differential equations present a fundamental challenge to current models.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Physical Simulation"
  - "Generative Models"
  - "Image-to-Image"
  - "Benchmark"
  - "Acoustic Wave Propagation"
date: 2026-05-08
content_hash: 0b714694f25a368a
---

# PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?

**Conference**: CVPR 2025  
**arXiv**: [2503.05333](https://arxiv.org/abs/2503.05333)  
**Code**: [Project Page](http://www.physics-gen.org)  
**Area**: Image Generation/Physical Simulation  
**Keywords**: Physical Simulation, Generative Models, Image-to-Image, Benchmark, Acoustic Wave Propagation

## TL;DR

This work introduces the PhysicsGen benchmark, which contains 300,000 image pairs covering three physical simulation tasks (acoustic wave propagation, lens distortion, and rolling/bouncing dynamics). It systematically evaluates the ability of generative models to learn physical relations, revealing that physical relations described by high-order differential equations present a fundamental challenge to current models.

## Background & Motivation

Generative models (GANs, diffusion models) have made remarkable progress in image-to-image translation tasks, but their potential in the field of physical simulation has not been systematically explored. Two core questions remain:

- **Can generative models learn complex physical relations from image input-output pairs?** Physical simulations usually rely on numerical solutions of differential equations. If generative models can learn these mappings, it would be of great significance.
- **How much speedup can be achieved?** Traditional physical simulations are computationally expensive (taking up to hundreds of seconds per sample), whereas generative models can perform inference extremely fast.

However, the generative AI field lacks physics-informed datasets and benchmarks, which limits the training and evaluation of models on complex physical systems. Existing works are either restricted to simple physical attribute prediction or lack systematic comparison across specific domains.

## Method

### Overall Architecture

PhysicsGen provides three simulation tasks with varying physical complexity (100,000 image pairs each) and uniformly evaluates various generative models (Pix2Pix, U-Net, ConvAE, VAE, DDPM, Stable Diffusion, DDBM) across all tasks.

### Key Design 1: Urban Acoustic Wave Propagation Task

- **Function**: To test the model's ability to learn iterative differential equation solving.
- **Mechanism**: Given an aerial view of a city (where buildings are black and open spaces are white), the model predicts the propagation distribution map of an acoustic source in this environment. It includes four subtasks: baseline (no diffraction/reflection), diffraction, reflection, and combination. Acoustic wave propagation is described by high-order partial differential equations.
- **Design Motivation**: Choosing an intuitively understandable physical problem (urban noise propagation) while incorporating iterative solving and high-order physical processes.

### Key Design 2: Lens Distortion Task

- **Function**: To test the model's ability to learn closed-form (non-iterative) physical relations.
- **Mechanism**: Based on the Brown-Conrady distortion model, simulated geometric distortion of images is generated given camera parameters. This is a deterministic, closed-form mapping.
- **Design Motivation**: Providing a physical task that does not require iterative solving as a comparison, thereby validating the variance in model performance under different solving strategies.

### Key Design 3: Rolling/Bouncing Dynamics Task

- **Function**: To test the model's ability to learn time-series prediction and high-order equations of motion.
- **Mechanism**: Simulating the rolling and bouncing motion of a ball on an inclined plane, involving linear and rotational dynamics. Given the current frame, the model predicts the ball's position and rotation in the next frame.
- **Design Motivation**: Equations of motion contain higher-order terms (angular acceleration, collision rebounds), testing the limits of models in handling high-order physical relations.

### Loss & Training

Each baseline model uses its standard training loss. Evaluation uses MAE and weighted MAPE (wMAPE, which specifically penalizes high-amplitude error predictions in low-amplitude regions).

## Key Experimental Results

### Main Results: Acoustic Wave Propagation Task (LoS/NLoS Region MAE)

| Model | Base LoS/NLoS | Diffraction LoS/NLoS | Inference Time / Sample |
|------|-------------|---------------------|------------|
| Physical Simulation | 0.0/0.0 | 0.0/0.0 | 204700 ms |
| Pix2Pix | **1.73/1.19** | 0.91/3.36 | 0.138 ms |
| U-Net | 2.29/1.73 | **0.94/3.27** | 0.138 ms |
| DDPM | 2.42/3.26 | - | 3986 ms |
| Stable Diff | 2.12/1.08 | - | 2971 ms |

### Speedup

| Model Type | Speedup (vs. Physical Simulation) |
|---------|-----------------|
| Pix2Pix / U-Net | **~1,500,000x** |
| DDPM | ~50x |
| Stable Diffusion | ~70x |

### Key Findings

- **Simple Task (Baseline)**: All models can learn reasonable physical relations, with Pix2Pix performing the best.
- **High-order Tasks (Diffraction + Reflection)**: Model performance drops significantly, with errors increasing dramatically, especially in NLoS (Non-Line-of-Sight) regions.
- **Stunning Speedups but Underwhelming Accuracy**: GANs/U-Nets achieve million-fold speedups, but there remains a clear gap in physical correctness.
- Diffusion models (DDPM, SD) do not outperform simple GANs/U-Nets on these types of tasks.

## Highlights & Insights

1. **Value of Systematic Benchmarks**: This work provides the first unified benchmark covering different physical complexities, enabling fair comparisons across different models.
2. **High-Order Physical Relations as the Fundamental Bottleneck**: Models perform well on simple first-order relations, but high-order relations (diffraction, rotational dynamics) present a fundamental limitation for current generative models.
3. **Enormous Speedup Potential**: Even with imperfect accuracy, million-fold speedups are still of practical value in coarse estimation scenarios.

## Limitations & Future Work

- Although the three tasks are diverse, they do not cover all types of physical phenomena.
- All models use the same architecture and training settings, without optimization specifically for physical tasks.
- No physics-informed loss constraints were introduced, which could potentially improve accuracy significantly.
- 3D physical simulation scenarios were not evaluated.

## Related Work & Insights

- **PUGAN, FEM-GAN**: Combining GANs with physical modeling.
- **Tenenbaum's Intuitive Physics Engine**: Pioneering work in machine learning for understanding coarse physical properties.
- Embedding physical constraints into the loss functions of generative models represents a crucial direction for the future.

## Rating

⭐⭐⭐⭐ — The benchmark is well-designed, and the core findings (high-order physical relations as a bottleneck) have important guidance value. The million-fold speedup numbers are striking. However, since the baseline models were not specifically optimized for physics, the conclusions might be somewhat conservative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](../../ICCV2025/image_generation/attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)
- [\[CVPR 2025\] Hiding Images in Diffusion Models by Editing Learned Score Functions](hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)
- [\[CVPR 2026\] LoFA: Learning to Predict Personalized Prior for Fast Adaptation of Visual Generative Models](../../CVPR2026/image_generation/lofa_learning_to_predict_personalized_prior_for_fast_adaptation_of_visual_genera.md)
- [\[CVPR 2025\] IDEA-Bench: How Far are Generative Models from Professional Designing?](idea-bench_how_far_are_generative_models_from_professional_designing.md)

</div>

<!-- RELATED:END -->
