---
title: >-
  [Paper Note] Cycle-Consistent Tuning for Layered Image Decomposition
description: >-
  [CVPR 2026][Image Generation][Image Decomposition] A cycle-consistent fine-tuning framework based on diffusion models is proposed to achieve image layer separation (e.g., logo-object decomposition) by jointly training decomposition and synthesis models. A progressive self-improving data augmentation strategy is introduced to achieve robust decomposition in scenarios with non-linear layer interactions.
tags:
  - CVPR 2026
  - Image Generation
  - Image Decomposition
  - Cycle Consistency
  - Diffusion Models
  - LoRA Fine-tuning
  - In-Context Learning
date: 2026-05-08
content_hash: 6c0fd549de1d4714
---

# Cycle-Consistent Tuning for Layered Image Decomposition

**Conference**: CVPR 2026  
**arXiv**: [2602.20989](https://arxiv.org/abs/2602.20989)  
**Code**: None (Project page available)  
**Area**: Image Decomposition / Image Editing  
**Keywords**: Image Decomposition, Cycle Consistency, Diffusion Models, LoRA Fine-tuning, In-Context Learning

## TL;DR

A cycle-consistent fine-tuning framework based on diffusion models is proposed to achieve image layer separation (e.g., logo-object decomposition) by jointly training decomposition and synthesis models. A progressive self-improving data augmentation strategy is introduced to achieve robust decomposition in scenarios with non-linear layer interactions.

## Background & Motivation

Image decomposition (splitting an image into semantically or physically meaningful layers) is a classic problem in CV and CG:
- Traditional methods (e.g., intrinsic decomposition) are limited to linear interactions (alpha blending) and struggle with non-linear coupling such as lighting, perspective distortion, and material reflections.
- Separating logos from product photos involves global non-linear interactions (shadows, perspective deformation, surface reflections).
- Existing generative editing methods (e.g., ICEdit, Flux-Kontext) can remove logos but struggle to accurately isolate and extract them.
- Decomposition is an ill-posed problem (more unknowns than inputs) requiring additional constraints.

Core Idea: Decomposition is the inverse process of synthesis. By simultaneously learning decomposition and synthesis while imposing cycle-consistency constraints, the determinism of synthesis is used to constrain the uncertainty of decomposition.

## Method

### Overall Architecture

Based on FLUX.1-Fill-dev (a pre-trained diffusion inpainting model), the model is adapted for decomposition tasks via lightweight LoRA fine-tuning. It adopts an In-Context Learning paradigm: the input is a three-panel grid image (Composite / Logo / Clean Object), and the model learns to separate the two layers from the composite.

### Key Designs

1.  **Cycle-Consistent Decomposition-Synthesis Framework**: Simultaneously learns a decomposition function $\mathcal{F}_D(I)=\langle A,B\rangle$ and a synthesis function $\mathcal{F}_C(\langle A,B\rangle)=I$, sharing the same LoRA parameter space. During training, it runs bidirectionally: (1) From $I$, decompose to get $\langle A',B'\rangle$ and re-synthesize back to $I'$; (2) From $A,B$, synthesize to get $I^*$ and re-decompose back to $\langle A^*,B^*\rangle$. The cycle-consistency loss aligns both directions:
    $$\mathcal{L}_{cyc} = \mathbb{E}\left[\|v_\theta(x_{t_1}^I, M_D, t_1, \tau_D) - v_\theta(x_{t_1}^{I^*}, M_D, t_1, \tau_D)\|_2^2\right] + \mathbb{E}\left[\|v_\theta(x_{t_2}^{\langle A,B\rangle}, M_C, t_2, \tau_C) - v_\theta(x_{t_2}^{\langle A',B'\rangle}, M_C, t_2, \tau_C)\|_2^2\right]$$
    This allows decomposition and synthesis to supervise each other, reducing reliance on densely labeled data.

2.  **Progressive Self-Improving Data Collection**: Addresses the scarcity of logo-object decomposition training data. It involves three stages: (a) Seed data—100 manually labeled triplets + GPT-4o assisted training of initial IC-LoRA; (b) Iterative data generation—generating candidate triplets using the current IC-LoRA, filtering high-quality samples with Qwen-VL, and retraining to improve generation stability round-by-round; (c) Cycle model self-improvement—using the cycle-consistent model to perform decomposition-resynthesis cycles on new composites, adding high-quality resynthesized samples to the training set. The selection rate consistently improved from round 1 to round 10.

3.  **Flow Matching-based ICL Training**: Fine-tunes the LoRA parameters of FLUX.1-Fill-dev using a flow matching loss:
    $$\mathcal{L}_{rec} = \mathbb{E}_{x,t}\left[\|v_\theta(x_t, M, t, \tau) - \frac{\partial x_t}{\partial x}\|_2^2\right]$$
    By using masks to distinguish regions to be generated (ones) from preserved regions (zeros), visual ICL with single-input multiple-output is achieved.

### Loss & Training

- Total Loss = Flow matching reconstruction loss + Cycle-consistency loss.
- Decomposition and synthesis share the same LoRA parameters to improve parameter efficiency and stabilize training.
- The self-improving data cycle uses Qwen-VL for automatic filtering combined with simple manual checks.

## Key Experimental Results

### Main Results

| Method | Logo VQAScore↑ | Object VQAScore↑ | VLMScore Mean↑ |
| :--- | :--- | :--- | :--- |
| AssetDropper | 0.42 | — | — |
| ICEdit | 0.31 | 0.31 | 2.55 |
| Flux-Kontext | 0.40 | 0.32 | 3.79 |
| Gemini | 0.42 | 0.32 | 4.20 |
| **Ours** | **0.43** | 0.31 | **4.22** |

Evaluated on 1.5K synthetic test samples, the method achieves the best logo extraction quality and the highest overall score.

### Ablation Study

| Configuration | Effect Description |
| :--- | :--- |
| Round 0 IC-LoRA only | Poor separation quality, severe logo residue |
| + Iterative data generation | Decomposition significantly improved |
| + Cycle-consistency | Significant improvement in logo fidelity |
| + Self-improvement process (Full) | Further improvement in object consistency and realism |

Generalization experiments: Intrinsic decomposition (MAW dataset) Intensity 0.57 / Chromaticity 3.54, close to specialized SOTA methods.

### Key Findings

- Cycle consistency is the largest single contributor to decomposition quality improvement.
- The selection rate of high-quality samples in the self-improving data strategy grows continuously across rounds (from ~20% to >60%).
- In user studies, the method was ranked first in over 50% of cases.
- The framework generalizes to different tasks like intrinsic decomposition and foreground-background decomposition.

## Highlights & Insights

- The insight that "decomposition and synthesis are dual processes" is elegant—using a deterministic process (synthesis) to constrain an ill-posed problem (decomposition).
- The progressive data bootstrapping starts from only 100 seed samples and gradually expands the high-quality training set, demonstrating high data efficiency.
- A single LoRA encodes both decomposition and synthesis capabilities, ensuring parameter efficiency.
- Unlike manipulation-based methods (e.g., Attend-and-Excite), this method requires zero modifications to the base model.

## Limitations & Future Work

- Performance degrades when overlapping elements dominate the frame (e.g., large-scale wall advertisements).
- Currently supports only two-layer decomposition and cannot handle multiple overlapping logos.
- Constrained by the grid paradigm of ICL; architectural adjustments are needed to scale to more layers.
- Training data is biased toward product logos; additional adaptation is required for other types of overlapping elements (e.g., watermarks, stickers).

## Related Work & Insights

- Comparison with AssetDropper: The latter uses reward-driven optimization to extract assets but cannot recover the underlying object.
- Difference from DecompDiffusion: The latter trains independent models for different layers, while ours shares the same model.
- The idea of cycle consistency could be extended to motion/lighting/multimodal decomposition.
- The progressive self-improving data strategy provides valuable reference for data-scarce scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of cycle consistency and self-improving data strategy is novel; the decomposition-synthesis dual perspective is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive quantitative, qualitative, ablation, user study, and generalization experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with smooth logic from motivation to verification.
- Value: ⭐⭐⭐ Academic application (logo extraction) is relatively niche, but the framework's philosophy is generalizable.
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[AAAI 2026\] EchoGen: Cycle-Consistent Learning for Unified Layout-Image Generation and Understanding](../../AAAI2026/image_generation/echogen_cycle-consistent_learning_for_unified_layout-image_generation_and_unders.md)
- [\[CVPR 2026\] Match-and-Fuse: Consistent Generation from Unstructured Image Sets](match-and-fuse_consistent_generation_from_unstructured_image_sets.md)
- [\[ICCV 2025\] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering](../../ICCV2025/image_generation/ouroboros_single-step_diffusion_models_for_cycle-consistent_forward_and_inverse_.md)
- [\[CVPR 2026\] AlignVAR: Towards Globally Consistent Visual Autoregression for Image Super-Resolution](alignvar_towards_globally_consistent_visual_autoregression_for_image_super-resol.md)

<!-- RELATED:END -->
