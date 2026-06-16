---
title: >-
  [Paper Note] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis
description: >-
  [ICCV 2025][Human-Object Interaction] This paper proposes SyncDiff, a unified multi-body human-object interaction (HOI) motion synthesis framework that achieves precise multi-body synchronization via alignment scores and…
tags:
  - "ICCV 2025"
  - "Human-Object Interaction"
  - "Motion Synthesis"
  - "Diffusion Models"
  - "Multi-Body Synchronization"
  - "Frequency Decomposition"
date: 2026-05-08
content_hash: 754b6f3a4001455b
---

# SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis

**Conference**: ICCV 2025
**arXiv**: [2412.20104](https://arxiv.org/abs/2412.20104)  
**Code**: [https://syncdiff.github.io/](https://syncdiff.github.io/)  
**Area**: Others
**Keywords**: Human-Object Interaction, Motion Synthesis, Diffusion Models, Multi-Body Synchronization, Frequency Decomposition

## TL;DR

This paper proposes SyncDiff, a unified multi-body human-object interaction (HOI) motion synthesis framework that achieves precise multi-body synchronization via alignment scores and an explicit synchronization strategy, while introducing frequency-domain decomposition to model high-frequency interaction semantics.

## Background & Motivation

Existing HOI motion synthesis methods are typically limited to specific interaction configurations (e.g., single-hand–single-object, two-hand–single-object) and lack the ability to handle generalized multi-body scenarios composed of arbitrary numbers of humans, hands, and objects. Two fundamental challenges arise in multi-body settings:

**High synchronization requirements**: Motions across different bodies are highly correlated and mutually influential. Naively concatenating all motions into a high-dimensional representation and modeling them with a single diffusion model can only implicitly capture inter-body dependencies, making it difficult to ensure precise alignment (e.g., contact consistency, interpenetration avoidance).

**High-frequency interactions are overwhelmed**: High-frequency, small-amplitude interactions between objects (e.g., periodic friction between a brush and a teapot during scrubbing) tend to be dominated by large-scale, low-frequency motions (e.g., global object displacement and contact), causing the generated motions to lack semantically critical details.

To address these issues, the authors argue for dedicated alignment scores to facilitate motion synchronization and frequency-domain decomposition to explicitly model high-frequency motion components.

## Method

### Overall Architecture

SyncDiff defines the diffusion process over a graph model where nodes represent individual motions (human/hand/object) and edges represent pairwise relative motions between bodies. The model operates on a higher-order representation comprising all individual and relative motions, making it the first unified multi-body HOI synthesis framework capable of handling an arbitrary number of bodies. The framework consists of three core designs: frequency decomposition, alignment scores (during training), and explicit synchronization (during inference).

### Key Designs

1. **Motion Representation**: For articulated bodies (humans/hands), motion is represented as 3D joint positions $x_h \in \mathbb{R}^{N \times 3D}$; for rigid bodies (objects), motion is represented as translation + quaternion $x_o \in \mathbb{R}^{N \times 7}$. Relative motion $x_{b_2 \to b_1}$ is computed via coordinate transformation, expressing $b_2$'s motion in $b_1$'s local frame. All individual and relative motions are concatenated into a higher-order representation $x \in \mathbb{R}^{N \times D_{sum}}$.

2. **Frequency Decomposition**: FFT is applied to decompose motion into low-frequency ($x_{dc}$) and high-frequency ($x_{ac}$) components, with a cutoff frequency of $L=16$ to discard high-frequency noise. The low-frequency component is supervised in the time domain, while the high-frequency component is represented in the frequency domain as Fourier coefficients $x_F$; both are fed into a Transformer backbone for denoising separately. The final output is reconstructed as $\hat{x} = \hat{x}_{dc} + \hat{x}_{ac}$. The design motivation is to prevent high-frequency semantic components from being overwhelmed by large low-frequency motions.

3. **Alignment Scores and Alignment Loss**: One of the core innovations. Analogous to data sample scores guiding denoising, alignment scores are used to promote consistency between individual motions and relative motions on each edge of the graph. Specifically, the alignment loss is defined as:

$$\mathcal{L}_{align} = \sum_{j_1 \neq j_2} \|\hat{x}_{o_{j_2} \to o_{j_1}} - \text{rel}(\hat{x}_{o_{j_1}}, \hat{x}_{o_{j_2}})\|_2^2 + \sum_{i,j} \|\hat{x}_{h_i \to o_j} - \text{rel}(\hat{x}_{o_j}, \hat{x}_{h_i})\|_2^2$$

This requires the model-predicted relative motion to be consistent with the relative motion computed from individual motions, which is mathematically equivalent to optimizing the negative log-likelihood of the alignment distribution.

4. **Explicit Synchronization (Inference)**: A synchronization operation is performed every $s=50$ steps (total steps $T=1000$) by maximizing the joint distribution of the data sample likelihood and the alignment likelihood. This yields a closed-form synchronization update formula, where the synchronized individual motion is a weighted average of the diffusion-predicted mean and the value derived from relative motions, with weights controlled by hyperparameter $\bar{\lambda}$ and noise scale $\sigma$. The authors prove this formula is equivalent to maximum likelihood sampling from a new Gaussian distribution.

### Loss & Training

The total loss is a weighted sum of four terms:

$$\mathcal{L} = \lambda_{dc}\mathcal{L}_{dc} + \lambda_{ac}\mathcal{L}_{ac} + \lambda_{align}\mathcal{L}_{align} + \lambda_{norm}\mathcal{L}_{norm}$$

- $\mathcal{L}_{dc}$, $\mathcal{L}_{ac}$: supervise reconstruction of low-frequency and high-frequency components, respectively
- $\mathcal{L}_{align}$: alignment loss to promote multi-body synchronization
- $\mathcal{L}_{norm}$: constrains rigid-body rotation quaternion norms to remain close to 1

The model adopts a latent diffusion paradigm, encoding action/object labels with CLIP and object geometry with BPS.

## Key Experimental Results

### Main Results

Evaluated on 5 datasets (TACO, CORE4D, GRAB, OAKINK2, BEHAVE), covering both hand-object and human-object interaction scenarios.

| Method | CSIoU(%) ↑ | IV(cm³) ↓ | FID ↓ | RA(%) ↑ |
|--------|-----------|----------|-------|---------|
| MACS | 56.81 | 13.18 | 10.56 | 58.40 |
| DiffH2O | 62.29 | 10.25 | 4.34 | 61.40 |
| **SyncDiff** | **73.00** | **6.64** | **2.70** | **73.28** |

*TACO Test1 results. SyncDiff substantially outperforms baselines in both contact quality and semantic accuracy.*

| Method | CRR(%) ↑ | FID ↓ | RA(%) ↑ |
|--------|---------|-------|---------|
| OMOMO | 5.31 | 13.22 | 68.02 |
| CG-HOI | 5.74 | 12.16 | 70.05 |
| **SyncDiff** | **6.15** | **6.45** | **92.89** |

*CORE4D Test1 results. Action semantic recognition accuracy exceeds the prior SOTA by approximately 23 percentage points.*

### Ablation Study

| Configuration | CSIoU(%) | FID | RA(%) |
|--------------|---------|-----|-------|
| SyncDiff (full) | 73.00 | 2.70 | 73.28 |
| w/o all | 62.96 | 10.63 | 57.39 |
| w/o frequency decomposition | 68.86 | 6.44 | 56.60 |
| w/o $\mathcal{L}_{align}$ + explicit sync | 63.74 | 4.13 | 64.47 |
| w/o $\mathcal{L}_{align}$ | 70.39 | 2.90 | 67.82 |
| w/o explicit sync | 65.51 | 3.39 | 67.27 |

*TACO Test1 ablation results. Removing each core component leads to varying degrees of performance degradation, with explicit synchronization having the largest impact.*

### Key Findings

- Frequency decomposition is particularly critical for scenarios requiring periodic relative motion (e.g., scrubbing actions); without it, objects tend to remain relatively static.
- Introducing frequency decomposition alone without synchronization mechanisms can paradoxically degrade contact metrics, indicating that high-frequency modeling places higher demands on synchronization.
- A user study (150 participants, 10 dataset splits) demonstrates that SyncDiff's advantage in multi-body scenarios becomes increasingly pronounced as the number of bodies grows.

## Highlights & Insights

- This work is the first to formalize multi-body HOI synthesis as a motion synchronization problem on a graph model and to derive a theoretical foundation for alignment scores.
- The explicit synchronization strategy is supported by rigorous mathematical derivation (equivalent to maximum likelihood sampling) rather than heuristic design.
- Frequency-domain decomposition is an elegant and effective solution to the pervasive problem of high-frequency semantics being masked by low-frequency motions.

## Limitations & Future Work

- The explicit synchronization step introduces additional computational overhead (executed every 50 steps).
- Validation is currently limited to mocap datasets, without addressing online deployment in real-world scenarios.
- The relative motion representation is restricted to rigid bodies as reference frames; relative representations between articulated bodies are omitted.

## Related Work & Insights

- Compared to methods such as CG-HOI, SyncDiff does not require predefined contact guidance.
- The frequency decomposition strategy is inspired by GID, extending it from scene generation to multi-body motion synthesis.
- The combination of graph models and diffusion models offers transferable insights for other generative tasks requiring multi-body coordination, such as multi-person dance and robotic collaboration.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer](layertracer_cognitive-aligned_layered_svg_synthesis_via_diffusion_transformer.md)
- [\[ICCV 2025\] Jigsaw++: Imagining Complete Shape Priors for Object Reassembly](jigsaw_imagining_complete_shape_priors_for_object_reassembly.md)
- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [\[AAAI 2026\] Human Cognitive Biases in Explanation-based Interaction: The Case of Within and Between Session Order Effect](../../AAAI2026/others/human_cognitive_biases_in_explanation-based_interaction_the_case_of_within_and_b.md)
- [\[ICML 2026\] Mapping Human Anti-collusion Mechanisms to Multi-agent AI Systems](../../ICML2026/others/mapping_human_anti-collusion_mechanisms_to_multi-agent_ai_systems.md)

</div>

<!-- RELATED:END -->
