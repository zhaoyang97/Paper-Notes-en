---
title: >-
  [Paper Note] OLAF-World: Orienting Latent Actions for Video World Modeling
description: >-
  [ICML 2026][Video Generation][Latent Action Learning] OLAF-World learns transferable latent actions through **Sequence-level Control-Effect Alignment** (Seq∆-REPA)—converting unlabeled videos into action-controllable vid…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Latent Action Learning"
  - "Video World Models"
  - "Cross-Context Transfer"
  - "Control Alignment"
date: 2026-05-08
content_hash: 8a3d2f244167f614
---

# OLAF-World: Orienting Latent Actions for Video World Modeling

**Conference**: ICML 2026  
**arXiv**: [2602.10104](https://arxiv.org/abs/2602.10104)  
**Code**: To be confirmed  
**Area**: Video Generation / World Models / Self-Supervised Learning  
**Keywords**: Latent Action Learning, Video World Models, Cross-Context Transfer, Control Alignment

## TL;DR
OLAF-World learns transferable latent actions through **Sequence-level Control-Effect Alignment** (Seq∆-REPA)—converting unlabeled videos into action-controllable video world models to achieve zero-shot action transfer across contexts. It matches the performance of AdaWorld (trained on 2 hours of data) using only 1 minute of labeled data (rotation control precision 0.4680 vs. 0.6420).

## Background & Motivation

**Background**: Video world models require large-scale frame-level action annotations for action control, but such annotations are costly and typically restricted to specific domains. Latent Action Models (LAM) promise to automatically discover control interfaces from unlabeled videos by inferring latent actions through an inverse dynamics encoder and predicting future frames with a forward decoder.

**Limitations of Prior Work**: Although latent action models reconstruct well within a **single video clip**, the learned latent actions **cannot transfer across contexts**. In different scenes, perspectives, or lighting conditions, the same semantic action (e.g., "moving forward") maps to entirely different directions in the latent space. Two failure modes exist:
- **Shortcut Learning**: Encoders tend to encode scene-related visual cues rather than actual controllable factors.
- **Cross-context Non-identifiability**: Since training objectives operate only within a single clip, the latent coordinate system can drift freely between different videos.

**Key Challenge**: Step-wise reconstruction objectives (per-frame prediction loss) fail to provide a shared reference frame across videos, leading to inconsistent representations of the same action in different environments, which undermines transferability.

**Goal**: To learn a structured, cross-context consistent latent action space that supports zero-shot action sequence transfer and rapid adaptation under low-data regimes.

**Key Insight**: Although explicit action annotations are unavailable, the **semantic effects of actions are observable**. The same underlying action should produce similar visual-semantic changes across different contexts. By utilizing a frozen self-supervised video encoder as a global reference, latent action sequences can be aligned with their effect directions (net change in features).

**Core Idea**: Align latent actions with visual-semantic changes captured by a self-supervised encoder through sequence-level control-effect alignment, establishing a unified latent coordinate system across different contexts.

## Method

### Overall Architecture
A two-stage pipeline:
1. **Latent Action Learning Stage (Seq∆-REPA)**: Train an inverse dynamics model on unlabeled videos to learn a transferable latent action space, ensuring the same semantic action maps to consistent directions in the latent space across environments.
2. **World Model Pre-training Stage**: Use the learned latent actions as a unified control interface to condition a pre-trained Video Diffusion Transformer.

### Key Designs

1. **Seq∆-REPA Sequence-level Effect Alignment**:
    - **Function**: Forces the latent coordinate system to be consistent across contexts by aligning latent action sequences with the feature change directions of a frozen video encoder.
    - **Mechanism**: For a video clip $x_{0:K}$, a frozen self-supervised encoder (V-JEPA ViT) extracts features $s_i \in \mathbb{R}^D$ frame-by-frame. The **effect direction** is calculated as the average of all frame feature differences: $\tau^* = \frac{1}{K} \sum_{i=0}^{K-1} (s_{i+1} - s_i)$. Simultaneously, the latent action encoder infers a sequence of latent actions $z_{0:K-1}$, which are aggregated and projected into the feature space via a trainable MLP: $\bar{z} = \frac{1}{K} \sum z_i$, $u = h_\psi(\bar{z})$. Alignment is performed using cosine similarity: $\mathcal{L}^{\text{Seq}\Delta\text{-REPA}}_\psi = 1 - \langle \text{norm}(u), \text{norm}(\tau^*) \rangle$.
    - **Design Motivation**: (1) **Temporal differencing naturally suppresses static details**, making the model robust to scene appearance changes by focusing on feature variations; (2) **Sequence-level rather than step-wise** aggregation avoids the non-identifiability of step-wise reconstruction objectives; (3) **Frozen encoder serves as a global reference**, ensuring all videos are aligned to the same semantic space.

2. **Hybrid Training Objective**:
    - **Function**: Combines the $\beta$-VAE reconstruction objective with the Seq∆-REPA alignment objective.
    - **Mechanism**: $\mathcal{L}_{\text{LAM}} = \mathcal{L}^{\text{VAE}}_{\theta, \phi} + \lambda \mathcal{L}^{\text{Seq}\Delta\text{-REPA}}_\psi$, where $\lambda = 0.02$. This ensures latent actions effectively reconstruct the next frame (encoding useful dynamics) while satisfying cross-context alignment constraints.
    - **Design Motivation**: Reconstruction objectives alone are prone to shortcut learning. With the alignment objective, the encoder is forced to learn action-related features rather than relying on scene-specific cues.

3. **Context-Specific Adaptation Strategy**:
    - **Function**: Rapidly adapts to a new action space when a small amount of labeled data is available in a target environment, while retaining the global alignment learned from unlabeled data.
    - **Mechanism**: A lightweight action adapter $A_\eta$ is learned to map environmental actions $a_t$ to the pre-trained latent space: $\hat{z}_t = A_\eta(a_t)$. For discrete action sets, an embedding table $E \in \mathbb{R}^{|A| \times d_z}$ is initialized (where each entry's initial value is the latent action prototype inferred from the frozen LAM using labeled data). Subsequently, only the adapter and LoRA layers need fine-tuning.
    - **Design Motivation**: Fully leverages the alignment properties learned from unlabeled data, enabling high-fidelity action tracking even with only 1 minute of labeled data.

## Key Experimental Results

### Main Results: Latent Space Structural Diagnosis (Linear Probe F1)

| Method | 1st→1st | 1st→3rd | 3rd→3rd | 3rd→1st |
|------|---------|---------|---------|---------|
| AdaWorld | 0.6004 | 0.4820 | 0.4827 | 0.4999 |
| **OLAF-World** | **0.8138** | **0.6250** | **0.8256** | **0.5904** |

In cross-domain evaluations (grey columns), 1st→3rd improved by +30% and 3rd→3rd by +71%, indicating that the latent coordinate system is truly aligned across contexts.

### Ablation Study: Data-Efficient Adaptation

| Method | Adaptation Videos | Image Quality ↑ | Trans RPE ↓ | Rot RPE ↓ |
|------|---------|---------------|------------|-----------|
| DirectAct | 0 | 0.7213 | 0.0703 | 1.4311 |
| AdaWorld | 0 | 0.5600 | 0.0470 | 1.0844 |
| Ours | 0 | 0.5400 | 0.0387 | 0.8773 |
| AdaWorld | 1 min | 0.5623 | 0.0318 | 0.6420 |
| **Ours** | 1 min | 0.5726 | 0.0284 | **0.4680** |
| AdaWorld | 2 hours | 0.6177 | 0.0263 | 0.3834 |
| **Ours** | 2 hours | 0.6312 | 0.0230 | **0.3785** |

OLAF-World achieves the best RPE metrics across zero, 1-minute, and 2-hour data scenarios; with 1 minute of data, rotation control precision is 27% higher than AdaWorld.

### Key Findings
- **Cross-domain Linear Separability**: While AdaWorld's F1 saturates at ~0.48, this method maintains ~0.83, significantly improving cross-context consistency.
- **Action Prototype Similarity**: Visualizing the cosine similarity matrix of action prototypes across scenes reveals a clear diagonal dominance—representations of the same action across different views are highly consistent, while different actions are well-separated.
- **Zero-shot Transfer Quality**: Qualitative comparisons show that AdaWorld often suffers from "temporal bleaching," disappearance of subjects, or trajectory drift during cross-context action transfer, whereas this method stably maintains scene consistency and action accuracy.

## Highlights & Insights
- **"Effect" rather than "Action Labels" as Alignment Reference**: Even without explicit action labels, action effects (feature change directions) can be automatically extracted via self-supervised encoders, bypassing the bottleneck of manual annotation. This can be generalized to other scenarios requiring consistent conceptual representations.
- **Sequence-level Aggregation Resolves Non-identifiability**: By aligning latent actions and effect directions across an entire clip (rather than step-wise), the cross-context non-identifiability problem is theoretically addressed.
- **Dual Role of Frozen Encoder**: Acting as both a feature extractor and an alignment reference, it stabilizes training and provides strong universality through self-supervised pre-training—eliminating the need to re-train the reference encoder for every type of video content.

## Limitations & Future Work
- Computational Overhead: Running two encoders (latent action + frozen video) increases computational costs relative to traditional inverse dynamics models.
- Dependence on Self-Supervised Encoders: The quality of alignment depends heavily on the frozen encoder; performance may degrade if the encoder handles certain video types poorly (e.g., extreme lighting, rare actions).
- Discrete vs. Continuous Actions: Experiments focused primarily on discrete actions (8-direction control); performance in continuous action control scenarios remains to be evaluated.
- Long Video Generation: Current world models are conditioned on relatively short sequences (97 frames); maintaining consistency under cumulative error in long videos requires further exploration.

## Related Work & Insights
- **vs. AdaWorld and other LAM Methods**: Previous methods used step-wise reconstruction or other pixel/feature-based losses, but operated within single clips without establishing a shared cross-context coordinate system. This work essentially improves problem identifiability through sequence-level alignment and global constraints from a frozen reference encoder.
- **vs. Representation Alignment Methods**: While there are feature-feature alignment works in video generation, this paper innovates with a **control-effect alignment** paradigm—directly targeting downstream controllability rather than just internal representation improvement.
- **vs. Reinforcement Learning and Imitation Learning**: Compared to methods requiring explicit rewards or demonstration sequences, this work automatically learns control interfaces from unlabeled videos, offering higher generalizability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Seq∆-REPA is a fundamental innovation for latent action learning. Sequence-level alignment plus a frozen reference encoder solves the long-standing cross-context non-identifiability problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Covers structural diagnosis, zero-shot transfer, and adaptation efficiency with clear comparisons and complete data.
- Writing Quality: ⭐⭐⭐⭐⭐  Problem statement is clear, the motivational chain is tight, and the methodology is concise and powerful.
- Value: ⭐⭐⭐⭐⭐  Enables the conversion of unlabeled videos into controllable world models, possessing significant application potential and providing new perspectives for cross-context alignment in representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](../../CVPR2026/video_generation/drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)
- [\[ICML 2026\] World-R1: Reinforcing 3D Constraints for Text-to-Video Generation](world-r1_reinforcing_3d_constraints_for_text-to-video_generation.md)
- [\[ICML 2026\] WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)
- [\[CVPR 2026\] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens](../../CVPR2026/video_generation/a_frame_is_worth_one_token_efficient_generative_world_modeling_with_delta_tokens.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/video_generation/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)

</div>

<!-- RELATED:END -->
