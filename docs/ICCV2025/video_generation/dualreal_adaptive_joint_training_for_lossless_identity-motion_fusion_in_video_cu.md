---
title: >-
  [Paper Note] DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization
description: >-
  [ICCV 2025][Video Generation][Video Customized Generation] DualReal is the first framework to propose adaptive joint training for identity and motion…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Video Customized Generation"
  - "Identity-Motion Fusion"
  - "Joint Training"
  - "Diffusion Transformer"
  - "Adaptive Control"
date: 2026-05-08
content_hash: 52aa5468726ed52a
---

# DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization

**Conference**: ICCV 2025
**arXiv**: [2505.02192](https://arxiv.org/abs/2505.02192)
**Code**: [https://wenc-k.github.io/dualreal-customization](https://wenc-k.github.io/dualreal-customization) (Project Page)
**Area**: Video Generation
**Keywords**: Video Customized Generation, Identity-Motion Fusion, Joint Training, Diffusion Transformer, Adaptive Control

## TL;DR

DualReal is the first framework to propose adaptive joint training for identity and motion, achieving lossless fusion along both dimensions via Dual-aware Adaptation and a StageBlender Controller, with average gains of 21.7% and 31.8% on CLIP-I and DINO-I metrics.

## Background & Motivation

Video Customized Generation is an important frontier in video generation, aiming to produce videos that simultaneously preserve identity consistency and motion consistency from user-provided reference images and motion videos. Existing methods such as VideoBooth, AnimateDiff, and DreamVideo predominantly adopt an **isolated customized paradigm**: identity and motion adapters are trained separately and their parameters are directly merged at inference time.

However, this isolated paradigm entirely neglects the **intrinsic mutual constraints and collaborative dependencies** between identity and motion. Specifically:

**Identity-Motion Conflict**: The addition of motion priors causes irreversible degradation of identity fidelity. Figure 2 of the paper demonstrates that, with a fixed number of identity training steps, identity quality deteriorates continuously as motion training steps increase.

**Inconsistent Optimal Step Counts**: The number of motion training steps that minimizes identity degradation differs across subjects, and no universally optimal step count exists.

**Violation of the Natural Denoising Process**: The denoising process inherently and dynamically adjusts its focus between spatial (identity) and temporal (motion) information at different steps, whereas isolated training enforces uniform sampling across all steps, leading to conflicting optimization directions.

**Key Challenge**: The parameter distributions of the two dimensions differ substantially across training data, and unconstrained joint updates cause destructive cross-dimensional interference. For instance, fine-tuning a motion adapter on static images during the motion training phase irreversibly impairs its capacity for dynamic generation.

**Core Idea**: Rather than training in isolation and merging afterward, DualReal builds mutual dependencies between the two dimensions during training itself — using a **frozen counterpart's prior to guide the current training dimension** — and adaptively allocates weights across dimensions by leveraging the **functional specialization of denoising stages and network depth**.

## Method

### Overall Architecture

DualReal is built upon the DiT (Diffusion Transformer) architecture, inserting a dual-adapter module (DA-Block) — comprising an identity adapter and a motion adapter — into each DiT block. The overall pipeline is as follows:

1. At each training step, the training mode $Z \in \{0, 1\}$ (identity or motion) is dynamically switched.
2. Noise is injected into the corresponding data and fed into the DiT together with text embeddings.
3. The StageBlender Controller computes the weight coefficient $\omega_i$ for each adapter within each DA-Block, conditioned on the current denoising timestep and fused features.
4. Within the DA-Block, the adapter corresponding to the active training dimension is updated while the frozen adapter provides prior guidance.
5. A gradient mask prevents parameter updates in the non-training dimension, avoiding knowledge leakage.

### Key Designs

1. **Dual-aware Adaptation**:

    - **Function**: Dynamically alternates between identity and motion training steps, using the frozen dimension's prior to guide learning in the active dimension.
    - **Mechanism**: Both adapters use a bottleneck structure. The identity adapter directly maps input features, while the motion adapter additionally maps reference image embeddings through a conditional linear layer. Outputs are fused via weighted summation:
        - $\hat{f}_{out}^i = \omega_i \cdot f_{mo}^i + (1 - \omega_i) \cdot f_{id}^i + f_{dit}^i$
    - **Regularization**: Parameter isolation is enforced via a gradient mask $M$. When training motion ($Z=1$), only motion adapter parameters $\theta_m$ are updated; when training identity ($Z=0$), only $\theta_i$ is updated:
        - $\theta^{(t+1)} = \theta^{(t)} - M \odot \nabla_\theta \mathcal{L}$
        - $M = Z \cdot M_m + (1-Z) \cdot M_i$
    - **Design Motivation**: The frozen adapter provides intrinsic regularization during the forward pass, constraining over-fitting within the active dimension; the gradient mask completely eliminates cross-dimensional interference. This "reference-guided learning" mechanism enables the two dimensions to collaborate within a shared parameter space.

2. **StageBlender Controller**:

    - **Function**: Adaptively generates weight coefficients for identity and motion adapters in each DA-Block, conditioned on the denoising timestep and DiT layer depth.
    - **Mechanism**:
        - Input features are pooled and modulated via Adaptive LayerNorm conditioned on the timestep embedding: $f'' = \text{MLP}(\text{LN}(f')) * \alpha + \beta$
        - Gated fusion: $f_g = f'' + \gamma \cdot f'$
        - A downstream MLP generates grouped weights: $\omega^{(1)}, ..., \omega^{(n)} = \text{softmax}(\Gamma \cdot \text{MLP}(f_g))$
        - Here $\Gamma: \mathbb{R}^L \to \mathbb{R}^n$ maps DiT layer depth to $n$ weight groups.
    - **Visual Analysis (Figure 7)**: Shallow blocks progressively increase identity weights as denoising proceeds (preserving identity details), while the deepest blocks continuously increase motion weights — validating the functional division of labor across DiT depths.
    - **Design Motivation**: Early denoising steps are dominated by coarse-grained layout (motion-dominant), while later steps handle fine-grained details (identity-dominant), and blocks at different depths exhibit distinct functional preferences. Fixed weights cannot capture this dynamic variation.

3. **Weight Groups ($n=7$)**:

    - **Function**: Partitions the 42 DiT blocks into 7 groups, with each group sharing a set of weight coefficients.
    - **Mechanism**: Avoids insufficient context when $n=1$ and detail dilution when $n=42$.
    - **Ablation**: $n=7$ significantly outperforms $n=1$, $n=2$, and $n=42$ on both CLIP-I and DINO-I.

### Loss & Training

- Standard video diffusion reconstruction loss $\mathcal{L}$ is used.
- The switching ratio between identity and motion training steps is controlled by predefined hyperparameters.
- CogVideoX-5B serves as the base DiT model.
- Training data: 50 identity subjects (3–10 images each), 21 motion sequences, and 50 prompts per case.

## Key Experimental Results

### Main Results

| Method | CLIP-T↑ | CLIP-I↑ | DINO-I↑ | T.Flickering↑ | T.Cons↑ | Motion Smooth↑ | DD Deviation |
|--------|---------|---------|---------|---------------|---------|----------------|--------------|
| MotionBooth | 0.317 | 0.566 | 0.459 | 0.962 | 0.972 | 0.973 | -1.07 |
| LoRA | 0.323 | 0.425 | 0.286 | 0.956 | 0.976 | 0.973 | +13.32 |
| CogVideoX-5B | 0.336 | 0.521 | 0.424 | 0.947 | 0.973 | 0.965 | +14.49 |
| DreamVideo | 0.278 | 0.458 | 0.334 | 0.949 | 0.963 | 0.968 | -3.18 |
| **DualReal** | **0.323** | **0.629** | **0.551** | **0.965** | **0.983** | **0.978** | +2.94 |

DualReal outperforms the second-best method by 11.1% on CLIP-I and 20.0% on DINO-I, and ranks first across all three motion quality metrics (T.Cons, Motion Smoothness, T.Flickering).

### Ablation Study

| Configuration | CLIP-T | CLIP-I | DINO-I | DD Deviation | Note |
|---------------|--------|--------|--------|--------------|------|
| w/o Dual-aware Adapt. | 0.334 | 0.616 | 0.647 | -5.53 | Isolated training + inference fusion |
| w/o StageBlender | 0.346 | 0.619 | 0.652 | -3.31 | Fixed weights, no timestep awareness |
| w/o Weight Groups | 0.335 | 0.662 | 0.766 | -3.12 | Uniform modulation across all blocks |
| **DualReal (full)** | **0.333** | **0.674** | **0.771** | **-2.70** | Full method |

Weight group ablation: $n=1$ (0.662/0.766), $n=2$ (0.632/0.660), $n=42$ (0.631/0.706), **$n=7$ (0.674/0.771)**.

### Key Findings

- Removing Dual-aware Adaptation (i.e., reverting to isolated training with inference-time fusion) degrades DD deviation from -2.70 to -5.53, causing near-complete motion collapse — confirming the necessity of joint training.
- Visual analysis of the StageBlender Controller validates two important patterns: (1) the model's attention to identity increases monotonically as denoising progresses; (2) the deepest blocks specialize in motion modeling.
- Fixed weights (without StageBlender) cause fine-grained details such as hand regions to over-adapt to motion patterns.

## Highlights & Insights

1. **The paper is the first to systematically expose the fundamental flaw of the isolated customization paradigm**: the issue is not insufficient technique, but rather that the paradigm itself ignores the intrinsic dependencies between dimensions. The visualization analysis in Figure 2 is particularly compelling.
2. The **gradient mask regularization** approach is simple yet effective, cleanly severing cross-dimensional gradient interference in a manner cleaner than soft regularization.
3. The **visual analysis of StageBlender** (Figure 7) provides a new perspective for understanding spatiotemporal specialization within DiT: shallow blocks handle identity while deep blocks handle motion — a finding with significant reference value for subsequent work on customized generation.
4. The optimality of $n=7$ in the grouping mechanism suggests the existence of approximately 7 functionally similar block clusters within the DiT.

## Limitations & Future Work

- The evaluation dataset is relatively small (50 subjects × 21 motions × 50 prompts); generalization requires validation at larger scale.
- The Dynamic Degree metric indicates below-average motion intensity (DD=14.96 vs. reference 12.02), suggesting insufficient support for highly dynamic scenes.
- The identity/motion switching ratio is a fixed hyperparameter; future work may explore adaptive switching strategies.
- Comparisons with recent methods such as IP-Adapter and Animate-Anyone are absent.

## Related Work & Insights

- DreamVideo's "separate training + inference fusion" paradigm serves as the primary isolated baseline this work targets.
- The expert transformer architecture of CogVideoX provides the strong foundation model for DualReal.
- The gradient masking strategy draws on classical methods for preventing catastrophic forgetting in multi-task learning.
- The joint training framework proposed here has broad reference value for tasks requiring simultaneous control over multiple dimensions in video/image generation (e.g., text + pose, style + structure).

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[ICCV 2025\] DreamRelation: Relation-Centric Video Customization](dreamrelation_relation-centric_video_customization.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[ICCV 2025\] MagicDrive-V2: High-Resolution Long Video Generation for Autonomous Driving with Adaptive Control](magicdrive-v2_high-resolution_long_video_generation_for_autonomous_driving_with_.md)
- [\[ICCV 2025\] Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer](decouple_and_track_benchmarking_and_improving_video_diffusion_transformers_for_m.md)

</div>

<!-- RELATED:END -->
