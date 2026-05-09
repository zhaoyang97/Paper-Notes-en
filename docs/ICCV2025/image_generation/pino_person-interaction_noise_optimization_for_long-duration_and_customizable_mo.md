---
title: >-
  [Paper Note] PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups
description: >-
  [ICCV 2025][Image Generation][multi-person interaction generation] This paper proposes Person-Interaction Noise Optimization (PINO), a training-free framework that decomposes complex multi-person group interactions into semantically well-defined dyadic interaction pairs. By leveraging a pretrained two-person interaction diffusion model with noise optimization and physical penalty terms, PINO sequentially synthesizes group interaction motions of arbitrary scale, supporting fine-grained user control and long-duration motion generation.
tags:
  - ICCV 2025
  - Image Generation
  - multi-person interaction generation
  - noise optimization
  - training-free
  - motion diffusion model
  - physical constraints
date: 2026-05-08
content_hash: 5f94488a043739e1
---

# PINO: Person-Interaction Noise Optimization for Long-Duration and Customizable Motion Generation of Arbitrary-Sized Groups

**Conference**: ICCV 2025
**arXiv**: [2507.19292](https://arxiv.org/abs/2507.19292)
**Code**: [GitHub](https://sinc865.github.io/pino/)
**Area**: Motion Generation / Diffusion Models
**Keywords**: multi-person interaction generation, noise optimization, training-free, motion diffusion model, physical constraints

## TL;DR

This paper proposes Person-Interaction Noise Optimization (PINO), a training-free framework that decomposes complex multi-person group interactions into semantically well-defined dyadic interaction pairs. By leveraging a pretrained two-person interaction diffusion model with noise optimization and physical penalty terms, PINO sequentially synthesizes group interaction motions of arbitrary scale, supporting fine-grained user control and long-duration motion generation.

## Background & Motivation

Generating realistic multi-person group interaction motions is of significant importance in animation, gaming, and robotics. However, as the number of participants grows, interaction complexity increases exponentially, making this problem highly challenging. Existing approaches exhibit the following critical limitations:

**Training data bottleneck**: Methods such as Shan et al. require training on dedicated multi-person datasets, but annotation costs scale sharply with group size, and models are confined to fixed group sizes.

**Lack of flexible control**: ControlNet-based methods such as InterControl and FreeMotion use a single shared prompt to describe the entire group, making it impossible to specify distinct interaction relationships for individual characters. This causes generated group motions to be overly uniform and simplistic (e.g., "three people walking hand in hand").

**Absence of physical realism**: Existing methods lack constraints against physical artifacts such as interpenetration and body overlap, problems that worsen as the number of characters increases. Although FreeMotion performs conditioned generation, it does not enforce relational constraints post-generation.

**Limited scalability**: Achieving fine-grained control (e.g., collision avoidance, orientation control, region constraints) typically requires retraining ControlNet for each additional control signal.

The core insight of PINO is that **group interactions are inherently composed of smaller, interconnected dyadic interactions**. For example, a group photo is not a monolithic interaction but a combination of dyadic interactions between the photographer and each subject, as well as among the subjects themselves. Shared characters (e.g., the photographer) serve as pivot nodes connecting different interaction pairs.

## Method

### Overall Architecture

PINO takes two inputs: (1) an ordered participant list specifying the target and reference characters for each dyadic interaction pair; and (2) independent text prompts describing each pair. The framework operates as follows:

1. Generate the first interaction pair using a pretrained two-person diffusion model.
2. Apply noise optimization to the first pair to eliminate physical artifacts.
3. Incrementally introduce new characters, each anchored to an existing character as a pivot (reference), and generate the new character's motion using an independent prompt.
4. Optimize the initial noise of each new character to enforce physical constraints with respect to all existing characters.

### Key Designs

1. **Mask-based conditioned diffusion generation**: A modified two-person diffusion model $G_\theta^{mask}$ is employed, which replaces the conditional sequence $\mathbf{x}_t^{cond}$ with a noised version of the reference character's motion sequence $\hat{\mathbf{x}}_0^{cond}$ during denoising, while only denoising the target character's noise $\mathbf{x}_T^{tgt}$. This ensures the newly generated motion remains consistent with the reference character's existing motion.

   For the $p$-th target character, the noise optimization is formulated as:

   $\hat{\mathbf{x}}_T^p \leftarrow \arg\min_{\mathbf{x}_T^p} \mathcal{L}\left(G_\theta^{mask}(\mathbf{x}_T^p, \hat{\mathbf{x}}_0^{k_p}, c_{k_p, p}), \{\mathbf{x}_0^{i \in \mathcal{I}}\}\right)$

   where $\mathcal{I}$ is a predefined subset of existing characters participating in the optimization.

2. **Physical penalty term design**:

   **Overlap avoidance loss**: A penalty is applied when the distance between root joint positions of two characters falls below threshold $\delta$:

   $\mathcal{L}_{overlap} = \sum_i \sum_n \max\left(0, \delta - \|\mathbf{p}_{root}^p(n) - \hat{\mathbf{p}}_{root}^i(n)\|_2\right)$

   **Spatiotemporal motion control loss** $\mathcal{L}_{control}$ comprises four differentiable penalties:
   - Root position penalty: constrains a character to reach a specified position at a given timestep.
   - Motion region penalty: restricts a character's movement to within a defined region.
   - Orientation penalty: controls the facing direction at specific frames.
   - Relative position penalty: maintains desired distance or orientation relationships between characters.

   The total optimization loss is: $\mathcal{L} = \mathcal{L}_{overlap} + \mathcal{L}_{control}$

3. **Long-duration motion generation**: Motion duration is extended via inpainting. The last $n$ frames of an existing sequence serve as context, and at each denoising step, the initial frames are replaced using a binary mask:

   $\mathbf{x}_t^i \leftarrow \mathbf{m} \odot \hat{\mathbf{x}}^i + (1 - \mathbf{m}) \odot \mathbf{x}_t^i$

   A boundary penalty is additionally applied to minimize joint acceleration, ensuring smooth transitions between old and new segments. Combined with prompt switching, this enables alternating interactions among characters over long sequences (e.g., three persons taking turns shaking hands).

### Loss & Training

The proposed method is entirely training-free and relies on the pretrained InterGen model (a two-person interaction diffusion model trained on the InterHuman dataset).
- 50-step DDIM sampler is used.
- Noise optimization learning rate: 0.003, with 100 optimization iterations.
- Initial noise is optimized via backpropagation through the diffusion process.

## Key Experimental Results

### Main Results

Two-person interaction generation (300 samples from the InterHuman test set):

| Method | Overlap ↓ | PenVol. (cm³) ↓ | FID ↓ | R-Prec. ↑ | Diversity |
|--------|-----------|-----------------|-------|-----------|-----------|
| GT | 0.029 | 471.75 | 0.983 | 0.715 | 7.921 |
| InterGen | 0.119 | 3112.72 | 13.278 | 0.674 | 7.793 |
| **PINO-InterGen** | **0.000** | **275.65** | **13.163** | **0.675** | **7.904** |

Multi-person interaction generation (5 persons, incrementally added, with person 1 as pivot):

| Method | Pair | FID ↓ | Overlap ↓ |
|--------|------|-------|-----------|
| FreeMotion | (1,5) | 25.671 | 0.991 |
| InterGen | (1,5) | 19.501 | 0.977 |
| **PINO-InterGen** | **(1,5)** | **16.911** | **0.069** |

### Ablation Study

Effect of incrementally adding penalty terms (24 generated sequences):

| Configuration | Position Error ↓ | Overlap ↓ | Region Violation ↓ | Orientation Error ↓ |
|--------------|------------------|-----------|-------------------|---------------------|
| InterGen (baseline) | 1.0 | 0.292 | 0.500 | 1.0 |
| + $\mathcal{L}_{root}$ | **0.0** | 0.333 | 0.917 | 1.0 |
| + $\mathcal{L}_{overlap}$ | 0.0 | **0.0** | 0.958 | 1.0 |
| + $\mathcal{L}_{region}$ | 0.083 | 0.0 | **0.043** | 1.0 |
| + $\mathcal{L}_{orientation}$ | 0.083 | 0.043 | 0.083 | **0.208** |

Each penalty term effectively reduces its corresponding error. The full combination of penalties achieves the lowest violation rate across all metrics.

### Key Findings

- PINO reduces Overlap from 0.119 to 0.000 (two-person) and from 0.991 to 0.069 (five-person), nearly eliminating character overlap entirely.
- Penetration volume is reduced from 3112.72 cm³ to 275.65 cm³, even surpassing the GT value of 471.75 cm³.
- Semantic quality metrics (FID, R-Precision) also improve alongside the enhancement in physical plausibility.
- In motion extension experiments, PINO reduces Foot Skate from 0.070 to 0.045, yielding more natural motion.

## Highlights & Insights

- The decomposition strategy of "group interaction = interconnected dyadic interactions" is both elegant and effective, reducing an exponentially complex problem to a linear one.
- Fully training-free: no multi-person dataset is required, no retraining is needed, and fine-grained control is achieved solely through noise optimization.
- Independent prompts can be specified for each interaction pair, offering substantially greater flexibility than a shared prompt.
- Physical penalty terms are designed in a differentiable form, enabling seamless integration into the noise optimization loop of the diffusion model.
- The combination of motion extension and prompt switching enables rich temporal interaction narratives.

## Limitations & Future Work

- Performance is bounded by the quality of the underlying two-person model (InterGen).
- The dyadic decomposition strategy cannot model higher-order coordinated behaviors (e.g., chain reactions where one person pushes another into a third).
- Joint-position-based penalties cannot fully prevent interpenetration in hand regions, as the InterHuman dataset lacks hand joint data.
- Noise optimization requires 100 iterations, and computational cost scales linearly with the number of characters.
- No direct comparison is made against methods that train dedicated multi-person models (e.g., Shan et al.).

## Related Work & Insights

- The noise optimization approach draws from Attend-and-Excite and InitNo in image generation, as well as ProgMoGen and DNO in motion generation.
- InterGen provides a strong prior for dyadic interactions; PINO extends it to arbitrary group sizes through post-hoc optimization.
- PINO is complementary to FreeMotion (ControlNet-based conditioned generation) and InterControl (joint-position-guided generation), as it requires no additional training.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The training-free approach extending dyadic interaction models to arbitrary group sizes is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers two-person, multi-person, extension, and ablation settings with comprehensive physical metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, with rich pseudocode and visualizations.
- **Value**: ⭐⭐⭐⭐ Directly applicable to animation, gaming, and related domains; framework design is elegant.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] InfiniDreamer: Arbitrarily Long Human Motion Generation via Segment Score Distillation](infinidreamer_arbitrarily_long_human_motion_generation_via_segment_score_distill.md)
- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](../../ICLR2026/image_generation/diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)

<!-- RELATED:END -->
