---
title: >-
  [Paper Note] StickMotion: Generating 3D Human Motions by Drawing a Stickman
description: >-
  [CVPR 2025][Human Understanding][Motion Generation] The StickMotion framework is proposed, which uses user-hand-drawn stickman drawings as fine-grained motion control conditions, combined with text descriptions, to achieve global and local 3D human motion generation. A Multi-Condition Module (MCM) is designed to efficiently process condition combinations, saving users 51.5% of their time for expressing creative motion ideas.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Motion Generation"
  - "Stickman Conditioning"
  - "Multi-Condition Diffusion"
  - "Dynamic Supervision"
  - "Text-to-Motion"
date: 2026-05-08
content_hash: e2d8b808f7da477a
---

# StickMotion: Generating 3D Human Motions by Drawing a Stickman

**Conference**: CVPR 2025  
**arXiv**: [2503.04829](https://arxiv.org/abs/2503.04829)  
**Code**: Coming soon  
**Area**: Human Understanding/Motion Generation  
**Keywords**: Motion Generation, Stickman Conditioning, Multi-Condition Diffusion, Dynamic Supervision, Text-to-Motion

## TL;DR

The StickMotion framework is proposed, which uses user-hand-drawn stickman drawings as fine-grained motion control conditions, combined with text descriptions, to achieve global and local 3D human motion generation. A Multi-Condition Module (MCM) is designed to efficiently process condition combinations, saving users 51.5% of their time for expressing creative motion ideas.

## Background & Motivation

Although text-to-motion generation has made significant progress, simple text descriptions (e.g., "high kick forward") cannot accurately convey complex limb poses imagined by users. Existing methods attempt to control details through more detailed text descriptions (e.g., FineMoGen describes by body parts), but this requires users to write long and precise descriptions, raising the barrier to usability.

Core Problem:
- **Insufficient Text Description Accuracy**: Natural language struggles to accurately describe the positions and angles of limbs in 3D space.
- **Inefficient Multi-Condition Fusion**: Traditional self-attention methods introduce unnecessary computation and performance degradation when processing multi-condition combinations.
- **Unnatural Fixed Frame Indexing**: Directly hard-binding the stickman pose to a specific frame disrupts the natural flow and continuity of the motion sequence.

Advantages of stickmen as control conditions: Drawing a simple stickman is much faster and more intuitive for users than writing a precise text description, while also providing exact limb position information that text cannot express.

## Method

### Overall Architecture

Based on diffusion models, StickMotion accepts a text description and up to three stickman drawings (located near the start, middle, and end of the sequence, respectively) as conditions. It efficiently handles four condition combinations: $(text)$, $(text, stick)$, $(stick)$, and $()$ through the Multi-Condition Module (MCM), outputting predicted noise and stickman frame index scores.

### Key Design 1: Stickman Generation Algorithm (SGA)

- `Function`: Automatically generates simulated hand-drawn style stickman training data from 3D joint coordinates of existing datasets.
- `Mechanism`: Hand-drawn characteristics are considered: (1) stroke smoothness (simulating jitter on different devices); (2) drawing deviations (global position offsets); (3) inconsistent scaling (proportional deviations of different body parts). The pose is observed from a frontal perspective, requiring the user to draw 6 single-stroke line segments (head, torso, and limbs), which are encoded using a Transformer encoder.
- `Design Motivation`: Hand-drawn stickman data collection is time-consuming and heavily influenced by the style of human annotators; SGA can automatically generate diverse training data from any motion dataset.

### Key Design 2: Multi-Condition Module (MCM)

- `Function`: Efficiently processes all possible combinations of text and stickman conditions.
- `Mechanism`: The data is split along the batch dimension into four groups $(B_1, B_2, B_3, B_4)$ corresponding to four condition combinations. In Condition Fusion, two Feat Decoders are utilized to separately process text and stickman conditions. Outputs for all combinations are achieved by selectively applying them along the batch dimension, without requiring attention masks. The Latent Encoder further fuses the information.
- `Design Motivation`: Traditional methods handle multi-condition combinations by using self-attention with masking, which calculates useless attention for masked tokens and suffers from performance degradation due to mutual interference between different representation spaces (stickman vs. text).

### Key Design 3: Dynamic Supervision Strategy (Dynamic Supervision)

- `Function`: Allows the network to automatically adjust the exact index of the frame corresponding to the stickman near the specified position.
- `Mechanism`: Users only need to specify the approximate position of the stickman (start/middle/end), and the network outputs an index score $\hat{I}_l$ for each frame. During training, frames are randomly sampled as ground-truth stickmen within respective position ranges (e.g., the middle position is within $[3L/8, 5L/8]$). Supervision is performed using a softmax-weighted index loss: $\mathcal{L}_{index} = M \cdot \sum_l softmax(\hat{I}_l) \cdot \|\hat{x}_l - x_i\|^2$.
- `Design Motivation`: Fixed frame indexing causes unnatural motions (e.g., a frame abruptly transitioning to a specified pose). Allowing the network to choose the most natural position among neighboring frames to insert the stickman pose resolves this.

### Loss & Training

$\mathcal{L}_{total} = \mathcal{L}_{index}^{start} + \mathcal{L}_{index}^{middle} + \mathcal{L}_ {index}^{end} + \mathcal{L}_{motion}$, where $\mathcal{L}_{motion}$ is the standard diffusion noise prediction loss. During inference, classifier-free guidance is used to control the preference weights of the text/stickman conditions.

## Key Experimental Results

### Main Results: HumanML3D Test Set

| Method | R Precision Top3 ↑ | FID ↓ | MM Dist ↓ | Diversity ↑ |
|------|-------------------|-------|----------|------------|
| Real motions | 0.797 | 0.002 | 2.974 | 9.503 |
| MDM | 0.611 | 0.544 | 5.566 | 9.559 |
| MLD | 0.772 | 0.473 | 3.196 | 9.724 |
| **StickMotion** | **~0.78** | **~0.3** | **~3.1** | ~9.5 |

### User Study

| Metric | StickMotion Advantage |
|------|----------------|
| Consistency with Imagination | Significantly improved compared to text-only methods |
| Time Saving | Saves **51.5%** of time compared to text descriptions |
| Interactive Satisfaction | Higher than text-only methods |

### Ablation Study

| Module | FID | R-Precision |
|------|-----|-------------|
| Self-Attention Baseline | Higher | Lower |
| MCM (Ours) | **Lower** | **Higher** |
| Without Dynamic Supervision (Fixed Frame) | Unnatural | - |
| With Dynamic Supervision | **Natural** | - |

### Key Findings

- StickMotion is comparable to SOTA methods on text-to-motion metrics, while additionally providing precise pose control.
- MCM reduces computational complexity and improves performance compared to the self-attention baseline.
- Dynamic supervision significantly improves the naturalness of the generated motions under stickman conditioning.
- User studies demonstrate that drawing a stickman saves 51.5% of the time compared to writing detailed text, and the generated results better match user imagination.

## Highlights & Insights

1. **Intuitiveness of Stickman as a Control Condition**: Drawing a simple sketch is far more intuitive and efficient than describing textually "raise the right arm to shoulder height, extend the left foot forward...".
2. **Batch Design of MCM**: Leveraging batch-dimension grouping elegantly implements multi-condition combinations, avoiding the computational waste of attention masks.
3. **Dynamic Supervision is Key**: Allowing the network to slightly adjust the stickman position is the perfect trade-off for ensuring naturalness and user-friendliness.

## Limitations & Future Work

- Stickmen express limited information (no finger details, no facial expressions).
- Currently, only 3 stickman positions (start/middle/end) are supported; denser keyframe control remains to be explored.
- Inferring 3D poses from 2D stickmen is ambiguous (the same 2D projection may correspond to different 3D poses).
- Interactive real-time generation scenarios have not yet been explored.

## Related Work & Insights

- **MDM, MLD**: Diffusion model-based baselines for text-to-motion generation.
- **FineMoGen**: Details controlled via body parts using detailed text.
- **Flame**: Allows appending text to edit motion sequences.
- The concept of stickman conditioning can be extended to other generation tasks requiring precise spatial control (e.g., gesture generation, dance choreography).

## Rating

⭐⭐⭐⭐ — The stickman conditioning design is intuitive and practical, with user studies validating its actual value. Both MCM and dynamic supervision are excellent designs addressing real-world problems. The 51.5% time saving is a persuasive utility metric.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[CVPR 2025\] Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions](shape_my_moves_text-driven_shape-aware_synthesis_of_human_motions.md)
- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)
- [\[ICML 2025\] Scaling Large Motion Models with Million-Level Human Motions](../../ICML2025/human_understanding/scaling_large_motion_models_with_million-level_human_motions.md)
- [\[ICLR 2026\] QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture](../../ICLR2026/human_understanding/quamo_quaternion_motion_kinematics.md)

</div>

<!-- RELATED:END -->
