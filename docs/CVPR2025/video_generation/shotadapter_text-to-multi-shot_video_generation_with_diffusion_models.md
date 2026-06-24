---
title: >-
  [Paper Note] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models
description: >-
  [CVPR 2025][Video Generation][Multi-shot Video Generation] ShotAdapter proposes a lightweight framework that converts a pretrained single-shot T2V model into a generator supporting text-to-multi-shot video generation (T2MSV) with only about 5000 fine-tuning iterations. This is achieved by introducing learnable "transition tokens" and a local attention masking strategy, enabling multi-shot video generation with consistent character identities and independently controllable sho…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Multi-shot Video Generation"
  - "Diffusion Transformer"
  - "Transition Control"
  - "Identity Consistency"
  - "Attention Masking"
date: 2026-05-08
content_hash: d4db3628b0021dcb
---

# ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2505.07652](https://arxiv.org/abs/2505.07652)  
**Code**: [https://shotadapter.github.io/](https://shotadapter.github.io/)  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Multi-shot Video Generation, Diffusion Transformer, Transition Control, Identity Consistency, Attention Masking

## TL;DR

ShotAdapter proposes a lightweight framework that converts a pretrained single-shot T2V model into a generator supporting text-to-multi-shot video generation (T2MSV) with only about 5000 fine-tuning iterations. This is achieved by introducing learnable "transition tokens" and a local attention masking strategy, enabling multi-shot video generation with consistent character identities and independently controllable shots.

## Background & Motivation

**Background**: Current diffusion models (such as OpenSora, MovieGen, Kling AI, etc.) have made significant progress in text-to-video generation, but all existing models can only generate short videos with a single continuous shot, lacking the capability to switch between different shots.

**Limitations of Prior Work**: Real-world applications (such as filmmaking) require multi-shot videos, where the same character transitions across different scenes and activities. Existing makeshift solutions suffer from severe limitations: (1) merging all descriptions into a single prompt fails to create hard cuts and cannot showcase distinct activities; (2) generating each shot separately and then splicing them leads to character identity inconsistency; (3) generating consistent keyframes from a reference image first and then animating them with an I2V model still suffers from insufficient identity and background consistency due to the limitations of existing tools.

**Key Challenge**: Multi-shot video generation simultaneously requires cross-shot character/background consistency (demanding global information interaction) and independent controllability of each shot's content (demanding localized control), two objectives that are inherently contradictory.

**Goal**: To design a model-agnostic framework that enables T2V models to support multi-shot generation with minimal fine-tuning cost, while allowing users to control the number, duration, and content of the shots.

**Key Insight**: The authors observe that in NLP, the [EOS] token is successfully utilized to mark sentence boundaries. By analogy, a learnable "transition token" can be employed to mark shot transition points in videos. Combined with local attention masks to restrict each text prompt to affect only its corresponding shot's visual tokens, localized control can be achieved.

**Core Idea**: Introducing learnable transition tokens to mark shot boundaries, combined with a local attention masking strategy to enable independent text control per shot within a single complete video, while maintaining global attention to preserve identity consistency.

## Method

### Overall Architecture

ShotAdapter is based on a DiT-structured T2V model (similar to OpenSora). The pipeline is as follows: the input consists of a set of text prompts for each shot and the desired shot count/duration configuration. A 3D-VAE encodes the video into latent representations, which are patchified into a sequence of visual tokens. The textual condition tokens of each shot are concatenated sequentially, with $n-1$ transition tokens appended at the end (where $n$ is the number of shots). The entire sequence is fed into the DiT, where local attention masks control the interaction among different tokens. The model's output is decoded by the 3D-VAE to yield a complete video containing multiple shots.

### Key Designs

1. **Transition Token**:

    - Function: Marks the shot transition positions in the video, teaching the model to generate "hard cuts" at designated frames.
    - Mechanism: Initializes a set of learnable parameters matching the hidden dimension $D$, which is duplicated $n-1$ times for an $n$-shot video and appended to the end of the input sequence. In the attention layers, transition tokens only interact with the visual tokens of frames where transitions occur. Consequently, the model learns to generate abrupt scene cuts at these frame locations.
    - Design Motivation: Analogous to special token mechanisms in NLP, using very few learnable parameters (just a single embedding vector) can encode the semantic concept of a "shot transition". Experiments demonstrate that the model can generalize to shot counts unseen during training (2-8 shots), with an average frame error of only 1-2 frames.

2. **Local Attention Masking**:

    - Function: Achieves independent text control per shot, ensuring that each text prompt only influences the visual generation of its corresponding shot.
    - Mechanism: Constructs a structured attention masking matrix to constrain three types of interactions: (a) transition tokens only pay attention to the visual tokens of transition frames; (b) each text token only attends to the visual tokens of its corresponding shot; (c) visual and textual tokens maintain self-attention individually. In this way, textual conditions for different shots exert their influence independently.
    - Design Motivation: Without masking, fully connected interactions among all tokens dilute the influence of individual shot prompt information. The local mask preserves global visual self-attention (ensuring character consistency) while restricting the scope of influence of the textual conditions.

3. **Multi-Shot Video Dataset Construction Pipeline**:

    - Function: Constructs multi-shot training data from existing single-shot video datasets.
    - Mechanism: Two methods are proposed: (a) randomly cropping and concatenating sub-segments from highly dynamic single-shot videos (preserving background consistency with different actions/viewpoints); (b) clustering and randomly splicing multiple independent videos of the same identity (introducing diversity across different backgrounds). Post-processing steps include generating shot-specific descriptions using LLaVA-NeXT, detecting the number of people using YOLO, and verifying identity consistency using DINOv2, which filters out 38% of unqualified samples.
    - Design Motivation: The lack of off-the-shelf multi-shot training data is a key bottleneck for this task. These two complementary strategies enable the automatic construction of high-quality multi-shot data from single-shot data without manual annotation.

### Loss & Training

Fine-tuning employs the standard diffusion training loss, freezing most parameters of the pretrained model and only updating the learnable parameters of the transition tokens and the mask-related components in the attention layers. It requires only about 5,000 iterations (less than 1% of the pretraining iterations) with a 90% smaller batch size. The training data includes multi-shot videos of 2, 3, and 4 shots.

## Key Experimental Results

### Main Results

| Method | 2-shot IC↑ | 3-shot IC↑ | 4-shot IC↑ | 2-shot BC↑ | TA (2-shot)↑ |
|------|-----------|-----------|-----------|-----------|-------------|
| Random Shots | 71.03/80.47 | 54.76/63.72 | 48.08/55.87 | 84.46 | 26.84 |
| Similar Shots | 73.94/82.55 | 55.15/66.17 | 49.25/58.67 | 88.85 | 26.40 |
| Shots by Ref. | 81.74/84.98 | 67.92/72.97 | 57.83/67.74 | 82.11 | 25.59 |
| **ShotAdapter** | **78.67/86.33** | **70.30/76.44** | **61.86/74.89** | **89.48** | **27.12** |

(IC = Identity Consistency, diff bg/same bg; BC = Background Consistency; TA = Text Alignment)

### Ablation Study

| Configuration | 2-shot IC↑ | 3-shot IC↑ | 4-shot IC↑ | 2-shot BC↑ |
|------|-----------|-----------|-----------|-----------|
| ShotAdapter (full) | 78.67/86.33 | 70.30/76.44 | 61.86/74.89 | 89.48 |
| w/o Transition Token | 77.17/84.78 | 68.95/70.98 | 58.83/70.24 | 87.94 |
| Trained only on 2-shot data | 78.05/85.46 | 70.12/71.53 | 56.99/68.37 | 89.08 |

### Key Findings

- ShotAdapter significantly outperforms all baselines in identity consistency for 3-shot and 4-shot scenarios, while showing comparable competitiveness to the Shots by Reference baseline in 2-shot scenarios.
- The incorporation of transition tokens contributes notably to background consistency and text alignment, indicating that it successfully teaches the model to perform "hard cuts".
- The model trained only on 2-shot data still performs well on 3-shot and 4-shot scenarios, showcasing the strong generalization capability of the transition tokens.
- The transition tokens can generalize to 2-8 shots, with a mean shot boundary deviation error (MSDE) of only 0.83-2.00 frames.
- In user studies, ShotAdapter wins in identity consistency and background consistency with a preference rate of approximately 73-82%.

## Highlights & Insights

- **Highly simple and elegant design of the transition token**—with just a single learnable embedding vector, repeating it $n-1$ times combined with attention masking can control an arbitrary number of shot transitions. This is transferable to other generation tasks requiring "structured segmentation".
- **Data construction pipeline solves the "no-data" bottleneck**—systematizing the single-shot to multi-shot data conversion and leveraging clustering and automatic filtering to ensure quality is a reusable methodology.
- **Local attention masking balances global consistency and local controllability**—maintaining global attention among visual tokens (identity consistency) while restricting text tokens to their corresponding shots (local control). This approach can be generalized to other multi-condition control scenarios.

## Limitations & Future Work

- Validated only on human-centric scenes; other subjects such as animals remain untested, largely limited by the data filtering pipeline.
- The maximum generation length is limited by the underlying base model (currently 128 frames); the authors suggest this could be extended in an autoregressive manner.
- Fine-tuning causes a slight degradation in visual quality (notable in user studies), which might be related to the smaller batch size.
- The resolution is limited to 192×320, leaving a significant gap to the requirements of actual film production.

## Related Work & Insights

- **vs Single-Shot T2V**: Single-shot models cannot generate "hard cuts"; merging multiple descriptions into a single prompt leads to mixed and indistinguishable actions.
- **vs StoryMaker + I2V**: The pipeline of first generating consistent keyframes and then animating them frame-by-frame is limited by the quality of off-the-shelf tools, showing severe identity degradation especially across multiple shots.
- **vs Image Story Generation (ConsiStory/StoryDiffusion)**: These approaches focus on consistency across image sequences but lack temporal continuity in video and the concept of "shot transitions".

## Rating

- Novelty: ⭐⭐⭐⭐ Defies the T2MSV task and provides an end-to-end solution for the first time, with a novel transition token design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Designs a comprehensive evaluation pipeline and multiple baselines, but lacks comparison with more existing methods.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear and the method representation is intuitive, though the data construction section could be more concise.
- Value: ⭐⭐⭐⭐ Opens up a new direction for multi-shot video generation, but current results are still far from practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2026\] MultiShotMaster: A Controllable Multi-Shot Video Generation Framework](../../CVPR2026/video_generation/multishotmaster_a_controllable_multi-shot_video_generation_framework.md)
- [\[CVPR 2025\] Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [\[CVPR 2025\] Multi-subject Open-set Personalization in Video Generation](multi-subject_open-set_personalization_in_video_generation.md)
- [\[CVPR 2026\] OneStory: Coherent Multi-Shot Video Generation with Adaptive Memory](../../CVPR2026/video_generation/onestory_coherent_multi-shot_video_generation_with_adaptive_memory.md)

</div>

<!-- RELATED:END -->
