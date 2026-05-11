---
title: >-
  [Paper Note] SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency
description: >-
  [NeurIPS 2025][Image Generation][scene consistency] SceneDecorator presents a training-free framework that, for the first time, systematically addresses scene planning and scene consistency in story generation via VLM-gu…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "scene consistency"
  - "story generation"
  - "scene planning"
  - "training-free"
  - "attention mechanism"
date: 2026-05-08
content_hash: dce886ecdcbbc202
---

# SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency

**Conference**: NeurIPS 2025
**arXiv**: [2510.22994](https://arxiv.org/abs/2510.22994)
**Code**: [https://lulupig12138.github.io/SceneDecorator](https://lulupig12138.github.io/SceneDecorator) (project page)
**Area**: Diffusion Models / Story Image Generation
**Keywords**: scene consistency, story generation, scene planning, training-free, attention mechanism

## TL;DR
SceneDecorator presents a training-free framework that, for the first time, systematically addresses scene planning and scene consistency in story generation via VLM-guided global-to-local scene planning and a long-term scene-sharing attention mechanism, achieving significant improvements over existing methods on scene alignment and consistency metrics.

## Background & Motivation
Story generation requires producing multiple images with conceptual consistency. Existing methods focus primarily on character consistency while overlooking the role of scenes in narrative:

**Lack of Scene Planning**: Existing methods generate scenes solely from text descriptions, lacking scene-level narrative coherence. Scenes across the same story lack semantic association.

**Unexplored Scene Consistency**: Real-world applications such as film storyboards require generating multiple stories under the same scene with different plot content, yet existing methods cannot maintain cross-story scene consistency.

**Character Homogenization**: Excessive focus on character consistency causes generated subjects to converge in style, reducing diversity.

Core Idea: Treat scenes as a central element of story generation, employ a VLM as a "director" for global-to-local scene planning, and maintain long-term scene consistency via attention sharing.

## Method

### Overall Architecture
SceneDecorator is a training-free framework consisting of two core components:
1. VLM-Guided Scene Planning: Decomposes user themes into local scenes and story sub-prompts.
2. Long-Term Scene-Sharing Attention: Maintains cross-story scene consistency while preserving subject diversity.

The framework builds on SDXL as the base model with IP-Adapter-XL for scene injection, requiring no additional training.

### Key Designs

1. **VLM-Guided Scene Planning**

    - **Global Scene Conceptualization**: A VLM (Qwen2-VL) generates a global scene description $Q$ from the user theme $T$.
    - **Global Scene Visualization**: FLUX.1-dev generates a global scene image $V$ from description $Q$.
    - **Local Scene Cropping**: The VLM acts as a "director" and identifies coordinates $\{L_i\}$ of $M$ storyboard scenes within $V$, cropping local scenes $\{V_i\}$.
    - **Story Sub-Prompt Generation**: For each local scene $V_i$, the VLM generates $N$ sequential story sub-prompts $P^{1:N}$.
    - In-context learning is used to enhance VLM performance.

2. **Mask-Guided Scene Injection**

    - Problem: Directly injecting scenes via IP-Adapter causes subject styles to over-fuse with the background, reducing diversity.
    - Solution: The activation region of subject tokens in cross-attention maps is used as a mask $M$.
    - Scene feature injection follows a weighted formula: $Z_c^{new} = A_c \cdot V_c + \lambda \cdot (1-M) \cdot A_c' \cdot V_c'$
    - The mask ensures scene semantics are primarily injected into background regions, while subject regions remain guided by text for diversity.

3. **Scene-Sharing Attention**

    - In self-attention, the latent representations of two branches mutually attend to each other's keys and values.
    - Crucially, masks restrict cross-story attention to background regions: $K' = [K, \tilde{K} \odot (1-\tilde{M})]$
    - This enables cross-story scene sharing while preventing character confusion.

4. **Extrapolable Noise Blending**

    - Motivation: The above mechanism is limited to generating two stories simultaneously; extension to $N$ stories is needed.
    - Method: Within denoising interval $[T_1, T_2]$, the $N$ latent representations are dynamically partitioned into complementary pairs $\langle Z_t^i, Z_t^j \rangle$.
    - Each story participates in $N{-}1$ pairings; predicted noise is averaged before updating the latent.
    - Key advantage: GPU memory only needs to accommodate two stories at a time, yet consistency across $N$ stories is achieved.

### Loss & Training
No training is required; all modules are plug-and-play at inference time. Hyperparameters: $M{=}4$ (local scenes), $N{=}5$ (stories per scene), $T_1{=}0$, $T_2{=}25$, 20-step DDIM sampling. Runs on a single RTX 3090.

## Key Experimental Results

### Main Results

| Method | CLIP-T↑ | DreamSim-I↓ | DINO-F↑ | Text Align% | Scene Align% | Image Quality% |
|---|---|---|---|---|---|---|
| CustomDiffusion | 0.306 | 0.752 | 0.373 | 7.9% | 3.4% | 6.0% |
| ConsiStory | 0.320 | 0.723 | 0.475 | 21.3% | 14.1% | 24.7% |
| StoryDiffusion | 0.311 | 0.735 | 0.340 | 14.3% | 6.3% | 11.8% |
| **SceneDecorator** | 0.312 | **0.605** | **0.571** | **56.5%** | **76.2%** | **57.5%** |

### Ablation Study

| Configuration | Effect |
|---|---|
| w/o Mask-Guided Scene Injection | Text lacks scene semantics; stories fail to match the scene |
| w/ Scene Injection, w/o Mask | Scene semantics injected but subject style lacks diversity |
| w/ Scene Injection + Mask | Scene injection and subject diversity achieved, but cross-story consistency is limited |
| **Full SceneDecorator** | **Scene consistency + subject diversity + narrative coherence** |

| Noise Blending Memory | 1 | 2 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|
| w/o Extrapolable | 11.4G | 12.7G | 14.5G | 17.5G | 20.4G | 23.5G | OOM |
| **w/ Extrapolable** | 11.4G | 12.7G | 12.7G | 12.7G | 12.7G | 12.7G | 12.7G |

### Key Findings
- In a user study (61 participants), SceneDecorator achieves a 76.2% preference rate on scene alignment, substantially outperforming the second-best method ConsiStory (14.1%).
- DreamSim-I (scene alignment) improves from 0.723 (ConsiStory) to 0.605; DINO-F (scene consistency) improves from 0.475 to 0.571.
- Extrapolable Noise Blending fixes GPU memory at 12.7G regardless of the number of stories generated.
- GPT-4o evaluation of VLM scene planning semantic quality: narrative coherence 90.06%, thematic relevance 92.57%, layout rationality 90.29%.

## Highlights & Insights
- The first work to systematically examine story generation from a scene-centric perspective, introducing the new challenges of scene planning and scene consistency.
- The global-to-local scene planning strategy emulates real film production workflows, casting the VLM in the role of a "director."
- Mask-guided scene injection elegantly balances the tension between scene consistency and subject diversity.
- Extrapolable Noise Blending achieves consistency across $O(N)$ stories with $O(1)$ memory complexity.
- The training-free design allows flexible integration with tools such as PhotoMaker, ControlNet, and style LoRAs.

## Limitations & Future Work
- Scene planning relies on the VLM's imagination and comprehension, which may be inaccurate for complex themes.
- The VLM occasionally predicts coordinates outside image boundaries (corrected via snapping).
- Generation quality is bounded by SDXL as the base model.
- The scene injection weight $\lambda$ requires manual tuning, with potentially different optimal values across scenes.
- The framework is currently limited to static images; extension to video story generation is a natural next step.

## Related Work & Insights
- Compared to ConsiStory, SceneDecorator shifts the focus from character consistency to scene consistency, offering a new perspective.
- Mask-guided attention manipulation is generalizable to other generation tasks requiring local control.
- The noise blending pairing strategy is a general method for cross-sample interaction.
- The framework supports scene evolution (e.g., morning to evening, summer to winter), demonstrating its flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[ICCV 2025\] Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency](../../ICCV2025/image_generation/free4d_tuning-free_4d_scene_generation_with_spatial-temporal_consistency.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](../../ICLR2026/image_generation/generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)
- [\[ICLR 2026\] Consistent Text-to-Image Generation via Scene De-Contextualization](../../ICLR2026/image_generation/consistent_text-to-image_generation_via_scene_de-contextualization.md)
- [\[AAAI 2026\] MDiff4STR: Mask Diffusion Model for Scene Text Recognition](../../AAAI2026/image_generation/mdiff4str_mask_diffusion_model_for_scene_text_recognition.md)

</div>

<!-- RELATED:END -->
