---
title: >-
  [Paper Note] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing
description: >-
  [CVPR 2025][Image Generation][Multi-human motion editing] This paper proposes InterEdit, the first text-guided multi-human 3D motion interaction editing framework. It achieves semantic editing while preserving the spatio-temporal coupling relationships between multiple humans in diffusion models through Semantic-Aware Plan Token Alignment and Interaction-Aware Frequency Token Alignment.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multi-human motion editing"
  - "text-guided"
  - "diffusion model"
  - "interaction preservation"
  - "DCT frequency control"
  - "classifier-free guidance"
date: 2026-05-08
content_hash: 7fc0dc6a37c95828
---

# InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing

**Conference**: CVPR 2025  
**arXiv**: [2603.13082](https://arxiv.org/abs/2603.13082)  
**Code**: TBD  
**Area**: Motion Generation / Multi-Human Interaction  
**Keywords**: Multi-human motion editing, text-guided, diffusion model, interaction preservation, DCT frequency control, classifier-free guidance

## TL;DR

This paper proposes InterEdit, the first text-guided multi-human 3D motion interaction editing framework. It achieves semantic editing while preserving the spatio-temporal coupling relationships between multiple humans in diffusion models through Semantic-Aware Plan Token Alignment and Interaction-Aware Frequency Token Alignment.

## Background & Motivation

**Rapid development of text-driven motion editing**: Recent years have witnessed significant progress in single-person motion editing based on diffusion models (e.g., MDM, MotionDiffuse), but these methods are limited to single-person scenarios.

**Complexity of multi-human interactions**: Multi-human motion involves complex spatio-temporal coupling, such as contact constraints, rhythm synchronization, and spatial coordination. Editing one person's motion may disrupt the overall interaction relationship.

**Limitations of Prior Work**: Multi-human motion generation methods like TIMotion can generate interactive motions from scratch, but cannot perform fine-grained editing on existing motions.

**Difference between editing and generation**: Motion editing requires executing specified textual editing instructions while preserving original motion characteristics (e.g., interaction patterns, rhythm), making it more constrained than motion generation.

**Challenges in rhythm preservation**: Edited motions need to maintain rhythmic and frequency characteristics similar to the original motion; otherwise, the interaction will appear unnatural.

**Core Idea**: Simultaneously achieve semantic editing and interaction preservation during the diffusion process via learnable Plan Tokens (for semantic alignment) and Frequency Tokens (for frequency control).

## Method

### Overall Architecture

InterEdit is based on a conditional diffusion model and consists of three core modules:

1. **Synchronized Classifier-Free Guidance**: Synchronously denoises the motions of both individuals during the diffusion process, executing conditional and unconditional denoising simultaneously.
2. **Semantic-Aware Plan Token Alignment (SAPTA)**: Aligns learnable tokens with the semantic knowledge of a teacher model.
3. **Interaction-Aware Frequency Token Alignment (IAFTA)**: A DCT-based frequency control token to maintain motion rhythm.

### Key Designs 1: Semantic-Aware Plan Token Alignment (SAPTA)

- Introduces learnable Plan Tokens, which are appended to the inputs of the diffusion model.
- Transfers the semantic understanding capability of a pre-trained teacher model (a single-person motion editing model) to the multi-human editing framework via knowledge distillation.
- Plan Tokens encode semantic information regarding "what edit to perform", guiding the diffusion process in the correct editing direction.
- Learned by aligning the intermediate features of the student and teacher models during training.

### Key Designs 2: Interaction-Aware Frequency Token Alignment (IAFTA)

- Decomposes the original multi-human motion using Discrete Cosine Transform (DCT) to extract frequency features.
- Encodes frequency features into learnable Frequency Tokens and injects them into the diffusion process.
- Low-frequency components correspond to global motion trends and rhythm, while high-frequency components correspond to detailed actions.
- Balances the editing flexibility and interaction preservation by controlling the preservation level of different frequency bands.
- Regularized during training using frequency dropout ($p_f$).

### Key Designs 3: Synchronized Diffusion and Interaction Consistency

- Motions of both individuals share timesteps and noise scheduling during the diffusion process.
- In classifier-free guidance, the conditional branch receives textual editing instructions while the unconditional branch maintains the interaction structure.
- Controllable guidance scale: Larger guidance weights emphasize textual edits, while smaller weights focus more on preserving the original interaction.

## Key Experimental Results

### Main Results

| Method | FID↓ | g2t R@1↑ | g2s R@1↑ |
|------|------|---------|----------|
| TIMotion | 0.4451 | 24.97% | 12.54% |
| **InterEdit** | **0.3707** | **30.82%** | **17.08%** |
| Gain | -16.7% | +5.85pp | +4.54pp |

### Ablation Study

| Configuration | FID↓ | g2t R@1↑ | Description |
|------|------|---------|------|
| Full InterEdit | **0.3707** | **30.82%** | Full method |
| w/o Plan Token | ~0.42 | ~27% | Semantic alignment degradation |
| w/o Frequency Token | ~0.41 | ~28% | Rhythm preservation degradation |
| High frequency dropout ($p_f$=0.2) | ~0.39 | ~29% | Excessive dropout |
| **Optimal $p_f$=0.05** | **0.3477** | — | Optimal FID |
| $p_f$=0.0 (w/o dropout) | ~0.38 | — | Slight overfitting |

### Key Findings

- Both Plan Tokens and Frequency Tokens are essential components; removing either leads to significant performance degradation.
- A moderate frequency dropout ($p_f$=0.05) yields the best FID (0.3477), indicating that discarding a moderate amount of frequency information provides a regularization effect.
- The simultaneous improvement in g2t (gesture-to-text) and g2s (gesture-to-gesture-score) shows that the edited motion aligns with textual semantics while maintaining interaction quality.
- Significantly outperforms TIMotion on text-to-motion retrieval metrics, indicating more accurate semantics for edited motions.

## Highlights & Insights

1. **First Multi-Human Motion Interaction Editing Task**: Defines a new task, filling the gap of motion editing from single-person to multi-human scenarios.
2. **Frequency Domain Control**: DCT frequency decomposition serves as an elegant tool for preserving motion rhythm, which is more natural than directly imposing constraints in the time domain.
3. **Modular Design**: Plan Tokens and Frequency Tokens can be independently added and removed, facilitating the analysis of their respective contributions.
4. **Knowledge Distillation with a Teacher Model**: Cleverly utilizes a mature model for single-person motion editing as the teacher, avoiding training semantic understanding from scratch.

## Limitations & Future Work

- Currently only supports two-person interaction editing; scenarios with three or more people have not been verified.
- Relies on a pre-trained teacher model, whose quality affects the learning of Plan Tokens.
- Evaluation relies heavily on FID and retrieval metrics, lacking direct measurement of physical plausibility (such as collision/penetration detection) of the interactions.
- The performance on long-sequence motion editing has not been sufficiently discussed.

## Related Work & Insights

- **MDM / MotionDiffuse**: Single-person motion diffusion models, which InterEdit extends to multi-human interaction.
- **TIMotion**: A multi-human motion generation method, functioning as the primary baseline for comparison.
- **Insights**: The DCT frequency control concept can be generalized to other editing tasks that require preserving temporal sequence characteristics (e.g., audio or video rhythm editing).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — New task definition + novel frequency control concept.
- **Experimental Thoroughness**: ⭐⭐⭐☆ — Comprehensive ablation study, but evaluation metrics could be enriched.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and well-articulated motivation.
- **Value**: ⭐⭐⭐☆ — Clear application scenarios, but currently limited to two people.
- **Overall Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)
- [\[CVPR 2025\] Towards Scalable Human-Aligned Benchmark for Text-Guided Image Editing](towards_scalable_human-aligned_benchmark_for_text-guided_image_editing.md)
- [\[CVPR 2025\] Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction](nonisotropic_gaussian_diffusion_for_realistic_3d_human_motion_prediction.md)
- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](../../CVPR2026/image_generation/vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[CVPR 2025\] Lifting Motion to the 3D World via 2D Diffusion](lifting_motion_to_the_3d_world_via_2d_diffusion.md)

</div>

<!-- RELATED:END -->
