---
title: >-
  [Paper Note] RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation
description: >-
  [ICLR 2026][Image Generation][3D asset reference] This paper proposes RefAny3D, a 3D asset-referenced image generation framework that achieves precise geometric and texture consistency between generated images and 3D ref…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "3D asset reference"
  - "dual-branch generation"
  - "point map"
  - "domain decoupling"
  - "subject-driven generation"
date: 2026-05-08
content_hash: fbc1f4c5100c456e
---

# RefAny3D: 3D Asset-Referenced Diffusion Models for Image Generation

**Conference**: ICLR 2026
**arXiv**: [2601.22094](https://arxiv.org/abs/2601.22094)
**Code**: [https://judgementh.github.io/RefAny3D](https://judgementh.github.io/RefAny3D)
**Area**: 3D-Guided Image Generation / Diffusion Models
**Keywords**: 3D asset reference, dual-branch generation, point map, domain decoupling, subject-driven generation

## TL;DR

This paper proposes RefAny3D, a 3D asset-referenced image generation framework that achieves precise geometric and texture consistency between generated images and 3D reference assets through a dual-branch generation strategy that jointly models RGB images and point maps.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Existing reference-based image generation methods (e.g., IP-Adapter, OminiControl) rely on 2D reference images and cannot effectively leverage 3D assets. In practice, creators often need to use 3D assets such as meshes directly as references to visualize objects across different scenes.

3D asset-referenced generation faces three major challenges:

**Insufficient Consistency**: Precise alignment with the geometric structure and texture of 3D assets is required.

**Limited Viewpoint Coverage**: Single-reference-image methods cannot capture the complete appearance of an object.

**Viewpoint Conflict**: Multi-image conditioning methods lack 3D structural priors, leading to cross-view inconsistencies.

## Method

### Overall Architecture

Built upon the Flux.1-dev model, the framework takes multi-view RGB images and point maps of a 3D asset as conditions, and jointly generates the target RGB image along with its corresponding point map.

### Key Design 1: Spatially Aligned Dual-Branch Generation

The generation process is formulated as joint distribution modeling: $p(x_I, x_P | y, c)$

- $x_I$: target RGB image
- $x_P$: corresponding point map
- $y$: reference 3D model
- $c$: text prompt

**Shared Positional Encoding**: Shared positional encoding is applied to tokens from the same viewpoint for both RGB and point map branches, leveraging the positional encoding properties of DiT to naturally assign higher attention scores to tokens at the same position. A unified positional offset term $(i-w, j)$ is introduced to avoid bias caused by inconsistent distances among condition tokens.

### Key Design 2: Domain-Decoupled Generation

RGB and point maps exhibit inherent information asymmetry: point maps only define 3D geometry and pose, whereas RGB images contain photorealistic details of the entire scene.

- **Domain-Specific LoRA**: Reference-LoRA (activated for all condition tokens) and Domain-LoRA (activated only for point map tokens) are introduced to separately learn reference generation and point map domain knowledge.
- **Text-Agnostic Attention**: An attention mask is applied in the point map branch to suppress the influence of text tokens on the point map, preventing background information from leaking into the point map.

### Dataset Construction

A 3D asset–pose aligned dataset is constructed based on the Subjects200K dataset through the following pipeline:
1. Target objects are extracted using GroundingDINO.
2. Objects are converted into 3D assets using Hunyuan3D.
3. The pose of 3D assets in images is estimated using FoundationPose.

## Key Experimental Results

### Main Results (GPT Evaluation + Visual Model Evaluation)

| Method | Texture↑ | Geometry↑ | Aesthetics↑ | Overall↑ | CLIP Avg↑ | DINO Avg↑ | GIM↑ |
|------|------|------|------|------|----------|----------|------|
| Textual Inversion | 2.89 | 4.42 | 6.26 | 4.53 | 0.827 | 0.548 | 3360 |
| DreamBooth | 5.37 | 6.68 | 6.89 | 6.32 | 0.867 | 0.695 | 3483 |
| OminiControl | 5.63 | 6.58 | 6.89 | 6.37 | 0.855 | 0.665 | 3474 |
| **RefAny3D** | **6.32** | **7.37** | **7.69** | **7.12** | **0.873** | **0.720** | **3901** |

### Ablation Study

| Setting | Effect |
|------|------|
| w/o shared positional encoding | Pixel-level correspondence between point map and RGB fails; geometric consistency degrades. |
| w/o text-agnostic attention | Point map is influenced by text tokens; color mixing appears in background regions. |
| w/o domain-specific LoRA | A single LoRA learns both domains simultaneously, causing background artifacts. |
| w/o point map branch | Absence of 3D cues leads to unstable training and poor 3D consistency. |

### User Study

RefAny3D outperforms all baselines in fidelity (4.655), identity preservation (4.737), aesthetic quality (4.632), and overall ranking (1.579).

## Highlights & Insights

- This work is the first to explore image generation conditioned on 3D assets as reference.
- The use of point maps as structural anchors effectively establishes pixel-level correspondence across viewpoints.
- The domain-decoupling strategy elegantly addresses the information asymmetry between RGB and point map branches.
- The framework can be integrated with multi-view-to-3D generation models to form a complete production workflow.

## Limitations & Future Work

- Performance degrades on non-rigid objects (e.g., ropes, cushions) due to dataset limitations.
- Using a large number of viewpoint conditions introduces significant computational and time overhead.
- Data construction quality depends on Hunyuan3D and FoundationPose.

## Related Work & Insights

- **Subject-Driven Generation**: Textual Inversion, DreamBooth, IP-Adapter, OminiControl, etc.
- **3D-Guided Generation**: ThemeStation, Phidias, etc., which focus on 3D asset generation rather than image generation.
- **Multimodal Generation**: Marigold, GeoWizard, etc., which jointly generate RGB and geometric information.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — 3D asset-referenced image generation defines an entirely new task.
- **Technical Soundness**: ⭐⭐⭐⭐ — The dual-branch and domain-decoupling design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation is comprehensive, combining GPT assessment, visual model metrics, and user studies.
- **Value**: ⭐⭐⭐⭐ — Offers practical value for 3D content creation workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)
- [\[ICLR 2026\] Contact-Guided 3D Genome Structure Generation of E. coli via Diffusion Transformers](contact-guided_3d_genome_structure_generation_of_e_coli_via_diffusion_transforme.md)
- [\[ICLR 2026\] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)
- [\[ICLR 2026\] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)
- [\[ICLR 2026\] RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](rider_3d_rna_inverse_design_with_reinforcement_learning-guided_diffusion.md)

</div>

<!-- RELATED:END -->
