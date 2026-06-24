---
title: >-
  [Paper Note] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics
description: >-
  [CVPR 2025][Image Generation][Unified Framework] UniReal proposes to unify various image generation and editing tasks into a "discontinuous frame generation" framework. By leveraging video data as a scalable, universal source of supervision, it achieves unified handling of multiple tasks such as instruction-based editing, customized generation, and object insertion within a single diffusion Transformer through hierarchical prompting and text-image association mechanisms.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Unified Framework"
  - "Image Editing"
  - "Video Data Supervision"
  - "Diffusion Transformer"
  - "Multi-task Generation"
date: 2026-05-08
content_hash: ab7bbcafb98c7821
---

# UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics

**Conference**: CVPR 2025  
**arXiv**: [2412.07774](https://arxiv.org/abs/2412.07774)  
**Code**: [Project Page](https://xavierchen34.github.io/UniReal-Page/)  
**Area**: Image Generation / Unified Generation and Editing  
**Keywords**: Unified Framework, Image Editing, Video Data Supervision, Diffusion Transformer, Multi-task Generation

## TL;DR

UniReal proposes to unify various image generation and editing tasks into a "discontinuous frame generation" framework. By leveraging video data as a scalable, universal source of supervision, it achieves unified handling of multiple tasks such as instruction-based editing, customized generation, and object insertion within a single diffusion Transformer through hierarchical prompting and text-image association mechanisms.

## Background & Motivation

The field of image generation and editing has become increasingly specialized with rising application demands. Different tasks require distinct methodologies and domain-specific datasets, which limits cross-domain knowledge learning. However, various tasks share a core requirement: maintaining consistency between input and output images while introducing controlled visual changes.

Video generation models (such as Sora-like approaches) can effectively balance inter-frame consistency and motion changes, which highly aligns with the requirements of image editing. Furthermore, existing image editing datasets are difficult to construct and limited in scale, whereas video data naturally contains inter-frame consistency and dynamics, serving as a scalable and universal source of supervision.

The core insight of UniReal is to view image-level tasks as "discontinuous video generation", utilizing large-scale video learning to capture world dynamics (such as lighting, reflections, object interactions, etc.), while eliminating ambiguity in multi-task joint training through hierarchical prompting.

## Method

### Overall Architecture

Based on a 5B-parameter video generation Transformer, UniReal treats different numbers of input/output images as video frames, modeling inter-frame relationships through full attention. Input images are encoded by VAE and patchified into visual tokens. Index embeddings and image prompts (asset/canvas/control) are added, and then concatenated with T5-encoded text embeddings into a long 1D tensor to be fed into the Transformer.

### Key Design 1: Text-Image Association and Hierarchical Prompting

- **Function**: Eliminates ambiguity during multi-task joint training in a unified framework, precisely guiding how the model should process different input images.
- **Mechanism**: Visual tokens are associated with corresponding text positions using reference words like "IMG1"/"RES1". A three-layer prompt is designed: (a) **Base Prompt** describes the task content; (b) **Context Prompt** provides attribute labels (e.g., "realistic/synthetic", "static/dynamic"); (c) **Image Prompt** distinguishes three types of input images—canvas (background editing target), asset (reference object), and control (layout/shape constraint)—via learnable category embeddings.
- **Design Motivation**: The same input requires different handling across various tasks; editing requires maintaining the layout while making local changes, whereas customization generates an entirely new scene while preserving only the reference object. Keywords in context prompts can be shared across tasks to force the learning of shared features.

### Key Design 2: Constructing Universal Supervision from Video Data

- **Function**: Automatically constructs large-scale training data supporting various tasks from video data.
- **Mechanism**: Starting from raw videos, two frames are randomly selected as pre- and post-editing images (Video Frame2Frame, 8M samples). Kosmos-2 is used to obtain bounding box annotations for entities, and SAM2 is employed to trace mask trajectories, supporting tasks such as multi-object customization (5M), object insertion (1M), and reference segmentation (5M). GPT-4o mini is used to generate precise instructions for a high-quality subset of 200K.
- **Design Motivation**: The natural transitions between video frames—such as addition, deletion, attribute changes, and structural changes—cover the underlying principles of most editing tasks and are significantly easier to obtain than constructing specialized editing datasets.

### Key Design 3: Progressive Multi-task Training

- **Function**: Starts from base text-to-image/video generation capabilities, progressively learning editing capabilities and upgrading image resolution.
- **Mechanism**: First, pretrain on T2I/T2V data to obtain base generation capabilities (256 resolution) $\rightarrow$ Train on all datasets from Tab.1 to learn multiple tasks (256 resolution) $\rightarrow$ Progressively scale up to 512 $\rightarrow$ 1024 resolution. Flow matching loss is used for training, with a learning rate of 1e-5 with warm-up.
- **Design Motivation**: Progressive training strategies help stabilize the multi-task learning process, preventing the editing capabilities learned during the low-resolution stage from degrading when resolution is increased.

### Loss & Training

A standard flow matching loss function is used for training.

## Key Experimental Results

### Main Results 1: Instruction Editing - EMU Edit Test Set

| Method | CLIP_dir↑ | CLIP_im↑ | CLIP_out↑ | L1↓ | DINO↑ |
|------|----------|---------|----------|-----|-------|
| InstructPix2Pix | 0.078 | 0.834 | 0.219 | 0.121 | 0.762 |
| UltraEdit | 0.107 | 0.793 | 0.283 | 0.071 | 0.844 |
| EMU Edit | 0.109 | 0.859 | 0.231 | 0.094 | 0.819 |
| OmniGen | - | 0.836 | 0.233 | - | 0.804 |
| **UniReal** | **0.127** | **0.851** | **0.285** | **0.099** | **0.790** |

### Main Results 2: Customized Generation - DreamBench

| Method | CLIP-T↑ | CLIP-I↑ | DINO↑ |
|------|---------|---------|-------|
| DreamBooth | 0.305 | 0.803 | 0.668 |
| BLIP-Diffusion | 0.302 | 0.805 | 0.670 |
| IP-Adapter(Flux) | - | 0.772 | - |
| **UniReal** | **Outperforms most methods** | **High CLIP-I** | **High DINO** |

### Key Findings

- UniReal achieves the best performance in CLIP_dir and CLIP_out, indicating optimal editing direction and output quality.
- It can correctly model complex world dynamics such as shadows, reflections, lighting effects, and object interactions.
- Video Frame2Frame data alone is sufficient to train a model with basic editing capabilities.
- The 8M+ samples provided by video frame pairs in the training data vastly exceed existing publicly available editing datasets.

## Highlights & Insights

1. **Elegance of Unified Formulation**: Unifying all image generation/editing tasks into discontinuous frame generation is conceptually simple yet highly practical.
2. **Supervisory Potential of Video Data**: Natural changes between video frames inherently cover various editing patterns, significantly reducing data construction costs.
3. **Composable Context Prompts**: The natural compositionality of text allows prompts from different tasks to be combined flexibly, enabling novel features unseen in the training data.

## Limitations & Future Work

- The 5B model scale is relatively large, resulting in higher inference costs.
- For customized generation, there is still room for improvement in reference object fidelity.
- World dynamics modeling is still limited by the diversity of the training videos.
- Future work could explore scaling up to larger video datasets and unifying even more tasks.

## Related Work & Insights

- **OmniGen**: Tokenizes both text and images into long tensors, but its generation/editing capabilities as a byproduct are of lower quality.
- **ACE**: Employs query/conditioning blocks to receive different input images, but does not fully utilize video data.
- **Instruct-Imagen**: Uses multimodal instructions to unify image generation tasks, but lacks the scalability of video supervision.
- Insight: The architectural design principles of video generation models can be directly transferred to image editing tasks.

## Rating

⭐⭐⭐⭐ — Elegant unified framework design; the idea of using video data as universal supervision is highly valuable. It achieves competitive or SOTA performance across multiple tasks, demonstrating the feasibility and potential of large-scale unified models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)
- [\[CVPR 2025\] Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation](trust_your_critic_robust_reward_modeling_and_reinforcement_learning_for_faithful.md)
- [\[ICCV 2025\] VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](../../ICCV2025/image_generation/visualcloze_a_universal_image_generation_framework_via_visual_in-context_learnin.md)
- [\[CVPR 2025\] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)
- [\[CVPR 2025\] DreamOmni: Unified Image Generation and Editing](dreamomni_unified_image_generation_and_editing.md)

</div>

<!-- RELATED:END -->
