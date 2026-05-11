---
title: >-
  [Paper Note] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion
description: >-
  [AAAI 2026][Video Generation][Multi-shot video generation] FilmWeaver is proposed as a framework that guides autoregressive diffusion models via a dual-level cache (Shot Cache + Temporal Cache), enabling multi-shot video generation of arbitrary length with cross-shot consistency.
tags:
  - AAAI 2026
  - Video Generation
  - Multi-shot video generation
  - autoregressive diffusion
  - cache mechanism
  - consistency
  - long video generation
date: 2026-05-08
content_hash: c9264cb18f0b5462
---

# FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion

**Conference**: AAAI 2026
**arXiv**: [2512.11274](https://arxiv.org/abs/2512.11274)
**Code**: [Project Page](https://filmweaver.github.io)
**Area**: Video Generation
**Keywords**: Multi-shot video generation, autoregressive diffusion, cache mechanism, consistency, long video generation

## TL;DR

FilmWeaver is proposed as a framework that guides autoregressive diffusion models via a dual-level cache (Shot Cache + Temporal Cache), enabling multi-shot video generation of arbitrary length with cross-shot consistency.

## Background & Motivation

### State of the Field
Current video diffusion models (e.g., HunyuanVideo, Wan) have demonstrated strong performance on single-shot video generation, yet **multi-shot video** generation remains a significant challenge. Multi-shot videos hold greater practical value in film production and narrative-driven creative scenarios.

### Limitations of Prior Work
Multi-shot video generation faces two critical challenges:

**Cross-shot consistency**: Character identities and backgrounds must remain visually coherent across shots, which cannot be achieved through text descriptions alone.

**Shot duration and count management**: Existing methods are limited in controlling the duration of individual shots and the total video length.

Existing approaches suffer from distinct drawbacks:
- **Multi-model pipeline methods** (VideoDirectorGPT, VideoStudio): Adopt a two-stage paradigm of keyframe generation followed by image-to-video, but independent generation of each segment leads to **visual discontinuities and scene jumps**.
- **Concurrent multi-shot methods** (ShotAdapter, Mask2DiT): Pack multiple shots into a single sequence, severely **constraining per-shot duration**.
- **TTT**: Introduces an RNN mechanism but lacks long-term memory and incurs high training costs.
- **LCT**: Requires two-stage training and supports only MM-DiT architectures.

### Core Idea

**Decouple the consistency problem into two sub-problems — cross-shot consistency and intra-shot coherence — managed separately via a dual-level cache system**: Shot Cache stores keyframes from prior shots to maintain character/scene identity, while Temporal Cache retains the frame history of the current shot to ensure smooth motion.

## Method

### Overall Architecture

FilmWeaver is built upon an autoregressive diffusion paradigm, centered on a **Dual-Level Cache mechanism**. When generating a new video chunk, the model is conditioned on a text prompt $\mathbf{c}_{\text{text}}$, Temporal Cache $C_{\text{temp}}$, and Shot Cache $C_{\text{shot}}$. The training objective is:

$$\mathcal{L} = \mathbb{E}_{\mathbf{v}_0, \mathbf{c}_{\text{text}}, \epsilon, t}\left[\left\|\epsilon - \epsilon_\theta(\mathbf{v}_t, t, \mathbf{c}_{\text{text}}, C_{\text{temp}}, C_{\text{shot}})\right\|^2\right]$$

The cache is injected via **in-context injection** without modifying the model architecture, making the approach compatible with existing pretrained T2V models.

### Key Designs

#### 1. **Temporal Cache (Intra-Shot Coherence)**
- **Function**: Acts as a sliding window storing conditioning information from the most recently generated frames of the current shot.
- **Mechanism**: Exploits high temporal redundancy in video via a **differential compression strategy**: recent frames are preserved at high fidelity, while distant frames are progressively compressed.
- **Implementation**: Three-level hierarchical compression — the most recent 1 latent is uncompressed, the next 2 are compressed at 4×, and the last 16 at 32×.
- **Design Motivation**: Maintains motion coherence while controlling computational overhead; each generation step produces 6 latents (24 frames, 1 second at 24 fps).

#### 2. **Shot Cache (Cross-Shot Consistency)**
- **Function**: Retrieves the Top-K keyframes most relevant to the current text prompt from prior shots.
- **Retrieval Mechanism**: Computes cosine similarity between CLIP text embeddings and candidate keyframe image embeddings:

$$C_{\text{shot}} = \underset{kf \in \mathcal{KF}}{\arg\,\text{top-k}}\left(\text{sim}(\phi_T(\mathbf{c}_{\text{text}}), \phi_I(kf))\right)$$

- **K=3**, chosen based on a performance-efficiency trade-off; three keyframes suffice to capture the diverse concepts required in complex multi-shot scenarios.
- **Design Motivation**: Provides a concise yet highly relevant visual summary of narrative history to guide the model in maintaining character and background consistency.

#### 3. **Four Inference Modes**
Based on the state of the dual-level cache, inference proceeds in four stages:
1. **No Cache** (first shot generation): Initializes the cache; operates in standard T2V mode.
2. **Temporal Only** (first shot extension): Ensures high temporal coherence; supports video extension.
3. **Shot Only** (new shot generation): Clears the Temporal Cache and injects prior keyframes from the Shot Cache; supports multi-concept injection.
4. **Full Cache** (new shot extension): Leverages both cache levels simultaneously.

### Loss & Training

#### Progressive Training Curriculum
- **Stage 1**: Trains long single-shot video generation using only the Temporal Cache (10K steps); Shot Cache is disabled, allowing the model to first master intra-shot dynamics.
- **Stage 2**: Activates the Shot Cache and fine-tunes on a mixed curriculum covering all four cache scenarios (10K steps); the progressive approach accelerates convergence.

#### Data Augmentation (Addressing the "Copy-Paste" Problem)
- **Problem**: The model over-relies on visual context, reducing motion diversity and text-following capability.
- **Negative Sampling**: Randomly introduces irrelevant keyframes into the Shot Cache, forcing the model to distinguish useful from distracting information.
- **Asymmetric Noising**: Strong noise (100–400 timesteps) is applied to the Shot Cache, and weak noise (0–100 timesteps) to the Temporal Cache, balancing copy-avoidance with motion coherence.

### Multi-Shot Dataset Construction

Expert models are applied for shot segmentation → sliding-window CLIP similarity for scene clustering → filtering (removal of segments shorter than 1 second and scenes with more than 3 persons) → **Group Captioning**: all shots within a scene are jointly fed into Gemini 2.5 Pro for unified description → a validation step ensures annotation accuracy.

## Key Experimental Results

### Main Results

| Method | Aes.↑ | Incep.↑ | Char. Cons.↑ | All Cons.↑ | Char. Align.↑ | All Align.↑ |
|--------|-------|---------|-------------|-----------|--------------|------------|
| VideoStudio | 32.02 | 6.81 | 73.34% | 62.40% | 20.88 | 31.52 |
| StoryDiffusion | 35.61 | 8.30 | 70.03% | 67.15% | 20.21 | 30.86 |
| IC-LoRA | 31.78 | 6.95 | 72.47% | 71.19% | 22.16 | 28.74 |
| **FilmWeaver** | **33.69** | **8.57** | **74.61%** | **75.12%** | **23.07** | **31.23** |

FilmWeaver achieves state-of-the-art performance on consistency and character-text alignment metrics, while also obtaining the highest Inception Score.

### Ablation Study

| Configuration | Aes.↑ | Incep.↑ | Char.↑ | All.↑ | Char. Align.↑ | All Align.↑ |
|---------------|-------|---------|--------|------|--------------|------------|
| w/o Augmentation | 30.04 | 7.77 | 72.36% | 75.92% | 21.88 | 28.12 |
| w/o Shot Cache | 33.92 | 8.63 | 68.11% | 65.44% | 22.41 | 31.79 |
| w/o Temporal Cache | 31.61 | 8.36 | 70.79% | 70.57% | 20.21 | 30.70 |
| **Full Model** | **33.69** | **8.57** | **74.61%** | **75.12%** | **23.07** | **31.23** |

### Key Findings
1. **Shot Cache is critical for cross-shot consistency**: Its removal causes All Consistency to drop sharply from 75.12% to 65.44%.
2. **Noise augmentation is critical for text-following**: Removing it leads to a significant decline in Text Alignment.
3. **Negative sampling provides fault tolerance**: The model can effectively ignore irrelevant keyframes when they are retrieved.
4. **Computational efficiency**: Attention complexity is reduced from $O(24^2)=576$ for the full sequence to approximately $3.5 \times 11^2 \approx 423.5$ with chunked processing.

## Highlights & Insights

1. **Elegant decoupling**: The consistency problem is explicitly decomposed into inter-shot and intra-shot sub-problems, each managed by a dedicated cache — a clean and effective design.
2. **High flexibility**: The four inference modes naturally support downstream tasks such as multi-concept injection and video extension without additional training.
3. **Architecture-agnostic**: Cache injection via in-context injection requires no modification to the model structure, enabling compatibility with diverse pretrained T2V models.
4. **Practical data construction pipeline**: The Group Captioning strategy addresses the problem of cross-shot annotation consistency.

## Limitations & Future Work

1. Visual quality still has room for improvement, which could be addressed through better data curation and training strategies.
2. The cache size (K=3) may be insufficient for highly complex scenes.
3. The data pipeline relies on Gemini 2.5 Pro, limiting annotation cost-efficiency and accessibility.
4. The evaluation benchmark is small (20 scenes × 5 shots), and standardized public multi-shot benchmarks are lacking.

## Related Work & Insights

- The differential compression strategy parallels that of FramePack, but is extended to cross-shot scenarios.
- The retrieval-augmented generation idea underlying the Shot Cache is transferable to other generation tasks requiring long-term consistency.
- The negative sampling strategy resembles hard negative training in contrastive learning, enhancing model robustness.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-level cache decoupling is elegant, though autoregressive diffusion and in-context injection are not themselves novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative and qualitative comparisons with sufficient ablations, though the evaluation scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, smooth logic, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Multi-shot video generation is an important problem, and the framework demonstrates strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](../../CVPR2026/video_generation/rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [\[CVPR 2026\] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](../../CVPR2026/video_generation/moviedrive_multimodal_multiview_video_diffusion.md)
- [\[AAAI 2026\] MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation](mofu_scale-aware_modulation_and_fourier_fusion_for_multi-subject_video_generatio.md)
- [\[CVPR 2026\] Gloria: Consistent Character Video Generation via Content Anchors](../../CVPR2026/video_generation/gloria_consistent_character_video_generation_via_content_anchors.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)

</div>

<!-- RELATED:END -->
