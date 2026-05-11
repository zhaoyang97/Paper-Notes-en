---
title: >-
  [Paper Note] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation
description: >-
  [AAAI 2026][Video Generation][Story video generation] This paper proposes DreamRunner, a framework that achieves fine-grained controllable multi-character multi-event story video generation via LLM-based dual-level plann…
tags:
  - "AAAI 2026"
  - "Video Generation"
  - "Story video generation"
  - "retrieval-augmented motion adaptation"
  - "regional attention"
  - "compositional video generation"
  - "LoRA injection"
date: 2026-05-08
content_hash: 47149824f6ce1dd2
---

# DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation

**Conference**: AAAI 2026
**arXiv**: [2411.16657](https://arxiv.org/abs/2411.16657)
**Code**: Not released
**Area**: Video Generation
**Authors**: Zun Wang, Jialu Li, Han Lin, Jaehong Yoon, Mohit Bansal (UNC Chapel Hill)
**Keywords**: Story video generation, retrieval-augmented motion adaptation, regional attention, compositional video generation, LoRA injection

## TL;DR

This paper proposes DreamRunner, a framework that achieves fine-grained controllable multi-character multi-event story video generation via LLM-based dual-level planning, retrieval-augmented motion prior learning, and a spatial-temporal region-based 3D attention injection module (SR3AI).

## Background & Motivation

Story video generation (SVG) aims to produce coherent multi-scene, character-driven videos from narrative scripts. Existing methods (e.g., VLogger, VideoDirectorGPT) primarily employ LLMs for high-level scene decomposition, generating each scene independently before concatenation. This paradigm faces three core challenges:

1. **Coherent composition**: Single-scene descriptions often contain multiple objects/characters with distinct motion trajectories, attributes, and sequential events that are difficult to compose coherently.
2. **Complex motion synthesis**: Fine-grained character motions (e.g., "ballroom dancing") described in complex scenes are difficult for baseline T2V models to generate directly.
3. **Character customization + sequential events**: Maintaining visual character consistency and temporal motion coherence simultaneously is non-trivial.

Existing SVG methods directly feed scene descriptions as text conditions into T2V models, providing limited constraints, which leads to poor fidelity, missing events/objects, and ambiguous motion.

## Core Problem

How to achieve in story video generation: (1) fine-grained compositional control over multiple objects and events; (2) retrieval-based customized synthesis of complex motions; (3) conflict-free injection of multi-character consistency and sequential motion priors?

## Method

### Overall Architecture

DreamRunner consists of three stages:

**Stage 1: Dual-Level Video Plan Generation**
- **Story-level coarse-grained planning**: GPT-4o generates 6–8 character-driven, motion-rich scene descriptions from the story theme, each containing three structured fields: `scene`, `motions`, and `narrations`.
- **Scene-level fine-grained planning**: Each scene description is decomposed into entity-level layout plans for 6 keyframes in the format `Frame: [entity name, entity motion, entity description], [x0,y0,x1,y1]`, with coordinates normalized to $[0,1]$. Overlapping regions receive merged descriptions.

**Stage 2: Motion Retrieval & Prior Learning**
- **Motion video retrieval** from the InternVid large-scale video database:
  - BM25 text retrieval (400 candidates) → attribute filtering (duration ≥ 2s, frames ≥ 40, aspect ratio ≥ 0.9) → YOLOv5 object tracking and cropping → CLIP + ViCLIP semantic similarity ranking.
  - 4–20 video clips retrieved per motion.
- **Motion prior training**: Test-time fine-tuning based on MotionDirector.
  - Even-indexed layers of CogVideoX's 3D full attention are manually designated as "spatial layers" and odd-indexed layers as "temporal layers."
  - Spatial LoRA learns appearance; temporal LoRA learns motion.
  - **Key innovation**: Per-video prompts (rather than a single shared prompt) are used to help the model disentangle motion from irrelevant background and appearance.
- **Character prior learning**: Reference images are repeated 48 times to construct pseudo-videos; LoRA is injected into spatial layers and only the first frame is reconstructed to prevent overfitting.

**Stage 3: Spatial-Temporal Region-Based Diffusion (SR3AI)**

### Key Designs

**SR3AI (Spatial-Temporal Region-Based 3D Attention and Prior Injection)**

SR3AI regionalizes the 3D full attention of CogVideoX through two mechanisms:

1. **Regional 3D Attention**:
   - Given $N$ region text descriptions $C_1, \ldots, C_N$ and corresponding layouts $L_1, \ldots, L_N$, each condition is encoded to obtain embeddings $T_1, \ldots, T_N$.
   - Attention masking rule: each region visual latent $L_i$ attends to its corresponding text $T_i$ and all visual latents $L_1, \ldots, L_N$; each text embedding $T_i$ attends only to itself and its corresponding $L_i$.
   - This ensures each region is constrained by its own text while visual latents maintain cross-region interaction (soft, rather than hard, isolation).

2. **Regional LoRA Injection**:
   - For each LoRA, a latent mask is computed based on the text description and layout.
   - Formulation: $Wx = W_0x + A_{\text{witch}}B_{\text{witch}}(Mask_{\text{witch}} \cdot x) + A_{\text{cat}}B_{\text{cat}}(Mask_{\text{cat}} \cdot x)$
   - Character LoRAs are injected into spatial layers; motion LoRAs into temporal layers — with no layer-level overlap, avoiding multi-LoRA conflicts.

### Loss & Training

Motion prior training uses two losses:

- **Standard diffusion loss** $L_{org}$: reconstructs all video frames.
$$L_{org} = \mathbb{E}_{z_0, y, \epsilon \sim \mathcal{N}(0,1), t \sim U(0,T)} [\|\epsilon - \epsilon_\theta(z_t, t, y)\|^2]$$

- **Appearance-debiased temporal loss** $L_{ad}$: focuses motion learning in a normalized latent space.
$$\phi(\epsilon) = \frac{\epsilon}{\sqrt{\beta^2 + 1}} - \beta \cdot \epsilon_{anchor}$$
$$L_{ad} = \mathbb{E}[\|\phi(\epsilon) - \phi(\epsilon_\theta(z_t, t, y))\|^2]$$

- Total loss: $L_{motion} = L_{org} + L_{ad}$ (no additional weighting required; simple and robust).

Character prior: LoRA injected into spatial layers, with only the first frame reconstructed. Fine-tuning is conducted without text conditioning; each prior requires approximately 5 min per A6000 GPU.

## Key Experimental Results

**Story Video Generation (DreamStorySet, Table 1)**

| Metric | VideoDirectorGPT | VLogger | DreamRunner | Gain |
|--------|:-:|:-:|:-:|:-:|
| Character CLIP ↑ | 54.3 | 62.5 | **70.7** | +13.1% |
| Character DINO ↑ | 9.5 | 41.3 | **55.1** | +33.4% |
| Fine-Grained CLIP ↑ | 23.7 | 23.5 | **24.7** | +5.1% |
| Fine-Grained ViCLIP ↑ | 21.7 | 23.1 | **23.7** | +2.6% |
| Full Text CLIP ↑ | 22.4 | 22.5 | **24.2** | +7.6% |
| Full Text ViCLIP ↑ | 22.5 | 22.2 | **24.1** | +8.6% |
| Transition DINO ↑ | 63.5 | 73.6 | **93.6** | +27.2% |
| Aesthetics ↑ | 42.3 | 43.4 | **55.4** | +27.6% |
| Imaging ↑ | 60.3 | 61.2 | **62.1** | +1.5% |
| Smoothness ↑ | 94.3 | 96.2 | **98.1** | +2.0% |

**Compositional T2V (T2V-CompBench, Table 2)**

The SR3A module (without LoRA injection) is applied to CogVideoX-2B/5B:
- Dynamic attribute binding improves by over 25%.
- Spatial binding improves by over 15%.
- Motion binding improves by at least 10%.
- CogVideoX-5B + SR3A achieves SoTA among open-source models across 5 dimensions.
- Surpasses all closed-source models (Gen-3, Dreamina, PixVerse, Kling) on dynamic attribute binding, spatial binding, and object interactions.

**RAG + Per-Video Prompt (Table 4)**

| Method | CLIP | ViCLIP |
|--------|:-:|:-:|
| CogVideoX-2B baseline | 23.39 | 20.84 |
| + RAG (single prompt) | 24.01 | 22.02 |
| + RAG (per-video prompt) | **24.67** | **23.04** |

### Ablation Study

**SR3AI + RAG Combined Effect (Table 3)**:
- SR3AI alone → significantly improves event transition smoothness, visual quality, and text alignment (divide-and-conquer effect).
- RAG alone → improves fine-grained and full-text video–text similarity.
- Both combined → optimal across all dimensions.

**RAG Pipeline Ablation (Table 5)**:
- Top-20 retrieval + CLIP/ViCLIP filtering → CLIP 25.47, ViCLIP 23.66 (best).
- Only 3 videos + filtering → CLIP 24.45, ViCLIP 22.80 (insufficient).
- 20 videos without filtering → CLIP 24.01, ViCLIP 22.51 (noisy but still effective).

**Layer Separation Strategy (Table 6)**:
- Interleaved injection (odd/even layers) > front/back half injection > no appearance debiasing.
- Interleaved injection: CLIP 25.5, ViCLIP 23.7.

**Visual Quality (Tables 7–8)**:
- Adding RAG and SR3AI does not degrade visual quality; all metrics improve.
- Overall quality score: 82.55 (DreamRunner) vs. 78.6 (VLogger) vs. 75.3 (VideoDirectorGPT).

**Computational Cost**:
- TTF ≈ 0.2 GPU hours per motion; 3–4 GPU hours per full story.
- Comparison: VLogger ≈ 6K GPU hours; VideoDirectorGPT ≈ 400 GPU hours.

## Highlights & Insights

1. **Retrieval-augmented motion customization** is a key contribution: it reframes motion synthesis as a customization problem, retrieving relevant videos from InternVid for test-time fine-tuning without manual data collection.
2. **Per-video prompting** is a simple yet effective design that helps the model suppress appearance/background noise and focus on motion patterns.
3. **SR3AI's regionalized design** jointly achieves spatial-temporal regional attention and regional LoRA injection without additional training, operating in a zero-shot manner.
4. **Natural spatial/temporal LoRA layer separation** avoids multi-LoRA conflicts through a clean architectural design.
5. DreamRunner surpasses closed-source models on several T2V-CompBench dimensions, demonstrating that open-source models paired with thoughtful design can close the gap.
6. Computational cost is substantially reduced (3–4 GPU hours vs. 6K), yielding strong practical utility.

## Limitations & Future Work

1. **Performance bounded by the backbone**: Built on CogVideoX-2B/5B, DreamRunner inherits the backbone's limitations on rare compositional scenarios or complex motions.
2. **Heuristic layer separation**: Manually designating even layers as spatial and odd layers as temporal lacks theoretical grounding; while ablations confirm effectiveness, generalizability to other architectures remains uncertain.
3. **Retrieval depends on external database coverage**: Motion retrieval relies on InternVid's coverage, which may be insufficient for rare or unusual motions.
4. **Coarse handling of overlapping regions**: Multi-character overlapping regions are processed by merging descriptions via LLM, and quality depends on the LLM's merging capability.
5. **Small self-constructed evaluation set**: DreamStorySet contains only 10 characters and 64 motions; multi-character scenarios are evaluated qualitatively only, limiting the rigor of evaluation.
6. **Sequential TTF per motion**: Although each fine-tuning takes approximately 5 minutes, sequentially processing 15–20 motions still requires several hours.
7. **Short video duration**: Each scene generates only 6-second videos at 8 fps, limiting expressiveness for long narratives.

## Related Work & Insights

| Method | Multi-Object Control | Motion Customization | Spatial-Temporal Region Control | Character Consistency | Training Cost |
|--------|:-:|:-:|:-:|:-:|:-:|
| VideoDirectorGPT | Spatial layout | ✗ | Spatial only | Weak | 400 GPU hrs |
| VLogger | ✗ | ✗ | ✗ | Moderate | 6K GPU hrs |
| Peekaboo | Regional mask | ✗ | Spatial only | — | — |
| TALC | ✗ | ✗ | Temporal only | — | — |
| MotionDirector | ✗ | Single-video FT | ✗ | — | Per video |
| **DreamRunner** | **Regional 3D attention** | **RAG + FT** | **Spatial-temporal joint** | **Strong** | **3–4 GPU hrs** |

Key distinction: DreamRunner is the first to simultaneously realize spatial-temporal regional control (SR3AI) and retrieval-augmented motion customization, achieving conflict-free multi-character multi-motion generation through regionalized LoRA injection.

**Broader implications**:
1. The **RAG paradigm for generative models** is worth adopting in other conditional generation tasks (audio, 3D): rather than training from scratch, retrieving from large-scale databases and performing lightweight adaptation is both effective and efficient.
2. **Regionalized LoRA injection** is generalizable to any multi-concept compositional generation scenario (e.g., multi-style image generation).
3. The success of **per-video prompting** suggests that existing motion learning methods with shared prompts suffer from significant motion–appearance entanglement.
4. The approach is tightly coupled with the 3D full attention characteristic of DiT architectures and may not transfer directly to UNet-based architectures.
5. Stronger future backbones (e.g., Sora-class models) combined with similar fine-grained control modules may yield qualitative leaps in performance.

## Rating ⭐⭐⭐⭐ (4/5)

**Strengths**: The framework is complete and internally coherent (planning → retrieval → injection); experiments are thorough (two tasks + detailed ablations); multiple metrics show substantial gains over baselines; computational efficiency is high.

**Weaknesses**: The self-constructed evaluation set is small and multi-character scenarios receive only qualitative assessment; the odd/even layer separation lacks theoretical justification; the 6-second video limit constrains narrative expressiveness; overall performance remains bounded by CogVideoX backbone capacity.

**Overall**: A strongly engineered and systematic work on story video generation that organically integrates RAG, regional attention, and LoRA customization. While the novelty of individual components is incremental, the combined effect is significant and the experiments are solid.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[CVPR 2026\] Training-free Motion Factorization for Compositional Video Generation](../../CVPR2026/video_generation/training-free_motion_factorization_for_compositional_video_generation.md)
- [\[ICCV 2025\] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](../../ICCV2025/video_generation/motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)
- [\[ICLR 2026\] TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](../../ICLR2026/video_generation/ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](../../ICCV2025/video_generation/etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)

</div>

<!-- RELATED:END -->
