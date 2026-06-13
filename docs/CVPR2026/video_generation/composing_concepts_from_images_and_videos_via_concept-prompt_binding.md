---
title: >-
  [Paper Note] Composing Concepts from Images and Videos via Concept-prompt Binding
description: >-
  [CVPR 2026][Video Generation][visual concept composition] Bind & Compose (BiCo) is a one-shot method that binds visual concepts to prompt tokens via hierarchical binders and achieves flexible image-video concept composit…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "visual concept composition"
  - "diffusion Transformer"
  - "video personalization"
  - "concept binding"
  - "temporal decoupling"
date: 2026-05-08
content_hash: 0010ee5a0994a710
---

# Composing Concepts from Images and Videos via Concept-prompt Binding

**Conference**: CVPR 2026  
**arXiv**: [2512.09824](https://arxiv.org/abs/2512.09824)  
**Code**: [Project Page](https://refkxh.github.io/BiCo_Webpage)  
**Area**: LLM / NLP (Other)  
**Keywords**: visual concept composition, diffusion Transformer, video personalization, concept binding, temporal decoupling

## TL;DR
Bind & Compose (BiCo) is a one-shot method that binds visual concepts to prompt tokens via hierarchical binders and achieves flexible image-video concept composition through token-level composition, comprehensively outperforming prior work in concept consistency, prompt fidelity, and motion quality.

## Background & Motivation
**Background**: Visual concept composition aims to integrate elements from different images and videos into a coherent output, serving as a fundamental capability for visual creation and filmmaking. With the development of DiT-architecture T2V diffusion models (e.g., Wan2.1), concept localization and customization capabilities have significantly improved.

**Limitations of Prior Work**: (i) Insufficient concept extraction precision — existing methods (LoRA/learnable embedding + masks) struggle to disentangle complex concepts with occlusion and temporal variation, and cannot extract non-object concepts (e.g., style); (ii) Limited image-video concept composition flexibility — prior work is restricted to subject from images + motion from videos, unable to flexibly compose arbitrary attributes (visual style, lighting changes, etc.).

**Key Challenge**: The challenges of precise concept decomposition (without mask input) and cross-modal concept composition (image + video) are mutually coupled.

**Goal**: Enable flexible extraction and composition of arbitrary visual concepts (including non-object concepts such as style and motion) from images and videos.

**Key Insight**: Leverage T2V diffusion models' concept localization capability by binding text tokens to corresponding visual concepts (one-shot training), then compose concepts through token-level composition.

**Core Idea**: First bind visual concepts to prompt tokens (Bind), then select bound tokens from different sources and compose them into the target prompt (Compose), achieved through hierarchical binder structures + diversified absorption mechanism + temporal decoupling strategy.

## Method

### Overall Architecture
BiCo is built on the Wan2.1-T2V-1.3B model, with a two-phase workflow:
1. **Concept Binding**: For each visual input, lightweight binder modules learn the mapping between text tokens and corresponding visual concepts
2. **Concept Composing**: Different parts of the target prompt are updated through corresponding binders, composing multi-source visual information into the updated prompt

Core operation is based on DiT's cross-attention conditional injection:

$$\mathbf{x}_{out} = \text{cross\_attention}(\mathbf{x}_{in}, \mathbf{p}, \mathbf{p})$$

### Key Designs
1. **Hierarchical Binder Structure**: Comprises a global binder $f_g(\cdot)$ and per-block binders $f_l^i(\cdot)$. Each binder is a residual MLP with zero-initialized scaling factors: $f(\mathbf{p}) = \mathbf{p} + \gamma \cdot \text{MLP}(\mathbf{p})$. Since DiT blocks behave differently during denoising, the hierarchical design enables global correlation + targeted fine-tuning. Coupled with a **two-stage reverse training strategy** — first strengthen the global binder at high noise levels ($\geq \alpha$, $\alpha=0.875$), then jointly train all binders.

2. **Diversified-Absorption Mechanism (DAM)**: Addresses concept-token binding precision in the one-shot setting. A VLM (Qwen2.5-VL) extracts spatial and temporal key concepts and generates diversified prompts (keeping key concept words fixed). Learnable **absorption tokens** $p_a^j$ absorb concept-irrelevant visual details during training; these tokens are discarded at inference to suppress unwanted details.

3. **Temporal Decoupling Strategy (TDS)**: Addresses image-video temporal heterogeneity. Video concept training is divided into two stages: Stage 1 trains on single frames (aligned with image concept training), Stage 2 trains on full videos with a dual-branch binder structure:

$$\text{MLP}(\mathbf{p}) \leftarrow (1-g(\mathbf{p})) \cdot \text{MLP}_s(\mathbf{p}) + g(\mathbf{p}) \cdot \text{MLP}_t(\mathbf{p})$$

$\text{MLP}_s$ inherits weights from Stage 1, and $g(\cdot)$ is zero-initialized to ensure proper initialization.

### Loss & Training
Standard diffusion model denoising loss is used to train binders. Each stage runs for 2400 iterations with learning rate $1.0 \times 10^{-4}$. Inference generates 81-frame videos. Experiments are conducted on NVIDIA RTX 4090.

## Key Experimental Results

### Main Results

| Method | CLIP-T↑ | DINO-I↑ | Concept↑ | Prompt↑ | Motion↑ | Overall↑ |
|--------|---------|---------|----------|---------|---------|----------|
| Textual Inversion† | 25.96 | 20.47 | 2.14 | 2.17 | 2.94 | 2.42 |
| DB-LoRA† | 30.25 | 27.74 | 2.76 | 2.76 | 2.51 | 2.68 |
| DreamVideo | 27.43 | 24.15 | 1.90 | 1.82 | 1.66 | 1.79 |
| DualReal | 31.60 | 32.78 | 3.10 | 3.11 | 2.78 | 3.00 |
| **BiCo (Ours)** | **32.66** | **38.04** | **4.71** | **4.76** | **4.46** | **4.64** |

BiCo improves subjective Overall Quality by **+54.67%** over DualReal (3.00 → 4.64).

### Ablation Study

| Configuration | Concept↑ | Prompt↑ | Motion↑ | Overall↑ |
|---------------|----------|---------|---------|----------|
| Baseline (global binder only) | 2.16 | 2.60 | 2.26 | 2.34 |
| + Hierarchical Binder | 2.63 | 2.88 | 2.93 | 2.81 |
| + Prompt Diversification | 3.40 | 3.34 | 3.04 | 3.26 |
| + Absorption Token | 3.55 | 3.43 | 3.43 | 3.47 |
| + TDS (w/o absorption) | 3.80 | 3.97 | 3.70 | 3.82 |
| ▲ w/o Reverse Training | 2.60 | 2.70 | 2.43 | 2.58 |
| **Full Model** | **4.43** | **4.47** | **4.32** | **4.40** |

### Key Findings
- Hierarchical binders significantly improve concept retention and motion quality (Motion: 2.26 → 2.93)
- Absorption tokens effectively suppress unwanted details (ablation visualization shows irrelevant elements appearing when removed)
- TDS is critical for image-video compatibility (Overall: 3.47 → 3.82)
- Two-stage reverse training is indispensable — removing it causes Overall to plummet from 4.40 to 2.58

## Highlights & Insights
- **Unified framework**: First to achieve flexible composition of arbitrary concepts from images and videos, supporting non-object concepts (style, motion)
- **No masks required**: Achieves implicit decomposition through text-conditioned concept composition, lowering the user barrier
- **Scalable design**: Binders are lightweight modules independently trained for different concept sources, composable on demand
- **Rich derivative applications**: Image/video decomposition (retaining only selected tokens), text-guided editing

## Limitations & Future Work
- All tokens are treated equally, but token importance for T2V generation is non-uniform — tokens representing subjects/motion are far more important than function words
- Based on a 1.3B model; scaling effects on larger T2V models (CogVideoX, Sora-scale) remain unverified
- Consistency between automatic metrics (CLIP-T, DINO-I) and human evaluation needs further confirmation
- Computational overhead: each concept source requires independent binder training (2400 iterations × 2 stages)

## Related Work & Insights
- Textual Inversion/DreamBooth-LoRA are foundational video personalization methods but offer coarse concept control granularity
- DreamVideo/DualReal support subject + motion composition but constrain input types and quantities
- TokenVerse achieves prompt-controlled image concept composition but relies on text-conditional modulation architecture, inapplicable to modern T2V models
- Break-A-Scene depends on explicit mask input and cannot extract non-object concepts
- BiCo unifies concept decomposition and composition through the binder + token composition paradigm
- Set-and-Sequence and Grid-LoRA achieve appearance/motion learning in LoRA space but cannot precisely specify concepts and composition methods

## Method Details Supplement
- **VLM Key Concept Extraction**: Qwen2.5-VL extracts spatial concepts (objects, style, lighting, etc.) and temporal concepts (motion patterns, speed changes, etc.), combined into spatial-only and spatiotemporal prompts respectively
- **Inference Process**: The target prompt $\mathbf{p}_d$ is decomposed by concept correspondence, each part updated through corresponding binders, then reassembled into $\mathbf{p}_u^i$
- **Derivative Applications**: Image/video decomposition (retain only dog-related tokens, discard cat-related tokens), text-guided editing (unchanged parts pass through binders, edited parts use original tokens)

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to achieve unified flexible composition of arbitrary image-video concepts
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Quantitative automatic + human evaluation + detailed ablation + visualization cases are comprehensive
- Writing Quality: ⭐⭐⭐⭐ — Concepts are clear; DAM/TDS design motivations are well articulated
- Value: ⭐⭐⭐⭐⭐ — Direct and broad application prospects for visual content creation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Where Concept Erasure Should Occur: Concept-Layer Alignment in Text-to-Video Diffusion Models](../../ICML2026/video_generation/where_concept_erasure_should_occur_concept-layer_alignment_in_text-to-video_diff.md)
- [\[CVPR 2026\] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)
- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](../../ICCV2025/video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)

</div>

<!-- RELATED:END -->
