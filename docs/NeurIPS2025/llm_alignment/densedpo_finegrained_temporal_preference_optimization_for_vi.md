---
title: >-
  [Paper Note] DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models
description: >-
  [NeurIPS 2025 (Spotlight)][LLM Alignment][video diffusion] This paper identifies and addresses the motion bias problem in video DPO — by constructing structurally aligned video pairs via noising and denoising GT videos t…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "LLM Alignment"
  - "video diffusion"
  - "DPO"
  - "motion bias"
  - "segment-level preference"
  - "guided generation"
date: 2026-05-08
content_hash: b9b16fbcc665e7b5
---

# DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2506.03517](https://arxiv.org/abs/2506.03517)  
**Code**: [https://snap-research.github.io/DenseDPO/](https://snap-research.github.io/DenseDPO/)  
**Area**: Video Generation / Preference Optimization
**Keywords**: video diffusion, DPO, motion bias, segment-level preference, guided generation

## TL;DR

This paper identifies and addresses the motion bias problem in video DPO — by constructing structurally aligned video pairs via noising and denoising GT videos to fix the motion dimension, annotating dense preferences at the temporal segment level for more precise learning signals, and leveraging off-the-shelf VLMs for automatic annotation to reduce cost. Using only 1/3 of the annotation data, the method substantially improves motion generation quality while matching visual quality and text alignment.

## Background & Motivation

**Background**: Direct Preference Optimization (DPO) has been successfully applied to post-training alignment of image diffusion models and is beginning to be transferred to video diffusion models. The standard pipeline generates two videos from independent noise, collects human preference annotations, and optimizes the model with a DPO loss.

**Limitations of Prior Work**: Directly transferring image DPO to video exposes a fundamental and previously overlooked flaw — **motion bias**. Since current video models excel at generating high-quality slow-motion videos but frequently produce artifacts in dynamic scenes, annotators systematically prefer static yet clean videos over dynamic but artifact-prone ones when the two are sampled from independent noise. After DPO training, the model further learns that "less motion = better," resulting in a significant drop in motion intensity. This phenomenon has been repeatedly observed at Snap and in multiple prior works.

**Key Challenge**: Video quality is multi-dimensional — visual quality and motion intensity are often negatively correlated. When annotators are asked to assign a single binary preference to an entire video, these dimensions become inevitably entangled. Moreover, human preference judgments over long videos (e.g., 5 seconds) are imprecise — artifacts may only appear in certain temporal segments, but a global label cannot capture such fine-grained distinctions.

**Goal**: (1) Eliminate motion bias in video DPO; (2) Obtain more precise temporally fine-grained preference signals; (3) Reduce the cost of video preference annotation.

**Key Insight**: Drawing on Pareto optimization — to optimize across multiple objectives, one should fix certain attributes and vary others. Specifically, inspired by SDEdit, the method generates video pairs with similar structure (motion-aligned) but differing local details by adding noise to GT videos and then denoising.

**Core Idea**: Use guided sampling to hold motion invariant, annotate dense preferences at the temporal segment level, and allow DPO to optimize only visual quality without harming motion.

## Method

### Overall Architecture

DenseDPO modifies two core components of the standard video DPO pipeline: **data construction** (replacing independent noise sampling with guided sampling) and **annotation granularity** (replacing whole-video binary preferences with segment-level dense preferences). The overall workflow: (1) select high-quality videos from a real-video dataset; (2) add partial noise to each video and denoise with two different random seeds, yielding motion-aligned video pairs with differing local details; (3) segment the videos into short clips (e.g., 1 second) and annotate preferences independently per segment; (4) train the model with a modified DPO loss.

### Key Designs

1. **StructuralDPO: Guided Video Pair Construction (Eliminating Motion Bias)**

    - **Function**: Generate video pairs with similar motion trajectories but differing local visual details.
    - **Mechanism**: Given a GT video $\mathbf{x}$ and a guidance level $\eta \in [0.65, 0.8]$, construct partially noised videos $\mathbf{x}_n^0 = (1-\eta)\mathbf{x} + \eta \boldsymbol{\epsilon}^0$ and $\mathbf{x}_n^1 = (1-\eta)\mathbf{x} + \eta \boldsymbol{\epsilon}^1$, then denoise from step $n = \text{round}(\eta \cdot N)$ to step 1. $\eta$ controls structural similarity — lower $\eta$ implies stronger guidance and more similar videos.
    - **Design Motivation**: Early denoising steps in diffusion models govern global motion and layout, while later steps govern local details. Starting from a partially noised video preserves the GT motion trajectory, causing the two videos to differ only in local details — annotators cannot base their preference on motion quantity and can only compare visual quality. This fundamentally eliminates motion bias.

2. **DenseDPO: Segment-Level Dense Preference Annotation**

    - **Function**: Expand a single preference label for a video pair into independent preference labels for multiple temporal segments.
    - **Mechanism**: Divide a $T$-frame video into $F = \lceil T/s \rceil$ segments (each of $s$ frames, default 1 second) and annotate preferences independently per segment as $\mathbf{l} \in \{-1, +1\}^F$. Because guided sampling establishes a one-to-one temporal correspondence between the two videos, cross-segment comparisons are valid. The DPO loss becomes $\mathcal{L} = -\mathbb{E}\left[\log\sigma\left(-\beta \sum_{f=1}^{F} l_f \cdot (s_f^0 - s_f^1)\right)\right]$.
    - **Design Motivation**: Experiments show that in over 60% of video pairs, preference directions differ across temporal segments — in some segments video A is better, while in others video B is. Whole-video annotation either results in ties or selects "the video with fewer artifacts," failing to exploit this fine-grained information. Segment-level annotation achieves three goals simultaneously: more precise signals, fewer ties (>80% of pairs contain at least one non-tie segment), and more effective training samples.

3. **VLM-Based Automatic Segment-Level Preference Annotation**

    - **Function**: Replace human annotation with off-the-shelf vision-language models (e.g., GPT-o3).
    - **Mechanism**: Feed the corresponding short segments (~1 second) from two videos to a VLM and ask it to judge which is better. The key insight is that while VLMs perform poorly when evaluating long videos (5 seconds), they yield judgments consistent with humans on short segments.
    - **Design Motivation**: Training a dedicated video reward model requires large-scale annotation, whereas DenseDPO reduces the problem to a granularity manageable by VLMs — short-segment comparison. This substantially lowers the barrier to deploying the method.

### Loss & Training

Based on the rectified flow extension of Flow-DPO. $\beta = 500$; the video model is fine-tuned with LoRA rank=128. AdamW optimizer, global batch size=256, trained for 1000 steps. From 55K high-quality videos, 30K prompts are selected; DenseDPO annotates dense preferences for only 10K video pairs.

## Key Experimental Results

### Main Results

Evaluated on VideoJAM-bench (128 high-motion prompts) and MotionBench (419 diverse dynamic prompts), using VBench and VisionReward metrics.

| Method | Dynamic Degree↑ | Visual Quality↑ | Text Alignment↑ | Motion Smoothness↑ | Annotation |
|---|---|---|---|---|---|
| Pre-trained | 84.16 | 0.192 | 0.770 | 92.40 | — |
| SFT | 83.25 | 0.205 | 0.773 | 92.72 | — |
| VanillaDPO | **80.25** ↓ | 0.371 | 0.867 | 93.43 | 10K pairs |
| StructuralDPO | 84.69 | 0.341 | 0.843 | 92.94 | 10K pairs |
| **DenseDPO** | **85.38** | **0.376** | **0.863** | **93.56** | **~3K pairs** |

VanillaDPO's Dynamic Degree drops sharply from 84.16 to 80.25 (direct evidence of motion bias), whereas DenseDPO recovers to 85.38 while matching VanillaDPO on visual quality and text alignment.

### Ablation Study

| Configuration | Dynamic Degree | Visual Quality | Notes |
|---|---|---|---|
| VanillaDPO | 80.25 | 0.371 | Severe motion bias |
| StructuralDPO | 84.69 | 0.341 | Motion restored but quality slightly reduced |
| DenseDPO (human annotation) | 85.38 | 0.376 | Full recovery |
| DenseDPO (GPT-o3 annotation) | ~85 | ~0.37 | Approaches human annotation |
| Segment length s=1s | Optimal | Optimal | Default choice |
| Segment length s=full video | Degrades to Structural | Degrades to Structural | Validates necessity of dense annotation |

### Key Findings

- **Motion bias is the core flaw of video DPO**: VanillaDPO significantly reduces motion intensity across all experiments, representing the cost of directly transferring image DPO. Controlling for motion via guided sampling is the key to resolution.
- **60%+ of video pairs exhibit mixed preferences**: Preference directions differ across temporal segments. Whole-video annotation either discards such samples (tie) or introduces noise — segment-level annotation fundamentally addresses the data utilization problem.
- **VLMs are reliable on short segments**: GPT-o3 shows high agreement with humans on 1-second segment preference judgments but fails on 5-second full videos. DenseDPO reduces preference granularity to within the capability range of VLMs.
- **Significant data efficiency gains**: DenseDPO surpasses VanillaDPO trained on 10K pairs using only approximately 3K effective video pairs (80% of 10K pairs contain non-tie segments, each yielding multiple training signals).

## Highlights & Insights

- **Precise diagnosis of motion bias**: This is a widely observed but previously unsystematically addressed problem. The paper not only defines the issue but also demonstrates through quantitative experiments that VanillaDPO consistently degrades motion across all tests. Correctly identifying the problem is itself a significant contribution.
- **Pareto control-variable design with broad transfer value**: Using SDEdit-inspired guided sampling to "fix motion and compare only visual quality" is a transferable control-variable design applicable to any multi-dimensional preference learning setting — e.g., fixing composition and comparing only color in images, or fixing logic and comparing only expression in text.
- **Segment-level annotation is the correct granularity for video preferences**: The temporal dimension of video, absent in images, necessitates dense signals along the time axis. This observation parallels the evolution from sentence-level to token-level preference learning in NLP.
- **Practical breakthrough for VLM-based automatic annotation**: The approach requires no dedicated video reward model training; it exploits the reliability of off-the-shelf VLMs on short segments, substantially lowering the barrier to deployment.

## Limitations & Future Work

- **Dependence on GT videos**: Guided sampling requires real videos as a starting point, precluding reference-free DPO based on purely generated samples. This is a limitation for scenarios lacking high-quality video data.
- **Content-dependent sensitivity of $\eta$**: The optimal guidance level $\eta$ may vary with base model capability and content type (degree of dynamism); the current uniform range of [0.65, 0.8] may lack sufficient granularity.
- **Unvalidated for long videos (>10s)**: As video length increases, the number of segments grows, annotation and training costs scale linearly, and the independence assumption across segments may no longer hold.
- **VLM annotation may degrade in complex scenes**: In cases involving occlusion, rapid motion, or subtle differences, VLM accuracy may be insufficient.
- **Ceiling of LoRA fine-tuning**: The current approach uses only LoRA fine-tuning; whether full-parameter fine-tuning could further unlock potential remains unexplored.

## Related Work & Insights

- **vs. Diffusion-DPO (Wallace et al., 2023)**: The seminal image DPO work. Direct transfer to video introduces motion bias — DenseDPO is a correction designed specifically for video. The key differences lie in data construction and annotation granularity.
- **vs. VisionReward / VideoAlign (2024)**: These methods train dedicated video reward models for DPO, requiring large-scale annotation. DenseDPO uses off-the-shelf VLMs for training-free annotation of segment-level preferences, making it considerably more lightweight. However, dedicated reward models may be more accurate in complex scenarios.
- **vs. Sentence-level DPO (language models)**: Prior work in language models has explored preference annotation at the sentence rather than full-document level. DenseDPO extends this fine-grained preference idea to the temporal dimension of video, representing a natural cross-modal analogue.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Triple innovation: motion bias diagnosis + guided control-variable construction + segment-level dense annotation, mutually reinforcing
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two benchmarks, multi-dimensional metrics, complete ablations, human vs. VLM annotation comparison
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain from problem diagnosis to solution is exceptionally clear; the Spotlight recognition is well deserved
- Value: ⭐⭐⭐⭐⭐ — Establishes a principled paradigm for video DPO; the guided control-variable design is broadly transferable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](../../CVPR2026/llm_alignment/mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)

</div>

<!-- RELATED:END -->
