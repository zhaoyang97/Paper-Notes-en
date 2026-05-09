---
title: >-
  [Paper Note] Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement
description: >-
  [ACL 2026][Video Generation][Text-to-Video Generation] This paper proposes VideoRepair, the first training-free, model-agnostic text-to-video self-correction framework that detects fine-grained text-video misalignment via MLLM, preserves correct regions, and selectively repairs problematic regions, consistently improving alignment quality across four T2V backbone models on EvalCrafter and T2V-CompBench.
tags:
  - ACL 2026
  - Video Generation
  - Text-to-Video Generation
  - Self-Correction
  - Localized Refinement
  - Text-Video Alignment
  - Diffusion Model
date: 2025-04-17
content_hash: 7f5733bd2d47694e
---

# Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement

**Conference**: ACL 2026  
**arXiv**: [2411.15115](https://arxiv.org/abs/2411.15115)  
**Code**: [video-repair](https://video-repair.github.io/)  
**Area**: Video Generation  
**Keywords**: Text-to-Video Generation, Self-Correction, Localized Refinement, Text-Video Alignment, Diffusion Model

## TL;DR

This paper proposes VideoRepair, the first training-free, model-agnostic text-to-video self-correction framework that detects fine-grained text-video misalignment via MLLM, preserves correct regions, and selectively repairs problematic regions, consistently improving alignment quality across four T2V backbone models on EvalCrafter and T2V-CompBench.

## Background & Motivation

**State of the Field**: Text-to-video (T2V) diffusion models have made significant advances in generation quality, but still struggle with following complex text prompts — particularly involving multiple objects, attribute binding, and spatial relationships. Common errors include incorrect object counts, confused attribute binding, or region distortion.

**Limitations of Prior Work**: Existing compositional T2V methods improve compositionality but lack explicit feedback mechanisms to detect and correct misalignments. Image-domain repair frameworks suffer from high computational overhead, reliance on external generators, or introduction of visual inconsistencies. The key issue is: even when generated videos contain misaligned portions, correctly generated regions should often be preserved rather than regenerated.

**Root Cause**: Global regeneration wastes correctly generated content, while simple inpainting/editing lacks semantically guided ability to introduce or correct entities that do not match the text. A mechanism is needed that can both precisely locate problematic regions and preserve faithful content.

**Paper Goals**: Design a training-free video repair framework that can automatically detect what is wrong, plan how to fix it, and then locally correct it.

**Starting Point**: Analogous to how humans revise creative works — modifying only erroneous parts while preserving correct ones. Through MLLM-generated fine-grained evaluation questions to identify misaligned regions, then leveraging the diffusion model's own regeneration capability for selective repair.

**Core Idea**: Preserve correct regions, selectively repair erroneous regions — transform MLLM evaluation feedback into actionable generation guidance.

## Method

### Overall Architecture

VideoRepair has three stages: (1) Misalignment Detection: extract semantic tuples from text prompts, generate evaluation question sets, use MLLM binary answers to identify misaligned regions; (2) Refinement Planning: determine entities to preserve and their instance counts, obtain preservation region masks through segmentation models, generate local prompts for regions to be repaired; (3) Localized Refinement: selectively reinitialize noise, apply different text guidance to preserved and repair regions, achieve seamless fusion through joint optimization.

### Key Designs

1. **MLLM-Driven Misalignment Detection**:

    - Function: Automatically identify which video elements do not match the text prompt
    - Mechanism: Extract semantic tuples (entities, attributes, relationships, actions) from the prompt, use LLM to generate evaluation question sets $Q$, divided into counting questions $Q_c$ (e.g., "Is there one bear?") and other questions $Q_{others}$ (attributes, actions, scenes). MLLM answers these questions for the initial video; counting questions return triplets (judgment, prompt count, video count), other questions return binary judgments. Aggregated into a $[0,1]$ alignment score
    - Design Motivation: More fine-grained than simple object existence checks — explicitly capturing quantity, attributes, spatiotemporal relationships, and actions, providing feedback that directly guides repair planning

2. **Region-Preserving Refinement Planning**:

    - Function: Determine what to preserve, what to repair, and what prompts to use for repair
    - Mechanism: (a) MLLM identifies correctly generated key entities $O^*$ and their preservation count $N^*$ based on QA results; (b) Pointing prompts and segmentation models obtain entity binary masks $\mathbf{M}$ per frame; (c) LLM generates local repair prompts $p^r$ excluding already preserved entities
    - Design Motivation: Transforms evaluation feedback into actionable generation guidance — masks precisely define which pixels to preserve and which to regenerate; local prompts ensure repair regions receive correct semantic guidance

3. **Localized Refinement and Fusion**:

    - Function: Repair problematic regions without destroying correct regions
    - Mechanism: Downscale masks to latent space; preserved regions use original noise while repair regions use resampled noise. Each denoising step runs the diffusion model twice: preserved regions with original prompt $p$, repair regions with local prompt $p^r$. Final fusion via joint optimization: $V_1 = \arg\min_{\tilde{V}} \|M_{pres} \otimes (\tilde{V} - \hat{V}_{pres})\|^2 + \|M_{refine} \otimes (\tilde{V} - \hat{V}_{refine})\|^2$, achieving seamless boundary transitions
    - Design Motivation: Pure mask inpainting cannot introduce new entities; pure editing cannot freely correct misalignments; dual-path denoising + joint optimization achieves both precise control and global consistency

### Loss & Training

Entirely training-free, using existing T2V diffusion models for inference. K repair candidate videos are generated (different random seeds), with the best selected via evaluation question scores. BLIP-BLEU score serves as tiebreaker when scores are tied.

## Key Experimental Results

### Main Results

| T2V Backbone | Method | EvalCrafter Avg↑ | Visual Quality | Motion Quality | Temporal Consistency |
|--------|------|------|----------|------|------|
| Wan 2.1-1.3B | Original | 44.83 | 63.2 | 61.0 | 62.1 |
| Wan 2.1-1.3B | + VideoRepair | 49.01 | 65.1 | 61.6 | 62.0 |
| VideoCrafter2 | Original | 45.97 | 61.8 | 62.6 | 62.9 |
| VideoCrafter2 | + VideoRepair | 48.83 | 62.1 | 62.4 | 62.0 |
| CogVideoX-5B | Original | 45.01 | 65.8 | 61.0 | 61.8 |
| CogVideoX-5B | + VideoRepair | 46.41 | 64.8 | 61.1 | 61.9 |

### Ablation Study

| Config | Metric | Note |
|------|---------|------|
| vs LLM paraphrasing | 43.12-45.81 | Simple prompt rephrasing, limited or negative improvement |
| vs SLD | 43.72-47.11 | Effective in some scenarios but severely damages visual/temporal quality |
| vs OPT2I | 45.63-48.69 | Clear improvement but lower than VideoRepair |
| VideoRepair | 46.41-49.01 | Consistently optimal without harming quality metrics |

### Key Findings

- VideoRepair provides consistent improvements across all four T2V backbones, validating model-agnosticism
- The key advantage is not harming visual quality, motion quality, and temporal consistency — while the SLD method sometimes approaches alignment scores, it severely damages these quality metrics (e.g., temporal consistency drops from 62.1 to 21.0)
- Count and Color subcategories show the most significant improvements, precisely the weakest areas of current T2V models

## Highlights & Insights

- **"Preserve correct, repair incorrect" paradigm**: An intuitively natural but technically non-trivial approach — compared to global regeneration or simple inpainting, region-preserving refinement is superior in both efficiency and quality. This paradigm is transferable to any generative task requiring post-processing correction
- **Evaluation-feedback-driven generation**: Directly transforming MLLM evaluation QA results into repair plans (masks + prompts) establishes a closed loop between evaluation and generation. This self-correction paradigm is more scalable than purely human feedback
- **Training-free + model-agnostic**: No additional model training required; can be used plug-and-play with any T2V diffusion model

## Limitations & Future Work

- Requires two diffusion model forward passes (preservation + repair), doubling inference overhead
- Depends on MLLM evaluation accuracy — misalignment state misjudgments may lead to unnecessary modifications or omissions
- Currently supports only single-round repair; iterative repair may lead to error accumulation
- Future exploration: combining with T2V model training for online self-correction, incorporating user interactive feedback

## Related Work & Insights

- **vs SLD/OPT2I**: SLD uses global semantic guidance but severely damages visual quality; OPT2I optimizes prompts but does not perform pixel-level repair; VideoRepair's region-preserving strategy achieves both alignment precision and quality maintenance
- **vs Image repair/editing methods**: Inpainting can only fill regions but cannot introduce new entities; editing cannot freely correct misalignments; VideoRepair's dual-path denoising overcomes both limitations

## Rating

- Novelty: ⭐⭐⭐⭐ First training-free video self-correction framework; region-preserving repair paradigm is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four backbones, two benchmarks, comprehensive ablation and quality metric evaluation
- Writing Quality: ⭐⭐⭐⭐ Three-stage pipeline diagram is clear; method description is systematic
- Value: ⭐⭐⭐⭐ Provides a general and practical post-processing improvement solution for T2V generation

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] OSCBench: Benchmarking Object State Change in Text-to-Video Generation](oscbench_benchmarking_object_state_change_in_text-to-video_generation.md)
- [\[AAAI 2026\] GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection](../../AAAI2026/video_generation/genvidbench_a_6-million_benchmark_for_ai-generated_video_detection.md)
- [\[CVPR 2026\] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](../../CVPR2026/video_generation/infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)
- [\[CVPR 2026\] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](../../CVPR2026/video_generation/from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)
- [\[ICLR 2026\] Language-guided Open-world Video Anomaly Detection under Weak Supervision](../../ICLR2026/video_generation/language-guided_open-world_video_anomaly_detection_under_weak_supervision.md)

<!-- RELATED:END -->
