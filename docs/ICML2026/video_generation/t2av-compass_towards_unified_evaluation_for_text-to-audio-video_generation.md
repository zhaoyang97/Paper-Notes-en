---
title: >-
  [Paper Note] T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation
description: >-
  [ICML 2026][Video Generation][Text-to-audio-video generation] T2AV-Compass is the first comprehensive evaluation benchmark for text-to-audio-video (T2AV) generation…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Text-to-audio-video generation"
  - "cross-modal alignment"
  - "evaluation benchmark"
  - "MLLM-as-judge"
  - "audiovisual imbalance"
date: 2026-05-08
content_hash: 646c68c3371d5bf2
---

# T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation

**Conference**: ICML 2026  
**arXiv**: [2512.21094](https://arxiv.org/abs/2512.21094)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Benchmark Evaluation  
**Keywords**: Text-to-audio-video generation, cross-modal alignment, evaluation benchmark, MLLM-as-judge, audiovisual imbalance

## TL;DR
T2AV-Compass is the first comprehensive evaluation benchmark for text-to-audio-video (T2AV) generation, featuring 500 complex prompts and a dual-layer evaluation framework (low-level signal metrics + high-level MLLM diagnostics). It systematically evaluates 15 cutting-edge T2AV systems, quantitatively revealing an "audio realism bottleneck" where even top-tier models achieve 85%+ realism in video but only 50% in audio.

## Background & Motivation

**Background**: T2AV generation is at the forefront of multimodal content creation, with breakthrough systems like Sora and Veo emerging. however, evaluation frameworks remain inadequate, mostly reusing unimodal or weak multimodal benchmarks (e.g., VBench for video only, AudioCaps for audio only), which fail to capture true multimodal synergy.

**Limitations of Prior Work**:
- Fragmented capture of cross-modal semantic alignment and temporal synchronization—existing metrics cannot determine if generated sounds correspond to visible events.
- Benchmark datasets are generally short and simplistic, failing to stress-test complex real-world scenarios.
- Evaluation dimensions are fragmented—some focus on vision, others on audio, with almost no end-to-end multidimensional diagnostic framework.
- Evaluation lacks interpretability—it is difficult to attribute specific failure reasons.

**Key Challenge**: T2AV generation requires simultaneous success across multiple axes (perceptual quality, cross-modal alignment, temporal synchronization, instruction following, and physical realism), yet evaluation frameworks often neglect one for another.

**Goal**: Construct the first professional evaluation benchmark for T2AV generation that satisfies both "comprehensiveness" (covering multidimensional evaluation) and "diagnosticity" (interpretable failure analysis).

**Key Insight**: A taxonomy-driven data construction combined with a dual-layer evaluation metric system—utilizing both low-level signal-based objective metrics and high-level semantic MLLM subjective diagnostics.

**Core Idea**: Use structured checklists (QA lists) to transform ambiguous instructions into verifiable constraints, supplemented by physical and knowledge realism checks, unifying technical fidelity, semantic alignment, and instruction following within a single framework.

## Method

### Overall Architecture
Three core components: (1) **Data Construction**—a taxonomy-driven hybrid pipeline generating 500 high-complexity multimodal prompts; (2) **Dual-layer Evaluation Framework**—low-level signal metrics plus high-level MLLM diagnostics; (3) **System Benchmarking**—evaluating 15 frontier T2AV systems to provide dimension-level comparisons and failure mode analysis.

### Key Designs

1. **Taxonomy-driven Multi-source Data Construction**:

    - Function: Ensures semantic coverage, complexity, and physical rationality of the prompt set.
    - Mechanism: (1) Aggregates high-quality community prompts (VidProM / Kling / LMArena / Shot2Story), deduplicated via cosine similarity at 0.8 to obtain 70K; (2) Rewrites prompts using Gemini-2.5-Pro to add visual, motion, sonic, and cinematographic constraints, increasing average length from 54 to 154 tokens and constraints from 5 to 10; (3) Introduces 100 high-fidelity 4-10s YouTube clips for video inversion to ensure alignment with realistic dynamics; (4) Three rounds of manual verification to filter non-compliant, overly long, or irrational prompts.
    - Design Motivation: Avoids pure text hallucinations and enhances real-world representation; multi-source fusion expands semantic space coverage; the taxonomy includes 8 metaphor types, 5 annotation dimensions, and 4 complexity factors.

2. **Dual-layer Evaluation Framework**:

    - Function: End-to-end diagnostics ranging from low-level signal integrity to high-level semantic rationality.
    - Mechanism:
      - **Objective Evaluation**—Video quality VT (DOVER++) + VA (Aesthetic Predictor V2.5); Audio quality AA + SQ (NISQA); Cross-modal alignment T-A (CLAP) + T-V (VideoCLIP-XL-V2) + A-V (ImageBind) + Temporal synchronization (Synchformer).
      - **Subjective Evaluation**—Instruction Following (IF) uses Gemini-2.5 to generate structured QA checklists across 7 dimensions and 17 sub-dimensions; Realism (RE) includes MSS (motion smoothness) / OIS (object integrity) / TCS (temporal coherence) + AAS (audio artifacts) / MTC (texture consistency).
    - Design Motivation: Signal metrics are fast and reliable but coarse-grained; MLLM judgments capture subtle semantics but are time-consuming—the two are complementary.

3. **Structured Checklist + MLLM-as-Judge Protocol**:

    - Function: Maps abstract instructions to verifiable and traceable constraint checks.
    - Mechanism: Automatically generates two types of checklists—"Instruction Following" and "Realism"—from 500 prompts; for each check, the MLLM is required to **output reasoning text before providing a 1-5 score**, saved as JSON for error analysis.
    - Design Motivation: The reasoning-before-scoring protocol enhances diagnostic interpretability and supports fine-grained failure attribution, avoiding credibility issues inherent in black-box scoring.

## Key Experimental Results

### Main Results: Comparison of 15 T2AV Systems

| Method | Open Source | Video Fidelity VT | Video Aesthetics VA | Audio Aesthetics AA | A-V Alignment | Sync DS ↓ | Avg Score |
|------|------|-----------|-----------|-----------|---------|----------|---------|
| Veo-3.1 | ✗ | 13.39 | 5.425 | 6.818 | 0.2856 | 0.6776 | 70.29 |
| Sora-2 | ✗ | 7.568 | 4.112 | 5.584 | 0.2419 | 0.8100 | 69.83 |
| Kling-2.6 | ✗ | 11.41 | 5.417 | 6.666 | 0.2495 | 0.7852 | 68.16 |
| Wan-2.6 | ✗ | 11.87 | 4.605 | 6.440 | 0.2149 | 0.8818 | 67.68 |
| LTX-2 | ✓ | 7.160 | 4.661 | 6.742 | 0.1851 | 0.8756 | 63.72 |
| Ovi-1.1 | ✓ | 9.336 | 4.368 | 6.531 | 0.1620 | 0.9624 | 61.23 |

Closed-source models dominate the top, but no single model leads across all dimensions.

### Dimension Diagnosis: Audiovisual Imbalance

| Config | IF (Video) | IF (Audio) | Video Realism | Audio Realism | Description |
|------|---------------|---------------|---------|---------|------|
| Veo-3.1 | 76.15% | 67.90% | 87.14% | 49.95% | Top models still show severe audio deficiency |
| Kling-2.6 | 73.72% | 63.89% | 87.98% | 47.03% | Excellent video but weak audio |
| Wan-2.2 + Hunyuan-Foley | 74.45% | 58.23% | 89.63% | 62.14% | Cascade pipeline: Great video but broken A-V alignment |
| AudioLDM2 + MTV | 68.30% | 65.80% | 76.45% | 58.92% | Pure AV synthesis lags behind |

### Key Findings
- **Audio Realism Bottleneck**: A gap of 30-50 points exists between video and audio realism, revealing a structural **audiovisual imbalance**—even when top models reach 85%+ in video integrity/temporal stability, audio remains at ~50%.
- **Dynamic Instruction Following is Most Challenging**: The "Dynamics" dimension is the most discriminative in video instruction following; frontier models lose significant points when executing complex motions and interactions (temporal coherence bottleneck).
- **Sound Effect Synthesis is the Weakest Link**: Sound Effects (SFX) is the most error-prone sub-category in audio instruction following; models struggle to associate diverse physical sound events with prompts and visual events.
- **Cascade Pipeline Disparity**: Cascade T2V → V2A pipelines can compete with end-to-end models in unimodal quality (Wan-2.2 + Hunyuan-Foley reaches 89.63 in video realism), but suffer from severe lags in global A-V alignment ("fragmented optimization").

## Highlights & Insights
- **Systematic Multidimensional Diagnostic System**: Integrates low-level signal metrics (DOVER++, CLAP, Synchformer) with high-level semantic verification (MLLM reasoning-based scoring) within a unified framework, upgrading failure attribution from "vague scores" to "traceable root causes."
- **Taxonomy-driven Data Construction**: Systematically designs constraints across cinematographic, physical causality, and acoustic environment dimensions, followed by LLM rewriting and three-round manual verification; this pipeline is reusable for other evaluation benchmarks.
- **Quantitative Reveal of Audiovisual Imbalance**: Explicitly characterizes why AV generation has not reached human levels—it is not that all dimensions lag, but rather a structural "bottleneck" in audio.
- **Interpretable Reasoning-before-Scoring**: Mandatory textual reasoning before MLLM judgment avoids the credibility issues of black-box scoring.

## Limitations & Future Work
- While 500 prompts are rich, they do not cover all possible scenarios (e.g., long videos > 10s, unconventional AV interactions).
- MLLM-as-Judge still exhibits MLLM bias and instability.
- Audio evaluation is not deep enough—lacking fine-grained metrics like spatial audio or spectral distortion.
- Many evaluation metrics rely on closed-source LLMs, limiting reproducibility; manual verification across languages and cultures is still needed.
- Temporal sync metrics assume single AV event alignment, potentially failing for complex multi-source soundscapes.

## Related Work & Insights
- **vs VBench / EvalCrafter**: These only evaluate video quality and text alignment, completely ignoring audio; T2AV-Compass promotes audio to a first-class citizen.
- **vs JavisBench / VABench**: While these involve joint AV evaluation, this work represents a qualitative leap in prompt complexity (154 vs 50-68 tokens), metric systematicity, and diagnostic depth.
- **vs AudioCaps / TTA-Bench**: These are pure audio benchmarks and cannot characterize multimodal synergy.
- **Insights**: The correct path for establishing professional benchmarks is "taxonomy-driven data design + hybrid objective/subjective metrics + interpretable diagnostic systems," which serves as a reference for other generative tasks (3D generation, controllable text generation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First comprehensive evaluation benchmark for T2AV generation; structured QA paradigm for MLLM-as-Judge + quantitative reveal of "audiovisual imbalance."
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 15 representative systems with detailed diagnostics; lacks large-scale correlation validation with human annotations and metric stability analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, excellent charts, and refined details; multiple contributions are presented clearly.
- Value: ⭐⭐⭐⭐⭐ Provides the first professional open evaluation framework for subsequent T2AV research; released code and data are expected to drive long-term progress in the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation](../../ICLR2026/video_generation/javisdit_unified_modeling_and_optimization_for_joint_audio-video_generation.md)
- [\[CVPR 2026\] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](../../CVPR2026/video_generation/unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)
- [\[ICCV 2025\] WorldScore: A Unified Evaluation Benchmark for World Generation](../../ICCV2025/video_generation/worldscore_a_unified_evaluation_benchmark_for_world_generation.md)
- [\[CVPR 2026\] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](../../CVPR2026/video_generation/uniavgen_unified_audio_and_video_generation_with_asymmetric_cross-modal_interact.md)
- [\[CVPR 2026\] SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](../../CVPR2026/video_generation/slvmeval_synthetic_meta_evaluation_benchmark_for_text-to-long_video_generation.md)

</div>

<!-- RELATED:END -->
