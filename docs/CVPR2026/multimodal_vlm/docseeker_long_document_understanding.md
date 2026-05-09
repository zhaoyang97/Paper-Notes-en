---
title: >-
  [Paper Note] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding
description: >-
  [CVPR 2026][Multimodal VLM][Long document understanding] DocSeeker is proposed to achieve structured reasoning and evidence grounding in long document understanding via an ALR (Analyze–Locate–Reason) visual reasoning paradigm combined with two-stage training (SFT + EviGRPO). The model is trained exclusively on short documents yet generalizes robustly to documents of extreme length.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Long document understanding
  - evidence grounding
  - structured reasoning
  - reinforcement learning
  - visual RAG
date: 2026-05-08
content_hash: 4332bc0b12c3dd45
---

# DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding

**Conference**: CVPR 2026
**arXiv**: [2604.12812](https://arxiv.org/abs/2604.12812)
**Code**: [https://github.com/yh-hust/DocSeeker](https://github.com/yh-hust/DocSeeker)
**Area**: Multimodal VLM / Document Understanding
**Keywords**: Long document understanding, evidence grounding, structured reasoning, reinforcement learning, visual RAG

## TL;DR

DocSeeker is proposed to achieve structured reasoning and evidence grounding in long document understanding via an ALR (Analyze–Locate–Reason) visual reasoning paradigm combined with two-stage training (SFT + EviGRPO). The model is trained exclusively on short documents yet generalizes robustly to documents of extreme length.

## Background & Motivation

**Background**: MLLMs suffer severe performance degradation on long-document VQA as document length increases. Pure-vision approaches treat each page as an image input to avoid OCR error propagation.

**Limitations of Prior Work**: (1) Low signal-to-noise ratio: key evidence is buried among a large number of irrelevant pages; (2) Supervision scarcity: datasets provide only short final answers without intermediate reasoning steps. Visual RAG faces a Top-$k$ dilemma—large $k$ introduces noise while small $k$ misses evidence.

**Key Challenge**: Models learn fragile shortcuts (memorization) rather than genuine reasoning ability, resulting in poor interpretability and weak OOD generalization.

**Goal**: Train models to adopt a structured "locate first, then reason" workflow rather than directly predicting answers.

**Key Insight**: Inspired by human cognitive processes—first analyze intent, then locate evidence, and finally reason.

**Core Idea**: The ALR paradigm requires the model to explicitly produce a structured chain of thought following the "Analyze → Locate → Reason" sequence, combined with two-stage training via SFT and evidence-aware GRPO.

## Method

### Overall Architecture

Built upon Qwen-2.5-VL-7B, with page IDs prefixed to each page as pointers. The output is constrained to follow the ALR structure: $\mathbf{Y} = (\mathbf{Y}_A \oplus \mathbf{Y}_L \oplus \mathbf{Y}_R) \oplus (\mathbf{Y}_E \oplus \mathbf{Y}_F)$, comprising question analysis, evidence localization (with cited page numbers), reasoning process, evidence page list, and final answer.

### Key Designs

1. **ALR Visual Reasoning Paradigm**:

    - **Function**: A structured "locate first, then reason" workflow.
    - **Mechanism**: Page-aware input (interleaved page IDs and visual tokens) combined with three-stage structured output. The model must first analyze user intent, then scan the document to locate relevant pages with justification, and finally synthesize reasoning from the grounded evidence.
    - **Design Motivation**: Enforced evidence localization compels the model to distinguish visual tokens across different pages, counteracting interference caused by long visual inputs.

2. **Evidence-Aware GRPO (EviGRPO)**:

    - **Function**: Jointly optimizes evidence localization and reasoning via reinforcement learning.
    - **Mechanism**: A multi-dimensional reward function $R = \lambda_1 R_{format} + \lambda_2 R_{evidence} + \lambda_3 R_{answer}$. The format reward enforces the ALR template; the evidence reward evaluates page localization accuracy using a weighted F1 score ($\beta > 1$, biased toward recall); the answer reward evaluates the final answer using ANLS.
    - **Design Motivation**: Reasoning paths produced by SFT tend to be suboptimal; RL enables the model to learn directly from outcome signals, surpassing imitation learning.

3. **Evidence-Guided Resolution Allocation (EGRA)**:

    - **Function**: Supports longer document inputs during training.
    - **Mechanism**: Evidence pages are kept at high resolution; 70% of non-evidence pages are downsampled (1024→256) while the remaining 30% retain high resolution. At inference time, all pages are processed at high resolution.
    - **Design Motivation**: This not only alleviates GPU memory constraints but also improves the signal-to-noise ratio of training data—outperforming the strategy of simply discarding non-evidence pages.

### Loss & Training

Stage I: Standard cross-entropy SFT on 13,986 ALR CoT samples distilled from Gemini-2.5-Flash. Stage II: EviGRPO (rollout group size 16; format/evidence/answer reward weights of 0.1/0.3/0.6). Training is conducted exclusively on documents with $\leq 20$ pages.

## Key Experimental Results

### Main Results

| Method | Params | DUDE↑ | MPDocVQA↑ | MMLong↑ | LongDocURL↑ |
|--------|--------|-------|-----------|---------|------------|
| Baseline | 7B | 35.2 | 70.1 | 25.4 | 37.8 |
| InternVL3 | 8B | 47.4 | 80.8 | 24.1 | 38.7 |
| GPT-4o | — | 54.1 | 67.4 | 42.8 | 64.5 |
| **DocSeeker** | **7B** | **56.8** | **87.2** | **48.5** | **58.3** |

### Ablation Study

| Configuration | DUDE | MPDocVQA | Notes |
|---------------|------|----------|-------|
| Full DocSeeker | 56.8 | 87.2 | SFT + EviGRPO + EGRA |
| SFT only | 52.1 | 84.5 | No RL |
| SFT + GRPO (no evidence reward) | 54.3 | 85.8 | Standard GRPO |
| Without EGRA | 50.8 | 82.1 | Uniform resolution |

### Key Findings

- Improvements of 30–60% over the baseline demonstrate the effectiveness of the ALR paradigm.
- Trained exclusively on documents with $\leq 20$ pages, the model generalizes robustly to documents of up to 468 pages.
- DocSeeker's localization capability naturally complements visual RAG and can serve as a backbone model for RAG systems.

## Highlights & Insights

- "Train short, generalize long" is a surprising result: the ALR paradigm learns transferable reasoning skills rather than memorized patterns.
- The EGRA strategy is simple yet effective: differentiated resolution allocation simultaneously reduces memory consumption and improves signal-to-noise ratio, outperforming the strategy of discarding non-evidence pages.
- The evidence-aware reward design makes the RL stage more targeted and purposeful.

## Limitations & Future Work

- Training data is drawn solely from MP-DocVQA and DUDE, limiting domain coverage.
- Reliance on Gemini-2.5-Flash for distillation constrains data quality to the teacher model's capability.
- The pure-vision approach still has limitations on dense-text pages.
- The framework is extensible to multi-document cross-document reasoning.

## Related Work & Insights

- **vs. VisRAG/SV-RAG**: These are retrieval-augmented methods; the ALR paradigm endows DocSeeker as an end-to-end approach with explicit localization capability.
- **vs. mPLUG-DocOwl2**: DocOwl2 employs visual token compression, whereas DocSeeker applies differentiated resolution via EGRA.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Both the ALR paradigm and EviGRPO represent significant innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive in-domain and out-of-domain evaluation with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Method and experiments are articulated clearly.
- **Value**: ⭐⭐⭐⭐⭐ Substantial advancement for long document understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[CVPR 2026\] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)
- [\[NeurIPS 2025\] iFinder: Structured Zero-Shot VLM Grounding for Dash-Cam Video Reasoning](../../NeurIPS2025/multimodal_vlm/ifinder_structured_zero-shot_vision-based_llm_grounding_for_dash-cam_video_reaso.md)

</div>

<!-- RELATED:END -->
