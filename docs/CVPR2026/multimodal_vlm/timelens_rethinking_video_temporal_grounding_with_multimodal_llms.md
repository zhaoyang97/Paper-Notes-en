---
title: >-
  [Paper Note] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs
description: >-
  [CVPR 2026][Multimodal VLM][video temporal grounding] This paper systematically investigates the key factors for building video temporal grounding (VTG) capabilities in MLLMs from two dimensions — data quality and algori…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "video temporal grounding"
  - "data quality"
  - "RLVR"
  - "timestamp encoding"
  - "benchmark refinement"
date: 2026-05-08
content_hash: 6a06384d90df8733
---

# TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

**Conference**: CVPR 2026
**arXiv**: [2512.14698](https://arxiv.org/abs/2512.14698)
**Code**: [timelens-arc-lab.github.io](https://timelens-arc-lab.github.io/)
**Area**: Video Temporal Grounding / Multimodal LLM
**Keywords**: video temporal grounding, data quality, RLVR, timestamp encoding, benchmark refinement

## TL;DR

This paper systematically investigates the key factors for building video temporal grounding (VTG) capabilities in MLLMs from two dimensions — data quality and algorithm design. It releases the high-quality benchmark TimeLens-Bench and training set TimeLens-100K, and constructs the TimeLens model series via interleaved textual timestamp encoding combined with a thinking-free RLVR training paradigm, achieving state-of-the-art performance among open-source models and surpassing GPT-5 and Gemini-2.5-Flash.

## Background & Motivation

**Background**: MLLMs excel at "what" understanding but are critically deficient in "when" understanding. VTG — localizing temporal segments in a video given a text query — is the core task for establishing temporal awareness, yet research approaches are highly fragmented with no unified best practices.

**Limitations of Prior Work**:

1. Existing VTG benchmarks suffer from serious quality issues: 20.6% of samples in Charades-STA violate query uniqueness, and 34.9% exhibit annotation precision problems; multiple datasets contain errors such as non-existent events, ambiguous queries, and information leakage.
2. Different open-source methods use heterogeneous training data and experimental setups, making fair comparison of design choices (e.g., temporal encoding, training strategies) infeasible.
3. Training data (aggregated from multiple source datasets) exhibits even higher error rates than the evaluation benchmarks.

**Key Challenge**: Model rankings shift dramatically after benchmark correction — open-source models that outperform GPT-5 on the original benchmark are completely reversed after fixing — demonstrating that prior evaluation standards are unreliable.

**Goal**: Establish a reliable data foundation for VTG and systematically explore optimal algorithmic design principles.

**Key Insight**: Rather than introducing new complex methods, this work conducts incremental but necessary systematic baseline studies along two axes: data quality and algorithm design.

**Core Idea**: Data quality refinement + interleaved textual timestamp encoding + thinking-free RLVR = a simple yet optimal solution for VTG.

## Method

### Overall Architecture

The work proceeds along two dimensions: data quality and algorithm design. On the data side: diagnose and refine three mainstream benchmarks → release TimeLens-Bench + automatically re-annotate training data → TimeLens-100K. On the algorithm side: systematically compare timestamp encoding schemes → training paradigms → RLVR training recipes → ultimately build TimeLens-7B/8B.

### Key Designs

1. **TimeLens-Bench: High-Quality Evaluation Benchmark**

    - Defines 6 strict annotation criteria: query clarity/uniqueness, event existence, avoidance of information leakage, annotation precision/completeness.
    - Diagnose-then-Refine workflow: the same annotator handles both error detection and correction to balance efficiency and quality.
    - Multi-round cross-validation with quality control: each batch is reviewed by different annotators; batches exceeding an error threshold are entirely reworked.
    - Produces Charades-TimeLens / ActivityNet-TimeLens / QVHighlights-TimeLens.

2. **Interleaved Textual Timestamp Encoding**

    - Compares three encoding families: position-encoding-based (MRoPE, etc.), visual overlay (rendering timestamps directly on frames), and textual encoding (interleaved vs. non-interleaved).
    - Each scheme is further compared across two time formats: raw timestamps (e.g., "10.2s") vs. frame indices (e.g., "1, 2, 3").
    - Conclusion: interleaved textual prefix with raw timestamps is optimal (mIoU: Charades 48.3, ActivityNet 43.1, QVHighlights 56.7), substantially outperforming position-encoding-based approaches (36.6, 33.1, 49.2), and requires no architectural modifications.

3. **Thinking-Free RLVR Training Paradigm**

    - Systematically compares four paradigms: SFT / thinking-based RLVR / SFT + thinking-free RLVR / pure thinking-free RLVR.
    - Core finding: VTG is fundamentally a perception task rather than a reasoning task; explicit thinking processes are detrimental.
    - Pure thinking-free RLVR achieves the best performance with 1.0× training time (~4h10m on 8×H20 GPUs).
    - A preceding SFT stage provides no significant benefit (SFT+RLVR at 2.9× time vs. pure RLVR at 1.0× time, with comparable performance).

### Loss & Training

- GRPO optimizer with verifiable rewards based on temporal segment IoU; no Chain-of-Thought is used.
- **Early stopping strategy**: training is halted when both IoU reward and within-group reward standard deviation plateau simultaneously; continued training leads to performance degradation.
- **Difficulty-based data sampling**: the model being trained performs offline inference on training data to compute IoU-based difficulty scores; Gaussian sampling biases toward hard samples (performance saturates when mean > 0.75); approximately 12K samples suffice.
- TimeLens-7B is based on Qwen2.5-VL-7B; TimeLens-8B is based on Qwen3-VL-8B.

## Key Experimental Results

### Main Results

mIoU comparison on TimeLens-Bench:

| Model | Charades | ActivityNet | QVHighlights | Type |
|-------|----------|-------------|--------------|------|
| GPT-4o | 41.8 | 40.4 | 52.1 | Commercial |
| GPT-5 | 40.5 | 42.9 | 56.8 | Commercial |
| Gemini-2.5-Flash | 48.6 | 52.5 | 64.3 | Commercial |
| Gemini-2.5-Pro | 52.8 | 58.1 | 70.4 | Commercial |
| Time-R1-7B | 36.6 | 33.1 | 49.2 | Open-source |
| MiMo-VL-7B | 39.6 | 35.5 | 41.5 | Open-source |
| Qwen2.5-VL-7B (baseline) | 39.3 | 31.4 | 31.6 | Open-source |
| **TimeLens-7B** | **48.8** | **46.2** | **56.0** | Open-source |
| Qwen3-VL-8B (baseline) | 48.3 | 46.8 | 59.4 | Open-source |
| **TimeLens-8B** | **55.2** | **53.2** | **65.5** | Open-source |

### Ablation Study

**Training paradigm comparison** (trained on TimeLens-100K):

| Training Paradigm | Charades mIoU | ActivityNet mIoU | QVHighlights mIoU | Training Time |
|-------------------|---------------|------------------|-------------------|---------------|
| SFT (32K) | 47.4 | 39.9 | 52.0 | 1.0× |
| SFT (100K) | 48.6 | 39.7 | 49.0 | 2.4× |
| Thinking-based RLVR | 42.7 | 41.2 | 57.8 | 1.9× |
| SFT + Thinking-free RLVR | 50.1 | 42.7 | 55.9 | 2.9× |
| **Thinking-free RLVR** | **48.3** | **43.1** | **56.7** | **1.0×** |

### Key Findings

- TimeLens-8B achieves mIoU of 55.2/53.2/65.5 across three benchmarks, surpassing GPT-5 (40.5/42.9/56.8) and Gemini-2.5-Flash (48.6/52.5/64.3).
- Open-source models appear stronger on the original benchmarks, but rankings are dramatically reversed after correction — confirming the unreliability of original benchmarks.
- Thinking-free RLVR achieves the best or near-best performance with minimal training time (1.0×); explicit thinking degrades Charades mIoU (42.7 vs. 48.3).
- Interleaved textual encoding consistently outperforms visual overlay and position-encoding approaches across all three benchmarks.
- Early stopping and difficulty-based sampling each contribute approximately 1–2 mIoU improvement while saving over 50% of training time.

## Highlights & Insights

- The positioning of this work as "necessary baselines rather than a new method" is remarkably candid; the benchmark refinement effort is substantial, and its impact far exceeds that of a typical methodology paper.
- The ranking reversal following benchmark correction is the most striking finding in the paper — implying that prior comparative conclusions drawn from the original benchmarks need to be revisited.
- The finding that "VTG is perception rather than reasoning" is counterintuitive: CoT/thinking is not only unhelpful for VTG but actively harmful.
- The two RLVR training insights (early stopping + difficulty-based sampling) have broad applicability to other tasks with verifiable rewards.
- The superiority of interleaved textual encoding demonstrates that simple approaches combined with high-quality data outperform complex architectural modifications.

## Limitations & Future Work

- Benchmark refinement requires substantial human involvement (annotator training, cross-validation), limiting scalability.
- Thinking-free RLVR may not generalize to more complex temporal reasoning tasks (e.g., event localization requiring causal inference).
- Validation is limited to Qwen2.5-VL/Qwen3-VL; the transferability of best practices to architectures such as InternVL and LLaVA remains to be examined.
- The quality gap between automatic re-annotation in TimeLens-100K and human annotation has not been quantitatively analyzed.
- Multi-granularity temporal localization (e.g., joint moment retrieval and video summarization) is not explored.

## Related Work & Insights

- **vs. Time-R1**: Both employ RLVR, but Time-R1 uses thinking-based RLVR and achieves only 36.6/33.1/49.2 mIoU, far below TimeLens's 48.8/46.2/56.0; the gap stems from data quality and the thinking-free design.
- **vs. TRACE/TRACE-uni**: Dedicated VTG models achieving only 27.1–28.1/32.7–33.6/39.0–39.8 mIoU, substantially inferior to approaches built on strong MLLMs.
- **vs. TimeSuite**: Another systematic VTG approach; its ActivityNet mIoU of only 19.8 suggests that data and training strategy matter more than model design.
- Takeaway: The research paradigm of data quality refinement → fair evaluation → best practice establishment is worth emulating in other tasks such as detection and segmentation.

## Rating

- Novelty: ⭐⭐⭐ The method itself is incremental; the value lies in its systematic nature rather than a single innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three encoding families × two formats + four training paradigms + thorough RLVR recipe exploration — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, with every finding supported by sufficient experiments; the ranking-reversal visualization in Fig. 2(a) is highly persuasive.
- Value: ⭐⭐⭐⭐⭐ Benchmark refinement and best practices are extremely valuable to the VTG community; TimeLens-Bench is poised to become the new standard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [\[ICCV 2025\] Enrich and Detect: Video Temporal Grounding with Multimodal LLMs](../../ICCV2025/multimodal_vlm/enrich_and_detect_video_temporal_grounding_with_multimodal_llms.md)
- [\[CVPR 2026\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](personavlm_long_term_personalized_multimodal_llms.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)

</div>

<!-- RELATED:END -->
