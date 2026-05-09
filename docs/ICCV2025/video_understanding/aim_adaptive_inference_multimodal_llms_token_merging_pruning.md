---
title: >-
  [Paper Note] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
description: >-
  [ICCV 2025][Video Understanding][Multimodal LLM] This paper proposes AIM, a training-free adaptive inference method for multimodal LLMs that achieves a 6.8× FLOPs reduction while maintaining performance, through similarity-based iterative visual token merging before the LLM and progressive PageRank-based token pruning within LLM layers. Under equal compute budgets, AIM even surpasses SOTA on long video understanding (+4.6 MLVU).
tags:
  - ICCV 2025
  - Video Understanding
  - Multimodal LLM
  - adaptive inference
  - token merging
  - token pruning
  - video understanding efficiency
date: 2026-05-08
content_hash: 5817caaca057e3c7
---

# AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

**Conference**: ICCV 2025
**arXiv**: [2412.03248](https://arxiv.org/abs/2412.03248)
**Code**: [https://github.com/LaVi-Lab/AIM](https://github.com/LaVi-Lab/AIM)
**Area**: Video Understanding
**Keywords**: Multimodal LLM, adaptive inference, token merging, token pruning, video understanding efficiency

## TL;DR
This paper proposes AIM, a training-free adaptive inference method for multimodal LLMs that achieves a 6.8× FLOPs reduction while maintaining performance, through similarity-based iterative visual token merging before the LLM and progressive PageRank-based token pruning within LLM layers. Under equal compute budgets, AIM even surpasses SOTA on long video understanding (+4.6 MLVU).

## Background & Motivation

**Background**: Multimodal LLMs rely on large numbers of visual tokens (up to thousands for video), incurring substantial computational overhead that limits real-time deployment and long-video processing.

**Limitations of Prior Work**: Methods such as FastV and PDrop prune only at specific LLM layers, lacking flexibility; LLaVA-Prumerge operates only before the LLM. None can adaptively accommodate varying computational budgets.

**Core Idea**: Merge similar tokens before the LLM to reduce redundancy, and progressively prune unimportant tokens within LLM layers — two tunable knobs that offer flexible control over compute.

## Method

### Key Designs

1. **Pre-LLM Token Merging**: Adjacent visual tokens are partitioned into sets A and B based on cosine similarity; the most similar pairs are identified and averaged. For video, merging is performed within frames (cross-frame merging would disrupt temporal order).

2. **Progressive In-Layer Token Pruning**: The PageRank algorithm is applied to the self-attention weight matrix to compute an importance score for each token. Only visual tokens are pruned; text tokens are retained (pruning text tokens causes severe performance degradation).

3. **Piecewise Linear Scheduler**: All tokens are retained for the first $l_1$ layers; token count linearly decreases from layer $l_1$ to $l_2$; visual tokens are fully removed beyond layer $l_2$. Early layers are responsible for cross-modal fusion (cannot be pruned aggressively), while later layers shift toward text reasoning (can be pruned substantially).

## Key Experimental Results

| Model | FLOPs (TB) | VideoMME | MLVU |
|-------|-----------|----------|------|
| LLaVA-OV-7B | 99.63 | 58.2 | 64.7 |
| AIM | 14.67 | 57.4 | **69.3** |
| FastV | 21.24 | 50.1 | 54.1 |

### Key Findings
- Retaining only 25% of visual tokens is sufficient to maintain near-full performance.
- Fewer tokens per frame enables processing more frames, which actually improves long-video understanding.
- Pruning visual tokens in early layers severely degrades performance, whereas aggressive pruning in later layers has minimal impact.

### Piecewise Linear Scheduler Parameter Effects

| $l_1$ | $l_2$ | Retention Ratio | VideoMME | FLOPs (TB) |
|-------|-------|----------------|---------|-----------|
| 4 | 20 | 25% | 57.4 | 14.67 |
| 8 | 24 | 25% | 56.8 | 15.23 |
| 4 | 20 | 50% | 58.0 | 28.45 |
| 0 | 16 | 25% | 52.1 | 12.34 |

### Results Across Different Models

| Model | Original FLOPs | AIM FLOPs | Performance Retention |
|-------|---------------|----------|----------------------|
| LLaVA-OV-7B | 99.6 TB | 14.7 TB | 98.6% |
| Qwen2-VL-7B | 85.3 TB | 12.5 TB | 97.8% |

## Highlights & Insights
- The counterintuitive finding that "reducing tokens can improve long-video performance" is particularly valuable: fewer tokens × more frames > more tokens × fewer frames.
- PageRank-based token importance estimation is more stable than naive attention weight aggregation.

## Limitations & Future Work
- Scheduler parameters ($l_1$, $l_2$, retention ratio) require manual selection; an automatic tuning mechanism is absent.
- Training-free methods may have a performance ceiling below that of training-based approaches, with potentially significant degradation under extreme compression.
- The computational overhead of the PageRank algorithm is not thoroughly analyzed and may partially offset the speedup from token pruning.
- Applicability to spoken-language multimodal models (e.g., audio-visual LLMs) remains unexplored.
- Pruning only visual tokens while retaining all text tokens may not be optimal for visually dominated tasks.
- The balance between intra-frame and cross-frame merging lacks systematic investigation.
- Integration with complementary acceleration techniques such as model quantization and knowledge distillation has not been explored.

## Related Work & Insights
- **vs. FastV/PDrop**: These methods prune only at specific LLM layers and lack flexibility; AIM operates both before the LLM and within LLM layers simultaneously.
- **vs. LLaVA-Prumerge**: Operates only before the LLM; AIM additionally introduces progressive in-layer pruning.
- **vs. ToMe**: ToMe is a general-purpose token merging approach; AIM incorporates intra-frame constraints and PageRank-based importance scoring tailored to video scenarios.

### Additional Discussion
- The core innovation lies in extending the problem analysis from a single dimension to multiple dimensions, enabling a more comprehensive understanding.
- The experimental design covers diverse scenarios and baseline comparisons, with statistically significant results.
- The modular design of the method facilitates extension to related tasks and new datasets.
- Open-sourcing the code and data is of considerable value to the community for reproduction and follow-up research.
- Compared to concurrent work, this paper demonstrates greater depth in problem formulation and more comprehensive experimental analysis.
- The paper is logically structured, forming a complete loop from problem definition to method design to experimental validation.
- The computational overhead of the method is reasonable, making it deployable in practical applications.
- Future work may consider integration with additional modalities such as audio and 3D point clouds.
- Validating the scalability of the method on larger datasets and models is an important subsequent direction.
- Combining the method with reinforcement learning for end-to-end optimization is worth exploring.
- Cross-domain transfer is a direction worth investigating — the generalizability of the method requires further validation.
- A lightweight variant of the method tailored for edge computing and mobile deployment scenarios warrants further study.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-stage merge-then-prune design is original
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers video and image benchmarks with thorough ablations
- Writing Quality: ⭐⭐⭐⭐ In-depth analysis with clear insights
- Value: ⭐⭐⭐⭐⭐ Substantial practical value for real-world deployment

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[ICCV 2025\] Breaking the Encoder Barrier for Seamless Video-Language Understanding](breaking_the_encoder_barrier_for_seamless_video-language_understanding.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding](4d-bench_benchmarking_multi-modal_large_language_models_for_4d_object_understand.md)

<!-- RELATED:END -->
