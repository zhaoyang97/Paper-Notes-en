---
title: >-
  [Paper Note] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism
description: >-
  [CVPR2026][Multimodal VLM][Long video understanding] This paper proposes FlexMem — a training-free visual memory mechanism that constructs a visual memory bank via iterative dual-pathway KV cache compression…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Long video understanding"
  - "visual memory"
  - "KV cache compression"
  - "training-free"
  - "streaming video"
date: 2026-05-08
content_hash: 38ab3408df844d56
---

# Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism

**Conference**: CVPR2026
**arXiv**: [2603.29252](https://arxiv.org/abs/2603.29252)  
**Code**: [FlexMem](https://github.com/FlexMem)  
**Area**: Multimodal VLM
**Keywords**: Long video understanding, visual memory, KV cache compression, training-free, streaming video

## TL;DR
This paper proposes FlexMem — a training-free visual memory mechanism that constructs a visual memory bank via iterative dual-pathway KV cache compression, and introduces both encoding-based and fast index-based memory retrieval strategies, enabling MLLMs to process 1000+ frame long videos on a single 3090 GPU while substantially outperforming existing efficient video understanding methods.

## Background & Motivation
Long video understanding represents a core challenge for MLLMs. The primary difficulty lies in the fact that dense visual tokens from long videos readily exceed the sequence length limit of MLLMs (e.g., 1024 frames can produce 200K+ tokens), leading to performance degradation and prohibitive memory overhead.

Limitations of existing approaches: (1) **RAG-based methods** (e.g., AKS): retrieve keyframes for local processing, excelling at needle-in-haystack tasks but incapable of handling tasks requiring global understanding, and still subject to memory constraints; (2) **Visual compression methods** (e.g., AdaRETAKE): compress KV caches to accommodate more input frames, but still require all compressed features to be fed simultaneously during decoding, leaving the computational bottleneck unresolved, with input length growing linearly with video duration.

**Core Idea**: Emulate human video-watching behavior — continuous viewing, memory formation, and answering questions based on relevant memory segments. By iteratively processing video segments to form a fixed-size memory bank, the method breaks the input length ceiling and theoretically supports videos of unlimited length.

## Method

### Overall Architecture
FlexMem divides a long video into $N$ segments $\{V_1, ..., V_N\}$ and processes them iteratively: (1) encode the current segment together with historical context memory; (2) produce context memory $C_i$ (for information propagation) and local memory $M_i$ (stored in the memory bank) via dual-pathway compression; (3) after all segments are processed, retrieve the most query-relevant memory segments from the memory bank to generate the answer.

### Key Designs

1. **Dual-Pathway Compression**: Two distinct KV cache compression strategies are designed to address the different demands of the prefill and decoding stages:

    - **Context memory $C_i$** (for information propagation during prefill): based on an **aggregation score** $s_j^l = \sum_{k \in C} a_{jk}^l + \sum_{h \in V_i} a_{hj}^l$, tokens that both aggregate information from historical context and propagate it forward are selected and retained at compression ratio $\alpha_c$.
    - **Local memory $M_i$** (stored in the memory bank for decoding): based on a **local saliency score** $\hat{s}_j^l = \sum_{k \in V_i} a_{kj}^l$, the most discriminative tokens within each segment are selected and retained at compression ratio $\alpha_s$.

   This design ensures that context memory exhibits **continuity** (information is propagated across iterations), while local memory exhibits **distinctiveness** (preserving the most salient visual evidence from each segment).

2. **Encoding-based Memory Retrieval**: Cross-modal attention weights from the MLLM during memory encoding (optionally conditioned on query $T_q$) serve as the relevance measure: $g_i = \sum_{l=3}^{L} \sum_{j \in T_q} \sum_{k \in V_i} a_{jk}^l$. The top-$n_a$ memory segments by score are retrieved for answer generation.

3. **MemIndex Fast Memory Indexing**: To address the repeated inference cost of encoding-based retrieval, a linear regression model is fitted to approximate encoding-based retrieval results: $\arg\min_\sigma \sum_i \|\sigma(R_i) - g_i\|_2$. In practice, only $K=3$ most important cache layers and $k=5$ most salient tokens are selected to construct a compact index tensor, with only the last token's feature used on the query side. This achieves 95%+ approximation of encoding-based retrieval while supporting offline indexing and multi-query scenarios.

### Loss & Training
- **Fully training-free**: FlexMem requires no additional training and integrates directly into existing MLLMs.
- The layer weights $\alpha^l$ of MemIndex are obtained via linear regression fitted on a small dataset.
- Applied to two base models: LLaVA-Video and LLaVA-OneVision.

## Key Experimental Results

### Main Results (LLaVA-Video 7B, single 3090 GPU)

| Method | Sampled Frames | Input Tokens | TimeScope | LVBench | Video-MME (All) | LongVideoBench (All) |
|--------|---------------|-------------|-----------|---------|-----------------|----------------------|
| LLaVA-Video Baseline | 32frm | 7k | 58.3 | 41.4 | 61.7 | 58.6 |
| AKS (RAG) | 1fps | 7k | 84.6 | 46.6 | 62.8 | 59.7 |
| AdaRETAKE (Compression) | 384frm | 40k | 78.2 | 46.8 | 63.6 | 59.8 |
| **FlexMem** | 512/1024frm | 13k | **85.6** | **50.2** | **64.6** | **63.0** |

### Comparison with SOTA MLLMs

| Method | TimeScope | LVBench | MLVU | Video-MME (All) | LongVideoBench |
|--------|-----------|---------|------|-----------------|----------------|
| GPT-4o | - | 27.0 | 64.6 | 71.9 | 66.7 |
| Gemini-1.5-Pro | - | 33.1 | - | 75.0 | 64.0 |
| LLaVA-Video+FlexMem | 85.9 | **51.0** | 72.4 | 64.7 | 63.6 |

### Ablation Study

| Design Choice | LongVideoBench (All) | LVBench |
|--------------|----------------------|---------|
| Context compression only | 62.5 | 49.9 |
| Local compression only | 62.6 | 49.7 |
| **Dual-pathway compression** | **63.6** | **51.0** |
| Full memory bank (no retrieval) | 59.8 | 49.3 |
| **Memory retrieval** | **63.6** | **51.0** |

### Key Findings
- FlexMem improves LLaVA-Video by 32.2% on TimeScope and 19.7% on LVBench.
- On a single 3090 GPU, it outperforms AKS and AdaRETAKE by 3.9% and 5.2%, respectively (average over five benchmarks).
- Elevates LLaVA-Video to near Gemini-1.5-Pro level, surpassing it by 54.1% on LVBench.
- Dual-pathway compression outperforms single-pathway by 1–1.3%, confirming the complementarity of contextual continuity and local saliency.
- Feeding the entire memory bank without retrieval leads to a substantial 3.8% drop, demonstrating the importance of retrieval-based noise filtering.
- 8 frames per segment is optimal; longer segments (16/32 frames) degrade performance due to information redundancy.
- MemIndex performs excellently in streaming QA scenarios, with a gap of less than 1% compared to encoding-based retrieval.

## Highlights & Insights
- **Elegant modeling of human video-watching behavior**: The paradigm of iterative processing, memory formation, and selective recall is both natural and effective.
- **Insightful dual-pathway compression design**: Context memory ensures continuity while local memory ensures distinctiveness, each serving different stage requirements.
- **Engineering value of MemIndex**: Through simple linear regression fitting with selective layer/token choices, memory retrieval overhead is reduced to a negligible level.
- **Extreme resource efficiency**: Processing 1000+ frames on a single 3090 GPU while outperforming methods that require substantially more resources.
- The training-free design makes FlexMem a plug-and-play enhancement for any video-MLLM.

## Limitations & Future Work
- Iterative processing of each segment still requires step-by-step inference; total processing time for extremely long videos may be non-trivial despite constant per-step memory usage.
- Context memory retains only the most recent $n_s$ segments, potentially missing long-range temporal dependencies.
- MemIndex linear regression fitting requires a small amount of annotated data, introducing distribution dependence despite the modest data volume.
- Validation is limited to LLaVA-Video and LLaVA-OneVision; generalizability to other architectures (e.g., Qwen-VL) remains to be verified.

## Related Work & Insights
- **vs. AKS (RAG-based)**: RAG excels at precise localization but lacks global understanding; FlexMem addresses both through iterative memory.
- **vs. AdaRETAKE (compression-based)**: AdaRETAKE feeds all compressed features simultaneously, retaining computational bottlenecks; FlexMem decouples encoding and decoding via memory bank and retrieval.
- **vs. Video-XL (special token-based)**: Video-XL introduces special tokens to summarize information but still suffers from linearly growing input length; FlexMem maintains a fixed-size memory bank.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-pathway compression and MemIndex designs are innovative; the visual memory paradigm is broadly applicable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five long-video benchmarks plus one streaming benchmark, two base models, detailed ablations, and resource-constrained comparisons.
- Writing Quality: ⭐⭐⭐ The method section contains numerous formulas but maintains clear logical flow; some language could be refined.
- Value: ⭐⭐⭐⭐⭐ The practical value of processing 1000+ frames on a single 3090 GPU is extremely high; the training-free, plug-and-play nature suits broad deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)
- [\[CVPR 2026\] Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)

</div>

<!-- RELATED:END -->
