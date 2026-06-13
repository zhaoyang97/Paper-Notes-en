---
title: >-
  [Paper Note] Unleashing Hour-Scale Video Training for Long Video-Language Understanding
description: >-
  [NeurIPS 2025][Video Understanding][Long Video Understanding] This work constructs VideoMarathon, the first large-scale hour-level video instruction-following dataset (9,700 hours, 3.3M QA pairs, 22 task types)…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Long Video Understanding"
  - "Video-LMM"
  - "Memory Augmentation"
  - "Instruction-Following Dataset"
  - "Hour-Scale Video"
date: 2026-05-08
content_hash: 56599d5299034b6c
---

# Unleashing Hour-Scale Video Training for Long Video-Language Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2506.05332](https://arxiv.org/abs/2506.05332)  
**Code**: [Project Page](https://videomarathon.github.io/)  
**Area**: Video Understanding / Multimodal VLM
**Keywords**: Long Video Understanding, Video-LMM, Memory Augmentation, Instruction-Following Dataset, Hour-Scale Video

## TL;DR

This work constructs VideoMarathon, the first large-scale hour-level video instruction-following dataset (9,700 hours, 3.3M QA pairs, 22 task types), and proposes Hour-LLaVA, a model that leverages a memory repository, forgetting mechanism, and MemAug module to enable efficient training and inference on hour-scale videos at 1 FPS, achieving state-of-the-art results among open-source models of comparable scale across four long video benchmarks.

## Background & Motivation

**Background**: Recent Video-LMMs have achieved notable progress on video QA and video summarization tasks, yet training data predominantly consists of short clips (average under 1 minute). Existing datasets such as LLaVA-Video-178K have an average video duration of only 0.6 minutes.

**Limitations of Prior Work**: Evaluation benchmarks (e.g., LVBench with an average of 67 minutes, Video-MME with an average of 17 minutes) require models to comprehend hour-scale videos, whereas models are trained exclusively on short clips of a few minutes, resulting in a severe train–test length mismatch. Existing approaches (e.g., uniform sampling of 64 frames) incur substantial information loss when processing long videos.

**Key Challenge**: The absence of high-quality long-video instruction-following data, combined with existing models' inability to efficiently handle the massive token count of hour-scale videos. GPU memory constraints prevent models from directly consuming all visual tokens produced by 1-FPS sampling over thousands of frames.

**Goal**: (1) Construct large-scale long-video training data; (2) Design a model architecture that exploits complete video context under limited computational budget.

**Key Insight**: Drawing inspiration from the human memory system—selectively retaining and recalling critical information while systematically discarding redundancy—the work designs a memory augmentation mechanism that balances token compression with information preservation.

**Core Idea**: A hierarchical video captioning pipeline generates large-scale long-video QA data; a memory repository caches full video features, which are then compressed via a forgetting mechanism and enriched through cross-attention, enabling learnable token compression.

## Method

### Overall Architecture

The system comprises two major components: (1) **VideoMarathon Dataset**—generated via hierarchical video captioning (clip → event → global) combined with DeepSeek-V3 to produce 3.3M QA pairs; (2) **Hour-LLaVA Model**—a video encoder (SigLIP) extracts features at 1 FPS → projector (2-layer MLP) → forgetting mechanism (spatial + temporal compression to 1/16) → MemAug module (4-layer Transformer with cross- and self-attention recall) → LLM decoder (Qwen2-7B) generates answers. The full video features are stored in a memory repository; the compressed, decayed tokens are enhanced by MemAug before being fed to the LLM.

### Key Designs

1. **VideoMarathon Dataset Construction**:

    - 28K long videos (3–60 minutes, totaling ~9,700 hours) are collected from five sources: Panda-70M, Ego4D, ActivityNet, YouCook2, and MovieChat-1K.
    - Hierarchical captioning pipeline: Qwen2VL-7B generates detailed per-clip descriptions across six dimensions (temporality, spatiality, objects, actions, scenes, and summary); DeepSeek-V3 then aggregates these into event-level and global-level descriptions.
    - QA pairs covering 22 task types (6 major themes) are generated from the hierarchical descriptions, comprising 1.73M open-ended QA pairs and 1.57M multiple-choice questions.
    - **Design Motivation**: Only by training on large-scale long videos can models explicitly learn long-range dependencies. Experiments confirm that LLaVA-Video cannot benefit from VideoMarathon training due to sparse sampling, which prevents learning long-term patterns.

2. **Forgetting Mechanism + MemAug Module**:

    - **Memory Repository**: Full video tokens sampled at 1 FPS are stored in the memory repository, granting the model permanent access to the complete video context.
    - **Forgetting Mechanism**: Spatially, 3/4 of tokens are randomly discarded (compression ratio 1/4); temporally, 3/4 of frames are uniformly discarded; the overall compression ratio is approximately 1/16, yielding *decayed video tokens* $\tilde{\mathbf{H}}_v$.
    - **MemAug Module**: A 4-layer Transformer block. In cross-attention, the decayed tokens $\tilde{\mathbf{H}}_v$ and question tokens $\mathbf{H}_q$ serve as queries while the memory repository $\mathbf{H}_v$ provides keys and values, retrieving discarded information from the complete context on demand. In self-attention, question information flows into video tokens, endowing them with question-awareness. Formally: $\hat{\mathbf{H}}_v = f_{\theta_M}(\tilde{\mathbf{H}}_v, \mathbf{H}_q | \mathbf{H}_v)$
    - **Design Motivation**: Unlike handcrafted keyframe selection or question-guided compression, MemAug is a learnable compression mechanism supervised end-to-end by the next-token prediction loss.

3. **Three-Stage Training**:

    - Stage 1: Image–language pre-training (3B image–text pairs; only MemAug is trained).
    - Stage 2: Video–language adaptation (0.6M mixed data; full-parameter training for 1 epoch).
    - Stage 3: Video instruction fine-tuning (4.4M mixed data, including 0.7M VideoMarathon long-video samples; visual encoder frozen; optimal long-to-short video ratio of 3:1).

### Loss & Training

Standard cross-entropy autoregressive loss. Learning rate 2e-5 with cosine annealing and AdamW optimizer. Hour-LLaVA-7B is trained on 64 AMD MI300X GPUs.

## Key Experimental Results

### Main Results

| Method | Params | TempCompass (11s) | LongVideoBench (459s) | Video-MME Long (2466s) | LVBench (4037s) |
|--------|--------|------------------|-----------------------|----------------------|-----------------|
| GPT-4o | — | 70.9 | 66.7 | 65.3 | 48.9 |
| LLaVA-Video-7B | 7B | 60.0 | 58.2 | 56.3 | 33.7 |
| Apollo-7B | 7B | 60.0 | 56.0 | 60.0 | 37.1 |
| Video-XL | 7B | 59.3 | 55.4 | 55.5 | 38.8 |
| **Hour-LLaVA-7B** | **7B** | **63.2** | **60.1** | **62.2** | **40.5** |

### Ablation Study

| Configuration | TempCompass | LongVideoBench | LVBench | Notes |
|---------------|-------------|----------------|---------|-------|
| Hour-LLaVA (MemAug) | 59.7 | 54.0 | 40.6 | Full 3B model |
| Uniform compression (no MemAug) | 59.3 | 52.1 | 38.3 | Learnable vs. uniform |
| Keyframe compression | 59.1 | 52.0 | 38.9 | Learnable vs. keyframe |
| Question-guided compression | 56.0 | 51.0 | 37.5 | Worst — premature filtering loses context |
| Memory repository 100% → <10% | — | — | ~35 | Reducing repository size causes ~5-point drop |

### Key Findings

- MemAug consistently outperforms all handcrafted compression baselines across benchmarks, exceeding the best handcrafted method by approximately 2 points on LVBench.
- LLaVA-Video cannot benefit from VideoMarathon training even when fine-tuned on it — sparse sampling is the fundamental bottleneck.
- After spatial forgetting retains 1/4 of tokens, MemAug achieves performance comparable to the no-compression baseline on image tasks (MMStar: 51.9 vs. 52.8), demonstrating its generalizability to token compression for Image-LMMs.
- Hour-LLaVA-3B already surpasses more than half of 7B-scale models and exhibits out-of-distribution generalization on LVBench, where average video length exceeds the training maximum.

## Highlights & Insights

- **Filling the Data Gap**: VideoMarathon is the first large-scale hour-level video instruction-following dataset. The three-level clip → event → global captioning strategy elegantly addresses the challenge of generating high-quality QA pairs for long videos using large models, and is transferable to any long-video annotation scenario.
- **Learnable vs. Handcrafted Compression**: A unified experimental framework compares uniform, keyframe-based, and question-guided compression strategies. Notably, question-guided compression performs worst — premature information filtering discards critical context — underscoring the advantage of end-to-end learning.
- **MemAug's Potential for Image-LMMs**: The spatial compression ablation shows that 1/4 tokens + MemAug can match full-token performance, suggesting that memory augmentation could directly accelerate Image-LMM inference.

## Limitations & Future Work

- Training data is entirely model-synthesized (Qwen2VL + DeepSeek-V3), introducing noise and hallucinations; no noise-robust training strategy is designed.
- Only video and text modalities are handled; audio information is ignored.
- Evaluation relies predominantly on multiple-choice QA, which may not comprehensively measure long-video understanding capabilities.
- The longest training videos are 60 minutes; performance on ultra-long videos (e.g., full-length films of 2–3 hours) remains to be validated.
- The forgetting mechanism is relatively naive; content-importance-aware adaptive forgetting warrants further exploration.

## Related Work & Insights

- **vs. LLaVA-Video**: Uniform sampling of 64 frames prevents learning from long-video data; the memory repository + MemAug architecture of Hour-LLaVA is the critical differentiator.
- **vs. LongVU / Video-XL**: These methods rely on handcrafted heuristic compression (keyframe selection, KV-cache retrieval); Hour-LLaVA with learnable MemAug consistently outperforms them under the same token budget.
- **vs. LongVILA**: LongVILA extends context to 2M tokens but demands substantial computational resources and sequence parallelism; Hour-LLaVA achieves greater efficiency through compression and recovery.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dataset construction pipeline is comprehensive and practical; the MemAug design is well-motivated though of moderate novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four benchmarks with detailed ablations covering forgetting strategies, data mixing, memory repository size, and compression method comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, detailed method descriptions, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ VideoMarathon fills a substantial gap in long-video training data, and Hour-LLaVA demonstrates the necessity of hour-scale video training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](../../ICCV2025/video_understanding/vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](../../AAAI2026/video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)

</div>

<!-- RELATED:END -->
