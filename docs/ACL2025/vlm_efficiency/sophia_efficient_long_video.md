---
title: >-
  [Paper Note] Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding
description: >-
  [ACL 2025 (Long Paper)][Multimodal Efficiency][Long Video Understanding] Proposes the Sophia model to handle hour-scale long videos: accurately selects query-relevant frames via Shot-adaptive Frame Pruning (a two-stage frame pruning based on shot segmentation), and replaces full attention with Hierarchical Attention of $O(N)$ complexity. It achieves state-of-the-art (SOTA) performance on 6 out of 8 long video benchmarks, while requiring only 1/8.5 of the attention FLOPs compa…
tags:
  - "ACL 2025 (Long Paper)"
  - "Multimodal Efficiency"
  - "Long Video Understanding"
  - "Shot-adaptive Frame Pruning"
  - "Hierarchical Attention"
  - "Shot Detection"
  - "Sparse Attention"
date: 2026-05-08
content_hash: 6506f6cbc97cdfef
---

# Sharper and Faster mean Better: Towards More Efficient Vision-Language Model for Hour-scale Long Video Understanding

**Conference**: ACL 2025 (Long Paper)  
**Code**: [HuggingFace](https://huggingface.co/Tao-tse/Sophia)  
**Area**: Multimodal VLM / Video Understanding / Efficient Inference  
**Keywords**: Long Video Understanding, Shot-adaptive Frame Pruning, Hierarchical Attention, Shot Detection, Sparse Attention  

## TL;DR
Proposes the Sophia model to handle hour-scale long videos: accurately selects query-relevant frames via Shot-adaptive Frame Pruning (a two-stage frame pruning based on shot segmentation), and replaces full attention with Hierarchical Attention of $O(N)$ complexity. It achieves state-of-the-art (SOTA) performance on 6 out of 8 long video benchmarks, while requiring only 1/8.5 of the attention FLOPs compared to InternVL2.

## Background & Motivation

**Triple Challenges of Long Videos**: (1) Context length overflow—a 10-minute video sampled at 2 fps amounts to 1200 frames, generating tens of thousands of visual tokens; (2) Massive memory consumption—the KV cache of standard quadratic attention requires 70GB+ at 128 frames; (3) Prohibitive computational complexity—full attention FLOPs grow quadratically with the number of frames.

**Limitations of Prior Work**: Existing approaches either compress the token count per frame (e.g., spatial pooling in LLaVA-OneVision), sacrificing spatial details, or apply uniform temporal segmentation, which discards substantial segments and ignores the temporal non-uniformity of events/shots in videos. LongVU selects frames based on DINOv2 feature clustering, but fails to leverage query information for targeted filtering.

**Key Insight**: Videos possess a natural structure—shot transitions. Leveraging this structure for two-level pruning (filtering shots first, then removing redundant frames) aligns better with video semantics than uniform segmentation or unstructured clustering. Meanwhile, frame-to-frame attention can be replaced with a hierarchical structure (intra-frame local + inter-frame global) instead of full connections, theoretically guaranteeing an information propagation distance of $O(1)$.

## Method

### Overall Architecture
Two core modules: (1) Shot-adaptive Frame Pruning—two-stage frame pruning based on shot detection; (2) Hierarchical Attention—replacing full attention with hierarchical sparse attention. The backbone consists of an InternViT-300M encoder, an MLP projector, and an InternLM2-Chat-7B language model.

### Key Designs

1. **Shot-adaptive Frame Pruning (Two-Stage)**

    - **Shot Detection**: Employs a pre-trained TransNet to detect shot transitions, naturally segmenting the video into variable-length shot clips.
    - **Inter-shot Pruning**: Extracts the visual embeddings of the key frames from each shot, computes the cosine similarity with the MLP-mapped query text, and discards the least relevant $\alpha\%$ of the shots.
    - **Intra-shot Pruning**: Calculates the cosine similarity between adjacent frames within the same shot, removing the highly redundant $\beta\%$ of frames (e.g., duplicated frames in prolonged static scenes).
    - **Differentiable Indexing**: Utilizes Gumbel-Softmax during training to achieve a differentiable approximation of frame selection, enabling end-to-end gradient propagation.

2. **Hierarchical Attention ($O(N)$ Complexity)**

    - Groups video tokens by frame, dividing attention into two levels: (a) Intra-frame local attention—fully connected among all tokens within the same frame; (b) Inter-frame global attention—fully connected among frame-level summary tokens.
    - **IPD Theoretical Guarantee**: The Information Propagation Distance (IPD) is $O(1)$—any two frames require at most 2 attention layers to exchange information (first aggregating to the frame summary token $\rightarrow$ propagating inter-frame $\rightarrow$ distributing to target frames), significantly superior to the $O(F/w)$ of sliding window attention.
    - **Efficient Implementation**: Implemented with custom Triton CUDA kernels to avoid the extra overhead associated with PyTorch sparse attention.

### Loss & Training
- Three-stage training: (1) MLP projector alignment $\rightarrow$ (2) Joint full-parameter fine-tuning $\rightarrow$ (3) Video instruction tuning.
- The Gumbel-Softmax temperature is gradually annealed during training.
- The TransNet shot detector remains frozen and does not participate in training.

## Experiments

### Main Results: Long Video Understanding

| Benchmark | Sophia | Prev. SOTA | Gain |
|-----------|--------|---------|------|
| EgoSchema | **64.4** | 54.9 (LongVU) | +17.3% |
| MovieChat-1K | **78.2** | 74.7 (LLaVA-OneVision) | +4.7% |
| LongVideoBench | **57.9** | 55.0 (InternVL2) | +5.3% |
| LVBench | **46.2** | 44.3 (LongVU) | +4.3% |
| MLVU | **68.3** | 65.4 (LongVU) | +4.4% |
| Video-MME (Long) | **47.1** | 45.5 (InternVL2) | +3.5% |

### Efficiency Comparison (128-frame input)

| Model | Attention FLOPs | Memory Usage |
|------|----------------|---------|
| LongVU | 87.03T | ~80GB |
| InternVL2-8B | 22.33T | ~70GB |
| Qwen2-VL-7B | 19.06T | ~65GB |
| **Sophia** | **2.64T** | **~27GB** |

Sophia’s attention FLOPs are only **1/8.5** of InternVL2 and **1/33** of LongVU.

### Ablation Study

| Ablation Dimension | Conclusion |
|----------|------|
| Shot detection vs. Uniform splitting | Shot-adaptive is 3.2% higher on EgoSchema, supporting that shot awareness aligns better with video semantics. |
| Two-stage (Inter + Intra) | Removing either stage leads to performance degradation, demonstrating their complementary and non-substitutable nature. |
| Hierarchical vs. Dense Attention | Performance discrepancy is <1%, while FLOPs are reduced by over 10 times, offering an exceptional efficiency-performance trade-off. |
| Query-guided vs. Query-free pruning | Query-guided Inter-shot Pruning contributes approximately 2-3% absolute improvement. |
| Gumbel-Softmax vs. Hard selection | Differentiable selection stabilizes training and accelerates convergence. |

### Key Findings
- The 8B-parameter Sophia outperforms the 34B LLaVA-NeXT-Video and the 40B InternVL2, demonstrating that architectural efficiency can bridge the gap in parameter scale.
- $\text{IPD} = O(1)$ implies that even when processing a 1-hour video (thousands of frames), there is no information decay across long distances between frames.
- The quality of shot detection significantly impacts final performance—TransNet performs best on videos with explicit shots, such as movies.

## Highlights & Insights
- **Shot awareness is the core innovation**: Leveraging the natural structure of videos (shot boundaries) instead of artificial partitioning aligns better with video semantic distribution.
- **Theoretically guaranteed $O(N)$ attention**: $\text{IPD} = O(1)$ balances efficiency and long-range modeling capability, unlike sliding window attention which decays with distance.
- **Solid engineering implementation**: Custom Triton kernel implementation combined with practical memory/speed comparisons provides empirical data support alongside theoretical advantages.
- **Smaller models outperforming larger ones**: The 8B Sophia outperforms 34-40B models on 6 out of 8 benchmarks, highlighting the importance of architectural design.

## Limitations & Future Work
- The $\alpha$ and $\beta$ values for frame pruning are fixed hyperparameters without adaptive adjustments (different videos/queries should have different optimal pruning rates).
- The TransNet shot detector remains frozen and is not jointly trained with the VLM, posing a pipeline bottleneck where detector errors will propagate downstream.
- Hierarchical Attention assumes visual tokens far outnumber textual tokens, which might not be applicable to short video scenarios.
- Not validated in real-time or streaming video understanding scenarios.
- Shot detection may have limited efficacy on videos without distinct shot boundaries (e.g., surveillance footage, continuous screen recordings).

## Related Work & Insights
- **vs. LongVU**: Performs frame selection based on DINOv2 feature clustering without leveraging query information; Sophia's shot-aware and query-guided selection is more precise.
- **vs. Qwen2-VL**: Handles dynamic resolution but still relies on full attention, whereas Sophia's hierarchical attention is more efficient.
- **vs. InternVL2**: Achieves comparable performance but with Sophia’s FLOPs being an order of magnitude lower (1/8.5).
- **vs. Video-LLaMA series**: Compresses via video Q-Former but loses details, whereas Sophia's frame pruning retains the complete information of key frames.

## Rating
- Novelty: ⭐⭐⭐⭐ Shot-aware segmentation and IPD theoretical analysis are novel; their combination addresses practical bottlenecks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 8 benchmarks, offering detailed efficiency analysis and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Excellent integration of theory and practice, with clear and intuitive efficiency comparison diagrams.
- Value: ⭐⭐⭐⭐⭐ Addresses core efficiency bottlenecks in long video understanding, possessing both solid engineering and academic value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding](../../CVPR2026/vlm_efficiency/timeviper_a_hybrid_mamba-transformer_vision-language_model_for_efficient_long_vi.md)
- [\[ACL 2025\] MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)
- [\[ACL 2026\] APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention](../../ACL2026/vlm_efficiency/apb-v_accelerating_long-video_understanding_via_sequence-parallelism-aware_appro.md)
- [\[ACL 2026\] ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs](../../ACL2026/vlm_efficiency/regate_learning_faster_and_better_with_fewer_tokens_in_mllms.md)
- [\[ICML 2025\] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](../../ICML2025/vlm_efficiency/sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere.md)

</div>

<!-- RELATED:END -->
