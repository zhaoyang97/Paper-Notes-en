---
title: >-
  [Paper Note] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding
description: >-
  [CVPR2026][Video Understanding][long video understanding] This paper proposes QViC-MF, a framework that achieves state-of-the-art performance on MLVU, LVBench…
tags:
  - "CVPR2026"
  - "Video Understanding"
  - "long video understanding"
  - "visual compression"
  - "memory feedback"
  - "question-guided attention"
  - "large multimodal models"
date: 2026-05-08
content_hash: 0d5f9b973a1868bd
---

# Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding

**Conference**: CVPR2026
**arXiv**: [2603.15167](https://arxiv.org/abs/2603.15167)
**Code**: [FujitsuResearch/QViC-MF](https://github.com/FujitsuResearch/QViC-MF)
**Area**: Video Understanding
**Keywords**: long video understanding, visual compression, memory feedback, question-guided attention, large multimodal models

## TL;DR

This paper proposes QViC-MF, a framework that achieves state-of-the-art performance on MLVU, LVBench, and VNBench through question-guided multi-frame visual compression (QMSA) and a contextual memory feedback mechanism, using as few as 16 visual tokens per frame.

## Background & Motivation

**Pressing need for long video understanding**: Real-world applications such as surveillance analysis and industrial process monitoring require models to comprehend videos spanning tens of minutes to several hours, yet the limited context windows of LMMs make processing full-length videos infeasible.

**Limitations of Prior Work**: Transformer-based visual compressors and memory-augmented methods typically compress each frame independently, failing to capture cross-frame temporal dependencies and underperforming on tasks that require holistic event understanding, such as temporal ordering.

**Unidirectional perception-to-memory pipeline**: Conventional frameworks compress visual information, store it in memory, and then perform inference sequentially; memory never feeds back to the perception stage, making the loss of question-relevant information irrecoverable.

**Memory capacity constraints**: Fixed-capacity memory banks with suboptimal update strategies tend to discard visually critical frames related to the query, degrading final inference quality.

**Compression hallucination**: In naive self-attention designs, textual token information leaks into context embeddings, causing compressed representations to contain semantics absent from the visual input.

**Question-insensitive attention**: When visual and textual tokens are jointly fed into a self-attention module, context tokens cannot adaptively attend to question-relevant visual regions.

## Method

### Overall Architecture (QViC-MF)

QViC-MF processes long videos in a clip-wise streaming fashion:

1. **Visual Encoder**: Concatenates the current clip ($K$ frames) with $K_r$ frames recalled from the context memory, extracts visual features via SigLIP So400m/14-384px, and projects them through an MLP projector to obtain visual embeddings $\mathcal{E}_{v,n} \in \mathbb{R}^{K_v \times P \times D_e}$.
2. **Visual Compressor**: A Transformer encoder centered on QMSA that compresses $P$ patch tokens per frame into $C$ context tokens ($C \ll P$); its input is the concatenation of visual embeddings, question embeddings, and learnable context seed embeddings.
3. **Context Memory**: Stores the context embeddings, relevance scores, and global frame indices for each frame, with capacity $L$; entries with the lowest relevance scores are pruned when the buffer overflows.
4. **Memory Feedback**: Retrieves the $K_r$ frames with the highest relevance scores from the current memory as recall frames for the next clip, forming a memory-to-perception feedback loop.
5. **Decoder**: A Qwen2-7B LLM that generates answers by concatenating all context embeddings stored in memory with the text prompt.

### Key Designs: QMSA (Question-guided Multimodal Selective Attention)

QMSA applies three operator matrices $\mathbf{M}, \mathbf{B}, \mathbf{G}$ to the self-attention logits:

$$\text{QMSA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{D_e}} + \mathbf{M} + \mathbf{B} + \mathbf{G}\right)\mathbf{V}$$

- **Masking ($\mathbf{M}$)**: Extends the causal mask across multiple frames to allow cross-frame temporal context, while restricting each frame's context tokens to attend only to visual/context tokens within the same frame, enabling frame-level compression.
- **Blocking ($\mathbf{B}$)**: Blocks the attention pathway from context tokens to text tokens, preventing textual semantics from leaking into the visual compressed representations (addressing compression hallucination).
- **Guiding ($\mathbf{G}$)**: Broadcasts the mean of text-to-visual attention logits as a guidance bias for context-to-visual attention, enabling the compression process to adaptively focus on question-relevant visual regions.

### Relevance Scores and Memory Update

The relevance score $r_{n,i}$ is computed as the mean over the Top-$K_h$ heads of text-to-visual attention weights from intermediate compressor layers ($L_1$ to $L_2$), and is used for memory pruning and recall ranking.

### Loss & Training

- Only the context seed embeddings and the visual compressor are trained; the base model is frozen.
- The compressor is fine-tuned with LoRA on LLaVA-Video-7B-Qwen2.
- Training data: 83K samples randomly drawn from LLaVA-Video-178K (domain-balanced, approximately 5%).
- Training hardware: 8×H200 GPUs.

## Key Experimental Results

### Main Results

| Method | LLM | Token/frame | MLVU test | LVBench | VideoMME Long | VNBench Long |
|--------|-----|-------------|-----------|---------|---------------|--------------|
| LLaVA-Video | Qwen2-7B | 169 | 53.3 | 41.8 | — | 40.4 |
| Flash-VStream | Qwen2-7B | 128 | — | 42.0 | 50.3 | — |
| Video-XL | Qwen2-7B | 16 | 45.5 | — | — | — |
| **QViC-MF (2fps)** | **Qwen2-7B** | **16** | **59.4** | **50.2** | **54.0** | **58.7** |

Using only 16 tokens per frame, QViC-MF surpasses the previous state of the art by 6.1% on MLVU test, 8.2% on LVBench, 18.3% on VNBench Long, and 3.7% on VideoMME Long.

### Ablation Study

| Configuration | MLVU test | VNBench Long |
|---------------|-----------|--------------|
| Vanilla Compressor (64 frames) | 48.8 | 39.3 |
| + Context Memory | 42.4 | 40.9 |
| + Memory Feedback | 52.2 | 55.1 |
| + Framewise Mask ($\mathbf{M}$) | 53.0 | 50.7 |
| + Blocking Ctx2Txt ($\mathbf{B}$) | 57.9 | 56.7 |
| + Guiding Ctx2Vis ($\mathbf{G}$) | **59.4** | **58.7** |

### Key Findings

- **Memory feedback is the primary source of gain**: Adding memory alone (without feedback) actually degrades MLVU performance (48.8→42.4), whereas introducing feedback yields a substantial improvement to 52.2.
- **QMSA components are incrementally effective**: Blocking contributes most on MLVU (+4.9), while Guiding contributes substantially on VNBench Long (+2.0).
- **Extreme compression preserves high accuracy**: Even when compressed to a single token per frame, QMSA retains over 80% of the original accuracy on MLVU, far outperforming single-frame compression and average pooling baselines.
- **MLVU sub-task analysis**: Gains are particularly pronounced on temporal reasoning tasks such as Ego Reasoning (71.7), Action Order (61.4), and Sports QA (58.3).

## Highlights & Insights

- **Feedback-driven closed-loop perception–memory architecture**: By breaking the conventional unidirectional pipeline and allowing historical context stored in memory to inform the compression of current frames, QViC-MF represents an important paradigm shift in long video understanding framework design.
- **Elegant QMSA design**: The three-matrix Mask/Block/Guide mechanism simultaneously addresses frame-level compression, compression hallucination, and question-adaptive attention in a unified and concise formulation.
- **Exceptional token efficiency**: At 16 tokens per frame—far fewer than LLaVA-Video (169) or Flash-VStream (128)—the method achieves a superior balance between efficiency and accuracy.
- **Substantial lead on VNBench (NIAH tasks)**: Outperforming LLaVA-Video by 18.3% on the Long subset demonstrates the effectiveness of the memory feedback mechanism in localizing sparse critical events.

## Limitations & Future Work

- Validation is limited to 7B-scale models; it remains unexplored whether larger LLMs (e.g., 13B/70B) would benefit further.
- The streaming architecture relies on fixed clip sizes and a fixed number of recall frames, lacking an adaptive sampling strategy.
- The sensitivity of hyperparameters such as memory capacity $L=256$ and recall frame count $K_r=32$ to different video lengths and types is not thoroughly discussed.
- Training on only 83K samples leaves the generalization to larger data scales or more diverse tasks unverified.
- Evaluation is restricted to multiple-choice question answering; more complex video understanding forms such as open-ended generation or grounding are not explored.

## Related Work & Insights

- **Memory-augmented LMMs**: MA-LMM, MovieChat, and Flash-VStream all adopt unidirectional memory strategies; QViC-MF is the first to introduce memory-to-perception feedback.
- **Visual compression**: LLaMA-VID and LLaVA-Mini compress each frame independently; Video-XL compresses long videos with summary tokens but without question guidance.
- **Question-aware encoding**: InstructBLIP's Q-Former, IQViC, and related methods explore question-guided compression for images or short videos, but all operate on single frames without cross-frame temporal modeling.
- **Frame selection**: Methods such as Frame-Voyager reduce redundancy through keyframe selection, complementing QViC-MF's compression-based approach.

## Rating

- Novelty: ⭐⭐⭐⭐ — The closed-loop design with memory feedback driving perception constitutes a new paradigm in the field; the triple-matrix manipulation in QMSA offers a novel perspective on visual compression.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across four benchmarks, layer-wise ablation validating each component, with rich compression ratio comparisons and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, figures are intuitive (especially the visual diagnosis of compression hallucination), and technical descriptions are rigorous.
- Value: ⭐⭐⭐⭐ — Achieves significant improvements across multiple long video benchmarks; the efficient 16-token-per-frame design is highly relevant to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[CVPR 2026\] Temporally Consistent Long-Term Memory for 3D Single Object Tracking](chronotrack_temporally_consistent_long_term_memory_for_3d_single_object_tracking.md)
- [\[CVPR 2026\] VSI: Visual-Subtitle Integration for Keyframe Selection to Enhance Long Video Understanding](vsi_visual-subtitle_integration_for_keyframe_selection_to_enhance_long_video_un.md)
- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[CVPR 2026\] VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](videoseek_long-horizon_video_agent_with_tool-guided_seeking.md)

</div>

<!-- RELATED:END -->
