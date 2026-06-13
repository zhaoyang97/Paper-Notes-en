---
title: >-
  [Paper Note] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens
description: >-
  [ICCV 2025][Model Compression][Vision Large Language Model] This paper proposes B-VLLM, a framework that dynamically balances spatio-temporal cues within the context window constraints of VLLMs via three modules: text-co…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Vision Large Language Model"
  - "Video Understanding"
  - "Token Balancing"
  - "Frame Selection"
  - "Token Merging"
date: 2026-05-08
content_hash: 5c54507dcfb1230b
---

# B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens

**Conference**: ICCV 2025  
**arXiv**: [2412.09919](https://arxiv.org/abs/2412.09919)  
**Code**: [https://github.com/zhuqiangLu/B-VLLM.git](https://github.com/zhuqiangLu/B-VLLM.git)  
**Area**: Model Compression / Video Understanding  
**Keywords**: Vision Large Language Model, Video Understanding, Token Balancing, Frame Selection, Token Merging

## TL;DR

This paper proposes B-VLLM, a framework that dynamically balances spatio-temporal cues within the context window constraints of VLLMs via three modules: text-conditioned adaptive frame selection, temporal frame token merging, and spatial token sampling. The approach achieves a 10% performance improvement on MVBench.

## Background & Motivation

Current VLLMs face a visual token overload problem when processing video—especially long video—where increasing frame counts cause token counts to grow rapidly, potentially exceeding the LLM context window limit and substantially increasing computational cost. Existing solutions suffer from spatio-temporal cue imbalance:

- **Uniform frame downsampling** (e.g., VideoLLaMA2 samples a fixed 8 frames): ignores temporal dynamics and may miss task-relevant keyframes.
- **Per-frame token compression** (e.g., LLaMA-VID compresses each frame to 2 tokens): fails to preserve intra-frame spatial detail, leading to poor performance on tasks requiring spatial understanding.

The authors characterize this as a "spatio-temporal token imbalance" problem: reducing frame-level tokens causes temporal cues to dominate, while uniform frame sampling causes spatial cues to be overwhelmed. A method capable of adaptively balancing spatio-temporal token allocation according to the task is therefore needed.

## Method

### Overall Architecture

The B-VLLM pipeline proceeds as follows: (1) a visual encoder encodes all frames into initial visual tokens (including [CLS] tokens); (2) a text-conditioned frame selection module uses [CLS] tokens and the question text to select the $L^*$ most relevant frames; (3) temporal frame token merging removes redundant frames; (4) spatial token sampling extracts the $R$ most relevant tokens from the selected frames; (5) an optional spatial token merging step further enforces a token budget $\theta$; (6) visual tokens are projected into the LLM feature space and concatenated with text tokens as input to the LLM.

### Key Designs

1. **Text-Conditioned Adaptive Frame Selection**:

    - Per-frame [CLS] tokens (carrying high-level semantic information) are used instead of all tokens to locate relevant frames, balancing computational efficiency.
    - A Q-Former serves as the frame selection network, jointly encoding the [CLS] token sequence $V_{[CLS]}$ and textual context $T$ to generate $L^*$ queries.
    - Differentiable discrete frame selection is achieved via Gumbel-Softmax: $V^* = \Phi(Q \cdot V_{[CLS]}^\top, \tau) \cdot V$
    - The temperature parameter $\tau$ controls discreteness; as $\tau \to 0$, the Gumbel-Softmax approximates a one-hot distribution (in practice $\tau = 0.1$).
    - Gumbel-Softmax is preferred over standard Softmax because Softmax tends to smooth out rich spatial visual cues, particularly when aggregating across multiple frames.
    - Temporal ordering of selected frames is restored after selection.

2. **Temporal Frame Token Merging**:

    - Frame selection may produce duplicates (e.g., when the total frame count $L < L^*$).
    - Duplicate frames are detected via cosine similarity between row vectors of the selection matrix $S_\tau$.
    - Frames whose similarity exceeds a threshold $\gamma$ are treated as duplicates and merged by averaging: $V_\alpha^* = \frac{1}{|D_\alpha|} \sum_{\beta \in D_\alpha} V_\beta^*$
    - The deduplication-and-merging process is applied iteratively.

3. **Spatial Visual Token Sampling**:

    - A spatial Q-Former samples $R$ most relevant tokens from the $M$ tokens of each selected frame ($R \ll M$), with frame tokens and textual context as input.
    - An optional progressive spatial token merging strategy is employed: when the token count exceeds budget $\theta$, Bipartite Merging is applied repeatedly to merge the most similar tokens until the budget is satisfied.
    - This constitutes the first proposal of an iterative token merging strategy for fine-grained token count control.

4. **Integration with the Backbone LLM**:

    - Visual tokens are projected into the LLM feature space via a trainable MLP.
    - The framework is flexible: it can be used standalone (with Qwen2 as the backbone) or integrated into existing VLLMs (e.g., LLaMA-VID, VideoLLaMA2).

### Loss & Training

- Training is conducted exclusively on the LLaMA-VID-Dataset and Valley datasets to ensure fair comparison.
- Standard autoregressive language modeling loss is employed.

## Key Experimental Results

### Main Results — Video Benchmarks

| Method | #Frames | MVBench | VideoMME-s | VideoMME-m | VideoMME-l |
|--------|---------|---------|------------|------------|------------|
| LLaMA-VID | 1fps | 39.0 | 34.2 | 34.7 | 27.1 |
| LLaMA-VID + Ours | 1fps | 43.5(+4.5) | 44.7(+10.5) | 38.8(+4.1) | 35.2(+8.1) |
| VideoLLaMA2 | 8 | 45.5 | 48.9 | 42.7 | 37.7 |
| VideoLLaMA2 + Ours | 1fps | 46.5(+1.0) | 47.2 | 44.4(+1.7) | 41.5(+3.8) |
| **B-VLLM** | 1fps | **50.8** | **60.8** | **51.8** | **47.9** |

Gains are most pronounced in long-video settings: LLaMA-VID achieves an 8.1% improvement on VideoMME-Long.

### Ablation Study — Module Contributions

| Frame Sel. | Temporal Merge | Spatial Sample | MVBench | VideoMME | MMBench | POPE |
|------------|---------------|---------------|---------|----------|---------|------|
| ✗ | ✗ | ✗ | 39.0 | 33.6 | 49.8 | 75.3 |
| ✓ | ✗ | ✗ | 39.4 | 36.5 | 58.0 | 81.1 |
| ✓ | ✓ | ✗ | 42.1 | 38.1 | 54.6 | 67.3 |
| ✓ | ✓ | ✓ | **43.5** | **39.6** | **59.3** | **83.8** |

### Key Findings

- **Effectiveness of frame selection**: Q-Former outperforms Resampler (MVBench: 46.5 vs. 44.3), as Q-Former's cross-attention is better suited for cross-modal token processing.
- **Value of [CLS] tokens**: Although less informative than Mean Pooling, [CLS] tokens offer twice the training efficiency (10 hrs vs. 19 hrs) while consistently outperforming random guessing.
- **Token count–performance trade-off**: B-VLLM performance saturates at approximately 512 tokens, demonstrating the viability of efficient budget control.
- **Hyperparameter sensitivity**: Low $\tau$ (0.1) combined with high $\gamma$ (0.75–1.0) yields the best results; lower $\tau$ drives Gumbel-Softmax closer to discrete selection, while higher $\gamma$ prevents merging of dissimilar frames.
- 8–32 keyframes suffice for most video understanding tasks.
- Video understanding tasks rely more on temporal dynamics than spatial detail (reducing frames from 32 to 16 incurs only a 0.9% drop, whereas reducing spatial tokens from 32 to 16 incurs a 0.6% drop).

## Highlights & Insights

- The paper decomposes the token management problem in video VLLMs into three cleanly decoupled stages—frame selection, deduplication, and spatial token selection—each with a well-motivated technical solution.
- The reuse of [CLS] tokens within VLLMs is an elegant and effective design choice; most VLLMs discard [CLS] tokens entirely.
- The use of Gumbel-Softmax to realize differentiable discrete frame selection is technically elegant, simultaneously enabling end-to-end trainability and genuine discrete frame selection.
- The plug-and-play nature of the framework confers broad practical utility.

## Limitations & Future Work

- [CLS] tokens as frame-level representations carry limited information, which may lead to suboptimal frame selection for tasks requiring fine-grained spatial information to determine relevance.
- Performance on image benchmarks requiring spatial reasoning and OCR (e.g., VizWiz, TextVQA) is limited by the token budget constraint.
- The frame selection module itself (Q-Former) introduces additional computational overhead and parameters.
- Spatial token sampling uses a fixed count $R$, without adaptive adjustment based on frame content.

## Related Work & Insights

- The proposed approach is complementary to LLaMA-VID (extreme spatial compression) and VideoLLaMA2 (uniform frame sampling).
- Token Merging (ToMe), originally developed for image generation, is adapted here for token management in VLLMs.
- The text-conditioned adaptive processing paradigm is extensible to other multimodal scenarios, such as multi-image reasoning and document understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of text-conditioned frame selection and iterative token merging is novel and practically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation, plug-in validation, detailed ablations, and visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; module design is presented in a well-structured, progressive manner.
- Value: ⭐⭐⭐⭐ A general-purpose video VLLM framework with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](../../NeurIPS2025/model_compression/vision-centric_token_compression_in_large_language_model.md)
- [\[NeurIPS 2025\] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models](../../NeurIPS2025/model_compression/learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)

</div>

<!-- RELATED:END -->
