---
title: >-
  [Paper Note] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models
description: >-
  [CVPR 2025][Model Compression][Video Large Language Models] DyCoke is proposed, a training-free dynamic visual token compression method. By employing a two-stage strategy—temporal token merging (eliminating $50-60\%$ of cross-frame redundancy) and dynamic KV Cache pruning (dynamically retaining the most relevant tokens at each decoding step to further reduce tokens by $70-90\%$)—it reduces the average number of tokens per frame in video LLMs to 15…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Video Large Language Models"
  - "Token Compression"
  - "Temporal Merging"
  - "KV Cache Pruning"
  - "Training-Free"
date: 2026-05-08
content_hash: 014f3f7b6fa167ae
---

# DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2411.15024](https://arxiv.org/abs/2411.15024)  
**Code**: [https://github.com/KD-TAO/DyCoke](https://github.com/KD-TAO/DyCoke)  
**Area**: Model Compression  
**Keywords**: Video Large Language Models, Token Compression, Temporal Merging, KV Cache Pruning, Training-Free

## TL;DR
DyCoke is proposed, a training-free dynamic visual token compression method. By employing a two-stage strategy—temporal token merging (eliminating $50-60\%$ of cross-frame redundancy) and dynamic KV Cache pruning (dynamically retaining the most relevant tokens at each decoding step to further reduce tokens by $70-90\%$)—it reduces the average number of tokens per frame in video LLMs to 15, achieving a $1.5\times$ speedup with comparable or slightly improved performance.

## Background & Motivation

**Background**: Video Large Language Models (such as LLaVA-OneVision) require processing multi-frame inputs. Each frame generates hundreds of visual tokens, leading to thousands of total tokens and extremely high inference costs.

**Limitations of Prior Work**: Existing token pruning methods (such as FastV) employ a one-time static pruning strategy. However, the attention of video LLMs dynamically shifts across different visual tokens at different decoding steps (when generating different words)—some tokens may be unimportant at the current step but critical in subsequent steps. One-time pruning permanently discards these "delayed-important" tokens.

**Key Challenge**: Video tokens contain significant temporal redundancy (similar content in adjacent frames) and spatial redundancy (most areas of each frame are unrelated to the question). However, which tokens are relevant is dynamic and depends heavily on the currently generated text content.

**Goal**: To simultaneously eliminate both temporal and spatial redundancy in video tokens without retraining the model, while maintaining dynamic token selection capability.

**Key Insight**: A two-stage approach: first, merge temporal redundancy based on inter-frame cosine similarity, and then dynamically select the most relevant tokens based on cross-attention at each decoding step.

**Core Idea**: First, merge similar tokens between adjacent frames to eliminate temporal redundancy. Then, dynamically select the top-$p\%$ high-attention tokens at each decoding step. Pruned tokens are sent to a buffer and can be reactivated in subsequent steps.

## Method

### Overall Architecture
The visual tokens of input video frames undergo two-stage compression: (1) Temporal Token Merging (TTM) uses a sliding window to group frames by four. After even-odd splitting, it computes the cosine similarity of tokens between adjacent frame groups and merges highly similar tokens, reducing tokens by $50-60\%$. (2) KV Cache Dynamic Pruning calculates the cross-attention weights of visual tokens at each text decoding step and retains only the top-$p\%$ for attention calculation. The KV values of pruned tokens are stored in a buffer for retrieval in subsequent steps.

### Key Designs

1. **Temporal Token Merging (TTM)**:

    - **Function**: Eliminate temporal redundancy across multi-frame videos
    - **Mechanism**: Use a sliding window (size of 4 frames) to sample frame groups. After splitting into even and odd groups, compute the cosine similarity of tokens at corresponding positions between adjacent frame groups. Token pairs with similarity higher than a threshold $K$ are merged (by averaging). This reduces the token count by $50-60\%$.
    - **Design Motivation**: Content in adjacent frames is highly similar (e.g., unchanged background over several frames). These redundant tokens provide no new information but consume significant computation.

2. **KV Cache Dynamic Pruning**:

    - **Function**: Dynamically select the most relevant visual tokens at each decoding step
    - **Mechanism**: At each decoding step, compute the cross-attention scores between the current predicted token and all visual tokens, and retain only the top-$p\%$ tokens for subsequent attention calculations. The key innovation is the introduction of a Dynamic Pruning (DP) Cache—specifically, the KV state of pruned tokens is not deleted but stored in a buffer, allowing them to be retrieved and reactivated in subsequent decoding steps.
    - **Design Motivation**: Different decoding steps generate different text, requiring attention to different visual regions. DP Cache ensures no information is permanently lost.

### Loss & Training
Completely training-free, directly applicable to pre-trained video LLMs (such as LLaVA-OneVision). Hyperparameters: TTM merging threshold $K=0.5$, pruning layer $L=3$, retention ratio $P=0.7$.

## Key Experimental Results

### Main Results (LLaVA-OV-7B)

| Method | Token Retention | FLOPs | Speedup | ActivityNet-QA | NextQA |
|------|-----------|-------|-------|---------------|--------|
| Baseline | 100% | 41.4T | 1.0× | 51.93 | 2.86 |
| FastV | 35% | 17.9T | - | 50.93 | 2.80 |
| DyCoke | 18.75% | 24.1T | 1.54× | **52.08** | **2.88** |

DyCoke retains only $18.75\%$ of the tokens but achieves a slight performance increase ($+0.15/+0.02$), while reducing FLOPs by $41\%$.

### Ablation Study

| K | L | P | Retention | FLOPs Ratio | ActivityNet |
|---|---|---|-------|---------|-------------|
| 0.3 | 3 | 0.7 | 23.25% | 75% | 51.80 |
| 0.5 | 3 | 0.7 | 18.75% | 59% | 52.08 |
| 0.7 | 3 | 0.7 | 14.25% | 43% | 51.80 |

### Key Findings
- Performance remains stable even under extreme compression (retaining only $14.25\%$), indicating massive redundancy in video tokens.
- DyCoke's performance slightly improves instead of dropping, possibly because removing distracting tokens allows the model to focus better on key information.
- Dynamic selection significantly outperforms static selection (vs. FastV), validating the dynamic nature of attention in video LLMs.

## Highlights & Insights
- **Core insight of "Dynamic > Static"**: Video LLMs attend to different visual content at different decoding steps, which dictates that a dynamic, rather than static, token selection strategy must be employed.
- **DP Cache Design**: Pruned tokens are buffered instead of discarded, ensuring no permanent loss of information. This is critical to the success of dynamic pruning.
- **Training-Free Practicality**: Plug-and-play without retraining the video LLMs, resulting in zero deployment cost.

## Limitations & Future Work
- The optimal combination of two-stage hyperparameters ($K$, $L$, and $P$) may vary depending on the video content and question types.
- Validation is limited to the LLaVA-OV series; other video LLM architectures (e.g., Video-ChatGPT) have not been tested.
- DP Cache introduces extra storage overhead, which may become a bottleneck for extremely long videos.

## Related Work & Insights
- **vs. FastV**: FastV statically prunes and permanently discards tokens, whereas DyCoke uses dynamic selection with cache recovery, achieving better performance at lower retention rates.
- **vs. ToMe (Token Merging)**: ToMe is used for spatial merging of image tokens, while DyCoke incorporates the temporal dimension, making it more suitable for video scenarios.
- **vs. Video Sampling Strategies**: Directly reducing the frame count loses temporal details; DyCoke preserves all frames while compressing the token count per frame.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of temporal merging, dynamic pruning, and DP Cache is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-metric evaluation and complete ablations are provided, but tests are restricted to a single model family.
- Writing Quality: ⭐⭐⭐⭐ The visualization analysis of dynamic attention is compelling.
- Value: ⭐⭐⭐⭐ The training-free approach provides direct value for video LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] From Language Models over Tokens to Language Models over Characters](../../ICML2025/model_compression/from_language_models_over_tokens_to_language_models_over_characters.md)
- [\[ICML 2025\] DLP: Dynamic Layerwise Pruning in Large Language Models](../../ICML2025/model_compression/dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [\[CVPR 2026\] Ultra-Fast Neural Video Compression](../../CVPR2026/model_compression/ultra-fast_neural_video_compression.md)
- [\[CVPR 2025\] Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion](good_cheap_and_fast_overfitted_image_compression_with_wasserstein_distortion.md)
- [\[CVPR 2025\] Towards Practical Real-Time Neural Video Compression](towards_practical_real-time_neural_video_compression.md)

</div>

<!-- RELATED:END -->
