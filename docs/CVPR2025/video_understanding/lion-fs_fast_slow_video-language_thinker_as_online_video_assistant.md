---
title: >-
  [Paper Note] LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant
description: >-
  [CVPR 2025][Video Understanding][Online Video Assistant] The LION-FS online video assistant framework is proposed, inspired by the "fast and slow thinking" cognitive theory. It utilizes a Fast Path (router-based token aggregation and dropping) to achieve efficient real-time response decision-making, and a Slow Path (multi-granularity keyframe augmentation) to inject fine-grained spatial and interaction features during response generation. It comprehensively outperforms existi…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Online Video Assistant"
  - "Fast and Slow Thinking"
  - "Token Routing"
  - "Keyframe Augmentation"
  - "First-person Video"
date: 2026-05-08
content_hash: 0eaff8a9d4d75e76
---

# LION-FS: Fast & Slow Video-Language Thinker as Online Video Assistant

**Conference**: CVPR 2025  
**arXiv**: [2503.03663](https://arxiv.org/abs/2503.03663)  
**Code**: [https://github.com/JiuTian-VL/LION-FS](https://github.com/JiuTian-VL/LION-FS)  
**Area**: Video Understanding / Multimodal Large Language Models  
**Keywords**: Online Video Assistant, Fast and Slow Thinking, Token Routing, Keyframe Augmentation, First-person Video

## TL;DR

The LION-FS online video assistant framework is proposed, inspired by the "fast and slow thinking" cognitive theory. It utilizes a Fast Path (router-based token aggregation and dropping) to achieve efficient real-time response decision-making, and a Slow Path (multi-granularity keyframe augmentation) to inject fine-grained spatial and interaction features during response generation. It comprehensively outperforms existing methods on the Ego4D and Ego-Exo4D benchmarks.

## Background & Motivation

**Background**: Online video assistants must continuously ingest first-person video streams, dynamically determine when to respond to the user, and deliver professional, accurate responses. The LIVE framework, proposed by VideoLLM-online, is a pioneering work in this field, establishing the foundational paradigm for streaming video dialogue.

**Limitations of Prior Work**: LIVE suffers from three severe limitations: (1) Low response decision accuracy—utilizing only low-frame-rate image features makes it difficult for LLMs to capture temporal relationships between frames; (2) Imprecise response content—retaining a fixed, small number of tokens per frame fails to leverage the unique nature of the first-person perspective, preventing the capture of adaptive, fine-grained information; (3) Poor training/inference efficiency—scaling up tokens for all frames to improve performance is highly inefficient, as the response decision phase does not require such a large volume of tokens; token expansion should instead be concentrated in the response generation stage for keyframes.

**Key Challenge**: Online video assistants must reconcile real-time operation (high frame rate, low latency) with high accuracy (fine-grained understanding, precise responses). These two objectives present an inherent conflict—more tokens yield better understanding but slower inference.

**Goal**: (1) To efficiently process high-frame-rate video streams and accurately determine when to respond; (2) To enhance response precision and granularity without sacrificing efficiency.

**Key Insight**: Drawing inspiration from Kahneman's "Fast and Slow Thinking" theory—simple response decision-making (whether to reply) corresponds to the fast, intuitive System 1, while select, complex response generation corresponds to the deliberate System 2. The two tasks are decoupled and optimized using distinct strategies.

**Core Idea**: Decouple online video dialogue into a Fast Path (router-driven efficient response decision-making) and a Slow Path (multi-granularity keyframe-augmented precise response generation), optimizing both effectiveness and efficiency.

## Method

### Overall Architecture

The overall workflow of LION-FS is split into two paths. Fast Path: A dual-encoder setup (SigLIP 2FPS + EgoVLPv2 8FPS) extracts general spatial features and first-person temporal features. These two types of features are adaptively fused (without increasing the token count) via a Token Aggregation Router, and redundant tokens are subsequently discarded using a Token Dropping Router to achieve sparse decoding, enabling efficient, frame-by-frame response decision-making. Slow Path: Once a response is triggered, the current frame is designated as a keyframe and enhanced at multiple granularities—utilizing global grid enhancement (Grid Tokens) and local object enhancement (Box Tokens). These are injected into a multimodal Thinking Template to guide more precise response generation.

### Key Designs

1. **Token Aggregation Router**:

    - **Function**: Adaptively fuses features from the general image encoder and the first-person video encoder without increasing the total token count.
    - **Mechanism**: SigLIP (2FPS) extracts 10 tokens per frame (1 CLS + 9 pooled tokens from a 3×3 grid), while EgoVLPv2 (8FPS) extracts 10 tokens for every group of 4 frames. Once temporally aligned, an MLP router uses the CLS token of SigLIP (Visual Guidance) to generate weight proportions, dynamically fusing both types of tokens via a weighted sum: $[\text{Frm}]_i = G_f(\text{[VG]})_0 \times [\text{Frm}_s]_i + G_f(\text{[VG]})_1 \times [\text{Frm}_t]_i$.
    - **Design Motivation**: Simply concatenating both types of tokens doubles the sequence length, severely hindering LLM decoding efficiency, while direct addition ignores the varying importance of the two features across different scenarios. The router dynamically determines which encoder to trust more during viewpoint or state transitions based on the scene content.

2. **Token Dropping Router**:

    - **Function**: Adaptively discards redundant visual tokens at each Transformer layer during LLM decoding to accelerate inference.
    - **Mechanism**: At each layer, routing weights are computed for each token as $r_{(i,n)}^l = w_\theta^T [\text{Frm}]_{(i,n)}^l$. Only tokens with weights exceeding the $\beta$-percentile threshold are retained for attention and FFN computations. Tokens below the threshold bypass the current layer, retaining their representations from the previous layer. The parameter $\beta$ controls the dropping ratio.
    - **Design Motivation**: In first-person scenarios, critical information is typically concentrated around hands and interaction regions; many tokens represent low-information backgrounds or duplicate information across nearly static consecutive frames. Discarding these redundant tokens significantly reduces FLOPs.

3. **Multi-granularity Keyframe Augmentation**:

    - **Function**: Injects fine-grained global and local features into keyframes during response generation in a **training-free** manner.
    - **Mechanism**: Global enhancement—the keyframe is divided into 4 grids, and a 3×3 pooling is applied to each grid to produce Grid Tokens, effectively quadrupling the spatial information density of a single frame. Local enhancement—a Faster R-CNN is employed to detect hand positions, and the bounding boxes of interacting objects are matched based on distance. Tokens within the box regions are selected from the 576 patch tokens and globally pooled to form Box Tokens. Both types of tokens are integrated into a Multimodal Thinking Template: "Stream: [Frame Tokens] [Grid Tokens] User: Please focus on [Box Tokens]. Assistant: ", which serves as a multimodal prompt directing refined response generation.
    - **Design Motivation**: The Fast Path provides only 10 tokens per frame, which is insufficient for detailed response generation. While incorporating fine-grained features for all frames is computationally prohibitive (compromising real-time capability), keyframes mark critical actions or event turning points, making keyframe-targeted enhancement the most cost-effective strategy.

### Loss & Training

The training objective comprises two components: Streaming Loss (for response decision-making), which supervises the network to predict the probability of the EOS token at each frame, and LM Loss (language modeling), which supervises the autoregressive generation of response text. The overall loss is formulated as $\text{Loss} = \frac{1}{N}\sum_j(-ws_j\log P_j^{[\text{EOS}]} - l_{j+1}\log P_j^{[\text{Txt}]_{j+1}})$. The Slow Path is training-free, and only the Fast Path requires training.

## Key Experimental Results

### Main Results

| Dataset | Method | LL-PPL↓ | TimeDiff↓ | Fluency↑ | LM-Correctness↑ |
|--------|------|---------|-----------|----------|------------------|
| Ego-Exo4D | VideoLLM-online | 2.24 | 0.78 | 33.7% | 44.8% |
| Ego-Exo4D | VideoLLM-MoD | 2.12 | 0.82 | 33.8% | 45.3% |
| Ego-Exo4D | **LION-FS** | **2.04** | **0.74** | **36.5%** | **48.2%** |
| Ego4D | VideoLLM-online | 2.40 | 2.04 | 45.3% | 49.0% |
| Ego4D | **LION-FS** | **2.09** | 2.15 | **46.1%** | **52.4%** |

### Ablation Study

| Configuration | LL-PPL↓ | TimeDiff↓ | Fluency↑ | LM-Correctness↑ |
|------|---------|-----------|----------|------------------|
| SigLIP only (10 tokens) | 2.24 | 0.78 | 33.7% | 44.8% |
| EgoVLP only (10 tokens) | 2.29 | 1.05 | 36.8% | 47.8% |
| Simple Concatenation (20 tokens) | 2.25 | 1.65 | 27.7% | 45.8% |
| Adaptive Routing (10 tokens) | **2.25** | **0.67** | **38.1%** | **48.0%** |
| + Token Dropping β=0.5 | 2.16 | 0.74 | 36.5% | 47.0% |

### Key Findings

- **Adaptive Routing is the optimal aggregation strategy**: Compared to concatenation (20 tokens but Fluency drops severely to 27.7%) and simple addition, routing aggregation improves both accuracy and temporal awareness without increasing the token count.
- **First-person features from EgoVLPv2 offer the most assistance to TimeDiff** ($0.67$ vs $0.78$), indicating that egocentric pre-training successfully captures crucial temporal action signals.
- **Token Dropping achieves the best trade-off at $\beta=0.5$**: FLOPs are reduced by 16% (61.44T $\rightarrow$ 51.40T), training is accelerated by 1.12×, with only a negligible decay in performance.
- The Slow Path is training-free, directly augmenting keyframes at inference time, offering high deployment flexibility.

## Highlights & Insights

- **The elegant "Fast-Slow decoupling" paradigm**: Separating response decision-making (simple task) and response generation (complex task), and optimizing them with efficient routing and fine-grained enhancement respectively, successfully circumvents the dilemma of "either low efficiency due to massive tokens, or poor quality due to over-compression". This task-difficulty adaptive design can be widely generalized to other VLM streaming scenarios.
- **The training-free Slow Path cleverly leverages task characteristics**: Expensive, fine-grained enhancement is invoked exclusively when a response is required, "injecting" the augmented tokens into the LLM's generation prefix via the Thinking Template. This yields substantial gains in response quality without incurring any additional training costs.
- **Dual-encoder routing fusion goes beyond mere feature concatenation**: It utilizes SigLIP's CLS token as a "dispatcher" to automatically prioritize the most reliable encoder based on visual context. This "visually guided" routing strategy is highly intuitive and effective.

## Limitations & Future Work

- The Slow Path relies on Faster R-CNN for hand and object detection, introducing additional inference latency and system complexity.
- Box Token extraction depends heavily on object detection models; detection failures can cause cascading performance degradation in response quality.
- The dropping ratio $\beta$ in the Token Dropping Router is globally fixed, whereas varying complexities of different scenarios may require an adaptive $\beta$.
- Evaluation was conducted exclusively on two egocentric datasets (Ego4D/Ego-Exo4D); the generalization capability to third-person scenarios remains unverified.
- The structure of the Thinking Template is hand-crafted; integrating more flexible prompt designs could further enhance generation quality.

## Related Work & Insights

- **vs VideoLLM-online (LIVE)**: LIVE processes low-frame-rate videos, does not distinguish between response decision and generation, and uses a fixed token count. LION-FS comprehensively improves on all three dimensions—achieving a 4× increase in frame rate, decoupling the fast-slow paths, and offering differentiated keyframe enhancement.
- **vs VideoLLM-MoD**: MoD introduces a Mixture-of-Depths strategy to mitigate computation but lacks both egocentric-specific features and multi-granularity keyframe augmentation. The Token Dropping Router in LION-FS shares a similar philosophy but integrates it with egocentric encoders.
- **vs SlowFast-LLaVA**: SlowFast-LLaVA applies different pooling granularities at varying frame rates to enrich features, but it operates in an offline manner. LION-FS extends the Fast-Slow concept to online streaming videos with an emphasis on real-time execution.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of fast-slow decoupling, dual-encoder routing, and training-free enhancement is novel, though individual components are not entirely unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐ The ablation study is highly detailed, though validation on only two datasets is slightly limiting.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, beautifully illustrated, and provides exceptionally clear motivation.
- Value: ⭐⭐⭐⭐ Establishes a highly practical fast-slow inference framework for online video assistants, providing valuable insights for wearable AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](../../NeurIPS2025/video_understanding/livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2025\] EgoLife: Towards Egocentric Life Assistant](egolife_towards_egocentric_life_assistant.md)
- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)

</div>

<!-- RELATED:END -->
