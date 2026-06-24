---
title: >-
  [Paper Note] Vision-centric Token Compression in Large Language Model
description: >-
  [NeurIPS 2025 Spotlight][Model Compression][token compression] Vist proposes a vision-centric slow-fast dual-path token compression framework that renders distant long-context text as images and compresses them with a lightweight vision encoder, coupled with a Probability-guided Visual Enhancement (PVE) training objective. Across 11 ICL benchmarks, it achieves comparable accuracy with 2.3× fewer tokens, reducing FLOPs by 16% and memory by 50%.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Model Compression"
  - "token compression"
  - "vision encoder"
  - "long context"
  - "in-context learning"
  - "frequency-based masking"
date: 2026-05-08
content_hash: 70fa6064d39cf342
---

# Vision-centric Token Compression in Large Language Model

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.00791](https://arxiv.org/abs/2502.00791)  
**Code**: [https://github.com/CSU-JPG/VIST](https://github.com/CSU-JPG/VIST)  
**Area**: Model Compression / Token Compression  
**Keywords**: token compression, vision encoder, long context, in-context learning, frequency-based masking

## TL;DR

Vist proposes a vision-centric slow-fast dual-path token compression framework that renders distant long-context text as images and compresses them with a lightweight vision encoder, coupled with a Probability-guided Visual Enhancement (PVE) training objective. Across 11 ICL benchmarks, it achieves comparable accuracy with 2.3× fewer tokens, reducing FLOPs by 16% and memory by 50%.

## Background & Motivation

Large language models (LLMs) face dual pressures from ever-growing context windows and expanding model scales. Real-world tasks such as long-document understanding and multi-turn question answering demand effective long-context modeling, yet processing extremely long token sequences incurs prohibitive computational and memory costs. Most existing token compression methods rely on the LLM itself to compute token-level entropy for importance estimation (e.g., the LLMLingua series), which makes the compression process inherently expensive as it requires the heavyweight LLM to participate.

Psycholinguistic research shows that skilled human readers skip approximately one-third of high-frequency function words (e.g., "the", "of") during rapid skimming, focusing attention on rare content words. This selective reading strategy naturally forms a slow-fast circuit: a fast visual channel glosses over low-salience distant context to maintain global awareness, while a slow cognitive channel deeply processes proximal critical sentences.

The core idea of this paper is: **render distant, less-relevant text as images and let a frozen lightweight vision encoder (e.g., CLIP ViT) serve as a "fast eye" for coarse scanning, while the LLM acts as the "brain" focusing on proximal key information for deep reasoning.** This approach leverages the OCR capabilities inherent in pretrained vision encoders, bypassing the vocabulary bottleneck and character-level noise sensitivity of conventional text tokenizers.

## Method

### Overall Architecture

Vist (Vision-centric Token Compression) adopts a slow-fast dual-path design:

1. **Fast visual path**: The first $T_e$ text tokens are uniformly rendered into $M$ RGB images (each 224×224) and fed into a frozen CLIP ViT-L/14 encoder for feature extraction, followed by a trainable Perceiver Resampler that compresses the features into a fixed number of visual tokens.
2. **Slow cognitive path**: The remaining $T_d$ original text tokens are fed directly into the LLM for fine-grained reasoning.
3. The compressed visual tokens are injected into the LLM via cross-attention and jointly processed with the original text tokens for next-token prediction.

### Key Designs

1. **Text-to-image rendering**: Text is rendered into RGB images using a 10px font size and Google Noto Sans (H=14, W=3584, C=3), equivalent to 224×224 resolution. Blank regions are excluded from attention and loss computation via masking. Every 1,024 text tokens require rendering into 7 images.

2. **Perceiver Resampler compression**: The frozen ViT-L/14 extracts image features $F \in \mathbb{R}^{M \times L \times D}$, which are compressed by a learnable Perceiver Resampler into $N+1$ visual tokens per image (including a CLS token), with $N=64$ by default. During training, 4,096 text tokens are rendered into 28 images and compressed into $64 \times 28 = 1792$ visual tokens, yielding a compression ratio of $\Delta = 2.3$.

3. **Probability-guided Visual Enhancement (PVE)**: The core training objective, comprising two key components:

    - **Text-anchored semantic consistency**: A contrastive learning loss that pulls the visual features $\hat{F}'$ output by the Resampler closer to the text token embeddings $\hat{F}^t$ produced by the LLM tokenizer.
    - **Frequency-based Masking (FM)**: Inspired by Shannon information theory ($I(y) = -\log_2 P(y)$), corpus-level token frequency is used as a proxy for semantic importance. High-frequency tokens (e.g., "the", "with") carry little information and are preferentially masked; low-frequency tokens (domain-specific or contextually critical keywords) are retained. The importance score is $s_w = \log \frac{|S|}{1+\text{count}(w)}$, with a masking rate of 50%, where lower-importance tokens are masked with higher probability.

### Loss & Training

- Training objective: next-token prediction loss + PVE contrastive loss
- PVE contrastive loss: $\mathcal{L}_{PVE}^{ij} = -\log \frac{\exp(\langle \hat{F}'_i, \hat{F}^t_j \rangle / \tau)}{\sum_{k=1}^B \exp(\langle \hat{F}'_i, \hat{F}^t_k \rangle / \tau)}$
- Training uses float16 precision + DeepSpeed Zero-2 + CPU offloading
- Base LLM: TinyLlama; pretraining data: RedPajama 11B tokens (7 domains including ArXiv, Book, C4, etc.)
- The Perceiver Resampler and LLM cross-attention are jointly trained end-to-end

## Key Experimental Results

### Main Results: Long-Context Language Modeling (PPL)

| Method | $T_e$ | $T_d$ | ArXiv | Book | PG19 | TFLOPs | MEM(GB) |
|--------|--------|--------|-------|------|------|--------|---------|
| TinyLlama | - | 4096 | >10³ | >10³ | >10³ | 8.47 | 5.46 |
| CEPE* | 6144 | 2048 | 3.005 | 14.919 | 11.112 | 13.27 | 7.74 |
| Vist | 6144 | 2048 | **2.989** | **14.894** | 12.737 | **11.65** | **4.94** |
| CEPE* | 14336 | 2048 | 3.003 | 14.921 | 10.909 | 23.30 | 13.59 |
| Vist | 14336 | 2048 | **2.965** | **14.815** | 11.933 | **19.52** | **6.75** |

### Open-Domain QA (Exact Match)

| Method | $k_e$ | $k_d$ | TriviaQA | NQ | PopQA |
|--------|--------|--------|----------|------|-------|
| TinyLlama | - | 10 | 21.45 | 8.45 | 10.79 |
| CEPE* | 20 | 10 | 16.56 | 6.75 | 5.78 |
| Vist | 20 | 10 | **25.67**(+9.11) | **8.81**(+2.06) | **11.84**(+6.06) |

### Ablation Study

| Configuration | NLUS | NLUI | TriviaQA | NQ | PopQA | Note |
|---------------|------|------|----------|------|-------|------|
| No masking | 9.9 | 26.4 | 17.14 | 6.51 | 5.72 | Baseline |
| Random masking | 8.3 | 30.2 | 24.88 | 8.35 | 10.19 | Helpful but insufficient |
| Frequency masking (FM) | **15.6** | **40.6** | **25.20** | **8.71** | **11.44** | FM is critical |

### Key Findings

- At 14K token input, Vist saves 3.78 TFLOPs and 6.84 GB memory compared to CEPE*, with a 2.3× throughput improvement.
- On open-domain QA, Vist outperforms CEPE* by an average of 5.7% (EM), because PVE guides the Resampler to focus on key semantics, whereas CEPE* introduces noise as more passages are added.
- The frequency-based masking strategy (50% masking rate) retains the majority of high information-gain (IG) tokens, demonstrating that token frequency is an effective proxy for semantic importance.
- 64 visual tokens per image is the optimal configuration; more tokens (e.g., 128) introduce noise.
- Scaling to Mistral 7B is equally effective, with PPL superior to the corresponding CEPE baseline.

## Highlights & Insights

- **Paradigm innovation**: This is the first work to address LLM long-context compression from a visual perspective, rendering text as images for processing by a lightweight vision encoder and bypassing the vocabulary bottleneck of conventional text tokenizers.
- **Biological inspiration**: The slow-fast dual-path design is inspired by the selective reading strategies of human readers as studied in psycholinguistics, elegantly translating an academic observation into an engineering solution.
- **Simple yet effective PVE**: Token frequency is used as a substitute for expensive LLM-based entropy computation to assess token importance, substantially reducing the computational overhead of compression.
- Four key advantages of using a vision encoder as a "visual text tokenizer": simplified tokenization, alleviation of vocabulary bottleneck, robustness to character-level noise, and efficient multilingual processing.

## Limitations & Future Work

- Experiments are currently limited to TinyLlama and Mistral 7B; validation on larger-scale LLMs (e.g., 70B+) is absent.
- On tasks with high category diversity (e.g., NLUS, TREC, TREF), the lightweight encoding path still lags behind full LLM processing.
- The text-to-image rendering approach theoretically benefits non-Latin scripts (Chinese, Japanese, etc.) by reducing token counts, but this remains empirically unverified.
- PPL on literary and mathematical text (PG19, Proof) is slightly inferior to CEPE*, indicating room for improvement in compressing purely textual semantic content.

## Related Work & Insights

- Vist forms a direct comparison with CEPE (text encoder compression), replacing the text encoder with a vision encoder to achieve competitive or superior performance while significantly reducing memory usage.
- Complementary to the LLMLingua series (selective compression based on LLM entropy): Vist does not rely on the LLM to compute token importance.
- The approach extends the ideas of Pixel (pretraining on text rendered as images) to the long-context compression setting.
- Insight: Lightweight vision encoders may serve as "front-end denoisers" for LLMs in certain scenarios, a concept potentially extensible to multimodal RAG and related applications.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](../../ICCV2025/model_compression/b-vllm_a_vision_large_language_model_with_balanced_spatio-temporal_tokens.md)
- [\[ICML 2025\] RADIO: Rate-Distortion Optimization for Large Language Model Compression](../../ICML2025/model_compression/radio_rate-distortion_optimization_for_large_language_model_compression.md)
- [\[NeurIPS 2025\] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)
- [\[ACL 2025\] Quantification of Large Language Model Distillation](../../ACL2025/model_compression/quantification_of_large_language_model_distillation.md)
- [\[CVPR 2025\] DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](../../CVPR2025/model_compression/dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)

</div>

<!-- RELATED:END -->
