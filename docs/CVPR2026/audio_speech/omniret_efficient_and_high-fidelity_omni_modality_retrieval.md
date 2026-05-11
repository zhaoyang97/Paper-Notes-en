---
title: >-
  [Paper Note] OmniRet: Efficient and High-Fidelity Omni Modality Retrieval
description: >-
  [CVPR 2026][Audio & Speech][omni-modal retrieval] This paper proposes OmniRet, the first unified retrieval model supporting composed queries across text, vision…
tags:
  - "CVPR 2026"
  - "Audio & Speech"
  - "omni-modal retrieval"
  - "multimodal embedding"
  - "Sliced Wasserstein"
  - "composed query"
  - "audio retrieval"
date: 2026-05-08
content_hash: f95cf8672ced8c09
---

# OmniRet: Efficient and High-Fidelity Omni Modality Retrieval

**Conference**: CVPR 2026
**arXiv**: [2603.02098](https://arxiv.org/abs/2603.02098)
**Code**: [hmchuong/omniret](https://github.com/hmchuong/omniret)
**Area**: Audio & Speech
**Keywords**: omni-modal retrieval, multimodal embedding, Sliced Wasserstein, composed query, audio retrieval

## TL;DR

This paper proposes OmniRet, the first unified retrieval model supporting composed queries across text, vision, and audio modalities. It introduces a Shared Media Resampler to improve computational efficiency and Attention Sliced Wasserstein Pooling (ASWP) to preserve fine-grained information, achieving state-of-the-art performance on 12 out of 13 retrieval tasks.

## Background & Motivation

**Real-world demand for multimodal retrieval**: Information retrieval has evolved from single-modality search to composite query scenarios spanning heterogeneous data such as images, videos, audio, and text, which existing systems struggle to handle across more than three modalities.

**Modality limitations of existing models**: Classic models such as CLIP, BLIP, and CLAP support alignment between only two modalities (text–vision or text–audio) and cannot handle composed queries involving all three modalities simultaneously.

**Information bottleneck**: Compressing rich multimodal inputs into a single embedding vector causes severe information loss; simple mean pooling or [EOS] token approaches discard fine-grained information from LLM outputs.

**Computational efficiency bottleneck**: Token sequences produced by media encoders typically exceed 500 tokens; feeding them directly into an LLM leads to prohibitive computational costs, constraining training batch size and thereby weakening contrastive learning.

**Cost of late interaction**: Methods such as ColBERT that retain token-level embeddings achieve high information fidelity but incur excessive storage and computational costs, making them impractical for large-scale retrieval systems.

**Absence of audio retrieval benchmarks**: No systematic evaluation benchmark exists for composed audio retrieval (audio+text→audio) or audio–visual retrieval (audio→image/video), limiting research progress in this direction.

## Method

### Overall Architecture

OmniRet uses GTE-Qwen2-1.5B-Instruct as the core LLM cross-modal composer, with visual inputs encoded by SigLIP-SO400M and audio inputs encoded by the QwenAudio Encoder. Tokens from each modality are compressed by a Shared Media Resampler and then interleaved according to an instruction template before being fed into the LLM. ASWP subsequently aggregates LLM outputs into a single embedding. Training updates only the resampler, projection layers, pooling layer, and LLM LoRA (rank=16), totaling approximately 84M trainable parameters.

### Key Design 1: Shared Media Resampler

A Perceiver architecture compresses the large token sequences (>500) produced by each modality encoder into a fixed number of compact latent vectors. The key design choice is to **share a single Perceiver module while introducing independent latent queries per modality**, preserving modality-specific characteristics while maintaining cross-modal generalization. For video inputs, 3D trilinear interpolation first reduces frame-level redundancy before resampling.

### Key Design 2: Attention Sliced Wasserstein Pooling (ASWP)

An attention resampler first compresses LLM outputs into $S$ latent embeddings $\mathbf{Z}$, which are then treated as a distribution. Monge coupling distances are computed along $L$ one-dimensional projection directions against $S$ learnable reference points $\mathbf{X}$, yielding an intermediate representation $\mathbf{Z}' \in \mathbb{R}^{S \times L}$. A Straight-Through Maximum (STM) mechanism then generates binary attention masks to select the most relevant reference point for each projection direction, and column-wise summation produces the final $L$-dimensional embedding. The default configuration uses $L=4096, S=128$, achieving the best balance between information fidelity and computational efficiency.

### Key Design 3: Diversity Regularization Loss

To ensure that resampled tokens capture diverse information, an orthogonality constraint is imposed on the output vectors $\mathbf{M}$: the pairwise similarity matrix $\mathbf{MM}^\top$ is computed, the diagonal (self-similarity) is removed, and Dropout-based sparse sampling is applied to the residual matrix before penalizing non-orthogonality with Smooth L1 loss ($\gamma=0.5$). Dropout ensures that only a random subset is used at each step, efficiently encouraging global diversity.

### Key Design 4: Two-Stage Training Strategy

- **Stage 1 (Warm-up)**: Projection layers, resampler, and pooling layer are trained on single-modality and text-binding tasks with the LLM frozen; batch size 2048, 2M samples total.
- **Stage 2 (Fine-tuning)**: Training continues on all 30 datasets (~6.2M query–target pairs) with LoRA fine-tuning of the LLM; batch size 3072, 4 tasks randomly sampled per batch, gradient accumulation of 2 steps, 18M samples total.

## Loss & Training

The total loss is a weighted combination of three terms:

$$\mathcal{L} = \mathcal{L}_{\text{cont}} + \mu_1 \mathcal{L}_{\text{triplet}} + \mu_2 \mathcal{L}_{\text{div}}$$

- $\mathcal{L}_{\text{cont}}$: Hard-negative InfoNCE contrastive loss, temperature $\tau=0.07$, adaptive weight $\beta=0.5$
- $\mathcal{L}_{\text{triplet}}$: Hinge-based triplet loss, margin $\eta=0.1$
- $\mathcal{L}_{\text{div}}$: Diversity regularization loss
- Weights: $\mu_1=1, \mu_2=0.1$

## Key Experimental Results

### Table 1: Extended M-BEIR 13-Task Recall Comparison (1.5B Models)

| Model | I→I | T→T | I→T | T→I | V→T | T→V | A→T | T→A | T→I,T | I,T→T | I,T→I | I,T→I,T | V,T→V |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-------|-------|-------|---------|-------|
| VLM2VecV2 | 30.0 | 81.1 | 43.4 | 39.8 | 17.6 | 18.4 | - | - | 61.6 | 24.5 | 28.7 | 33.6 | 76.4 |
| **OmniRet** | 24.4 | **86.7** | **50.6** | **46.9** | **43.8** | **43.2** | **66.8** | **62.4** | **70.5** | **44.4** | **36.5** | **64.8** | **86.2** |

OmniRet achieves the best performance on 12 of 13 tasks, substantially outperforming all task-specific models on audio and video tasks.

### Table 2: MMEBv2 Subset Generalization (Recall@1)

| Model | Image CLS | Image RET | Video CLS | Video RET | Video MRET |
|-------|-----------|-----------|-----------|-----------|------------|
| VLM2VecV2 | 62.9 | 69.5 | 39.3 | 28.8 | 38.5 |
| **OmniRet** | 51.7 | 65.3 | **48.6** | **36.5** | **43.3** |

OmniRet achieves state-of-the-art on all video tasks and maintains competitive performance on image tasks despite not using the corresponding training data.

### Table 3: ACM Benchmark (Recall@5)

| Model | A,T→A | A→V | V→A | A→I | I→A |
|-------|-------|-----|-----|-----|-----|
| ImageBind | 7.32 | **35.5** | **36.3** | **30.1** | **29.7** |
| **OmniRet** | **23.0** | **35.5** | 34.4 | 24.5 | 26.0 |

OmniRet substantially outperforms on composed audio retrieval (A,T→A) and matches ImageBind on audio–video retrieval.

### Ablation Study

- Replacing ASWP with [EOS] vector: Recall drops by **6.8%**
- Removing the Media Resampler: drops by **3.5%**
- Removing $\mathcal{L}_{\text{div}}$: drops by **3.1%**
- Replacing STM with Average Pooling in ASWP: drops by **29.5%**

## Highlights & Insights

- **First tri-modal unified retrieval**: OmniRet is the first system to support composed queries across text, vision, and audio, filling the gap left by the absence of audio modality in general-purpose retrieval.
- **Efficiency–fidelity balance**: The Shared Media Resampler compresses 500+ tokens to a fixed number of latents, while ASWP retains token-level fine-grained information within a single-vector format compatible with ANN indexing.
- **New benchmark contribution**: The ACM Benchmark introduces two novel tasks—composed audio retrieval and audio–visual retrieval—with quality verified through human evaluation.
- **Thorough ablation**: Five ablation groups covering embedding type, number of projections/references, pooling method, resampler design, and loss function quantitatively validate each component's contribution.

## Limitations & Future Work

- Due to computational constraints, the scaling behavior of larger LLM backbones and additional training data was not explored.
- Coverage is limited to text, vision, and audio; modalities such as depth maps, 3D point clouds, and speech are not addressed.
- The ACM Benchmark scenarios are relatively simple and do not involve complex retrieval over interleaved mixed-media documents.
- Single-modality image retrieval (I→I) still lags behind specialized models such as PE-Core (24.4 vs. 32.0).

## Related Work & Insights

- **Multimodal Embedding**: The CLIP/BLIP family focuses on text–vision alignment; CLAP focuses on text–audio; ImageBind attempts a six-modality joint space but lacks computational efficiency.
- **General Multimodal Retrieval**: UniIR pioneered general-purpose retrievers trained across multiple datasets; VLM2Vec leverages VLMs for embedding; MMEmbed exploits LLM instruction-following capabilities.
- **Embedding Pooling**: From mean pooling/[EOS] to ColBERT-style late interaction and NV-Embed's learnable queries, OmniRet's ASWP finds a balance between single-vector representations and late interaction.

## Rating

- Novelty: ⭐⭐⭐⭐ — First tri-modal unified retrieval framework; ASWP pooling and ACM Benchmark are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluation across 13+ tasks, MMEBv2 generalization, a new benchmark, and five ablation groups provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-defined problem formulation, rich figures and tables, and complete mathematical derivations.
- Recommendation: ⭐⭐⭐⭐ — Advances modality coverage and the efficiency–quality trade-off in multimodal retrieval with high practical value.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition](../../ICCV2025/audio_speech/lyra_an_efficient_and_speechcentric_framework_for_omnicognit.md)
- [\[CVPR 2026\] Omni-MMSI: Toward Identity-Attributed Social Interaction Understanding](omni-mmsi_toward_identity-attributed_social_interaction_understanding.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[ICLR 2026\] TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization](../../ICLR2026/audio_speech/triplesumm_adaptive_triple-modality_fusion_for_video_summarization.md)

</div>

<!-- RELATED:END -->
