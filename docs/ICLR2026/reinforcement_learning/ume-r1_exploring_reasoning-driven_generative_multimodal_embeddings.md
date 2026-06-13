---
title: >-
  [Paper Note] UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings
description: >-
  [ICLR 2026][Reinforcement Learning][multimodal embeddings] This paper proposes UME-R1, the first framework to explore a reasoning-driven generative multimodal embedding paradigm. Through a two-stage training pipeline (co…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "multimodal embeddings"
  - "reasoning-driven generation"
  - "reinforcement-learning"
  - "MLLM"
  - "inference-time scaling"
date: 2026-05-08
content_hash: f17be8ec0a4ab6ae
---

# UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings

**Conference**: ICLR 2026
**arXiv**: [2511.00405](https://arxiv.org/abs/2511.00405)  
**Code**: [GitHub](https://github.com/XMUDeepLIT/UME-R1)  
**Area**: Reinforcement Learning
**Keywords**: multimodal embeddings, reasoning-driven generation, reinforcement-learning, MLLM, inference-time scaling

## TL;DR

This paper proposes UME-R1, the first framework to explore a reasoning-driven generative multimodal embedding paradigm. Through a two-stage training pipeline (cold-start SFT followed by reinforcement learning), the embedding model learns to reason before generating representations, achieving significant improvements over traditional discriminative embedding models across 78 tasks on the MMEB-V2 benchmark.

## Background & Motivation

**Background**: MLLM-based embedding models (e.g., VLM2Vec, MM-Embed) have achieved notable progress on multimodal embedding tasks, substantially outperforming traditional dual-encoder vision-language models such as CLIP. Concurrently, large reasoning models (LRMs) represented by DeepSeek-R1 have demonstrated breakthroughs on complex reasoning tasks.

**Limitations of Prior Work**: Existing MLLM-based multimodal embedding models are fundamentally discriminative — they directly encode inputs and extract the hidden state of the last token as the embedding, without generating any new tokens. This prevents them from benefiting from reasoning-driven generative paradigms. While some works (e.g., CAFe) incorporate next-token prediction loss during training to preserve generative capacity, inference remains discriminative.

**Key Challenge**: A fundamental gap exists between reasoning capability and embedding quality — embedding tasks lack verification mechanisms with ground-truth answers analogous to mathematics, making it difficult to directly apply reinforcement learning to optimize embedding models.

**Goal**: How can multimodal embedding models operate under a generative paradigm, enabling them to reason before generating higher-quality embeddings? How can RL be successfully applied to embedding tasks that lack ground-truth answers?

**Key Insight**: Unify embedding tasks under a generative paradigm where the model first generates a reasoning process and summary, and subsequently produces embeddings conditioned on this context. RL optimization is enabled through a composite reward combining ranking score and similarity margin.

**Core Idea**: The embedding model first engages in deliberate reasoning before producing representations; RL continuously improves reasoning quality, realizing inference-time scaling for embedding tasks.

## Method

### Overall Architecture

UME-R1 is a universal multimodal embedding framework built on a two-stage training strategy: (1) cold-start SFT endows the model with reasoning capability and the ability to simultaneously produce discriminative and generative embeddings; (2) RL further enhances reasoning and refines generative embedding quality. At inference time, the model can flexibly switch between discriminative and generative embedding modes as needed.

### Key Designs

**Design 1: Dual-Mode Embedding Architecture**
- **Function**: A unified template is designed so that the model simultaneously produces discriminative and reasoning-driven generative embeddings.
- **Mechanism**: A `<disc_emb>` token is inserted into the prompt for discriminative embedding; the model then generates `<think>...</think><answer>` for reasoning and summarization, followed by a `<gen_emb>` token for the generative embedding. Both embedding types are obtained by extracting the hidden state of the corresponding token at the final layer.
- **Design Motivation**: Discriminative embeddings incur no additional overhead, while generative embeddings leverage reasoning information to provide richer semantic representations. The two are complementary — oracle experiments show that their combined upper bound substantially exceeds either modality used alone.

**Design 2: Cold-Start Dataset Construction**
- **Function**: Generate chain-of-thought (CoT) annotations for 1.76M query–target pairs.
- **Mechanism**: GLM-4.1V-Thinking is used to generate separate reasoning traces for the query and target of each pair. After filtering (removing samples with repeated tokens, excessively long reasoning, or malformed formats), 1.46M SFT pairs and 11K RL training pairs are obtained.
- **Design Motivation**: Cold-start data enables the model to acquire fundamental reasoning and embedding generation capabilities during the SFT stage.

**Design 3: Embedding Reward Function**
- **Function**: Design a verifiable RL reward for embedding tasks.
- **Mechanism**: Reward = ranking score × similarity margin. The ranking score measures the proportion of samples whose similarity to the positive example is ranked above a given threshold; the similarity margin captures the mean similarity difference between positive and negative pairs. The two components are multiplied to yield a composite reward.
- **Design Motivation**: Addresses the challenge that embedding tasks lack ground-truth answers. Using a threshold alone leads to zero policy gradient issues when pairs are either too easy or too hard. The ranking-plus-margin composite design is more robust.

### Loss & Training

**SFT Stage**: Sum of three loss terms:
- Discriminative contrastive loss $\mathcal{L}_{dctr}$ (InfoNCE)
- Generative contrastive loss $\mathcal{L}_{gctr}$ (InfoNCE conditioned on reasoning trajectories)
- Autoregressive cross-entropy loss $\mathcal{L}_{ce}$ (applied to reasoning and summary tokens)

**RL Stage**: Optimized with GRPO:
- Format reward: whether the model strictly adheres to the `<think>...</think><answer>` template
- Embedding reward: composite ranking × similarity margin reward
- Group size $G=8$, $\varepsilon=0.2$, $\beta=0.04$, batch size 256, learning rate $1\times10^{-6}$

## Key Experimental Results

### Main Results

| Model | Image | Video | VisDoc | All |
|------|-------|-------|--------|-----|
| VLM2Vec-V2 (2B) | 64.9 | 34.9 | 65.4 | 58.0 |
| CAFe (7B) | 67.6 | 42.4 | 63.9 | 60.6 |
| DUME (2B) | 62.5 | 33.2 | 52.8 | 52.7 |
| **UME-R1 (2B)** | **66.6** | **42.2** | **63.9** | **60.1** |
| UME-R1 Oracle (2B) | +4.3 | — | — | — |
| UME-R1 Oracle (7B) | +3.6 | — | — | — |

Using only two-thirds of the training data of VLM2Vec-V2, UME-R1 achieves an overall improvement of 2.1 points.

### Ablation Study

| Component | Image | Video | VisDoc |
|------|-------|-------|--------|
| DUME (discriminative only) | 62.5 | 33.2 | 52.8 |
| + Generative embedding (SFT) | 66.6 (+4.1) | 42.2 (+9.0) | 63.9 (+11.1) |
| + RL | Further improvement | — | — |
| Oracle (best of disc. + gen.) | +4.3 (2B) / +3.6 (7B) | — | — |

### Key Findings

1. **Generative embeddings substantially outperform discriminative ones**: Under identical data, UME-R1 improves by 4.1 / 9.0 / 11.1 points on image, video, and document modalities, respectively.
2. **The two embedding types are highly complementary**: The oracle upper bound far exceeds either mode used alone, suggesting that adaptive selection strategies hold significant practical value.
3. **RL effectively improves generative embeddings**: Demonstrating that RLVR is extensible to embedding tasks that lack ground-truth answers.
4. **Inference-time scalability**: Repeated sampling improves pass@k coverage, indicating that inference-time scaling has potential for embedding tasks.

## Highlights & Insights

- **Paradigm innovation**: This is the first work to introduce a reasoning-driven generative paradigm into multimodal embeddings, challenging the conventional assumption that embedding models must be discriminative.
- **Breakthrough in embedding RL**: The ranking × similarity margin reward elegantly resolves the zero-gradient problem inherent in RL training for embedding tasks without ground-truth answers.
- **Flexibility**: The model can simultaneously output both discriminative and generative embeddings, allowing users to select the appropriate mode for their use case.
- **Inference-time scaling**: The pass@k results suggest that inference-time scaling exists for embedding tasks as well — a notably forward-looking finding.
- **Data efficiency**: Superior performance is achieved using only two-thirds of the training data of VLM2Vec-V2.

## Limitations & Future Work

1. **Reasoning overhead**: Generative embeddings require prior generation of reasoning traces and summaries, incurring substantially higher inference latency that is unsuitable for latency-sensitive applications.
2. **CoT annotation dependency**: SFT data relies on GLM-4.1V-Thinking to generate CoT annotations, making annotation quality contingent on the teacher model's capabilities.
3. **Large oracle gap**: The gap between the oracle and single-mode embedding still reaches 3–4 points, indicating that the current mode selection strategy has room for improvement.
4. **Evaluation primarily on MMEB-V2**: Validation on additional downstream tasks (e.g., retrieval engines, RAG systems) is needed to assess real-world effectiveness.
5. **Small RL dataset**: Only 11K RL training pairs are used; scaling up RL data may yield further gains.

## Related Work & Insights

- **Relationship to VLM2Vec / VLM2Vec-V2**: The same discriminative embedding framework serves as the backbone, upon which generative capabilities are extended.
- **Relationship to DeepSeek-R1**: The reasoning-driven generative paradigm is adopted and transferred from question answering to embedding tasks.
- **Implications for RAG systems**: Reasoning-driven embeddings may yield significant gains on complex retrieval tasks that require understanding query intent.
- **Implications for inference-time scaling research**: The pass@k results indicate that inference-time scaling is a viable direction for embedding tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce reasoning-driven generative paradigm into embedding tasks, opening an entirely new research direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Coverage of 78 tasks across three modalities with thorough ablations; however, latency analysis and real-world application validation are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear and methodology is well described, though notation is dense in places.
- **Value**: ⭐⭐⭐⭐⭐ — Establishes a new paradigm for embedding models; the RL reward design and inference-time scaling findings carry broad implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)
- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](rm-r1_reward_modeling_as_reasoning.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICLR 2026\] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)

</div>

<!-- RELATED:END -->
