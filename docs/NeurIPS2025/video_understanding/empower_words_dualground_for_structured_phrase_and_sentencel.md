---
title: >-
  [Paper Note] Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding
description: >-
  [NeurIPS 2025][Video Understanding][Video Temporal Grounding] DualGround identifies a critical issue in existing VTG models — over-reliance on the global semantics of the [EOS] token while neglecting word-level signals — and proposes a sentence-level + phrase-level dual-path architecture. Through an Adaptive Cross-Attention (ACA) module and a Recurrent Phrase Generator (RPG), the model captures global and local semantics respectively, achieving state-of-the-art performance on QVHighlights and Charades-STA.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Video Temporal Grounding
  - Dual-Path Architecture
  - Phrase-Level Alignment
  - Moment Retrieval
  - Attention Decoupling
date: 2026-05-08
content_hash: ee996bc5812518d7
---

# Empower Words: DualGround for Structured Phrase and Sentence-Level Temporal Grounding

**Conference**: NeurIPS 2025
**arXiv**: [2510.20244](https://arxiv.org/abs/2510.20244)
**Code**: None
**Area**: Video Understanding / Temporal Grounding
**Keywords**: Video Temporal Grounding, Dual-Path Architecture, Phrase-Level Alignment, Moment Retrieval, Attention Decoupling

## TL;DR

DualGround identifies a critical issue in existing VTG models — over-reliance on the global semantics of the [EOS] token while neglecting word-level signals — and proposes a sentence-level + phrase-level dual-path architecture. Through an Adaptive Cross-Attention (ACA) module and a Recurrent Phrase Generator (RPG), the model captures global and local semantics respectively, achieving state-of-the-art performance on QVHighlights and Charades-STA.

## Background & Motivation

### State of the Field

Video Temporal Grounding (VTG) encompasses two subtasks: Moment Retrieval (MR) and Highlight Detection (HD). Existing models typically process text features from pretrained encoders such as CLIP and InternVideo2 by treating all text tokens (word tokens + [EOS] token) uniformly.

Through controlled experiments, the authors identify a critical issue:

### Limitations of Prior Work

**[EOS] Bias**: Models rely almost exclusively on the global semantics of the [EOS] token, while word-level tokens are severely neglected.

### Root Cause

Using only the [EOS] token yields performance comparable to or even better than using the full token sequence.

### Starting Point

Even for video segments irrelevant to the query, attention concentrates on [EOS] rather than on semantically relevant keywords (e.g., "red jacket").

### Supplementary Remarks

This leads to degraded performance in scenarios requiring fine-grained word-to-video alignment.

## Method

### Overall Architecture

The dual-path architecture processes sentence-level and phrase-level semantics separately:
1. **Sentence-Level Path**: Utilizes the [EOS] token with Adaptive Cross-Attention (ACA) to capture global alignment.
2. **Phrase-Level Path**: Word tokens → Recurrent Phrase Generator (RPG) → Slot Attention refinement → phrase-segment interaction.
3. The outputs of both paths, $V^s + V^p$, are fused and fed into the decoder.

### Key Designs

1. **Adaptive Cross-Attention (ACA, Sentence-Level Path)**:
    - **Function**: Enables each video segment to selectively align with the global semantics of the [EOS] token.
    - **Mechanism**: Introduces learnable dummy tokens as "attention attractors"; video segments semantically unrelated to the query distribute attention to dummy tokens rather than to [EOS], reducing noise.
    - **Design Motivation**: A single-token ([EOS]) key sequence is incompatible with standard attention; dummy tokens provide the basis for a contrastive distribution.
    - **Implementation**: Dummy tokens are processed through a conditional encoder and concatenated with [EOS] as key-value pairs; video features serve as queries.

2. **Recurrent Phrase Generation + Slot Attention Refinement (Phrase-Level Path)**:
    - **Function**: Clusters word tokens into $N$ semantically coherent phrase units and models fine-grained interactions between phrases and video segments.
    - **Mechanism**: The RPG module sequentially generates $N$ phrases conditioned on the global semantics and the previous phrase; Slot Attention iteratively refines the phrases to remove semantic overlap.
    - **Design Motivation**: Word-level semantics are context-dependent; phrase-level abstraction provides more coherent alignment units, and sequential generation encourages adjacent words to be grouped into the same phrase.
    - **Phrase-Segment Interaction**: Contextual embeddings $C \in \mathbb{R}^{N \times T \times d}$ are computed for all phrases across all timesteps via Hadamard product.

### Loss & Training

- **MR Loss**: Focal classification loss + L1 boundary regression loss.
- **HD Loss**: Ranking loss + contrastive loss (applied to saliency scores and attention weights respectively).
- **Phrase-Level Loss**:
    - DQA Loss: Regularizes the orthogonality of phrase attention distributions via $\|AA^T - rI\|_F^2$, encouraging semantic diversity.
    - Global Reconstruction Loss: Encourages $P_{[EOS]}$ to reconstruct global semantics, ensuring consistency between the phrase-level and sentence-level paths.
- Total Loss: $\mathcal{L} = \mathcal{L}_{mr} + \mathcal{L}_{hd} + \mathcal{L}_{phrase}$

## Key Experimental Results

### Main Results

QVHighlights test split (InternVideo2 features):

| Method | MR R1@0.5 | MR R1@0.7 | MR mAP@0.5 | HD mAP | HD HIT@1 |
|--------|-----------|-----------|------------|--------|----------|
| FlashVTG | 74.7 | 60.0 | 54.8 | 40.0 | 66.2 |
| R2-Tuning | 72.5 | 57.3 | 52.1 | 39.4 | 65.0 |
| **DualGround** | **>75** | **>61** | **>55** | **>40** | **>67** |

Charades-STA (InternVideo2 features):

| Method | R1@0.5 | R1@0.7 |
|--------|--------|--------|
| FlashVTG | ~73 | ~55 |
| **DualGround** | **Higher** | **Higher** |

DualGround achieves state-of-the-art performance on both benchmarks across MR and HD tasks.

### Ablation Study

- Sentence-level path vs. phrase-level path vs. dual-path: the dual-path configuration significantly outperforms either single-path variant.
- Number of dummy tokens: 4–8 tokens yield optimal performance.
- Number of phrases $N$: $N=4$ achieves the best balance.
- Slot Attention refinement: contributes approximately 0.5–1.0% mAP improvement.
- DQA Loss: removal leads to phrase homogenization and performance degradation.

### Key Findings

- Existing VTG models exhibit significantly lower performance in the word-only configuration compared to [EOS]-only, confirming the global semantic bias.
- DualGround successfully reduces over-reliance on [EOS]: word-only performance improves substantially.
- The phrase-level path yields the largest gains for long queries requiring fine-grained alignment.
- The sentence-level path remains effective and necessary for short queries.

## Highlights & Insights

- **Precise Problem Diagnosis**: Controlled experiments clearly expose the [EOS] bias issue; the experimental design is methodologically instructive.
- **Principled Architecture**: The necessity of the dual-path design is derived directly from the identified problem, forming a complete logical chain.
- **Phrases as Intermediate Representations**: Introducing a phrase-level granularity between words and sentences constitutes a well-motivated semantic abstraction.
- The dummy token design in ACA elegantly resolves the attention compatibility issue arising from a single-token key sequence.

## Limitations & Future Work

- The number of phrases $N$ is fixed, which limits flexibility for queries of varying length and complexity.
- Phrase clustering relies entirely on feature-space similarity, without leveraging syntactic structural information.
- Generalization to additional datasets (e.g., ActivityNet, TACoS) has not been validated.
- Inference speed may increase due to the dual-path architecture and Slot Attention.

## Related Work & Insights

- **Connection to Keyword-DETR**: Both approaches address the differential importance of text tokens, but DualGround explicitly separates global and local pathways.
- The phrase generation mechanism in LGI (Local-Global Interaction) inspired the RPG design.
- Transferring Slot Attention from object discovery to NLP phrase clustering represents a meaningful cross-domain application.

## Rating

⭐⭐⭐⭐ — Solid problem identification, clear dual-path design rationale, and state-of-the-art results in the competitive VTG field.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DualGround: Structured Phrase and Sentence-Level Temporal Grounding](dualground_phrase_temporal.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](tempsamp_r1_temporal_grounding.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)

<!-- RELATED:END -->
