---
title: >-
  [Paper Note] GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization
description: >-
  [CVPR 2026][Audio & Speech][temporal forgery localization] GEM-TFL is proposed to bridge the gap between weak and full supervision for temporal forgery localization via a two-stage classification-regression framework. Th…
tags:
  - "CVPR 2026"
  - "Audio & Speech"
  - "temporal forgery localization"
  - "weak supervision"
  - "EM algorithm"
  - "graph diffusion"
  - "temporal consistency"
date: 2026-05-08
content_hash: 8fc36b35154c3bd7
---

# GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization

**Conference**: CVPR 2026
**arXiv**: [2603.05095](https://arxiv.org/abs/2603.05095)
**Code**: None
**Area**: Speech/Audio
**Keywords**: temporal forgery localization, weak supervision, EM algorithm, graph diffusion, temporal consistency

## TL;DR

GEM-TFL is proposed to bridge the gap between weak and full supervision for temporal forgery localization via a two-stage classification-regression framework. Three core modules are introduced: EM-based decomposition of binary labels into multi-dimensional latent attributes, training-free temporal consistency refinement (TCR), and graph diffusion proposal refinement (GPR). The method achieves an average mAP improvement of 4–8% on weakly supervised temporal forgery localization benchmarks.

## Background & Motivation

Temporal Forgery Localization (TFL) aims to precisely localize manipulated segments in video/audio. Fully supervised methods require frame-level annotations, which are costly to obtain. Weakly supervised TFL (WS-TFL) relies only on video-level binary labels for training, but faces the following challenges:

1. Mismatch between training objective (classification) and inference objective (localization)
2. Insufficient supervisory signal from binary labels
3. Non-differentiability of top-k aggregation causing gradient blocking
4. Fragmentation due to independently generated proposals

## Method

### Overall Architecture

Two stages: a classification stage (MIL + LAD + TCR + GPR to generate pseudo proposals) followed by a localization stage (a regression branch trained under pseudo-proposal supervision; only the regression branch is used at inference time).

### Key Designs

#### 1. Latent Attribute Decomposition (LAD)

Binary labels are decomposed into an $(m+1)$-dimensional latent attribute set: class 0 denotes authentic content, and classes $1\ldots m$ denote $m$ learnable forgery attributes. Optimization is performed via EM:

- **E-Step**: Compute the posterior $P(c|x,y;\theta^{(t)})$ — authentic samples are assigned to class 0, while forged samples are distributed across multiple attributes according to model confidence.
- **M-Step**: Minimize $\mathcal{L}_{bin} + \lambda_1 \mathcal{L}_{nll} + \lambda_2 \mathcal{L}_{ent}$ to update parameters, with EMA updates applied to attribute priors.

#### 2. Temporal Consistency Refinement (TCR)

Addresses the non-differentiability of top-k aggregation. Frame-level attribute predictions $S_t$ are re-aligned to video-level attribute priors $q$ via a KL-based Bregman projection formulation, solved using Iterative Proportional Scaling (IPS). TCR is training-free and alternately projects onto row and column constraint spaces until convergence.

#### 3. Graph Diffusion Proposal Refinement (GPR)

An undirected graph $G=(V,E)$ is constructed with proposals as nodes; edge weights combine temporal similarity (DIoU) and semantic similarity. Confidence scores are propagated via iterative diffusion:

$$\omega^{(t+1)} = \beta \mathcal{T} \omega^{(t)} + (1-\beta) \omega^{(0)}$$

with closed-form solution $\omega^* = (1-\beta)(I - \beta\mathcal{T})^{-1}\omega^{(0)}$.

### Loss & Training

Localization stage: $\mathcal{L} = \mathcal{L}_{bce}(\hat{y},y) + \gamma \cdot \mathcal{L}_{main}(\hat{\mathcal{P}}, \mathcal{P})$, where $\gamma$ increases linearly from 0.5 to 1.0.

## Key Experimental Results

### LAV-DF Dataset

| Method | Supervision | Avg. mAP |
|--------|-------------|----------|
| UMMAFormer | Full | 96.8 |
| MFMS | Full | 97.3 |
| MDP | Weak | 60.0 |
| WMMT | Weak | 73.3 |
| **GEM-TFL** | Weak | **77.6** |

### AV-Deepfake1M Dataset

| Method | Supervision | Avg. mAP |
|--------|-------------|----------|
| GEM-TFL vs. Prev. SOTA | Weak | +8% absolute Gain |

### Key Findings

- The two-stage design effectively bridges the training–inference gap
- EM decomposition enriches the supervisory signal from binary labels
- The training-free nature of TCR avoids gradient blocking

## Highlights & Insights

1. EM decomposition of binary labels into multi-dimensional attributes elegantly extracts richer semantics from weak supervision.
2. Graph diffusion replaces the hard-coded outer-region scoring of OIC, reducing manual bias.
3. The training-free property of TCR enables inference-level refinement without additional training overhead.

## Limitations & Future Work

1. A gap of approximately 20% mAP remains compared to fully supervised methods.
2. The number of latent attributes $m$ requires manual specification.
3. Hyperparameters such as $\beta$ in graph diffusion require tuning.

## Related Work & Insights

- Compared to PseudoFormer: EM attribute decomposition and graph diffusion are added to improve pseudo-proposal quality.
- The EM latent attribute decomposition idea is transferable to other weakly supervised tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of EM decomposition, graph diffusion, and TCR is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons on two datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Framework diagrams are clear.
- **Value**: ⭐⭐⭐⭐ Bridging the weak/full supervision gap is an important research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization](../../AAAI2026/audio_speech/deformtrace_a_deformable_state_space_model_with_relay_tokens_for_temporal_forger.md)
- [\[CVPR 2026\] Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)
- [\[ICLR 2026\] LogicReward: Incentivizing LLM Reasoning via Step-Wise Logical Supervision](../../ICLR2026/audio_speech/logicreward_incentivizing_llm_reasoning_via_step-wise_logical_supervision.md)
- [\[NeurIPS 2025\] Segment-Factorized Full-Song Generation on Symbolic Piano Music](../../NeurIPS2025/audio_speech/segment-factorized_full-song_generation_on_symbolic_piano_music.md)
- [\[NeurIPS 2025\] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](../../NeurIPS2025/audio_speech/seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)

</div>

<!-- RELATED:END -->
