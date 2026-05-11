---
title: >-
  [Paper Note] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis
description: >-
  > SemTalk decomposes co-speech motion into rhythm-aligned base motions and semantics-aware sparse motions, and adaptively fuses them via learned semantic scores to achieve high-quality holistic co-speech motion generati…
tags:

date: 2026-05-08
content_hash: d98c6dec506ee774
---

# SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis

- **Conference**: ICCV 2025
- **arXiv**: 2412.16563
- **Code**: https://xiangyue-zhang.github.io/SemTalk
- **Area**: Other
- **Keywords**: co-speech motion generation, semantic emphasis, rhythm consistency, RVQ-VAE, semantic gating

## TL;DR

> SemTalk decomposes co-speech motion into rhythm-aligned base motions and semantics-aware sparse motions, and adaptively fuses them via learned semantic scores to achieve high-quality holistic co-speech motion generation with frame-level semantic emphasis.

## Background & Motivation

Holistic co-speech motion generation requires synthesizing gestures, facial expressions, and body movements that align with speech. The core challenge lies in balancing common rhythmic motions and rare but important semantic motions.

Through analysis of the semantic annotations in the BEAT2 dataset, the authors identify a key observation: **semantically relevant motions are temporally sparse** — most frames contain rhythm-driven gestures, while only a small number of key frames carry gestures that convey specific semantics. Existing methods suffer from the following issues:

**Dominance of rhythmic features**: Most methods (TalkSHOW, EMAGE, etc.) rely on rhythm-related audio features, causing semantic motions to be overwhelmed by rhythmic signals.

**Lack of frame-level precision in semantic guidance**: Methods such as LivelySpeaker employ global CLIP-based semantic control, which cannot precisely emphasize key semantic moments at the frame level.

**Coupled modeling of rhythm and semantics**: Modeling two fundamentally different motion patterns jointly makes it difficult to optimize both simultaneously.

These observations motivate the core design of SemTalk: **separate modeling with adaptive fusion**.

## Method

### Overall Architecture

SemTalk consists of three core stages:

1. **Base Motion Blocks** $f_r(\cdot)$: generate rhythm-aligned base motion codes $q^b$
2. **Sparse Motion Blocks** $f_s(\cdot)$: generate frame-level semantic codes $q^s$ and semantic scores $\psi$
3. **Adaptive Fusion** $\mathcal{E}$: adaptively fuse $q^b$ and $q^s$ based on $\psi$ to obtain the final code $q^m$

Formally:

$$q^m = \mathcal{E}(q^b, q^s; \psi)$$

### RVQ-VAE Pre-training

The body is divided into four parts — face, upper body, hands, and lower body — each equipped with an independent RVQ-VAE to prevent feature entanglement and preserve the unique dynamics of each part.

### Base Motion Generation

**Rhythmic speech encoding**: Two types of rhythmic features are extracted — beat features $\gamma_b$ derived from amplitude/short-time energy, and HuBERT features $\gamma_h$ — combined with seed pose $\tilde{m}$ and speaker identity $id$.

**Coarse2Fine cross-attention module**: Base motions are hierarchically refined following a cascaded propagation order: face → hands → upper body → lower body. Lip movements correspond closely to speech phonemes and thus guide hand gestures; natural hand swinging influences the upper body, which in turn drives the lower body.

**Rhythm consistency learning**: An InfoNCE loss is used for motion–rhythm alignment:

$$\mathcal{L}_{\text{Rhy}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(h(f_i), \gamma_h^i) / \tau)}{\sum_{j=1}^{N}\exp(\text{sim}(h(f_i), \gamma_h^j) / \tau)}$$

This is decomposed into a local frame-level term $\mathcal{L}_{\text{Rhy}}^{(L)}$ and a global sequence-level term $\mathcal{L}_{\text{Rhy}}^{(G)}$, ensuring precise frame-level alignment and smooth sequence-level rhythmic flow, respectively.

### Semantics-Aware Sparse Motion Generation

**Semantic speech encoding**: Three sources of semantic information are fused — frame-level text embeddings $\phi_l$, CLIP sentence-level features $\phi_g$, and emotion2vec emotion features $\phi_e$.

**Semantic Gating (Sem-gate)**: The core innovation. Multimodal inputs are used to compute frame-level semantic scores $\psi$, which enhance semantic frames via two weighting strategies:

1. **Feature weighting** $\mathcal{W}_f$: scales semantic features $f_t$ by $\psi$ to activate semantically salient frames
2. **Loss weighting** $\mathcal{W}_l$: supervises $\psi$ with classification loss $\mathcal{L}_{cls}^G$ to reinforce key frames based on semantic annotations

Sparse motion codes are blended via alpha-blending:

$$q^s = MLP(\psi f_s + (1-\psi) f_b)$$

### Semantic Score Fusion

For each frame $i$, if the semantic score $\psi_i > \beta$ (threshold), the sparse semantic code $q_i^s$ replaces the base motion code $q_i^r$; otherwise the base code is retained. The convolutional structure of the RVQ-VAE decoder naturally ensures smooth transitions between frames.

## Experiments

### Main Results (BEAT2 Dataset)

| Method | FGD ↓ | BC ↑ | DIV ↑ | MSE ↓ | LVD ↓ |
|--------|-------|------|-------|-------|-------|
| TalkSHOW | 6.209 | 6.947 | 13.47 | 7.791 | 7.771 |
| EMAGE | 5.512 | 7.724 | 13.06 | 7.680 | 7.556 |
| DiffSHEG | 8.986 | 7.142 | 11.91 | 7.665 | 8.673 |
| **SemTalk** | **4.278** | **7.770** | 12.91 | **6.153** | **6.938** |

SemTalk achieves state-of-the-art performance on FGD (−22.4%), MSE (−19.9%), and LVD (−8.2%).

### Ablation Study on Sem-gate

| Configuration | FGD ↓ | BC ↑ | DIV ↑ | Acc(%) ↑ |
|---------------|-------|------|-------|----------|
| w/o Sem-gate | 4.893 | 7.702 | 12.42 | - |
| SAG (LivelySpeaker) | 4.618 | 7.682 | 12.45 | - |
| Random $\psi$ | 4.634 | 7.700 | 12.44 | 50.07 |
| Sem-gate (w/o $\mathcal{W}$) | 4.495 | 7.633 | 12.26 | 72.32 |
| Sem-gate (w/ $\mathcal{W}_f$) | 4.408 | 7.679 | 12.28 | 78.52 |
| Sem-gate (w/ $\mathcal{W}_l$) | 4.366 | 7.772 | 11.94 | 77.83 |
| **Sem-gate (Full)** | **4.278** | **7.770** | **12.91** | **82.76** |

### Key Findings

1. **Effectiveness of disentangled modeling**: Compared to SemTalk* (Base Motion only), the full SemTalk generates more expressive semantic gestures (e.g., raising the hand with an extended index finger when saying "my opinion").
2. **Sem-gate outperforms SAG**: LivelySpeaker's SAG relies solely on text–motion alignment and lacks emotional information, making it prone to overfitting text; Sem-gate achieves higher accuracy via ground-truth supervision and dual weighting.
3. **Same text, different emotions**: SemTalk generates different gestures for identical text spoken with different emotional prosody, demonstrating that the model does not overfit to text.
4. **User study**: Among 25 participants, SemTalk received the highest preference ratings across four dimensions: naturalness, semantic consistency, motion–speech synchrony, and diversity.

## Highlights & Insights

- The decomposition of "base motion + sparse motion" is elegant and cognitively intuitive — speakers indeed produce semantically meaningful gestures only at a small number of key moments.
- The cascaded order of the Coarse2Fine cross-attention (face → hands → upper body → lower body) is grounded in physiological intuition and constitutes a well-motivated design choice.
- The learned semantic scores $\psi$ align closely with the distribution of semantically significant words (e.g., peaks at "comes," "fantastic," and "captured"), validating the model's semantic understanding capability.
- The framework demonstrates strong generalizability: on the SHOW dataset, which lacks semantic annotations, Sem-gate pre-trained on BEAT2 can be transferred to generate semantic scores.

## Limitations & Future Work

- The quality of semantic scores depends on frame-level semantic annotations from the BEAT2 dataset, which may be subject to annotator subjectivity.
- The threshold $\beta$ is set empirically at 0.5 and may require adjustment for different data distributions.
- Lower-body motion is primarily driven by cascading from the upper body, with no independent lower-body modeling.
- The overall computational cost is substantial: separate RVQ-VAEs, dual-branch base/sparse generation, and fusion introduce significant overhead.

## Related Work & Insights

- **Co-speech gesture generation**: TalkSHOW (VQ-VAE cross-conditioning), EMAGE (multi-encoder), DiffSHEG (diffusion model)
- **Semantic enhancement**: LivelySpeaker (CLIP + diffusion), HA2G (hierarchical network), DisCo (content–rhythm disentanglement)
- **Holistic motion generation**: ProbTalk (PQ-VAE), TM2D (decomposing dance motion into music-related and music-independent components)

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐⭐ |
| Clarity | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |
| Overall | 8.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement](../../AAAI2026/others/mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)
- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](../../CVPR2026/others/next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](../../AAAI2026/others/leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)

</div>

<!-- RELATED:END -->
