---
title: >-
  [Paper Note] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking
description: >-
  [CVPR 2026][Self-Supervised Learning][Procedural Activity Understanding] This paper proposes PL-Stitch, a self-supervised framework that leverages the Plackett-Luce probabilistic ranking model to use temporal ordering of…
tags:
  - "CVPR 2026"
  - "Self-Supervised Learning"
  - "Procedural Activity Understanding"
  - "Temporal Ordering"
  - "Plackett-Luce Model"
  - "Surgical Video"
date: 2026-05-08
content_hash: ae5202141c5d8f05
---

# A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking

**Conference**: CVPR 2026
**arXiv**: [2511.17805](https://arxiv.org/abs/2511.17805)  
**Code**: [https://github.com/visurg-ai/PL-Stitch](https://github.com/visurg-ai/PL-Stitch)  
**Area**: Video Understanding / Self-Supervised Learning
**Keywords**: Procedural Activity Understanding, Temporal Ordering, Plackett-Luce Model, Self-Supervised Learning, Surgical Video

## TL;DR

This paper proposes PL-Stitch, a self-supervised framework that leverages the Plackett-Luce probabilistic ranking model to use temporal ordering of video frames as a pretraining signal. The method learns "procedure-aware" video representations and consistently outperforms existing self-supervised approaches on surgical phase recognition and cooking action segmentation.

## Background & Motivation

Procedural activities (e.g., cooking, surgery) consist of a series of steps executed in strict temporal order. Understanding this temporal structure is critical for downstream tasks such as robot-assisted surgery and action prediction. Mainstream self-supervised methods (e.g., DINO, MAE, VideoMAE) primarily learn features via instance discrimination or masked reconstruction, yet are fundamentally "procedure-agnostic"—they learn *what* appears in a frame but not *when* it appears.

The authors design a key diagnostic experiment: SSL models are pretrained on forward- and reverse-order videos separately, and the resulting frame-level features are found to be nearly identical (extremely low cosine distance). This directly demonstrates that existing methods are entirely insensitive to temporal direction—they cannot distinguish a surgical workflow played forward from one played in reverse.

Existing attempts to exploit temporal information also suffer from drawbacks: (1) pairwise comparison methods require $\mathcal{O}(k^2)$ comparisons, yielding fragmented and inefficient signals; (2) permutation classification misframes a relative ordering problem as an absolute classification task, penalizing a nearly correct ordering (e.g., two frames swapped) as harshly as a completely wrong one.

**Core insight**: Temporal ordering is inherently a listwise ranking problem and should be modeled with a probabilistic ranking model, so that penalties are proportional to the degree of error. The Plackett-Luce model naturally satisfies this requirement—it defines a probability distribution over all possible permutations and assigns higher probability to near-correct orderings.

## Method

### Overall Architecture

PL-Stitch consists of a shared ViT backbone encoder $f_\theta$ and two complementary branches. The **Video Branch** sparsely samples $k=8$ frames from a video and trains the model to predict their correct temporal order via a PL ranking loss, learning global workflow progression. The **Image Branch** jointly optimizes masked image modeling (MIM) and spatio-temporal jigsaw objectives on triplets of frames (past/present/future), learning fine-grained local features and cross-frame correspondences.

### Key Designs

1. **Plackett-Luce-Based Listwise Temporal Ranking (Video Branch)**

    - **Function**: Learn global procedural progression representations.
    - **Mechanism**: $k=8$ frames are uniformly sampled from a video to form a clip $C_v$. Each frame is encoded to a [CLS] token, which is fed into a temporal head $h_{vid}$ (MLP → Transformer Encoder → MLP) that outputs PL distribution parameters $s_{clip} = (s_1, \ldots, s_k)$. The loss is the PL negative log-likelihood $\mathcal{L}_{vid} = -\log P(r^*|s)$, where $P(r|s) = \prod_{i=1}^{K} \frac{\exp(s_{r(i)})}{\sum_{j=i}^{K} \exp(s_{r(j)})}$. The model learns to assign higher ranking scores to earlier frames.
    - **Design Motivation**: Compared to pairwise comparison (+1.3pp linear, +3.5pp kNN) and permutation classification (+2.6pp linear, +8.8pp kNN), the listwise PL optimization provides a globally consistent ranking signal. Its probabilistic nature allows near-correct orderings to receive higher probability, making it more robust than hard classification.

2. **Spatio-Temporal Jigsaw Objective (Image Branch)**

    - **Function**: Learn fine-grained cross-frame object correspondences.
    - **Mechanism**: Given a frame triplet (past/present/future), the patch features of the masked center frame serve as Query, while patch features from the surrounding frames serve as Key/Value (with positional encodings removed to force reliance on visual content rather than positional shortcuts). Cross-attention aggregates temporal context, self-attention models spatial relationships, and the output PL parameters predict the original spatial arrangement $r^*_{jigsaw} = (1, 2, \ldots, N)$ of the patches.
    - **Design Motivation**: Standard jigsaw tasks rely on single-frame information and cannot capture cross-frame motion correspondences. This design exploits cues from temporally adjacent frames (e.g., positional shifts of the same instrument across frames) to reason about masked spatial locations. Ablations show that adding the jigsaw objective improves kNN accuracy from 78.9% to 80.2%.

3. **Masked Image Modeling (MIM Baseline Objective)**

    - **Function**: Establish a robust frame-level semantic feature foundation.
    - **Mechanism**: Adopts iBOT-style masked image modeling with 30% block masking and reconstruction on the current frame.
    - **Design Motivation**: MIM provides dense pixel/semantic-level feature learning and serves as a complementary signal to temporal ranking and jigsaw objectives. Ablations show MIM is an indispensable foundational component—removing it causes a significant performance drop.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{vid} + \lambda_2 \mathcal{L}_{MIM} + \lambda_3 \mathcal{L}_{jigsaw}$, where $\lambda_1=1, \lambda_2=1, \lambda_3=0.4$. A ViT-B/16 backbone and AdamW optimizer are used with a base learning rate of $4 \times 10^{-4}$. For surgical data, pretraining is conducted on the LEMON dataset for 30 epochs; for cooking data, pretraining is conducted on the respective training sets for 100 epochs. Training uses 4 A100 GPUs.

## Key Experimental Results

### Main Results

**Surgical Phase Recognition (Linear Probing / kNN)**

| Dataset | Method | Linear Acc | kNN Acc | Prev. SOTA | Gain |
|--------|------|-----------|---------|---------|------|
| Cholec80 | PL-Stitch | **80.4** | **81.7** | 74.6 / 70.3 (iBOT) | +5.8 / +11.4 |
| AutoLaparo | PL-Stitch | **79.9** | **82.5** | 76.3 / 75.3 (iBOT) | +3.6 / +7.2 |
| M2CAI16 | PL-Stitch | **76.4** | **77.1** | 71.0 / 68.0 (iBOT) | +5.4 / +9.1 |

**Cooking Action Segmentation (Linear Probing Acc / kNN Acc)**

| Dataset | PL-Stitch | Prev. SOTA | Gain |
|--------|-----------|---------|------|
| GTEA | **54.1** / **62.4** | 52.2 / 60.0 (DINO) | +1.9 / +2.4 |
| Breakfast | **21.6** / **10.9** | 15.9 / 7.5 (DINO/iBOT) | +5.7 / +3.4 |

### Ablation Study

| Configuration | Linear Acc | kNN Acc | Note |
|------|-----------|---------|------|
| $\mathcal{L}_{MIM}$ only | 73.4 | 69.4 | iBOT baseline |
| $\mathcal{L}_{MIM}$ + $\mathcal{L}_{vid}$ | 77.1 | 78.9 | +9.5pp kNN; temporal ranking contributes most |
| $\mathcal{L}_{MIM}$ + $\mathcal{L}_{jigsaw}$ | 75.3 | 71.4 | Jigsaw provides auxiliary benefit |
| **Full PL-Stitch** | **77.8** | **80.2** | All three objectives are complementary |

**Comparison of Temporal Objectives (combined with MIM)**

| Objective Type | Linear | kNN | Note |
|---------|--------|-----|------|
| Pairwise | 75.8 | 75.4 | $\mathcal{O}(k^2)$ local signal |
| Permutation CE | 74.5 | 70.1 | Hard classification ill-suited for ranking |
| **PL Ranking** | **77.1** | **78.9** | Probabilistic listwise ranking is optimal |

### Key Findings

- The temporal ranking objective $\mathcal{L}_{vid}$ is the primary driver of performance gains (+9.5pp kNN), far exceeding the jigsaw objective's +2.0pp contribution.
- PL ranking significantly outperforms both pairwise and permutation classification approaches, validating the theoretical advantages of probabilistic listwise ranking.
- $k=8$ frames represents the optimal trade-off between computational efficiency and performance ($k=16$ yields only +0.1pp but quadruples computation).
- t-SNE visualizations show that PL-Stitch's feature space exhibits clear phase separation (ARI 0.35 vs. best baseline 0.10).
- Attention maps show that PL-Stitch consistently tracks surgical instruments, whereas baseline attention is scattered and unstable.

## Highlights & Insights

- **The forward/reverse invariance experiment is elegantly designed**: A simple yet powerful experiment directly exposes the blind spot of existing SSL methods, and is more convincing than downstream performance comparisons alone.
- **PL model as an SSL pretext task**: Introducing listwise learning-to-rank from information retrieval into visual self-supervision opens up a new design space for pretext tasks. The probabilistic nature of PL gently penalizes near-correct orderings, avoiding the excessive harshness of classification losses.
- **Cross-attention design in spatio-temporal jigsaw**: Removing positional encodings forces the model to reason about spatial positions via visual content rather than positional shortcuts; combining this with temporal neighbor frames as motion cues is a clever refinement of the classic jigsaw task.

## Limitations & Future Work

- Only frame-level encoders (ViT-B/16) are used; video-specific architectures with spatio-temporal attention are not exploited.
- Pretraining relies on large-scale in-domain data (LEMON for surgery); transferability to domains lacking such data remains to be verified.
- Evaluation is limited to representation quality (linear probe/kNN); full fine-tuning and downstream generative tasks (e.g., action prediction) are not assessed.
- The PL model assumes independent item selection (Luce's axiom), which may not fully apply to action sequences with strong causal dependencies.

## Related Work & Insights

- **vs. iBOT**: Both employ self-distillation + MIM, but iBOT lacks temporal signals; PL-Stitch achieves +11.4pp on Cholec80 kNN.
- **vs. T-CoRe**: Also leverages temporally adjacent frames for cross-frame reconstruction, but lacks global temporal structure learning and underperforms even the standalone MIM baseline.
- **vs. VideoMAEv2**: Symmetric treatment of spatial and temporal dimensions in spatio-temporal masked reconstruction leads to temporal blindness, making it one of the weakest baselines.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First introduction of the PL ranking model into visual SSL; the motivating experiment is highly convincing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks, two evaluation protocols, extensive ablations, and qualitative analysis (t-SNE / attention maps / prediction visualizations).
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative is coherent, with a clear logical chain from motivating experiment → problem formulation → method → experiments.
- Value: ⭐⭐⭐⭐ Provides a new self-supervised paradigm for procedural video understanding, though applicability is currently limited to domains with strong temporal structure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](../../ICML2026/self_supervised/understanding_self-supervised_learning_via_latent_distribution_matching.md)

</div>

<!-- RELATED:END -->
