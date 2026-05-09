---
title: >-
  [Paper Note] Real-World Point Tracking with Verifier-Guided Pseudo-Labeling
description: >-
  [CVPR 2026][Video Understanding][point tracking] This paper proposes a learnable Verifier meta-model trained on synthetic data to assess the reliability of tracker predictions and transfer this capability to the real world. By evaluating per-frame predictions from six pretrained trackers and selecting the most reliable as pseudo-labels, the proposed Track-On-R model is fine-tuned on only ~5K real videos and achieves comprehensive state-of-the-art performance across four real-world benchmarks.
tags:
  - CVPR 2026
  - Video Understanding
  - point tracking
  - pseudo-labeling
  - verifier
  - multi-teacher ensemble
  - sim-to-real adaptation
date: 2026-05-08
content_hash: 20687e8c6c6159c3
---

# Real-World Point Tracking with Verifier-Guided Pseudo-Labeling

**Conference**: CVPR 2026
**arXiv**: [2603.12217](https://arxiv.org/abs/2603.12217)
**Code**: [kuis-ai.github.io/track_on_r](https://kuis-ai.github.io/track_on_r)
**Area**: Video Understanding / Point Tracking
**Keywords**: point tracking, pseudo-labeling, verifier, multi-teacher ensemble, sim-to-real adaptation

## TL;DR

This paper proposes a learnable Verifier meta-model trained on synthetic data to assess the reliability of tracker predictions and transfer this capability to the real world. By evaluating per-frame predictions from six pretrained trackers and selecting the most reliable as pseudo-labels, the proposed Track-On-R model is fine-tuned on only ~5K real videos and achieves comprehensive state-of-the-art performance across four real-world benchmarks.

## Background & Motivation

**Background**: Long-range point tracking models (CoTracker, TAPIR, Track-On, etc.) are typically trained on synthetic data (TAP-Vid Kubric). Recent self-training methods (BootsTAPIR, CoTracker3) fine-tune on real videos using pseudo-labels to bridge the sim-to-real gap.

**Limitations of Prior Work**:

1. Individual trackers exhibit highly variable reliability across different frames and scenes — each excels under different challenges such as fast motion, low texture, occlusion, and identity switching.
2. Naive self-training (randomly selecting a single teacher for pseudo-labels) or fixed fusion strategies cannot handle this heterogeneity and propagate systematic errors.
3. Methods such as BootsTAPIR require millions of real videos for large-scale distillation, resulting in low data efficiency.

**Key Challenge**: Oracle experiments show that selecting the best tracker per frame could yield substantial performance gains (with a large gap over existing methods), yet no automated method achieves such per-frame adaptive selection.

**Goal**: Learn to automatically assess the per-frame reliability of multiple trackers and select the most accurate predictions as pseudo-labels.

**Key Insight**: Train a "Verifier" meta-model — one that does not perform tracking itself, but learns to judge "who tracks well." Reliability estimation is learned on synthetic data via a "corrupt-then-detect" paradigm (applying random perturbations to GT trajectories to simulate tracker errors).

**Core Idea**: Rather than tracking points, the Verifier scores tracker predictions, converting multi-model complementarity into high-quality pseudo-labels.

## Method

### Overall Architecture

Given a video and query points, six pretrained teacher trackers each produce a candidate trajectory $\mathbf{C} \in \mathbb{R}^{L \times M \times 2}$. The Verifier evaluates per-frame reliability scores $\hat{\mathbf{s}}_t \in \mathbb{R}^M$ over these candidates and selects the highest-scoring candidate as the pseudo-label for each frame. Frame-level optimal predictions are concatenated into a complete pseudo-label trajectory used to fine-tune the student model Track-On2. At inference time, the Verifier can also serve as a plug-and-play ensemble module.

### Key Designs

1. **Verifier Training: "Corrupt-then-Detect" on Synthetic Data**

    - Trained on the K-EPIC synthetic dataset (11K videos, 24 frames/video).
    - Six types of random perturbations (drift, jump, occlusion, identity switching, etc., with displacements of 1–128 pixels) are applied to GT trajectories to generate candidate trajectories $\mathbf{C}$.
    - Training objective: candidates closer to the GT should receive higher scores — implemented via soft contrastive learning with target distribution $\mathbf{s}_t = \text{Softmax}(-\|\mathbf{C}_t - \mathbf{p}_t\| / \tau_s)$, $\tau_s = 0.1$.
    - Loss: cross-entropy $\mathcal{L} = \sum_t v_t \cdot \text{CE}(\hat{\mathbf{s}}_t, \mathbf{s}_t)$, with occluded frames masked.
    - Requires zero real-world annotations; the learned reliability estimation transfers across domains.

2. **Localized Feature Extraction + Candidate Transformer**

    - Uses the frozen CNN encoder from CoTracker3 to extract frame-level dense features $\mathbf{F}_t \in \mathbb{R}^{H' \times W' \times D}$.
    - Local features are extracted at query and candidate positions via deformable attention, rather than global reasoning.
    - Positional encoding: sinusoidal encoding of candidate displacements relative to the query point $\boldsymbol{\Delta}_t = \mathbf{C}_t - \mathbf{q}_{t_0}$, augmented with learnable identity encodings to distinguish queries from candidates.
    - **Candidate Transformer**: constrained cross-attention (per-frame query attends only to $M$ candidates in the current frame) + temporal self-attention (propagating context across frames) → outputs a temperature-scaled softmax reliability distribution over $M$ candidates per frame.
    - Key formulation: $\hat{\mathbf{s}}_t = \text{Softmax}(\mathbf{f}_t^q \cdot \mathbf{f}_t / \tau)$, $\tau = 0.1$.

3. **Verifier-Guided Real-World Fine-Tuning**

    - Six teacher models: Track-On2, BootsTAPIR, BootsTAPNext, Anthro-LocoTrack, AllTracker, CoTracker3 (window).
    - Real video sources: TAO + OVIS + VSPW (videos with >48 frames, totaling 4,864 clips, requiring no annotations).
    - Query point sampling: 2/3 from SIFT detections, 1/3 from motion-salient regions.
    - Visibility estimation: majority voting among teacher models.
    - Training strategy: mixed training on synthetic data (with GT) and real data (with Verifier pseudo-labels), with the weight of real data progressively increased.

### Loss & Training

- Verifier training: cross-entropy on soft reliability targets, with occluded frames masked.
- Student model fine-tuning: standard point tracking loss (position L1 + visibility BCE), with mixed synthetic and real data and progressively increasing loss weight on real data.
- Base student model: Track-On2, pretrained on TAP-Vid Kubric.

## Key Experimental Results

### Main Results

Comparison on real-world point tracking benchmarks:

| Model | EgoPoints δ_avg | RoboTAP AJ | Kinetics AJ | DAVIS AJ | Type |
|------|-----------------|------------|-------------|----------|------|
| Track-On2 | 61.7 | 68.1 | 55.3 | 67.0 | Synthetic pretrain |
| BootsTAPIR | 55.7 | 64.9 | 54.6 | 61.4 | Real fine-tune (M-scale) |
| BootsTAPNext-B | 33.6 | 64.0 | 57.3 | 65.2 | Real fine-tune (M-scale) |
| CoTracker3 | 54.0 | 66.4 | 55.8 | 63.8 | Real fine-tune |
| AllTracker† | 62.0 | 68.8 | 56.8 | 63.7 | Extra optical flow data |
| **Track-On-R (Ours)** | **67.3** | **70.9** | **57.8** | **68.1** | Real fine-tune (~5K) |

### Ablation Study

**Verifier ensemble vs. individual teachers (δ_avg / AJ on DAVIS & RoboTAP)**:

| Method | DAVIS δ_avg | RoboTAP δ_avg |
|------|-------------|---------------|
| Random teacher selection | 79.5 | 77.4 |
| Best single teacher | ~79–80 | ~80 |
| **Verifier selection** | **80.6** | **81.8** |

**Effect of number of teachers**: Increasing the number of teachers monotonically improves Verifier performance — even adding weak teachers does not degrade the Verifier (though it degrades the random-selection baseline).

**Data efficiency**: The majority of adaptation gains are achieved with only ~3K videos (TAO subset), far fewer than the millions required by BootsTAPIR.

**Verifier vs. non-learning ensemble strategies**: The Verifier surpasses all non-learning baselines including geometric median, consistency voting, and Kalman filtering.

### Key Findings

- Track-On-R achieves comprehensive state-of-the-art results on all four benchmarks, with EgoPoints δ_avg of 67.3 surpassing AllTracker by 5.3 points.
- With only ~5K real videos, Track-On-R outperforms BootsTAPIR/BootsTAPNext trained on millions of videos, representing >100× improvement in data efficiency.
- Real-world fine-tuning does not degrade performance on synthetic benchmarks; δ_avg on PointOdyssey even improves by +8.3.
- Teacher rankings vary substantially across datasets (BootsTAPNext ranks lowest on RoboTAP but second on DAVIS), validating the necessity of adaptive per-frame selection.

## Highlights & Insights

- The Verifier meta-model design is elegant — rather than performing tracking, it learns "who tracks well," converting multi-tracker complementarity into high-quality pseudo-labels.
- The oracle gap analysis (Fig. 2) compellingly demonstrates the large potential of adaptive selection.
- The trajectory perturbation scheme (six perturbation types) is carefully designed to cover diverse real-world tracker failure modes, and is conducted entirely on synthetic data.
- Extremely high data efficiency (~3K videos near optimal, 100× fewer than BootsTAPIR) is a key practical advantage.
- The Verifier can also be used directly as a plug-and-play ensemble module at inference time, providing gains without fine-tuning.

## Limitations & Future Work

- The Verifier's upper bound is constrained by the teacher tracker pool — if all teachers fail on a given frame, the Verifier cannot recover.
- Using six teacher trackers at inference introduces substantial computational overhead (6× inference cost).
- The approach is only validated on point tracking; whether the Verifier paradigm generalizes to optical flow, VOS, and related tasks remains to be explored.
- Fine-tuning effectiveness depends on the quality and diversity of real video data; applicability to extreme out-of-domain scenarios (e.g., underwater, thermal imaging) is unknown.
- The generalization of the Verifier itself — trained on K-EPIC synthetic data — may be limited in real-world scenes with larger domain gaps.

## Related Work & Insights

- **vs. CoTracker3**: Both adopt teacher pseudo-label strategies, but CoTracker3 randomly selects teachers (Kinetics AJ 55.8 vs. 57.8); the core gap lies in pseudo-label quality.
- **vs. BootsTAPIR/BootsTAPNext**: Large-scale student-teacher distillation requires millions of videos; this work exceeds their performance with only ~5K, representing a massive difference in data efficiency.
- **vs. AllTracker**: AllTracker leverages additional real optical flow annotation data (EgoPoints 62.0 vs. 67.3), demonstrating that Verifier pseudo-labels can even outperform supervision from real optical flow annotations.
- Broader implication: The Verifier paradigm is transferable to any scenario requiring multi-model fusion or pseudo-label selection — e.g., per-sample selection of the most reliable teacher in multi-teacher distillation for object detection or semantic segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The Verifier meta-model elevates reliability estimation from heuristics to a learnable paradigm — a novel and elegant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 4 real-world and 2 synthetic benchmarks, with comprehensive ablations over teacher combinations, data scale, non-learning baselines, and oracle comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ The oracle gap analysis is highly convincing; method descriptions are clear; Figs. 1/2/3 are excellent.
- Value: ⭐⭐⭐⭐⭐ The Verifier paradigm is broadly applicable, highly data-efficient, and offers important insights for point tracking and the wider self-training/pseudo-labeling community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VideoSeek: Long-Horizon Video Agent with Tool-Guided Seeking](videoseek_long-horizon_video_agent_with_tool-guided_seeking.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](cinesrd_leveraging_visual_acoustic_and_linguistic_cues_for_open-world_visual_med.md)
- [\[CVPR 2026\] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](rethinking_twostage_referringbytracking_in_referri.md)
- [\[CVPR 2026\] SVAgent: Storyline-Guided Long Video Understanding via Cross-Modal Multi-Agent Collaboration](svagent_storyline_guided_long_video_understanding_via_cross_modal_multi_agent_collaboration.md)

<!-- RELATED:END -->
