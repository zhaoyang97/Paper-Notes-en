---
title: >-
  [Paper Note] RELO: Reinforcement Learning to Localize for Visual Object Tracking
description: >-
  [ICML 2026][Video Understanding][Visual Tracking] RELO reframes the "where is the target" problem in visual single-object tracking as an MDP over a spatial feature map…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Visual Tracking"
  - "RL Localization"
  - "MDP"
  - "AUC Reward"
  - "Temporal Token Propagation"
date: 2026-05-08
content_hash: 1f4b04d2bcd41735
---

# RELO: Reinforcement Learning to Localize for Visual Object Tracking

**Conference**: ICML 2026  
**arXiv**: [2605.07379](https://arxiv.org/abs/2605.07379)  
**Code**: https://github.com/Multimedia-Analytics-Laboratory/RELO (available)  
**Area**: Video Understanding / Visual Object Tracking / Reinforcement Learning  
**Keywords**: Visual Tracking, RL Localization, MDP, AUC Reward, Temporal Token Propagation

## TL;DR
RELO reframes the "where is the target" problem in visual single-object tracking as an MDP over a spatial feature map, treating each spatial location as an action. It replaces traditional handcrafted center heatmap supervision with actor-critic and direct IoU/AUC rewards, and introduces two stabilization designs—"warmup regression" and "layer-aligned temporal token propagation." On LaSOText, it achieves SOTA with 57.5% AUC.

## Background & Motivation

**Background**: Modern single-object tracking is dominated by the one-stream Transformer + center heatmap classification + regression branch paradigm (OSTrack, ODTrack, ARTrack, SUTrack, etc.). Typically, a Gaussian-smoothed center heatmap or binary mask is used as classification supervision to indicate "where high response should be," while the regression branch outputs the box.

**Limitations of Prior Work**: This "prior-driven localization" paradigm has two fundamental issues—**(1) Misalignment between supervision and evaluation metrics**: During training, the model is forced to fit an artificially designed center distribution, but evaluation only cares about IoU and AUC; the model cannot directly know "what is the final IoU if this location is chosen." **(2) Introduction of non-essential handcrafted assumptions**: The variance of the Gaussian, threshold of the binary mask, and boundary of corner expectation are all heuristics, sensitive across datasets and not end-to-end learnable.

**Key Challenge**: The essence of tracking is "select a spatial location → its corresponding box has the highest IoU with GT," but current pipelines forcibly insert a "center classification" proxy task, whose optimal solution does not coincide with that of the real task—especially in cross-category, long-term tracking scenarios like LaSOText, where the center prior is unreliable under appearance drift.

**Goal**: Eliminate all handcrafted spatial priors, enabling the model to learn "which location to select" directly from the ultimate feedback of "how good is the tracking result," while ensuring stability—since RL training from scratch is notoriously unstable.

**Key Insight**: The authors observe that the search feature map $\mathbf{F}^{(t)} \in \mathbb{R}^{H\times W \times C}$ of a one-stream tracker naturally forms a discrete action space of size $H\times W$, with each position $(i,j)$ having a candidate box predicted by the regression head—this is a standard discrete-action MDP, and IoU/AUC are non-differentiable but numerically computable reward signals, making policy gradient methods directly applicable.

**Core Idea**: Learn "where to localize" via RL, without any center/corner prior; use a two-stage "warmup regression then freeze, then learn policy" for stability, and provide inter-frame context via layer-aligned temporal token propagation.

## Method

### Overall Architecture
Given a template frame $\mathbf{I}_{\text{temp}}$ and a search frame $\mathbf{I}^{(t)}$, a one-stream Transformer encoder (HiViT-T/B/L, Fast-iTPN pre-trained) jointly processes both to output $\mathbf{F}^{(t)} \in \mathbb{R}^{H\times W\times C}$ as the state. Three parallel heads: the regression head $g_{\text{reg}}$ predicts 4D box coordinates $\mathbf{B}^{(t)} \in \mathbb{R}^{H\times W\times 4}$ at each location; the policy head $g_{\text{policy}}$ outputs $H\times W$ logits forming a softmax distribution $\pi(a_{ij}|\mathbf{F}^{(t)})$; the value head $g_{\text{value}}$ provides a scalar reward estimate $v^{(t)}$. Training is two-stage: **warmup** phase uses GIoU+L1 supervision only at the GT center to teach the regression head the "local feature→box" mapping, then freezes it; **RL phase** samples actions, computes rewards, and updates policy/value via actor-critic on $T=8$ frame video clips. During inference, the argmax policy location is selected frame-by-frame to output the box, with no test-time adaptation.

### Key Designs

1. **Spatial Location = Action MDP Formalization + Actor-Critic Learning**:

    - **Function**: Explicitly models the core tracking decision of "which location to select" as a policy, directly optimized by IoU/AUC, bypassing the center heatmap proxy task.
    - **Mechanism**: Action space $\mathcal{A} = \{(i,j) | i \in \{1,...,H\}, j \in \{1,...,W\}\}$; policy is a categorical distribution over logits; reward is $r^{(t)} = \text{IoU}(\boldsymbol{b}^{(t)}_{a^{(t)}}, \boldsymbol{b}^{(t)}_{\text{gt}}) + \lambda \cdot \text{AUC}(\{\boldsymbol{b}^{(\tau)}_{a^{(\tau)}}, \boldsymbol{b}^{(\tau)}_{\text{gt}}\}_{\tau=1}^T)$, providing both immediate per-frame and global trajectory rewards (default $\lambda=1$); value head provides baseline, advantage $A^{(t)} = r^{(t)} - v^{(t)}$; loss is REINFORCE form $\ell_{\text{policy}} = -\frac{1}{T}\sum_t A^{(t)} \log \pi(a^{(t)}|\mathbf{F}^{(t)})$ plus value MSE, total loss $\ell = \ell_{\text{policy}} + \beta \ell_{\text{value}}$ ($\beta=0.5$). Crucially, the reward need not be differentiable and can directly use evaluation metrics like IoU, AUC, ensuring strict train-test alignment.
    - **Design Motivation**: The core issue with center heatmap supervision is "objective ≠ evaluation metric"; RL directly aligns with the evaluation metric. The discrete action space ($H\times W = 16\times 16 = 256$) is manageable for REINFORCE + actor-critic, without needing heavy frameworks like PPO.

2. **Regression Warmup + Freeze → Stable RL Training**:

    - **Function**: Addresses the instability of RL from scratch over 256 actions.
    - **Mechanism**: The authors found that joint training of regression + policy from random initialization leads to poor box quality for explored actions, resulting in noisy reward signals and failed learning. Thus, a warmup is performed: only the GT center spatial location is supervised with GIoU + L1 for the regression head, teaching the model "given the correct location, how to decode an accurate box from its feature"—essentially teaching box decoding, not localization. After warmup, the encoder and regression head are frozen; only the policy and value heads are updated during RL. This ensures that any action sampled by the policy yields a reasonable box, providing clean reward signals and stable policy learning.
    - **Design Motivation**: RL sample efficiency is poor in large action spaces; decoupling "how to map" (learned with dense supervision at one location) from "which to select" (learned with RL over 256 actions) is a practical stabilization technique.

3. **Layer-Aligned Temporal Token Propagation**:

    - **Function**: Provides temporal context across frames while avoiding semantic mismatch between deep and shallow layers.
    - **Mechanism**: Methods like ODTrack inject the previous frame's deep-layer temporal token into the current frame's shallow layer, causing high-level semantics to mix with low-level appearance features and semantic mismatch. The authors instead align layers: the temporal token $\mathbf{T}^{(t-1)}_l$ from layer $l$ of the previous frame is passed only to layer $l$ of the current frame. Specifically, layer $l$ input is $\mathbf{H}^{(t)}_l = [\mathbf{Z}^{(t)}_l, \mathbf{X}^{(t)}_l, \mathbf{T}^{(t)}_l, \mathbf{T}^{(t-1)}_l]$ (template, search, current frame temporal, previous frame same-layer temporal); after $\mathcal{f}_l$, $\mathbf{P}^{(t-1)}_{l+1}$ is discarded, only $\mathbf{T}^{(t)}_{l+1}$ is kept, and at layer $l+1$ it is concatenated with $\mathbf{T}^{(t-1)}_{l+1}$. Thus, the number of tokens per layer remains constant, avoiding linear growth with depth, and cross-frame information exchange always occurs at matching semantic levels.
    - **Design Motivation**: Different Transformer layers learn different abstraction levels; cross-layer mixing disrupts the hierarchical structure. Layer alignment respects the encoder's design for temporal propagation, with negligible computational overhead.

### Loss & Training

- **Warmup**: GIoU + L1, supervised only at the GT center location, for a few epochs.
- **RL**: $\ell = \ell_{\text{policy}} + 0.5 \ell_{\text{value}}$, 90 epochs, 2500 sequences per epoch, sequence length $T=8$, $\lambda=1$; AdamW lr=$10^{-4}$, lr ×0.1 after 72 epochs.
- **Data**: COCO + LaSOT + GOT-10k + TrackingNet + VastTrack, template box ×2 enlargement, search box ×4 enlargement, horizontal flip + brightness jitter.
- **Inference**: Frame-by-frame, select the box at $\arg\max \pi(\cdot|\mathbf{F}^{(t)})$ location, template update disabled by default.

## Key Experimental Results

### Main Results

| Method | LaSOT AUC | LaSOText AUC | TrackingNet AUC | GOT-10k AO |
|---|---|---|---|---|
| OSTrack-B256 | 69.1 | 47.4 | 83.1 | 71.0 |
| SeqTrack-B256 | 69.9 | 49.5 | 83.3 | 74.7 |
| ARTrack-B256 | 70.4 | 46.4 | 84.2 | 73.5 |
| SUTrack-B224 | 73.2 | 53.1 | 85.7 | 77.9 |
| ARPTrack-B256 | 72.6 | 52.0 | 85.5 | 77.7 |
| **RELO-B256** | **73.3** | **54.2** | **86.4** | **80.5** |
| ODTrack-L384 | 74.0 | 53.9 | 86.1 | 78.2 |
| LoRAT-L224 | 74.2 | 52.8 | 85.0 | 75.7 |
| SUTrack-L224 | 73.5 | 54.0 | 86.5 | 81.0 |
| ARPTrack-L384 | 74.2 | 54.2 | 86.6 | 81.5 |
| **RELO-L256** | **75.1** | **57.5** | **87.3** | **81.8** |

| Method | TNL2K | NFS | UAV123 |
|---|---|---|---|
| LoRAT-L224 | 61.1 | 66.6 | 71.9 |
| ARTrack-L384 | 60.3 | 67.9 | 71.2 |
| **RELO-B256** | 60.9 | 70.0 | 70.4 |
| **RELO-L256** | **63.6** | **71.3** | 71.4 |

### Ablation Study

| Model Variant | #Params (M) | FLOPs (G) | RTX4090 FPS | LaSOT AUC |
|---|---|---|---|---|
| RELO-T256 (HiViT-T) | 22 (+2 value) | 8 | 91 | 70.4 |
| RELO-B256 (HiViT-B) | 70 (+2) | 34 | 50 | 73.3 |
| RELO-L256 (HiViT-L) | 247 (+2) | 114 | 32 | 75.1 |

| Efficiency Track (Edge) | LaSOT / LaSOText | TrackingNet | GOT-10k | i9-CPU / Jetson FPS |
|---|---|---|---|---|
| MixFormerV2-S | 60.6 / 43.6 | 75.8 | - | 47 / 70 |
| AsymTrack-B | 64.7 / 44.6 | 80.0 | 67.7 | 38 / 64 |
| SUTrack-T224 | 69.6 / 50.2 | 82.7 | 72.7 | 23 / 34 |
| **RELO-T256** | **70.4 / 51.1** | **83.6** | **75.6** | 21 / 32 |

### Key Findings
- **Most significant gain on LaSOText**: From SUTrack-L's 54.0 to RELO-L256's 57.5, a +3.5 AUC improvement far exceeds the +0.9 on LaSOT—long-term tracking and cross-category scenarios are where center priors are least reliable, and RL reward's direct optimization shows the clearest advantage.
- **RL is effective even for small T256 models**: RELO-T256, with only 22M parameters and 91 FPS, still outperforms SUTrack-T224, indicating that the benefit of reward-driven localization mainly comes from the change in learning objective rather than model capacity, which is valuable for edge deployment.
- **Template update is disabled by default**: The authors deliberately avoid template update on LaSOT/LaSOText to prevent "benchmark overfitting," yet still achieve SOTA, indicating the RL-learned policy is inherently robust.
- **Value head adds only +2M parameters and is removed at test time**: It is an auxiliary parameter for training, with zero inference cost—this is standard practice for actor-critic and noteworthy.

## Highlights & Insights
- **Train-test alignment is truly enabling**: The center heatmap paradigm persisted because it was assumed "AUC is non-differentiable and thus cannot be directly optimized"; RELO breaks this assumption with REINFORCE, showing that RL is fully controllable on a 256-dimensional discrete action space—this approach is transferable to any vision task with non-differentiable but computable metrics (segmentation mIoU, detection AP, HOI, etc.).
- **Warmup-then-freeze is a key engineering trick for RL in visual perception**: The authors decouple "feature decoding" and "decision making" into separate training stages, using dense supervision for the former and policy search in a stable feature space for the latter—this "teach the hand first, then the brain" paradigm is worth reusing in robotics, RL-based detection, and similar scenarios.
- **Layer-aligned temporal propagation**: Though an engineering detail, it highlights a long-overlooked design principle in Transformer temporal trackers—cross-frame information flow should respect hierarchical semantic levels, not simply deep→shallow copying; it brings stable gains with almost zero computational cost.
- **L256 outperforms ARTrack-L/ODTrack-L with lower input resolution**: Indicates RL learns "where is more accurate" better, compensating for lower input resolution.

## Limitations & Future Work
- Only validated on single-object short-term/long-term tracking; multi-object tracking (MOT) association and detection-by-tracking for target disappearance/reappearance require further study.
- Reward design currently uses IoU + λ·AUC weighting, with $\lambda=1$ empirically chosen; adaptive scheduling for different sequence lengths is not discussed.
- No exploration of common RL techniques such as reward shaping or policy entropy annealing; future work may further exploit RL's potential.
- Freezing the entire encoder + regression during warmup means encoder features are not further optimized for the "action selection" task; enabling end-to-end RL for the encoder while maintaining stability is a future direction.
- No comparison with the latest LLM-based trackers (using VLMs for open-vocabulary tracking); cross-paradigm localization remains to be explored.

## Related Work & Insights
- **vs OSTrack / SUTrack / ODTrack (prior-driven center heatmap)**: These use Gaussian-smoothed center maps for classification supervision; RELO discards this entirely, using RL to optimize IoU/AUC—on the same backbone (HiViT), RELO's gains on LaSOText are especially notable, indicating the benefit comes mainly from the change in learning objective.
- **vs SLT (Kim 2022)**: SLT also uses RL to optimize average IoU, but only as fine-tuning on a prior-driven Siamese tracker, with the policy constrained by the underlying tracker's inductive bias; RELO uses RL as the fundamental localization mechanism from scratch (with warmup), not as a post-hoc correction.
- **vs Yun 2017 / Ren 2018 (early RL trackers)**: These early works used RL for box adjustment (translation/scale actions), with coarse granularity and sparse rewards; RELO uses dense rewards and spatial-grid actions, making the design more modern.
- **vs ARTrack / SeqTrack (autoregressive tracking)**: These regress boxes as token sequences, essentially still supervised learning; RELO is policy learning, conceptually closer to AlphaGo—using RL for metric-aligned optimization.

## Rating
- Novelty: ⭐⭐⭐⭐ RL is used as the main localization mechanism (not auxiliary) in visual tracking, with clear conceptual clarity
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 benchmarks + T/B/L model scales + edge deployment evaluation + complete ablation
- Writing Quality: ⭐⭐⭐⭐ Rigorous formulas, clear pipeline diagram (Fig.2), standard reward and RL formalization
- Value: ⭐⭐⭐⭐ The "train-test alignment" paradigm (using RL to replace handcrafted priors) is inspiring for the vision community, and open-source code enhances reusability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Unified Multimodal Visual Tracking with Dual Mixture-of-Experts](unified_multimodal_visual_tracking_with_dual_mixture-of-experts.md)
- [\[CVPR 2026\] Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry](../../CVPR2026/video_understanding/dual-agent_reinforcement_learning_for_adaptive_and_cost-aware_visual-inertial_od.md)
- [\[CVPR 2026\] Drift-Resilient Temporal Priors for Visual Tracking](../../CVPR2026/video_understanding/drift-resilient_temporal_priors_for_visual_tracking.md)
- [\[CVPR 2026\] Learning to Assist: Physics-Grounded Human-Human Control via Multi-Agent Reinforcement Learning](../../CVPR2026/video_understanding/learning_to_assist_physics-grounded_human-human_control_via_multi-agent_reinforc.md)
- [\[CVPR 2026\] SpikeTrack: A Spike-driven Framework for Efficient Visual Tracking](../../CVPR2026/video_understanding/spiketrack_a_spike-driven_framework_for_efficient_visual_tracking.md)

</div>

<!-- RELATED:END -->
