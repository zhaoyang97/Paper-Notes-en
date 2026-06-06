---
title: >-
  [Paper Note] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning
description: >-
  [ICCV 2025][Video Generation][Visual Reinforcement Learning] This paper proposes DisWM, a framework that pre-trains disentangled representations from "distracting videos" offline…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Visual Reinforcement Learning"
  - "World Models"
  - "Disentangled Representation"
  - "Knowledge Transfer"
  - "Latent Space Distillation"
date: 2026-05-08
content_hash: 5969c8342089649e
---

# Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning

**Conference**: ICCV 2025
**arXiv**: [2503.08751](https://arxiv.org/abs/2503.08751)  
**Code**: [https://qiwang067.github.io/diswm](https://qiwang067.github.io/diswm)  
**Area**: Video Generation
**Keywords**: Visual Reinforcement Learning, World Models, Disentangled Representation, Knowledge Transfer, Latent Space Distillation

## TL;DR

This paper proposes DisWM, a framework that pre-trains disentangled representations from "distracting videos" offline, then transfers semantic knowledge to downstream world models via offline-to-online latent space distillation, improving sample efficiency and robustness of visual reinforcement learning under environmental variations.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Visual reinforcement learning (VRL) faces severe challenges in real-world scenarios, where environmental complexity, variability, and visual distractions lead to significant performance degradation. Even minor environmental changes (e.g., lighting condition shifts) can cause large pixel-level perturbations that invalidate learned policies.

**Limitations of Prior Work**: Existing disentangled representation learning (DRL) methods, while promising for improving VRL interpretability and robustness, suffer from a critical limitation: they learn from scratch without prior world knowledge, requiring extensive environment interactions to acquire desired behaviors.

**Goal**: The core idea is to extract semantic prior knowledge from readily available "distracting videos" (videos containing visual distractions) and transfer this disentanglement capability to downstream control tasks via latent space distillation. Crucially, the pre-training videos and downstream tasks may originate from different domains (e.g., DMC and MuJoCo), differing in visual appearance, physical dynamics, action spaces, and reward functions.

## Method

### Overall Architecture

DisWM comprises three stages: (1) offline pre-training of an action-free video prediction model on distracting videos to extract disentangled features; (2) transferring semantic knowledge to an online world model via latent space distillation; and (3) online fine-tuning of the world model, incorporating action and reward information to further enhance disentangled representations.

### Key Designs

1. **Disentangled Representation Pretraining**: An action-free video prediction model based on $\beta$-VAE is trained on distracting videos. It consists of three components: a posterior learner (encoding observation $o_t$ into latent state $z_t$), a prior module (predicting future latent states from historical states), and a decoder (reconstructing $\hat{o}_t$ from $z_t$). The training loss contains three terms: image reconstruction loss, action-free KL divergence loss (ensuring prior-posterior consistency), and a disentanglement loss $\beta_2 \text{KL}[q_{\phi'}(\mathbf{z}_t|o_t) \| p(\mathbf{z}_t)]$ — which promotes orthogonality and disentanglement in the latent space by pushing the posterior distribution toward the standard normal $\mathcal{N}(\mathbf{0}, I)$. **Design Motivation**: The rich visual variations in distracting videos naturally provide the factor variations required for disentanglement learning, injecting "world knowledge" into downstream tasks.

2. **Offline-to-Online Latent Distillation**: A naive pre-train-then-fine-tune paradigm causes the disentangled information to be overwritten under large domain gaps. DisWM therefore transfers the disentanglement capability of the pre-trained model's latent variable $\mathbf{z}_{disen}$ to the world model's $\mathbf{z}_{task}$ via KL divergence distillation: $\mathcal{L}_{distill} = \text{KL}(\mathbf{z}_{disen} \| \mathbf{z}_{task}) = \sum \mathbf{z}_{disen} \cdot \log(\frac{\mathbf{z}_{disen}}{\mathbf{z}_{task}})$. The distillation weight $\eta$ is gradually annealed from 0.1 to 0.01 during the adaptation phase, enabling a smooth transition from heavy reliance on pre-trained knowledge to increasingly autonomous learning.

3. **Disentangled World Model Adaptation**: A full world model $\mathcal{M}_\phi$ is built upon DreamerV2, comprising a recurrent transition function $h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$, posterior/prior states, reconstruction, reward prediction, and discount factor prediction. The total training loss augments the standard world model objective with disentanglement regularization and distillation terms: $\mathcal{L}(\phi) = \text{Reconstruction} + \text{Reward Prediction} + \text{Discount Prediction} + \alpha\text{KL Divergence} + \beta\text{KL}[q_\phi(\mathbf{z}_t|o_t) \| p(\mathbf{z}_t)] + \eta\mathcal{L}_{distill}$. **Key Insight**: The introduction of action and reward signals during online fine-tuning enriches data diversity, which in turn enhances disentangled representation learning — forming a positive feedback loop.

### Loss & Training

The pre-training stage uses a dataset of 1 million frames of distracting videos (generated by DreamerV2 interacting with environments featuring visual color distractions). Online fine-tuning is limited to $1\times10^6$ environment steps. The latent dimensions of $\mathbf{z}_{disen}$ and $\mathbf{z}_{task}$ are both set to 20. Visual observations are resized to $64\times64$. Training on a single RTX 3090 GPU takes approximately 16 hours and requires 55 GB of GPU memory. Color schemes are switched at the midpoint of training to simulate changing distractions.

## Key Experimental Results

### Main Results

| Task | DisWM | DreamerV2 | APV | TED | DV2 Finetune | Notes |
|------|-------|-----------|-----|-----|--------------|-------|
| Cheetah Run→Walker Walk | Best | Poor | Medium | Medium | 2nd | Cross-domain transfer |
| Reacher Easy→Cheetah Run | Best | Poor | Below Medium | Medium | 2nd | Cross-domain transfer |
| Cheetah Run→Hopper Stand | Best | Poor | Low | Medium | 2nd | Intra-DMC transfer |
| Finger Spin→Reacher Easy | Best | Poor | Low | Medium | 2nd | Intra-DMC transfer |
| Finger Spin→Cartpole Swingup | Best | Poor | Medium | Medium | 2nd | Intra-DMC transfer |
| Reacher Easy→Pusher (MuJoCo) | Best | Poor | Low | Medium | Medium | Cross-domain (DMC→MuJoCo) |

DisWM consistently achieves optimal or near-optimal sample efficiency and final performance across all six tasks.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| DisWM (full) | Best episode return | Baseline |
| w/o latent distillation | Degraded early-training performance | Distillation provides critical knowledge transfer in early training |
| w/o disentanglement constraint (pretrain+finetune) | Significant performance drop | DRL training and disentangled representations are essential for learning efficiency |
| β too small (insufficient disentanglement) | Entangled representations learned | Cannot effectively handle environmental variations |
| β too large (excessive disentanglement) | Degraded reconstruction quality | Impairs environment modeling accuracy |
| η too low | Insufficient knowledge transfer | Downstream agent acquires inadequate prior knowledge |
| η too high | Overfitting to pre-trained model | Hinders adaptive learning on downstream tasks |
| Different pre-training video sources | All outperform DreamerV2 w/o pre-training | Framework is robust to pre-training domain selection |

### Key Findings

- DisWM effectively transfers semantic knowledge even under extreme domain gaps (DMC→MuJoCo, with mismatched dynamics, action spaces, and reward functions).
- Pre-training with distracting videos from any DMC task improves Cartpole Swingup performance, demonstrating robustness to pre-training domain selection.
- Traversal visualizations of the $\beta$-VAE clearly show that the pre-trained model successfully learns independent factors such as color and position.
- Fine-tuning stage visualizations on MuJoCo Pusher demonstrate fine-grained disentanglement of attributes such as arm color and object position.

## Highlights & Insights

- Framing VRL robustness to environmental variation as a domain transfer learning problem offers a novel perspective.
- The use of "distracting videos" is elegant: visual distractions naturally provide the factor variations required for disentanglement learning.
- The offline-to-online distillation design prevents knowledge forgetting caused by direct fine-tuning, and the progressive weight decay strategy is practically effective.
- The positive feedback loop — whereby action and reward signals during online fine-tuning reinforce disentangled representation learning — is an interesting and noteworthy finding.

## Limitations & Future Work

- Disentanglement learning remains challenging under more complex environmental variations, such as temporally varying background video distractions.
- The generation of the pre-training dataset (1 million frames) itself requires environment interaction; future work could explore the use of unlabeled real-world videos.
- The current validation is limited to continuous control tasks; the framework could be extended to discrete action spaces or more complex robotic manipulation tasks.
- Disentanglement quality depends on the tuning of $\beta$, and an adaptive mechanism is lacking.

## Related Work & Insights

- In dialogue with video pre-training world model methods such as APV and IPV, DisWM provides stronger robustness to environmental variation through disentanglement constraints.
- The combination of disentangled representations and world models offers a promising direction for interpretable VRL.
- The cross-domain transfer approach via latent space distillation generalizes naturally to more practical scenarios such as sim-to-real transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ The pipeline design of distracting videos → disentangled pre-training → world model distillation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six tasks, comprehensive ablations, sensitivity analysis, and cross-domain validation.
- Writing Quality: ⭐⭐⭐⭐ Clear framework presentation with rich visualizations.
- Value: ⭐⭐⭐⭐ Provides a practical solution for VRL under environmental variation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](../../CVPR2026/video_generation/goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] DisMo: Disentangled Motion Representations for Open-World Motion Transfer](../../NeurIPS2025/video_generation/dismo_disentangled_motion_representations_for_openworld_moti.md)
- [\[NeurIPS 2025\] RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation](../../NeurIPS2025/video_generation/rlgf_reinforcement_learning_with_geometric_feedback_for_autonomous_driving_video.md)
- [\[ICCV 2025\] NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors](normalcrafter_learning_temporally_consistent_normals_from_video_diffusion_priors.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)

</div>

<!-- RELATED:END -->
