---
title: >-
  [Paper Note] Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback
description: >-
  [AAAI 2026][Image Generation][Audio-driven animation] This paper proposes a DiT-based audio-driven human video generation framework built on Wan2.1, featuring a LoRA training strategy for long video generation…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Audio-driven animation"
  - "multi-character animation"
  - "Diffusion Transformer"
  - "training-free inference"
  - "DPO"
date: 2026-05-08
content_hash: b2754ae753e1a1ec
---

# Playmate2: Training-Free Multi-Character Audio-Driven Animation via Diffusion Transformer with Reward Feedback

**Conference**: AAAI 2026
**arXiv**: [2510.12089](https://arxiv.org/abs/2510.12089)  
**Code**: [https://playmate111.github.io/Playmate2/](https://playmate111.github.io/Playmate2/)  
**Area**: Image Generation
**Keywords**: Audio-driven animation, multi-character animation, Diffusion Transformer, training-free inference, DPO

## TL;DR

This paper proposes a DiT-based audio-driven human video generation framework built on Wan2.1, featuring a LoRA training strategy for long video generation, partial parameter updates combined with DPO reward feedback to enhance lip synchronization and motion naturalness, and a novel training-free Mask-CFG method that enables multi-character (≥3 persons) audio-driven animation for the first time.

## Background & Motivation

### State of the Field

Audio-driven character animation is a core capability in digital human research, with broad applications in film, gaming, and virtual reality. Driven by advances in diffusion models, the field has progressed substantially beyond the GAN era. Current methods fall into two categories:

- **Portrait Animation**: Focuses exclusively on facial expression synthesis (EMO, Hallo, Sonic, etc.), neglecting background and full-body motion, yielding poor results in complex scenes.
- **Human Animation**: Leverages video diffusion models for full-body animation (OmniHuman, FantasyTalking, etc.), but faces multiple challenges.

### Limitations of Prior Work

**Conflict between lip-sync and body motion**: Existing methods tend to sacrifice natural body motion in pursuit of precise lip synchronization, or vice versa.

**Poor temporal consistency in long videos**: Long video generation suffers from motion jitter and abrupt transitions, with no guarantee of temporal coherence.

**Limited multi-character support**: Most existing methods support only single-person animation. The few that support multiple characters (e.g., MultiTalk, HunyuanVideo-Avatar) require constructing multi-speaker datasets and substantially modifying model architectures, making them resource-intensive and non-scalable.

### Root Cause

How can high-quality multi-character audio-driven animation be achieved without building multi-person datasets or modifying model architectures? How can lip synchronization, motion naturalness, and long-term temporal consistency be simultaneously addressed?

### Starting Point

The paper leverages the large-scale video diffusion model Wan2.1 as a backbone and addresses the above challenges through three design levels: a LoRA strategy for long video generation, DPO reward feedback for lip-sync and motion quality, and training-free Mask-CFG inference for multi-character support.

## Method

### Overall Architecture

The framework is built upon the Wan2.1 video diffusion model and comprises three core components:

1. LoRA-based long video generation strategy
2. Partial parameter update + DPO reward feedback training
3. Mask-CFG training-free multi-character inference

### Key Designs

#### 1. **LoRA-based Long Video Generation**

**Function**: Addresses a problem specific to Wan2.1's long video generation—its $1+T$ input format processes the first frame independently, causing forgetting and drift in long videos.

**Mechanism**: Rather than using Wan2.1's original $1+T$ chunking scheme, the video is divided into $T/4$ chunks, each encoded as a single latent representation. LoRA training is applied only to the self-attention and cross-attention modules within the Wan2.1 DiT blocks, without adding audio cross-attention.

**Why prior approaches fail**:
- OmniAvatar's final latent extension strategy: errors accumulate over time, severely degrading long-video quality.
- HunyuanVideo-Avatar's Time-aware Position Shift Fusion: produces visible artifacts under the DiT backbone's special input format.

**Design Motivation**: The LoRA strategy preserves the base model's capabilities while enabling adaptation at low training cost. This stage is completed with 16 A100 GPUs over 5,000 steps.

#### 2. **Partial Parameter Update + DPO Reward Feedback**

**Function**: Improves lip synchronization accuracy and facial expression naturalness in two steps.

**Step 1: Audio Cross-Attention Module Training**

Following the first-stage LoRA training, an audio cross-attention module is introduced. Multi-scale audio features are extracted via Wav2Vec, and parameters are updated using a Flow Matching objective:

$$\mathcal{L} = \mathbb{E}_{z_0, z_1, z_a, t} \|v_{\theta_a}(z_t, z_a, t; \theta_a) - v_t\|^2$$

Audio features are aggregated every 4 frames into a single representation, ensuring temporal alignment with the compressed video latents.

**Step 2: DPO Reward Feedback**

Unlike Hallo4, which requires human annotators to construct preference datasets, this paper adopts a more efficient automated approach:
- For each training sample, 5 segments are randomly selected.
- LatentSync is used to compute Sync-C scores.
- The highest-scoring segment is designated $y^w$ (preferred); the lowest is $y^l$ (dispreferred).
- Training uses the Flow-DPO loss:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{y^w, y^l, t}\left[\log\sigma\left(-\frac{\beta_t}{2}(\|v^w - v_{\theta_a}(y_t^w,t)\|^2 - \|v^w - v_{\text{ref}}(y_t^w,t)\|^2 - \|v^l - v_{\theta_a}(y_t^l,t)\|^2 + \|v^l - v_{\text{ref}}(y_t^l,t)\|^2)\right)\right]$$

The total training loss is: $\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{DPO}}$, with $\lambda = 0.1$. This stage uses 32 A100 GPUs for 100K steps of audio attention training followed by 100K steps of DPO.

#### 3. **Mask-CFG Training-Free Multi-Character Animation**

**Function**: Enables multi-character audio-driven animation at inference time without any training or model modification.

**Core Idea**: Given an audio condition set $A = \{a_1, a_2, \ldots, a_n\}$ and a corresponding set of mutually exclusive binary masks $M = \{m_1, m_2, \ldots, m_n\}$—where $a_1$ denotes silent audio and $m_1$ the background mask—the following conditional independence property is derived mathematically:

$$p(a_i \mid x_t) = p(a_i \mid m_i \odot x_t)$$

Substituting into the CFG formulation yields the Mask-CFG velocity field:

$$\hat{v}_\theta(x_t, a, t) = v_\theta(x_t, t) + \sum_{i=1}^n \lambda_i m_i \odot [v_\theta(x_t, a_i, t) - v_\theta(x_t, t)]$$

Each character's audio condition is routed to its corresponding spatial region via a mask; background regions and silent characters follow unconditional generation, with $\lambda = 5.0$.

**Design Motivation**: Existing multi-character methods (MultiTalk, HunyuanVideo-Avatar) all require constructing multi-person datasets and modifying cross-attention mechanisms, incurring high cost and lacking generality. Mask-CFG operates entirely at inference time with no training or model modifications, achieving training-free audio-driven animation for ≥3 characters for the first time.

### Loss & Training

Training proceeds in three stages:
1. **LoRA stage**: 16×A100, 5,000 steps, LoRA only.
2. **Audio attention stage**: 32×A100, 100K steps, Flow Matching.
3. **DPO stage**: 32×A100, 100K steps; $v_{\text{ref}}$ is updated every 10K steps.

## Key Experimental Results

### Main Results

Quantitative comparison on the HDTF and CelebV-HQ datasets:

| Method | FID ↓ (HDTF/CelebV) | FVD ↓ (HDTF/CelebV) | Sync-C ↑ (HDTF/CelebV) | Sync-D ↓ (HDTF/CelebV) |
|--------|---------------------|---------------------|------------------------|------------------------|
| Sonic | 46.47/87.61 | 213.15/232.65 | 6.91/5.28 | 8.57/8.15 |
| HunyuanVideo-Avatar | 34.80/78.85 | 175.00/230.41 | 7.43/4.81 | 8.12/8.11 |
| MultiTalk | 38.51/77.92 | 172.02/206.46 | 8.57/5.64 | 6.97/7.67 |
| OmniAvatar | 36.19/82.40 | 137.19/169.66 | 7.72/5.36 | 7.66/7.76 |
| **Ours (w/ DPO)** | **27.63/66.11** | **81.86/133.78** | **8.15/5.49** | **7.32/7.66** |

On HDTF, FID improves from the second-best 29.05 to 27.63, and FVD from 86.10 to 81.86, surpassing all competing methods across all metrics.

### Ablation Study

**User Study** (MOS scores from 50 participants, 5-point scale):

| Method | Lip Sync ↑ | Video Clarity ↑ | Naturalness ↑ | Visual Appeal ↑ |
|--------|-----------|----------------|--------------|----------------|
| MultiTalk | 3.93 | 3.79 | 3.93 | 3.79 |
| OmniAvatar | 3.71 | 3.77 | 3.21 | 3.29 |
| **Ours** | **4.02** | **3.98** | **3.90** | **4.11** |

**DPO ablation**: Incorporating DPO yields consistent improvements across all metrics (FID: 29.05→27.63; FVD: 86.10→81.86). Visualizations show that the DPO model generates richer, contextually appropriate facial expressions under singing audio, whereas the non-DPO variant produces flat expressions.

**Long video ablation**: OmniAvatar's latent extension strategy suffers from severe quality degradation due to error accumulation; HunyuanVideo-Avatar's Position Shift produces visible artifacts at transition regions; the proposed method generates temporally coherent, identity-consistent long videos.

### Key Findings

1. The proposed method achieves state-of-the-art FID and FVD on HDTF; notably, FVD of 81.86 substantially outperforms the second-best method.
2. DPO not only improves lip synchronization but also significantly enhances the richness and naturalness of facial expressions.
3. Mask-CFG is the first training-free audio-driven method supporting ≥3 characters, requiring neither multi-person datasets nor model modifications.
4. A model supporting multi-character animation can be trained on 300K single-person videos (800+ hours), at a cost far lower than constructing multi-person datasets.

## Highlights & Insights

1. **Mathematical elegance of Mask-CFG**: By leveraging the masked conditional independence assumption and deriving from the CFG formulation, the multi-character problem is reduced to a simple inference-time operation—a seamless integration of theory and practice.
2. **Efficient DPO implementation**: Preference pairs are constructed automatically using LatentSync without human annotation, making the approach more economical and practical than Hallo4.
3. **Systematic engineering**: The three-stage training pipeline addresses distinct problems incrementally, with clear objectives at each stage, avoiding the complexity of end-to-end training.
4. **High practicality**: Mask-CFG is a plug-and-play, architecture-agnostic method that can be directly applied to any audio-driven diffusion model.

## Limitations & Future Work

1. The quality of multi-character animation depends on accurate mask segmentation; errors in automatic segmentation directly degrade results.
2. Mask-CFG assumes independence of audio conditions across different regions, which may break down in scenes with close character interactions.
3. All training data consist of single-person videos, limiting the model's ability to handle physical collisions and occlusions in multi-person interaction scenes.
4. Multi-character inference at test time requires multiple forward passes (one per audio condition), resulting in linearly increasing computation.
5. Evaluation is conducted only on HDTF and CelebV-HQ; generalization to real-world application scenarios remains to be verified.

## Related Work & Insights

- **Hallo series** (Hallo/Hallo3/Hallo4): An evolutionary trajectory from portrait to full-body animation; Hallo4 is the first to introduce DPO but requires human annotation.
- **MultiTalk**: Achieves multi-person animation via Label Rotary Position Embedding, but requires a multi-person dataset.
- **Extended applications of CFG**: The Mask-CFG concept is generalizable to other multi-entity conditional generation scenarios (e.g., multi-character text-driven generation, region-controllable editing).
- **OmniHuman/OmniAvatar**: Comprehensive full-body animation methods, but with limited capabilities for long video and multi-person scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The training-free Mask-CFG multi-character approach is innovative; the automated preference data construction via DPO also demonstrates ingenuity.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers quantitative evaluation, qualitative results, user studies, and ablations, though limited to two datasets.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — The first training-free multi-character solution, with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation](../../ICML2026/image_generation/semantic-aware_motion_encoding_for_topology-agnostic_character_animation.md)
- [\[AAAI 2026\] MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment](macs_multi-source_audio-to-image_generation_with_contextual_significance_and_sem.md)
- [\[AAAI 2026\] ProCache: Constraint-Aware Feature Caching with Selective Computation for Diffusion Transformer Acceleration](procache_constraint-aware_feature_caching_with_selective_computation_for_diffusi.md)
- [\[ICLR 2026\] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](../../ICLR2026/image_generation/training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)

</div>

<!-- RELATED:END -->
