---
title: >-
  [Paper Note] InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild
description: >-
  [ICCV 2025][LLM Evaluation][Text-to-motion generation] This paper proposes the InterSyn framework, which jointly models single-person and multi-person motions within a unified interleaved sequence via an Interleaved Lear…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "Text-to-motion generation"
  - "multi-person interaction"
  - "diffusion models"
  - "interleaved learning"
  - "motion coordination"
date: 2026-05-08
content_hash: b4e43ed8a1048bd3
---

# InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild

**Conference**: ICCV 2025
**arXiv**: [2508.10297](https://arxiv.org/abs/2508.10297)  
**Code**: Not released (open-source planned)  
**Area**: Motion Generation
**Keywords**: Text-to-motion generation, multi-person interaction, diffusion models, interleaved learning, motion coordination

## TL;DR

This paper proposes the InterSyn framework, which jointly models single-person and multi-person motions within a unified interleaved sequence via an Interleaved Learning strategy, combined with a Relative Coordination Refinement (REC) module, to generate more natural and coordinated human interaction motions. On the InterHuman test set, FID is reduced by 6.1% and R Precision Top-1 is improved by 2.8% compared to FreeMotion.

## Background & Motivation

Text-driven human motion generation (T2M) has broad applications in animation, virtual reality, and related fields. Despite significant progress on single-person datasets such as HumanML3D, generating diverse interaction motions in real-world scenarios remains challenging.

Existing methods exhibit two critical limitations:

**Single-person and interaction motions are handled in isolation**: Semantic motions (walking, talking, etc.) are learned independently on HumanML3D, while social motions (hugging, handshaking, etc.) are modeled separately on InterHuman. In reality, however, human motion **fluidly transitions** between solitary and social activities — this dynamic interleaving is an intrinsic characteristic of natural movement.

**Lack of inter-person cue modeling**: In multi-person scenarios, individuals continuously adjust their movements in response to subtle signals from others. Existing methods cannot effectively model this **mutually adaptive** dynamic process.

The core hypothesis of this paper is inspired by educational psychology theories including **situated learning theory** and **cultural-historical activity theory**: semantic and social motions mutually reinforce each other, and joint learning yields better generalization.

## Method

### Overall Architecture

InterSyn consists of two stages:
1. **Interleaved Interaction Synthesis (INS)**: Single-person and multi-person interaction motions are fused into a unified interleaved sequence, generated using a conditional diffusion model.
2. **Relative Coordination Refinement (REC)**: A coordinator network refines the spatial relationships and temporal synchronization among multiple agents in the generated interaction motions.

### Key Designs

1. **Interleaved Data Construction**: A motion bucket $u = (u_x, u_y) \in \mathbb{R}^{2 \times T \times K \times C}$ is initialized. Single-person motions $p_s$ and two-person interaction pairs $(p_x, p_y)$ are randomly sampled and combined into a continuous sequence via a fusion function $U(\cdot)$: $u = U(p_x, p_y, p_s, t_i, t_s)$, where $t_i$ and $t_s$ denote the start time indices of interaction and solitary motions, respectively. $U(\cdot)$ handles motion alignment, smooth transitions, and orientation adjustment. Since the two datasets use different skeletons, rotation and translation alignment is performed via forward kinematics (FK).

2. **Conditional Motion Diffusion Model (CMDM)**: A Transformer-based diffusion network $M_s$ takes the interaction time steps $t_i$ and $t_s$ as conditional inputs, with a temporal embedding layer encoding $t_i, t_s$ and a text embedding layer encoding the concatenated description $w_u$. During training, noise is added to motion $u$ to obtain $u^t$, and the model predicts the denoised motion: $\hat{u} = M_s(u, w_u, t_i, t_s)$. By conditioning on temporal signals, the model learns to generate continuous sequences that smoothly transition between solitary and interaction motions.

3. **Relative Coordination Refinement (REC)**: A Transformer-based coordinator network $M_c$ refines the interaction motions. For a two-person interaction, the predicted motion of the first person is refined with reference to the second person's motion: $\phi_x = M_c(\hat{u}_x, \hat{u}_y, w_u)$. The refined $\phi_x$ is then used to fine-tune $\hat{u}_y$: $\phi_y = M_c(\hat{u}_y, \phi_x, w_u)$. The key constraint is the **relative coordination loss**: when $\phi_x$ is already a reasonable interaction motion relative to $\hat{u}_y$, the fine-tuning adjustment to $\hat{u}_y$ should be minimal: $\mathcal{L}_{\text{rela}} = \|\phi_y - \hat{u}_y\|_2$.

### Loss & Training

Two-stage training:
- **Stage 1** (INS): $\mathcal{L}_I = \lambda_1 \mathcal{L}_{\text{rec}} + \lambda_2 \mathcal{L}_{\text{smooth}}$, where $\mathcal{L}_{\text{smooth}}$ enforces smoothness within a ±5-frame window at fusion boundaries.
- **Stage 2** (REC): INS is frozen; the coordinator is trained. $\mathcal{L}_R = \lambda_3 \mathcal{L}_{\text{rela}} + \lambda_4 \mathcal{L}_{\text{dm}}$, where $\mathcal{L}_{\text{dm}}$ is the masked joint distance map loss from InterGen.
- Hyperparameters: $\lambda_1=1,\ \lambda_2=0.1,\ \lambda_3=1,\ \lambda_4=0.5$
- Diffusion time steps: 1,000; DDIM sampling at inference
- Text encoder: frozen CLIP-ViT-L-14
- Trained on a single H100 GPU for 31 hours; batch size 256; 44 GB VRAM
- Alternating training strategy: reconstruction loss is computed alternately on single-person data and interleaved data

## Key Experimental Results

### Main Results (InterHuman Test Set)

| Method | R Precision Top1 ↑ | FID ↓ | MM Dist ↓ | Diversity → | MModality ↑ |
|---|---|---|---|---|---|
| TEMOS | 0.224 | 17.375 | 5.342 | 6.939 | 0.535 |
| MDM | 0.153 | 9.167 | 6.125 | 7.602 | **2.355** |
| InterGen | 0.264 | 13.404 | 3.882 | 7.770 | 1.451 |
| FreeMotion | 0.326 | 6.740 | **3.848** | 7.828 | 1.226 |
| **InterSyn** | **0.335** | **6.332** | 3.856 | **7.763** | 1.601 |

### Ablation Study

| Configuration | R Precision Top1 | FID ↓ | MM Dist ↓ |
|---|---|---|---|
| s-i-s (default) | 0.298 | 0.417 | 3.707 |
| s-i-s-i (more transitions) | 0.242 | 0.469 | 3.958 |
| s-i-s-i-s (5 segments) | 0.115 | 0.638 | 4.436 |
| w/o coordinator | 0.103 | 0.847 | 5.842 |
| w/o $\mathcal{L}_{\text{rela}}$ | 0.283 | 0.537 | 3.838 |
| w/o $\mathcal{L}_{\text{smooth}}$ | 0.295 | 0.431 | 3.712 |

### Key Findings

1. **Number of interleaved segments**: The s-i-s configuration (solo–interaction–solo, 3 segments) achieves the best performance; increasing the number of transitions (4 or 5 segments) leads to significant degradation, as excessive switching within a fixed frame budget truncates critical motion phases.
2. **The coordinator is critical**: Removing the coordinator causes FID to surge by 103.1% and MM Dist to increase by 57.6%.
3. **Unique role of $\mathcal{L}_{\text{rela}}$**: Removing it reduces Top-1 R Precision by 5.0%; it is specifically responsible for aligning the interactive dynamics between multiple agents.
4. **Qualitative role of $\mathcal{L}_{\text{smooth}}$**: Although removing $\mathcal{L}_{\text{smooth}}$ has a small quantitative impact (FID increases by only 3.4%), qualitative analysis reveals noticeable jitter at gait transitions.
5. **Dynamic environment evaluation**: On the interleaved motion benchmark (unified HumanML3D + InterHuman test set), InterSyn reduces FID by 42.1% compared to FreeMotion.
6. **Time step configuration**: $t_s=0,\ \text{random } t_i$ is the optimal setting, balancing accuracy, diversity, and multimodality.

## Highlights & Insights

- **Learning paradigm innovation**: Shifting from "separately learning single-person and interaction motions" to "jointly learning interleaved sequences" better reflects the cognitive process of human motor learning.
- **Unified first-person perspective**: All single-person and interaction motions are processed from a first-person viewpoint, with FK-based skeleton alignment resolving cross-dataset skeleton incompatibility.
- **Symmetric design of the REC module**: Through a two-step refinement (first refining $u_x$, then using $\phi_x$ to reversely fine-tune $u_y$), the module implicitly models the bidirectional dependency of interaction.
- **Key distinction from FreeMotion**: FreeMotion unifies single-person and multi-person generation through conditional motion distributions but handles the two modalities separately; InterSyn jointly encodes them in a unified latent space, enabling seamless transitions.

## Limitations & Future Work

- Training and evaluation are limited to dyadic interactions; extension to groups of more than two persons is only partially addressed through the coordinator design at inference time, and its quality remains unknown.
- Performance is best with at most two interleaved segments, limiting the complexity of generated motions.
- Skeleton alignment relies on the accuracy of FK transforms, which may introduce cumulative errors for complex motions.
- Integration of additional conditioning signals such as audio or scene context has not been explored.
- The maximum number of generated frames is constrained by the fixed 196-frame limit.

## Related Work & Insights

- InterSyn inherits the direct $x_0$ prediction diffusion paradigm from MDM, extending it to multi-person interleaved motions.
- InterGen's mutual attention mechanism and FreeMotion's number-free generation are important predecessors.
- The interleaved learning concept is generalizable to other multi-agent interaction scenarios, such as multi-robot collaboration and multi-vehicle trajectory prediction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The interleaved learning strategy is a novel entry point grounded in cognitive theories of human motor learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies comprehensively cover data organization, time step configuration, and loss function design.
- **Writing Quality**: ⭐⭐⭐⭐ The method is described in detail with intuitive visual comparisons, though some notation is inconsistent.
- **Value**: ⭐⭐⭐⭐ InterSyn provides a new paradigm for human interaction motion generation, and the dynamic environment evaluation benchmark is also a valuable contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing](dista-net_dynamic_closely-spaced_infrared_small_target_unmixing.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[AAAI 2026\] MindVote: When AI Meets the Wild West of Social Media Opinion](../../AAAI2026/llm_evaluation/mindvote_when_ai_meets_the_wild_west_of_social_media_opinion.md)
- [\[NeurIPS 2025\] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](../../NeurIPS2025/llm_evaluation/memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](../../ACL2026/llm_evaluation/dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)

</div>

<!-- RELATED:END -->
