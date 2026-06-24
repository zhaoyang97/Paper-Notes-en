---
title: >-
  [Paper Note] World2Act: Latent Action Post-Training via Skill-Compositional World Models
description: >-
  [CVPR 2025][Video Generation][World Models] World2Act proposes a VLA post-training method based on latent space alignment: it aligns the latent video dynamic representations of a World Model with the action representations of a VLA via contrastive learning (instead of supervision in pixel space). It also introduces an LLM-driven skill decomposition pipeline to enable arbitrary-length video generation, achieving SOTA on RoboCasa and LIBERO with only 50 synthetic trajectories a…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "World Models"
  - "VLA Post-Training"
  - "Latent Alignment"
  - "Skill Decomposition"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 56819e8f86142fc8
---

# World2Act: Latent Action Post-Training via Skill-Compositional World Models

**Conference**: CVPR 2025  
**arXiv**: [2603.10422](https://arxiv.org/abs/2603.10422)  
**Code**: [https://wm2act.github.io/](https://wm2act.github.io/)  
**Area**: Video Generation  
**Keywords**: World Models, VLA Post-Training, Latent Alignment, Skill Decomposition, Contrastive Learning

## TL;DR
World2Act proposes a VLA post-training method based on latent space alignment: it aligns the latent video dynamic representations of a World Model with the action representations of a VLA via contrastive learning (instead of supervision in pixel space). It also introduces an LLM-driven skill decomposition pipeline to enable arbitrary-length video generation, achieving SOTA on RoboCasa and LIBERO with only 50 synthetic trajectories and a 6.7% improvement in real-world environments.

## Background & Motivation
**Background**: VLAs ($\pi_0$, GR00T-N1.6) learn via behavior cloning, but lack generalization to environmental changes and novel contact conditions. World Models (Cosmos-Predict2) can generate physically consistent future trajectories.

**Limitations of Prior Work**: (1) WM post-training typically uses pixel-space supervision (inverse dynamics models/pixel rewards), but pixel rollouts from WMs amplify noise and hallucinations; (2) Video diffusion models are trained on fixed-length clips, whereas robot tasks vary significantly in duration, making arbitrary-length generation a bottleneck; (3) Collecting real-world datasets that contain both camera movements and manipulation labels is extremely expensive.

**Key Challenge**: WMs contain rich dynamic priors, but transferring these priors at the pixel level introduces hallucinations and artifacts.

**Goal**: How to transfer the dynamic priors of a WM into a VLA policy without relying on pixels? How to enable WMs to support arbitrary-length video generation?

**Key Insight**: Perform action-video alignment in the latent space of the WM rather than the pixel space; employ an LLM to decompose long tasks into atomic skill segments to achieve stable long-video generation.

**Core Idea**: WM latent representation + VLA action representation contrastive alignment + skill-decomposed arbitrary-length WM = data-efficient VLA post-training.

## Method

### Overall Architecture
Two-stage post-training: Stage 1 trains a Video Adapter and an Action Adapter via contrastive learning to map both modalities to a shared latent space; Stage 2 freezes the VLA backbone and uses a lightweight Residual Policy to align VLA actions with the WM dynamic priors.

### Key Designs

1. **Skill-Compositional World Model (Skill-WM)**:

    - **Function**: Decomposes long tasks into atomic skill segments to support arbitrary-length video generation.
    - **Mechanism**: Segments video streams based on changes in gripper states, and uses an LLM (deepseek) to decompose global instructions into an ordered sequence of atomic skill descriptions, aligning video segments with language. During inference, the LLM generates a skill list, and the WM generates video segment-by-segment, using the final frame of the previous segment as the initial condition for the next.
    - **Design Motivation**: The length distribution of atomic skills is more uniform and concentrated (density improved by 17-72%), reducing error accumulation caused by long-tail distributions.

2. **Stage 1: Latent Space Alignment**:

    - **Function**: Trains a Video Adapter $\mathcal{B}_v$ (CNN) and an Action Adapter $\mathcal{B}_a$ (MLP) to map video latent representations and actions into a shared space.
    - **Mechanism**: Bidirectional InfoNCE contrastive loss + action reconstruction MSE loss. Chunk-wise alignment (one chunk per $M$ frames) is used instead of global trajectory alignment to prevent the model from exploiting shortcuts such as task identity. Hard negatives are sourced from different demos of the same skill.
    - **Architectural Details**: The Video Adapter is a 3-layer 1D temporal CNN mapping the WM's DiT latent features (token dimension ~4096) to a 256-dimensional shared space. The Action Adapter is a 2-layer MLP (hidden layer 512) that concatenates $M$-step action vectors and maps them to 256 dimensions. The InfoNCE temperature parameter is set to $\tau = 0.07$, and all unpaired chunks within the batch serve as easy negatives.
    - **Design Motivation**: Chunk-wise alignment encourages fine-grained temporal dynamic matching, unlike global embeddings which might ignore temporal details.

3. **Stage 2: Residual Policy Post-Training**:

    - **Function**: Freezes the VLA backbone and learns a lightweight residual correction $f^\theta$, resulting in $a_{\text{final}} = a_{\text{base}} + a_{\text{residual}}$.
    - **Mechanism**: Online rollout of the current policy, using the frozen WM to generate target video latent representations. The contrastive loss between $z^v$ and $z^a$ is then computed to train the residual network, without requiring rewards or env success signals.
    - **Design Motivation**: The residual policy preserves the base capabilities of the VLA (preventing catastrophic forgetting) and exhibits high sample efficiency (requiring only a lightweight network).

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{contrastive}}$$

## Key Experimental Results

### Main Results (RoboCasa)

| Method | Real Demos | Synthetic | SR |
|------|-----------|-----------|-----|
| $\pi_0$ | 300 | 0 | 62.5% |
| GR00T-N1.6 | 300 | 0 | 66.2% |
| Cosmos Policy | 50 | 0 | 65.7% |
| GR00T-N1.6-ft + DreamGen | 350 | +50 | 70.5% |
| **GR00T-N1.6-ft + World2Act** | 350 | **+50** | **72.6%** |
| **Cosmos + World2Act** | 50 | **+50** | **66.3%** |

### LIBERO (Average over 4 suites)

World2Act improves Cosmos Policy from 85.2% to 89.6% and GR00T-N1.6-ft from 87.6% to 91.2% on LIBERO-Long.

| Suite | Cosmos | +World2Act | GR00T-N1.6-ft | +World2Act |
|-------|--------|-----------|--------------|------------|
| LIBERO-Spatial | 91.0% | 93.4% | 92.8% | 95.0% |
| LIBERO-Object | 93.2% | 95.0% | 94.6% | 96.4% |
| LIBERO-Goal | 88.4% | 91.8% | 90.2% | 93.4% |
| LIBERO-Long | 85.2% | 89.6% | 87.6% | 91.2% |

LIBERO-Long (which requires multi-step long-sequence reasoning) shows the most significant improvement, validating the advantages of Skill-WM in long-horizon tasks.

### Ablation Study & Key Findings
- Latent space alignment vs. pixel space supervision: Latent space methods are more robust to hallucinations in WM rollouts.
- Skill-WM vs. Base-WM: Video generation temporal consistency is significantly improved (lower FVD) after skill decomposition.
- Chunk-wise contrastive > trajectory-wise contrastive: Fine-grained temporal alignment is more effective.
- Meaningful improvements with just 50 synthetic trajectories, showing extremely high data efficiency.
- Real-world experiments show a 6.7% improvement, validating sim-to-real transferability.

## Highlights & Insights
- **Latent Alignment Over Pixel Supervision**: The core insight is that WM latent representations are more robust to hallucinations than pixels. Pixel-level supervision amplifies noise, whereas latent representations preserve the essence of dynamic priors.
- **LLM-Driven Automated Skill Decomposition**: Visual stream segmentation using gripper states paired with instruction decomposition via an LLM. It is fully automated with a synchronization rate of >86%, serving as a practical data engineering solution.
- **Elegant Design of Residual Policy**: Without modifying original VLA weights, it only learns lightweight corrections, balancing capability retention and new knowledge injection.
- **Connecting WM and VLA via Contrastive Learning**: InfoNCE acts as a reward-free WM $\to$ VLA knowledge transfer signal, avoiding the instability of RL.

## Limitations & Future Work
- The choice of WM (Cosmos-Predict2) has a significant impact on post-training performance, and the applicability across different WMs is not fully explored; the authors only validated on the Cosmos family, leaving open-source alternatives (such as OpenSora) unverified.
- The residual policy assumes that the base VLA has basic capabilities—it might be ineffective for completely failing base policies. Fundamentally, it is "fine-tuning" rather than "learning from scratch."
- Skill decomposition relies on changes in gripper states and is not applicable to non-grasping tasks (such as pushing or sliding).
- Validated only on manipulation tasks; other embodied tasks like navigation are yet to be tested.

## Related Work & Insights
- **vs. DreamGen**: DreamGen infers pseudo-actions from WM rollouts using pixel-space IDMs, which suffers from hallucinations. World2Act operates in the latent space and is more robust (RoboCasa 72.6% vs. 70.5%).
- **vs. UWM/Cosmos Policy**: Unified video-action representation methods, but high-dimensional joint embedding is unstable. World2Act aligns them in a moderate dimension using contrastive learning.
- **vs. VLA-RFT/Ctrl-World**: Reward-based post-training methods that rely on RL policy gradients, whereas World2Act is reward-free.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of latent space alignment and skill-decomposed WM is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on RoboCasa + LIBERO + real world, with multi-baseline comparisons and ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear two-stage design with detailed technical execution.
- **Value**: ⭐⭐⭐⭐⭐ A new paradigm for WM $\to$ VLA knowledge transfer, both practical and highly efficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Navigation World Models](navigation_world_models.md)
- [\[CVPR 2025\] SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)
- [\[ICML 2025\] Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[CVPR 2026\] Inference-time Physics Alignment of Video Generative Models with Latent World Models](../../CVPR2026/video_generation/inference-time_physics_alignment_of_video_generative_models_with_latent_world_mo.md)

</div>

<!-- RELATED:END -->
