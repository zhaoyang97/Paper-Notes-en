---
title: >-
  [Paper Note] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning
description: >-
  [CVPR 2026][Image Generation][Video customization] DreamVideo-Omni is proposed as a two-stage progressive training paradigm—omni-motion identity supervised fine-tuning followed by latent identity reward feedback learning…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Video customization"
  - "multi-subject identity preservation"
  - "omni-motion control"
  - "latent space reinforcement learning"
  - "DiT"
date: 2026-05-08
content_hash: ca6e6f65306172df
---

# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

**Conference**: CVPR 2026
**arXiv**: [2603.12257](https://arxiv.org/abs/2603.12257)  
**Code**: [Project Page](https://dreamvideo-omni.github.io)  
**Area**: Image Generation
**Keywords**: Video customization, multi-subject identity preservation, omni-motion control, latent space reinforcement learning, DiT

## TL;DR

DreamVideo-Omni is proposed as a two-stage progressive training paradigm—omni-motion identity supervised fine-tuning followed by latent identity reward feedback learning—that, within a single DiT architecture, for the first time unifies multi-subject customization with full-granularity motion control (global bounding boxes + local trajectories + camera motion).

## Background & Motivation

**Background**: Large-scale diffusion models have achieved breakthroughs in text-to-video generation, yet real-world applications demand simultaneous high-fidelity generation with precise control over multi-subject identity and multi-granularity motion. Existing methods focus either on subject customization (e.g., ConsisID, VideoMage) or on motion control (e.g., Tora, Wan-Move), with few unified frameworks.

**Limitations of Prior Work**: Current unification attempts face three major bottlenecks: (a) **Limited motion control granularity**: most methods rely on a single signal (bounding box / depth map / sparse trajectory), precluding simultaneous control over global position, local dynamics, and camera motion; (b) **Control ambiguity**: in multi-subject scenes, models cannot determine which motion signal corresponds to which subject; (c) **Identity degradation**: introducing motion control reduces identity fidelity, since identity preservation requires pixel-level consistency while motion control requires pixel-level dynamics—a contradiction that standard diffusion reconstruction losses cannot reconcile.

**Key Challenge**: Identity preservation (encouraging static pixel-level consistency) and motion control (requiring dynamic pixel evolution) are inherently conflicting objectives that standard diffusion losses are insufficient to satisfy simultaneously.

**Goal**: To achieve multi-subject customization and full-granularity motion control (global + local + camera) simultaneously within a single framework without sacrificing identity fidelity.

**Key Insight**: (a) Explicitly binding motion signals to their corresponding subjects to eliminate ambiguity; (b) employing reinforcement learning from human preferences—rather than reconstruction loss—to optimize identity preservation, as identity assessment is fundamentally a subjective perceptual alignment task.

**Core Idea**: A two-stage paradigm: Stage 1 performs joint training with structured triplets ⟨reference subject, global bounding box, local trajectory⟩ and introduces group/role embeddings to resolve ambiguity; Stage 2 trains a Latent Identity Reward Model (LIRM) to compute identity rewards directly in latent space, bypassing the VAE decoder for efficient reinforcement learning.

## Method

### Overall Architecture

Built upon the Wan2.1-1.3B T2V DiT pre-trained model, the framework adopts a two-stage progressive training scheme:

- **Stage 1 (Omni-Motion Identity SFT)**: Reference images, bounding boxes, and trajectories are jointly injected into the DiT as conditioning signals, with joint training covering single/multi-subject customization, global/local motion control, and camera motion control.
- **Stage 2 (Latent Identity Reinforcement Learning)**: A LIRM reward model is first trained, then Reward Feedback Learning (LIReFL) is applied to directly optimize identity preservation in latent space.

### Key Designs

#### 1. Condition-Aware 3D RoPE

- **Function**: Assigns distinct temporal position indices to heterogeneous inputs (video frames, reference images, trajectories, padding).
- **Mechanism**: Video frames use sequential temporal indices $t \in [0, T-1]$; reference images share a special temporal index $t_{\text{ref}}$ so the model treats them as static conditions; padding uses an invalid index $t_{\text{pad}}$ to be ignored by the model; trajectories inherit video frame indices to ensure spatiotemporal alignment.
- **Design Motivation**: Directly concatenating heterogeneous tokens causes positional encoding confusion. Ablation experiments show that removing this component causes training collapse (R-DINO drops catastrophically from 0.499 to 0.139).

#### 2. Group & Role Embeddings

- **Function**: Resolves the ambiguity of "which motion signal controls which subject" in multi-subject scenes.
- **Mechanism**: Each control unit ⟨reference subject, bounding box, trajectory⟩ is assigned a unique group embedding; a separate role embedding distinguishes "appearance assets" (reference images) from "motion guidance" (bounding boxes/trajectories).
- **Design Motivation**: Without this design, multi-subject mIoU drops from 0.532 to 0.459 and EPE increases from 6.80 to 20.69.

#### 3. Hierarchical Bounding Box Injection

- **Function**: Injects the bounding box latent representation into the output of each DiT block via layer-wise zero convolutions.
- **Mechanism**: $\bm{h}_0 = \bm{z}_t + \mathcal{Z}_{\text{in}}(\bm{z}_{\text{box}})$, $\bm{h}_{l+1} = \text{Block}_l(\bm{h}_l) + \mathcal{Z}_l(\bm{z}_{\text{box}})$, with independent zero convolutions at each layer.
- **Design Motivation**: Input-level fusion alone is insufficient; removing hierarchical injection causes multi-subject mIoU to collapse from 0.532 to 0.289.

#### 4. Latent Identity Reward Model (LIRM)

- **Function**: Evaluates identity consistency between generated videos and reference images in latent space, providing a reward signal.
- **Mechanism**: Uses the first 8 layers of a pre-trained VDM as the backbone; reference image features serve as Q and video features as K/V in cross-attention; an MLP head predicts a scalar reward. Trained with BCE loss on ~27.5K manually annotated win-lose video pairs.
- **Design Motivation**: Compared to static encoders such as CLIP/DINO, the VDM backbone possesses spatiotemporal priors and can perceive identity consistency under motion; latent-space operation avoids the overhead of VAE decoding.

### Loss & Training

- **Stage 1 Loss**: Weighted diffusion loss $\mathcal{L}_{\text{sft}} = \mathbb{E}[(1 + \lambda_1 \mathbf{M}) \cdot \|\epsilon - \epsilon_\theta(\bm{z}_t, \mathcal{C}, t)\|_2^2]$, where $\lambda_1=2$ amplifies foreground region learning.
- **Stage 2 Loss**: $\mathcal{L} = \mathcal{L}_{\text{sft}} + \lambda_2 \mathcal{L}_{\text{LIReFL}}$, with $\lambda_2=0.1$. LIReFL is initialized from Gaussian noise, denoised without gradients to a random intermediate step $t_m$, then one gradient-enabled denoising step is executed; the frozen LIRM computes the reward and backpropagates. The SFT loss serves as regularization to prevent reward hacking.
- **Training Scale**: Stage 1 trains for 40K steps on 64×A100 GPUs; LIRM trains for 4K steps; LIReFL fine-tuning runs for 3.4K steps on 16×A100 GPUs.

## Key Experimental Results

### Main Results

**Table 1: Joint Customization + Motion Control Comparison on DreamOmni Bench**

| Method | R-CLIP↑ | R-DINO↑ | Face-S↑ | mIoU↑ | EPE↓ | CLIP-T↑ |
|--------|---------|---------|---------|-------|------|---------|
| DreamVideo-2 | 0.731 | 0.429 | 0.157 | 0.212 | 24.05 | 0.297 |
| **DreamVideo-Omni** | **0.739** | **0.499** | **0.301** | **0.558** | **9.31** | **0.308** |

**Table 2: Motion Control Comparison (Single-Subject / Multi-Subject)**

| Method | Single mIoU↑ | Single EPE↓ | Multi mIoU↑ | Multi EPE↓ |
|--------|-------------|------------|------------|-----------|
| Tora (1.1B) | 0.163 | 31.74 | 0.162 | 32.84 |
| Wan-Move (14B) | 0.507 | 14.43 | 0.541 | 9.02 |
| **DreamVideo-Omni (1.3B)** | **0.558** | **9.31** | **0.570** | **6.08** |

DreamVideo-Omni at 1.3B parameters comprehensively outperforms Wan-Move at 14B.

### Ablation Study

**Table 3: Component Ablation (Single-Subject Mode)**

| Configuration | R-DINO↑ | Face-S↑ | mIoU↑ | EPE↓ |
|--------------|---------|---------|-------|------|
| w/o Cond-Aware 3D RoPE | 0.139 | 0.039 | 0.274 | 30.22 |
| w/o Group & Role Emb. | 0.486 | 0.254 | 0.524 | 26.24 |
| w/o Hierarchical BBox | 0.508 | 0.257 | 0.400 | 31.84 |
| Stage 1 Only | 0.483 | 0.251 | 0.556 | 10.53 |
| w/o LIReFL (Stage 2 SFT only) | 0.487 | 0.266 | 0.561 | 10.01 |
| **Full Model** | **0.499** | **0.301** | **0.558** | **9.31** |

### Key Findings

1. **Condition-Aware 3D RoPE is foundational**: Its removal causes catastrophic degradation across all metrics and training collapse.
2. **Group/Role Embeddings are critical for multi-subject control**: Removal causes multi-subject EPE to increase from 6.80 to 20.69 (3× degradation).
3. **Hierarchical injection vs. input-level fusion gap is substantial**: Multi-subject mIoU drops from 0.532 to 0.289 without hierarchical injection.
4. **LIReFL effectively improves identity fidelity**: Under Stage 2 training, pure SFT yields limited gains, while LIReFL provides additional improvements of 0.013 on multi-subject Face-S and 0.012 on R-DINO.
5. **All-timestep reward > last-3-step reward**: Applying reward feedback across all timesteps outperforms restricting feedback to the final 3 denoising steps.
6. **Emergent capabilities**: Despite being trained on a T2V model, the framework naturally develops zero-shot I2V generation and first-frame-conditioned trajectory control.
7. **User study**: Overall quality preference rate reaches 89.2% on the joint task (vs. 10.8% for DreamVideo-2).

## Highlights & Insights

- **First unified framework for multi-subject customization + omni-motion control**: A single DiT simultaneously handles subject appearance, global motion, local dynamics, and camera motion.
- **Group/Role Embedding binding mechanism**: An elegant solution to multi-subject control ambiguity, assigning group embeddings to each ⟨subject, bounding box, trajectory⟩ triplet to explicitly bind signals to subjects.
- **Latent-space reward learning**: Avoids the substantial overhead of VAE decoding, making video-level ReFL genuinely practical; the VDM backbone is better suited than CLIP/DINO for evaluating identity consistency under motion.
- **Camera motion = background trajectory**: No additional 3D camera parameters are required; the trajectory control mechanism is directly reused for camera motion control, reducing training overhead.
- **Dataset (2.12M) and Benchmark (1027 videos)**: Both are new contributions to the community.

## Limitations & Future Work

1. Based on a 1.3B model; the upper bound on video quality is constrained by the base model capacity, which can be scaled to larger models.
2. Resolution is limited to 480×832 at 49 frames, far from high-definition long video generation.
3. Manual annotation cost for LIRM (27.5K pairs) is high; automated preference data generation warrants exploration.
4. Camera motion control is achieved indirectly via background trajectories, which may lack precision for exact 3D camera parameter control.
5. Scalability and quality when the number of subjects exceeds 2–3 require further validation.

## Related Work & Insights

- **DreamVideo-2**: The predecessor, supporting only single-subject + bounding box control; the present work represents a comprehensive upgrade.
- **Wan-Move**: A 14B-parameter I2V trajectory control model surpassed by the proposed 1.3B model, demonstrating that architectural design outweighs parameter scale.
- **IPRO / Identity-GRPO**: Related identity reinforcement learning approaches that compute rewards in pixel space at high cost and restrict feedback to the final denoising steps. The proposed latent-space approach is more efficient and enables full-timestep reward feedback.
- **PRFL**: A concurrent work on latent-space reward modeling targeting general video quality rather than identity preservation.
- **Insights**: (1) The latent-space reward learning paradigm is generalizable to other video control tasks; (2) the explicit binding mechanism of Group/Role embeddings is applicable to other multi-condition generation scenarios.

## Rating

⭐⭐⭐⭐⭐ A highly complete systems-level contribution in both engineering and methodology: the first unification of multi-subject customization and omni-motion control, with a well-motivated two-stage design, thorough ablation studies, new dataset and benchmark contributions, and the impressive result of a 1.3B model surpassing a 14B counterpart.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)
- [\[NeurIPS 2025\] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions](../../NeurIPS2025/image_generation/omnivcus_feedforward_subject-driven_video_customization_with_multimodal_control_.md)
- [\[CVPR 2026\] PureCC: Pure Learning for Text-to-Image Concept Customization](purecc_pure_learning_for_text-to-image_concept_customization.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](../../ICCV2025/image_generation/bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[CVPR 2026\] When Identities Collapse: A Stress-Test Benchmark for Multi-Subject Personalization](when_identities_collapse_a_stress-test_benchmark_for_multi-subject_personalizati.md)

</div>

<!-- RELATED:END -->
