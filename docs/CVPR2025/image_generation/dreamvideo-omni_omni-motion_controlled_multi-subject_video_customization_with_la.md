---
title: >-
  [Paper Note] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning
description: >-
  [CVPR 2025][Image Generation][Multi-Subject Customization] DreamVideo-Omni is proposed, achieving collaborative generation of multi-subject customization and omni-motion control (global bbox + local trajectory + camera motion) within a unified DiT framework through a progressive two-stage training paradigm (Omni-Motion SFT + Latent Identity Reward Feedback Learning).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multi-Subject Customization"
  - "Omni-Motion Control"
  - "Identity Preservation"
  - "Latent Reward Learning"
  - "Video DiT"
date: 2026-05-08
content_hash: 9771d8316274bd79
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

**Conference**: CVPR 2025  
**arXiv**: [2603.12257](https://arxiv.org/abs/2603.12257)  
**Code**: [Project Homepage](https://dreamvideo-omni.github.io)  
**Area**: Image Generation  
**Keywords**: Multi-Subject Customization, Omni-Motion Control, Identity Preservation, Latent Reward Learning, Video DiT

## TL;DR

DreamVideo-Omni is proposed, achieving collaborative generation of multi-subject customization and omni-motion control (global bbox + local trajectory + camera motion) within a unified DiT framework through a progressive two-stage training paradigm (Omni-Motion SFT + Latent Identity Reward Feedback Learning).

## Background & Motivation

**Background**: Diffusion models have significantly advanced the quality of video generation, but controlling multi-subject identities and multi-granularity motion simultaneously remains an open challenge. Subject customization methods and motion control methods currently develop along two independent paths.

**Limitations of Prior Work**:
   - **Insufficient Motion Control Granularity**: Existing methods rely solely on a single motion signal (bbox / depth / trajectory) and cannot simultaneously control global position, local dynamics, and camera motion.
   - **Control Ambiguity**: In multi-subject scenarios, models cannot distinguish which motion signal corresponds to which subject, leading to control confusion.
   - **Identity Degradation**: Identity fidelity decreases after introducing motion control, as identity preservation requires pixel consistency while motion control requires pixel changes.

**Key Challenge**: The objectives of identity preservation (striving for consistency with the reference image) and motion control (striving for dynamic changes) are inherently in conflict, which cannot be reconciled by standard diffusion reconstruction loss.

**Goal**: Simultaneously achieve multi-subject identity preservation, omni-motion control (global/local/camera), and control ambiguity elimination within a single framework.

**Key Insight**: Two-stage progressive training—first establishing unified control capability via SFT, and then enhancing identity fidelity using latent identity reward reinforcement learning.

**Core Idea**: Explicitly bind motion signals to the corresponding subjects via Group/Role Embedding, and directly reinforce identity preservation in the latent space using a VDM-based latent identity reward model.

## Method

### Overall Architecture

DreamVideo-Omni is based on Wan2.1-1.3B T2V DiT and employs a two-stage training process:

- **Stage 1 (Omni-Motion & Identity SFT)**: Jointly trains subject appearance, global bbox motion, local trajectory motion, and camera motion.
- **Stage 2 (Latent Identity Reward Feedback Learning)**: Trains the Latent Identity Reward Model (LIRM) and reinforces identity preservation using ReFL.

### Key Designs

1. **Condition-Aware 3D RoPE**

    - **Function**: Assigns different temporal indices to heterogeneous inputs (video frames, reference images, trajectory tokens).
    - **Mechanism**: Video frames use sequential indices $[0, T-1]$; reference images are marked as static conditions with a unified $t_{\text{ref}}$; padding is marked as invalid with $t_{\text{pad}}$; trajectories inherit video frame indices to maintain spatio-temporal alignment.
    - **Design Motivation**: Eliminates temporal confusion of heterogeneous inputs; removing it leads to training collapse (catastrophic drop in all metrics in ablation studies).

2. **Group & Role Embedding**

    - **Function**: Explicitly binds motion signals to the corresponding subjects, eliminating multi-subject control ambiguity.
    - **Mechanism**: Group Embedding binds the triplet $\langle\text{reference image, bbox, trajectory}\rangle$ into a group, while Role Embedding distinguishes between "appearance assets" (object embedding) and "motion control" (control embedding).
    - **Design Motivation**: In multi-subject scenarios, motion signals of different subjects must be explicitly associated; otherwise, the model cannot distinguish between them.

3. **Hierarchical Motion Injection**

    - **Function**: Injects bbox conditions at each block level of the DiT.
    - **Mechanism**: Bbox latents are added to the input and the output of each block via learnable zero-convolution: $h_0 = z_t + Z_{\text{in}}(z_{\text{box}})$, $h_{l+1} = \text{Block}_l(h_l) + Z_l(z_{\text{box}})$.
    - **Design Motivation**: Merely merging bbox at the input layer is insufficient for precise global motion control; multi-layer injection significantly improves mIoU ($0.289 \rightarrow 0.532$).

4. **Latent Identity Reward Model (LIRM)**

    - **Function**: Evaluates the identity consistency between the generated video and the reference image in the latent space.
    - **Mechanism**: Utilizes the first 8 layers of VDM as the backbone, takes the reference image features as $Q$ to perform cross-attention with the noisy video features, and outputs a scalar reward.
    - **Design Motivation**:
        - More motion-aware than static encoders like CLIP/DINO, enabling it to distinguish copy-paste artifacts.
        - Operating in the latent space avoids expensive VAE decoding costs.

5. **Latent Identity Reward Feedback Learning (LIReFL)**

    - **Function**: Backpropagates the reward signals of LIRM to optimize the video generation model.
    - **Mechanism**: Denoises from noise to a random intermediate step $t_m$, performs one step of gradient-enabled denoising to obtain $z_{t_m}$, passes it into the frozen LIRM to calculate reward, and maximizes this reward.
    - **Design Motivation**: Bypasses VAE decoding, supports reward feedback at arbitrary timesteps (not just the final steps), and fully leverages the potential of ReFL.

### Loss & Training

- **Stage 1 SFT Loss**: Reweighted diffusion loss, where the region inside the bbox is weighted by $\lambda_1 = 2$ to enhance subject learning.
- **Stage 2 Total Loss**: $L = L_{\text{sft}} + \lambda_2 \cdot L_{\text{LIReFL}}$, where $\lambda_2 = 0.1$.
- **LIRM Training**: Binary cross-entropy loss, trained on a preference dataset of 27,500 training videos + 500 test videos.
- **Conditional Dropout**: Bbox and trajectory conditions are randomly dropped with $p=0.5$, and reference image enhancement shares the same probability.

## Key Experimental Results

### Main Results

**Comprehensive Comparison on DreamOmni Bench**

| Method | R-CLIP↑ | R-DINO↑ | Face-S↑ | mIoU↑ | EPE↓ | CLIP-T↑ |
|------|---------|---------|---------|-------|------|---------|
| DreamVideo-2 | 0.731 | 0.429 | 0.157 | 0.212 | 24.05 | 0.297 |
| **DreamVideo-Omni** | **0.739** | **0.499** | **0.301** | **0.558** | **9.31** | **0.308** |

**Comparison on MSRVTT-Personalization (Subject Mode / Face Mode)**

| Method | CLIP-T↑ | R-DINO↑ | EPE↓ | Face-S↑ |
|------|---------|---------|------|---------|
| Video Alchemist | 0.268 | 0.626 | - | 0.411 |
| Tora2 | 0.273 | 0.615 | 17.43 | 0.419 |
| **DreamVideo-Omni** | **0.273** | **0.628** | **11.21** | **0.417** |

**Comparison on Motion Control (DreamOmni Bench)**

| Method | Single-Subject mIoU↑ | Single-Subject EPE↓ | Multi-Subject mIoU↑ | Multi-Subject EPE↓ |
|------|-------------|------------|-------------|------------|
| Tora (1.1B) | 0.163 | 31.74 | 0.162 | 32.84 |
| Wan-Move (14B) | 0.507 | 14.43 | 0.541 | 9.02 |
| **Ours (1.3B)** | **0.558** | **9.31** | **0.570** | **6.08** |

### Ablation Study

**Ablation of Individual Components (DreamOmni Bench - Single/Multi Subject)**

| Method | R-DINO↑ | Face-S↑ | mIoU↑ | EPE↓ |
|------|---------|---------|-------|------|
| w/o Cond-Aware 3D RoPE (Single Subject) | 0.139 | 0.039 | 0.274 | 30.22 |
| w/o Group & Role Emb. (Multi Subject) | 0.503 | 0.289 | 0.459 | 20.69 |
| w/o Hierarchical BBox Inject. (Multi Subject) | 0.510 | 0.269 | 0.289 | 25.56 |
| Ours Stage 1 (Multi Subject) | 0.506 | 0.287 | 0.532 | 6.80 |
| w/o LIReFL (Multi Subject) | 0.512 | 0.316 | 0.556 | 6.29 |
| **Ours Full (Multi Subject)** | **0.524** | **0.329** | **0.570** | **6.08** |

**User Study (Joint Subject + Motion vs. DreamVideo-2)**

| Dimension | DreamVideo-2 | Ours |
|------|-------------|------|
| Subject Fidelity | 22.4% | **77.6%** |
| Motion Consistency | 18.3% | **81.7%** |
| Overall Quality | 10.8% | **89.2%** |

### Key Findings

- DreamVideo-Omni with 1.3B parameters outperforms the 14B Wan-Move in motion control, demonstrating extreme parameter efficiency.
- Removing Condition-Aware 3D RoPE causes training collapse, proving it is a foundational component of the framework.
- After removing Hierarchical BBox Injection, the multi-subject mIoU drops from $0.532$ to $0.289$, proving that multi-layer injection is indispensable.
- LIReFL steadily improves all identity preservation metrics without compromising motion control accuracy.
- The framework exhibits the emergence of zero-shot I2V generation and first-frame conditional trajectory control capabilities (despite being trained only on T2V).

## Highlights & Insights

1. **Explicit Binding of Control Signals**: Group + Role Embedding binds $\langle\text{subject, bbox, trajectory}\rangle$ into structured triplets, fundamentally solving the multi-subject ambiguity problem with a clear and effective approach.
2. **Latent Space Reward Learning**: The VDM-based LIRM calculates rewards directly in the latent space, avoiding VAE decoding overhead. It is motion-aware, making it more suitable for video scenarios than CLIP/DINO.
3. **Unified Control of Camera Motion**: Treats camera motion as background point trajectories, unifying it with local motion under the same trajectory conditioning mechanism, which avoids extra 3D camera parameter estimation.
4. **Large-scale Data Engineering**: A complete pipeline utilizing 2.12M videos (motion filtering $\rightarrow$ subject discovery $\rightarrow$ spatio-temporal annotation $\rightarrow$ reference image construction); data quality is the guarantee of performance.
5. **Emergent Capabilities**: Multi-task training naturally unlocks I2V and first-frame conditional trajectory control, demonstrating the generalization capability of the framework.

## Limitations & Future Work

1. **Resolution and Frame Count Limitations**: Currently only supports $480 \times 832$ / $49$ frames; scaling to high-resolution long videos remains to be verified.
2. **Dependency on Base Models**: Based on Wan2.1-1.3B, larger models (such as 14B) may have room for further improvement.
3. **Generalization of LIRM**: The reward model is trained on specific preference data, and its generalization to uncovered subject types has not been fully verified.
4. **Upper Limit of Multi-Subject Count**: The padding mechanism in grid attention implies a fixed capability limit $N_{\text{max}}$.
5. **Upper Bound of Complex Motion**: Performance under extreme motion and occlusion scenarios is not discussed in detail.

## Related Work & Insights

- **Relationship with DreamVideo-2**: This work is a continuation of the DreamVideo series, extending from single-subject + bbox to multi-subject + omni-motion.
- **Relationship with Wan-Move**: Wan-Move focuses on point trajectory control (14B I2V); this work surpasses it with a 1.3B T2V model, demonstrating the importance of architectural design.
- **Relationship with IPRO/Identity-GRPO**: These methods calculate rewards in the pixel space requiring VAE decoding, whereas the proposed latent space scheme in this work is more efficient.
- **Insights**: The design concept of the latent space reward model can be generalized to other generative tasks (such as audio, 3D), and the design of the binding mechanism is universally valuable for controllable generation of multiple entities.

## Rating

- Novelty: ⭐⭐⭐⭐ (Each component is innovative but not a paradigm breakthrough; latent space reward learning is the main highlight.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Self-built benchmark + extensive ablations + user study, highly comprehensive.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure but quite long; some parts could be condensed.)
- Value: ⭐⭐⭐⭐⭐ (The unified framework for multi-subject + omni-motion holds significant practical value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](../../ICCV2025/image_generation/bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[NeurIPS 2025\] OmniVCus: Feedforward Subject-driven Video Customization with Multimodal Control Conditions](../../NeurIPS2025/image_generation/omnivcus_feedforward_subject-driven_video_customization_with_multimodal_control_.md)
- [\[CVPR 2025\] DreamRelation: Bridging Customization and Relation Generation](dreamrelation_bridging_customization_and_relation_generation.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)
- [\[CVPR 2026\] Scaling Multi-Identity Consistency for Image Customization via Multi-to-Multi Matching Paradigm](../../CVPR2026/image_generation/scaling_multi-identity_consistency_for_image_customization_via_multi-to-multi_ma.md)

</div>

<!-- RELATED:END -->
