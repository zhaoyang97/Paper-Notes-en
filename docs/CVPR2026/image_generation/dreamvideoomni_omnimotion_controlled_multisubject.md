---
title: >-
  [Paper Note] DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning
description: >-
  [CVPR 2026][Image Generation][video customization] This paper proposes DreamVideo-Omni, a unified DiT framework for multi-subject identity customization and omni-motion control (global bbox + local trajectory + camera motion). It resolves multi-subject ambiguity via condition-aware 3D RoPE and Group/Role Embeddings, and introduces Latent Identity Reward Feedback Learning (LIReFL) to provide dense identity rewards at arbitrary denoising timesteps, enabling efficient identity reinforcement by bypassing the VAE decoder.
tags:
  - CVPR 2026
  - Image Generation
  - video customization
  - multi-subject
  - motion control
  - identity preservation
  - reward learning
date: 2026-05-08
content_hash: 2e7524d45514f7f0
---

# DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning

**Conference**: CVPR 2026
**arXiv**: [2603.12257](https://arxiv.org/abs/2603.12257)
**Code**: [https://dreamvideo-omni.github.io](https://dreamvideo-omni.github.io/)
**Area**: Image Generation
**Keywords**: video customization, multi-subject, motion control, identity preservation, reward learning

## TL;DR

This paper proposes DreamVideo-Omni, a unified DiT framework for multi-subject identity customization and omni-motion control (global bbox + local trajectory + camera motion). It resolves multi-subject ambiguity via condition-aware 3D RoPE and Group/Role Embeddings, and introduces Latent Identity Reward Feedback Learning (LIReFL) to provide dense identity rewards at arbitrary denoising timesteps, enabling efficient identity reinforcement by bypassing the VAE decoder.

## Background & Motivation

**Background**: Large-scale diffusion models have achieved high-fidelity video generation, yet precisely controlling multi-subject identity alongside multi-granularity motion remains an open challenge. Existing approaches bifurcate into two independent directions: subject customization methods (e.g., ConsisID, Phantom) preserve appearance but offer no motion controllability; motion control methods (e.g., Wan-Move, Tora) enable precise motion control but cannot specify subject appearance.

**Limitations of Prior Work**: The few methods that attempt unification (DreamVideo-2, VACE) face three fundamental difficulties: (1) limited motion control granularity—most support only a single signal (bbox, depth, or sparse trajectory), unable to simultaneously cover global position, local dynamics, and camera motion; (2) multi-subject control ambiguity—all conditioning signals are injected indiscriminately, leaving the model unable to associate each motion with the correct subject; (3) identity degradation—motion control demands pixel-level temporal variation while identity preservation demands consistency with a static reference, a contradiction that standard diffusion reconstruction losses cannot reconcile.

**Key Challenge**: Identity preservation requires the output to resemble the reference image (pixel-level consistency), while motion control requires it to differ (temporal variation). These objectives are inherently contradictory at the gradient level. Low-level reconstruction losses cannot express the high-level semantic notion of "same identity, different pose."

**Key Insight**: (1) Resolve multi-subject ambiguity via explicit binding mechanisms (Group/Role Embeddings); (2) Elevate identity preservation from low-level pixel consistency to high-level semantic consistency aligned with human preference, by training a dedicated identity reward model to supply gradient signals.

**Core Idea**: Explicitly bind motion signals to subject identities through Group/Role Embeddings, and train a VDM-based latent-space identity reward model (LIRM) to provide dense identity feedback at arbitrary timesteps during denoising.

## Method

### Overall Architecture

Built upon Wan2.1-1.3B T2V DiT with a progressive two-stage training scheme. Stage 1 (Omni-Motion & Identity SFT) jointly trains on reference image, global bbox, and local trajectory triplets. Stage 2 (Latent Identity Reward Feedback Learning) trains a latent-space identity reward model (LIRM) that supplies identity-preserving reward signals directly during denoising. Training data consists of a self-constructed dataset of 2.12M videos; evaluation is conducted on the newly proposed DreamOmni Bench (1,027 videos).

### Key Designs

1. **Condition-Aware 3D RoPE + Group/Role Embeddings**:

    - Function: Handle heterogeneous inputs (video frames, reference images, trajectories) and resolve multi-subject control ambiguity.
    - Mechanism: The spatial dimensions of 3D RoPE retain standard indexing, while the temporal dimension is assigned per input type: video frames use continuous indices $t \in [0, T-1]$ (marked as sequence), reference images share a fixed index $t_{ref}$ (marked as static condition), padding uses invalid indices $t_{pad}$, and trajectories share indices with video frames (maintaining spatiotemporal alignment). Group embeddings assign a unique identifier to each $\langle$reference, bbox, trajectory$\rangle$ triplet, ensuring motion signals are bound to the correct subject; Role embeddings distinguish "appearance assets" (object embeddings) from "control signals" (control embeddings).
    - Design Motivation: Removing condition-aware RoPE directly causes training collapse. Removing Group/Role Embeddings substantially degrades motion control accuracy, especially in the multi-subject setting.

2. **Hierarchical BBox Injection**:

    - Function: Inject bbox conditioning into every transformer layer rather than superimposing it only at the input layer.
    - Mechanism: The bbox is first encoded into a latent representation $\mathbf{z}_{box}$ via 3D VAE. The initial superposition is $\mathbf{h}_0 = \mathbf{z}_t + \mathcal{Z}_{in}(\mathbf{z}_{box})$, and the output of each layer is further augmented as $\mathbf{h}_{l+1} = \text{Block}_l(\mathbf{h}_l) + \mathcal{Z}_l(\mathbf{z}_{box})$, where $\mathcal{Z}_l$ are layer-specific zero-convolutions. This design does not increase token sequence length.
    - Design Motivation: Ablations show that removing hierarchical injection causes multi-subject mIoU to collapse from 0.570 to 0.289 (−49.3%), demonstrating that injecting only at the input layer is insufficient to maintain spatial control in deeper layers.

3. **Latent Identity Reward Model (LIRM) + Latent Identity Reward Feedback Learning (LIReFL)**:

    - Function: Evaluate video–reference image identity consistency in latent space and provide dense gradient feedback during denoising.
    - Mechanism: The LIRM architecture consists of the first 8 layers of a pretrained VDM (backbone) + identity cross-attention layers + an MLP prediction head. Reference image features serve as queries $\mathbf{Q} = f_{ref}\mathbf{W_Q}$, while spatiotemporal video features serve as keys/values; identity alignment is computed via cross-attention as $r_t = \mathcal{H}(\mathbf{h}_{attn} + \mathbf{Q})$. The model is trained on ~27,500 videos with human preference annotations using BCE loss. During denoising, LIReFL randomly samples a timestep $t_m$, performs a single gradient-propagable denoising step, and passes the intermediate latent to the frozen LIRM to obtain a reward. The overall loss is $\mathcal{L} = \mathcal{L}_{sft} + 0.1 \cdot \mathcal{L}_{LIReFL}$.
    - Design Motivation: (1) Using a VDM backbone rather than a static image encoder (CLIP/DINO)—the VDM carries spatiotemporal priors that distinguish "same identity, different pose" from "copy-paste." A key finding is that reference images must serve as queries (using them as keys/values causes accuracy to drop from 0.720 to 0.455). (2) Operating entirely in latent space bypasses the VAE decoder, substantially improving training efficiency. (3) Providing feedback at arbitrary timesteps $t_m$ rather than only at the final step covers structural information across the entire denoising trajectory.

### Loss & Training

Stage 1: $\mathcal{L}_{sft} = \mathbb{E}[(1 + \lambda_1 \mathbf{M}) \cdot \|\epsilon - \epsilon_\theta(\mathbf{z}_t, \mathcal{C}, t)\|_2^2]$, with upweighted loss inside bbox regions ($\lambda_1=2$). Stage 2: LIRM is trained for ~4,000 steps; LIReFL is trained for 3,400 steps. SFT uses 64 × A100 GPUs; the RL stage uses 16 × A100 GPUs.

## Key Experimental Results

### Main Results (DreamOmni Bench)

| Method | R-CLIP↑ | R-DINO↑ | Face-S↑ | mIoU↑ | EPE↓ | CLIP-T↑ |
|------|---------|---------|---------|-------|------|---------|
| DreamVideo-2 | 0.731 | 0.429 | 0.157 | 0.212 | 24.05 | 0.297 |
| **DreamVideo-Omni** | **0.739** | **0.499** | **0.301** | **0.558** | **9.31** | **0.308** |

### Motion Control Accuracy (DreamOmni Bench)

| Method | Single-subject mIoU↑ | Single-subject EPE↓ | Multi-subject mIoU↑ | Multi-subject EPE↓ |
|------|-------------|------------|-------------|------------|
| Tora (1.1B) | 0.163 | 31.74 | 0.162 | 32.84 |
| Wan-Move (14B) | 0.507 | 14.43 | 0.541 | 9.02 |
| **Ours (1.3B)** | **0.558** | **9.31** | **0.570** | **6.08** |

### Ablation Study

| Configuration | mIoU↑ | EPE↓ | Face-S↑ |
|------|-------|------|---------|
| w/o Hierarchical BBox | 0.289 | 13.88 | - |
| w/o Group & Role Emb. | 0.458 | 10.38 | - |
| w/o Condition-aware RoPE | Training collapse | - | - |
| w/o LIReFL (Stage 1 only) | - | - | 0.271 |
| Full DreamVideo-Omni | 0.570 | 6.08 | 0.329 |

### Key Findings

- **A 1.3B model surpasses a 14B model**: DreamVideo-Omni (1.3B) comprehensively outperforms Wan-Move (14B) on both single- and multi-subject motion control, reducing EPE by 35% (single-subject) and 33% (multi-subject), demonstrating that precise conditioning injection matters more than model scale.
- **Hierarchical BBox injection is critical**: Its removal causes mIoU to collapse by 49.3%, indicating that spatial control signals must be reinforced at every layer within a DiT.
- **Query direction in LIRM is critical**: Using reference images as queries (0.720 accuracy) substantially outperforms using them as keys/values (0.455)—querying from reference identity to find correspondences in the video is more effective than the reverse.

## Highlights & Insights

- **Group/Role Embeddings resolve multi-subject ambiguity**: The explicit binding of control signals to subjects is an elegant and general design transferable to any generative task requiring independent control of multiple entities.
- **Latent-space identity rewards bypass the VAE bottleneck**: This represents a key breakthrough for ReFL in video generation. Computing rewards in pixel space incurs prohibitive GPU overhead, making video ReFL practically infeasible; operating in latent space makes it tractable. Furthermore, providing feedback at arbitrary timesteps—rather than only the final step—captures structural information throughout the denoising process.
- **Unifying camera motion and local motion as trajectory control**: Representing camera motion via the trajectories of background pixels avoids explicit 3D camera parameter estimation and eliminates the need for additional training data, substantially simplifying the pipeline.

## Limitations & Future Work

- **High training cost**: Stage 1 requires 64 × A100 GPUs for 40K steps; Stage 2 requires 16 × A100 GPUs, resulting in a substantial total compute budget.
- **Complex data construction pipeline**: The pipeline involves RAFT optical flow, RAM++ tagging, Qwen3-VL captioning, GroundingDINO detection, SAM2 segmentation, and CoTracker3 trajectory tracking, making full reproduction difficult.
- **Evaluation primarily on a self-constructed benchmark**: DreamOmni Bench is newly proposed and its community adoption has yet to be established.
- **Based solely on Wan2.1-1.3B**: Larger VDM backbones (14B) may further unlock performance.

## Related Work & Insights

- **vs. DreamVideo-2**: DreamVideo-2 supports only single-subject coarse bbox control and exhibits weak identity preservation (Face-S 0.157 vs. 0.301) and poor motion accuracy (EPE 24.05 vs. 9.31). The performance gap is attributable to Group/Role Embeddings and LIReFL.
- **vs. Wan-Move**: Wan-Move achieves precise trajectory control with 14B parameters but does not support identity customization. The proposed 1.3B model surpasses it across all motion metrics, indicating that architectural design is more important than parameter count.
- **vs. IPRO / Identity-GRPO**: Both require decoding latents to pixels before computing identity rewards, incurring large GPU overhead and limiting feedback to the final denoising steps. LIReFL's fully latent-space operation is a decisive advantage.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First framework to unify multi-subject customization, omni-motion control, and latent-space identity RL.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple benchmarks, multi-dimensional metrics, user studies, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Systematic and complete, though extremely detail-heavy.
- Value: ⭐⭐⭐⭐⭐ Defines a new state of the art and evaluation standard for video customization with motion control.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation](ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)
- [\[CVPR 2026\] GIST: Towards Design Compositing](gist_towards_design_compositing.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](consistcompose_multimodal_layout_control.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)

<!-- RELATED:END -->
