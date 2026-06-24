---
title: >-
  [Paper Note] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction
description: >-
  [CVPR 2025][Image Generation][Video Virtual Try-On] This paper proposes DPIDM (Dynamic Pose Interaction Diffusion Models), which injects synchronized human and garment skeleton poses into the denoising network via a pose adapter. A hierarchical attention module is designed to model intra-frame human-garment pose spatial interactions and inter-frame human pose temporal dynamics. Combined with a temporal regulative attention loss to enhance temporal consistency…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Video Virtual Try-On"
  - "Diffusion Models"
  - "Pose Interaction"
  - "Temporal Consistency"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: e960b2c19a93c9fb
---

# Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction

**Conference**: CVPR 2025  
**arXiv**: [2505.16980](https://arxiv.org/abs/2505.16980)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Video Virtual Try-On, Diffusion Models, Pose Interaction, Temporal Consistency, Attention Mechanism

## TL;DR

This paper proposes DPIDM (Dynamic Pose Interaction Diffusion Models), which injects synchronized human and garment skeleton poses into the denoising network via a pose adapter. A hierarchical attention module is designed to model intra-frame human-garment pose spatial interactions and inter-frame human pose temporal dynamics. Combined with a temporal regulative attention loss to enhance temporal consistency, the method achieves a VFID of 0.506 on the VVT dataset, representing a 60.5% improvement over the SOTA.

## Background & Motivation

**Background**: Video Virtual Try-On (VTON) aims to transfer a target garment onto a person in a video. Early GAN-based methods (e.g., CP-VTON, ClothFormer) utilize a two-stage "warp-and-blend" paradigm, which is limited by the accuracy of deformation estimation. Recent diffusion model-based methods (e.g., TunnelTry-on, ViViD, GPD-VVTO) leverage the generative capabilities of pretrained text-to-image models. They extract detailed garment features via a reference U-Net and incorporate temporal attention layers to enhance inter-frame consistency.

**Limitations of Prior Work**: Most existing video try-on methods process videos simply by inserting standard temporal attention after spatial attention, **ignoring the critical spatiotemporal pose interaction between the human body and the garment**. Specifically: (1) the **spatial alignment** between the human pose and the garment coverage in each frame (e.g., clothing wrinkles should change with body poses) is neglected; (2) the **temporal dynamics** of human poses across the entire video (human motion is a continuous sequence of coherent poses) are not adequately modeled. These issues are particularly prominent when the tried-on garment style differs significantly from the original clothing.

**Key Challenge**: Video try-on must simultaneously satisfy two objectives: (1) garment visual fidelity—the pattern, texture, and color of the clothing must strictly match the reference image; (2) temporal consistency—the garment appearance and body motion across frames must play out smoothly and naturally. Existing methods struggle to balance the two, especially during large-motion sequences.

**Goal**: Incorporate spatial pose alignment and temporal pose dynamics into the attention mechanism of diffusion models to fundamentally improve the quality of video try-on.

**Key Insight**: Introduce synchronized skeleton poses of the human body and the garment as explicit guidance, using a hierarchical attention module to separately handle intra-frame spatial interactions and inter-frame temporal dynamics.

**Core Idea**: Utilize a Pose Adapter to inject pose information into the Q/K/V components of both spatial and temporal attention (rather than only at the input layer), achieving fine-grained, pose-aware control at every layer.

## Method

### Overall Architecture

DPIDM adopts a dual-branch architecture: the Main U-Net (initialized from the SD v1.5 Inpainting model) processes a 9-channel input (4ch noisy latent + 4ch clothing-free latent + 1ch clothing-free mask); the Garment U-Net (initialized from SD v1.5) extracts fine-grained garment features and injects them into the Main U-Net. DW-Pose is used to estimate the human pose in the video, and a self-trained garment pose estimator extracts garment keypoints corresponding to the human keypoints. The poses are injected into each attention module of the Main U-Net via a lightweight Pose Encoder and Pose Adapter.

### Key Designs

1. **Pose-Aware Spatial Attention (PASA)**:

    - Function: Captures intra-frame human-garment pose spatial interactions to achieve pose-guided garment warping.
    - Mechanism: In the original spatial self-attention, human features $f_h$ and garment features $f_g$ are concatenated to compute attention. On top of this, PASA introduces a Pose Adapter: human pose embedding $p_h$ and garment pose embedding $p_g$ are concatenated as $p = [p_h, p_g]$ and mapped to $\text{Adpt}(p)$ via two fully connected (FC) layers (down → GELU → up, where the up-layer is zero-initialized to maintain the initial feature space). This is added to the features before computing self-attention: $h = \text{Attn}(\psi_q(f + \text{Adpt}(p)), \psi_k(f + \text{Adpt}(p)), \psi_v(f + \text{Adpt}(p)))$.
    - Design Motivation: Compared to standard PoseGuider methods that only feed pose images into the first layer, PASA injects pose information into the attention blocks of every layer, achieving finer multi-scale pose control. Moreover, the synchronous injection of human and garment poses provides explicit geometric guidance for spatial alignment.

2. **Temporal Shift Attention (TSA)**:

    - Function: Captures short-range temporal relationships between adjacent frames at a low computational cost.
    - Mechanism: Borrowing the concept of Latent-Shift, the patch tokens of the previous $L$ frames are shifted in the temporal dimension to the current frame to form the shifted features $h_{\text{shift}}$. These are concatenated with the current frame's features $h$ to serve as the K and V of the attention operation: $\hat{h} = \text{Attn}(\psi_q(h), \psi_k([h, h_{\text{shift}}]), \psi_v([h, h_{\text{shift}}]))$.
    - Design Motivation: Direct 3D spatiotemporal joint attention has a complexity of $O((H \times W \times T)^2)$. TSA integrates temporal information into the 2D attention framework via a simple shift operation, keeping the complexity at the $O((H \times W)^2)$ level.

3. **Pose-Aware Temporal Attention (PATA) + Temporal Regulative Attention Loss (TRA)**:

    - Function: PATA models long-range temporal dynamics of human poses in the video; TRA further constrains the consistency of attention maps across frames.
    - Mechanism: Similar to PASA, PATA adds human pose embeddings to the cross-attention output via a Pose Adapter, which is then processed by standard temporal attention, making temporal modeling aware of pose changes. The TRA loss enhances temporal consistency by minimizing the difference in PASA attention maps between consecutive frames: $\mathcal{L}_{\text{TRA}} = \sum_i^N \sum_{j=2}^T \gamma_i |\mathcal{A}_i^{(j)} - \mathcal{A}_i^{(j-1)}|$.
    - Design Motivation: Standard temporal attention does not consider continuous changes in human poses, which causing garment appearances to jitter during dramatic motion frames. PATA enables temporal modeling to recognize "where actions originate and where they lead," while TRA directly constrains inter-frame structural consistency at the loss level.

### Loss & Training

- Total Loss: $\mathcal{L} = \mathcal{L}_{\text{LDM}} + \lambda \mathcal{L}_{\text{TRA}}$, with $\lambda=0$ for image data and $\lambda=10^{-3}$ for video data.
- Joint Image-Video Training Strategy: Alternates between image training (updating only PASA + Cross Attention) and video training (updating only TSA + PATA), which reduces GPU memory overhead and speeds up convergence.
- Pose keypoints are randomly dropped with a 5% probability, forcing the model to infer poses from neighboring frames, thus enhancing robustness and temporal consistency.
- Trained for 80k steps with a batch size of 32, using 24 frames/sequence on 16×A100 GPUs.
- Inference uses the DDIM sampler, a classifier-free guidance scale of 1.5, and a sliding window for long videos.

## Key Experimental Results

### Main Results

| Method | VVT VFID_I ↓ | VVT LPIPS ↓| ViViD VFID_I ↓ | ViViD LPIPS ↓ |
|------|-------------|-------------|----------------|---------------|
| ClothFormer | 3.967 | 0.081 | - | - |
| TunnelTry-on | 3.345 | 0.054 | - | - |
| ViViD | 3.405 | 0.068 | 1.894 | 0.118 |
| GPD-VVTO | 1.280 | 0.056 | - | - |
| **DPIDM** | **0.506** | **0.041** | **0.488** | **0.081** |

On VVT, VFID drops from 1.280 to 0.506, marking a **relative improvement of 60.5%**; on ViViD, it shows a relative improvement of 74.2% compared to ViViD.

### Ablation Study

| Configuration | SSIM↑ | LPIPS↓ | VFID_I↓ | VFID_R↓ |
|------|-------|--------|---------|---------|
| (a) Baseline (Standard temporal attention) | 0.893 | 0.084 | 3.451 | 2.435 |
| (b) + PAA (Pose-Aware Attention) | 0.925 | 0.050 | 1.068 | 0.153 |
| (c) + PAA + TSA | 0.929 | 0.043 | 0.721 | 0.075 |
| (d) + PAA + TSA + TRA | **0.930** | **0.041** | **0.506** | **0.047** |

### Key Findings

- **PAA makes the greatest contribution**: Moving from (a) to (b), the VFID drops from 3.451 to 1.068. This single module surpasses all former SOTA methods, validating the core value of modeling spatiotemporal pose interactions.
- TSA and TRA primarily improve VFID (the temporal consistency metric) with minimal impact on SSIM/LPIPS, indicating that their function is centered on inter-frame smoothing.
- DPIDM also achieves SOTA on the image try-on task (VITON-HD FID 8.15, KID 0.32), showing that the pose-aware design also benefits single-frame quality.
- Qualitative analysis indicates that DPIDM maintains garment pattern consistency and natural wrinkles even under large motions.

## Highlights & Insights

- **Injection of the Pose Adapter at each layer**: Compared to feeding the pose map only once at the input layer, injecting pose information within the attention blocks of every layer achieves fine-grained multi-scale control. This concept is similar to ControlNet but more lightweight, and can be naturally transferred to other video generation tasks requiring spatial conditional control.
- **Joint Image-Video Training Strategy**: Alternating the training of spatial and temporal modules reduces VRAM constraints and avoids interference between the two types of modules. This training strategy provides a valuable reference for other joint spatiotemporal models.
- **Pose Drop-out**: Randomly dropping pose keypoints during training forces the model to refer to neighboring frames. This simple trick simultaneously enhances robustness to pose estimation errors and boosts temporal consistency.
- **Garment Pose Estimator**: Training a dedicated garment keypoint detector to establish human-garment pose correspondence is a novelty rarely seen in prior try-on works.

## Limitations & Future Work

- Requires 16×A100 GPUs for training, placing high demands on computational resources.
- The annotation data for the garment pose estimator is manually labeled, which is costly to scale up to more diverse clothing categories.
- Inference requires the full diffusion denoising process for every frame, leading to slow video inference speeds.
- Processing of back-facing people has been excluded due to inaccurate masks in these scenarios.
- Currently only handles single-person scenes; pose interactions in multi-person scenarios remain more complex.

## Related Work & Insights

- **vs GPD-VVTO**: GPD-VVTO integrates garment features into temporal attention to enhance fidelity but neglects poses. DPIDM simultaneously models spatial and temporal pose interactions via PASA and PATA, reducing the VFID from 1.280 to 0.506.
- **vs ViViD**: ViViD introduces a large-scale video try-on dataset and a reference U-Net architecture. While its SSIM is the highest on VVT (owing to larger training data), its VFID is far poorer than DPIDM (3.405 vs 0.506), demonstrating that temporal consistency is its weakness.
- **vs TunnelTry-on**: TunnelTry-on addresses the issue of off-center subjects through Focus-Tunnel cropping and alignment, but still relies on standard temporal attention for frame connections. DPIDM's hierarchical pose attention leads by a large margin in terms of temporal consistency.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of pose-aware attention and hierarchical spatiotemporal modeling is novel, and the garment pose estimation is also a fresh contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation across three datasets (VVT, ViViD, and VITON-HD) with detailed ablation studies and comprehensive qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐ The architecture diagram is clear, and the methodology is described in detail, although it contains somewhat numerous mathematical notations.
- Value: ⭐⭐⭐⭐ The massive 60.5% improvement in VFID demonstrates the key significance of modeling pose interactions for video virtual try-on.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Shining Yourself: High-Fidelity Ornaments Virtual Try-on with Diffusion Model](shining_yourself_high-fidelity_ornaments_virtual_try-on_with_diffusion_model.md)
- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [\[CVPR 2025\] Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model](re-hold_video_hand_object_interaction_reenactment_via_adaptive_layout-instructed.md)
- [\[CVPR 2025\] BooW-VTON: Boosting In-the-Wild Virtual Try-On via Mask-Free Pseudo Data Training](boow-vton_boosting_in-the-wild_virtual_try-on_via_mask-free_pseudo_data_training.md)
- [\[ECCV 2024\] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](../../ECCV2024/image_generation/wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)

</div>

<!-- RELATED:END -->
