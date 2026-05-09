---
title: >-
  [Paper Note] Mobile-VTON: High-Fidelity On-Device Virtual Try-On
description: >-
  [CVPR 2026][Human Understanding][Virtual try-on] This paper proposes Mobile-VTON, the first diffusion-based virtual try-on system capable of running fully offline on mobile devices. Through a TeacherNet-GarmentNet-TryonNet (TGT) architecture and a Feature-Guided Adversarial (FGA) distillation strategy, the system achieves high-quality try-on results comparable to server-side baselines with only 415M parameters and 2.84GB memory.
tags:
  - CVPR 2026
  - Human Understanding
  - Virtual try-on
  - mobile deployment
  - knowledge distillation
  - diffusion models
  - privacy protection
date: 2026-05-08
content_hash: 079fc42f6310f41d
---

# Mobile-VTON: High-Fidelity On-Device Virtual Try-On

**Conference**: CVPR 2026
**arXiv**: [2603.00947](https://arxiv.org/abs/2603.00947)
**Code**: [Project Page](https://zhenchenwan.github.io/Mobile-VTON/)
**Area**: Human Understanding
**Keywords**: Virtual try-on, mobile deployment, knowledge distillation, diffusion models, privacy protection

## TL;DR

This paper proposes Mobile-VTON, the first diffusion-based virtual try-on system capable of running fully offline on mobile devices. Through a TeacherNet-GarmentNet-TryonNet (TGT) architecture and a Feature-Guided Adversarial (FGA) distillation strategy, the system achieves high-quality try-on results comparable to server-side baselines with only 415M parameters and 2.84GB memory.

## Background & Motivation

Virtual try-on (VTON) has seen substantial improvements in visual quality in recent years, yet mainstream systems rely on cloud-based GPU inference, requiring users to upload personal photos. This introduces three core issues: (1) diffusion models are too large in parameter count, memory, and latency for mobile hardware; (2) garment representations drift semantically across diffusion steps, causing inconsistencies; (3) existing methods depend on large-scale pre-training (e.g., text-to-image), making it difficult for lightweight architectures to independently acquire sufficient garment synthesis capability. Furthermore, uploading personal photos poses privacy risks.

## Method

### Overall Architecture

Mobile-VTON adopts a modular TGT architecture: **TeacherNet** (based on SD 3.5 Large, frozen) provides supervision signals; **GarmentNet** (lightweight Light-UNet) extracts garment features and maintains timestep consistency; **TryonNet** (lightweight Light-UNet) fuses person–garment representations to synthesize the final try-on image. The core mechanism is Feature-Guided Adversarial (FGA) Distillation, which transfers teacher model knowledge into the lightweight student network.

### Key Designs

1. **Feature-Guided Adversarial (FGA) Distillation**: Combines two complementary objectives:

    - **Feature-level distillation**: Aligns the score function estimates of teacher and student at each diffusion step, rather than regressing pixel values. Given a noisy latent $\tilde{\mathbf{z}}^{(t)}$, the frozen teacher $D_t$ and student $D_s$ produce $s_{\text{true}}$ and $s_{\text{fake}}$, respectively, and the $\ell_2$ distance is minimized:
    $\mathcal{L}_{\text{feature}} = \mathbb{E}_t \| s_{\text{true}}(\tilde{\mathbf{z}}^{(t)}, t) - s_{\text{fake}}(\tilde{\mathbf{z}}^{(t)}, t) \|_2^2$
    - **Adversarial augmentation**: A lightweight discriminator $D$ distinguishes real from generated images, while TryonNet improves perceptual realism by fooling the discriminator:
    $\mathcal{L}_{\text{GAN}} = \mathbb{E}_{X \sim \mathcal{R}}[\log D(X)] + \mathbb{E}_{\hat{X} \sim \mathcal{G}}[\log(1 - D(\hat{X}))]$

2. **Trajectory-Consistent GarmentNet (TCG)**: Addresses semantic drift of garment features across diffusion steps. The diffusion process is applied deterministically at each timestep $t$, requiring the model to consistently reconstruct the original garment image across all steps:
    $\mathcal{L}_{\text{cons}} = \mathbb{E}_{t \sim [1,T]} [\| \hat{X}_g^{(t)} - X_g \|_2^2]$
   This temporal regularization stabilizes garment semantics along the diffusion trajectory, preventing texture distortion and shape warping.

3. **Garment-Aware TryonNet**: Trained from scratch directly on the try-on task without large-scale pre-training:

    - **Latent Concatenation**: Person and garment images are concatenated along the height dimension and encoded as $\mathbf{z}_{\text{concat}}$; a reference input $X_{\text{condi}} = \text{Concat}_{\text{height}}(X_t, X_g)$ is constructed to guide identity and garment appearance preservation.
    - **Multi-scale Feature Fusion**: At each self-attention layer, multi-scale garment features $\mathbf{F}_g^{(i)}$ from GarmentNet are concatenated with TryonNet's hidden states; dual-branch cross-attention simultaneously fuses text conditioning and visual garment semantics from the Light-Adapter.
    - **Light-Adapter**: DINOv2-base (replacing large CLIP visual encoders) is used to extract garment visual features, injected via IP-Adapter-style decoupled cross-attention.

### Loss & Training

GarmentNet: $\mathcal{L}_{\text{GarmentNet}} = \lambda_1 \mathcal{L}_{\text{feature}}^G + \lambda_2 \mathcal{L}_{\text{cons}}$

TryonNet: $\mathcal{L}_{\text{TryonNet}} = \mathcal{L}_{\text{Diff}} + \lambda_1 \mathcal{L}_{\text{feature}}^T + \lambda_3 \mathcal{L}_{\text{GAN}}$

Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GarmentNet}} + \mathcal{L}_{\text{TryonNet}}$, with $\lambda_1=1\text{e-}2, \lambda_2=0.5, \lambda_3=5\text{e-}3$.

Two-stage training: Stage 1 trains for 140 epochs on a combined DressCode+VITON-HD dataset (lr=1e-4); Stage 2 fine-tunes for 100 epochs on a DressCode subset (lr=5e-5). Training uses 8×A100 GPUs with batch size 256.

## Key Experimental Results

### Main Results

| Method | VITON-HD LPIPS↓ | VITON-HD SSIM↑ | DressCode LPIPS↓ | Memory (GB) | Mobile |
|------|----------------|----------------|-----------------|----------|--------|
| IDM-VTON | 0.102 | 0.868 | 0.065 | 18.47 | ✗ |
| BooW-VTON | 0.107 | 0.864 | 0.051 | 18.47 | ✗ |
| CatVTON | 0.161 | 0.872 | 0.092 | 5.80 | ✗ |
| StableVITON | 0.142 | 0.875 | 0.113 | 13.84 | ✗ |
| **Mobile-VTON** | **0.088** | **0.893** | **0.053** | **2.84** | **✓** |

### Ablation Study

| Configuration | LPIPS↓ | SSIM↑ | CLIP-I↑ | FID↓ | Note |
|------|--------|-------|---------|------|------|
| w/o TCG, w/o LC | 0.119 | 0.874 | 0.798 | 11.231 | Baseline |
| +TCG | 0.111 | 0.879 | 0.805 | 10.814 | Improved garment semantic stability |
| +TCG +LC | 0.088 | 0.893 | 0.833 | 10.211 | Full component combination |

### Key Findings

- Mobile-VTON achieves the best LPIPS (0.088) and SSIM (0.893) on VITON-HD with only 2.84GB memory and 415M parameters, outperforming all server-side baselines.
- On DressCode, it achieves the best SSIM (0.935) and second-best LPIPS (0.053).
- Fully mask-free (no segmentation masks required); the model must synthesize the entire image (including body and background), making FID/KID evaluation more challenging, yet results remain competitive.
- TCG and LC each contribute significantly; their combination yields cumulative gains (LPIPS reduced from 0.119 to 0.088).
- The method also performs well in in-the-wild settings (LPIPS 0.133 vs. IDM-VTON's 0.137).

## Highlights & Insights

- **First mobile-deployable diffusion VTON**: Demonstrates that high-quality virtual try-on can run entirely on a smartphone, with practical commercial value for privacy protection.
- **Score-based distillation + adversarial training**: The FGA strategy avoids the blurring artifacts of pixel-level regression and combines adversarial loss to improve perceptual realism.
- **Trajectory consistency constraint**: A simple yet effective solution to garment semantic drift across diffusion steps.
- **No large-scale pre-training required**: Through latent concatenation and teacher supervision, the lightweight model learns directly from task data, lowering the training barrier.
- **DINOv2 as a CLIP replacement**: Lightweight yet semantically rich visual feature extraction, offering a useful reference for mobile optimization.

## Limitations & Future Work

- FID/KID metrics are slightly higher than some server-side methods (due to the harder mask-free task), leaving room for improved distribution alignment.
- Only upper-body try-on is supported; full-body, lower-body, and accessory scenarios remain to be explored.
- Actual on-device inference speed and power consumption are not reported.
- The two-stage training pipeline is relatively complex; end-to-end joint training may be more efficient.
- Robustness under extreme poses and heavy occlusion has not been thoroughly evaluated.

## Related Work & Insights

- **DMD2**: Source of inspiration for the FGA distillation strategy (score-based distillation).
- **CatVTON**: Reference for the latent concatenation strategy; Mobile-VTON further integrates it with distillation.
- **IDM-VTON**: Reference for garment self-attention fusion; Mobile-VTON achieves similar effects with a lightweight design.
- **SnapGen**: Reference architecture for Light-UNet, adapted for efficient mobile inference.
- Takeaway: The combination of knowledge distillation, adversarial training, and domain-specific constraints is a valuable reference for model compression and deployment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First mobile-deployable diffusion VTON; the TGT architecture and FGA distillation are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 3 datasets with multiple baselines and complete ablations, though actual on-device latency data is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear and loss derivations are complete.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses industry needs (e-commerce, privacy); first demonstration of high-quality VTON feasibility on mobile devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](referencefree_image_quality_assessment_for_virtual.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)
- [\[CVPR 2026\] IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)
- [\[CVPR 2026\] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)

</div>

<!-- RELATED:END -->
