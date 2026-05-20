---
title: >-
  [Paper Note] Mobile-VTON: High-Fidelity On-Device Virtual Try-On
description: >-
  [CVPR 2026][Human Understanding][Virtual Try-On] The first fully offline, on-device diffusion-based virtual try-on framework. Built upon a TeacherNet-GarmentNet-TryonNet (TGT) architecture…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Virtual Try-On"
  - "Mobile Deployment"
  - "Knowledge Distillation"
  - "Diffusion Models"
  - "Privacy Preservation"
date: 2026-05-08
content_hash: 65fe52499a51afa0
---

# Mobile-VTON: High-Fidelity On-Device Virtual Try-On

**Conference**: CVPR 2026
**arXiv**: [2603.00947]([https://arxiv.org/abs/2603.00947](https://arxiv.org/abs/2603.00947))  
**Code**: Available ([https://zhenchenwan.github.io/Mobile-VTON/](https://zhenchenwan.github.io/Mobile-VTON/))  
**Area**: Human Understanding
**Keywords**: Virtual Try-On, Mobile Deployment, Knowledge Distillation, Diffusion Models, Privacy Preservation

## TL;DR
The first fully offline, on-device diffusion-based virtual try-on framework. Built upon a TeacherNet-GarmentNet-TryonNet (TGT) architecture, it transfers the capabilities of SD3.5 Large to a 415M-parameter lightweight student network via Feature-Guided Adversarial (FGA) distillation. The method matches or surpasses server-side baselines at 1024×768 resolution on VITON-HD and DressCode, with an end-to-end inference time of approximately 80 seconds on a Xiaomi 17 Pro Max.

## Background & Motivation
Virtual try-on (VTON) is highly practical for fashion e-commerce, yet existing high-quality methods almost universally rely on cloud-side GPUs: users must upload personal photos to remote servers for inference, incurring latency and energy overhead while posing serious privacy risks—particularly under stringent data protection regulations. Deploying diffusion-based VTON on mobile devices presents three core challenges: (1) large model parameter counts, memory footprints, and latency far exceed the capacity of mobile NPUs/GPUs; (2) garment representations undergo semantic drift across diffusion timesteps, leading to texture distortion and detail loss; (3) existing methods depend heavily on large-scale pretraining (e.g., ImageNet or large-scale text-to-image), and lightweight architectures cannot directly acquire sufficient generative capability from task-specific data alone.

## Core Problem
How can high-fidelity virtual try-on be achieved on a commodity smartphone—without uploading user data, using only a person image and a garment image as input? The central tension is that the model must be small enough to run on mobile hardware while achieving generation quality comparable to server-side methods with 5–17× more parameters.

## Method

### Overall Architecture
Mobile-VTON adopts a modular TGT architecture: TeacherNet (frozen SD 3.5 Large, serving as the knowledge source) + GarmentNet (lightweight student for extracting consistent garment features) + TryonNet (lightweight student for fusing human body and garment information to synthesize try-on images). A Light-Adapter replaces the large CLIP visual encoder with DINOv2-base, injecting garment semantics via an IP-Adapter mechanism. The entire system is trained directly on task data without reliance on external pretraining.

### Key Designs
1. **FGA Distillation (Feature-Guided Adversarial Distillation)**: The student network is trained with two complementary objectives. (i) Feature-level distillation: the score functions of TeacherNet and the student network are aligned at each diffusion timestep via $\mathcal{L}_\text{feature} = \mathbb{E}_t[\|s_\text{true} - s_\text{fake}\|^2]$, employing DMD2-style score matching rather than pixel-wise regression, enabling the student to learn the distributional behavior of the teacher. (ii) Adversarial augmentation: a lightweight discriminator $D$ is introduced to distinguish real images from TryonNet-generated images, enhancing realism and detail sharpness through a standard GAN loss $\mathcal{L}_\text{GAN}$.
2. **TCG (Trajectory-Consistent GarmentNet)**: Addresses semantic drift of garment features across diffusion timesteps. A reconstruction constraint is directly applied to GarmentNet at each timestep $t$: $\mathcal{L}_\text{cons} = \mathbb{E}_t[\|\hat{X}_g(t) - X_g\|^2]$, requiring the network to consistently reconstruct the original garment image throughout the diffusion trajectory. This temporal regularization stabilizes garment colors, textures, and logos across timesteps.
3. **Garment-Aware TryonNet**: (i) Latent Concatenation (LC): the person image and garment image are concatenated along the height dimension before encoding into latent space, with a reference conditional input (concatenated encoding of the target person and garment) that enables TryonNet to learn garment–body alignment without pretraining. (ii) Feature fusion: multi-scale features from corresponding GarmentNet layers are concatenated into each self-attention layer of TryonNet; cross-attention simultaneously receives textual tokens and visual K-V pairs from the Light-Adapter, enabling multi-level garment semantic injection.
4. **Light-Adapter**: Replaces the large CLIP visual encoder with DINOv2-base, projecting garment image features into K and V tensors and injecting them into TryonNet via decoupled cross-attention, balancing semantic richness with mobile-side efficiency.

### Loss & Training
- GarmentNet total loss: $\mathcal{L}_\text{GarmentNet} = \lambda_1 \cdot \mathcal{L}_\text{featureG} + \lambda_2 \cdot \mathcal{L}_\text{cons}$
- TryonNet total loss: $\mathcal{L}_\text{TryonNet} = \mathcal{L}_\text{Diff} + \lambda_1 \cdot \mathcal{L}_\text{featureT} + \lambda_3 \cdot \mathcal{L}_\text{GAN}$ (where $\mathcal{L}_\text{Diff}$ is a garment-aware reconstruction loss)
- Hyperparameters: $\lambda_1 = 1\text{e-}2$, $\lambda_2 = 0.5$, $\lambda_3 = 5\text{e-}3$
- Two-stage training: Stage 1 trains for 140 epochs on a combined DressCode + VITON-HD dataset (lr = 1e-4); Stage 2 fine-tunes for 100 epochs on DressCode (lr = 5e-5)
- Hardware: 8× A100 80GB, batch size = 256, AdamW optimizer

## Key Experimental Results

| Dataset | Metric | Ours (Mobile-VTON) | Prev. SOTA | Notes |
|--------|------|------|----------|------|
| VITON-HD | LPIPS↓ | 0.088 | 0.102 (IDM-VTON) | Surpasses best server-side method (mask-based) |
| VITON-HD | SSIM↑ | 0.893 | 0.890 (SD-VITON) | Best |
| DressCode | LPIPS↓ | 0.053 | 0.0513 (BooW-VTON) | Near best |
| DressCode | SSIM↑ | 0.935 | 0.928 (BooW-VTON) | Best |
| VITON-HD In-Wild | LPIPS↓ | 0.133 | 0.137 (IDM-VTON) | Best |
| Memory | GPU Memory | 2.84 GB | 5.80–18.47 GB | 51%–85% reduction |
| Deployment | Mobile | ✓ (Xiaomi 17 Pro Max, ~80s) | All ✗ | Only method deployable on mobile |

### Ablation Study
- **Contribution of TCG**: Adding TCG reduces LPIPS from 0.119 to 0.111, improves SSIM from 0.874 to 0.879, and CLIP-I from 0.798 to 0.805. Visually, logos and stripes become sharper and color localization more accurate.
- **Contribution of LC**: Building on TCG, adding LC further reduces LPIPS from 0.111 to 0.088, raises SSIM to 0.893, and CLIP-I to 0.833. LC provides explicit garment geometry and appearance cues, compensating for the absence of pretraining.
- **Criticality of distillation**: Removing distillation causes FID to surge from 10.2 to 113.6—a complete collapse—demonstrating that lightweight models trained from scratch without teacher guidance entirely fail to converge.
- **Effect of dataset quality**: Fine-tuning on DressCode outperforms fine-tuning on VITON-HD, indicating that lightweight models are more sensitive to data quality; DressCode offers more uniform resolution and clearer visual content.

## Highlights & Insights
- The most significant technical contribution is FGA distillation: the combination of score-based distillation and GAN training enables a 415M-parameter student network to achieve generation quality comparable to a 2B+ parameter teacher.
- TCG is remarkably simple yet effective—merely a cross-timestep reconstruction consistency constraint—yet it directly resolves the core problem of garment semantic drift in diffusion models.
- The entire system is trained directly from task data without large-scale pretraining, offering a valuable reference for resource-constrained deployment scenarios.
- The choice of DINOv2-base over CLIP as the visual encoder deserves attention—it represents a favorable efficiency–quality trade-off for mobile deployment.
- The complete pipeline is validated on a real smartphone with reported inference times (80s), constituting a practical rather than theoretical deployment demonstration.

## Limitations & Future Work
- An end-to-end inference time of 80 seconds remains too long for a smooth user experience; no step-count reduction, pruning, or system-level acceleration has been applied.
- The method struggles to accurately reproduce text-bearing garments (logos, brand names, slogans) due to the lack of text-aware pretraining and limited text-garment examples in training data.
- Only upper-body try-on is supported; full-body, dress, and other garment categories are not covered.
- As a mask-free method that must synthesize the entire image (including background and body), the approach is inherently disadvantaged compared to mask-based methods on FID/KID metrics.
- INT8 quantization is executed on Android NPUs, but quantization-induced accuracy degradation is not quantitatively reported.

## Related Work & Insights
- **vs. IDM-VTON (18.47 GB)**: IDM-VTON is the strongest server-side mask-based baseline, achieving CLIP-I of 0.875 on VITON-HD. Mobile-VTON surpasses it on LPIPS and SSIM, with slightly lower CLIP-I (0.833 vs. 0.875), while requiring only 2.84 GB memory and supporting mobile deployment—the two methods operate in fundamentally different regimes.
- **vs. CatVTON**: Both are mask-free methods employing latent concatenation. Mobile-VTON comprehensively outperforms CatVTON on LPIPS/SSIM (0.088 vs. 0.161; 0.893 vs. 0.872), demonstrating that the TGT architecture combined with FGA distillation is substantially superior to naively applying CatVTON's concatenation strategy.
- **vs. BooW-VTON**: BooW-VTON is the strongest server-side mask-free baseline, leading on FID/KID. Mobile-VTON surpasses it on SSIM on DressCode (0.935 vs. 0.928) and achieves comparable LPIPS (0.053 vs. 0.051), while requiring only 2.84 GB vs. 18.47 GB memory.

The FGA distillation strategy (score-based + adversarial) is transferable to other diffusion model tasks requiring edge deployment. The temporal consistency constraint in TCG can be adapted to video generation, 3D-consistent generation, and other temporal or multi-view tasks. The finding that data quality matters more for lightweight models than for large models warrants verification in other distillation research. Related idea: `20260316_convnet_dit_hybrid_distill.md` (diffusion model distillation).

## Rating
- Novelty: ⭐⭐⭐⭐ [The TGT architecture and FGA distillation strategy constitute systematic innovation; being the first mobile-deployable diffusion VTON carries practical engineering value]
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ [Three datasets, multiple baselines, detailed ablations, real-device deployment, and dataset quality analysis—comprehensive coverage]
- Writing Quality: ⭐⭐⭐⭐ [Clear structure, rich figures and tables, thorough method description]
- Value: ⭐⭐⭐⭐ [On-device deployment of diffusion models is an important engineering direction; the FGA distillation strategy demonstrates good generalizability]

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)
- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[ICLR 2026\] Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals](../../ICLR2026/human_understanding/inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth.md)
- [\[CVPR 2026\] ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body](vibes_a_conversational_agent_with_behaviorally_intelligent_3d_virtual_body.md)

</div>

<!-- RELATED:END -->
