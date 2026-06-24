---
title: >-
  [Paper Note] OFER: Occluded Face Expression Reconstruction
description: >-
  [CVPR 2025][Image Generation][Occluded face reconstruction] OFER utilizes two conditional diffusion models to generate the shape and expression coefficients of the FLAME parametric model, respectively, and integrates a ranking network to select the optimal shape from multiple candidates, achieving diverse and realistic 3D facial expression reconstruction under occlusion.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Occluded face reconstruction"
  - "diffusion models"
  - "multi-hypothesis reconstruction"
  - "ranking mechanism"
  - "FLAME parametric model"
date: 2026-05-08
content_hash: 0f8b53c9bbb5ebe6
---

# OFER: Occluded Face Expression Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2410.21629](https://arxiv.org/abs/2410.21629)  
**Code**: [https://ofer.is.tue.mpg.de/](https://ofer.is.tue.mpg.de/)  
**Area**: Image Generation  
**Keywords**: Occluded face reconstruction, diffusion models, multi-hypothesis reconstruction, ranking mechanism, FLAME parametric model

## TL;DR

OFER utilizes two conditional diffusion models to generate the shape and expression coefficients of the FLAME parametric model, respectively, and integrates a ranking network to select the optimal shape from multiple candidates, achieving diverse and realistic 3D facial expression reconstruction under occlusion.

## Background & Motivation

**Background**: Single-image 3D face reconstruction is a classic inverse problem, typically recovering geometry by regressing or optimizing 3DMM (e.g., FLAME) parameters. Existing methods like MICA, DECA, and EMOCA perform reasonably well under mild occlusion, but they are mostly deterministic methods that can only output a single solution.

**Limitations of Prior Work**: Under severe occlusion (masks, sunglasses, hair, large profile angles, etc.), facial information is extremely incomplete, and the occluded regions can correspond to infinitely many plausible 3D shapes. Deterministic methods lack generative diversity and cannot capture the multi-hypothesis nature of the problem. Although Diverse3D uses DPP sampling to produce multiple solutions, its generated results are often exaggerated and distorted.

**Key Challenge**: The ambiguity introduced by occlusion requires methods that can model the output distribution rather than point estimation. However, existing multi-hypothesis methods (such as those based on VAE+DPP) fail to adequately learn the true distribution of facial geometry, resulting in poor generation quality.

**Goal**: (1) Generate multiple realistic and diverse 3D faces under occlusion; (2) Select the optimal identity shape from multiple shape candidates to ensure consistency across expressions.

**Key Insight**: Diffusion models are inherently capable of learning data distributions, making them suitable for modeling multimodal outputs. The authors observe that the variation of shape (identity) is much smaller than that of expression. Therefore, one can first determine an optimal shape and then superimpose diverse expressions on top of it.

**Core Idea**: Two DDPMs are employed to separately generate the shape and expression coefficients of FLAME, followed by a ranking network to select the optimal solution from the shape candidates.

## Method

### Overall Architecture

The input is a face image that may contain occlusions, and the output is a set of diverse 3D face reconstructions. The overall pipeline consists of three steps: (1) IdGen generates N=100 FLAME shape coefficient candidates; (2) IdRank ranks these candidates to select the optimal identity; (3) ExpGen generates N expression coefficients, which are combined with the selected shape to produce the final diverse reconstructions.

### Key Designs

1. **Identity Generative Network (IdGen)**:

    - **Function**: Generates a set of FLAME shape coefficients from Gaussian noise conditioned on the input image.
    - **Mechanism**: Uses a DDPM with a 1D U-Net structure. The input is 300-dimensional noise, and the condition is a 512-dimensional image feature encoded by ArcFace. The shape parameter $S \in \mathbb{R}^{300}$ is generated through 1000 denoising steps. Sampling with N=100 different initial noises yields 100 shape hypotheses. The training loss is the standard DDPM noise prediction loss.
    - **Design Motivation**: Diffusion models learn the true distribution of shape parameters, and different starting noises naturally produce plausible variations in the occluded regions, avoiding the distribution modeling flaws of sampling methods like DPP.

2. **Identity Ranking Network (IdRank)**:

    - **Function**: Selects the optimal shape that best matches the input image from the N shape candidates output by IdGen.
    - **Mechanism**: Decodes each shape coefficient into a mesh via FLAME, removes the back of the head to retain only the frontal vertices, and subtracts the mean to obtain a residual mesh. A 5-layer MLP, conditioned on a joint ArcFace+FaRL encoding (1024-dimensional), scores each candidate. During training, ranking labels are generated on-the-fly: the L1 distance between each candidate and the GT is calculated and passed through a softmax to serve as the ranking target, which is trained with cross-entropy loss.
    - **Design Motivation**: Experiments reveal that the minimum error among the 100 candidates is already close to SOTA; the key is how to select it. Shape is more stable than expression, and a good ranking mechanism can filter out low-quality samples. This is the first time a ranking selection mechanism has been introduced in diffusion models.

3. **Expression Generative Network (ExpGen)**:

    - **Function**: Generates diverse FLAME expression coefficients.
    - **Mechanism**: The structure is identical to IdGen, but it generates 50-dimensional expression parameters (the first 50 expression components of FLAME), conditioned on the joint ArcFace+FaRL feature (1024-dimensional). During inference, N expression candidates are generated and combined with the shape selected by IdRank.
    - **Design Motivation**: Expressions exhibit greater ambiguity under occlusion, making them suitable for modeling multiple possibilities using diffusion models. Decoupling shape and expression helps maintain identity consistency.

### Loss & Training

- IdGen training loss: Standard DDPM L1 noise prediction loss.
- IdRank training loss: Cross-entropy between the predicted ranking distribution and the ground-truth distance ranking distribution.
- ExpGen training loss: Same DDPM loss as IdGen.
- The three networks are trained independently. During IdRank training, the gradients of IdGen are frozen, and candidates are generated on-the-fly.
- Training data: IdGen/IdRank use four 2D-3D paired datasets: Stirling, FaceWarehouse, LYHM, and Florence; ExpGen uses the FaMoS dynamic face dataset.

## Key Experimental Results

### Main Results

| Dataset | Metric | OFER-rank | FOCUS (MP) | MICA (8DS) | Diverse3D |
|---|---|---|---|---|---|
| NoW (Full) Med | MSE↓ | **0.98** | 1.03 | 0.90 | 1.41 |
| NoW (Occlusion) Med | MSE↓ | **1.01** | 1.08 | N/A | N/A |
| CO-545 (Occlusion) | SE RMSE↓ | **1.95** | - | - | 2.16 |
| CO-545 | CSE RMSE↓ | **0.17** | - | - | 0.30 |

| Dataset | Metric | OFER | Diverse3D |
|---|---|---|---|
| Erakiotan (mask) | STD-S↑ | **34.04** | 11.81 |
| Erakiotan (sunglasses) | STD-S↑ | **34.38** | 21.28 |
| Erakiotan (mask) | ODE↓ | **0.002** | 0.95 |

### Ablation Study

| Configuration | Ranked Med↓ | Avg Med↓ | Ideal Min Med↓ |
|---|---|---|---|
| OFER (100 samples) | **0.98** | 1.02 | 0.81 |
| FLAME(50)+OFER(50) | 1.08 | 1.33 | 0.84 |
| FLAME(80)+OFER(20) | 1.34 | 1.60 | 0.86 |
| Random FLAME(1000) | N/A | 1.75 | 0.90 |

### Key Findings

- The candidate search space generated by IdGen is significantly superior to random FLAME sampling (ideal min: 0.81 vs. 0.90), indicating that the diffusion model indeed learns a better shape distribution.
- The ranking mechanism effectively selects low-error samples from the candidates, with the Ranked result (0.98) being far superior to the average (1.02) and close to the ideal minimum (0.81).
- The expression diversity (STD-S) generated by OFER in occluded regions far exceeds that of Diverse3D (34+ vs. 11-21), with almost no out-of-distribution errors (ODE ≈ 0).
- On the NoW occlusion subset, OFER-rank outperforms FOCUS, a method specifically designed for occlusions.

## Highlights & Insights

- The design of **shape-expression decoupling combined with ranking** is ingenious: it utilizes the prior that "identity is more certain than expression under occlusion", first pinning down the shape and then generating diverse expressions, which ensures consistency while retaining diversity.
- This is the **first time a ranking mechanism has been introduced into the sample selection pipeline of diffusion models**. Drawing inspiration from listwise ranking in information retrieval, it holds general value for other generative tasks that require selecting the best from multiple samples.
- The zero-mean residual input design of the ranking network reduces redundancy and speeds up convergence.
- The CO-545 dataset fills the gap in quantitative evaluation for occluded face reconstruction.

## Limitations & Future Work

- The training data volume is limited (only a few thousand 2D-3D pairs), which may cause the expression generation to lack fine details.
- The ranking network's use of softmax leads to decreased score discriminability when the number of candidates is large, meaning it might not always select the absolute optimal candidate.
- Inference requires generating 100 candidates and ranking them one by one, which incurs a large computational overhead.
- The ranking network is fixed during training on N=100 samples and cannot directly transfer to other candidate sizes.
- Future directions: (1) Integrating ranking as a feedback signal in diffusion training to constrain quality during generation; (2) Fusing 2D and 3D supervision to expand the training data; (3) Employing faster sampling strategies (e.g., DDIM).

## Related Work & Insights

- **vs. MICA**: MICA regresses FLAME shapes using ArcFace features, achieving high performance but being deterministic and lacking expression support. OFER reuses MICA's encoder design concept but shifts to a generative paradigm.
- **vs. Diverse3D**: Both perform multi-hypothesis occlusion reconstruction. However, Diverse3D's use of VAE+DPP sampling leads to poor distribution modeling and exaggerated generation results; OFER uses a diffusion model to directly learn the distribution, coupled with ranking selection.
- **vs. EMOCA**: Deterministic expression regression degrades severely under occlusion and cannot provide multi-hypothesis estimates.
- RankGAN introduced ranking into GAN generation; OFER transfers this idea to diffusion models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of a ranking selection mechanism and diffusion-based multi-hypothesis generation is novel, though the individual components themselves are pre-existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Proposes a new dataset CO-545, presents comprehensive ablation studies, and evaluates across multiple metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with a smooth logical flow of motivation.
- **Value**: ⭐⭐⭐⭐ The idea of ranking-based selection has general transfer value, and the approach is practical for occluded reconstruction scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models](../../AAAI2026/image_generation/realistic_face_reconstruction_from_facial_embeddings_via_diffusion_models.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](svfr_a_unified_framework_for_generalized_video_face_restoration.md)
- [\[CVPR 2025\] FDeID-Toolbox: Face De-Identification Toolbox](fdeid-toolbox_face_de-identification_toolbox.md)
- [\[CVPR 2025\] GIF: Generative Inspiration for Face Recognition at Scale](gif_generative_inspiration_for_face_recognition_at_scale.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)

</div>

<!-- RELATED:END -->
