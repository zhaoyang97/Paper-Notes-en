---
title: >-
  [Paper Note] EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations
description: >-
  [ICLR 2026][3D Vision][View translation] EgoWorld proposes an end-to-end exocentric-to-egocentric view translation framework that extracts three complementary observations from a single third-person image—3D point clouds…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "View translation"
  - "third-person to first-person"
  - "diffusion models"
  - "hand-object interaction"
  - "point cloud projection"
date: 2026-05-08
content_hash: 9dbbc28406cd961d
---

# EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations

**Conference**: ICLR 2026
**arXiv**: [2506.17896](https://arxiv.org/abs/2506.17896)
**Code**: [Available](https://redorangeyellowy.github.io/EgoWorld/)
**Area**: 3D Vision
**Keywords**: View translation, third-person to first-person, diffusion models, hand-object interaction, point cloud projection

## TL;DR

EgoWorld proposes an end-to-end exocentric-to-egocentric view translation framework that extracts three complementary observations from a single third-person image—3D point clouds, hand poses, and text descriptions—projects the point cloud to obtain a sparse egocentric RGB map, and reconstructs a complete high-fidelity egocentric image via diffusion-based inpainting, achieving state-of-the-art performance across four datasets under diverse unseen settings.

## Background & Motivation

**Background**: Egocentric vision is essential for AR/VR, robotic manipulation, and instructional video understanding, particularly for hand-object interaction analysis. However, the vast majority of real-world video data is captured from third-person perspectives, as head-mounted cameras and wearable devices remain far less prevalent than conventional cameras. Automatically generating egocentric views from exocentric images is therefore a problem of significant practical value.

**Limitations of Prior Work**: Existing exocentric-to-egocentric view translation methods impose severe input constraints. Exo2Ego-V requires multi-view 360° input; 4Diff requires known relative camera poses; EgoExo-Gen requires an initial egocentric reference frame and text instructions. The most closely related work, Exo2Ego, operates on a single exocentric image but relies on 2D hand layout prediction to establish structural transformations—a strategy that is highly unreliable under occlusion, viewpoint ambiguity, and cluttered environments, leading to poor generalization.

**Key Challenge**: A substantial geometric and semantic gap exists between exocentric and egocentric viewpoints. The exocentric view provides global context but lacks hand-object interaction detail; the egocentric view focuses on close-range hand-object interactions but lacks global context. This discrepancy results in extensive occlusion, appearance variation, and invisible regions that 2D alignment alone cannot bridge.

**Goal**: (1) How to obtain sufficient 3D geometric information from a single exocentric image without requiring multi-view inputs or camera pose priors; (2) how to complete sparse reprojection information into a dense, semantically coherent egocentric image.

**Key Insight**: The authors observe that a single exocentric image contains rich 3D information—depth maps can recover scene geometry, hand poses can establish cross-view correspondences, and text can provide semantic priors. Unifying these three complementary observations as conditioning inputs to a diffusion model reframes view translation as a multimodal conditional image inpainting problem.

**Core Idea**: Replace 2D layout estimation with 3D point cloud reprojection to obtain sparse structural priors, then combine hand poses and text descriptions as multimodal conditions to drive a diffusion model toward high-fidelity egocentric image reconstruction.

## Method

### Overall Architecture

EgoWorld is a two-stage pipeline. **Stage 1 (Exocentric View Observation $\Phi_{exo}$)**: Given a single exocentric image $I_{exo}$, the stage produces three intermediate representations—a sparse egocentric RGB map $S_{ego}$, a 3D egocentric hand pose $P_{ego}$, and a text description $T_{exo}$. **Stage 2 (Egocentric View Reconstruction $\Phi_{ego}$)**: Using these three observations as conditions, a pretrained Latent Diffusion Model (LDM) inpainting backbone reconstructs a dense egocentric image $\hat{I}_{ego}$ from the sparse RGB map. The entire pipeline requires neither multi-view inputs, nor known camera poses, nor egocentric reference frames.

### Key Designs

1. **Depth Estimation + Hand-Pose-Driven Scale Alignment**

    - **Function**: Construct a metric-scale 3D point cloud from the exocentric image.
    - **Mechanism**: An off-the-shelf monocular depth estimator (DepthAnythingV2) first produces a relative depth map $D_{exo}$ from $I_{exo}$. Because monocular depth suffers from inherent scale ambiguity, a hand pose estimator (ACR) extracts a MANO mesh from the exocentric image, yielding a metric-scale 3D exocentric hand pose $P_{exo}$ and a corresponding hand depth map $D_{hand}$. Within the valid hand region $\Omega_{hand}$, the median ratio between $D_{hand}$ and $D_{exo}$ serves as a global scale factor: $s^* = \text{median}_{(u,v) \in \Omega_{hand}} \frac{D_{hand}(u,v)}{D_{exo}(u,v)}$, calibrating the relative depth to metric depth $D'_{exo} = s^* D_{exo}$. Combined with the RGB image and camera intrinsics $K_{exo}$, this yields an RGBD point cloud $C_{exo} \in \mathbb{R}^{(H \times W) \times 6}$.
    - **Design Motivation**: The hand is nearly always visible in exocentric hand-object interaction scenes, and the MANO mesh provides a reliable metric-scale depth anchor—making hand-based scale alignment more robust than relying on other uncertain geometric constraints.

2. **Cross-View 3D Hand Pose Estimation + Umeyama Transform**

    - **Function**: Compute a rigid transformation from the exocentric to the egocentric viewpoint and project the point cloud accordingly.
    - **Mechanism**: This constitutes the core technical contribution. The authors train the first model $\phi_{ego}$ that directly predicts egocentric 3D hand poses from exocentric images. The architecture is intentionally simple: a ViT-224 backbone followed by a 2-layer MLP regressor (768→512→126 dimensions, corresponding to 21 keypoints × 3D × two hands). Given both the exocentric pose $P_{exo}$ and the predicted egocentric pose $P_{ego}$, the Umeyama algorithm solves for the optimal similarity transformation $(s, \mathbf{R}, \mathbf{t})$ between the two 3D point sets; its inverse yields $X = (X_{ego \to exo})^{-1}$. The exocentric point cloud $C_{exo}$ is transformed by $X$ and projected via $K_{ego}$ to produce the sparse egocentric RGB map $S_{ego}$.
    - **Design Motivation**: No prior model directly predicts egocentric hand poses from exocentric images. Hands are chosen as cross-view correspondences because they are nearly always visible in exocentric hand-object interaction footage (the body is often occluded by a table). Ablation studies confirm that the ViT backbone substantially outperforms CNN backbones, as global context is critical for cross-view reasoning.

3. **Multimodal Conditional Diffusion Reconstruction**

    - **Function**: Complete the sparse RGB map into a dense egocentric image.
    - **Mechanism**: A pretrained LDM inpainting model serves as the backbone. The sparse map $S_{ego}$ is VAE-encoded into a 4-channel latent embedding $s_{ego}$; the hand pose $P_{ego}$ is projected into 2D via $K_{ego}$, VAE-encoded, and then reduced to a 1-channel embedding $p_{ego}$ via a single convolutional layer. The concatenation of $s_{ego}$ (4ch), $p_{ego}$ (1ch), and a noisy latent $z_t$ (4ch) forms a 9-channel input to the U-Net. The text description $T_{exo}$ (scene and hand-object interaction descriptions generated by a VLM) is CLIP-encoded into $c_{exo} \in \mathbb{R}^{77 \times 768}$ and injected via cross-attention. Classifier-Free Guidance (CFG) is applied at inference to strengthen text control.
    - **Design Motivation**: Point cloud reprojection provides only partial observations due to occlusion and invisible regions, necessitating a powerful generative model to complete missing content. LDM inpainting is naturally suited to this "known partial pixels, complete the rest" formulation. Text and pose play complementary roles: text controls semantics and appearance (object category, scene style), while pose controls hand geometry configuration.

### Loss & Training

- **Hand pose estimator $\phi_{ego}$**: MSE (L2 regression) loss; ViT-224 backbone + 2-layer MLP; batch size 64, lr $1 \times 10^{-4}$, Adam optimizer, 100 epochs (~20 hours).
- **Diffusion model $\epsilon_\theta$**: Standard LDM denoising objective $\mathcal{L} = \mathbb{E}\|\epsilon_t - \epsilon_\theta(z'_t, t, c_{exo})\|_2^2$; fine-tuned from a pretrained LDM inpainting model; batch size 3, lr $1 \times 10^{-5}$, AdamW, 5 epochs (~10 hours).
- Inference uses CFG (scale factor $w$) conditioned on text embeddings vs. unconditional embeddings.
- All training is conducted on a single NVIDIA RTX 4090 GPU, imposing modest resource requirements.

## Key Experimental Results

### Main Results: Four Unseen Settings on H2O

| Setting | Method | FID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | PA-MPJPE ↓ | CLIPScore ↑ |
|---|---|---|---|---|---|---|---|
| Unseen Objects | pix2pixHD | 436.25 | 25.01 | 0.299 | 0.606 | 18.01 | 0.230 |
| Unseen Objects | CFLD | 59.62 | 25.92 | 0.431 | 0.454 | 7.997 | 0.266 |
| Unseen Objects | **EgoWorld** | **41.33** | **31.17** | **0.481** | **0.348** | **7.318** | **0.273** |
| Unseen Actions | CFLD | 50.95 | 28.53 | 0.432 | 0.459 | 8.120 | 0.270 |
| Unseen Actions | **EgoWorld** | **33.28** | **31.62** | **0.457** | **0.378** | **7.260** | **0.282** |
| Unseen Scenes | CFLD | 118.10 | 29.03 | 0.370 | 0.684 | 7.877 | 0.251 |
| Unseen Scenes | **EgoWorld** | **90.89** | **31.00** | **0.410** | **0.652** | **7.409** | **0.259** |
| Unseen Subjects | CFLD | 129.30 | 21.05 | 0.400 | 0.627 | 9.561 | 0.246 |
| Unseen Subjects | **EgoWorld** | **96.43** | **24.85** | **0.461** | **0.619** | **8.103** | **0.258** |

EgoWorld outperforms the strongest baseline CFLD—which uses ground-truth 2D hand layouts as input and thus constitutes an upper bound for Exo2Ego—across all unseen settings. In the Unseen Objects setting, FID decreases by 30% (59.62→41.33) and PSNR improves by over 5 dB; in Unseen Actions, FID decreases by 35%. Even in the most challenging Unseen Scenes setting, FID is reduced by 23%.

### Cross-Dataset Generalization (Unseen Actions)

| Dataset | Method | FID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | PA-MPJPE ↓ | CLIPScore ↑ |
|---|---|---|---|---|---|---|---|
| TACO | CFLD | 61.36 | 28.77 | 0.401 | 0.503 | 7.908 | 0.272 |
| TACO | **EgoWorld** | **37.19** | **30.16** | **0.424** | **0.403** | **7.359** | **0.283** |
| Assembly101 | CFLD | 53.93 | 21.00 | 0.399 | 0.557 | 11.11 | 0.246 |
| Assembly101 | **EgoWorld** | **50.23** | **25.37** | **0.410** | **0.514** | **10.56** | **0.256** |
| Ego-Exo4D | CFLD | 70.48 | 21.58 | 0.361 | 0.598 | 15.01 | 0.267 |
| Ego-Exo4D | **EgoWorld** | **61.23** | **24.99** | **0.399** | **0.548** | **13.99** | **0.286** |

EgoWorld consistently outperforms CFLD on three additional datasets, with FID reduced by approximately 39% on TACO and PSNR improved by over 3 dB on Ego-Exo4D.

### Ablation Study: Conditioning Modality Analysis

| Pose | Text | FID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | PA-MPJPE ↓ |
|---|---|---|---|---|---|---|
| ✗ | ✗ | 56.12 | 27.05 | 0.446 | 0.445 | 7.802 |
| ✓ | ✗ | 55.02 | 27.54 | 0.445 | 0.412 | 7.801 |
| ✗ | ✓ | 44.24 | 28.57 | 0.457 | 0.382 | 7.745 |
| ✓ | ✓ | **41.33** | **31.17** | **0.481** | **0.348** | **7.318** |

### Reconstruction Backbone Comparison

| Backbone | FID ↓ | PSNR ↑ | LPIPS ↓ | PA-MPJPE ↓ |
|---|---|---|---|---|
| MAE | 169.91 | 24.62 | 0.504 | 10.98 |
| MAT | 89.93 | 28.92 | 0.476 | 9.544 |
| MAT (Refined) | 68.63 | 29.75 | 0.451 | 8.256 |
| **LDM (EgoWorld)** | **41.33** | **31.17** | **0.348** | **7.318** |

### Key Findings

- **Text description contributes most**: Adding text alone reduces FID from 56.12 to 44.24 (a 21% reduction), whereas adding pose alone reduces FID only marginally from 56.12 to 55.02. This demonstrates that semantic information is critical for object and scene reconstruction.
- **Multimodal synergy is substantial**: Using both pose and text simultaneously yields PSNR of 31.17, an additional gain of 2.6 dB over text-only and 4.1 dB over the unconditioned baseline. Text controls "what to generate" while pose controls "where to place the hands."
- **Geometry–semantics decoupling**: Even when provided with incorrect text descriptions, the model maintains correct geometric structure (e.g., table surface tilt angle), indicating that the sparse map encodes geometric priors independently of the semantic priors provided by text.
- **ViT outperforms CNN**: The ViT backbone in the hand pose estimator substantially outperforms ResNet50 (FID 42.32 vs. 61.16), confirming that global context is essential for cross-view reasoning.
- **Robustness to noisy inputs**: Under challenging samples with hand occlusion or blur, EgoWorld degrades only slightly (FID 33.28→34.91), far outperforming the degradation observed in other baselines.
- **Hand pose representation has minor impact**: MANO mesh vs. keypoints yields nearly identical final performance (FID 33.21 vs. 33.28), suggesting that hand pose information is absorbed into the overall multimodal conditioning.

## Highlights & Insights

- **Paradigm upgrade from 2D to 3D**: Exo2Ego relies on 2D hand layout prediction as the structural transformation intermediary—a fundamentally unreliable strategy under occlusion. EgoWorld instead employs 3D point cloud reprojection for sparse structural priors, which is more robust and naturally encodes scene-level 3D geometry rather than hand position alone.
- **View translation as conditional inpainting**: The geometrically challenging cross-view synthesis problem is elegantly reformulated as an inpainting task well-suited to diffusion models. The sparse RGB map provides "known pixels," and the diffusion model completes only the missing regions, substantially reducing the generative burden. This problem reformulation is transferable to other cross-view generation tasks.
- **First cross-view hand pose prediction**: Directly predicting egocentric 3D hand poses from exocentric images had not been attempted previously. The model's architecture is remarkably simple (ViT + MLP), indicating that ViT global features already capture sufficient cross-view correspondence information.

## Limitations & Future Work

- **Error accumulation risk**: The pipeline depends on multiple off-the-shelf models (depth estimation, hand pose estimation, VLM), and their errors propagate through successive stages. Although experiments demonstrate robustness to noise, failure cases may arise under extreme occlusion or atypical poses.
- **Static image limitation**: The current method processes single frames without exploiting temporal information from video sequences. Extending to video-level egocentric synthesis would require additional temporal consistency mechanisms (e.g., temporal attention), representing a natural direction for future work.
- **Hallucination in invisible regions**: Regions entirely invisible from the exocentric viewpoint (e.g., book interior pages, palm surfaces) are fully generated by the diffusion model's imagination, with no guarantee of factual accuracy.
- **Camera intrinsic estimation**: The pipeline requires both exocentric and egocentric camera intrinsics, currently inferred from the depth estimator, which may introduce additional errors in real-world deployment.

## Related Work & Insights

- **vs. Exo2Ego**: Both address single exocentric-to-egocentric translation, but Exo2Ego uses 2D hand layouts (highly sensitive to occlusion) whereas EgoWorld uses 3D point cloud reprojection (more robust). Notably, CFLD uses ground-truth hand layouts—serving as an upper bound for Exo2Ego—yet EgoWorld surpasses it comprehensively.
- **vs. 4Diff**: Both leverage point clouds for view translation, but 4Diff does not incorporate hand pose or text conditions. The no-pose, no-text ablation of EgoWorld (FID 56.12) approximates 4Diff behavior and falls well short of the full model (FID 41.33), validating the necessity of multimodal conditioning.
- **vs. EgoExo-Gen**: EgoExo-Gen requires an initial egocentric reference frame, text instructions, and an exocentric video sequence—substantially more constrained inputs. EgoWorld requires only a single exocentric image.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Replacing 2D layout estimation with 3D point cloud reprojection is intuitive and effective; the cross-view hand pose predictor is a genuine first; however, each individual module (depth estimation, Umeyama alignment, LDM inpainting) draws on existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, four unseen settings, six metrics, comprehensive ablations (modality, backbone, pose representation, noise robustness), and real-world qualitative evaluation constitute a very complete experimental design.
- **Writing Quality**: ⭐⭐⭐⭐ The method is clearly described with rich figures and tables; the paper is well-structured with well-motivated problem formulation.
- **Value**: ⭐⭐⭐⭐ Direct applicability to AR/VR, robotics, and instructional video domains; the low barrier to entry—single-image input and single-GPU training—makes the system easy to deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views](../../NeurIPS2025/3d_vision/look_and_tell_a_dataset_for_multimodal_grounding_across_egocentric_and_exocentri.md)
- [\[ICLR 2026\] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [\[ICLR 2026\] Sharp Monocular View Synthesis in Less Than a Second](sharp_monocular_view_synthesis_in_less_than_a_second.md)
- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](dynamic_novel_view_synthesis_in_high_dynamic_range.md)
- [\[ICLR 2026\] Stylos: Multi-View 3D Stylization with Single-Forward Gaussian Splatting](stylos_multi-view_3d_stylization_with_single-forward_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
